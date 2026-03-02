import numpy as np
import torch 
import mdtraj as md
from scipy.spatial.transform import Rotation as R
from torch.nn.functional import softplus

def get_all_chiralities_vec(traj):
    chiralities = []

    for frame_idx in range(traj.n_frames):
        frame_chiralities = []
        volumes = []
        for residue in traj.top.residues:
            try:
                # Get atom indices for N, CA, C, and CB
                n_idx = residue.atom('N').index
                ca_idx = residue.atom('CA').index
                c_idx = residue.atom('C').index
                cb_idx = residue.atom('CB').index

                # Get the coordinates of the atoms for the current frame
                n = traj.xyz[frame_idx, n_idx]
                ca = traj.xyz[frame_idx, ca_idx]
                c = traj.xyz[frame_idx, c_idx]
                cb = traj.xyz[frame_idx, cb_idx]

                # Calculate the vectors
                v1 = n - ca
                v2 = c - ca
                v3 = cb - ca

                # Compute the volume of the parallelepiped formed by v1, v2, and v3
                volume = np.dot(np.cross(v1, v2), v3)
                if volume > 0.:
                    frame_chiralities.append(-1)
                elif volume < 0:
                    # print(volume)
                    frame_chiralities.append(1)
                else:
                    frame_chiralities.append(0)
                # frame_chiralities.append(volume)

            except KeyError:
                frame_chiralities.append(0)

        chiralities.append(frame_chiralities)

    return np.array(chiralities)

def get_all_volumes_vec(traj):
    chiralities = []

    for frame_idx in range(traj.n_frames):
        frame_chiralities = []
        volumes = []
        for residue in traj.top.residues:
            try:
                # Get atom indices for N, CA, C, and CB
                n_idx = residue.atom('N').index
                ca_idx = residue.atom('CA').index
                c_idx = residue.atom('C').index
                cb_idx = residue.atom('CB').index

                # Get the coordinates of the atoms for the current frame
                n = traj.xyz[frame_idx, n_idx]
                ca = traj.xyz[frame_idx, ca_idx]
                c = traj.xyz[frame_idx, c_idx]
                cb = traj.xyz[frame_idx, cb_idx]

                # Calculate the vectors
                v1 = n - ca
                v2 = c - ca
                v3 = cb - ca

                # Compute the volume of the parallelepiped formed by v1, v2, and v3
                volume = np.dot(np.cross(v1, v2), v3)
                frame_chiralities.append(volume)
                # frame_chiralities.append(volume)

            except KeyError:
                frame_chiralities.append(0)

        chiralities.append(frame_chiralities)

    return np.array(chiralities)

import torch

def build_residue_maps(top):
    """
    Pre-compute atom indices for every residue that has the five atoms we need.
    Returns a list of dicts (one per residue).
    """
    res_maps = []
    for res in top.residues:
        try:
            res_maps.append(
                dict(
                    n   = res.atom('N' ).index,
                    ca  = res.atom('CA').index,
                    c   = res.atom('C' ).index,
                    cb  = res.atom('CB').index,
                    o   = res.atom('O' ).index,
                    side=[a.index for a in res.atoms
                          if a.name not in ('N', 'CA', 'C', 'O')]
                )
            )
        except KeyError:      # glycine, missing O, etc.
            continue
    return res_maps


def chirality_fix_tensor(pos, res_maps, t,
                         k_side_init=0.02, k_oxy_init=-0.01,
                         eps_target=2e-4, k_max=3.0):
    """
    Flip or encourage chirality for a single structure (batch = 1).

    Parameters
    ----------
    pos   : (1, N, 3) *or* (N, 3) absolute coordinates  [nm]
    res_maps : list from build_residue_maps(top)
    t     : current diffusion time in [0, 1]
    eps_target : signed volume we insist on after the flip branch
    k_max : hard cap on any per-residue velocity            [nm ps⁻¹]

    Returns
    -------
    dv    : (N, 3) velocity increments                     [nm ps⁻¹]
    """
    # ---- normalise shapes ---------------------------------------------------
    if pos.ndim == 3:            # (1, N, 3) → (N, 3)
        pos = pos[0]
    N = pos.size(0)
    dv = torch.zeros_like(pos)   # (N, 3)

    # cubic track we want to follow
    V_cubic   = 0.002 * t**3
    time_left = max(1.0 - t, 1e-8)     # duration until t = 1

    for rm in res_maps:
        n, ca, c, cb, o, side = (
            rm[k] for k in ('n', 'ca', 'c', 'cb', 'o', 'side')
        )
        side = torch.as_tensor(side, device=pos.device)

        # backbone vectors (3-D each)
        v1 = pos[n]  - pos[ca]          # N–CA
        v2 = pos[c]  - pos[ca]          # C–CA
        v3 = pos[cb] - pos[ca]          # CA→CB

        n_vec = torch.cross(v1, v2)     # normal to the plane
        A     = n_vec.norm() + 1e-8
        n_hat = n_vec / A               # unit normal

        V = torch.dot(n_hat, v3)        # signed volume  (scalar)
        
        # choose branch -------------------------------------------------------
        if V < 0:                       # ----- flip branch -----
            delV       = eps_target - V           # positive amount to add
            k_needed = torch.clamp(delV / (A * time_left),
                                   0.0, k_max)

        elif V < V_cubic:               # ----- encourage branch -----
            delV       = V_cubic - V
            k_needed = torch.clamp(delV / (A * time_left),
                                   0.0, k_max)

        else:                           # already on track
            continue

        # make push ⟂ CA–CB to keep bond length unchanged --------------------
        u_cb   = v3 / (v3.norm() + 1e-8)
        proj   = torch.dot(n_hat, u_cb) * u_cb
        n_perp = (n_hat - proj)
        n_perp = n_perp / (n_perp.norm() + 1e-8)

        # apply velocities ----------------------------------------------------
        dv[side] += k_needed * n_perp           # every side-chain atom incl. CB
        dv[o]    += -0.5 * k_needed * n_perp    # oxygen counter-kick

    return dv         # (N, 3)

def get_atom_indices_by_name(topology, residue, atom_names):
    indices = []
    for atom_name in atom_names:
        try:
            indices.append([atom.index for atom in residue.atoms if atom.name == atom_name][0])
        except IndexError:
            indices.append(None)  # Append None if the atom is not found (e.g., CB in glycine)
    return indices

def get_dihed_idxs(top):

    # List to hold the atom indices for each residue
    atom_indices = []

    # Get the indices of N, CA, CB, and C atoms for each residue
    for residue in top.residues:
        if residue.name != 'GLY':  # Skip glycine residues
            indices = get_atom_indices_by_name(top, residue, ['N', 'CA', 'CB', 'C'])
            if None not in indices:  # Ensure all atoms are present
                atom_indices.append(indices)

    return atom_indices


def invert_chirality(traj, chi):
    """
    Invert the chirality of specific residues.

    Parameters:
    - traj: MDTraj trajectory object
    - res_list: List of residue indices to invert chirality
    """
    
    for frame_index in range(traj.n_frames):
        res_list = np.where(chi[frame_index] > 0.001)[0]
    
        for residue_index in res_list:

            # Get the specific residue
            residue = traj.topology.residue(residue_index)

            # Identify the atoms in the chiral center (N, CA, C, CB)
            try:
                n_idx = residue.atom('N').index
                ca_idx = residue.atom('CA').index
                c_idx = residue.atom('C').index
                cb_idx = residue.atom('CB').index
            except KeyError:
                print(f"Residue {residue_index} does not have the required atoms for chirality inversion.")
                continue

            # Get the indices of all side chain atoms
            side_chain_indices = [atom.index for atom in residue.atoms if atom.name not in ['N', 'CA', 'C', 'O', 'OXT']]

            # Include the CB atom in the side chain if it's not already included
            if cb_idx not in side_chain_indices:
                side_chain_indices.append(cb_idx)
        
            # Get the coordinates of the chiral center atoms for the first frame
            n = traj.xyz[frame_index, n_idx]
            ca = traj.xyz[frame_index, ca_idx]
            c = traj.xyz[frame_index, c_idx]
            cb = traj.xyz[frame_index, cb_idx]

            # Calculate the centroid of the chiral center (excluding CB for the rotation axis calculation)
            centroid = (n + ca + c) / 3.0
        
            # Calculate the rotation axis (from CA to the centroid of N, CA, and C)
            rotation_axis = np.cross(ca - centroid, c - centroid)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # Normalize the axis

            # Calculate the 180-degree rotation matrix
            # MJ add negative to rotation
            rot_matrix = -R.from_rotvec(np.pi * rotation_axis).as_matrix()

            # Apply the rotation to each side chain atom
            for atom_idx in side_chain_indices:
                atom_coords = traj.xyz[frame_index, atom_idx]  # Get coordinates for all frames
                # Translate to the origin (centroid)
                translated_coords = atom_coords - centroid
                # Rotate the coordinates
                rotated_coords = np.dot(rot_matrix, translated_coords.T).T
                # Translate back to the original position
                new_coords = rotated_coords + centroid
                # Update the coordinates in the trajectory
                traj.xyz[frame_index, atom_idx] = new_coords

    return traj


def invert_chirality_reflection(traj, chi):
    """
    Invert the chirality of specific residues.
    Flips everything across the pplan formed by n-ca and c-ca 
    Seems to increase clash but preserve bond, not sure if different from rotation
    """
    
    for frame_index in range(traj.n_frames):
        res_list = np.where(chi[frame_index] > 0.001)[0]
    
        for residue_index in res_list:

            # Get the specific residue
            residue = traj.topology.residue(residue_index)

            # Identify the atoms in the chiral center (N, CA, C, CB)
            try:
                n_idx = residue.atom('N').index
                ca_idx = residue.atom('CA').index
                c_idx = residue.atom('C').index
                cb_idx = residue.atom('CB').index
            except KeyError:
                print(f"Residue {residue_index} does not have the required atoms for chirality inversion.")
                continue

            # Get the coordinates of the chiral center atoms
            n = traj.xyz[frame_index, n_idx]
            ca = traj.xyz[frame_index, ca_idx]
            c = traj.xyz[frame_index, c_idx]
            cb = traj.xyz[frame_index, cb_idx]

            # Calculate the normal vector to the plane defined by N, CA, and C
            normal_vector = np.cross(n - ca, c - ca)
            normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the vector

            # Get the indices of all side chain atoms
            side_chain_indices = [atom.index for atom in residue.atoms if atom.name not in ['N', 'CA', 'C', 'O', 'OXT']]

            # Reflect the side chain atoms through the plane defined by N, CA, and C
            for atom_idx in side_chain_indices:
                atom_coords = traj.xyz[frame_index, atom_idx]
                reflected_coords = atom_coords - 2 * np.dot(atom_coords - ca, normal_vector) * normal_vector
                traj.xyz[frame_index, atom_idx] = reflected_coords

    return traj


def invert_chirality_reflection_ter(traj, chi):
    """
    Invert the chirality of specific residues.
    Flips everything across the plane formed by N-CA and C-CA.
    """
    
    for frame_index in range(traj.n_frames):
        res_list = np.where(chi[frame_index] > 0.001)[0]
    
        for residue_index in res_list:

            # Get the specific residue
            residue = traj.topology.residue(residue_index)

            # Identify the atoms in the chiral center (N, CA, C, CB)
            try:
                n_idx = residue.atom('N').index
                ca_idx = residue.atom('CA').index
                c_idx = residue.atom('C').index
                cb_idx = residue.atom('CB').index
                o_idx = residue.atom('O').index
            except KeyError:
                print(f"Residue {residue_index} does not have the required atoms for chirality inversion.")
                continue

            # Check if the residue is at the N-terminus or C-terminus
            is_n_terminus = False
            is_c_terminus = False
            
            if residue_index == 0:
                is_n_terminus = True
            elif residue_index == traj.n_residues-1:
                is_c_terminus = True
            elif residue.chain.index != traj.topology.residue(residue_index-1).chain.index:
                is_n_terminus = True
            elif residue.chain.index != traj.topology.residue(residue_index+1).chain.index:
                is_c_terminus = True

            # Get the coordinates of the chiral center atoms
            n = traj.xyz[frame_index, n_idx]
            ca = traj.xyz[frame_index, ca_idx]
            c = traj.xyz[frame_index, c_idx]
            cb = traj.xyz[frame_index, cb_idx]
            o = traj.xyz[frame_index, o_idx]

            if is_n_terminus:
                normal_vector = np.cross(c - ca, cb - ca)
                normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the vector
                
                # Reflect the N atom through the plane
                n_coords = traj.xyz[frame_index, n_idx]
                reflected_coords = n_coords - 2 * np.dot(n_coords - ca, normal_vector) * normal_vector
                traj.xyz[frame_index, n_idx] = reflected_coords

            elif is_c_terminus:
                normal_vector = np.cross(n - ca, cb - ca)
                normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the vector
                
                # Reflect the C and O atoms through the plane
                co_indices = [atom.index for atom in residue.atoms if atom.name in ['C', 'O', 'OXT']]
                for co_idx in co_indices:
                    co_coords = traj.xyz[frame_index, co_idx]
                    reflected_coords = co_coords - 2 * np.dot(co_coords - ca, normal_vector) * normal_vector
                    traj.xyz[frame_index, co_idx] = reflected_coords

            else:
                # Calculate the normal vector to the plane defined by N, CA, and C
                normal_vector = np.cross(n - ca, c - ca)
                normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the vector
                
                # Get the indices of all side chain atoms
                # side_chain_indices = [atom.index for atom in residue.atoms if atom.name not in ['N', 'CA', 'C', 'O', 'OXT']]
                side_chain_indices = [atom.index for atom in residue.atoms if atom.name not in ['N', 'CA', 'C']]
                # Reflect the side chain atoms through the plane defined by N, CA, and C
                for atom_idx in side_chain_indices:
                    atom_coords = traj.xyz[frame_index, atom_idx]
                    reflected_coords = atom_coords - 2 * np.dot(atom_coords - ca, normal_vector) * normal_vector
                    traj.xyz[frame_index, atom_idx] = reflected_coords

    return traj


def euler_integrator_chi_check(model, x, ca_pos, nsteps=100, x0_diff=False, t_flip=1.01, top_ref=None, keep_flip=False, type_flip='ref-ter'):
    
    # try adding small amounts of noise during integration
    
    ode_list = []
    dt = 1./(nsteps-1)
    x0 = x.detach()
    n_real = top_ref.n_atoms
    device = x.device
    
    chi_list = []
    chiral_flipped = False
    
    # select the flipping function
    if keep_flip or not chiral_flipped:
        if type_flip=='rot':
            flip_func = invert_chirality
        elif  type_flip=='ref':
            flip_func = invert_chirality_reflection
        elif  type_flip=='ref-CB':
            flip_func = invert_chirality_CB_only
        elif  type_flip=='ref-ter':
            flip_func = invert_chirality_reflection_ter
    
    for t in np.linspace(0, 1, nsteps):
        
        # Evaluate dx/dt using the model for the current state and time
        with torch.no_grad():
            if x0_diff:
                dx_dt = model(t, x.detach(), x0=x0)
            else:
                dx_dt = model(t, x.detach())

        # Compute the next state using Euler's
        x = (x + dx_dt * dt).detach() 
        
        # Along the way check for enantiomers and try flip them
        # first frame only for now
        if t > t_flip:
            if keep_flip or not chiral_flipped:
                x_ca = x + ca_pos
                # add this to the t_flip loop to save time (especially if don't keep flipping)
                trj = md.Trajectory(x_ca.cpu().numpy(), top_ref)
                chi = get_all_chiralities_vec(trj)
                trj = flip_func(trj, chi)
                x = torch.Tensor(trj.xyz).to(device) - ca_pos
                chiral_flipped = True
        
        ode_list.append(x.cpu().numpy())

    x_ca = x + ca_pos
    # final flip to ensure no enantiomiers are left
    trj = md.Trajectory(x_ca.cpu().numpy(), top_ref)
    chi = get_all_chiralities_vec(trj)
    trj = flip_func(trj, chi)
    x = torch.Tensor(trj.xyz).to(device) - ca_pos
    chiral_flipped = True
    
    ode_list.append(x.cpu().numpy())
    chi_list.append(get_all_chiralities_vec(trj))

    return np.array(ode_list)


def chirality_volumes(x, idx_n, idx_ca, idx_c, idx_cb):
    """
    x: (N,3) or (B,N,3) positions [nm]
    idx_*: LongTensor of shape (R,) with atom indices per residue
    Returns: vols of shape (B,R) or (R,) with v = ((N-CA) x (C-CA)) · (CB-CA)
    """
    batched = (x.ndim == 3)
    if not batched:
        x = x.unsqueeze(0)  # (1,N,3)

    Npos = x[:, idx_n]   # (B,R,3)
    CA   = x[:, idx_ca]
    Cpos = x[:, idx_c]
    CB   = x[:, idx_cb]

    v1 = Npos - CA
    v2 = Cpos - CA
    v3 = CB   - CA
    vols = torch.einsum('brc,brc->br', torch.cross(v1, v2, dim=-1), v3)  # (B,R)
    
    return vols if batched else vols.squeeze(0)


def _rodrigues(u, theta, device):
    # u: (B,R,3) unit axis, theta: (B,R,1)
    ux, uy, uz = u[...,0], u[...,1], u[...,2]
    zero = torch.zeros_like(ux, device=device)
    K = torch.stack([
        torch.stack([ zero, -uz,  uy], dim=-1),
        torch.stack([  uz,  zero,-ux], dim=-1),
        torch.stack([ -uy,   ux, zero], dim=-1),
    ], dim=-2)                                   # (B,R,3,3)
    I = torch.eye(3, device=device).view(1,1,3,3)
    sin_t = torch.sin(theta)[..., None]          # (B,R,1,1)
    cos_t = torch.cos(theta)[..., None]
    K2 = K @ K
    return I + sin_t * K + (1.0 - cos_t) * K2    # (B,R,3,3)


def chirality_velocity(
    x, t,
    idx_n, idx_ca, idx_c, idx_cb, idx_o, idx_side,
    idx_prev_c=None,  # list/array of C'(i-1) per residue or None
    idx_next_n=None,  # list/array of N(i+1) per residue or None
    w_sc=0.9, w_bb=0.1,         # split of rotation budget (sum needn't be 1; we renorm)
    noise=0.003,
):
    """
    Smooth chirality fix: blend side-chain χ1-like rotation and tiny backbone pivots.
    - Side chain atoms rotate about CA by theta_sc.
    - N and C rotate about CA by theta_bb; O rotates about C by theta_bb.
    - Optionally drag C'(i-1) with N and N(i+1) with C to preserve peptide bonds.
    """
    # if t < 0.3:
    #     return torch.zeros_like(x)

    dev = x.device
    B = x.shape[0] if x.ndim == 3 else 1
    x_req = x  # assuming already batched (B,N,3)

    # --- geometry for signed volume and ascent axis (same as your code) ---
    v1 = x_req[:, idx_n] - x_req[:, idx_ca]                   # (B,R,3)
    v2 = x_req[:, idx_c] - x_req[:, idx_ca]                   # (B,R,3)
    cross_nc = torch.cross(v1, v2, dim=-1)                    # (B,R,3)
    A  = cross_nc.norm(dim=-1, keepdim=True).clamp_min(1e-12) # (B,R,1)
    n  = cross_nc / A                                         # (B,R,3)

    r_cb = x_req[:, idx_cb] - x_req[:, idx_ca]                # (B,R,3)
    rr   = (r_cb*r_cb).sum(-1, keepdim=True).clamp_min(1e-12) # (B,R,1)

    base_omega = torch.cross(r_cb, n, dim=-1) / rr            # (B,R,3)

    d      = (n * r_cb).sum(-1, keepdim=True)                 # (B,R,1)
    v_tgt  = 0.003 * t**3                                     # ramp target
    d_tgt  = v_tgt / A
    gap    = (d_tgt - d).clamp_min(0.0)                       # only act if below target
    omega  = gap * base_omega                                 # (B,R,3)
    theta  = omega.norm(dim=-1, keepdim=True).clamp_min(1e-12)# (B,R,1)
    u      = omega / theta                                    # (B,R,3)

    # ---- split angles (renormalize weights) ----
    ws, wb = float(w_sc), float(w_bb)
    s = max(ws + wb, 1e-8)
    ws, wb = ws/s, wb/s

    theta_sc = ws * theta
    theta_bb = wb * theta

    # caps (in radians)
    cap_cb = 1
    
    cap_sc = max(0, min(100 * noise, 0.5) - (t/4))
    theta_cb = torch.clamp(theta_sc, max=cap_cb)
    theta_sc = torch.clamp(theta_sc, max=cap_sc)
    theta_bb = torch.clamp(theta_bb, max=cap_cb)
    R = x_req[:, idx_ca].shape[1]
    # is_terminal = x_req.new_zeros((R,), dtype=torch.bool)
    # is_terminal[0] = True
    # is_terminal[-1] = True  # handles R>=1; if R==1, both are same residue
    # rotation matrices
    R_cb = _rodrigues(u, theta_cb, dev)   # (B,R,3,3)
    R_sc = _rodrigues(u, theta_sc, dev)   # (B,R,3,3)
    R_bb = _rodrigues(u, theta_bb, dev)   # (B,R,3,3)
    dv = torch.zeros_like(x_req)
    CA = x_req[:, idx_ca]                                  # (B,R,3)
    r_cb = x_req[:, idx_cb] - CA                           # (B,R,3)
    r_cb_rot = torch.matmul(R_cb, r_cb.unsqueeze(-1)).squeeze(-1) 
    idx_cb_t = torch.as_tensor(idx_cb, device=dev, dtype=torch.long)  # (R,)

    # residues that actually have a CB
    has_cb = (idx_cb_t >= 0)
    
    # --- CB update only where CB exists ---
    if has_cb.any():
        idx_cb_v = idx_cb_t[has_cb]                       # (Rv,)
        r_cb_v   = r_cb[:, has_cb, :]                     # (B,Rv,3)
        R_cb_v   = R_cb[:, has_cb, :, :]                  # (B,Rv,3,3)
    
        r_cb_rot_v = torch.matmul(R_cb_v, r_cb_v.unsqueeze(-1)).squeeze(-1)  # (B,Rv,3)
        delta_cb_v = (r_cb_rot_v - r_cb_v)                                    # (B,Rv,3)
    
        # robust scatter-add (no advanced indexing weirdness)
        dv.index_add_(dim=1, index=idx_cb_v, source=delta_cb_v)
        
    
    side_atoms = torch.tensor([a for grp in idx_side for a in grp],
                              device=dev, dtype=torch.long)      # (A,)
    side_resix = torch.tensor([j for j, grp in enumerate(idx_side) for _ in grp],
                              device=dev, dtype=torch.long)      # (A,)

 
    # keep = is_terminal[side_resix]
    # side_atoms = side_atoms[keep]
    # side_resix = side_resix[keep]

    if side_atoms.numel() > 0:
        CA_per_atom = CA[:, side_resix]                           # (B,A,3)
        r_atom = x_req[:, side_atoms] - CA_per_atom               # (B,A,3)
        Rpa = R_sc[:, side_resix]                                 # (B,A,3,3)
        r_rot = torch.matmul(Rpa, r_atom.unsqueeze(-1)).squeeze(-1)  # (B,A,3)
        dv[:, side_atoms] += (r_rot - r_atom) 
            

    C  = x_req[:, idx_c]              # (B,R,3)

    # ---- (2) Backbone tiny pivots (θ_bb)
    # N about CA
    rN = x_req[:, idx_n] - CA
    rN_rot = torch.matmul(R_bb, rN.unsqueeze(-1)).squeeze(-1)
    dN = (rN_rot - rN)
    dv[:, idx_n] += dN 
    # C about CA
    rC = x_req[:, idx_c] - CA
    rC_rot = torch.matmul(R_bb, rC.unsqueeze(-1)).squeeze(-1)
    dC = (rC_rot - rC)
    dv[:, idx_c] += dC

    # O about C (carbonyl follows C)
    rO = x_req[:, idx_o] - C
    rO_rot = torch.matmul(R_bb, rO.unsqueeze(-1)).squeeze(-1)
    dv[:, idx_o] += (rO_rot - rO) 
    # # ---- (3) Optional rigid followers to preserve peptide bonds
    # if idx_prev_c is not None:
    #     dv[:, idx_prev_c] += dN  # move C'(i-1) like N
    # if idx_next_n is not None:
    #     dv[:, idx_next_n] += dC  # move N(i+1) like C
    
    return dv.detach()


def chirality_reflection_velocity(
    x, t,
    idx_n, idx_ca, idx_c, idx_cb, idx_o, idx_side,
    idx_prev_c=None,   # unused here (kept for drop-in compatibility)
    idx_next_n=None,   # unused here (kept for drop-in compatibility)
    noise=0.003,       # unused here (kept for drop-in compatibility)
    V_cubic_scale=0.10,
    eps_target=2e-4,
    k_max=1.6,
    oxy_kick=-0.5,     # match old: dv[o] += -0.5 * k * n_perp
):
    """
    Old-style chirality fix using a "reflection-like" push rather than Rodrigues rotations.

    This adapts `chirality_fix_tensor` to the batched/indexed style of `chirality_velocity`:
      - Compute per-residue signed volume V = <n_hat, (CB-CA)>.
      - If V < 0: "flip" branch; push to reach eps_target by t=1.
      - Else if V < V_cubic(t): "encourage" branch; push to stay on a cubic track.
      - Apply velocity along n_perp (component of n_hat perpendicular to CA->CB),
        to preserve the CA-CB bond length to first order.
      - Apply to ALL side-chain atoms (including CB), and counter-kick O.

    Args
    ----
    x : (B,N,3) or (N,3)
    t : scalar float in [0,1] (python float or 0-d tensor)
    idx_* : per-residue atom indices of shape (R,) (list/np/torch ok)
    idx_side : list length R; each element is a list of side-chain atom indices
               (typically excludes CB; we will include CB automatically)
    """

    # --- normalize x shape ---
    squeeze_out = False
    if x.ndim == 2:
        x = x.unsqueeze(0)
        squeeze_out = True
    if x.ndim != 3:
        raise ValueError(f"x must be (B,N,3) or (N,3); got shape {tuple(x.shape)}")

    dev = x.device
    dtype = x.dtype
    B, N, _ = x.shape

    # --- normalize t ---
    if isinstance(t, torch.Tensor):
        t_val = float(t.detach().item())
    else:
        t_val = float(t)

    # --- indices to tensors on device ---
    idx_n  = torch.as_tensor(idx_n,  device=dev, dtype=torch.long)
    idx_ca = torch.as_tensor(idx_ca, device=dev, dtype=torch.long)
    idx_c  = torch.as_tensor(idx_c,  device=dev, dtype=torch.long)
    idx_cb = torch.as_tensor(idx_cb, device=dev, dtype=torch.long)
    idx_o  = torch.as_tensor(idx_o,  device=dev, dtype=torch.long)
    
    R = idx_ca.numel()
    if not (idx_n.numel() == idx_c.numel() == idx_cb.numel() == idx_o.numel() == R):
        raise ValueError("idx_n/idx_ca/idx_c/idx_cb/idx_o must all have the same length (R).")

    valid_cb = (idx_cb >= 0)  # (R,)
    
    side_atoms = torch.tensor([a for grp in idx_side for a in grp],
                              device=dev, dtype=torch.long)      # (A,)
    side_resix = torch.tensor([j for j, grp in enumerate(idx_side) for _ in grp],
                              device=dev, dtype=torch.long)      # (A,)


    # if t == 0.99:
    #     print(torch.argwhere(side_resix == 46), len(side_resix))
    dv = torch.zeros_like(x)

    # ---- geometry: per-residue signed volume ----
    CA = x[:, idx_ca]                         # (B,R,3)
    v1 = x[:, idx_n] - CA                     # (B,R,3)  N-CA
    v2 = x[:, idx_c] - CA                     # (B,R,3)  C-CA
    n_vec = torch.cross(v1, v2, dim=-1)       # (B,R,3)
    A = n_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # (B,R,1)
    n_hat = n_vec / A                         # (B,R,3)

    idx_cb_safe = idx_cb.clamp_min(0)         # -1 -> 0 (arbitrary safe), will be masked out
    r_cb = x[:, idx_cb_safe] - CA             # (B,R,3)

    # mask out invalid CB residues so they *cannot* trigger flip/encourage
    valid_cb_broadcast = valid_cb.view(1, R, 1)  # (1,R,1)
    r_cb = torch.where(valid_cb_broadcast, r_cb, torch.zeros_like(r_cb))
    
    V = (n_hat * r_cb).sum(dim=-1, keepdim=True)  # (B,R,1) signed "volume" proxy
    
    # if t_val == 0.99:
    #     print(V[:,-1])
    # ---- target track / time-left ----
    V_cubic = (V_cubic_scale * t_val)          # scalar
    time_left = max(1.0 - t_val, 1e-8)                # scalar

    # ---- choose branch and compute k_needed ----
    # flip: V < 0  -> drive to eps_target
    # encourage: 0 <= V < V_cubic -> drive to V_cubic
    V_cubic_t = torch.full_like(V, V_cubic)           # (B,R,1)
    eps_t = torch.full_like(V, float(eps_target))     # (B,R,1)

    flip_mask = (V < 0.0)
    
    enc_mask  = (~flip_mask) & (V < V_cubic_t)

    delV_flip = (eps_t - V).clamp_min(0.0)            # (B,R,1)
    delV_enc  = (V_cubic_t - V).clamp_min(0.0)        # (B,R,1)

    k_flip = delV_flip / (A * time_left)
    k_enc  = delV_enc  / (A * time_left)

    k_needed = torch.zeros_like(V)
    k_needed = torch.where(flip_mask, k_flip, k_needed)
    k_needed = torch.where(enc_mask,  k_enc,  k_needed)
    k_needed = k_needed.clamp(0.0, float(k_max))      # (B,R,1)

    k_needed = torch.where(valid_cb_broadcast, k_needed, torch.zeros_like(k_needed))

    # ---- n_perp: component of n_hat perpendicular to CA->CB ----
    u_cb = r_cb / r_cb.norm(dim=-1, keepdim=True).clamp_min(1e-8)           # (B,R,3)
    proj = (n_hat * u_cb).sum(dim=-1, keepdim=True) * u_cb                  # (B,R,3)
    n_perp = n_hat - proj
    n_perp = n_perp / n_perp.norm(dim=-1, keepdim=True).clamp_min(1e-8)    # (B,R,3)

    # ---- apply velocities ----
    # Side chain (incl CB): dv[side] += k_needed * n_perp
    if side_atoms.numel() > 0:
        npa = n_perp[:, side_resix]        # (B,A,3)
        kpa = k_needed[:, side_resix]      # (B,A,1)
        dv[:, side_atoms] += kpa * npa

    
    # if t == 0.99:
    #     print(dv[:, idx_cb[-1]], dv[:, idx_cb[45]])
    # Oxygen counter-kick: dv[O] += oxy_kick * k_needed * n_perp
    # dv[:, idx_o] += float(oxy_kick) * k_needed * n_perp

    if squeeze_out:
        dv = dv.squeeze(0)
    return dv.detach()

    
