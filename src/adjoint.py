import torch
import torch.nn as nn
import torch.optim as optim
from .utils.model import *
from .utils.energy import *
from datetime import datetime
import numpy as np
import mdtraj as md
import os
import gc
from .utils.chi import *
from copy import deepcopy



energy_funcs = {
        'RDKit': rdkit_traj_to_energy,
        'CHARMM': charmm_traj_to_energy,
        'amber': amber_solv_traj_to_energy,
        'CHARMM-solv': charmm_solv_traj_to_energy,
        'CHARMM-lr': lambda topology, xyz: charmm_solv_traj_to_energy(topology, xyz, groups=["Bond", "Coulomb", "Coulomb-14"])
    }
energy_constants = {
    'RDKit': {'alpha': 0, 'k_bond': 0, 'k_angle': 0, 'k_lj': 0, 'k_hbond': 1.0, 'k_chiral': 10.0},
    'amber': {'alpha': 10, 'k_bond': 1, 'k_angle': 1, 'k_lj': 1, 'k_hbond': 1.0, 'k_chiral': 10.0},
    'CHARMM': {'alpha': 20, 'k_bond': 1.0, 'k_angle': 0.0, 'k_lj': 0.1, 'k_hbond': 0.0, 'k_chiral': 10.0},
    'CHARMM-solv': {'alpha': 12, 'k_bond': 2.0, 'k_angle': 0.5, 'k_lj': 1.5, 'k_hbond': 1.0, 'k_chiral': 10.0},
    'CHARMM-lr': {'alpha': 12, 'k_bond': 2.0, 'k_angle': 0.5, 'k_lj': 1.5, 'k_hbond': 1.0, 'k_chiral': 10.0},
}

def sigma(t, dt, cg_noise):
    return cg_noise * torch.sqrt((2 * (1 - t + dt)) / (t + dt))

def stochastic_trajectory(v_finetune, sigma_t, **kwargs):
    n_coords = kwargs.get('n_coords')
    batch_size = kwargs.get('batch_size')
    topology = kwargs.get('topology')
    timesteps = kwargs.get('num_steps')
    cg_noise = kwargs.get('cg_noise')
    device = kwargs.get('device')
    int_ff = kwargs.get('int_ff')
    ff = kwargs.get('ff')
    int_chi = kwargs.get('int_chi')
    beta = kwargs.get('beta')
    lj_cache = kwargs.get('lj_cache')
    bond_cache = kwargs.get('bond_cache')
    angle_cache = kwargs.get('angle_cache')
    hbond_cache = kwargs.get('hbond_cache')
    use_angles = kwargs.get('use_angles')
    use_hbonds = kwargs.get('use_hbonds')
    if 'solv' not in ff:
        MAX_LJ = 5.0
        MAX_BOND = 1.0
        W_BOND = 1.0
        use_angles = False
        use_hbonds = False
        MAX_ANGLE = 0.0
        MAX_HBOND = 0.0
        MAX_CHIRAL = 0.2
        T_LJ = 0.85
        T_BOND = 0.95
    else:
        MAX_LJ = 0.5
        MAX_BOND = 2.0
        MAX_ANGLE = 0.5
        MAX_HBOND = 0.5
        MAX_CHIRAL = 0.2
        T_LJ = 0.5
        T_BOND = 0.75

    ca_pos = torch.tensor(kwargs.get('ca_pos'), device=device)
  
    charmm_ff = kwargs.get('charmm_ff', 'auto')
    
    energy_func = energy_funcs[ff]
    alpha = energy_constants[ff]['alpha']
    k_bond = energy_constants[ff]['k_bond']
    k_angle = energy_constants[ff]['k_angle'] if use_angles else 0
    k_lj = energy_constants[ff]['k_lj']
    k_hbond = energy_constants[ff]['k_hbond']
    k_chiral = energy_constants[ff]['k_chiral']
    dt = torch.tensor(1.0 / timesteps)
    count = 0
    trajectory = torch.zeros(batch_size, timesteps + 1, n_coords, 3, device=device)
    trajectory[:, 0, :, :] = torch.randn(batch_size, n_coords, 3).to(device) * cg_noise  # Samples from initial Gaussian
    x_t = trajectory[:, 0, :, :]
    res_maps = build_residue_maps(topology)
    idx_n  = torch.as_tensor([res.atom('N').index  for res in topology.residues],  dtype=torch.long, device=device)
    idx_ca = torch.as_tensor([res.atom('CA').index for res in topology.residues],  dtype=torch.long, device=device)
    idx_c  = torch.as_tensor([res.atom('C').index  for res in topology.residues],  dtype=torch.long, device=device)
    idx_o =  torch.as_tensor([res.atom('O').index  for res in topology.residues],  dtype=torch.long, device=device)
    idx_cb = torch.as_tensor([res.atom('CB').index if any(a.name=='CB' for a in res.atoms) else -1
                              for res in topology.residues], dtype=torch.long, device=device)
    idx_side = [[a.index for a in res.atoms if a.name not in ['C', 'CA', 'N', 'O']]
                                                  for res in topology.residues]
    
 
    for t in range(timesteps):
        t_val = t / timesteps
        alpha_t = (t + 1) / timesteps
        is_last = (t == timesteps - 1)
        
    
        with torch.no_grad():
            ff_velocity = torch.zeros_like(x_t)
            if int_ff:
                xt_cap = x_t + ca_pos
                clamp_lj   = lambda z: torch.clamp(k_lj * lj_velocity_fn(z, t_val, dt, lj_cache),  -MAX_LJ,  MAX_LJ)
                clamp_bond = lambda z: torch.clamp(k_bond * bond_velocity_fn(z, t_val, dt, bond_cache), -MAX_BOND,  MAX_BOND)
                clamp_angle = lambda z: torch.clamp(k_angle * angle_velocity_fn(z, dt, angle_cache), -MAX_ANGLE, MAX_ANGLE)
                if use_hbonds:
                    clamp_hbond = lambda z: torch.clamp(k_hbond * hbond_velocity_fn(z, dt, hbond_cache), -MAX_HBOND, MAX_HBOND)
                
                stack = lambda f: torch.stack([f(xt_cap[i:i+1]) for i in range(x_t.shape[0])])
                # if use_hbonds and t_val > 0.5:
                #     ff_velocity += custom_function(t_val) * stack(clamp_hbond)
         
                if t_val > T_LJ:
                    ff_velocity = (t_val ** alpha) * stack(clamp_lj)
                    if use_hbonds:
                        ff_velocity += custom_function(t_val) * stack(clamp_hbond)
                    if use_angles and t_val > T_BOND:
                        ff_velocity += (t_val**alpha) * stack(clamp_angle)
                    if t_val > T_BOND:
                        ff_velocity += (t_val**alpha) * stack(clamp_bond)

                
                # ff_velocity += torch.clamp(k_chiral * chirality_reflection_velocity(x_t, t_val, idx_n, idx_ca, idx_c, idx_cb, idx_o, idx_side), -MAX_CHIRAL, MAX_CHIRAL)
    
            elif int_chi:
                # Chirality ascent field (uses a local tape w.r.t. x only; no link to model)
                ff_velocity = torch.clamp(k_chiral * chirality_reflection_velocity(x_t, t_val, idx_n, idx_ca, idx_c, idx_cb, idx_o, idx_side), -MAX_CHIRAL, MAX_CHIRAL)
                
    
            base = v_finetune(t_val, x_t)
            if is_last:
                drift = base + (ff_velocity if (int_ff or int_chi) else 0.0)
                diffusion = torch.zeros_like(x_t)
            else:
                drift = 2 * (base) + ff_velocity - (1 / alpha_t) * x_t 
                diffusion = sigma_t[t] * torch.randn_like(x_t)
    
            x_t = x_t + dt * drift + torch.sqrt(dt) * diffusion
            trajectory[:, t + 1, :] = x_t

    
    energy_model = EnergyModel(energy_func, topology)
    
    frame = trajectory[:, -1, :, :].requires_grad_(True) + ca_pos
    energy = energy_model(frame)
    frame.retain_grad()

    
    energy.backward()
    returned_grads = frame.grad.view(batch_size, n_coords * 3)   
    cur_traj = trajectory[0] + ca_pos
   
       
    return trajectory, energy.item(), returned_grads



def lean_adjoint_ode(X, v_base, grad, **kwargs):
    job_dir = kwargs.get('job_dir')
    device = kwargs.get('device')
    topology = kwargs.get('topology')
    lam = kwargs.get('lam')
    n_coords = kwargs.get('n_coords')
    timesteps = kwargs.get('num_steps') 
    batch_size = kwargs.get('batch_size')
    max_grad = kwargs.get('max_grad')
    torch.tensor(kwargs.get('ca_pos'), device=device)
    int_ff = kwargs.get('int_ff')
    ff = kwargs.get('ff')
    lj_cache = kwargs.get('lj_cache')
    bond_cache = kwargs.get('bond_cache')
    angle_cache = kwargs.get('angle_cache')
    hbond_cache = kwargs.get('hbond_cache')
    use_angles = kwargs.get('use_angles')
    use_hbonds = kwargs.get('use_hbonds')
    ca_pos = torch.tensor(kwargs.get('ca_pos'), device=device)
    charmm_ff = kwargs.get('charmm_ff', 'auto')
   
    if 'solv' not in ff:
        MAX_LJ = 5.0
        MAX_BOND = 1.0
        W_BOND = 1.0
        use_angles = False
        use_hbonds = False
        MAX_ANGLE = 0.0
        MAX_HBOND = 0.0
        MAX_CHIRAL = 0.2
        T_LJ = 0.85
        T_BOND = 0.95
    else:
        MAX_LJ = 0.5
        MAX_BOND = 2.0
        MAX_ANGLE = 0.5
        MAX_HBOND = 0.5
        MAX_CHIRAL = 0.2
        T_LJ = 0.5
        T_BOND = 0.75
    
    dt = 1.0 / timesteps
    a_t = torch.zeros(batch_size, timesteps+1, n_coords * 3, device=device)
    
    a_t[:, -1, :] =  torch.clamp(lam * grad, -max_grad, max_grad)
    alpha = energy_constants[ff]['alpha']
    k_bond = energy_constants[ff]['k_bond']
    k_angle = energy_constants[ff]['k_angle'] if use_angles else 0
    k_lj = energy_constants[ff]['k_lj']
    k_hbond = energy_constants[ff]['k_hbond']
    k_chiral = energy_constants[ff]['k_chiral']

    

    res_maps = build_residue_maps(topology)
    idx_n  = torch.as_tensor([res.atom('N').index  for res in topology.residues],  dtype=torch.long, device=device)
    idx_ca = torch.as_tensor([res.atom('CA').index for res in topology.residues],  dtype=torch.long, device=device)
    idx_c  = torch.as_tensor([res.atom('C').index  for res in topology.residues],  dtype=torch.long, device=device)
    idx_cb = torch.as_tensor([res.atom('CB').index if any(a.name=='CB' for a in res.atoms) else -1
                              for res in topology.residues], dtype=torch.long, device=device)
    idx_o =  torch.as_tensor([res.atom('O').index  for res in topology.residues],  dtype=torch.long, device=device)
    idx_side = [[a.index for a in res.atoms if a.name not in ['C', 'CA', 'N', 'O']]
                                                  for res in topology.residues]
    
    for t in range(timesteps - 1, -1, -1):
        alpha_t = (t+1) / timesteps
        alpha_t_dot = 1
        t_val = t / timesteps
        xgrad = X[:, t, :].requires_grad_(True)

        xt_cap = xgrad + ca_pos
        clamp_lj   = lambda z: torch.clamp(k_lj * lj_velocity_fn(z, t_val, dt, lj_cache),  -MAX_LJ,  MAX_LJ)
        clamp_bond = lambda z: torch.clamp(k_bond * bond_velocity_fn(z, t_val, dt, bond_cache), -MAX_BOND,  MAX_BOND)
        clamp_angle = lambda z: torch.clamp(k_angle * angle_velocity_fn(z, dt, angle_cache), -MAX_ANGLE, MAX_ANGLE)
        if use_hbonds:
            clamp_hbond = lambda z: torch.clamp(k_hbond * hbond_velocity_fn(z, dt, hbond_cache), -MAX_HBOND, MAX_HBOND)
        
        stack = lambda f: torch.stack([f(xt_cap[i:i+1]) for i in range(xgrad.shape[0])])
        ff_velocity = torch.zeros_like(xgrad)
        if int_ff:
            if t_val > T_LJ:
                ff_velocity = (t_val ** alpha) * stack(clamp_lj)
                if use_hbonds:
                    ff_velocity += custom_function(t_val) * stack(clamp_hbond)
                if use_angles and t_val > T_BOND:
                    ff_velocity += (t_val**alpha) * stack(clamp_angle)
                if t_val > T_BOND:
                    ff_velocity += (t_val**alpha) * stack(clamp_bond)

            # ff_velocity += torch.clamp(k_chiral * chirality_velocity(xgrad, t_val, idx_n, idx_ca, idx_c, idx_cb, idx_o, idx_side), -MAX_CHIRAL, MAX_CHIRAL)
    
        
        
        grad_input = lambda xi: (2 * v_base(t_val, xi.view(1, n_coords, 3)) + ff_velocity - (alpha_t_dot / alpha_t) * xi.view(1, n_coords, 3)).flatten()

        vjp = torch.stack([torch.autograd.functional.vjp(grad_input, xgrad[i].flatten(), a_t[i, t + 1, :])[0] for i in range(batch_size)])
        with torch.no_grad():
            a_t[:, t, :] = torch.clamp((a_t[:, t + 1, :] + dt * vjp).detach(), -max_grad, max_grad)
    return a_t
    
def adjoint_matching_loss(X, v_finetune, v_base, a_t, selected_timesteps, sigma_t, **kwargs):
    device = kwargs.get('device')
    timesteps = kwargs.get('num_steps')
    batch_size = kwargs.get('batch_size')
    n_coords = kwargs.get('n_coords')
    cg_noise = kwargs.get('cg_noise')
    loss = torch.tensor(0.0, device=device)
    
    for k, t in enumerate(selected_timesteps):
        t_val = t / timesteps
        diff = (v_finetune(t_val, X[:, t, :, :]) - v_base(t_val, X[:, t, :, :])).view(batch_size, n_coords * 3)
        loss_t = torch.sum(torch.norm((2 / sigma_t[k]) * diff * (cg_noise ** 2) + sigma_t[k] * a_t[:, t, :], dim=1) ** 2)
        loss += loss_t
        
    return loss


def trajectory_and_adjoint(v_base, v_finetune,  **kwargs):
    n_coords = kwargs.get('n_coords')
    cg_noise = kwargs.get('cg_noise')
    batch_size = kwargs.get('batch_size')
    num_steps = kwargs.get('num_steps')
    job_dir = kwargs.get('job_dir')
    device = kwargs.get('device')
    compare = kwargs.get('compare')
    sigma_all = sigma(torch.linspace(0, 1, num_steps + 1), 1 / num_steps, cg_noise)
    xyz, energy, gradients = stochastic_trajectory(v_finetune, sigma_all, **kwargs)
    if compare:
        return energy
    else:
        traj = xyz.view(batch_size, num_steps+1, n_coords, 3)
        with open(f'{job_dir}/energies.out', 'a') as f:
            f.write(str(energy) + '\n')
        a_t = lean_adjoint_ode(traj, v_base, gradients, **kwargs)
        return traj, a_t, energy


    


