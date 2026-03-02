import os
import re
import shutil
import warnings
from collections import defaultdict, Counter, deque
from pathlib import Path
from typing import Dict, List, Tuple
import MDAnalysis as mda
import mdtraj as md
import numpy as np
from openmm import NonbondedForce
from openmm.unit import elementary_charge, kilojoule_per_mole
from src.file_config import FLOWBACK_FF
from src.utils.model import get_lj_info, get_bond_info
from pdbfixer import PDBFixer
from openmm import *
from openmm.app import *
from openmm.unit import *
import torch
from dataclasses import dataclass

def _osremove(f):
    try:
        os.remove(f)
    except FileNotFoundError:
        pass


def compute_all_distances(traj):
    idxs = np.arange(traj.top.n_atoms)
    grid = np.array(np.meshgrid(idxs, idxs)).T.reshape(-1, 2)
    pairs = grid[grid[:, 0] > grid[:, 1]]
    return md.compute_distances(traj, pairs)

_TOPLEVEL_SUBSECTIONS = {"replace", "add", "delete"}
_BLOCK_HEADER_RE = re.compile(r'^\[\s*([^\]]+?)\s*\]\s*$', re.MULTILINE)


def _is_subsection(name: str) -> bool:
    return name.strip().lower() in _TOPLEVEL_SUBSECTIONS


def _find_first_existing(files: List[Path], preferred_names: Tuple[str, ...]) -> List[Path]:
    preferred, others = [], []
    for f in files:
        (preferred if f.name in preferred_names else others).append(f)
    return preferred + others


def _gather_ff_tdb_files(ff_dir: Path, suffix: str) -> List[Path]:
    return sorted(ff_dir.glob(f"*{suffix}.tdb"))


def _extract_named_blocks_from_text(text: str, wanted: List[str]) -> Dict[str, str]:
    results: Dict[str, str] = {}
    matches = list(_BLOCK_HEADER_RE.finditer(text))
    headers = [(m.group(1).strip(), m.start(), not _is_subsection(m.group(1))) for m in matches]
    for i, (name, start, is_top) in enumerate(headers):
        if not is_top:
            continue
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        if name in wanted and name not in results:
            results[name] = text[start:end].rstrip() + "\n"
    return results


def _extract_blocks(ff_dir: Path, tdb_suffix: str, names_to_find: List[str], preferred_files: Tuple[str, ...]) -> Dict[str, str]:
    candidates = _gather_ff_tdb_files(ff_dir, tdb_suffix)
    if not candidates:
        raise RuntimeError(f"No '*{tdb_suffix}.tdb' files found in '{ff_dir}'")
    ordered_files = _find_first_existing(candidates, preferred_files)
    found: Dict[str, str] = {}
    remaining = set(names_to_find)
    for f in ordered_files:
        if not remaining:
            break
        blocks = _extract_named_blocks_from_text(f.read_text(), list(remaining))
        found.update(blocks)
        remaining -= set(blocks)
    if remaining:
        searched = ", ".join(p.name for p in ordered_files)
        missing = ", ".join(sorted(remaining))
        raise RuntimeError(
            f"Missing required terminal entries [{missing}] in {tdb_suffix} databases. Searched files: {searched}"
        )
    return found

def ensure_charmm_ff(version: str = 'auto') -> Path:
    keyword = 'charmm' if version in ('auto', 'charmm') else version
    dest = Path(FLOWBACK_FF) / f"{keyword}.ff"
    if dest.exists():
        return dest
    gmxlibrary = os.environ.get('GMXLIB')
    if not gmxlibrary:
        raise RuntimeError('$GMXLIB is not set; cannot locate CHARMM force field')
    entries = [e for e in os.listdir(gmxlibrary) if e.lower().startswith(keyword) and e.endswith('.ff')]
    if not entries:
        raise RuntimeError(f"No CHARMM force field found in GMXLIB '{gmxlibrary}'")
    entries.sort(key=lambda n: int(re.search(r"(\d+)", n).group(1)) if re.search(r"(\d+)", n) else 0)
    src = Path(gmxlibrary) / entries[-1]
    shutil.copytree(src, dest, dirs_exist_ok=True)
    n_names = ["PRO-NH2+", "GLY-NH3+", "NH3+"]
    n_blocks = _extract_blocks(src, ".n", n_names, ("merged.n.tdb",))
    c_blocks_all = _extract_blocks(src, ".c", ["COO-"], ("merged.c.tdb",))
    if "COO-" in c_blocks_all:
        c_block_text = c_blocks_all["COO-"]
    else:
        c_block_text = _extract_blocks(src, ".c", ["CTER"], ("merged.c.tdb",))["CTER"]
    for pattern in ("*.n.tdb", "*.c.tdb"):
        for f in dest.glob(pattern):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
    merged_n = "".join(n_blocks[name].rstrip() + "\n\n" for name in n_names).rstrip() + "\n"
    (dest / "merged.n.tdb").write_text(merged_n)
    (dest / "merged.c.tdb").write_text(c_block_text)
    return dest

def map_original_to_processed_indices(original_pdb, processed_pdb):
    orig_u = mda.Universe(original_pdb)
    proc_u = mda.Universe(processed_pdb)
    index_map = -1 * np.ones(len(orig_u.atoms), dtype=int)
    proc_dict = {(a.resnum, a.resname, a.name): a.ix for a in proc_u.atoms}
    for atom in orig_u.atoms:
        identifier = (atom.resnum, atom.resname, atom.name)
        if identifier in proc_dict:
            index_map[atom.ix] = proc_dict[identifier]
        elif atom.name == 'O':
            identifier = (atom.resnum, atom.resname, 'OT1')
            index_map[atom.ix] = proc_dict[identifier]
        elif atom.resname == 'ILE' and atom.name == 'CD1':
            identifier = (atom.resnum, atom.resname, 'CD')
            index_map[atom.ix] = proc_dict[identifier]
    return index_map

def silence_atoms_and_shift_charge(nbforce, topology, mute, context=None):
    # mute = [i-1 for i in mute]
    # print(mute)
    atoms = list(topology.atoms())

    shifts = defaultdict(lambda: 0.0 * elementary_charge)
    nbforce.setUseDispersionCorrection(False)

    # OpenMM topology helpers
    atoms = list(topology.atoms())
    def is_hydrogen(idx: int) -> bool:
        el = atoms[idx].element
        if el is None:
            return False
        # Robust check across OpenMM versions
        return getattr(el, "atomic_number", None) == 1 or getattr(el, "symbol", "") == "H" or getattr(el, "name", "").lower() == "hydrogen"
    # Neighbor list from OpenMM bonds()
    neighbours = defaultdict(list)
    for bond in topology.bonds():
        # Bond may be a Bond object (atom1/atom2) or a tuple(Atom, Atom)
        a1 = getattr(bond, "atom1", bond[0])
        a2 = getattr(bond, "atom2", bond[1])
        i, j = a1.index, a2.index
        neighbours[i].append(j)
        neighbours[j].append(i)

    # Zero muted particles; shift their charge to first non-muted neighbor
    for idx in mute:
        q, sigma, eps = nbforce.getParticleParameters(idx)
        # print(idx, q, sigma, eps)
        if abs(q.value_in_unit(elementary_charge)) > 1e-12:
            try:
                parent = next(n for n in neighbours[idx] if n not in mute)
                shifts[parent] += q
            except StopIteration:
                warnings.warn(
                    f"Atom {idx} is muted but has no non-muted neighbours; total charge will not be conserved!"
                )
        nbforce.setParticleParameters(idx, 0.0 * elementary_charge, sigma, 0.0 * kilojoule_per_mole)
    # Apply the accumulated charge shifts
    for idx, dq in shifts.items():
        q, sigma, eps = nbforce.getParticleParameters(idx)
        nbforce.setParticleParameters(idx, q + dq, sigma, eps)

    # Recompute exceptions with updated qi, qj; zero epsilon if any H involved
    for k in range(nbforce.getNumExceptions()):
        i, j, qprod, sigma, eps = nbforce.getExceptionParameters(k)
        qi = nbforce.getParticleParameters(i)[0]
        qj = nbforce.getParticleParameters(j)[0]
        if is_hydrogen(i) or is_hydrogen(j):
            nbforce.setExceptionParameters(k, i, j, qi * qj, sigma, 0.0 * kilojoule_per_mole)
        else:
            nbforce.setExceptionParameters(k, i, j, qi * qj, sigma, eps)



def atom_key(atom):
    """Stable identity key for set-diff. Avoids relying on indices."""
    chain = atom.residue.chain
    chain_id = getattr(chain, "id", None)
    res_id   = getattr(atom.residue, "id", None)
    # Fallbacks if IDs are None
    if chain_id is None:
        chain_id = f"chain#{chain.index}"
    if res_id is None:
        res_id = f"res#{atom.residue.index}"
    return (str(chain_id), str(res_id), atom.residue.name, atom.name)

def counter_from_topology(top):
    """Multiset of atoms keyed by identity + a lookup to final indices."""
    ctr = Counter()
    idx_lookup = defaultdict(list)
    for a in top.atoms():
        k = atom_key(a)
        ctr[k] += 1
        idx_lookup[k].append(a.index)
    return ctr, idx_lookup

def diff_added_atoms(ctr_before, ctr_after, idx_lookup_after):
    """Return list of (final_index, chain_id, res_id, resname, atomname) for added atoms."""
    added = []
    added_idxs = []
    for k, n_after in ctr_after.items():
        n_before = ctr_before.get(k, 0)
        if n_after > n_before:
            # take exactly the extra occurrences' indices from the end of the list
            new_count = n_after - n_before
            new_indices = idx_lookup_after[k][-new_count:]
            chain_id, res_id, resname, atomname = k
            for idx in new_indices:
                added.append((idx, chain_id, res_id, resname, atomname))
                added_idxs.append(idx)
    # sort by final index for readability
    return sorted(added, key=lambda x: x[0]), added_idxs

def reset_nonH_nonOXT_positions(sim: Simulation, ref_positions):
    """
    Reset positions of all atoms that are NOT hydrogens and NOT named 'OXT'
    to their coordinates in `ref_positions`.

    Args:
        sim: OpenMM Simulation with current context.
        ref_positions: Quantity[n_atoms,3] of reference coordinates (e.g. from the start).
    """
    # Get current positions from simulation
    state = sim.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True)  # Quantity in nm
    heavy_idxs = np.zeros(sim.topology.getNumAtoms(), dtype=bool)
    # Loop over topology atoms and reset conditionally
    for atom in sim.topology.atoms():
        if atom.element.symbol != "H" and atom.name != "OXT":
            pos[atom.index] = ref_positions[atom.index]
            heavy_idxs[atom.index] = 1

    # Write updated positions back
    sim.context.setPositions(pos)
    return heavy_idxs

def decompose_energy(sim):
    """
    Decompose the potential energy into contributions from each Force object in the System.
    """
    context = sim.context
    system = sim.system
    
    energy_contribs = {}
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        # Make a copy of the state but only for this force
        state = context.getState(getEnergy=True, groups=1 << i)
        e = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        energy_contribs[force.__class__.__name__] = e

    # Also collect total energy and forces
    state_total = context.getState(getEnergy=True, getForces=True)
    total_energy = state_total.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    forces = state_total.getForces(asNumpy=True)

    return energy_contribs, total_energy, forces
    
# ---------- Parameter containers ----------
@dataclass
class AtomParams:
    mass: torch.Tensor    # [N]
    sigma: torch.Tensor   # [N] (nm)
    epsilon: torch.Tensor # [N] (kJ/mol)

@dataclass
class BondParams:
    idx_i: torch.Tensor   # [B] long
    idx_j: torch.Tensor   # [B] long
    length: torch.Tensor  # [B] (nm)
    k: torch.Tensor       # [B] (kJ/mol/nm^2)

# ---------- Builders using your helpers ----------
def build_atom_params(top, rtp_data, lj_data, ff, device="cpu"):
    """
    top: mdtraj/openmm Topology (must have atoms with .name and .residue.name)
    Uses your get_lj_info(...) to assemble per-atom sigma/epsilon/mass tensors.
    """
    masses, sigmas, eps = [], [], []
    for a in top.atoms:
        m, s, e = get_lj_info(rtp_data, lj_data, a.residue.name, a.name, ff)
        masses.append(m); sigmas.append(s); eps.append(e)
    return AtomParams(
        mass=torch.tensor(masses, dtype=torch.float32, device=device),
        sigma=torch.tensor(sigmas, dtype=torch.float32, device=device),
        epsilon=torch.tensor(eps, dtype=torch.float32, device=device),
    )

def build_bond_params(top, bond_data, rtp_data, ff, device="cpu"):
    """
    Gathers harmonic bond terms using your get_bond_info(...).
    """
    idx_i, idx_j, length, k = [], [], [], []
    for b in top.bonds:
        L, K, mi, mj = get_bond_info(rtp_data, bond_data, b, ff)
        idx_i.append(b[0].index); idx_j.append(b[1].index)
        length.append(L); k.append(K)
    if len(idx_i) == 0:
        # handle molecules with no bonds (rare, but keep code robust)
        return BondParams(
            idx_i=torch.empty(0, dtype=torch.long, device=device),
            idx_j=torch.empty(0, dtype=torch.long, device=device),
            length=torch.empty(0, dtype=torch.float32, device=device),
            k=torch.empty(0, dtype=torch.float32, device=device),
        )
    return BondParams(
        idx_i=torch.tensor(idx_i, dtype=torch.long, device=device),
        idx_j=torch.tensor(idx_j, dtype=torch.long, device=device),
        length=torch.tensor(length, dtype=torch.float32, device=device),
        k=torch.tensor(k, dtype=torch.float32, device=device),
    )

# ---------- Core energy terms ----------
def lj_energy_amber(positions, atom_params, pairs_12, pairs_13, pairs_14,
                    lj_cutoff=None, amber_scnb=2.0):
    """
    AMBER-style LJ:
      - exclude 1–2 and 1–3 completely
      - include 1–4 with epsilon scaled by 1/scnb (default 0.5)
    """
    x = positions
    N = x.shape[-2]
    rij = x.unsqueeze(1) - x.unsqueeze(0)                     # [N,N,3]
    r2  = (rij * rij).sum(-1)
    r2.requires_grad_(True)
    # kill self
    r2 = r2 + torch.eye(N, device=x.device) * 1e12
    cutoff_mask = (r2 <= lj_cutoff**2) if lj_cutoff is not None else torch.ones_like(r2, dtype=torch.bool)

    # Lorentz–Berthelot mixing
    sigma_ij = 0.5 * (atom_params.sigma[:,None] + atom_params.sigma[None,:])        # [N,N]
    eps_ij   = torch.sqrt(atom_params.epsilon[:,None] * atom_params.epsilon[None,:]) # [N,N]

    inv_r6 = (sigma_ij**2 / r2).clamp_min(1e-30)**3
    inv_r6.requires_grad_(True)
    e_pair = 4.0 * eps_ij * (inv_r6**2 - inv_r6)  # base LJ
    e_pair.requires_grad_(True)
    # start from upper triangle
    mask = torch.triu(torch.ones_like(r2, dtype=torch.bool), diagonal=1) & cutoff_mask
    

    # Exclude 1–2 and 1–3
    if pairs_12 is not None and pairs_12.numel() > 0:
        mask[pairs_12[:,0], pairs_12[:,1]] = False
    if pairs_13 is not None and pairs_13.numel() > 0:
        mask[pairs_13[:,0], pairs_13[:,1]] = False
    # Handle 1–4 scaling (override those entries with scaled eps)
    if pairs_14 is not None and pairs_14.numel() > 0:
        m14 = torch.zeros_like(mask)
        m14[pairs_14[:,0], pairs_14[:,1]] = True
        mask = mask | m14  # ensure 1–4 included even if previously excluded
        e14 = 4.0 * (eps_ij / amber_scnb) * (inv_r6**2 - inv_r6)  # scale epsilon
        e14.requires_grad_(True)
        e_pair = torch.where(m14, e14, e_pair)

    return e_pair[mask].sum()

def bonds_energy(positions, bond_params: BondParams):
    """
    Harmonic bonds: 0.5 * k * (r_ij - r0)^2
    """
    if bond_params.idx_i.numel() == 0:
        return positions.sum()*0.0  # zero, keep grad graph
    xi = positions[bond_params.idx_i]    # [B,3]
    xj = positions[bond_params.idx_j]    # [B,3]
    dij = torch.linalg.norm(xi - xj, dim=-1)           # [B]
    return 0.5 * (bond_params.k * (dij - bond_params.length)**2).sum()

def chirality_energy(positions, res_maps, k_chiral=1000.0, eps_target=2e-4):
    """
    Smooth hinge on the (normalized) signed volume: E = 0.5*k*ReLU(V_target(t)-V)^2
    Mirrors your chirality_fix_tensor schedule/intent but via a scalar potential.
    """
    x = positions
    V_cubic = 0.002
    E = positions.sum()*0.0   # zero w/ grad

    for rm in res_maps:
        n, ca, c, cb = (rm[k] for k in ('n', 'ca', 'c', 'cb'))
        v1 = x[n]  - x[ca]             # N–CA
        v2 = x[c]  - x[ca]             # C–CA
        v3 = x[cb] - x[ca]             # CA→CB

        # unit normal to the plane via normalized cross product
        n_vec = torch.cross(v1, v2)                       # [3]
        A     = torch.linalg.norm(n_vec) + 1e-8
        n_hat = n_vec / A

        V = torch.dot(n_hat, v3)                          # signed (normalized) volume
        V_target = max(eps_target, V_cubic)               # emulate flip/encourage preference
        penalty = torch.relu(V_target - V)                # act only when below target
        E = E + 0.5 * k_chiral * (penalty**2)

    return E


def compute_ljb_energy(
    positions,                      # [N,3], requires_grad=True if any term is on
    top,
    atom_params,     # needed iff lj_on
    bond_params,               # needed iff bonds_on
    res_maps,                  # needed iff chirality_on
    lj_on=True,
    bonds_on=True,
    chiral_on=True,
    w_lj=1.0,
    w_bonds=1.0,
    lj_cutoff=1.2,
    amber_scnb=2.0,
    k_chiral=1000.0,
    eps_target=2e-4,
    return_parts=False,
):
    zero = positions.sum() * 0.0  # graph-friendly zero

    E_lj = zero
    if lj_on:
        pairs_12, pairs_13, pairs_14 = enumerate_12_13_14(top)
        E_lj = lj_energy_amber(positions, atom_params,  pairs_12=pairs_12, pairs_13=pairs_13, pairs_14=pairs_14, lj_cutoff=lj_cutoff, amber_scnb=amber_scnb)

    E_bonds = zero
    if bonds_on:
        if bond_params is None:
            raise ValueError("Bonds are on, but 'bond_params' is None.")
        E_bonds = bonds_energy(positions, bond_params)

   

    E_total = w_lj * E_lj + w_bonds * E_bonds
    if return_parts:
        return E_total, {"lj": E_lj, "bonds": E_bonds}
    return E_total



def build_residue_maps(top) -> List[Dict[str, object]]:
    """
    Returns a list of residue maps:
      {'n','ca','c','cb','o','side'}
    Skips residues missing the needed atoms (e.g., GLY without CB will be skipped).
    """
    res_maps = []
    backbone_names = {"N", "CA", "C", "O", "OXT"}
    for res in top.residues:
        # Collect atom indices by name
        name2idx = {a.name: a.index for a in res.atoms}
        needed = ("N", "CA", "C", "CB", "O")
        if not all(n in name2idx for n in needed):
            # Skip residues that don't have all required atoms (e.g., GLY w/o CB)
            continue

        # side chain = all atoms in residue except backbone (keep CB in side)
        side = [a.index for a in res.atoms if a.name not in backbone_names]

        res_maps.append({
            "n":   name2idx["N"],
            "ca":  name2idx["CA"],
            "c":   name2idx["C"],
            "cb":  name2idx["CB"],
            "o":   name2idx["O"],
            "side": side,
        })
    return res_maps

# -------- helper: exclusions from bonds (1-2 pairs) -------------------------
def make_exclusions_from_bonds(top, device):
    if len([b for b in top.bonds]) == 0:
        return None
    pairs = torch.tensor([[b[0].index, b[1].index] for b in top.bonds],
                         dtype=torch.long, device=device)
    return pairs

def _bond_graph(top):
    G = defaultdict(list)
    for b in top.bonds:
        i, j = b[0].index, b[1].index
        G[i].append(j); G[j].append(i)
    return G

def enumerate_12_13_14(top):
    G = _bond_graph(top)
    N = top.n_atoms
    pairs_12, pairs_13, pairs_14 = set(), set(), set()
    for s in range(N):
        # BFS up to depth 3
        q = deque([(s, 0)])
        seen = {s}
        while q:
            v, d = q.popleft()
            if d == 1: pairs_12.add(tuple(sorted((s, v))))
            elif d == 2: pairs_13.add(tuple(sorted((s, v))))
            elif d == 3: pairs_14.add(tuple(sorted((s, v))))
            if d == 3: continue
            for w in G[v]:
                if w not in seen:
                    seen.add(w); q.append((w, d+1))
    def to_tensor(S):
        return torch.tensor(sorted(S), dtype=torch.long, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')) if S else None
    return to_tensor(pairs_12), to_tensor(pairs_13), to_tensor(pairs_14)


# ---------- grouping helpers ----------
def _make_nb_like(template: NonbondedForce) -> NonbondedForce:
    f = NonbondedForce()
    f.setNonbondedMethod(template.getNonbondedMethod())
    f.setCutoffDistance(template.getCutoffDistance())
    f.setReactionFieldDielectric(template.getReactionFieldDielectric())
    f.setEwaldErrorTolerance(template.getEwaldErrorTolerance())
    f.setUseSwitchingFunction(template.getUseSwitchingFunction())
    f.setSwitchingDistance(template.getSwitchingDistance())
    return f

def split_coulomb(system: System):
    """Replace the single NonbondedForce with 4 forces:
       (LJ, Coulomb, LJ_1-4, Coulomb_1-4). Return their indices.
    """
    nb_index = None
    for i in range(system.getNumForces()):
        if isinstance(system.getForce(i), NonbondedForce):
            nb_index = i
            break
    assert nb_index is not None, "No NonbondedForce found."

    nb = system.getForce(nb_index)
    nb_Coul    = _make_nb_like(nb)   # non-1-4 Coulomb only
    nb_Coul_14 = _make_nb_like(nb)   # 1-4 Coulomb only

    # Particles
    for i_p in range(nb.getNumParticles()):
        q, sig, eps = nb.getParticleParameters(i_p)
        # Non-1-4 parts get particle params; 1-4 parts get zeros at particle level
        nb_Coul.addParticle(q, sig, 0*eps)   # Coul uses q, not epsilon
        nb_Coul_14.addParticle(0*q, sig, 0*eps)

    # Exceptions encode 1-4 interactions
    for k in range(nb.getNumExceptions()):
        i, j, qprod, sig, eps = nb.getExceptionParameters(k)
        # Turn OFF 1-4 contributions in the "non-1-4" forces:
        nb_Coul.addException(i, j, 0*qprod, sig, 0*eps)
        # Route 1-4 to the 1-4 forces:
        nb_Coul_14.addException(i, j,  qprod,  sig, 0*eps)

    # Replace original with split forces
    system.removeForce(nb_index)
    idx_Coul    = system.addForce(nb_Coul)
    idx_Coul_14 = system.addForce(nb_Coul_14)
    return idx_Coul, idx_Coul_14

def tag_force_groups(system: System, idx_Coul, idx_Coul_14):
    """
    Assign force groups:
      SR_GROUP: all bonded + 1-4 LJ/Coulomb (+ optionally non-1-4 LJ if you want it short-range)
      LR_GROUP: non-1-4 Coulomb + GB (CustomGBForce) (+ optionally non-1-4 LJ)
    """
    MISC_GROUP = 0
    BOND_GROUP = 1
    ANGLE_GROUP = 2
    TORSION_GROUP = 3
    LJ_GROUP = 4
    LJ_14_GROUP = 5
    COUL_GROUP = 6
    COUL_14_GROUP = 7
    GB_GROUP = 8

    # First, send everything to SR by default
    for i in range(system.getNumForces()):
        system.getForce(i).setForceGroup(MISC_GROUP)

    idx_bond = 99
    idx_angle = 99
    idx_torsion = 99
    idx_cnb = 99
    idx_lj14 = 99
    idx_gb = 99
    # Identify GB force(s) and move to LR
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, HarmonicBondForce):
            idx_bond = i
        if isinstance(f, HarmonicAngleForce):
            idx_angle = i
        if isinstance(f, PeriodicTorsionForce):
            idx_torsion = i
        # Both CustomGBForce and GBSAOBCForce would qualify as LR polar solvation
        if isinstance(f, CustomGBForce):
            idx_gb = i

        # CHARMM vdW (non-1-4) lives here
        if isinstance(f, CustomNonbondedForce):
            idx_cnb = i
    
        # Many CHARMM ports put 1-4 vdW as a CustomBondForce with an r^-12 - r^-6 form
        elif isinstance(f, CustomBondForce):
            expr = f.getEnergyFunction()
            e = expr.replace(" ", "").lower()
            if "^12" in e:
                idx_lj14 = i

    # force_groups = {
    #     idx_gb: GB_GROUP,
    #     idx
    # }
    system.getForce(idx_bond).setForceGroup(BOND_GROUP)
    system.getForce(idx_angle).setForceGroup(ANGLE_GROUP)
    system.getForce(idx_torsion).setForceGroup(TORSION_GROUP)
    system.getForce(idx_gb).setForceGroup(GB_GROUP)
    system.getForce(idx_lj14).setForceGroup(LJ_14_GROUP)

    # Move non-1-4 Coulomb to LR
    system.getForce(idx_cnb).setForceGroup(LJ_GROUP)
    system.getForce(idx_Coul).setForceGroup(COUL_GROUP)
    
    
    # 1-4 terms stay SR
    system.getForce(idx_Coul_14).setForceGroup(COUL_14_GROUP)

    # Non-1-4 LJ: default SR OFF (kept in SR so it's excluded from LR queries).
    # If you want to include it in LR, toggle here:
    
    
   
    return MISC_GROUP, BOND_GROUP, ANGLE_GROUP, TORSION_GROUP, LJ_GROUP, LJ_14_GROUP, COUL_GROUP, COUL_14_GROUP, GB_GROUP

def get_group_energy_forces(sim: Simulation, GROUP: int):
    st = sim.context.getState(getEnergy=True, getForces=True, groups={GROUP})
    E = st.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    F = st.getForces(asNumpy=True)  # in kJ/(mol*nm); OpenMM "N" units -> convert if needed
    return E, F


