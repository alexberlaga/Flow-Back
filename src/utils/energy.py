from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import mdtraj as md
import torch
from rdkit import rdBase
from openmm import *
import os
from openmm.app import *
from openmm.unit import *
import tempfile
import subprocess
from .energy_helpers import *
import warnings
from src.file_config import fb_temp_dir
from collections import defaultdict
from mdtraj.reporters import XTCReporter
from pdbfixer import PDBFixer
from tqdm import tqdm

warnings.filterwarnings('ignore')


class EnergyModel(torch.nn.Module):
    def __init__(self, energy_func, topology):
        super().__init__()
        self.energy_func = energy_func
        self.topology_pdb = topology


    def forward(self, x, **kwargs):
        
        return EnergyFunction.apply(x, self.energy_func, self.topology_pdb).requires_grad_(True)
        
class EnergyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, energy_func, topology):
        """
        ctx: A context object for saving information for backward computation.
        input_tensor: The input tensor for which energy is computed.
        external_energy_func: A function that returns (energy, gradient).
        REQUIRES input_tensor to be coordinates IDENTICAL to topology_pdb
        """
        input_numpy = input_tensor.detach().cpu().numpy()  # Convert to NumPy
        energies, gradients = energy_func(topology, input_numpy)
        ctx.save_for_backward(torch.from_numpy(gradients).to(input_tensor.device))  # Save gradient
        return input_tensor.new_tensor(energies).requires_grad_(True)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradient by applying the chain rule.
        """
        (external_gradient,) = ctx.saved_tensors
        result = grad_output * external_gradient
      
        return result, None, None, None  # Gradient w.r.t. input, ignore func



def charmm_traj_to_energy(topology: md.Topology, xyz: np.ndarray, ff_version: str = "auto"):
    gradients = np.zeros_like(xyz)
    energies = np.zeros(xyz.shape[0])
    for i in range(xyz.shape[0]):
        energy, gradient = charmm_structure_to_energy(topology, xyz[i:i+1], ff_version=ff_version)  # External function call
        energies[i] = energy
        gradients[i] = gradient
    #Divide by ten to convert from angstroms to nm
    return energies, gradients * angstrom / nanometer



def charmm_structure_to_energy(topology: md.Topology, xyz: np.ndarray, nonbonded=True, ff_version: str = "auto"):
    t = md.Trajectory(xyz, topology)
    # ff_dir = ensure_charmm_ff(ff_version)
    if np.max(compute_all_distances(t)) > 4 * t.top.n_residues ** 0.5:
        raise RuntimeError("Crazy Structure. Could Not Compute Energy")
    with tempfile.TemporaryDirectory(prefix=fb_temp_dir()) as temp_dir:

        pdb_file = f'{temp_dir}/temp.pdb'
        t.save_pdb(pdb_file)
    # --- AMBER14 with implicit solvent (GBn2) ---
        fixer = PDBFixer(filename=pdb_file)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        
    
        fixer.findMissingAtoms()
        ctr0, _ = counter_from_topology(fixer.topology)
        
        fixer.addMissingAtoms()
        ff = ForceField("charmm36.xml")
       
        fixer.addMissingHydrogens(pH=7.0, forcefield=ff)
        ctr1, idx_after_atoms = counter_from_topology(fixer.topology)
        added, added_idxs = diff_added_atoms(ctr0, ctr1, idx_after_atoms)
        # Define the output PDB filename
        positions = fixer.positions
        # 1. Get positions and calculate the required box size
        pos_np = np.array(fixer.positions.value_in_unit(nanometer))
        min_coords = pos_np.min(axis=0)
        max_coords = pos_np.max(axis=0)
        molecule_size = max_coords - min_coords
        padding = 2.0  # nm
        box_dim = molecule_size + padding
        
        # 2. MANDATORY: Set the box vectors on the Topology object
        # Without this, ff.createSystem(nonbondedMethod=PME) will fail.
        box_vectors = (
            Vec3(box_dim[0], 0, 0) * nanometer,
            Vec3(0, box_dim[1], 0) * nanometer,
            Vec3(0, 0, box_dim[2]) * nanometer
        )
        fixer.topology.setPeriodicBoxVectors(box_vectors)
        
        # 3. Create the System (Now PME will work because the topology has a box)
        system = ff.createSystem(
            fixer.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * nanometer,
            constraints=HBonds
        )
        added = np.asarray(added_idxs, dtype=int)
        n_atoms = system.getNumParticles()
        keep = np.ones(n_atoms, dtype=bool)
        keep[added] = False
        
        # selected_atoms = index_map
    
        mask_added_atoms(system, fixer.topology, added_idxs)
            # Set integrator
        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        context = Context(system, integrator)
        context.setPositions(positions)
    
        
        state = context.getState(getEnergy=True, getForces=True)
        
        forces = state.getForces(asNumpy=True)[keep] 
    
        energy = state.getPotentialEnergy()

    return energy.value_in_unit(kilojoules_per_mole), -1 * forces



# def mask_added_atoms(system: System, context: Context, topology, added_idxs):
#     """
#     Zero out bonded (k) and nonbonded (q, epsilon, 1-4) terms that touch any atom in added_idxs.
#     Returns a restore() function that puts everything back exactly as before.
#     """
#     added = np.asarray(added_idxs, dtype=int)
#     keep = np.ones(system.getNumParticles(), dtype=bool)
#     keep[added] = False
#     # Collect forces
#     bond = angle = proper = c_tors = nb = None
#     forces = [system.getForce(i) for i in range(system.getNumForces())]
#     for f in forces:
#         if isinstance(f, HarmonicBondForce):
#             bond = f
#         elif isinstance(f, HarmonicAngleForce):
#             angle = f
#         elif isinstance(f, PeriodicTorsionForce):
#             proper = f
#         elif isinstance(f, CustomTorsionForce):
#             c_tors = f
#         elif isinstance(f, NonbondedForce):
#             nb = f


#     if bond is not None:
#         for i in range(bond.getNumBonds()):
#             p1, p2, r0, k = bond.getBondParameters(i)
#             if not (keep[p1] and keep[p2]):
#                 bond.setBondParameters(i, p1, p2, r0, 0.0)

#     if angle is not None:
#         for i in range(angle.getNumAngles()):
#             p1, p2, p3, theta0, k = angle.getAngleParameters(i)
#             if not (keep[p1] and keep[p2] and keep[p3]):
#                 angle.setAngleParameters(i, p1, p2, p3, theta0, 0.0)

#     if proper is not None:
#         for i in range(proper.getNumTorsions()):
#             p1, p2, p3, p4, per, phase, k = proper.getTorsionParameters(i)
#             if not (keep[p1] and keep[p2] and keep[p3] and keep[p4]):
#                 proper.setTorsionParameters(i, p1, p2, p3, p4, per, phase, 0.0)

#     if c_tors is not None:
#         for i in range(c_tors.getNumTorsions()):
#             p1, p2, p3, p4, params = c_tors.getTorsionParameters(i)
#             if not (keep[p1] and keep[p2] and keep[p3] and keep[p4]):
#                 # assume params = [theta0, k]
#                 params = list(params)
#                 if len(params) >= 2:
#                     params[1] = 0.0
#                 c_tors.setTorsionParameters(i, p1, p2, p3, p4, params)

#     if nb is not None:
#         # for i in range(nb.getNumParticles()):
#         #     q, sig, eps = nb.getParticleParameters(i)
#         #     if not keep[i]:
#         #         nb.setParticleParameters(i, 0.0*q, sig, 0.0*eps)
#         # for e in range(nb.getNumExceptions()):
#         #     i, j, qprod, sig, eps = nb.getExceptionParameters(e)
#         #     if not (keep[i] and keep[j]):
#         #         nb.setExceptionParameters(e, i, j, 0.0*qprod, sig, 0.0*eps)
#         silence_atoms_and_shift_charge(nb, topology, added_idxs, context)

    
#     context.reinitialize(preserveState=True)
    
def mask_added_atoms(system, topology, added_idxs):
    """
    High-performance masking that nullifies energy of added atoms 
    without destroying the OpenMM Context.
    """
    added = set(added_idxs) # Changed to set for O(1) lookups
    for force in system.getForces():
        # Bonded Terms: Zero out the force constant (k)
        if isinstance(force, HarmonicBondForce):
            for i in range(force.getNumBonds()):
                p1, p2, r0, k = force.getBondParameters(i)
                if p1 in added or p2 in added:
                    force.setBondParameters(i, p1, p2, r0, 0.0)
            # force.updateParametersInContext(context) # Efficient GPU update

        elif isinstance(force, HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                p1, p2, p3, t0, k = force.getAngleParameters(i)
                if any(p in added for p in [p1, p2, p3]):
                    force.setAngleParameters(i, p1, p2, p3, t0, 0.0)
            # force.updateParametersInContext(context)

        elif isinstance(force, (PeriodicTorsionForce, CustomTorsionForce)):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, *params = force.getTorsionParameters(i)
                if any(p in added for p in [p1, p2, p3, p4]):
                    # Set the force constant (last param for Periodic, index 1 for Custom) to 0
                    if isinstance(force, PeriodicTorsionForce):
                        force.setTorsionParameters(i, p1, p2, p3, p4, params[0], params[1], 0.0)
                    else:
                        new_p = list(params[0]); new_p[1] = 0.0
                        force.setTorsionParameters(i, p1, p2, p3, p4, new_p)
            # force.updateParametersInContext(context)

        # CMAP: Critical for CHARMM alignment
        elif isinstance(force, CMAPTorsionForce):
            continue

        # Nonbonded: Shift charges and zero Van der Waals
        elif isinstance(force, NonbondedForce):
            silence_atoms_and_shift_charge(force, topology, added_idxs)
            # force.updateParametersInContext(context)




def generic_traj_to_energy(topology: md.Topology, xyz: np.ndarray, ff_name, solv_name, groups=None):
    gradients = np.zeros_like(xyz)
    energies = np.zeros(xyz.shape[0])
    for i in range(xyz.shape[0]):
        energy, gradient = generic_structure_to_energy(topology, xyz[i:i+1], ff_name, solv_name, groups=groups) 
        energies[i] = energy
        gradients[i] = gradient
    return energies, gradients * angstrom / nanometer

def amber_solv_traj_to_energy(topology: md.Topology, xyz: np.ndarray):
    return generic_traj_to_energy(topology, xyz, "amber14-all.xml", "implicit/gbn2.xml")

def charmm_solv_traj_to_energy(topology: md.Topology, xyz: np.ndarray, groups=None):
    return generic_traj_to_energy(topology, xyz, "charmm36.xml", "implicit/gbn2.xml", groups=groups)

def amber_solv_structure_to_energy(topology: md.Topology, xyz: np.ndarray):
    return generic_structure_to_energy(topology, xyz, "amber14-all.xml", "implicit/gbn2.xml")

def charmm_solv_structure_to_energy(topology: md.Topology, xyz: np.ndarray, groups=None):
    # return generic_structure_to_energy(topology, xyz, "charmm/charmm36.xml", None)
    return generic_structure_to_energy(topology, xyz, "charmm36.xml", "implicit/gbn2.xml", groups=groups)

def generic_structure_to_energy(topology: md.Topology, xyz: np.ndarray, ff_name, solv_name, groups=None):
    t = md.Trajectory(xyz, topology)


    with tempfile.TemporaryDirectory(prefix=fb_temp_dir()) as temp_dir:
        pdb_file = f'{temp_dir}/temp.pdb'
        t.save_pdb(pdb_file)
    # --- AMBER14 with implicit solvent (GBn2) ---
        fixer = PDBFixer(filename=pdb_file)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    
    # ---- Example usage ----
    # Build OpenMM objects from the fixed structure
    
    
    # Snapshot BEFORE adding atoms
    fixer.findMissingAtoms()
    ctr0, _ = counter_from_topology(fixer.topology)
    
    fixer.addMissingAtoms()
    
    # Snapshot AFTER addMissingAtoms, BEFORE hydrogens
    # fixer.addMissingAtoms()
    if solv_name is None:
        ff = ForceField(ff_name)
    else:
        ff = ForceField(ff_name, solv_name)
    fixer.addMissingHydrogens(pH=7.0, forcefield=ff)
    ctr1, idx_after_atoms = counter_from_topology(fixer.topology)
    added, added_idxs = diff_added_atoms(ctr0, ctr1, idx_after_atoms)
    # Define the output PDB filename
    positions = fixer.positions
    
    # print(f"\nAdded {len(added_h)} hydrogens:")
    system = ff.createSystem(
        fixer.topology,
        nonbondedMethod=NoCutoff,   # implicit solvent: NoCutoff / CutoffNonPeriodic / CutoffPeriodic only
        constraints=HBonds
    )

    if groups is not None:
        idx_Coul, idx_Coul_14 = split_coulomb(system)
            
        MISC_GROUP, BOND_GROUP, ANGLE_GROUP, TORSION_GROUP, LJ_GROUP, LJ_14_GROUP, COUL_GROUP, COUL_14_GROUP, GB_GROUP = tag_force_groups(
            system, idx_Coul, idx_Coul_14,
        )

    move_set = set(added_idxs)  # e.g., all hydrogens + all atoms in N- and C-termini
    # Build a 0/1 mask per DoF (1 = movable, 0 = frozen)
    n = system.getNumParticles()
    # mask_vecs = [Vec3(1,1,1) if i in move_set else Vec3(0,0,0) for i in range(n)]
    mask_vecs = [Vec3(1,1,1) for i in range(n)]
    # Custom "minimizer" integrator: naive steepest descent with a tiny step
    integ = CustomIntegrator(0.0)
    integ.addPerDofVariable("m", 0)         # per-DoF mask
    integ.addGlobalVariable("alpha", 1e-7)  # small step size (nm / (kJ/mol/nm)); tuned empirically
    integ.addComputePerDof("x", "x + alpha*m*f")
    integ.setConstraintTolerance(1e-3)
    integ.addConstrainPositions()
    
    

    platform = None
    platform_props = {}

    
    for name, props in [
        ("CUDA", {"DeviceIndex": "0", "Precision": "mixed", "DeterministicForces": "true"}),
        ("OpenCL", {"OpenCLPlatformIndex": "0", "OpenCLDeviceIndex": "0", "Precision": "single"}),
        ("CPU", {}),
    ]:
        try:
            platform = Platform.getPlatformByName(name)
            platform_props = props
            if platform.getName() == "CPU":
                warnings.warn("Falling back to CPU platform. This will be slower.", RuntimeWarning)
            
            break
        except Exception:
            continue

    # CREATE Simulation first
    sim = Simulation(fixer.topology, system, integ, platform, platform_props)
    sim.context.setPositions(positions)
    
    # NOW set the mask on the Context (critical)
    integ.setPerDofVariableByName("m", mask_vecs)
    # sim = Simulation(topology, system, integ, platform, platform_props)
    # sim.context.setPositions(positions)
    

    ref_positions = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
    # heavy_idxs = reset_nonH_nonOXT_positions(sim, ref_positions)
    for _ in range(500):
        sim.step(10)
        heavy_idxs = reset_nonH_nonOXT_positions(sim, ref_positions)
    for _ in range(500):
        sim.step(1)
        heavy_idxs = reset_nonH_nonOXT_positions(sim, ref_positions)
        
    if groups is not None:  
        group_dict = {
            "Misc": MISC_GROUP, 
            "Bond": BOND_GROUP, 
            "Angle": ANGLE_GROUP, 
            "Torsion": TORSION_GROUP, 
            "LJ": LJ_GROUP, 
            "LJ-14": LJ_14_GROUP, 
            "Coulomb": COUL_GROUP, 
            "Coulomb-14": COUL_14_GROUP, 
            "GB": GB_GROUP
        }
        state = sim.context.getState(getEnergy=True, getForces=True, groups=set(group_dict[g] for g in groups))
    else:
        state = sim.context.getState(getEnergy=True, getForces=True)
    
    forces = state.getForces(asNumpy=True)
    heavy_forces = forces[heavy_idxs, :] 

    energy = state.getPotentialEnergy()
    return energy.value_in_unit(kilojoules_per_mole), -1 * heavy_forces



   

# ---- Helpers to EXCLUDE selected forces from total energy ----
def _force_group_map(system):
    """Return {group_index: Force} after you've assigned per-force groups."""
    return {system.getForce(g).getForceGroup(): system.getForce(g)
            for g in range(system.getNumForces())}

def _included_groups(system, exclude_groups):
    """Return a set of group indices to include = all groups minus exclude_groups."""
    include = set()
    for g in range(system.getNumForces()):
        # we set ForceGroup(g)=g below, so 'g' is the group id
        if g not in exclude_groups:
            include.add(g)
    return include


#OLD 


def rdkit_traj_to_energy(topology: md.Topology, xyz: np.ndarray):
    gradients = np.zeros_like(xyz)
    energies = np.zeros(xyz.shape[0])
    for i in range(xyz.shape[0]):
        energy, gradient = rdkit_structure_to_energy(topology, xyz[i:i+1])  # External function call
        energies[i] = energy
        gradients[i] = gradient
    return energies, gradients

def rdkit_structure_to_energy(topology: md.Topology, xyz: np.ndarray):
    """
    Convert an MDTraj topology and XYZ coordinates to an RDKit molecule.
    
    Args:
        topology (mdtraj.Topology): MDTraj topology object.
        xyz (np.ndarray): (N, 3) array of atomic coordinates.
        
    Returns:
        Chem.Mol: RDKit molecule with 3D coordinates.
    """
    blocker = rdBase.BlockLogs()
    t = md.Trajectory(xyz, topology)
    with tempfile.TemporaryDirectory(prefix=fb_temp_dir()) as temp_dir:
        t.save_pdb(f'{temp_dir}/temp.pdb')
        mol = Chem.MolFromPDBFile(f'{temp_dir}/temp.pdb')
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol)  # Generates a 3D conformation
    # Set up the Universal Force Field (UFF)
    ff = AllChem.UFFGetMoleculeForceField(mol)
    
    # Compute Energy
    energy = ff.CalcEnergy()
    
    # Compute Gradients (negative of forces)
    num_atoms = mol.GetNumAtoms()
    
    gradients = np.array(ff.CalcGrad())
    # gradients = torch.from_numpy(gradients)
    return energy, gradients.reshape(-1, 3)
