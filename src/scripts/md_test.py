import subprocess
import tempfile
import shutil
import numpy as np
from openmm import *
import os
from openmm.app import *
from openmm.unit import *
import argparse
import glob
import mdtraj as md
import MDAnalysis as mda
from collections import defaultdict
from src.file_config import FLOWBACK_DATA, FLOWBACK_OUTPUTS
from src.utils.energy_helpers import *

def _osremove(f):
    try:
        os.remove(f)
    except FileNotFoundError:
        pass

def check_nan_positions(simulation):
    """Check for NaN positions in the system and print the atom indices."""
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    positions = state.getPositions(asNumpy=True)
    potential_energy = state.getPotentialEnergy()

    # Check if any position contains NaN
    nan_mask = np.any(np.isnan(positions / unit.nanometer), axis=1)
    if np.any(nan_mask):
        nan_indices = np.where(nan_mask)[0]
        print(f"NaN detected in positions at atom indices: {nan_indices}")
        for i in nan_indices:
            print(f"Atom {i} Position: {positions[i]}")

    # Check for NaN energy
    if potential_energy._value is None or np.isnan(potential_energy._value):
        print("NaN detected in potential energy!")

def _map_original_to_processed_indices(original_pdb, processed_pdb):
    """
    Create a mapping from original PDB atom indices to processed PDB indices after pdb2gmx.
    
    Parameters:
    -----------
    original_pdb : str
        Path to the original PDB file
    processed_pdb : str
        Path to the processed PDB file (after pdb2gmx)
        
    Returns:
    --------
    list
        List where index position is the original atom index and value is the new index.
        -1 indicates no match was found for that original atom.
    """
    # Load both structures
    orig_u = mda.Universe(original_pdb)
    proc_u = mda.Universe(processed_pdb)
    
    # Initialize mapping list with -1 (indicating no match)
    n_orig_atoms = len(orig_u.atoms)
    index_map = -1 * np.ones(n_orig_atoms).astype(int)
    
    # Create unique identifiers for each atom in processed structure
    proc_dict = {}
    for atom in proc_u.atoms:
        identifier = (atom.resnum, atom.resname, atom.name)
        proc_dict[identifier] = atom.ix
    
    # Map original atoms to processed atoms
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
    """
    Zero LJ and charge on each atom in *mute* and transfer that charge
    to its first bonded neighbour that is **not** in *mute*.

    Parameters
    ----------
    nbforce : openmm.NonbondedForce
        The force we are modifying.
    topology : openmm.app.Topology
        Needed to discover bonding so we know where to move the charge.
    mute : iterable of int
        Particle indices to silence.
    context : openmm.Context, optional
        Pass the existing Context if you already created one so we can
        push the new parameters with `updateParametersInContext`.
    """
    mute    = set(int(i) for i in mute)
    shifts  = defaultdict(lambda: 0.0*elementary_charge)   # chsarge → heavy atom
    nbforce.setUseDispersionCorrection(False)
    # nbforce.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
    # ------------------------------------------------------------------
    # 1. build a quick neighbour map from the Topology
    # ------------------------------------------------------------------
    neighbours = defaultdict(list)
    for bond in topology.bonds:
        i, j = bond[0].index, bond[1].index
        neighbours[i].append(j)
        neighbours[j].append(i)

    # ------------------------------------------------------------------
    # 2. zero LJ & charge on the muted atoms, remember the charge removed
    # ------------------------------------------------------------------
    for idx in mute:
        q, sigma, eps = nbforce.getParticleParameters(idx)

        # accumulate the charge so we can add it to the parent later
        if abs(q.value_in_unit(elementary_charge)) > 1e-12:
            # first bonded neighbour that is NOT muted
            try:
                parent = next(n for n in neighbours[idx] if n not in mute)
                shifts[parent] += q
            except StopIteration:
                warnings.warn(f"Atom {idx} is muted but has no non-muted neighbours;"
                              " total charge will not be conserved!")

        # zero LJ *and* charge
        nbforce.setParticleParameters(idx,
                                      0.0*elementary_charge,
                                      sigma,
                                      0.0*kilojoule_per_mole)

    # ------------------------------------------------------------------
    # 3. apply the accumulated charge shifts to the parent atoms
    # ------------------------------------------------------------------
    for idx, dq in shifts.items():
        q, sigma, eps = nbforce.getParticleParameters(idx)
        nbforce.setParticleParameters(idx, q + dq, sigma, eps)

    # ------------------------------------------------------------------
    # 4. update every exception so chargeProd matches the new charges
    # ------------------------------------------------------------------
    for k in range(nbforce.getNumExceptions()):
        i, j, qprod, sigma, eps = nbforce.getExceptionParameters(k)
        
        qi = nbforce.getParticleParameters(i)[0]
        qj = nbforce.getParticleParameters(j)[0]
        if topology.atom(i).element.name == 'hydrogen' or topology.atom(j).element.name =='hydrogen':
            nbforce.setExceptionParameters(k, i, j, qi*qj, sigma, 0.0*kilojoule_per_mole)
        else:
            nbforce.setExceptionParameters(k, i, j, qi*qj, sigma, eps)
        



def run_simulation(index, pdb_file, ff_name='charmm36.xml', solv_name='implicit/gbn2.xml'):
    print(f"Starting simulation {index} with {pdb_file}...")
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

   

    move_set = set(added_idxs)  # e.g., all hydrogens + all atoms in N- and C-termini
    # Build a 0/1 mask per DoF (1 = movable, 0 = frozen)
    n = system.getNumParticles()
    # mask_vecs = [Vec3(1,1,1) if i in move_set else Vec3(0,0,0) for i in range(n)]
    mask_vecs = [Vec3(1,1,1) for i in range(n)]
    # Custom "minimizer" integrator: naive steepest descent with a tiny step
    integ_min = CustomIntegrator(0.0)
    integ_min.addPerDofVariable("m", 0)         # per-DoF mask
    integ_min.addGlobalVariable("alpha", 1e-7)  # small step size (nm / (kJ/mol/nm)); tuned empirically
    integ_min.addComputePerDof("x", "x + alpha*m*f")
    integ_min.setConstraintTolerance(1e-3)
    integ_min.addConstrainPositions()
    
    timestep = 0.002*picoseconds 
    integ_sim = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, timestep)
    integrator = CompoundIntegrator()
    integrator.addIntegrator(integ_min)   
    integrator.addIntegrator(integ_sim) 
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
    try:
        sim = Simulation(fixer.topology, system, integrator, platform, platform_props)
        sim.context.setPositions(positions)
        system.addForce(CMMotionRemover(1000))
        # NOW set the mask on the Context (critical)
        integ_min.setPerDofVariableByName("m", mask_vecs)
        integrator.setCurrentIntegrator(0)
        
        ref_positions = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        for _ in range(500):
            sim.step(10)
            heavy_idxs = reset_nonH_nonOXT_positions(sim, ref_positions)
        for _ in range(500):
            sim.step(1)
            heavy_idxs = reset_nonH_nonOXT_positions(sim, ref_positions)
        
        integrator.setCurrentIntegrator(1)
        sim.context.setVelocitiesToTemperature(300*kelvin)
        
        # Run a short step to check stability before full simulation
        sim.step(100)

        # Extract energy and check for NaNs
        state = sim.context.getState(getEnergy=True, getForces=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        forces = state.getForces(asNumpy=True)
    
        force_mags = np.linalg.norm(forces, axis=1)[heavy_idxs]

        if potential_energy != potential_energy:  # NaN check
            print(f"Simulation {index} failed: NaN in potential energy.")
            return False, np.max(force_mags)

        # Run the full simulation
        sim.step(20000)

    except Exception as e:
        print(f"Simulation {index} crashed: {e}")
        return False, np.max(force_mags)

    print(f"Simulation {index} completed successfully.")
    return True, np.max(force_mags)

def run_simulation_gmx(index, pdb_file, noh, save):
    print(f"Starting simulation {index} with {pdb_file}...")

    # Use a temporary directory to avoid unwanted backup files
    with tempfile.TemporaryDirectory(prefix='/project2/andrewferguson/berlaga') as temp_dir:
        if save:
            temp_dir = 'saved_trajs'
        structure_file = f"{temp_dir}/structure_{index}.gro"
        topology_file = f"{temp_dir}/topol_{index}.top"

        commands = [
           "gmx_mpi", "pdb2gmx", "-f", pdb_file, "-o", structure_file, "-p", topology_file,
            "-ff", "charmm27", "-water", "spce", "-ter", "-nobackup", "-quiet", "-i", f"{temp_dir}/posre"
        ]
        
        process = subprocess.Popen(commands, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate('0\n0\n')
        
        if process.returncode != 0:
            print(f"Simulation {index} failed at pdb2gmx step. Error:\n{stderr}")
            return False, np.inf
        
        index_map = _map_original_to_processed_indices(pdb_file, structure_file)
        _osremove('posre.itp')
        try:
            # Load topology and coordinates
            gro = GromacsGroFile(structure_file)
            original_box = gro.getPeriodicBoxVectors()
            expanded_box = (
                original_box[0] + Vec3(2, 0, 0) * nanometer,
                original_box[1] + Vec3(0, 2, 0) * nanometer,
                original_box[2] + Vec3(0, 0, 2) * nanometer,
            )
            topology = GromacsTopFile(topology_file, periodicBoxVectors=expanded_box, includeDir=temp_dir)
            positions = gro.positions
            
            # Create simulation system
            system = topology.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer,
                                           constraints=HBonds)
            if noh:
                t2 = md.load(structure_file)
        
                selected_atoms = index_map
                new_bond_force = HarmonicBondForce()
                new_bond_force.setForceGroup(10)
                new_angle_force = HarmonicAngleForce()
                new_angle_force.setForceGroup(11)
                new_torsion_force = PeriodicTorsionForce()
                new_torsion_force.setForceGroup(12)
                new_custom_torsion_force = CustomTorsionForce("0.5*k*(thetap-theta0)^2; thetap = step(-(theta-theta0+pi))*2*pi+theta+step(theta-theta0-pi)*(-2*pi); pi = 3.14159265358979")
                new_custom_torsion_force.addPerTorsionParameter('theta0')
                new_custom_torsion_force.addPerTorsionParameter('k')
                new_custom_torsion_force.setForceGroup(13)
                
                for force in enumerate(system.getForces()):
                    force_ = force[1]
                    if isinstance(force_, HarmonicBondForce):
                        for i in range(force_.getNumBonds()):
                            p1, p2, length, k = force_.getBondParameters(i)
                            if p1 in selected_atoms and p2 in selected_atoms:
                                new_bond_force.addBond(p1, p2, length, k)
                
                    elif isinstance(force_, HarmonicAngleForce):
                        for i in range(force_.getNumAngles()):
                            p1, p2, p3, angle, k = force_.getAngleParameters(i)
                            if p1 in selected_atoms and p2 in selected_atoms and p3 in selected_atoms:
                                new_angle_force.addAngle(p1, p2, p3, angle, k)
                
                    elif isinstance(force_, PeriodicTorsionForce):
                        for i in range(force_.getNumTorsions()):
                            p1, p2, p3, p4, periodicity, phase, k = force_.getTorsionParameters(i)
                            if p1 in selected_atoms and p2 in selected_atoms and p3 in selected_atoms and p4 in selected_atoms:
                                new_torsion_force.addTorsion(p1, p2, p3, p4, periodicity, phase, k)
        
                    
                    elif isinstance(force_, CustomTorsionForce):
                        for i in range(force_.getNumTorsions()):
                            p1, p2, p3, p4, params = force_.getTorsionParameters(i)
                            if p1 in selected_atoms and p2 in selected_atoms and p3 in selected_atoms and p4 in selected_atoms:
                                new_custom_torsion_force.addTorsion(p1, p2, p3, p4, params)
                # Remove old forces and add the new ones
                for fi in range(len(system.getForces()) - 1, -1, -1):
                    force_ = system.getForce(fi)
                    if isinstance(force_, CMAPTorsionForce):
                        force_.setForceGroup(14)
                        continue
                    elif isinstance(force_, NonbondedForce):
                        mute_atoms = np.setdiff1d(np.arange(force_.getNumParticles()), selected_atoms)
                        silence_atoms_and_shift_charge(force_, t2.top, mute_atoms)
                        force_.setForceGroup(15)
                        continue
                    else:
                        system.removeForce(fi)  # Remove each old force
                
                system.addForce(new_bond_force)
                system.addForce(new_angle_force)
                system.addForce(new_torsion_force)
                system.addForce(new_custom_torsion_force)
            # Set integrator
            timestep = 0.002*picoseconds if noh else 0.001*picoseconds
            integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, timestep)
            system.addForce(CMMotionRemover(1000))

            # # Create simulation
            # for i in range(Platform.getNumPlatforms()):
            try:
                platform = Platform.getPlatformByName('OpenCL') if Platform.getNumPlatforms() > 1 else Platform.getPlatformByName('CPU')
            except:
                platform = Platform.getPlatformByName('CUDA') if Platform.getNumPlatforms() > 1 else Platform.getPlatformByName('CPU')
            simulation = Simulation(topology.topology, system, integrator, platform)
            simulation.context.setPositions(positions)
            simulation.context.setVelocitiesToTemperature(300*kelvin)

            # Run a short step to check stability before full simulation
            simulation.step(100)

            # Extract energy and check for NaNs
            state = simulation.context.getState(getEnergy=True, getForces=True)
            potential_energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
            forces = state.getForces(asNumpy=True)
        
            force_mags = np.linalg.norm(forces, axis=1)[index_map]

            if potential_energy != potential_energy:  # NaN check
                print(f"Simulation {index} failed: NaN in potential energy.")
                return False, np.max(force_mags)

            # Run the full simulation
            simulation.step(20000)

        except Exception as e:
            print(f"Simulation {index} crashed: {e}")
            return False, np.max(force_mags)

        print(f"Simulation {index} completed successfully.")
        return True, np.max(force_mags)


def main():
    parser = argparse.ArgumentParser(description="MD Test")
    parser.add_argument("--model", help="Input Model")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--noise", type=str)
    parser.add_argument("--solver", type=str)
    parser.add_argument("--protein", help="Input protein")
    parser.add_argument("--benchmark", action='store_true')
    parser.add_argument("--nosolver", action='store_true')
    parser.add_argument("--noh", action='store_true')
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--num_files", "-n", type=int, default=0, help="number of files")
    
    args = parser.parse_args()
    model = args.model
    noise = args.noise
    checkpoint = args.checkpoint
    if not args.nosolver:
        solver = args.solver
    
    if args.benchmark:
        pdb_files = glob.glob(f"{FLOWBACK_DATA}/{args.protein}_clean_AA/*.pdb")
        stat_file = f"{FLOWBACK_OUTPUTS}/stat_files/{args.protein}_benchmark_stats.txt"
    elif args.nosolver: 
        pdb_files = glob.glob(f"{FLOWBACK_OUTPUTS}/{args.protein}/{model}_ckp-{checkpoint}_noise-{noise}/*.pdb")
        stat_file = f"{FLOWBACK_OUTPUTS}/stat_files/{args.protein}_{model}_ckp-{checkpoint}_noise-{noise}_stats.txt"
    elif args.num_files == 0:
        pdb_files = glob.glob(f"{FLOWBACK_OUTPUTS}/{args.protein}/{model}_ckp-{checkpoint}_noise-{noise}_{solver}/*.pdb")
        stat_file = f"{FLOWBACK_OUTPUTS}/stat_files/{args.protein}_{model}_ckp-{checkpoint}_noise-{noise}_{solver}_stats.txt"
    else:
        pdb_files = [f"{FLOWBACK_OUTPUTS}/{args.protein}/{model}_noise-0.003/frame_{i}_1.pdb" for i in range(args.num_files)]
        stat_file = f"{FLOWBACK_OUTPUTS}/stat_files/{args.protein}_{model}_stats.txt"
    
    if args.noh:
        stat_file = f'{stat_file[:-4]}_noh.txt'
    _osremove(stat_file)
    failed_simulations = []
    for i, pdb_file in enumerate(pdb_files):
    #     if args.save:
    #         success, max_force = run_simulation(i, pdb_file, args.noh, True)
    #         if not success:
    #             failed_simulations.append(i)
    #     else:
        with open(stat_file, 'a') as f:
            success, max_force = run_simulation(i, pdb_file)
            if not success:
                failed_simulations.append(i)
            f.write(f'{i}\t{success}\t{max_force:.1f}\n')

    print("\nSummary:")
    print(f"Total simulations: {len(pdb_files)}")
    print(f"Failed simulations: {len(failed_simulations)}")
    if failed_simulations:
        print("Failed simulations indices:", failed_simulations)


if __name__ == "__main__":
    main()

