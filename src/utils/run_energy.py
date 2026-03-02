import mdtraj as md
import multiprocessing as mp
import numpy as np
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from scipy.stats import sem
import torch
import re
mp.set_start_method("forkserver", force=True)

from .energy import (
    amber_solv_structure_to_energy,
    charmm_structure_to_energy,
    charmm_solv_structure_to_energy
)

from .model import get_amber_data

def _detect_openmm_platform():
    """Return the first available OpenMM platform in preference order."""
    try:
        from openmm import Platform          # OpenMM >= 8
    except ImportError:
        from simtk.openmm import Platform     # OpenMM 7.x

    names = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
    for pref in ("CUDA", "OpenCL", "HIP", "CPU"):
        if pref in names:
            return pref
    return "CPU"
def keep_numeric(s):
    return re.sub(r'[^0-9\.\-]', '', s)
def compute_energy(pdb_path: str, ff: str, charmm_ff: str = "auto", w_lj=1.0, w_bonds=1.0) -> float | None:
    """Return the potential energy for a single PDB file using the chosen backend."""
    try:
        pdb = md.load(pdb_path)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        energy_funcs = {
            "CHARMM": lambda top, xyz: charmm_structure_to_energy(
                top, xyz, nonbonded=True,
            ),
            "charmm": lambda top, xyz: charmm_structure_to_energy(
                top, xyz, nonbonded=True,
            ),
            "CHARMM-solv": charmm_solv_structure_to_energy,
            "amber": amber_solv_structure_to_energy,
            
        }
        energy_func = energy_funcs[ff]
        energy, _ = energy_func(pdb.top, pdb.xyz)
        print(keep_numeric(pdb_path) + ':', energy)
        return energy
    except Exception as e:
        print(f"Error processing {pdb_path}: {e}")
        return None
        
def run_energy_pipeline(
    pdb_paths: list[str],
    output_file: str,
    ff: str,
    charmm_ff: str = "auto",
    w_lj: float = 1.0,
    w_bonds: float = 1.0
) -> None:
    """Compute energies for all paths and save them to *output_file*."""
    n_total = len(pdb_paths)
    print(f"Computing energies for {n_total} structures -> {output_file}")
    print(w_lj, w_bonds)
    plat = _detect_openmm_platform()
    print("Using:", plat, 'n_workers:', min(mp.cpu_count(), 8))

    if plat == "OpenCL":
        mp.set_start_method("spawn", force=True)
        with mp.get_context("spawn").Pool(processes=min(mp.cpu_count(), 8)) as pool:
            func = partial(compute_energy, ff=ff, charmm_ff=charmm_ff, w_lj = w_lj, w_bonds = w_bonds)
            energies = pool.map(func, pdb_paths)
    else:
        with mp.Pool(processes=min(mp.cpu_count(), 8)) as pool:
            func = partial(compute_energy, ff=ff, charmm_ff=charmm_ff, w_lj = w_lj, w_bonds = w_bonds)
            energies = pool.map(func, pdb_paths)
   
    energies = np.array([e for e in energies if e is not None])
    print(
        f"Finished. {n_total - len(energies)} structures failed. "
        f"Median: {np.median(energies[~np.isnan(energies)])} +/- {sem(energies[~np.isnan(energies)])} kJ/mol. \n"
        f"Energies saved to {output_file}."
    )
    np.save(output_file, energies)

    
