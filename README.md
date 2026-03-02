# FlowBack & FlowBack-Adjoint

**FlowBack**: Generative backmapping of coarse-grained molecular systems using an equivariant graph neural network and a flow-matching objective. Implementation of https://openreview.net/forum?id=mhUasr0j5X.

**FlowBack-Adjoint**: Extends FlowBack with an **adjoint matching** scheme that adds **physics-aware, energy-guided corrections** during **post-training**. You can post-train the original FlowBack model using **RDKit** energies or **CHARMM** energies (**recommended**). For CHARMM-based post-training you must have a working installation of **GROMACS** and the desired **CHARMM** force field (e.g., CHARMM27/CHARMM36). **Inference/integration is unchanged** relative to the original FlowBack pipeline.

---

## Installation

### Env setup
Clone the repo and run the setup script to create and activate a local virtual environment using `pip`:

    git clone https://github.com/alexberlaga/Flow-Back.git
    cd Flow-Back
    source setup.sh

If you prefer Conda, use the provided environment specification instead:

    conda env create -f environment.yml
    conda activate flowback_env

All commands below use the pattern:

    python -m src.scripts.<script> [options] --config configs/<script>.yaml

### Quick test
    python -m src.scripts.eval --load_dir PDB_example --config configs/eval.yaml

---

## Inference (FlowBack & FlowBack-Adjoint)

The **integration (inference)** procedure is identical for base FlowBack and FlowBack-Adjoint models. Edit `configs/eval.yaml` to control parameters such as `n_gens`, `CG_noise`, and clash checking.

Generate samples for coarse-grained traces:

    python -m src.scripts.eval --load_dir PDB_example --config configs/eval.yaml

To compute bond, clash, or diversity metrics, set `retain_AA`, `check_bonds`, `check_clash`, and `check_div` in `configs/eval.yaml` and run the same command.

To increase diversity, adjust `CG_noise` in the config and rerun the command.

Backmap a short (10-frame) CG trajectory:

    python -m src.scripts.eval --load_dir pro_traj_example --config configs/eval.yaml

---

## Pre-Training (base FlowBack)

Download training PDBs and pre-processed features from:

    https://zenodo.org/records/13375392

Unzip and move `train_features` into the "inputs" folder of the working directory. Edit `configs/pre_train.yaml` to specify `load_path` and `top_path` for the feature and topology pickles.

Run pre-training:

    python -m src.scripts.pre_train --config configs/pre_train.yaml

---

## Post-Training with FlowBack-Adjoint (Energy-Guided)

**Goal**: Refine a pre-trained FlowBack model by incorporating **energy terms** in an adjoint matching objective. Edit `configs/post_train.yaml` to choose the energy backend (`ff`), energy-loss weight (`lam`), and other training hyperparameters.

If `ff` is set to `CHARMM`, ensure that both GROMACS and the desired CHARMM force field are installed; they are required to compute CHARMM energies during post-training.

Run post-training:

    python -m src.scripts.post_train --config configs/post_train.yaml

---

## Running `run_energies` on evaluation outputs

After you've generated PDB files via evaluation, compute their CHARMM energies with:
```
python -m src.scripts.run_energies --data PDB_example --model post_train --checkpoint 7000 --noise 0.003
```
The script searches `outputs/PDB_example` for a directory matching
`post_train_ckp-7000_noise-0.003/`, processes all contained .pdb files, and saves the resulting energy array to `outputs/energies/energies_PDB_example_post_train_ckp-7000_noise-0.003.npy`.

---

## Repository layout (updated)

This repository now includes training, evaluation, post-processing, force-field data, checkpoints, and tests.

### Top-level files

- `README.md`: project overview and usage.
- `pyproject.toml`: project/package and tool configuration.
- `environment.yml`: Conda environment definition.
- `setup.sh`: convenience script to create and activate a local virtual environment.
- `pytest.ini`: pytest defaults.

### `src/` (core package)

- `src/conditional_flow_matching.py`: main flow-matching model logic.
- `src/adjoint.py`: adjoint/energy-guided training objective and utilities.
- `src/chirality_predictor.py`: chirality classifier used for structure checks.
- `src/file_config.py`: central path/config helpers used by scripts.
- `src/utils/`: shared utility code for models, energy terms, evaluation, and chi/chirality:
  - `model.py`, `evaluation.py`, `chi.py`
  - `energy.py`, `energy_helpers.py`, `run_energy.py`
- `src/egnn_pytorch_se3/`: EGNN building blocks and geometric utilities.

### `src/scripts/` (CLI entry points)

All scripts can be run as `python -m src.scripts.<name> ...`.

- `pre_train.py`: base FlowBack training.
- `post_train.py`: FlowBack-Adjoint post-training.
- `eval.py`: inference/backmapping and metric generation.
- `train.py`: generic training entry point/helper.
- `run_energies.py`: batch CHARMM energy evaluation on generated PDBs.
- `get_val_energies.py`: validation-energy extraction helper.
- `featurize_pro.py`: protein featurization/preprocessing.
- `md_test.py`: molecular-dynamics style sanity/integration test helper.

### `configs/`

- `pre_train.yaml`: pre-training config.
- `post_train.yaml`: post-training config (standard).
- `post_train_no_solv.yaml`: post-training config variant without solvent terms.
- `eval.yaml`: inference/evaluation config (standard).
- `eval_no_solv.yaml`: inference/evaluation config variant without solvent terms.

### `tests/`

- `test_end_to_end.py`: end-to-end workflow checks.
- `test_scripts_cli.py`: CLI-level tests.
- `test_model_utils.py`: model utility tests.
- `test_energy.py`: energy-function tests.
- `test_chi.py`: chi/chirality tests.
- `conftest.py`: shared fixtures.

### Data and assets

- `data/`
  - `PDB_example/`: example coarse-grained protein inputs.
  - `pro_traj_example/`: short protein trajectory example.
  - `DNApro_example/`: DNA-protein complex examples.
  - `DNApro_traj_example/`: DNA-protein trajectory example.
- `forcefield/`
  - `aminoacids.rtp`, `bondtypes.csv`, `ffnonbonded.csv`: force-field lookup tables/files used in physics-aware routines.
- `models/`
  - `pre_train/`: example pre-trained checkpoint + params/config.
  - `post_train/`: example post-trained checkpoint + config.
  - `old/`: legacy checkpoints retained for compatibility/reference.
- `chirality_checkpoints/`
  - `ckpt_epoch_0075.pt`: trained chirality predictor checkpoint.
- `notebooks/run_energies.ipynb`: notebook workflow for energy analysis.

## Cite as

    @article{jones2025flowback,
      title={FlowBack: A Generalized Flow-Matching Approach for Biomolecular Backmapping},
      author={Jones, Michael S and Khanna, Smayan and Ferguson, Andrew L},
      journal={Journal of Chemical Information and Modeling},
      year={2025},
      publisher={ACS Publications}
    }

    @misc{berlaga2025flowbackadjointphysicsawareenergyguidedconditional,
      title={FlowBack-Adjoint: Physics-Aware and Energy-Guided Conditional Flow-Matching for All-Atom Protein Backmapping},
      author={Alex Berlaga and Michael S. Jones and Andrew L. Ferguson},
      year={2025},
      eprint={2508.03619},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2508.03619}
    }
