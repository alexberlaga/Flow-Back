import argparse
import mdtraj as md
import os 
from argparse import ArgumentParser
from src.file_config import FLOWBACK_OUTPUTS, FLOWBACK_DATA, FLOWBACK_MODELS, FLOWBACK_BASE
from src.utils.run_energy import run_energy_pipeline, compute_energy
from src.utils.energy_helpers import ensure_charmm_ff
from pathlib import Path
import yaml


def setup_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--config', type=str, default=f'{FLOWBACK_BASE}/configs/eval.yaml',
                        help='Path to config file')
    parser.add_argument('--data', default='PDB', type=str,
                        help='Path to input pdbs -- Can be AA or CG')
    parser.add_argument('--model', default=f'{FLOWBACK_MODELS}/post_train', type=str,
                        help='Trained model')
    parser.add_argument('--no_angle', action='store_true', help='No angles in ff integrator')
    parser.add_argument('--hbond', action='store_true', help='hbonds in ff integrator')
    parser.add_argument('--reflect_chi', action='store_true', help='Reflect, not rotate, chiralities')
    parser.add_argument('--ckp', default=None, type=str, help='Checkpoint for given mode')
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Use raw data, not a model"
    )
    return parser


def config_to_args(config):
    return argparse.Namespace(**config)


def get_args():
    parser = ArgumentParser()
    parser = setup_args(parser)
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config_args = config_to_args(config)
    return args, config_args



def main() -> None:
    
    args, config_args = get_args()
    
    ff_label = config_args.ff
    output_dir = Path(FLOWBACK_OUTPUTS) / "energies"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.benchmark:
        pdb_dir = Path(FLOWBACK_DATA) / str(args.data + "_clean_AA")
        output_file = output_dir / f"energies_{ff_label}_{args.data}_benchmark.npy"
    elif args.ckp is None or args.ckp.lower() == 'none':
        pdb_dir = Path(FLOWBACK_OUTPUTS) / args.data / args.model
        output_file = output_dir / f"energies_{args.data}_{args.model}.npy"
    else:
        base_dir = Path(FLOWBACK_OUTPUTS) / args.data
        model_name = os.path.basename(args.model.rstrip(os.sep))
        # if model_name == 'pre_train':
        #     ff_label_ = 'LJBC'
        # if args.ff != '':
        #     ff_label_ = args.ff
        # else:
        #     ff_label_ = ff_label
        pattern = f"{model_name}_ckp-{args.ckp}_*noise-{config_args.CG_noise}"
        if config_args.solver == 'euler':
            pattern += "_euler"
        if config_args.solver == 'euler_chi' or config_args.solver == 'euler_chi_old':
            pattern += "_eulerchi"

        if args.no_angle:
            pattern += "_noangle"
        elif args.hbond:
            pattern += "_hbond"
        elif args.reflect_chi:
            pattern += "_reflect"
        matches = sorted(base_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No directory matching {pattern} in {base_dir}")
        pdb_dir = matches[0]
        model_ckp = pdb_dir.name
        output_file = output_dir / f"energies_{args.data}_ff-{config_args.ff}_{model_ckp}.npy"
        
    
    pdb_paths = sorted(str(p) for p in pdb_dir.glob("*.pdb"))
    if config_args.ff.lower() == 'charmm':
        run_energy_pipeline(pdb_paths, str(output_file), ff=config_args.ff)
    else:
        run_energy_pipeline(pdb_paths, str(output_file), ff=config_args.ff)
        

if __name__ == "__main__":
    main()

