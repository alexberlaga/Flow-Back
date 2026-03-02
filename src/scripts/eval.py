### input directory to pdbs/trajs and return N generated samples of each ###

import os
import glob
import pickle as pkl
from tqdm import tqdm
import time
import datetime
import yaml
import argparse
from argparse import ArgumentParser
import random
from src.file_config import FLOWBACK_OUTPUTS, FLOWBACK_DATA, FLOWBACK_MODELS, FLOWBACK_BASE, fb_temp_dir

# need to test these for preproccessing
from src.utils.evaluation import *

# import functions to check and correct chirality
from src.utils.chi import *
from src.utils.model import get_charmm_data, get_amber_data


def setup_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--config', type=str, default=f'{FLOWBACK_BASE}/configs/eval.yaml',
                        help='Path to config file')
    parser.add_argument('--load_dir', default='PDB', type=str,
                        help='Path to input pdbs -- Can be AA or CG')
    parser.add_argument('--model_path', default=f'{FLOWBACK_MODELS}/post_train', type=str,
                        help='Trained model')
    parser.add_argument('--no_angle', action='store_true', help='No angles in ff integrator')
    parser.add_argument('--ckp', default='7000', type=str, help='Checkpoint for given mode')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing files')
    parser.add_argument('--hbond', action='store_true', help='add hbond in integrator')
    parser.add_argument('--chi_test', action='store_true', help='Chirality Test Mode')
    parser.add_argument('--save_chi', action='store_true', help='Save Chiralities')
    parser.add_argument('--no_checks', action='store_true', help='Do not check quantities')
    parser.add_argument('--repeat', type=int, default=0, help='repeat number')
    parser.add_argument('--reflect_chi', action='store_true', help='Reflect, not rotate, Chiralities')
    parser.add_argument('--external', action='store_true', help='Data not in data folder')
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

if __name__ == "__main__":
    args, config_args = get_args()
    
    load_dir = args.load_dir
    model_path = args.model_path
    ckp = args.ckp

    LJ_CACHE = {}
    BOND_CACHE = {}
    ANGLE_CACHE = {}
    HBOND_CACHE = {}
    
    CG_noise = config_args.CG_noise
    n_gens = config_args.n_gens
    solver = config_args.solver
    stride = config_args.stride
    check_clash = config_args.check_clash and not args.no_checks
    check_bonds = config_args.check_bonds and not args.no_checks
    check_div = config_args.check_div and not args.no_checks
    mask_prior = config_args.mask_prior
    retain_AA = config_args.retain_AA
    tol = config_args.tolerance
    nsteps = config_args.nsteps
    t_flip = config_args.t_flip
    type_flip = config_args.type_flip
    system = config_args.system
    vram = config_args.vram
    save_traj = config_args.save_traj
    save_dcd = config_args.save_dcd
    overwrite = config_args.overwrite or args.overwrite
    save_dir_cfg = config_args.save_dir
    ff = config_args.ff
    external = config_args.external or args.external
    no_angle = args.no_angle
    hbond = args.hbond
    save_chi = args.save_chi
    chirality_test_mode = args.chi_test
    os.environ['FLOWBACK_TEMP_DIR_LOC'] = getattr(config_args, "temp_dir_loc", "~")
    reflect_chi = args.reflect_chi or getattr(config_args, "reflect_chi", False)
    CACHE_BYTES = {
        "lj": 0,
        "bond": 0,
        "angle": 0,
        "hbond": 0,
    }

    
    if save_dir_cfg == '':
        save_dir = f'{FLOWBACK_OUTPUTS}/{load_dir}'
        split_path = model_path.split("/")[-1]
        if split_path == '':
            split_path = model_path.split("/")[-2]
        if solver == 'euler_ff':
            if reflect_chi:
                save_prefix = f'{save_dir}/{split_path}_ckp-{ckp}_noise-{CG_noise}_reflect/'
            elif no_angle:
                save_prefix = f'{save_dir}/{split_path}_ckp-{ckp}_noise-{CG_noise}_noangle/'
            elif hbond:
                save_prefix = f'{save_dir}/{split_path}_ckp-{ckp}_noise-{CG_noise}_hbond/'
            else:
                save_prefix = f'{save_dir}/{split_path}_ckp-{ckp}_noise-{CG_noise}/'
        elif solver == 'euler':
            save_prefix = f'{save_dir}/{split_path}_ckp-{ckp}_noise-{CG_noise}_euler/'
        elif solver == 'euler_chi' or solver == 'euler_chi_old':
            save_prefix = f'{save_dir}/{split_path}_ckp-{ckp}_noise-{CG_noise}_eulerchi/'
    else:
        save_dir = save_dir_cfg + '/'
        save_prefix = save_dir_cfg + '/'
    
    if not external:
        load_dir = f'data/{load_dir}'
    
    os.makedirs(save_prefix, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chiral_model_path = "models/chirality"
    chiral_ckp = 0
    chiral_egnn = load_model(chiral_model_path, chiral_ckp, device)
    CHIRAL_PRED_MODEL = load_chirality_predictor(CHIRAL_CKPT_PATH, base_egnn=chiral_egnn, device=device)
    CHIRAL_PRED_MODEL.eval()
    
    # add optional preprocessing to save files -- skip if already cleaned
    if system == 'pro':
        if retain_AA:
            load_dir = process_pro_aa(load_dir)
        else:
            load_dir = process_pro_cg(load_dir)
    elif system == 'DNApro':
        if retain_AA:
            # accounted for in featurization
            pass
        else:
            pass
            # either combine standard order with protein function (for aa and cg)
            # or call each seperately and recombine (keep pro-DNA order consistent)
            #load_dir = process_DNApro_cg(load_dir)
            
    else:
        print('Invalid system type')
    
    # load model
    model = load_model(model_path, ckp, device) 
    
    # Track scores
    bf_list, clash_list, div_list = [], [], []
    
    # save time for inference as a function of size -- over n-res?
    time_list, res_list = [], []
    trj_list = sorted(glob.glob(f'{load_dir}/*.pdb'))
    repeat = args.repeat
    if chirality_test_mode and repeat > 0:
        n = len(trj_list)
        start_idx = int((repeat - 1) / 84 * n)
        end_idx = int(repeat / 84 * n)
        trj_list = trj_list[start_idx:end_idx]
    print(f'Found {len(trj_list)} trajs to backmap')

    if 'CHARMM' in ff:
        rtp_data, lj_data, bond_data, angle_data = get_charmm_data()
    else:
        rtp_data, lj_data, bond_data, angle_data = get_amber_data()
        
    for trj_name in tqdm(trj_list, desc='Iterating over trajs'):
        save_name = f'{save_prefix}{trj_name.split("/")[-1]}'
        if save_traj:
            save_fn = save_name.replace('.pdb', f'_dt.pdb')
        else:
            save_fn = save_name.replace('.pdb', f'_1.pdb')

        
        
        if overwrite or (not os.path.exists(save_fn) and not os.path.exists(save_fn.replace('.pdb', '.pt'))):
            trj = md.load(trj_name)[::stride]
            n_frames = trj.n_frames
            start_time = datetime.datetime.now()
        
            # load features for the given topology and system type
            if system=='pro':
                res_ohe, atom_ohe, xyz, aa_to_cg, mask, n_atoms, top = load_features_pro(trj)
            elif system=='DNApro':
                res_ohe, atom_ohe, xyz, aa_to_cg, mask, n_atoms, top = load_features_DNApro(trj)
            else:
                print('Invalid system type')
            if device.type == 'cpu':
                print("building LJ Cache")
            lj_cache = get_or_build_cache(top, LJ_CACHE, 'lj', CACHE_BYTES, build_lj_cache, rtp_data, lj_data, ff, device, onefour_scale=0.5, cutoff=None)
            
            if device.type == 'cpu':
                print("building Bond Cache")
            bond_cache = get_or_build_cache(top, BOND_CACHE, 'bond', CACHE_BYTES, build_bond_cache, rtp_data, bond_data, ff, device)
            if device.type == 'cpu':
                print("building Angle Cache")
            angle_cache = get_or_build_cache(top, ANGLE_CACHE, 'angle', CACHE_BYTES, build_angle_cache, rtp_data, angle_data, ff, device, heavy_only=True)
            if device.type == 'cpu':
                print("building hbond Cache")
            hbond_cache = get_or_build_cache(top, HBOND_CACHE, 'hbond', CACHE_BYTES, build_hbond_cache, rtp_data, ff, device)
            
            test_idxs = list(np.arange(n_frames))*n_gens
            xyz_ref = xyz[test_idxs]
             
            print(f'{trj_name.split("/")[-1]}   {n_frames} frames   {n_atoms} atoms   {n_gens} samples')
        
            # ensure input will fit into specified VRAM (16GB by default)
            n_iters = int(len(test_idxs) * len(res_ohe) / (vram*6_000)) + 1
            idxs_lists = split_list(test_idxs, n_iters)
            print(f'breaking up into {n_iters} batches:\n')
                  
            xyz_gen = []
            frames_10 = []
            frames_20 = []
            frames_30 = []
            frames_40 = []
            prior = np.random.randn(n_gens, xyz_ref.shape[1], xyz_ref.shape[2]) * CG_noise
            
            for n, test_idxs in enumerate(idxs_lists):
                # print(test_idxs)
                n_test = len(test_idxs)
                print(f'iter {n+1} / {n_iters}')
                
                xyz_test_real = [xyz[i] for i in test_idxs]
                # print([x_[0] for x_ in xyz_test_real])
                map_test =      [aa_to_cg]*n_test
                mask_test =     [mask]*n_test
                res_test =      [res_ohe]*n_test
                atom_test =     [atom_ohe]*n_test
                ca_pos_test =   get_ca_pos(xyz_test_real, map_test)
                # wrap model -- update this so that the function multiplies by the dim of n_gens * n_frames 
                model_wrpd = ModelWrapper(model=model, 
                                feats=torch.tensor(np.array(res_test)).int().to(device), 
                                mask=torch.tensor(np.array(mask_test)).bool().to(device), 
                                atom_feats=torch.tensor(np.array(atom_test)).to(device),
                                ca_pos=torch.tensor(np.array(ca_pos_test)).to(device))
        
                # tensors that are constant across frames for this traj
                res_t  = torch.as_tensor(res_ohe, dtype=torch.int16).cpu()
                atom_t = torch.as_tensor(atom_ohe, dtype=torch.int16).cpu()
                mask_t = torch.as_tensor(mask, dtype=torch.bool).cpu()
                a2cg_t = torch.as_tensor(aa_to_cg).cpu()
                # prior is per-gen in your current code; map frame -> gen_idx
            
                with torch.no_grad():
                    if solver == 'euler_chi':
                        idx_n  = torch.as_tensor([res.atom('N').index  for res in top.residues],  dtype=torch.long, device=device)
                        idx_ca = torch.as_tensor([res.atom('CA').index for res in top.residues],  dtype=torch.long, device=device)
                        idx_c  = torch.as_tensor([res.atom('C').index  for res in top.residues],  dtype=torch.long, device=device)
                        idx_cb = torch.as_tensor([res.atom('CB').index if any(a.name=='CB' for a in res.atoms) else -1
                                                  for res in top.residues], dtype=torch.long, device=device)
                        idx_side = [[a.index for a in res.atoms if a.name not in ['C', 'CA', 'N', 'O']]
                                                  for res in top.residues]
                        
                        # If some residues lack CB (e.g., Gly), either mask them out or map their CB to CA so volume=0.
                        mask_valid = (idx_cb >= 0)
                        # For simplicity, map invalid CBs to CA so their contribution is sigmoid(beta*0)=0.5 and grad is zero:
                        idx_cb = torch.where(mask_valid, idx_cb, idx_ca)
                        ode_traj = euler_integrator_with_chirality(model_wrpd, torch.tensor(prior, dtype=torch.float32).to(device), idx_n, idx_ca, idx_c, idx_cb, idx_side)
                    elif solver == 'euler_chi_old':
                        ode_traj = euler_integrator_chi_check(model_wrpd, 
                                          torch.tensor(prior, dtype=torch.float32).to(device), torch.tensor(ca_pos_test).to(device),
                                              nsteps=nsteps, t_flip=t_flip, top_ref=top, type_flip=type_flip)
                    elif solver == 'euler_ff':
                        idx_n  = torch.as_tensor([res.atom('N').index  for res in top.residues],  dtype=torch.long, device=device)
                        idx_ca = torch.as_tensor([res.atom('CA').index for res in top.residues],  dtype=torch.long, device=device)
                        idx_c  = torch.as_tensor([res.atom('C').index  for res in top.residues],  dtype=torch.long, device=device)
                        idx_o  = torch.as_tensor([res.atom('O').index  for res in top.residues],  dtype=torch.long, device=device)
                        idx_cb = torch.as_tensor([res.atom('CB').index if any(a.name=='CB' for a in res.atoms) else -1
                                                  for res in top.residues], dtype=torch.long, device=device)
                        idx_side = [[a.index for a in res.atoms if a.name not in ['C', 'CA', 'N', 'O']]
                                                  for res in top.residues]
                        ff_params = {
                            # data
                            "ca_pos": torch.tensor(ca_pos_test, dtype=torch.float32, device=device),
                        
                            # caches / topology / forcefield context
                            "lj_cache": lj_cache,
                            "bond_cache": bond_cache,
                            "angle_cache": angle_cache,
                            "hbond_cache": hbond_cache,
                            "top": top,
                            "ff": ff,
                            "device": device,
                        
                            # atom index maps
                            "idx_n": idx_n,
                            "idx_ca": idx_ca,
                            "idx_c": idx_c,
                            "idx_o": idx_o,
                            "idx_cb": idx_cb,
                            "idx_side": idx_side,
                        
                            # integrator / options
                            "nsteps": 100,
                            "alpha": 12,
                            "no_angle": no_angle,
                            "hbond": hbond,
                            "chirality_test_mode": chirality_test_mode,
                            "chiral_pred": CHIRAL_PRED_MODEL,
                            "chiral_threshold": 0.3,
                            # explicitly into EGNN / wrapper
                            "res_ohe": res_t,
                            "atom_ohe": atom_t,
                            "cg_mask": mask_t,
                            # what you asked to store
                            "aa_to_cg": a2cg_t,
                            "cg_noise": CG_noise,
                            "reflect_chi": reflect_chi,
                        }
                        if 'solv' not in ff:
                            print('no solv')
                            ff_params.update({
                                "T_LJ": 0.85,
                                "T_BOND": 0.95,
                                "alpha": 20,
                            
                                "no_angle": True,
                                "hbond": False,
                            
                                "W_LJ": 0.1,
                                "W_BOND": 1.0,
                                "W_CHIRAL": 1.0,
                                "MAX_LJ": 5.0,
                                "MAX_BOND": 1.0,
                            })
                        ode_traj = euler_ff_integrator(model_wrpd, torch.tensor(prior, dtype=torch.float32).to(device), ff_params)
                        # ode_traj = euler_ff_integrator(model_wrpd, torch.tensor(prior,
                        #                                                 dtype=torch.float32).to(device), torch.tensor(ca_pos_test).to(device), lj_cache, bond_cache, angle_cache, hbond_cache, top, ff, device, idx_n, idx_ca, idx_c, idx_o, idx_cb, idx_side, no_angle=no_angle, hbond=hbond, chirality_test_mode=chirality_test_mode, chiral_egnn=chiral_egnn) 
                    elif solver == 'euler':
                        ode_traj = euler_integrator(model_wrpd, torch.tensor(prior,
                                                                        dtype=torch.float32).to(device))
               
    
                # end time and save
                time_diff = datetime.datetime.now() - start_time
                time_list.append(time_diff.total_seconds())
                res_list.append(trj.n_residues)
                # save trj -- optionally save ODE integration not just last structure -- only for one gen
                if save_traj:
                    xyz_gen.append(ode_traj.squeeze() + ca_pos_test)
                else:
                    if chirality_test_mode:
                        frames_10.append(ode_traj[10])
                        frames_20.append(ode_traj[20])
                        frames_30.append(ode_traj[30])
                        frames_40.append(ode_traj[40])
                    xyz_gen.append(ode_traj[-1] + ca_pos_test) 

            if chirality_test_mode:
                frames_10 = np.concatenate(frames_10)
                frames_20 = np.concatenate(frames_20)
                frames_30 = np.concatenate(frames_30)
                frames_40 = np.concatenate(frames_40)


            xyz_gen = np.concatenate(xyz_gen)
                  
            # don't include DNA virtual atoms in top 
            aa_idxs = top.select(f"not name DS and not name DP and not name DB")
            trj_gens = md.Trajectory(xyz_gen[:, :top.n_atoms], top).atom_slice(aa_idxs)
            trj_refs = md.Trajectory(xyz_ref[:, :top.n_atoms], top).atom_slice(aa_idxs)
        
            # Can only calculate bonds and div if an AA reference is provided
            if not chirality_test_mode:
                
                if check_bonds:
                    bf = [bond_fraction(t_ref, t_gen) for t_gen, t_ref in zip(trj_gens, trj_refs)]
                    bf = np.array(bf).reshape(n_frames, n_gens)
                    bf_list.append(bf)
                      
                # protein only clash only for now
                if check_clash:
                    clash = [clash_res_percent(t_gen) for t_gen in trj_gens]
                    clash = np.array(clash).reshape(n_frames, n_gens)
                    print('clash', np.mean(clash))
                    clash_list.append(clash)
                      
                # Need multiple gens to calculate diversity
                if check_div:
                    div_frames = []
                    for f in range(n_frames):
                        trj_ref_div = trj_refs[f]
                        trj_gens_div = trj_gens[f::n_frames]
                        div, _ = sample_rmsd_percent(trj_ref_div, trj_gens_div)
                        div_frames.append(div)
                    div_list.append(div_frames)
        
            # save gen using same pdb name -- currently saving as n_frames * n_gens
            save_name = f'{save_prefix}{trj_name.split("/")[-1]}'
                  
            # if saving dt, only save a single gen
            
            if save_traj:
                save_n = save_name.replace('.pdb', f'_1.pdb')
                save_i = save_name.replace('.pdb', f'_dt.pdb')
                trj_gens.save_pdb(save_i)
                trj_gens[-1].save_pdb(save_n)
            elif not chirality_test_mode:
                for i in range(n_gens):
                    save_i = save_name.replace('.pdb', f'_{i+1}.pdb')
                    if save_chi:
                        chi_vec = get_all_chiralities_vec(trj_gens[i])
                        chi_file = save_name.replace('.pdb', f'_chi.npy')
                        failures = np.where(chi_vec == 1)[1]
                        residue_fail = [[r for r in trj_gens.top.residues][ff] for ff in failures]
                        print('resiue failures:', residue_fail)
                        np.save(chi_file, chi_vec)
                    if save_dcd:
                        trj_gens[i*n_frames].save_pdb(save_i)
                        trj_gens[i*n_frames].save_dcd(save_i.replace('.pdb', '.dcd'))
                    else:
                        trj_gens[i*n_frames:(i+1)*n_frames].save_pdb(save_i)
                        
            else:
                for i in range(n_gens):
                    chi_vec = get_all_chiralities_vec(trj_gens[i])
                    save_i = save_name.replace('.pdb', f'_{i+1}.pt')
                    # compute chirality for EVERY generated frame using your existing function
                
                    # tensors that are constant across frames for this traj
                    res_t  = torch.as_tensor(res_ohe, dtype=torch.int16).cpu()
                    atom_t = torch.as_tensor(atom_ohe, dtype=torch.int16).cpu()
                    mask_t = torch.as_tensor(mask, dtype=torch.bool).cpu()
                    a2cg_t = torch.as_tensor(aa_to_cg).cpu()
                
                    # prior is per-gen in your current code; map frame -> gen_idx
                
            
                    payload = {
                        # explicitly into EGNN / wrapper
                        "res_ohe": res_t,
                        "atom_ohe": atom_t,
                        "mask": mask_t,
                        "ca_pos": torch.as_tensor(ca_pos_test),
            
                        # what you asked to store
                        "aa_to_cg": a2cg_t,
                        "prior": torch.as_tensor(prior[i], dtype=torch.float32),
                        "decision_frame_1": torch.as_tensor(frames_10, dtype=torch.float32),
                        "decision_frame_2": torch.as_tensor(frames_20, dtype=torch.float32),
                        "decision_frame_3": torch.as_tensor(frames_30, dtype=torch.float32),
                        "decision_frame_4": torch.as_tensor(frames_40, dtype=torch.float32),
                        # label
                        "chi": torch.as_tensor(chi_vec, dtype=torch.float32),
                    }
            
                    torch.save(payload, save_i)
    
    # save all scores to same dir
    if not chirality_test_mode:
        if check_bonds:
            np.save(f'{save_prefix}/bf.npy', np.array(bf_list))
        if check_clash:
            np.save(f'{save_prefix}/cls.npy', np.array(clash_list))
        if check_div:
            np.save(f'{save_prefix}/div.npy', np.array(div_list))
                  
        # save ordered list of trajs
        with open(f'{save_prefix}/trj_list.pkl', "wb") as output_file:
            pkl.dump(trj_list, output_file)
             
        try:
            np.save(f'{save_prefix}/time_gen-{n_gens}.npy', np.array([res_list, time_list]))     
        except:
            pass
    
    print(f'\nSaved to:  {save_prefix}\n')
    
    #
