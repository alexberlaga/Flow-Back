import os
import matplotlib.pyplot as plt  # removing this thows scipy.optimize gcc error
import numpy as np
import torch
from collections import defaultdict, deque
from Bio.PDB import PDBParser, PDBIO, Select
import mdtraj as md
from tqdm import tqdm
import hashlib
from torch.utils.data import Dataset, DataLoader
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from itertools import islice, count
import math
import re
from openmm import *
from openmm.app import *
from openmm.unit import *
from scipy.spatial import cKDTree
from ..egnn_pytorch_se3.egnn_pytorch import EGNN_SE3, EGNN
import pandas as pd
from joblib import Parallel, delayed            # NEW!
from src.file_config import *
from src.chirality_predictor import ChiralityPredictor, PRIOR_NAME, ca_atomidx_to_residx
from .chi import *
se3_avail = True
    

RES_MAP = {
    "ALA": 1,  # Alanine
    "ARG": 2,  # Arginine
    "ASN": 3,  # Asparagine
    "ASP": 4,  # Aspartic acid
    "CYS": 5,  # Cysteine
    "GLU": 6,  # Glutamic acid
    "GLN": 7,  # Glutamine
    "GLY": 8,  # Glycine
    "HIS": 9,  # Histidine
    "ILE": 10, # Isoleucine
    "LEU": 11, # Leucine
    "LYS": 12, # Lysine
    "MET": 13, # Methionine
    "PHE": 14, # Phenylalanine
    "PRO": 15, # Proline
    "SER": 16, # Serine
    "THR": 17, # Threonine
    "TRP": 18, # Tryptophan
    "TYR": 19, # Tyrosine
    "VAL": 20  # Valine
}

ATOM_MAP = atom_types = {
        'C':1,
        'CA':2,
        'CB':3,
        'CD':4,
        'CD1':5,
        'CD2':6,
        'CE':7,
        'CE1':8,
        'CE2':9,
        'CE3':10,
        'CG':11,
        'CG1':12,
        'CG2':13,
        'CH2':14,
        'CZ':15,
        'CZ2':16,
        'CZ3':17,
        'N':18,
        'ND1':19,
        'ND2':20,
        'NE':21,
        'NE1':22,
        'NE2':23,
        'NH1':24,
        'NH2':25,
        'NZ':26,
        'O':27,
        'OD1':28,
        'OD2':29,
        'OE1':30,
        'OE2':31,
        'OG':32,
        'OG1':33,
        'OH':34,
        'SD':35,
        'SG':36,
}

def smoothstep(x, a, b):
    """Smoothly interpolates from 0 to 1 as x goes from a to b."""
    x = np.clip((x - a) / (b - a), 0, 1)
    return x * x * (3 - 2 * x)

def custom_function(t):
    low = 0.1      # value at t=0.5
    peak = 1.0     # maximum at t=0.6
    small = 0.02   # value after t=0.85
    if t < 0.6:
        return low + (peak - low) * smoothstep(t, 0.5, 0.6)
    elif t < 0.85:
        return peak + (small - peak) * smoothstep(t, 0.6, 0.85)
    else:
        return small

def exists(val):
    return val is not None

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context, mask = None):
        h = self.heads

        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask):
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b n -> b () () n')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class GlobalLinearAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        self.norm_seq = nn.LayerNorm(dim)
        self.norm_queries = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, heads, dim_head)
        self.attn2 = Attention(dim, heads, dim_head)

        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, queries, mask = None):
        res_x, res_queries = x, queries
        x, queries = self.norm_seq(x), self.norm_queries(queries)

        induced = self.attn1(queries, x, mask = mask)
        out     = self.attn2(x, induced)

        x =  out + res_x
        queries = induced + res_queries

        x = self.ff(x) + x
        return x, queries
    
def generate_cos_pos_encoding(n, dim, device, scale=10000.0):
    '''MJ -- replace pos_emb with sin/cos'''

    assert dim % 2 == 0, "dim must be even for alternating sin/cos encoding"
    
    pos_enc = torch.zeros(n, dim, device=device)
    position = torch.arange(0, n, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(scale) / dim))
    
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)

    return pos_enc

# adapted from https://github.com/lucidrains/egnn-pytorch with added time conditioning
class EGNN_Network_time(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        num_tokens = None,
        num_edge_tokens = None,
        num_positions = None,
        emb_cos_scale = None,
        edge_dim = 0,
        num_adj_degrees = None,
        adj_dim = 0,
        global_linear_attn_every = 0,
        global_linear_attn_heads = 8,
        global_linear_attn_dim_head = 64,
        num_global_tokens = 4,
        time_dim=0, 
        res_dim=20,  # change to 21
        atom_dim=3,  # change to 5
        esm_dim=1280,
        sym='e3',
        **kwargs    # MJ -- include seq_features, and seq_decay in kwargs
    ):
        super().__init__()
        assert not (exists(num_adj_degrees) and num_adj_degrees < 1), 'make sure adjacent degrees is greater than 1'
        self.num_positions = num_positions
        self.emb_cos_scale = emb_cos_scale
        
        self.res_emb = nn.Embedding(res_dim, dim)
        self.atom_emb = nn.Embedding(atom_dim, dim)
        # self.esm_emb = nn.Linear(esm_dim, dim)
        # self.time_emb = nn.Embedding(1, dim)

        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None
        self.pos_emb = nn.Embedding(num_positions, dim) if exists(num_positions) else None
        self.edge_emb = nn.Embedding(num_edge_tokens, edge_dim) if exists(num_edge_tokens) else None
        self.has_edges = edge_dim > 0

        self.num_adj_degrees = num_adj_degrees
        self.adj_emb = nn.Embedding(num_adj_degrees + 1, adj_dim) if exists(num_adj_degrees) and adj_dim > 0 else None

        edge_dim = edge_dim if self.has_edges else 0
        adj_dim = adj_dim if exists(num_adj_degrees) else 0

        has_global_attn = global_linear_attn_every > 0
        self.global_tokens = None
        if has_global_attn:
            self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, dim))

        self.layers = nn.ModuleList([])
        for ind in range(depth):
            is_global_layer = has_global_attn and (ind % global_linear_attn_every) == 0

            if sym=='e3':
                self.layers.append(nn.ModuleList([
                    GlobalLinearAttention(dim = dim, heads = global_linear_attn_heads, dim_head = global_linear_attn_dim_head) if is_global_layer else None,
                    EGNN(dim = dim, edge_dim = (edge_dim + adj_dim), norm_feats = True, **kwargs),
                ]))
                
            elif sym=='se3' and se3_avail:
                self.layers.append(nn.ModuleList([
                    GlobalLinearAttention(dim = dim, heads = global_linear_attn_heads, dim_head = global_linear_attn_dim_head) if is_global_layer else None,
                    EGNN_SE3(dim = dim, edge_dim = (edge_dim + adj_dim), norm_feats = True, **kwargs),
                ]))

        # MJ -- add an MLP to encode time
        self.time_dim = time_dim
        if self.time_dim > 0:
            self.time_net = torch.nn.Sequential(
                torch.nn.Linear(1, self.time_dim),
                torch.nn.SELU(),
                torch.nn.Linear(self.time_dim, self.time_dim),
                torch.nn.SELU(),
                torch.nn.Linear(self.time_dim, dim),
            )
    
        #self.generate_cos_pos_encoding = generate_cos_pos_encoding

    def forward(self, x):
        return self.net(x)

    def forward(
        self,
        feats,
        coors,
        ca_pos,
        time,
        atom_feats=None,
        esm_feats=None,
        adj_mat = None,
        edges = None,
        mask = None,
        return_coor_changes = False
    ):
        b, a, device = feats.shape[0], feats.shape[1], feats.device

        if exists(self.token_emb):
            feats = self.token_emb(feats)
        
        if atom_feats != None:
            feats += self.atom_emb(atom_feats)

        # if esm_feats != None:
        #     feats += self.esm_emb(esm_feats)
        if exists(self.pos_emb):
            n = feats.shape[1]
            assert n <= self.num_positions, f'given sequence length {n} must be less than the number of positions {self.num_positions} set at init'
            pos_emb = self.pos_emb(torch.arange(n, device = device))
            feats += rearrange(pos_emb, 'n d -> () n d')

        
        # don't use both the linear and cos embeddings
        elif exists(self.emb_cos_scale):
            n, dim = feats.shape[1], feats.shape[2]
            pos_emb_cos = generate_cos_pos_encoding(n, dim, device=device, scale=self.emb_cos_scale)
            feats += rearrange(pos_emb_cos, 'n d -> () n d')
            
        else:
            pass
          

        # if time passed as single float or dim 0 tensor
        if isinstance(time, float) or time.dim()==0:
            time = time*torch.ones((b, a, 1)).to(device)
            
        # if time is passed as a tensor of size b
        elif time.dim() == 1:
            time = time[:, None].repeat(1, a)[:, :, None]
            
        #else:
        #    feats += time[:, None, None]
        
        # use a time embedding MLP
        if self.time_dim > 0:
            time = self.time_net(time)

        feats += time

        if exists(edges) and exists(self.edge_emb):
            edges = self.edge_emb(edges)
        
        # create N-degrees adjacent matrix from 1st degree connections
        if exists(self.num_adj_degrees):
            assert exists(adj_mat), 'adjacency matrix must be passed in (keyword argument adj_mat)'

            if len(adj_mat.shape) == 2:
                adj_mat = repeat(adj_mat.clone(), 'i j -> b i j', b = b)

            adj_indices = adj_mat.clone().long()

            for ind in range(self.num_adj_degrees - 1):
                degree = ind + 2

                next_degree_adj_mat = (adj_mat.float() @ adj_mat.float()) > 0
                next_degree_mask = (next_degree_adj_mat.float() - adj_mat.float()).bool()
                adj_indices.masked_fill_(next_degree_mask, degree)
                adj_mat = next_degree_adj_mat.clone()

            if exists(self.adj_emb):
                adj_emb = self.adj_emb(adj_indices)
                edges = torch.cat((edges, adj_emb), dim = -1) if exists(edges) else adj_emb

        # setup global attention

        global_tokens = None
        if exists(self.global_tokens):
            global_tokens = repeat(self.global_tokens, 'n d -> b n d', b = b)
            

        coors_real = coors + ca_pos
        coor_changes = [coors_real]
        for global_attn, egnn in self.layers:
            if exists(global_attn):
                feats, global_tokens = global_attn(feats, global_tokens, mask = mask)
            feats, coors_real = egnn(feats, coors_real, adj_mat = adj_mat, edges = edges, mask = mask)
            # coors -= ca_pos
            coor_changes.append(coors_real)

        coors = coors_real - ca_pos
        if return_coor_changes:
            return feats, coors, coor_changes

        return feats, coors

# def LJ_velocities(feats, coors)
    
def get_adj_mat(top):

    num_atoms = top.n_atoms
    
    bonded_pairs = []
    for bond in top.bonds:
        bonded_pairs.append((bond[0].index, bond[1].index))

    adj_mat = np.zeros((num_atoms, num_atoms), dtype=int)

    # Fill the adjacency matrix based on bonded pairs
    for pair in bonded_pairs:
        adj_mat[pair[0], pair[1]] = 1
        adj_mat[pair[1], pair[0]] = 1

    adj_mat = torch.tensor(adj_mat, dtype=torch.bool)
    return adj_mat

def get_adj_CG(xyz, mask_idxs, cut=1.0):
    '''Directly connect all CG atoms only'''

    num_atoms = xyz.shape[1]
    adj_mat = np.zeros((num_atoms, num_atoms), dtype=int)
    
    for i in range(num_atoms):
        if i in mask_idxs:
            dists = np.sqrt(((xyz[i] - xyz[mask_idxs])**2).sum(axis=-1))
            include_idxs = np.where(dists < cut)[0]
            adj_mat[i, mask_idxs[include_idxs]] = 1
        
    adj_mat = torch.tensor(adj_mat, dtype=torch.bool)
    return adj_mat

def sig(t, dt, cg_noise):
    return cg_noise * torch.sqrt((2 * (1 - t + dt)) / (t + dt))
    
def get_aa_to_cg(top, msk):
    '''Mapping between AA and CG
       Assign to Ca positions for now with mask, but will need to generalize this'''
    
    aa_to_cg = []
    for atom_idx, atom in enumerate(top.atoms):
        res_idx = atom.residue.index
        aa_to_cg.append(msk[res_idx])
        
    return np.array(aa_to_cg)

def get_ca_pos(xyz, aa_to_cg):
    '''Ca positions of every atom'''
    xyz_ca = []
    for xyz_ref, map_ref in zip(xyz, aa_to_cg):
        xyz_ca_i = xyz_ref[map_ref]
        xyz_ca.append(xyz_ca_i)
    
    return xyz_ca
    
def get_prior(xyz, aa_to_cg, mask_idxs=None, scale=1.0, frames=None):
    '''Normally distribute around respective Ca center of mass'''
    
    # set center of distribution to each CA and use uniform scale
    xyz_ca = xyz[:, aa_to_cg]
    scale = scale * np.ones(xyz_ca.shape)
    xyz_prior = np.random.normal(loc=xyz_ca, scale=scale, size=xyz.shape)
    
    # don't add noise to masked values
    if mask_idxs is not None:
        xyz_prior[:, mask_idxs] = xyz[:, mask_idxs]
    
    return xyz_priorx


def get_prior_mix(xyz, aa_to_cg, scale=1.0):
    '''Normally distribute around respective Ca center of mass'''
    
    # set center of distribution to each CA and use uniform scale
    xyz_prior = []
    
    for xyz_ref, map_ref in zip(xyz, aa_to_cg):
    
        xyz_ca = xyz_ref[map_ref]
        xyz_prior.append(np.random.normal(loc=xyz_ca, scale=scale * np.ones(xyz_ca.shape), size=xyz_ca.shape))
    
    return xyz_prior #np.array(xyz_prior, dtype=object)  # fix ragged nest warning

def get_prior_mask(xyz, aa_to_cg, masks=None, scale=1.0):
    '''Normally distribute around respective masked coordinates
       Optionally mask out CG values so they are identical in CG and AA traces'''
    
    # set center of distribution to each CA and use uniform scale
    xyz_prior = []
    for i, (xyz_ref, map_ref) in enumerate(zip(xyz, aa_to_cg)):
    
        xyz_ca = xyz_ref[map_ref]
        xyz_ca = np.random.normal(loc=xyz_ca, scale=scale * np.ones(xyz_ca.shape), size=xyz_ca.shape)

        # ensure masked values are not noised at all
        if masks is not None:
            mask = ~masks[i].astype(bool)
            xyz_ca[mask] = xyz_ref[mask]
        xyz_prior.append(xyz_ca)
    
    return xyz_prior



    
def str_to_ohe(string_list):
    unique_strings = list(set(string_list))
    string_to_index = {string: index for index, string in enumerate(unique_strings)}
    indices = [string_to_index[string] for string in string_list]
    return np.array(indices)

def get_node_ohes(top):
    '''get one-hot encodings of residue and atom element identities'''
    
    res_list, atom_list = [], []

    for a in top.atoms:
        if a.element.name != 'hydrogen': 
            res_list.append(a.residue.name)
            atom_list.append(a.element.name)

    res_ohe = str_to_ohe(res_list)
    atom_ohe = str_to_ohe(atom_list)
    
    return res_ohe, atom_ohe
    
# load data set
class CustomDataset(Dataset):
    def __init__(self, data_list1, data_list2):
        self.data_list1 = data_list1
        self.data_list2 = data_list2

    def __len__(self):
        return len(self.data_list1)  # Assuming both lists have the same length

    def __getitem__(self, index):
        sample1 = torch.tensor(self.data_list1[index], dtype=torch.float32)
        sample2 = torch.tensor(self.data_list2[index], dtype=torch.float32)

        return sample1, sample2


# load data set
class StructureDataset(Dataset):
    def __init__(self, xyz_diff, xyz_prior, res_feats, atom_feats, ca_pos, mask):
        self.xyz_diff = xyz_diff
        self.xyz_prior = xyz_prior
        self.res_feats = res_feats
        self.atom_feats = atom_feats
        self.ca_pos = ca_pos
        self.mask = mask

    def __len__(self):
        return len(self.xyz_diff)  # Assuming both lists have the same length

    def __getitem__(self, index):
        sample1 = torch.tensor(self.xyz_diff[index], dtype=torch.float32)
        sample2 = torch.tensor(self.xyz_prior[index], dtype=torch.float32)
        sample3 = torch.tensor(self.res_feats[index], dtype=torch.int)
        sample4 = torch.tensor(self.atom_feats[index], dtype=torch.int)
        sample5 = torch.tensor(self.ca_pos[index], dtype=torch.float32)
        sample6 = torch.tensor(self.mask[index], dtype=torch.bool)
        
        return sample1, sample2, sample3, sample4, sample5, sample6

class PostTrainDataset(Dataset):
    def __init__(self, res_feats, atom_feats, ca_pos, mask, topologies):
        self.res_feats = res_feats
        self.atom_feats = atom_feats
        self.ca_pos = ca_pos
        self.mask = mask
        self.topologies = topologies

    def __len__(self):
        return len(self.res_feats)  # Assuming both lists have the same length

    def __getitem__(self, index):
        sample1 = torch.tensor(self.res_feats[index], dtype=torch.int)
        sample2 = torch.tensor(self.atom_feats[index], dtype=torch.int)
        sample3 = torch.tensor(self.ca_pos[index], dtype=torch.float32)
        sample4 = torch.tensor(self.mask[index], dtype=torch.bool)
        sample5 = self.topologies[index]

        return sample1, sample2, sample3, sample4, sample5

    
def pro_res_to_ohe(string_list):
    
    amino_acids = RES_MAP

    indices = [amino_acids[string] for string in string_list]
    return np.array(indices)

def pro_atom_to_ohe(string_list):
    
    atom_types = {
    "carbon": 1,
    "oxygen": 2, 
    "nitrogen": 3, 
    "sulfur": 4,
    }

    indices = [atom_types[string] for string in string_list]
    return np.array(indices)

def pro_allatom_to_ohe(string_list):    
    atom_types = ATOM_MAP
  
    indices = [atom_types[string] for string in string_list]
    return np.array(indices)

def pro_ohe_to_allatom(ohe_list):
    inv_atom_map = {v: k for k, v in ATOM_MAP.items()}
    strings = [inv_atom_map[key] for key in ohe_list]
    return np.array(strings)

def pro_ohe_to_res(ohe_list):
    inv_atom_map = {v: k for k, v in RES_MAP.items()}
    strings = [inv_atom_map[key] for key in ohe_list]
    return np.array(strings)
    
def get_pro_ohes(top):
    '''get one-hot encodings of residue and atom element identities'''
    
    res_list, atom_list, allatom_list = [], [], []

    for a in top.atoms:
        if a.element.name != 'hydrogen': 
            res_list.append(a.residue.name) 
            atom_list.append(a.element.name) 
            allatom_list.append(a.name)       
    
    res_ohe = pro_res_to_ohe(res_list)
    atom_ohe = pro_atom_to_ohe(atom_list)
    allatom_ohe = pro_allatom_to_ohe(allatom_list)
    
    return res_ohe, atom_ohe, allatom_ohe

# set up alagous protein parse function
def parse_pro_CA(pro_trj):
    '''Extract GNN parameters compatible with 3sn2 CG representation of DNA
       Ensure that pro_trj only includes dna residues'''
    
    res_ohe, atom_ohe, all_atom_ohe = get_pro_ohes(pro_trj.top)
    mask_idxs = pro_trj.top.select('name CA')
    aa_to_cg = get_aa_to_cg(pro_trj.top, mask_idxs)
    xyz = pro_trj.xyz
    
    return xyz, mask_idxs, aa_to_cg, res_ohe, all_atom_ohe
    
class ModelWrapper(EGNN_Network_time):
    def __init__(
        self,
        model,
        feats,
        mask,
        ca_pos,
        atom_feats=None,
        esm_feats=None,
        adj_mat=None
    ):
        super(EGNN_Network_time, self).__init__()  # Call the nn.Module constructor
        self.model = model
        self.feats = feats
        self.mask = mask
        self.atom_feats = atom_feats
        self.esm_feats = esm_feats
        self.ca_pos = ca_pos
        
    def forward(self, t, x, y=None, *args, **kwargs):
        
        feats_out, coors_out = self.model(self.feats, x, mask=self.mask, time=t,
                                    atom_feats=self.atom_feats, esm_feats=self.esm_feats, ca_pos=self.ca_pos)
        return coors_out - x


# ------------------------------------------------------------
# Helper: build atom mask from a per-residue D-mask
# ------------------------------------------------------------
def residue_mask_to_atom_mask(d_res_mask: torch.Tensor,
                              idx_n, idx_ca, idx_c, idx_o, idx_cb, idx_side,
                              n_atoms: int,
                              device=None) -> torch.Tensor:
    """
    d_res_mask: (R,) bool, True for residues predicted D-chiral
    Returns: (n_atoms,) bool mask for atoms belonging to D residues
    """
    if device is None:
        device = d_res_mask.device

    d_atoms = torch.zeros(n_atoms, dtype=torch.bool, device=device)

    # idx_* are assumed per-residue atom indices (shape (R,))
    # idx_side is assumed a list/array per residue of side atom indices,
    # OR a tensor (R, K) with -1 padding, OR a flat list you can adapt below.
    def mark(indices):
        if indices is None:
            return
        ii = torch.as_tensor(indices, device=device)
        # filter invalid if present
        ii = ii[d_res_mask]
        ii = ii[ii >= 0]
        d_atoms[ii] = True

    mark(idx_n)
    mark(idx_ca)
    mark(idx_c)
    mark(idx_o)
    mark(idx_cb)

    # Handle idx_side in a flexible way
    if idx_side is not None:
        if isinstance(idx_side, (list, tuple)):
            # common pattern: idx_side is list length R, each entry is list of atom indices
            for r, is_d in enumerate(d_res_mask.tolist()):
                if not is_d:
                    continue
                side_r = idx_side[r]
                if side_r is None:
                    continue
                side_r = torch.as_tensor(side_r, device=device)
                side_r = side_r[side_r >= 0]
                if torch.any(side_r):
                    d_atoms[side_r] = True
        else:
            # tensor/ndarray case: (R, K) with -1 padding
            side = torch.as_tensor(idx_side, device=device)  # (R, K)
            side = side[d_res_mask]                          # (Rd, K)
            side = side.reshape(-1)
            side = side[side >= 0]
            d_atoms[side] = True

    return d_atoms


# ------------------------------------------------------------
# Helper: run chirality predictor at t=0.3
# ------------------------------------------------------------
def load_chirality_predictor(ckpt_path: str, base_egnn: nn.Module, device: torch.device):
    """
    Construct + load your ChiralityPredictor.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # print(ckpt, ckpt["model_state_dict"])
    # ---- You MUST adapt these two lines to your project ----
    # Example patterns I’ve seen in your repo before:
    dim = base_egnn.res_emb.embedding_dim
    res_vocab = base_egnn.res_emb.num_embeddings
    atom_vocab = base_egnn.atom_emb.num_embeddings
    model = ChiralityPredictor(dim, res_vocab, atom_vocab)
    sd = ckpt["model_state_dict"]

    # remove all EGNN weights
    sd_no_egnn = {k: v for k, v in sd.items() if "egnn" not in k}
    model.load_state_dict(sd_no_egnn)
    model.to(device).eval()
    return model



def make_chirality_batch(
    *,
    feats, atom_feats, mask, ca_pos, x, aa_to_cg, n_residues,
    device=None,
):
    """
    Build a (B=1)-batch dict compatible with what collate_chi() returns.
    Assumes inputs are already for ONE structure (no batching).
    """

    if device is None:
        device = prior.device

    # --- ensure dtypes ---
    feats      = feats.to(device=device, dtype=torch.long)          # (A,)
    atom_feats = atom_feats.to(device=device, dtype=torch.long)     # (A,)
    mask       = mask.to(device=device, dtype=torch.bool)           # (A,)

    # res_mask in your dataset is just all-ones (same length as mask)
    res_mask = torch.ones_like(mask, dtype=torch.bool, device=device)  # (A,)

    ca_pos = ca_pos.to(device=device, dtype=torch.float32)          # typically (A,3)
    x  = x.to(device=device, dtype=torch.float32)           # (A,3)

    aa_to_cg = aa_to_cg.to(device=device, dtype=torch.long)         # (R,) or (A,) depending on your mapping

    # --- add batch dimension + match collate shapes ---
    # collate_chi makes ca_pos shape (B,1,Amax,3)
    ca_pos_batched = ca_pos.expand(n_residues, -1, -1)

    # collate_chi uses prior shape (B,Amax,3)
    x_batched = x.expand(n_residues, -1, -1)

    feats_batched      = feats.unsqueeze(0).expand(n_residues, -1)
    atom_feats_batched = atom_feats.unsqueeze(0).expand(n_residues, -1)                    # (1,A)
    mask_batched       = mask.unsqueeze(0).expand(n_residues, -1)                          # (1,A)
    res_mask_batched   = res_mask.unsqueeze(0).expand(n_residues, -1)                      # (1,A)
    aa_to_cg_batched = ca_atomidx_to_residx(aa_to_cg).unsqueeze(0).expand(n_residues, -1)  # (1,A) assumed

    return {
        "feats": feats_batched,
        "atom_feats": atom_feats_batched,
        "mask": mask_batched,
        "res_mask": res_mask_batched,
        "ca_pos": ca_pos_batched,
        PRIOR_NAME: x_batched,
        "aa_to_cg": aa_to_cg_batched,
    }

@torch.inference_mode()
def predict_d_mask_per_residue(chiral_pred_model, atom_h, res_feats, atom_feats, cg_mask, ca_pos, x_t, aa_to_cg, device, n_residues,
                               threshold: float = 0.16) -> torch.Tensor:
    """
    Returns d_res_mask: (R,) bool where True means predicted D (label 1).

    The only requirement for the rest of the integrator is you return a (R,) bool mask.
    """
    # Put coords in absolute frame if your predictor expects that:
 
    # Option A: model returns per-residue probabilities/logits for class "D" (1)
    #   p_d: (B, R) or (R,)
    chi_batch = make_chirality_batch(
        feats=res_feats,
        atom_feats=atom_feats,
        mask=cg_mask,
        ca_pos=ca_pos,          # (A,3)
        x=x_t,            # (A,3) or x_t depending on your design
        aa_to_cg=aa_to_cg,      # should be (A,) with -1 padding already
        n_residues=n_residues,
        device=device,
    )

    with torch.no_grad():
        # match your model forward signature; example:
        out = chiral_pred_model(
            atom_h=atom_h.expand(n_residues, -1, -1),
            feats=chi_batch["feats"],
            atom_feats=chi_batch["atom_feats"],
            mask=chi_batch["mask"],
            res_mask=chi_batch["res_mask"],
            ca_pos=chi_batch["ca_pos"],
            prior=chi_batch[PRIOR_NAME],
            aa_to_cg=chi_batch["aa_to_cg"],
            res_idx=torch.arange(n_residues).to(device),
        )

    
    cutoff = math.log(threshold / (1.0 - threshold))
    d_res_mask = (out >= cutoff)
    # print(p_d, threshold, d_res_mask)
    return d_res_mask

# add custom integrators
def euler_integrator(model, x, nsteps=100, mask=None, noise=False):
    
    # try adding small amounts of noise during integration
    
    ode_list = []
    dt = 1./(nsteps-1)
    
    for t in np.linspace(0, 1, nsteps):
        
        # Evaluate dx/dt using the model for the current state and time
        with torch.no_grad():
            dx_dt = model(t, x.detach())

        x = (x + dx_dt * dt).detach()
        # track each update to show diffusion path
        ode_list.append(x.cpu().numpy())

    return np.array(ode_list)


def euler_ff_integrator(model, x, ff_params):

    # ---- required extracts ----
    ca_pos      = ff_params["ca_pos"]
    lj_cache    = ff_params["lj_cache"]
    bond_cache  = ff_params["bond_cache"]
    angle_cache = ff_params["angle_cache"]
    hbond_cache = ff_params["hbond_cache"]
    top         = ff_params["top"]
    ff          = ff_params["ff"]
    device      = ff_params["device"]

    idx_n       = ff_params["idx_n"]
    idx_ca      = ff_params["idx_ca"]
    idx_c       = ff_params["idx_c"]
    idx_o       = ff_params["idx_o"]
    idx_cb      = ff_params["idx_cb"]
    idx_side    = ff_params["idx_side"]

    reflect_chi = ff_params["reflect_chi"]
    # ---- optional extracts (with defaults) ----
    nsteps              = ff_params.get("nsteps", 100)
    cg_noise            = ff_params.get("cg_noise", 0.003)
    alpha               = ff_params.get("alpha", 12)
    no_angle            = ff_params.get("no_angle", False)
    hbond               = ff_params.get("hbond", False)
    chirality_test_mode = ff_params.get("chirality_test_mode", False)
    chiral_pred_model   = ff_params.get("chiral_pred", None)
    chiral_threshold    = ff_params.get("chiral_threshold", 0.16)
    
    MAX_LJ     = ff_params.get("MAX_LJ",     0.5)
    MAX_BOND   = ff_params.get("MAX_BOND",   2.0)
    MAX_ANGLE  = ff_params.get("MAX_ANGLE",  0.5)
    MAX_HBOND  = ff_params.get("MAX_HBOND",  0.5)
    MAX_CHIRAL = ff_params.get("MAX_CHIRAL", 2.0)
    
    # -------------------------
    # Force weights (defaults preserved)
    # -------------------------
    W_LJ     = ff_params.get("W_LJ",     1.5)
    W_BOND   = ff_params.get("W_BOND",   2.0)
    W_ANGLE  = ff_params.get("W_ANGLE",  0.5)
    W_HBOND  = ff_params.get("W_HBOND",  1.0)
    W_CHIRAL = ff_params.get("W_CHIRAL", 10.0)

    T_LJ = ff_params.get("T_LJ", 0.5)
    T_BOND = ff_params.get("T_BOND", 0.75)
    print('t_lj', T_LJ, 't_bond', T_BOND)

    res_feats   = ff_params["res_ohe"].to(device)
    atom_feats  = ff_params["atom_ohe"].to(device)
    cg_mask     = ff_params["cg_mask"].to(device)
    res_mask    = torch.ones_like(cg_mask).to(device)
    aa_to_cg    = ff_params["aa_to_cg"].to(device)
    ode_list = []
    dt = 1.0 / (nsteps - 1)
    device = x.device
    sig = topology_signature_by_sequence(top)
    x_t = x

    chirality_atom_mask = None     # (A,) bool, computed once at t≈0.3
    chirality_pred_done = False
    for t in range(nsteps):
        if device.type == 'cpu':
            print(t)
        t_val = t / nsteps
        alpha_t = (t + 1) / nsteps
        is_last = (t == nsteps - 1)
        
    
        with torch.no_grad():
            ff_velocity = torch.zeros_like(x_t)
            
            xt_cap = x_t + ca_pos
            clamp_lj   = lambda z: torch.clamp(W_LJ * lj_velocity_fn(z, t_val, dt, lj_cache),  -MAX_LJ,  MAX_LJ)
            clamp_bond = lambda z: torch.clamp(W_BOND * bond_velocity_fn(z, t_val, dt, bond_cache), -MAX_BOND,  MAX_BOND)
            clamp_angle = lambda z: torch.clamp(W_ANGLE * angle_velocity_fn(z, dt, angle_cache), -MAX_ANGLE, MAX_ANGLE)
            clamp_hbond = lambda z: torch.clamp(W_HBOND * hbond_velocity_fn(z, dt, hbond_cache), -MAX_HBOND, MAX_HBOND)
            stack = lambda f: torch.stack([f(xt_cap[i:i+1]) for i in range(x_t.shape[0])])

            if t_val > T_LJ:
                ff_velocity = t_val ** alpha * stack(clamp_lj)
                if hbond:
                    ff_velocity += custom_function(t_val) * stack(clamp_hbond)
                if not no_angle and t_val > T_BOND:
                    ff_velocity += (t_val**alpha) * stack(clamp_angle)
                if t_val > T_BOND:
                    ff_velocity += (t_val**alpha) * stack(clamp_bond)
            

            if (not chirality_test_mode) and (not chirality_pred_done):
                # “when t = 0.3” in discrete steps: trigger on the closest step to 0.3
                # This fires once when t_val crosses 0.3 within half a step.
                if abs(t_val - 0.3) <= (0.5 / nsteps):
                    chi_mask = []
                    for j in range(x_t.shape[0]):
                        atom_h, _ = model.model(res_feats.unsqueeze(0).long(), x_t[j], ca_pos[j:j+1], t_val, atom_feats=atom_feats.unsqueeze(0).long(), mask=cg_mask.unsqueeze(0))
                        d_res_mask = predict_d_mask_per_residue(
                            chiral_pred_model, atom_h, res_feats, atom_feats, cg_mask, ca_pos[j:j+1], x_t[j:j+1], aa_to_cg, device, top.n_residues, threshold=chiral_threshold
                        )
                        print(torch.where(d_res_mask))
                        # print(torch.sum(d_res_mask) / len(d_res_mask))
                        # Convert residue mask -> atom mask (A,) bool
                        n_atoms = x_t.shape[-2]  # x_t: (B, A, 3)
                        chi_mask.append(residue_mask_to_atom_mask(
                            d_res_mask,
                            idx_n, idx_ca, idx_c, idx_o, idx_cb, idx_side,
                            n_atoms=n_atoms,
                            device=x_t.device
                        ))  # (A,) bool
                        # chirality_atom_mask = torch.ones_like(atom_feats)
                        chirality_pred_done = True
                    chirality_atom_mask = torch.stack(chi_mask)
            
            if not chirality_test_mode:
                # ff_velocity += torch.clamp(W_CHIRAL * chirality_velocity(x_t, t_val, idx_n, idx_ca, idx_c, idx_cb, idx_o,  idx_side), -MAX_CHIRAL, MAX_CHIRAL)
                if chirality_atom_mask is not None and chirality_atom_mask.any():
                    if reflect_chi:
                        v_chi = chirality_reflection_velocity(x_t, t_val, idx_n, idx_ca, idx_c, idx_cb, idx_o, idx_side, noise=cg_noise)
                        # print(torch.max(v_chi).item())
                    else:
                        v_chi = chirality_velocity(x_t, t_val, idx_n, idx_ca, idx_c, idx_cb, idx_o, idx_side, noise=cg_noise)
                        # print(torch.max(v_chi[:, -10:, :]).item())
                    v_chi = torch.clamp(W_CHIRAL * v_chi, -MAX_CHIRAL, MAX_CHIRAL)
                    
                    # Apply mask: (A,) -> (1, A, 1) to broadcast over batch + xyz
                    atom_mask = chirality_atom_mask.view(x_t.shape[0], -1, 1)
                    v_chi = v_chi * atom_mask
                    ff_velocity += v_chi
                else:
                    # No predicted D residues (or we haven’t hit t≈0.3 yet): no chirality velocity
                    pass
    
            base = model(t_val, x_t)
            drift = base + ff_velocity 
    
            x_t = x_t + dt * drift
        
        # Track each update to show diffusion path
        ode_list.append(x_t.cpu().numpy())
    
    if not chirality_test_mode:
        trj = md.Trajectory(x_t.cpu().numpy(), top)
        chiralities = get_all_chiralities_vec(trj)
        if np.any(chiralities == 1):
            print('bad chirality!', np.where(chiralities == 1))
        trj = invert_chirality_reflection_ter(trj, chiralities)
        ode_list[-1] = trj.xyz
    
    return np.array(ode_list)

def euler_integrator_with_chirality(
    model,
    x0,
    idx_n, idx_ca, idx_c, idx_cb, idx_side,
    nsteps=100,
    w_chiral=10.0,
    t0=0.0, t1=1.0,
):
    """
    dx/dt = v_theta(t, x) + w_chiral * v_chiral(x)
    - v_chiral(x) = ∂/∂x sum(sigmoid(beta * volume_per_residue(x)))
    - All ops GPU-safe. No randomness. No gradients are baked through time.

    model(t, x): returns dx_dt (same shape as x); called under no_grad() here to
                 match your memoryless adjoint style. If you need to train, remove no_grad().
    """
    x = x0
    device = x.device
    ts = torch.linspace(t0, t1, nsteps, device=device, dtype=x.dtype)
    dt = (t1 - t0) / (nsteps - 1)
    path = []

    for t in ts:
        with torch.no_grad():
            dx_model = model(t, x.detach())

        
        dv_chiral = chirality_velocity(x, t, idx_n, idx_ca, idx_c, idx_o, idx_cb, idx_side)
      
        
        dx_dt = dx_model + w_chiral * dv_chiral

        # Euler step and detach to keep the trajectory memoryless for adjoint matching
        x = (x + dt * dx_dt).detach()
        
        path.append(x.cpu().detach().numpy())

    return np.array(path)



def bond_velocity_fn(x, t, dt, cache):
    """
    x: (N,3) or (1,N,3) tensor on the same device as cache tensors
    t: float in [0,1] (or tensor scalar)
    cache: dict from build_bond_cache
    dt: float or tensor scalar
    """
    # Accept both (N,3) and (1,N,3)
    if x.dim() == 3:
        x = x[0]  # assume batch size 1 for parity with your original
    with torch.enable_grad():
        i = cache["i"]; j = cache["j"]
        b0 = cache["b0"]; kb = cache["kb"]
        mi = cache["mi"]; mj = cache["mj"]
        # mask = cache["mask"]
    
        # Bond vectors for all bonds at once
        rij = x[i] - x[j]                         # (M,3)
        dij = torch.linalg.norm(rij, dim=1)       # (M,)
    
        # Your original formula (kept as-is): F = kb * (|r|-b0*t) * r
        # (Note: this is not normalized by |r|; preserving behavior intentionally.)
        Fmag = kb * (dij - (b0 * t))              # (M,)
        Fvec = (Fmag[:, None] * rij)              # (M,3)
    
        # Zero-out S-S bonds
        # if mask is not None:
        #     Fvec = torch.where(mask[:, None], Fvec, torch.zeros_like(Fvec))
    
        # Allocate velocities and scatter-accumulate contributions
        v = x.new_zeros(cache["n_atoms"], 3)      # (N,3)
        # i gets -F / m_i * dt, j gets +F / m_j * dt
        v.scatter_add_(0, i[:, None].expand(-1, 3), -(Fvec / mi[:, None]) * dt)
        v.scatter_add_(0, j[:, None].expand(-1, 3),  (Fvec / mj[:, None]) * dt)
    return v


def angle_velocity_fn(x, dt, cache, eps=1e-12):
    """
    Harmonic angle velocities via a single autograd pass.
    Potential: U = 0.5 * K * (theta - theta0)^2 over all angles (i-j-k).
    Returns velocities with v = -(1/m) * dU/dx * dt.
    """
    if x.dim() == 3:
        x = x[0]  # assume batch size 1 to match your other fns

    N = cache["N"]
    if cache["i"].numel() == 0:
        return x.new_zeros(N, 3)

    i = cache["i"]; j = cache["j"]; k = cache["k"]
    theta0 = cache["theta0"]; K = cache["K"]
    mass = cache["mass"]

    # Enable autograd on a detached copy to avoid polluting upstream graphs
    x_req = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        ri = x_req[i]      # (M,3)
        rj = x_req[j]      # (M,3)
        rk = x_req[k]      # (M,3)
    
        u = ri - rj        # (M,3)
        v = rk - rj        # (M,3)
        nu = torch.linalg.norm(u, dim=1).clamp_min(eps)  # (M,)
        nv = torch.linalg.norm(v, dim=1).clamp_min(eps)  # (M,)
    
        cos_th = (u * v).sum(dim=1) / (nu * nv)
        cos_th = cos_th.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.arccos(cos_th)                     # (M,)
    
        # Harmonic energy over all angles
        dtheta = theta - theta0
        E = 0.5 * K * (dtheta ** 2)
        E_sum = E.sum()
    
        # Forces = -grad_x E
        grad_x, = torch.autograd.grad(E_sum, x_req, create_graph=False, retain_graph=False)
        F = -grad_x  # (N,3)
    
        # v = F / m * dt
        v = F / mass[:, None] * dt
    return v
    


def hbond_velocity_fn(x, dt, cache, eps=1e-8):
    """
    Distance-only HB capture on side-chain N···O pairs, converted to a torque
    about χ1 (CA–CB) and applied as a rigid rotation to each residue's side chain.

    Inputs:
      x: (N,3) or (1,N,3)
      dt: scalar
      cache: from build_hb_sidechain_torque_cache
    Returns:
      velocities: (N,3)
    """
    if x.dim() == 3:
        x = x[0]
    device = x.device
    N = cache["N"]
    if cache["pivot_idx"].numel() == 0:
        return x.new_zeros(N,3)

    iu, ju = cache["iu"], cache["ju"]
    is_side = ~cache["is_bb"]
    rc, r0= cache["rc"], cache["r0"], 
    # cache["k_pair"]
    is_acceptor_sc = cache["is_acceptor_sc"]; is_donor_sc = cache["is_donor_sc"]
    k_omega = cache["k_omega"]
    is_bb   = cache["is_bb"];
    k_rep = cache["k_rep"]; r_rep = cache["r_rep"]
    alpha = cache["alpha"]     # 1/nm
    De    = cache["De"]        # kJ/mol

    # -------- neighbor list (upper-tri inside rc) --------
    rij = x[iu] - x[ju]                    # (P,3), i->j
    r   = rij.norm(dim=1)                  # (P,)
    mask_cut = (r < rc)

        # unit vectors and selected pairs (upper-tri within rc)
    mask_rc = mask_cut
    u = rij / r.clamp_min(1e-8)[:, None]
    
    # ---- N–O (and O–N) attraction ----
    DA_mask = mask_cut & (
        (is_donor_sc[iu] & is_acceptor_sc[ju]) |
        (is_acceptor_sc[iu] & is_donor_sc[ju])
    )
    sel_DA = DA_mask.nonzero(as_tuple=False).squeeze(1)
    
    Fi_list = []; Fj_list = []; i_list = []; j_list = []
    
    if sel_DA.numel() > 0:
        r_DA  = r[sel_DA].clamp_min(1e-8)
        u_DA  = u[sel_DA]
        z = torch.exp(-alpha * (r_DA - r0))
        dUdr_DA = 2.0 * De * alpha * z * (1.0 - z) 
        Fi_DA = -dUdr_DA[:,None] * u_DA             # force on i
        Fj_DA =  dUdr_DA[:,None] * u_DA              # force on j
        Fi_list.append(Fi_DA); Fj_list.append(Fj_DA)
        i_list.append(iu[sel_DA]); j_list.append(ju[sel_DA])
    
    # ---- O–O repulsion (capped quadratic) ----
    OO_mask = mask_rc & (
        (is_acceptor_sc[iu] & is_acceptor_sc[ju]) 
    )
    sel_OO = OO_mask.nonzero(as_tuple=False).squeeze(1)
    
    if sel_OO.numel() > 0:
        r_OO  = r[sel_OO].clamp_min(1e-8)
        u_OO  = u[sel_OO]
        # only active for r < r_rep
        active = (r_OO < r_rep)
        if active.any():
            act_idx = active.nonzero(as_tuple=False).squeeze(1)
            r_act   = r_OO[act_idx]
            u_act   = u_OO[act_idx]
            # F = k_rep * (r_rep - r) * u  (repulsive)
            Fmag = k_rep / (r_rep - r_act) ** 0.5
            Fi_OO =  Fmag[:,None] * u_act                # on i
            Fj_OO = -Fmag[:,None] * u_act                # on j
            Fi_list.append(Fi_OO); Fj_list.append(Fj_OO)
            i_list.append(iu[sel_OO][act_idx]); j_list.append(ju[sel_OO][act_idx])
    
    # --- if no pairs, return zeros
    if not Fi_list:
        return x.new_zeros(N,3)
    
    # Concatenate all pairwise forces to feed the torque accumulator
    Fi = torch.cat(Fi_list, dim=0)
    Fj = torch.cat(Fj_list, dim=0)
    i_sel = torch.cat(i_list, dim=0)
    j_sel = torch.cat(j_list, dim=0)

    # -------- convert to torques about χ1 axis per residue --------
    # map atoms to a local residue index that has χ1 (=-1 if none)
    atom2res = cache["atom2res"]          # (N,), indices into right_off / pivot_idx / axis_idx
    res_i = atom2res[i_sel]               # (P,)
    res_j = atom2res[j_sel]               # (P,)

    # Only atoms that belong to a χ1-controlled right-group contribute torque
    valid_i = res_i >= 0
    valid_j = res_j >= 0

    pivot = cache["pivot_idx"]            # (R,)
    axis  = cache["axis_idx"]             # (R,)

    # axis unit vectors per residue, and pivot positions
    r_pivot = x[pivot]                    # (R,3)
    u_axis  = (x[axis] - x[pivot])
    u_axis  = u_axis / (u_axis.norm(dim=1, keepdim=True).clamp_min(eps))  # (R,3)

    R = pivot.numel()
    tau_res = x.new_zeros(R)              # scalar torque about axis per residue

    # torque contributions from i atoms
    if valid_i.any():
        ri = x[i_sel[valid_i]]                            # positions
        Fi_sel = Fi[valid_i]
        ridx = res_i[valid_i]
        r_rel = ri - r_pivot[ridx]                        # (Pi,3)
        tau_i = torch.cross(r_rel, Fi_sel, dim=1)         # vector torque
        tau_i_ax = (tau_i * u_axis[ridx]).sum(dim=1)      # scalar along axis
        tau_scatter = x.new_zeros(R)
        tau_scatter.scatter_add_(0, ridx, tau_i_ax)
        tau_res += tau_scatter

    # torque contributions from j atoms
    if valid_j.any():
        rj = x[j_sel[valid_j]]
        Fj_sel = Fj[valid_j]
        ridx = res_j[valid_j]
        r_rel = rj - r_pivot[ridx]
        tau_j = torch.cross(r_rel, Fj_sel, dim=1)
        tau_j_ax = (tau_j * u_axis[ridx]).sum(dim=1)
        tau_scatter = x.new_zeros(R)
        tau_scatter.scatter_add_(0, ridx, tau_j_ax)
        tau_res += tau_scatter

    # -------- per-residue moment of inertia about axis (right-group only) --------
    right_idx = cache["right_idx"]          # packed right-group atoms
    right_off = cache["right_off"]          # offsets per residue (len=R+1)
    mass      = cache["mass"]

    # Build I_res by iterating residues in vectorized chunks
    I_res = x.new_zeros(R)
    # vectorized: compute r_perp norms for all right-group atoms, then scatter by residue
    if right_idx.numel() > 0:
        # map each right-group atom to its residue index
        # We can reconstruct by expanding offsets:
        # create an index array res_of_right matching right_idx length
        counts = (right_off[1:] - right_off[:-1]).to(torch.long)
        res_ids = torch.arange(R, device=device).repeat_interleave(counts)
        r_rg   = x[right_idx]
        # distance to axis line: |(r - r_pivot) - ((r - r_pivot)·u_axis) u_axis|
        r_rel  = r_rg - r_pivot[res_ids]
        proj   = (r_rel * u_axis[res_ids]).sum(dim=1, keepdim=True) * u_axis[res_ids]
        r_perp = r_rel - proj
        I_contrib = mass[right_idx] * (r_perp.norm(dim=1)**2)
        I_scatter = x.new_zeros(R)
        I_scatter.scatter_add_(0, res_ids, I_contrib)
        I_res = I_scatter.clamp_min(1e-8)

    # -------- angular velocity and distribution to atoms --------
    # ω_res = k_omega * tau_res / I_res  (scalar about the axis, signed)
    omega = k_omega * (tau_res / I_res)               # (R,)

    # per-atom linear velocities for right-group as rigid rotation: v = ω * (u_axis × (r - r_pivot))
    v = x.new_zeros(N, 3)
    if right_idx.numel() > 0:
        res_ids = torch.arange(R, device=device).repeat_interleave((right_off[1:] - right_off[:-1]).to(torch.long))
        r_rel = x[right_idx] - r_pivot[res_ids]
        rot_dir = torch.cross(u_axis[res_ids], r_rel, dim=1)    # (M,3)
        v_rg = (omega[res_ids][:,None] * rot_dir)               # (M,3)

        # rot_dir = torch.cross(r_rel, u_axis[res_ids], dim=1)
        # v_rg = (omega[res_ids][:, None] * rot_dir)
        v.scatter_add_(0, right_idx[:,None].expand(-1,3), v_rg)

    # -------- pair-type weighting to suppress backbone↔backbone **forces** at source --------
    # (Optional): If you’d like BB–BB and BB–SC pairs to contribute less torque in the first place,
    # you can multiply Fi/Fj by pair weights before torque accumulation, using is_bb masks on i_sel/j_sel.
    # Left out here for clarity since we already focus on side-chain N/O and rotate only side chains.

    return v * dt


def lj_velocity_fn(x, t, dt, cache):
    """
    x: (N,3) or (1,N,3) tensor on same device as cache
    t: scalar (float or 0-dim tensor)
    dt: scalar
    Includes: all nonbonded except 1–2 and 1–3; 1–4 is INCLUDED (scaled).
    Optional distance cutoff (from cache).
    """
    if x.dim() == 3:
        x = x[0]
    device = x.device

    N        = cache["N"]
    mass     = cache["mass"]
    sigma    = cache["sigma"]
    epsilon  = cache["epsilon"]
    res_idx  = cache["res_idx"]
    tri_12   = cache["tri_12"]
    tri_13   = cache["tri_13"]
    tri_14   = cache["tri_14"]
    onefour  = cache["onefour_scale"]
    cutoff   = cache["cutoff"]

    # Pair indices (upper triangle i<j) – build once per call, cheap
    iu, ju = torch.triu_indices(N, N, offset=1, device=device)
    with torch.enable_grad():
        # Displacements & distances
        rij = x[iu] - x[ju]                               # (P,3)
        r2  = (rij * rij).sum(dim=1)                      # (P,)
        r   = torch.sqrt(r2.clamp_min(1e-18))             # (P,)
    
        # Optional cutoff mask
        if cutoff is not None:
            mask_cut = (r < cutoff)
        else:
            mask_cut = torch.ones_like(r, dtype=torch.bool)
    
        # Exclude 1–2 and 1–3; include 1–4
        # (tri_* are upper-tri flags; index them with [iu, ju])
        mask_12 = tri_12[iu, ju]
        mask_13 = tri_13[iu, ju]
        mask_14 = tri_14[iu, ju]
    
        # Base: all nonbonded; remove 1–2 and 1–3
        mask_nb = mask_cut & (~mask_12) & (~mask_13)
        if t > 0.75:
            mask_nb = mask_nb & (res_idx[iu] != res_idx[ju])
        
        # Combine rules for epsilon scaling: 1–4 gets scaled, others unscaled
        # (We’ll compute epsilon_ij then multiply by onefour where mask_14 is True)
        eps_i = epsilon[iu]
        eps_j = epsilon[ju]
        sig_i = sigma[iu]
        sig_j = sigma[ju]
    
        # Lorentz–Berthelot combining; keep your t-dependent sigma correction for near residues
        # Compute |res_i - res_j| and blend sigma if < 3
        dr_res = (res_idx[iu] - res_idx[ju]).abs()
        sig_LB = 0.5 * (sig_i + sig_j)
        sig_near = t * sig_LB
        sig_ij = torch.where(dr_res < 3, sig_near, sig_LB)
    
        eps_ij = torch.sqrt(eps_i * eps_j)
        # 1–4 scaling on epsilon
        eps_ij = torch.where(mask_14, eps_ij * onefour, eps_ij)
    
        # Only compute forces where mask_nb is True
        if not mask_nb.any():
            return x.new_zeros(N, 3)
    
        sel = mask_nb.nonzero(as_tuple=False).squeeze(1)
        iu_s = iu[sel]; ju_s = ju[sel]
        rij_s = rij[sel]
        r2_s  = r2[sel]
        sig_s = sig_ij[sel]
        eps_s = eps_ij[sel]
    
        inv_r2 = 1.0 / r2_s
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2
    
        # Classic LJ force magnitude: 24*eps * (2*(sig^12)/r^14 - (sig^6)/r^8) * (vector r_ij)
        # We compute as: 24*eps*inv_r2*( 2*(sig^12)*inv_r12 - (sig^6)*inv_r6 )
        sig6 = sig_s ** 6
        Fmag = 24.0 * eps_s * inv_r2 * (2.0 * (sig6 ** 2) * inv_r12 - sig6 * inv_r6)  # (P_sel,)
    
        Fvec = Fmag[:, None] * rij_s  # (P_sel,3), direction i->j
    
        # Accumulate atom-wise velocities (v_i -= F/m_i * dt ; v_j += F/m_j * dt)
        v = x.new_zeros(N, 3)
        v.scatter_add_(0, iu_s[:, None].expand(-1, 3),  (Fvec / cache["mass"][iu_s, None]) * dt)
        v.scatter_add_(0, ju_s[:, None].expand(-1, 3), -(Fvec / cache["mass"][ju_s, None]) * dt)
    return v

def _infer_element(atom_tpl):
    # Prefer template element; fall back on simple first-letter guess if missing
    if atom_tpl.element is not None:
        return atom_tpl.element
    sym = (atom_tpl.name or "C")[0].upper()
    sym = sym if sym in {"C", "N", "O", "S", "H"} else "C"
    return app.Element.getBySymbol(sym)



import torch

import torch
from collections import defaultdict

def build_hbond_cache(top, rtp_data, ff, device,
                                    # pair selection
                                    rc=0.70,                    # capture cutoff (nm)
                                    # radial potential (distance-only)
                                    r0=0.29, De=500.0, alpha=1.0,       # U = 0.5*k_pair*(r-r0)^2
                                    k_rep=50.0,      
                                    r_rep=0.60,
                                    # backbone suppression
                                    w_bb_bb=0.1, w_bb_sc=0.1, w_sc_sc=1.0,
                                    # torque → angular velocity gain
                                    k_omega=2.0):
    """
    Precompute:
      - polar *side-chain* atoms (N or O, excluding backbone N/O),
      - χ1 axis per residue that has CA–CB (pivot = CA; axis = CA→CB),
      - right-group atoms per residue (all atoms distal to CB, i.e., side chain),
      - per-atom → residue map,
      - a packed list of right-group atom indices with offsets (for fast reductions),
      - masses, upper-tri index (iu, ju),
      - scalar params (rc, r0, k_pair, weights, k_omega).
    """
    N = top.n_atoms
    elem  = [(top.atom(i).element.name.lower() if top.atom(i).element else "") for i in range(N)]
    aname = [top.atom(i).name for i in range(N)]
    rname = [top.atom(i).residue.name for i in range(N)]
    rseq  = [top.atom(i).residue.resSeq for i in range(N)]

    # Backbone identification by name
    is_bb = torch.tensor([nm in {"N","CA","C","O","OXT"} for nm in aname], device=device, dtype=torch.bool)

    # Polar side-chain atoms (exclude backbone)
    is_side = ~is_bb
    is_N = torch.tensor([e=="nitrogen" for e in elem], device=device)
    is_O = torch.tensor([e=="oxygen"  for e in elem], device=device)
    # --- residue exclusions (e.g., ban amide side chains) ---
    exclude_res = {"ASN","GLN"} 
    is_excl_res = torch.tensor([rn in exclude_res for rn in rname],
                               device=device, dtype=torch.bool)
    

    is_sc = ~is_bb
    


    # Build adjacency graph
    adj = [[] for _ in range(N)]
    for b in top.bonds:
        u, v = b[0].index, b[1].index
        is_ss = (elem[u] == "sulfur") and (elem[v] == "sulfur")
        if not is_ss:
            adj[u].append(v); adj[v].append(u)

    # χ1 axis per residue: CA–CB (skip residues lacking CB, e.g., GLY)
    res_to_atoms = defaultdict(list)
    for i, rs in enumerate(rseq):
        res_to_atoms[rs].append(i)

    residues = sorted(res_to_atoms.keys())
    has_axis    = []
    axis_resids = []
    pivot_CA    = []
    axis_CB     = []

    for rs in residues:
        atoms = res_to_atoms[rs]
        CA = next((i for i in atoms if aname[i]=="CA"), None)
        CB = next((i for i in atoms if aname[i]=="CB"), None)
        if CA is not None and CB is not None:
            has_axis.append(True)
            axis_resids.append(rs)
            pivot_CA.append(CA)
            axis_CB.append(CB)

    if len(axis_resids)==0:
        # Nothing to rotate; still return a cache (the velocity fn will return zeros)
        iu, ju = torch.triu_indices(N, N, offset=1, device=device)
        mass = torch.ones(N, device=device)
        return dict(N=N, iu=iu, ju=ju, mass=mass,
                    valid_res=torch.tensor([], device=device, dtype=torch.long),
                    atom2res=torch.full((N,), -1, device=device, dtype=torch.long),
                    right_idx=torch.tensor([], device=device, dtype=torch.long),
                    right_off=torch.tensor([0], device=device, dtype=torch.long),
                    pivot_idx=torch.tensor([], device=device, dtype=torch.long),
                    axis_idx=torch.tensor([], device=device, dtype=torch.long),
                    is_bb=is_bb,
                    rc=float(rc), r0=float(r0), 
                    De=float(De), alpha=float(alpha),
                    # k_pair=float(k_pair),
                    w_bb_bb=float(w_bb_bb), w_bb_sc=float(w_bb_sc), w_sc_sc=float(w_sc_sc),
                    k_omega=float(k_omega))

    pivot_idx = torch.tensor(pivot_CA, device=device, dtype=torch.long)  # CA
    axis_idx  = torch.tensor(axis_CB, device=device, dtype=torch.long)   # CB
    valid_res = torch.tensor(axis_resids, device=device, dtype=torch.long)

    # right-group atoms: all atoms in the residue reachable from CB WITHOUT passing through CA
    right_lists = []
    atom2res = torch.full((N,), -1, device=device, dtype=torch.long)
    for ridx, rs in enumerate(axis_resids):
        CA = pivot_CA[ridx]; CB = axis_CB[ridx]
        # BFS from CB excluding CA
        rg = []
        stack = [CB]
        visited = {CA}
        while stack:
            u = stack.pop()
            if rseq[u] == rs and u not in visited:
                visited.add(u)
                rg.append(u)
                for w in adj[u]:
                    if w != CA and rseq[w]==rs and w not in visited:
                        stack.append(w)
        # Ensure CA itself is NOT in right-group; CB should be in there
        right_lists.append(sorted(set(rg)))
        for a in rg:
            atom2res[a] = ridx

    # Pack right-group lists into one index tensor + offsets
    right_idx = torch.tensor([a for lst in right_lists for a in lst], device=device, dtype=torch.long)
    off = [0]
    for lst in right_lists:
        off.append(off[-1] + len(lst))
    right_off = torch.tensor(off, device=device, dtype=torch.long)

    # masses
    mass = torch.empty(N, device=device, dtype=torch.float32)
    charge = torch.empty(N, device=device, dtype=torch.float32)
    for i in range(N):
        if "CHARMM" in ff:
            atype, m, q = _charmm_type_mass_charge(rtp_data, rname[i], aname[i]) 
        else: 
            atype, m, q = _amber_type_mass_charge(rtp_data, rname[i], aname[i]) 
        mass[i] = float(m)
        charge[i] = float(q)

    
    is_donor_sc    = ((charge > +0.60) | is_N) & is_sc & (~is_excl_res)   # e.g., LYS/ARG cations
    is_acceptor_sc = ((charge < -0.60) | is_O) & is_sc    # e.g., ASP/GLU anions
    
    iu, ju = torch.triu_indices(N, N, offset=1, device=device)

    return dict(
        N=N, iu=iu, ju=ju,
        mass=mass,
        # residue-wise χ1 axes
        valid_res=valid_res,           # list of resSeq that have CA–CB
        atom2res=atom2res,             # atom -> local residue index (w.r.t valid_res), -1 if none
        right_idx=right_idx,           # packed indices of all right-group atoms
        right_off=right_off,           # offsets per residue into right_idx (len = nres+1)
        pivot_idx=pivot_idx,           # CA per residue
        axis_idx=axis_idx,             # CB per residue (defines axis direction CA->CB)
        # polar side-chain masks
        is_bb=is_bb,
        is_donor_sc=is_donor_sc, is_acceptor_sc=is_acceptor_sc,
        # parameters
        rc=float(rc), r0=float(r0), De=De, alpha=alpha,
        w_bb_bb=float(w_bb_bb), w_bb_sc=float(w_bb_sc), w_sc_sc=float(w_sc_sc),
        k_rep=k_rep,
        r_rep=r_rep,
        k_omega=float(k_omega)
    )



def bond_fraction(trj_ref, trj_gen, fraction=0.1):
    '''Fraction of bonds within X percent of the reference'''
    bond_pairs = [[b[0].index, b[1].index] for b in trj_ref.top.bonds]
    bond_atoms = [[b[0], b[1]] for b in trj_ref.top.bonds]
    ref_dist = md.compute_distances(trj_ref, bond_pairs)
    gen_dist = md.compute_distances(trj_gen, bond_pairs)
    np.set_printoptions(suppress=True, precision=4)

    bond_frac = np.sum((gen_dist < (1+fraction)*ref_dist) & 
                       (gen_dist > (1-fraction)*ref_dist))

    bond_frac = bond_frac / np.size(ref_dist)
    
    return bond_frac

def get_res_idxs_cut(trj, thresh=0.12, Ca_cut=2.0):
    Ca_idxs = []
    for i, atom in enumerate(trj.top.atoms):
        if 'CA' in atom.name:
            Ca_idxs.append(i)
    Ca_idxs = np.array(Ca_idxs)
    Ca_xyzs = trj.xyz[0, Ca_idxs]
    n_res = trj.n_residues
    pairs = []
    for i in range(n_res):
        for j in range(i-1):
            if np.linalg.norm(Ca_xyzs[i]-Ca_xyzs[j]) < Ca_cut:
                pairs.append((i, j))
    dist, pairs = md.compute_contacts(trj, contacts=pairs, scheme='closest')
    # look at sidechain-heavy only
    
    neighbor_pairs = [(i, i+1) for i in range(trj.n_residues-1) if (
        trj.top.residue(i).name != 'GLY' and trj.top.residue(i+1).name != 'GLY')]
    
    neighbor_dist, neighbor_pairs = md.compute_contacts(trj, contacts=neighbor_pairs, scheme='sidechain-heavy')
    dist = np.concatenate([dist, neighbor_dist], axis=-1)
    pairs = np.concatenate([pairs, neighbor_pairs], axis=0)
    res_closes = list()
    for n_res in range(trj.top.n_residues):
        pair_mask = np.array([n_res in i for i in pairs])
        res_close = np.any(dist[0, pair_mask] < thresh)
        res_closes.append(res_close)
        # if res_close:
    res_closes = np.array(res_closes)

    return res_closes

def clash_res_percent(viz_gen, thresh=0.12, Ca_cut=2.0):
    all_res_closes = list()
    #for n in tqdm(range(len(viz_gen))):
    for n in range(len(viz_gen)):
        res_closes = get_res_idxs_cut(viz_gen[n], thresh=thresh, Ca_cut=Ca_cut)
        all_res_closes.append(res_closes)
    return 100 * sum([sum(i) for i in all_res_closes]) / sum([i.shape[0] for i in all_res_closes])
    
    
# for calculating generative diversity
def ref_rmsd(trj_ref, trj_sample_list):
    
    rmsd_list = []
    for i, trj_i in enumerate(trj_sample_list):
        

        for k, (trj_if, trj_rf) in enumerate(zip(trj_i, trj_ref)):
            rmsd = md.rmsd(trj_if, trj_rf)*10
            rmsd_list.append(rmsd)
    return np.mean(rmsd_list), np.std(rmsd_list)

def sample_rmsd(trj_sample_list):
    
    rmsd_list = []
    for i, trj_i in enumerate(trj_sample_list):
        for j, trj_j in enumerate(trj_sample_list[:i]):

            for k, (trj_if, trj_jf) in enumerate(zip(trj_i, trj_j)):
                rmsd = md.rmsd(trj_if, trj_jf)*10
                rmsd_list.append(rmsd)
        

    return np.mean(rmsd_list), np.std(rmsd_list)

def sample_rmsd_percent(trj_ref, trj_sample_list):
    
    R_ref, S_ref = ref_rmsd(trj_ref, trj_sample_list)
    R_sam, S_sam = sample_rmsd(trj_sample_list)
    
    R_per = (R_ref-R_sam) / R_ref
    S_per = np.sqrt( (S_sam/R_ref)**2 + ((R_sam*S_ref)/(R_ref)**2)**2 )
    
    return R_per, S_per

# get uncertainties on diversity scores 
def jackknife_div(trj_ref, trj_sample_list):
    gen_ref = get_ref_gen_rmsds(trj_ref, trj_gens)
    gen_gen = get_sample_rmsds(trj_gens) 
    
    assert len(gen_ref) == len(gen_gen)
    
    div_mat = np.zeros((trj_ref.n_frames, len(trj_sample_list)))
    for frame_idx, (gen_gen_i, gen_ref_i) in enumerate(zip(gen_gen, gen_ref)):
        for targ in range(len(trj_sample_list)):
            gen_gen_mean = np.mean([v for i,v in gen_gen_i.items() if targ not in i])
            gen_ref_mean = np.mean(np.delete(gen_ref_i, targ))
            div_mat[frame_idx][targ] = 1 - (gen_gen_mean / gen_ref_mean)
    return div_mat.mean(0)

# Define a custom selector for filtering
class ProteinDNASelect(Select):
    def accept_residue(self, residue):
        # Accept only protein and DNA residues
        if "CA" in residue or residue.id[0] == " " and residue.resname.strip() in ["DA", "DC", "DG", "DT"]:
            return 1  # Return 1 to indicate acceptance
        else:
            return 0  # Return 0 to exclude everything else




def read_rtp_file(file_path):
    with open(file_path, 'r') as file:
        rtp_data = file.readlines()  # Read the file as a list of lines
    return rtp_data


def parse_rtp(rtp_data):
    rtp_dict = {}
    residue_type = None
    atom_dict = {}
    
    # Process each line in the RTP data
    for line in rtp_data:
        line = line.strip()
        
        # If a new residue type is found, initialize a new dictionary for it
        residue_match = re.match(r"\[\s*[A-Z0-9]*\s*\]", line)
        if residue_match:
            if residue_type:  # If there is an existing residue, save it before moving on
                rtp_dict[residue_type] = atom_dict
            
            residue_type = residue_match.group(0)[2:-2]
            atom_dict = {}
            reading_atoms = True  # Start reading atoms
        # Process atoms section (inside [ atoms ] block)
        elif residue_type and re.match(r"\[ atoms \]", line):
            # Start reading atoms after "[ atoms ]" section header
            continue
        elif re.match(r"\[ bonds \]", line) or re.match(r"\[ impropers \]", line) or re.match(r"\[ cmap \]", line) or re.match(r"\[ bondedtypes \]", line):
            # Stop reading when the "[ bonds ]" section is reached
            reading_atoms = False
            continue
        elif residue_type and reading_atoms:
            # Parse atom data line
            parts = line.split()
            if len(parts) >= 2:
                atom_name = parts[0]
                atom_type = parts[1]
                if atom_name.isupper() and atom_type.isupper():
                    atom_dict[atom_name] = atom_type

    # Don't forget to add the last residue type to the dictionary
    if residue_type:
        rtp_dict[residue_type] = atom_dict
    
    return rtp_dict

def get_charmm_data(rtp_path=f'{FLOWBACK_FF}/aminoacids.rtp', nb_path=f'{FLOWBACK_FF}/ffnonbonded.csv', bond_path=f'{FLOWBACK_FF}/bondtypes.csv'):
    lj_df   = pd.read_csv(f"{FLOWBACK_FF}/charmm_lj.csv")        
    bond_df = pd.read_csv(f"{FLOWBACK_FF}/charmm_bonds.csv")      
    angle_df = pd.read_csv(f"{FLOWBACK_FF}/charmm_angles.csv")
    map_df  = pd.read_csv(f"{FLOWBACK_FF}/pdb_to_charmm.csv")
    return map_df, lj_df, bond_df, angle_df

def get_amber_data():
    lj_df   = pd.read_csv(f"{FLOWBACK_FF}/amber_lj.csv")        
    bond_df = pd.read_csv(f"{FLOWBACK_FF}/amber_bonds.csv")      
    angle_df = pd.read_csv(f"{FLOWBACK_FF}/amber_angles.csv")
    map_df  = pd.read_csv(f"{FLOWBACK_FF}/pdb_to_amber.csv")
    return map_df, lj_df, bond_df, angle_df

def topology_signature_by_sequence(top) -> str:
    """
    Compute a stable hash key for an MDTraj topology based solely on
    its residue sequence (and chain order).

    Two topologies with the same chain ordering and residue names
    will produce the same hash, even if they are distinct objects.
    """
    # Extract residue names per chain in order
    seq_per_chain = []
    for chain in top.chains:
        residues = [res.name for res in chain.residues]
        seq_per_chain.append("-".join(residues))
    seq_str = "|".join(seq_per_chain)  # chain separator

    # Compute deterministic SHA1 hash
    return hashlib.sha1(seq_str.encode("utf-8")).hexdigest()

def tensor_bytes(obj):
    if torch.is_tensor(obj):
        return obj.nelement() * obj.element_size()
    elif isinstance(obj, dict):
        return sum([tensor_bytes(v) for v in obj.values()])
    elif isinstance(obj, (list, tuple)):
        return sum([tensor_bytes(v) for v in obj])
    else:
        return 0

def get_or_build_cache(top, cache, cache_name, cache_bytes, cache_build_fn, *args, **kwargs):
    sig = topology_signature_by_sequence(top)
    if sig not in cache:
        built = cache_build_fn(top, *args, **kwargs)
        size_increase = tensor_bytes(built)
        if cache_bytes[cache_name] + size_increase >= 1024 ** 3:
            cache.clear()
        cache[sig] = built
        cache_bytes[cache_name] += size_increase
    else:
        built = cache[sig]
    return built
    
def get_lj_info(rtp_data, lj_data, residue, atom, ff):
    if 'CHARMM' in ff:
        atype, mass, _ = _charmm_type_mass_charge(rtp_data, residue, atom)
        row = lj_data[lj_data["atom_type"] == atype].iloc[0]
        return mass, row["sigma"], row["epsilon"]
    else:
        atype, mass, _ = _amber_type_mass_charge(rtp_data, residue, atom)
        row = lj_data[lj_data["atom_type"] == atype].iloc[0]
        return mass, row["sigma"], row["epsilon"]

def get_elec_info(rtp_data, residue, atom, ff):
    if 'CHARMM' in ff:
        atype, mass, charge = _charmm_type_mass_charge(rtp_data, residue, atom)
    else:
        atype, mass, charge = _amber_type_mass_charge(rtp_data, residue, atom)
    
    
def get_bond_info(rtp_data, bond_data, bond, ff):
    i, j = bond[0], bond[1]
    if 'CHARMM' in ff:
        ti, mi, _ = _charmm_type_mass_charge(rtp_data, i.residue.name, i.name)
        tj, mj, _ = _charmm_type_mass_charge(rtp_data, j.residue.name, j.name)
    else:
        ti, mi, _ = _amber_type_mass_charge(rtp_data, i.residue.name, i.name)
        tj, mj, _ = _amber_type_mass_charge(rtp_data, j.residue.name, j.name)
    row = _bond_row(bond_data, ti, tj)
    return row["length"], row["k"], mi, mj


def get_angle_info(rtp_data, angle_data,
                   residue_i, atom_i, residue_j, atom_j, residue_k, atom_k, ff):
    """
    Return (theta0_rad, K) for angle (i-j-k).
    Adapt this to your angle_data schema (AMBER/CHARMM).
    """
    if "CHARMM" in ff:
        # Example: map to atom types, then lookup
        ti, _, _ = _charmm_type_mass_charge(rtp_data, residue_i, atom_i)
        tj, _, _ = _charmm_type_mass_charge(rtp_data, residue_j, atom_j)
        tk, _, _ = _charmm_type_mass_charge(rtp_data, residue_k, atom_k)
    else:
        # AMBER-style: use atom types from your existing helper
        ti, _, _ = _amber_type_mass_charge(rtp_data, residue_i, atom_i)
        tj, _, _ = _amber_type_mass_charge(rtp_data, residue_j, atom_j)
        tk, _, _ = _amber_type_mass_charge(rtp_data, residue_k, atom_k)
        # Try (ti,tj,tk) then (tk,tj,ti) if angle table is unordered
    try:
        row = angle_data[(angle_data["type1"]==ti) & (angle_data["type2"]==tj) & (angle_data["type3"]==tk)].iloc[0]
    except IndexError:
        row = angle_data[(angle_data["type1"]==tk) & (angle_data["type2"]==tj) & (angle_data["type3"]==ti)].iloc[0]
    theta0_rad = float(row["theta0"])
    K = float(row["k"])
    return theta0_rad, K


def build_bond_cache(top, rtp_data, bond_data, ff, device):
    """Precompute per-bond tensors so the per-step function is vectorized."""
    i_idx, j_idx = [], []
    b0_list, kb_list = [], []
    mi_list, mj_list = [], []
    keep_mask = []  # True = include bond in calculations

    # Precompute element names once
    elem_name = [top.atom(a).element.name.lower() if top.atom(a).element is not None else "" 
                 for a in range(top.n_atoms)]
    atom_name = [top.atom(a).name for a in range(top.n_atoms)]

    for b in top.bonds:
        ai, aj = b[0].index, b[1].index
        # skip S-S bonds only
        is_ss = (elem_name[ai] == "sulfur") and (elem_name[aj] == "sulfur")
        # keep_mask.append(not is_ss)
        if not is_ss:
        #     b0 = 1.0
        #     kb = 0.0
        #     mass_i = 32.07
        #     mass_j = 32.07
        # else:
            b0, kb, mass_i, mass_j = get_bond_info(rtp_data, bond_data, b, ff)
            i_idx.append(ai); j_idx.append(aj)
            b0_list.append(float(b0))
            kb_list.append(float(kb))
            mi_list.append(float(mass_i))
            mj_list.append(float(mass_j))

    cache = {
        "i": torch.tensor(i_idx, device=device, dtype=torch.long),
        "j": torch.tensor(j_idx, device=device, dtype=torch.long),
        "b0": torch.tensor(b0_list, device=device, dtype=torch.float32),
        "kb": torch.tensor(kb_list, device=device, dtype=torch.float32),
        "mi": torch.tensor(mi_list, device=device, dtype=torch.float32),
        "mj": torch.tensor(mj_list, device=device, dtype=torch.float32),
        # "mask": torch.tensor(keep_mask, device=device, dtype=torch.bool),
        "n_atoms": top.n_atoms,
    }
    return cache

def build_angle_cache(top, rtp_data, angle_data, ff, device, heavy_only=True):
    """
    Enumerate angles (i, j, k) with central atom j, and cache theta0, K, and per-atom masses.
    angle_data: user-supplied table; adapt `get_angle_info(...)` to your format.
    """
    N = top.n_atoms

    # Basic per-atom info
    residues = [top.atom(i).residue.name for i in range(N)]
    atoms    = [top.atom(i).name for i in range(N)]
    elem     = [(top.atom(i).element.name.lower() if top.atom(i).element else "") for i in range(N)]

    # Optional: skip hydrogens in the angle set (keeps them physically present; just not angle drivers)
    include_atom = [e != "hydrogen" for e in elem] if heavy_only else [True]*N

    # Build adjacency (bonds)
    adj = [[] for _ in range(N)]
    for b in top.bonds:
        u, v = b[0].index, b[1].index
        is_ss = (elem[u] == "sulfur") and (elem[v] == "sulfur")
        if not is_ss:
            adj[u].append(v); adj[v].append(u)

    # Enumerate unique angles i-j-k (i<k to avoid duplicates)
    I, J, K = [], [], []
    for j in range(N):
        if not include_atom[j]:
            continue
        nbrs = [n for n in adj[j] if include_atom[n]]
        ln = len(nbrs)
        if ln < 2: 
            continue
        for a in range(ln):
            i = nbrs[a]
            for b in range(a+1, ln):
                k = nbrs[b]
                # record angle i-j-k once; order matters only for orientation, not energy
                I.append(i); J.append(j); K.append(k)

    if len(I) == 0:
        # No angles to process
        mass = torch.ones(N, device=device, dtype=torch.float32)
        return {
            "N": N, 
            "i": torch.tensor([], device=device, dtype=torch.long),
            "j": torch.tensor([], device=device, dtype=torch.long),
            "k": torch.tensor([], device=device, dtype=torch.long),
            "theta0": torch.tensor([], device=device, dtype=torch.float32),
            "K": torch.tensor([], device=device, dtype=torch.float32),
            "mass": mass,
        }

    i_t = torch.tensor(I, device=device, dtype=torch.long)
    j_t = torch.tensor(J, device=device, dtype=torch.long)
    k_t = torch.tensor(K, device=device, dtype=torch.long)

    # Per-atom masses (reuse whatever mapping you already use; this version assumes
    # you have `_amber_type_and_mass` or equivalent available from your codebase).
    mass = torch.empty(N, device=device, dtype=torch.float32)
    for a in range(N):
        # You likely already have this helper; adapt if needed:
        if "CHARMM" in ff:
            # Using CHARMM: map residue+atom -> atom_type -> mass (you can route via your LJ table if convenient)
            atype, m, _ = _charmm_type_mass_charge(rtp_data, residues[a], atoms[a])  # existing helper in your code
            mass[a] = float(m)
        else:
            atype, m, _ = _amber_type_mass_charge(rtp_data, residues[a], atoms[a])  # existing helper in your code
            mass[a] = float(m)

    # Angle parameters (theta0, K) lookups
    # You need to implement `get_angle_info` for your angle_data format.
    theta0_list, K_list = [], []
    for ii, jj, kk in zip(I, J, K):
        th0, Kval = get_angle_info(
            rtp_data, angle_data,
            residue_i=residues[ii], atom_i=atoms[ii],
            residue_j=residues[jj], atom_j=atoms[jj],
            residue_k=residues[kk], atom_k=atoms[kk],
            ff=ff
        )
        theta0_list.append(float(th0))  # radians
        K_list.append(float(Kval))      # energy/rad^2

    theta0 = torch.tensor(theta0_list, device=device, dtype=torch.float32)
    Kparam = torch.tensor(K_list, device=device, dtype=torch.float32)

    return {"N": top.n_atoms, "i": i_t, "j": j_t, "k": k_t,
            "theta0": theta0, "K": Kparam, "mass": mass}

def build_lj_cache(top, rtp_data, lj_data, ff, device, onefour_scale=1.0, cutoff=None):
    """
    Precompute per-atom LJ params + pair-type (1-2, 1-3, 1-4) masks.

    onefour_scale: multiply epsilon_ij by this factor for 1–4 pairs (e.g., 0.5 for AMBER-like)
    cutoff: optional float (in same units as x) to mask pairs by distance at runtime
    """
    N = top.n_atoms
    residues = [top.atom(i).residue.name for i in range(N)]
    atoms    = [top.atom(i).name   for i in range(N)]
    res_idx  = torch.tensor([top.atom(i).residue.resSeq for i in range(N)], device=device, dtype=torch.long)
    elem     = [(top.atom(i).element.name.lower() if top.atom(i).element else "") for i in range(N)]
    # Per-atom: mass, sigma, epsilon
    mass = torch.empty(N, device=device, dtype=torch.float32)
    sigma = torch.empty(N, device=device, dtype=torch.float32)
    epsilon = torch.empty(N, device=device, dtype=torch.float32)

    # Fill params from your existing helper
    for i in range(N):
        m_i, s_i, e_i = get_lj_info(rtp_data, lj_data, residues[i], atoms[i], ff)
        mass[i]    = float(m_i)
        sigma[i]   = float(s_i)
        epsilon[i] = float(e_i)

    # Build 1-2, 1-3, 1-4 sets via graph expansion (once)
    adj = [[] for _ in range(N)]
    for b in top.bonds:
        u, v = b[0].index, b[1].index
        is_ss = (elem[u] == "sulfur") and (elem[v] == "sulfur")
        if not is_ss:
            adj[u].append(v); adj[v].append(u)

    bonds_12 = set()
    for u in range(N):
        for v in adj[u]:
            if u < v: bonds_12.add((u, v))

    bonds_13 = set()
    for u in range(N):
        for v in adj[u]:
            for w in adj[v]:
                if w == u: continue
                a, b = sorted((u, w))
                if a != b and (a, b) not in bonds_12:
                    bonds_13.add((a, b))

    bonds_14 = set()
    for u in range(N):
        for v in adj[u]:
            for w in adj[v]:
                if w == u: continue
                for z in adj[w]:
                    if z in (v, u): continue
                    a, b = sorted((u, z))
                    if a != b and (a, b) not in bonds_12 and (a, b) not in bonds_13:
                        bonds_14.add((a, b))

    # Pack pair-type masks into a single lookup on device (upper-tri indexing)
    # We’ll populate three boolean tensors for quick membership testing.
    tri_12 = torch.zeros((N, N), device=device, dtype=torch.bool)
    tri_13 = torch.zeros((N, N), device=device, dtype=torch.bool)
    tri_14 = torch.zeros((N, N), device=device, dtype=torch.bool)

    if bonds_12:
        i12, j12 = zip(*bonds_12)
        tri_12[torch.tensor(i12, device=device), torch.tensor(j12, device=device)] = True
    if bonds_13:
        i13, j13 = zip(*bonds_13)
        tri_13[torch.tensor(i13, device=device), torch.tensor(j13, device=device)] = True
    if bonds_14:
        i14, j14 = zip(*bonds_14)
        tri_14[torch.tensor(i14, device=device), torch.tensor(j14, device=device)] = True

    

    # ---------------- precompute upper-tri pairs ----------------
    iu, ju = torch.triu_indices(N, N, offset=1, device=device)

    # ---------------- bundle cache ----------------
    cache = {
        "N": N,
        "mass": mass,
        "sigma": sigma,
        "epsilon": epsilon,
        "res_idx": res_idx,
        "tri_12": tri_12,
        "tri_13": tri_13,
        "tri_14": tri_14,
        "onefour_scale": float(onefour_scale),
        "cutoff": None if cutoff is None else float(cutoff),

        # torque-based rotation fields
        # "pivot_idx": pivot_idx,     # CA per rotatable residue
        # "axis_idx": axis_idx,       # CB per rotatable residue
        # "right_idx": right_idx,     # packed indices of atoms distal to CB
        # "right_off": right_off,     # offsets into right_idx (len = R+1)
        # "atom2res": atom2res,       # atom -> local residue-with-axis index (−1 if none)

        # neighbor pairs to reuse each step
        "iu": iu, "ju": ju,
    }
    return cache
        
def atom_to_steps(n_atoms):
    return int(np.floor((21.28 - 0.00105 * n_atoms) / (0.0007 * n_atoms)))

# --- helpers ---
def _amber_type_mass_charge(map_df, resname, atom_name):
    """Map (resname, atom) from PDB to (amber_type, mass) using pdb_to_amber.csv."""
    if resname == 'HIS':
        resname = 'HID'
    row = map_df.loc[
        (map_df["resname"] == resname) & (map_df["atom"] == atom_name)
    ].iloc[0]
    return row["amber_type"], row["mass_dalton"], row["charge_e"]

def _charmm_type_mass_charge(map_df, resname, atom_name):
    """Map (resname, atom) from PDB to (amber_type, mass) using pdb_to_amber.csv."""
    if resname == 'HIS':
        resname = 'HSD'
    row = map_df.loc[
        (map_df["resname"] == resname) & (map_df["atom"] == atom_name)
    ].iloc[0]
    return row["charmm_type"], row["mass_dalton"], row["charge_e"]

def _bond_row(bond_df, type_i: str, type_j: str):
    """Lookup an unordered bond (type_i, type_j) in amber_bonds.csv."""
    row = bond_df[(bond_df["type1"] == type_i) & (bond_df["type2"] == type_j)]
    if row.empty:
        row = bond_df[(bond_df["type1"] == type_j) & (bond_df["type2"] == type_i)]
    return row.iloc[0]
    
