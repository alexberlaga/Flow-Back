import os, glob
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset, DataLoader

import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import os
import argparse

from torch.utils.data import Subset
# ---------------------------
# Dataset: loads your .pt
# ---------------------------
CA_TYPE = 2
CB_TYPE = 3
N_TYPE  = 18
PRIOR_NAME = "decision_frame_3"


def pad_to_Amax_1d(x: torch.Tensor, Amax: int, pad_val=0):
    out = x.new_full((Amax,), pad_val)
    out[: x.shape[0]] = x
    return out

def pad_to_Amax_2d(x: torch.Tensor, Amax: int, pad_val=0.0):
    # x: (A, D)
    out = x.new_full((Amax, x.shape[1]), pad_val)
    out[: x.shape[0]] = x
    return out

def pad_ca_pos(ca_pos: torch.Tensor, Amax: int, pad_val=0.0):
    # input: (A,3) or (1,A,3) -> output: (1,Amax,3)
    if ca_pos.ndim == 2:
        ca_pos = ca_pos.unsqueeze(0)
    A = ca_pos.shape[1]
    out = ca_pos.new_full((1, Amax, 3), pad_val)
    out[:, :A, :] = ca_pos
    return out

def ca_atomidx_to_residx(ca_atom_idx_per_atom: torch.Tensor) -> torch.Tensor:
    """
    ca_atom_idx_per_atom: (A,) long, each atom stores the CA atom index of its residue (or -1).
    returns: (A,) long, dense residue ids [0..R-1] (or -1 where invalid).
    """
    ca_ids = ca_atom_idx_per_atom
    res_id = torch.full_like(ca_ids, -1)

    valid = ca_ids >= 0
    if not valid.any():
        return res_id

    uniq = torch.unique(ca_ids[valid])
    uniq, _ = torch.sort(uniq)

    lut = torch.full((int(uniq[-1].item()) + 1,), -1, device=ca_ids.device, dtype=torch.long)
    lut[uniq] = torch.arange(len(uniq), device=ca_ids.device, dtype=torch.long)

    res_id[valid] = lut[ca_ids[valid]]
    return res_id


def collate_chi(batch):
    Amax = max(b["atom_feats"].shape[0] for b in batch)
    Rmax = max(b["chi_vec"].shape[1] for b in batch)

    def pad_R(x: torch.Tensor, pad_val=0.0):
        # x: (1,R) or (something,R). Your chi_vec looks like (1,R) or (B,R) depending on saved.
        out = x.new_full((Rmax,), pad_val)
        out[: x.shape[1]] = x.squeeze(0)
        return out

    feats      = torch.stack([pad_to_Amax_1d(b["feats"],      Amax, 0).long() for b in batch], dim=0)
    atom_feats = torch.stack([pad_to_Amax_1d(b["atom_feats"], Amax, 0).long() for b in batch], dim=0)
    mask       = torch.stack([pad_to_Amax_1d(b["mask"].to(torch.long),     Amax, 0).bool() for b in batch], dim=0)
    res_mask   = torch.stack([pad_to_Amax_1d(b["res_mask"].to(torch.long), Amax, 0).bool() for b in batch], dim=0)

    ca_pos = torch.stack([pad_ca_pos(b["ca_pos"], Amax, 0.0) for b in batch], dim=0)                 # (B,1,Amax,3)
    prior  = torch.stack([pad_to_Amax_2d(b[PRIOR_NAME][0], Amax, 0.0) for b in batch], dim=0)        # (B,Amax,3)

    aa_to_cg = torch.stack([pad_to_Amax_1d(b["aa_to_cg"], Amax, -1).long() for b in batch], dim=0)   # (B,Amax)
    chi_vec  = torch.stack([pad_R(b["chi_vec"], 0.0) for b in batch], dim=0)                         # (B,Rmax)

    return dict(
        feats=feats,
        atom_feats=atom_feats,
        mask=mask,
        res_mask=res_mask,
        ca_pos=ca_pos,
        **{PRIOR_NAME: prior},
        aa_to_cg=aa_to_cg,
        chi_vec=chi_vec,
    )

    
class ChiralityPTDataset(Dataset):
    def __init__(self, dirs: List[str], frame: bool = False, max_n: int = 100000):
        self.files = []
        for d in dirs:
            if frame:
                candidates = sorted(glob.glob(os.path.join(d, "**", "frame*.pt"), recursive=True))
            else:
                candidates = sorted(glob.glob(os.path.join(d, "**", "*.pt"), recursive=True))
            self.files.extend(candidates)
        if not self.files:
            raise FileNotFoundError(f"No .pt files found under: {dirs}")
        if len(self.files) > max_n:
            self.files = random.sample(self.files, max_n)
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location="cpu")
        ca_pos = d["ca_pos"]
        if not torch.is_tensor(ca_pos):
            ca_pos = torch.tensor(ca_pos, dtype=torch.float32)

        prior = d[PRIOR_NAME]
        if not torch.is_tensor(prior):
            prior = torch.tensor(prior, dtype=torch.float32)

        # --- keep full chirality vector ---
        chi_vec = d["chi"]
        R = chi_vec.shape[1]
        
        return {
            "feats": d["res_ohe"].long(),        # (A,)
            "atom_feats": d["atom_ohe"].long(),  # (A,)
            "mask": d["mask"].bool(),            # (A,)
            "res_mask": torch.ones_like(d["mask"]).bool(),
            "ca_pos": ca_pos,                    # (A,3)
            PRIOR_NAME: prior,                      # (A,3)
            "aa_to_cg": ca_atomidx_to_residx(d["aa_to_cg"].long()),   # shape (A,)  each entry is an atom index
            "chi_vec": chi_vec,                  # (R,)
            "R": R,                              # int (optional)
            # keep res_idx if you want, but you won’t use it for k-sampling
            "path": self.files[idx],             # optional, useful for deterministic eval/debug
        }


# ---------------------------
# EGNN -> residue head model
# ---------------------------
class ChiralityPredictor(nn.Module):
    def __init__(self, dim, res_vocab, atom_vocab, head_hidden: int = 256):
        super().__init__()
        # self.egnn = base_egnn_model
        # self.dim = base_egnn_model.res_emb.embedding_dim
        # res_vocab = base_egnn_model.res_emb.num_embeddings
        # atom_vocab = base_egnn_model.atom_emb.num_embeddings
        self.dim = dim
        self.res_type_emb = nn.Embedding(res_vocab, self.
                                         dim)
        self.pos_mlp = nn.Sequential(
            nn.Linear(4, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.prior_enc = ResiduePriorEncoder(atom_vocab=atom_vocab, dim=self.dim)

        self.q_proj = nn.Linear(2 * self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)

        self.head = nn.Sequential(
            nn.Linear(2 * self.dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(head_hidden, 1),
        )

    @staticmethod
    def _true_lengths_from_aa_to_cg(aa_to_cg: torch.Tensor) -> torch.Tensor:
        # aa_to_cg: (B,Amax) dense residue ids per atom, or -1
        B = aa_to_cg.shape[0]
        L = []
        for b in range(B):
            valid = aa_to_cg[b] >= 0
            Lb = int(aa_to_cg[b, valid].max().item() + 1) if valid.any() else 1
            L.append(Lb)
        return aa_to_cg.new_tensor(L, dtype=torch.float32)  # (B,)

    def _pos_emb(self, res_idx: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        # res_idx: (B,), L: (B,)
        i = res_idx.to(torch.float32)
        den = torch.clamp(L - 1.0, min=1.0)
        p  = i / den
        dC = (den - i) / den
        k = 3.0
        isN = (i < k).to(torch.float32)
        isC = ((den - i) < k).to(torch.float32)
        return self.pos_mlp(torch.stack([p, dC, isN, isC], dim=-1))  # (B,dim)

    def _prior_repr_for_batch(self, prior_in, atom_feats, aa_to_cg, res_mask, res_idx):
        """
        prior_in:   (B,Amax,3)
        atom_feats: (B,Amax)
        aa_to_cg:   (B,Amax) dense residue ids per atom (or -1)
        res_mask:   (B,Amax) (unchanged semantics)
        res_idx:    (B,)
        Returns:
          prior_repr: (B,dim)
          atom_mask:  (B,Amax) EXACT SAME MASK as before
        """
        B, Amax, _ = prior_in.shape
        prior_repr_list = []
        atom_mask_list  = []

        for b in range(B):
            # EXACT SAME mask/logic you had:
            atom_mask = (aa_to_cg[b] == res_idx[b]) & res_mask[b]
            atom_mask_list.append(atom_mask)

            pos_xyz = prior_in[b][atom_mask]         # (Na,3)
            types   = atom_feats[b][atom_mask]       # (Na,)

            if pos_xyz.numel() == 0:
                prior_repr_list.append(prior_in.new_zeros((self.dim,)))
                continue

            pos_xyz_local = residue_local_frame_coords(pos_xyz, types, rms_normalize=True)
            tokens = self.prior_enc(pos_xyz_local, types)  # (Na,dim)

            # intra-residue all-to-all (same spirit as what we wrote before)
            Na = tokens.shape[0]
            if Na == 1:
                prior_repr_b = tokens[0]
            else:
                rel = (tokens @ tokens.transpose(0, 1)) / (self.dim ** 0.5)
                rel = rel.clone()
                rel.fill_diagonal_(float("-inf"))
                w_rel = torch.softmax(rel, dim=-1)          # (Na,Na)
                ctx = w_rel @ tokens                         # (Na,dim)
                prior_repr_b = ctx.mean(dim=0)               # (dim,)

            prior_repr_list.append(prior_repr_b)

        prior_repr = torch.stack(prior_repr_list, dim=0)     # (B,dim)
        atom_mask  = torch.stack(atom_mask_list, dim=0)      # (B,Amax)
        return prior_repr, atom_mask

    def forward(self, atom_h, feats, atom_feats, mask, res_mask, ca_pos, prior, aa_to_cg, res_idx):
        # EGNN atom embeddings (unchanged call + masks)
        # atom_h, _ = self.egnn(
        #     feats=feats,
        #     coors=prior,
        #     ca_pos=ca_pos,
        #     time=0.0,
        #     atom_feats=atom_feats,
        #     mask=mask
        # )  # (B,Amax,dim)

        prior_in = prior * 100.0
        B, Amax, dim = atom_h.shape
        res_idx = res_idx.long()  # (B,)

        # position embedding (same features/logic, just isolated)
        L = self._true_lengths_from_aa_to_cg(aa_to_cg)        # (B,)
        pos_emb = self._pos_emb(res_idx, L)                   # (B,dim)

        # prior repr and atom_mask (EXACT SAME atom_mask definition)
        prior_repr, atom_mask = self._prior_repr_for_batch(
            prior_in=prior_in,
            atom_feats=atom_feats,
            aa_to_cg=aa_to_cg,
            res_mask=res_mask,
            res_idx=res_idx
        )  # (B,dim), (B,Amax)

        # residue type token from first atom in residue (same method as before)
        has_any = atom_mask.any(dim=1)
        first_atom = atom_mask.float().argmax(dim=1)
        first_atom = torch.where(has_any, first_atom, torch.zeros_like(first_atom))

        b_idx = torch.arange(B, device=atom_h.device)
        res_type_token = feats[b_idx, first_atom]             # (B,)
        rtype_emb = self.res_type_emb(res_type_token)         # (B,dim)

        # cross-attention over EGNN atom embeddings (SAME attn scope: atom_mask)
        q = self.q_proj(torch.cat([pos_emb, rtype_emb], dim=-1))  # (B,dim)
        k = self.k_proj(atom_h)                                   # (B,Amax,dim)
        v = self.v_proj(atom_h)                                   # (B,Amax,dim)

        scores = (k * q[:, None, :]).sum(dim=-1) / (dim ** 0.5)   # (B,Amax)
        scores = scores.masked_fill(~atom_mask, float("-inf"))    # SAME mask
        w = torch.softmax(scores, dim=-1)                         # (B,Amax)
        atom_ctx = (w[:, :, None] * v).sum(dim=1)                 # (B,dim)

        x = torch.cat([atom_ctx, prior_repr], dim=-1)             # (B,2*dim)
        return self.head(x).squeeze(-1)                           # (B,)


def residue_local_frame_coords(pos_xyz: torch.Tensor, atom_types: torch.Tensor, eps=1e-6, rms_normalize=True):
    m_ca = (atom_types == CA_TYPE)
    m_cb = (atom_types == CB_TYPE)
    m_n  = (atom_types == N_TYPE)
    if not (m_ca.any() and m_cb.any() and m_n.any()):
        m_cb = (atom_types == 27)
        # raise ValueError("Missing CA/CB/N in residue atoms; cannot build local frame.")

    CA = pos_xyz[m_ca].mean(dim=0)
    CB = pos_xyz[m_cb].mean(dim=0)
    N  = pos_xyz[m_n ].mean(dim=0)

    e1 = (CB - CA)
    e1 = e1 / e1.norm().clamp(min=eps)

    v2 = (N - CA)
    v2 = v2 - (v2 @ e1) * e1
    e2 = v2 / v2.norm().clamp(min=eps)

    e3 = torch.cross(e1, e2, dim=0)
    e3 = e3 / e3.norm().clamp(min=eps)

    R = torch.stack([e1, e2, e3], dim=1)          # (3,3)
    pos_local = (pos_xyz - CA) @ R                # (Na,3)

    if rms_normalize:
        rms = pos_local.pow(2).sum(dim=1).mean().sqrt().clamp(min=eps)
        pos_local = pos_local / rms

    return pos_local


class ResiduePriorEncoder(nn.Module):
    """
    Minimal: treat local-frame coordinates as features.
    - Takes pos_local (Na,3) and atom_types (Na,)
    - Outputs per-atom tokens (Na,dim)
    """
    def __init__(self, atom_vocab: int, dim: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.atom_emb = nn.Embedding(atom_vocab, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim + 4, hidden),   # atom_emb + [x,y,z,r]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, pos_local: torch.Tensor, atom_types: torch.Tensor) -> torch.Tensor:
        # pos_local: (Na,3)
        # atom_types: (Na,)
        r = pos_local.norm(dim=-1, keepdim=True)          # (Na,1)
        aemb = self.atom_emb(atom_types)                  # (Na,dim)
        x = torch.cat([aemb, pos_local, r], dim=-1)       # (Na,dim+4)
        tok = self.mlp(x)                                 # (Na,dim)
        return tok



def make_test_subset_loader(ds_test, n_sub=100, seed=0, batch_size=1, num_workers=0):
    rng = random.Random(seed)
    n = len(ds_test)
    n_sub = min(n_sub, n)
    idxs = list(range(n))
    rng.shuffle(idxs)
    idxs = idxs[:n_sub]
    ds_sub = Subset(ds_test, idxs)
    dl_sub = DataLoader(ds_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dl_sub, idxs

def true_lengths_from_aa_to_cg_batch(aa_to_cg: torch.Tensor):
    # aa_to_cg: (B,Amax)
    B = aa_to_cg.shape[0]
    Lb = []
    for b in range(B):
        valid = aa_to_cg[b] >= 0
        Lb.append(int(aa_to_cg[b, valid].max().item() + 1) if valid.any() else 1)
    return Lb

def sample_res_idxs(Lb, q: int, device):
    # returns: (q,B)
    B = len(Lb)
    out = torch.empty((q, B), device=device, dtype=torch.long)
    for b, L in enumerate(Lb):
        out[:, b] = torch.randint(low=0, high=L, size=(q,), device=device)
    return out

def batch_logits_and_labels(egnn, model, batch, res_idx_vec):
    """
    res_idx_vec: (B,) long; one residue index per sample.
    Returns:
      logit: (B,)
      y:     (B,) float
    """
    atom_h, _ = egnn(
                feats=batch["feats"],
                coors=batch[PRIOR_NAME],
                ca_pos=batch["ca_pos"][:, 0],
                time=0.3,
                atom_feats=batch["atom_feats"],
                mask=batch["mask"]
            )    
    logit = model(
        atom_h=atom_h,
        feats=batch["feats"],
        atom_feats=batch["atom_feats"],
        mask=batch["mask"],
        res_mask=batch["res_mask"],
        ca_pos=batch["ca_pos"][:, 0],          # unchanged
        prior=batch[PRIOR_NAME],
        aa_to_cg=batch["aa_to_cg"],
        res_idx=res_idx_vec,
    )

    rows = torch.arange(res_idx_vec.shape[0], device=res_idx_vec.device)
    chi_sel = batch["chi_vec"][rows, res_idx_vec]   # unchanged meaning: select chi at residue
    y = (chi_sel > 0).float()
    return logit, y
    
@torch.no_grad()
def eval_epoch(egnn, model, dl, device, pos_weight_val: float = 3.0, q: int = 16):
    model.eval()
    pos_weight = torch.tensor(float(pos_weight_val), device=device)

    all_logits, all_y = [], []
    losses = []

    for batch in dl:
        for k in batch:
            batch[k] = batch[k].to(device)

        Lb = true_lengths_from_aa_to_cg_batch(batch["aa_to_cg"])
        res_idxs = sample_res_idxs(Lb, q=q, device=device)  # (q,B)

        for r in res_idxs:  # r: (B,)
            logit, y = batch_logits_and_labels(egnn, model, batch, r)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
            losses.append(loss.item())
            all_logits.append(logit.detach().cpu())
            all_y.append(y.detach().cpu())

    loss_avg = float(np.mean(losses)) if losses else float("nan")
    if not all_y:
        return loss_avg, float("nan"), float("nan")

    y_all = torch.cat(all_y).numpy()
    logit_all = torch.cat(all_logits).numpy()
    prob_all = 1.0 / (1.0 + np.exp(-logit_all))

    if np.unique(y_all).size < 2:
        return loss_avg, float("nan"), float("nan")

    return loss_avg, float(roc_auc_score(y_all, prob_all)), float(average_precision_score(y_all, prob_all))



def save_checkpoint(path, model, opt, epoch, metrics: dict):
    ckpt = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "opt_state_dict": opt.state_dict(),
        "metrics": metrics,
    }
    torch.save(ckpt, path)
    
if __name__ == "__main__":
    from src.utils.evaluation import load_model
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="chirality_checkpoints",
        help="Directory to save checkpoints (can be outside the current folder).",
    )
    args = parser.parse_args()

    proteins = "1FME 2WAV lambda NTL9 chignolin 2JOF NuG2 UVF A3D".split()

    train_dir = "data/train/pre_train_ckp-15_noise-0.01"
    test_dirs = [f"outputs/{p}/train_charmm_n10_post_ckp-200_noise-0.01/" for p in proteins]

    thresh = 0.7
    seed = 0
    batch_size = 4

    print(f"TRAIN dir: {train_dir}")
    print(f"TEST  proteins ({len(proteins)}): {proteins}")

    # Train set = train folder only
    ds_train = ChiralityPTDataset([train_dir], frame=False)

    # Test set = protein folders, capped at max_n=1000 (handled by dataset subsampling)
    ds_test  = ChiralityPTDataset(test_dirs, frame=False, max_n=1000)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_chi)
    dl_test  = DataLoader(ds_test, batch_size=batch_size, shuffle=False, collate_fn=collate_chi)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load FlowBack EGNN ----
    model_path_train = "models/pre_train"
    ckp_train = 15
    base_egnn_train = load_model(model_path_train, ckp_train, device)

    model_path_test = "jobs/train_charmm_n10_post"
    ckp_test = 200
    base_egnn_test = load_model(model_path_test, ckp_test, device)
    pred_dim = base_egnn_train.res_emb.embedding_dim
    res_vocab = base_egnn_train.res_emb.num_embeddings
    atom_vocab = base_egnn_train.atom_emb.num_embeddings
    model = ChiralityPredictor(pred_dim, res_vocab, atom_vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ---------------------------
    # loss logging
    # ---------------------------
    n_epochs = 1000
    train_losses, test_losses = [], []
    train_auc, test_auc = [], []
    train_ap, test_ap = [], []

    # ---- FREEZE EGNN ONCE (not every epoch) ----
    for p in base_egnn_train.parameters():
        p.requires_grad = False
    base_egnn_train.eval()
    for p in base_egnn_test.parameters():
        p.requires_grad = False
    base_egnn_test.eval()

    # class imbalance weight (you already used w=3)
    pos_w = 7.0

    # >>> NEW: checkpoint dir from CLI
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    q = 1
    for epoch in range(n_epochs):
        model.train()


        w = torch.tensor(float(pos_w), device=device)
        PRINT_EVERY = 20_000
        all_y = []
        all_logits = []
        running_losses = []
        running_logits = []
        running_y = []
        
        seen = 0
        for batch in tqdm(dl_train, desc=f"epoch {epoch:03d} train"):
            for k in batch:
                batch[k] = batch[k].to(device)
            chi_vec = batch["chi_vec"].squeeze()  # (R,)
            R = chi_vec.shape[1]
            B = batch[PRIOR_NAME].shape[0]
            
            # true residue counts per sample from aa_to_cg (atom->res mapping)
            Lb = []
            for b in range(B):
                valid = batch["aa_to_cg"][b] >= 0
                Lb.append(int(batch["aa_to_cg"][b, valid].max().item() + 1) if valid.any() else 1)
            
            # res_idxs: (q, B), each column sampled from [0, Lb[b])
            res_idxs = torch.empty((q, B), device=device, dtype=torch.long)
            for b in range(B):
                res_idxs[:, b] = torch.randint(low=0, high=Lb[b], size=(q,), device=device)
            losses = []
            atom_h, _ = base_egnn_train(
                feats=batch["feats"],
                coors=batch[PRIOR_NAME],
                ca_pos=batch["ca_pos"][:, 0],
                time=0.3,
                atom_feats=batch["atom_feats"],
                mask=batch["mask"]
            )  # (B,Amax,dim)
            for r in res_idxs:
                logit = model(
                    atom_h=atom_h,
                    feats=batch["feats"],
                    atom_feats=batch["atom_feats"],
                    mask=batch["mask"],
                    res_mask=batch["res_mask"],
                    ca_pos=batch["ca_pos"][:, 0],
                    prior=batch[PRIOR_NAME],
                    aa_to_cg=batch["aa_to_cg"],
                    res_idx=torch.tensor(r),
                )
                rows = torch.arange(r.shape[0], device=r.device)
                chi_sel = chi_vec[rows, r]          # (k,)
                y = (chi_sel > 0).float()
                all_y.append(y.detach().flatten().cpu())
                all_logits.append(logit.detach().flatten().cpu())
                running_logits.append(logit.detach().flatten().cpu())
                running_y.append(y.detach().flatten().cpu())
                losses.append(F.binary_cross_entropy_with_logits(logit, y, pos_weight=w))
            
            loss = torch.stack(losses).mean()
            running_losses.append(float(loss.item()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            
            seen += y.numel()   # number of proteins / residue-samples seen
            if seen >= PRINT_EVERY:
                y_win = torch.cat(running_y).numpy()
                logit_win = torch.cat(running_logits).numpy()
                prob_win = 1.0 / (1.0 + np.exp(-logit_win))
            
                win_loss = float(np.mean(running_losses))
            
                win_auc = float("nan")
                win_ap = float("nan")
                if np.unique(y_win).size == 2:
                    win_auc = roc_auc_score(y_win, prob_win)
                    win_ap = average_precision_score(y_win, prob_win)
            
                print(
                    f"[train @ {PRINT_EVERY:5d} samples] "
                    f"loss={win_loss:.6f}  auc={win_auc:.4f}  ap={win_ap:.4f}"
                )
               
                # reset window
                running_losses.clear()
                running_logits.clear()
                running_y.clear()
                seen = 0

        # ---- TRAIN metrics (epoch-level) ----
        train_loss = float(np.mean(running_losses)) if running_losses else float("nan")
        y_tr = torch.cat(all_y).numpy() if all_y else np.array([])
        
        logit_tr = torch.cat(all_logits).numpy() if all_logits else np.array([])
        prob_tr = 1.0 / (1.0 + np.exp(-logit_tr)) if logit_tr.size else np.array([])

        tr_auc = float("nan")
        tr_ap = float("nan")
        if y_tr.size and (np.unique(y_tr).size == 2):
            tr_auc = float(roc_auc_score(y_tr, prob_tr))
            tr_ap = float(average_precision_score(y_tr, prob_tr))

        # ---- TEST metrics (epoch-level) ----
        test_loss, te_auc, te_ap = eval_epoch(base_egnn_test, model, dl_test, device, pos_weight_val=pos_w)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_auc.append(tr_auc)
        test_auc.append(te_auc)
        train_ap.append(tr_ap)
        test_ap.append(te_ap)

        print(
            f"epoch {epoch:03d}  "
            f"train_loss={train_loss:.6f} train_auc={tr_auc:.4f} train_ap={tr_ap:.4f}  "
            f"test_loss={test_loss:.6f}  test_auc={te_auc:.4f}  test_ap={te_ap:.4f}"
        )

        # ---- checkpoint every 10 epochs ----
        
        metrics = {
            "train_loss": train_loss, "train_auc": np.nanmean(train_auc[-10:]), "train_ap": np.nanmean(train_ap[-10:]),
            "test_loss": test_loss,  "test_auc": np.nanmean(test_auc[-10:]),  "test_ap": np.nanmean(test_ap[-10:]),
            "pos_weight": float(pos_w),
            "model_path_train": model_path_train, "ckp_train": int(ckp_train),
            "model_path_test": model_path_test, "ckp_test": int(ckp_test),
            "seed": int(seed), 
        }
        ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch+1:04d}.pt")
        save_checkpoint(ckpt_path, model, opt, epoch=epoch, metrics=metrics)
        print(f"Saved checkpoint: {ckpt_path}")

    # save for plotting later
    out_path = f"{ckpt_dir}/chirality_losses_seed0.npz"
    np.savez(
        out_path,
        train_losses=np.array(train_losses, dtype=np.float32),
        test_losses=np.array(test_losses, dtype=np.float32),
        train_auc=np.array(train_auc, dtype=np.float32),
        test_auc=np.array(test_auc, dtype=np.float32),
        train_ap=np.array(train_ap, dtype=np.float32),
        test_ap=np.array(test_ap, dtype=np.float32),
        seed=np.int32(seed),
        model_path_train=model_path_train,
        ckp_train=np.int32(ckp_train),
        model_path_test=model_path_test,
        ckp_test=np.int32(ckp_test),
    )
    print(f"Saved losses/metrics to {out_path}")
