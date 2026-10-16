"""Architecture and hyperparameter search for the age model on top of the 14 trained"""
import argparse
import os
import sys

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.amp import autocast, GradScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import channel_encoder_search_arch as A  # noqa: E402
import channel_encoder_search_multi as S  # noqa: E402
import config_review as C  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
CKPT = os.path.join(HERE, "final_encoders")
DB_PATH = os.path.join(HERE, "age_model_optuna.db")
STORAGE = "sqlite:///" + DB_PATH + "?timeout=120"
STUDY = os.environ.get("AGE_STUDY", "age_attention")
EPOCHS = int(os.environ.get("AGE_EPOCHS", 12))
WIN_STRIDE = int(os.environ.get("AGE_WIN_STRIDE", 2))
# Training pools. "2ds" is what phase 1 used (ds005385 + ds003775, eyes-closed).
# "3ds" adds ds004148, the only other dataset with a genuine eyes-closed condition.
POOL_FILES = {"2ds": "public_pool"}
FLOOR = 14.07
# Same shape of objective as teh encoder search, for the same reason: optimising
# within-pool fit alone produced encoders that reached r = 0.71 and transferred to
IH_W = 1.0
# WHICH IN-HOUSE SUBSETS THE OBJECTIVE READS. Requested 2026-10-08: optimise on
# in-house train + validation + hold-out + all four pooled, not validation alone.
# 1. LOCKED IS NO LONGER A HELD-OUT SET. With IH_W_LOCK > 0 the trial that wins is
#    partly chosen for its locked-half result, so that result stops being an
#    out, independent or test. Every trial carries locked_in_objective so a table
IH_W_FIT = float(os.environ.get("AGE_IH_W_FIT", 1.0))
IH_W_VAL = float(os.environ.get("AGE_IH_W_VAL", 1.0))
IH_W_LOCK = float(os.environ.get("AGE_IH_W_LOCK", 1.0))
IH_W_ALL = float(os.environ.get("AGE_IH_W_ALL", 1.0))
PUB_PEN = float(os.environ.get("AGE_PUB_PEN", 4.0))
# Softened 2026-10-08 from r>=0.75 / MAE<=8.0. The public fit is a MEANS -- it
# exists to show teh features were not fitted to the in-house cohort -- so a
R_MIN = float(os.environ.get("AGE_R_MIN", 0.72))
MAE_MAX = float(os.environ.get("AGE_MAE_MAX", 8.5))
# Weight on the cohort-parity term (see run_fold). 0 = off, the old
# behaviour. 1.0 makes a 0.1 parity gap cost as much as 0.1 of teh
GAP_W = float(os.environ.get("AGE_GAP_W", 1.0))
# Which in-house subset the parity term is measured on. "val" by default: it
# leaves the locked 41 out of the score so they remain a genuine hold-out.
GAP_ON = os.environ.get("AGE_GAP_ON", "val")
# Repeats per config. >1 is what makes a score a mean rather than one draw.
N_SEEDS = int(os.environ.get("AGE_N_SEEDS", 2))
# Cost guard. bag_passes x bag_k x ih_oversample / bag_subjects is unbounded in
# the search space; one draw (24 x 32 x 20 / 16) ran 92 min and held a worker
COST_MAX = float(os.environ.get("AGE_COST_MAX", 4000.0))

# ---------------------------------------------------------------------------
# LONG-RUN TRACE MODE. Off by default, so the sweep is untouched. retrain_top3.py
TRACE = None
# dict filled with per-subject predictions at the SAME epoch best_m is taken
# from, so a figure drawn from it shows exactly teh reported model.
PRED_SINK = None
# capture predictions on the PUBLIC TRAINING pool too (539 subjets). Costs one
# extra bag_predict per improving epoch, so it stays off unless asked for.
PRED_TRAIN = False
# None -> PRED_SINK captures teh objective's best epoch (the historical
# behaviour). An int -> capture exactly that epoch, whatever teh objective says.
PRED_AT_EPOCH = None
# When a list, LogEuclidHead.readout appends the 14x14 channel-interaction
# matrices of every batch it sees: G (the normalised Gram, i.e. the inner
SPD_SINK = None
# When a dict, run_fold leaves the trained model in it instead of freeing it.
MODEL_SINK = None
# "cosine" (over TRACE_MAX_EPOCHS), "plateau" (on in-house val MAE), or "none"
SCHED = "none"
# early stopping: which curve to watch, and how many epochs without improvement
#   "val"  = in-house validation MAE   -- legitimate, generalises
STOP_ON = "none"
PATIENCE = 5
TRACE_MAX_EPOCHS = 60
# print each traced epoch as it lands, so a long retrain is watchable
TRACE_VERBOSE = bool(int(os.environ.get("AGE_TRACE_VERBOSE", "1")))
# ---------------------------------------------------------------------------
IH_SELECT_N = int(os.environ.get("IH_SELECT_N", 40))
# Of the IH_SELECT_N "select" subjects, this many are used for TRAINING in the
# joint arm; the remainder are the in-house VALIDATION set that the objective
# (in-sample IHval +0.199 against held-out IHtest +0.162). The locked half remains
IH_FIT_N = int(os.environ.get("IH_FIT_N", 20))
# K-FOLD over the in-house SELECT subjects, for the objective's in-house term.
# REVERTED TO 1 for phase 3: one fixed 20 fit / 20 val split, as in phase 1.
IH_FOLDS = int(os.environ.get("IH_FOLDS", 1))
# In-house widnow spacing. The eyes-closed recordings are only ~25 s (median),
# giving 885 non-overlapping 512-sample segments across all 75 subjects. Finer
# windows share 511/512 samples -- so they do not improve the per-subject estimate,
IH_STRIDE = int(os.environ.get("IH_STRIDE", 64))
# Bag-mode compute budget, as a multiple of one window-mode epoch's windows.
BAG_BUDGET = float(os.environ.get("BAG_BUDGET", 2.0))
# Windows per subject at EVALUATION. Using every widnow was costing a full pass over
# ~190 windows per subject per epoch for 240 subjets across three cohorts; 64 is
EVAL_BAG_K = int(os.environ.get("EVAL_BAG_K", 64))
SPLIT_PATH = os.path.join(CKPT, "inhouse_split.json")
# In-house stimuli to EXCLUDE from the rest_ec pool, comma separated. The pool
# is four recordings and two of them (gyrus_left/right_closed_eyes, 31.5% of the
IH_DROP_STIM = tuple(x.strip() for x in
                     os.environ.get("AGE_IH_DROP_STIM", "").split(",") if x.strip())
# Restrict teh LOCKED (test) split to these stimuli only, leaving fit and val on
# the full pool. This is the asymmetric design: train on everything so the
IH_TEST_STIM = tuple(x.strip() for x in
                     os.environ.get("AGE_IH_TEST_STIM", "").split(",") if x.strip())


def get_storage():
    return optuna.storages.RDBStorage(
        url=STORAGE, engine_kwargs={"connect_args": {"timeout": 120}})


# ------------------------------------------------------------------ the model
class ChannelAttentionLayer(nn.Module):
    """One self-attention block over the 14 channel tokens, returning its weights."""

    def __init__(self, d_model, n_heads, ff_mult, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * ff_mult, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.n1(x)
        # average_attn_weights=False keeps every head separate, which the
        # connectivity analysis needs -- averaging destroys head specialisation.
        a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.n2(x)))
        return x, w


class ChannelAdapter(nn.Module):
    """Low-rank residual adapter on one channel's embedding."""

    def __init__(self, dim, rank=4):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)        # starts as the identity map

    def forward(self, z):
        return z + self.up(self.down(z))


def build_unc(emb_dims, hidden=64):
    """Per-widnow log-variance head, for precision-weighted aggregation."""
    d = sum(emb_dims)
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(),
                         nn.Linear(hidden, 1))


def build_adapters(emb_dims, rank):
    """None when rank is 0, so the frozen/full paths are untouched."""
    if not rank:
        return None
    return nn.ModuleList([ChannelAdapter(d, rank) for d in emb_dims])


def apply_adapters(zs, adapters):
    return [a(z) for a, z in zip(adapters, zs)] if adapters is not None else zs


class LogEuclidHead(nn.Module):
    """Log-Euclidean SPD pooling over the 14 channel embeddings."""

    def __init__(self, encoders, emb_dims, spd_dim=16, spd_eps=1e-3,
                 spd_norm="l2", spd_with_mean=False, hidden=128, dropout=0.2,
                 freeze_encoders=True, spd_recenter="none", spd_tri_sqrt2=True,
                 n_domains=2, recenter_momentum=0.05, spd_bn=False,
                 adapter_rank=0, agg="mean", **_):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.freeze = freeze_encoders
        if freeze_encoders:
            for e in self.encoders:
                for p in e.parameters():
                    p.requires_grad_(False)
        self.adapters = build_adapters(emb_dims, adapter_rank)
        self.unc = build_unc(emb_dims) if agg == "precision" else None
        self.proj = nn.ModuleList([nn.Linear(d, spd_dim) for d in emb_dims])
        self.eps = spd_eps
        self.spd_norm = spd_norm
        self.with_mean = spd_with_mean
        n = len(encoders)
        feat = n * (n + 1) // 2 + 1 + (spd_dim if spd_with_mean else 0)
        self.head = (nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, hidden),
                                   nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(hidden, 1))
                     if hidden else
                     nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, 1)))
        iu = torch.triu_indices(n, n)
        self.register_buffer("iu0", iu[0], persistent=False)
        self.register_buffer("iu1", iu[1], persistent=False)
        # Off-diagonal entries of a symmetric matrix appear twice, so vectorising
        # teh upper triangle without a sqrt(2) factor silently down-weights the
        self.tri_sqrt2 = spd_tri_sqrt2
        w = torch.where(iu[0] == iu[1], 1.0, float(np.sqrt(2.0)))
        self.register_buffer("tri_w", w, persistent=False)
        # RIEMANNIAN RE-CENTERING. The log map expands around the IDENTITY, which
        # assumes both cohorts' matrices are already centred there. They are not:
        self.recenter = spd_recenter
        self.rc_mom = recenter_momentum
        # RIEMANNIAN BATCH NORM (SPDNet-BN, Brooks et al.): whiten teh batch of
        # Gram matrices by their running mean so the batch is centred at the
        self.spd_bn = spd_bn
        if spd_bn:
            self.register_buffer("bn_mean", torch.eye(n).unsqueeze(0))
            self.register_buffer("bn_seen", torch.zeros(1))
            # The bias is parameterised DIRECTLY as a matrix L applied by
            # congruence, G <- L G L^T, which keeps G symmetric positive-definite
            self.bn_bias = nn.Parameter(torch.eye(n))
        if spd_recenter != "none":
            self.register_buffer("dom_mean",
                                 torch.eye(n).unsqueeze(0).repeat(n_domains, 1, 1))
            self.register_buffer("dom_seen", torch.zeros(n_domains))

    def channel_embeddings(self, x):
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            zs = [e(x[:, :, i].contiguous())[0] for i, e in enumerate(self.encoders)]
        return apply_adapters(zs, getattr(self, "adapters", None))

    def win_logvar(self, zs):
        """[N] log-variance per window, clamped so weights cannot explode."""
        return self.unc(torch.cat(zs, 1)).squeeze(-1).clamp(-6.0, 6.0)

    @staticmethod
    def _inv_sqrt(M):
        """M^(-1/2) for a symmetric positive-definite matrix, via its spectrum."""
        w, Q = torch.linalg.eigh(M)
        return (Q * w.clamp_min(1e-6).rsqrt().unsqueeze(-2)) @ Q.transpose(-1, -2)

    def _recenter(self, G, domain, n):
        """Whiten each sample by its own domain's running geometric-mean proxy."""
        out = G
        for dom in torch.unique(domain):
            m = domain == dom
            if m.sum() == 0:
                continue
            di = int(dom)
            if self.training:
                batch_mean = G[m].detach().mean(0)
                if self.dom_seen[di] == 0:
                    self.dom_mean[di] = batch_mean
                else:
                    self.dom_mean[di] = ((1 - self.rc_mom) * self.dom_mean[di]
                                         + self.rc_mom * batch_mean)
                self.dom_seen[di] += 1
            W = self._inv_sqrt(self.dom_mean[di].to(G.dtype)
                               + 1e-6 * torch.eye(n, device=G.device,
                                                  dtype=G.dtype))
            out = out.clone()
            out[m] = W @ G[m] @ W
        return out

    def readout(self, zs, return_attn=False, return_feat=False, domain=None):
        V = torch.stack([p(z) for p, z in zip(self.proj, zs)], dim=1).float()
        if self.spd_norm == "l2":
            V = V / V.norm(dim=2, keepdim=True).clamp_min(1e-8)
        elif self.spd_norm == "trace":
            # keep relative channel amplitudes, fix only teh overall scale
            V = V / V.pow(2).sum((1, 2), keepdim=True).sqrt().clamp_min(1e-8)
        with autocast(device_type="cuda", enabled=False):
            n = V.shape[1]
            G = V @ V.transpose(1, 2)
            G = 0.5 * (G + G.transpose(1, 2))         # exact symmetry for eigh
            # Normalise by the mean diagonal so `spd_eps` is a relative ridge
            # rather than an absolute one -- the raw Gram scale varies by orders of
            scale = G.diagonal(dim1=1, dim2=2).mean(1).clamp_min(1e-12)
            if self.spd_norm == "corr":
                # CORRELATION rather than covariance: divide out each channel's own
                # magnitude so the diagonal is exactly 1 and ONLY the inter-channel
                # amplitudes do not -- log(scale) is still appended, so teh overall
                dg = G.diagonal(dim1=1, dim2=2).clamp_min(1e-12).rsqrt()
                G = G * dg[:, :, None] * dg[:, None, :]
            else:
                G = G / scale[:, None, None]
            G = G + self.eps * torch.eye(n, device=G.device,
                                         dtype=G.dtype).unsqueeze(0)
            if self.recenter != "none" and domain is not None:
                G = self._recenter(G, domain, n)
            if self.spd_bn:
                if self.training:
                    bm = G.detach().mean(0, keepdim=True)
                    self.bn_mean.copy_(bm if self.bn_seen == 0 else
                                       (1 - self.rc_mom) * self.bn_mean
                                       + self.rc_mom * bm)
                    self.bn_seen += 1
                # the running mean is a detached statistic, so no gradient should
                # flow through its eigendecomposition
                with torch.no_grad():
                    W = self._inv_sqrt(self.bn_mean[0].to(G.dtype)
                                       + 1e-6 * torch.eye(n, device=G.device,
                                                          dtype=G.dtype))
                G = W @ G @ W
                L = self.bn_bias.to(G.dtype)
                G = L @ G @ L.transpose(-1, -2)
                G = 0.5 * (G + G.transpose(-1, -2))      # keep it exactly symmetric
                G = G + self.eps * torch.eye(n, device=G.device,
                                             dtype=G.dtype).unsqueeze(0)
            try:
                w, Q = torch.linalg.eigh(G)
            except Exception:
                # eigh can still fail to converge on a near-degenerate spectrum;
                # float64 on CPU is slower but does not fall over, and a trial must
                w, Q = torch.linalg.eigh(G.double().cpu())
                w, Q = w.to(G.device).float(), Q.to(G.device).float()
            L = torch.log(w.clamp_min(1e-6))
            Glog = (Q * L.unsqueeze(1)) @ Q.transpose(1, 2)
            if SPD_SINK is not None:
                SPD_SINK.append({"G": G.detach().cpu().numpy(),
                                 "Glog": Glog.detach().cpu().numpy()})
            tri = Glog[:, self.iu0, self.iu1]
            if self.tri_sqrt2:
                tri = tri * self.tri_w
            f = [tri, torch.log(scale)[:, None]]
            if self.with_mean:
                f.append(V.mean(1))
            feat = torch.cat(f, dim=1)
            if return_feat:
                return feat
            y = self.head(feat).squeeze(-1)
        return (y, []) if return_attn else y

    def forward(self, x, return_attn=False, domain=None):
        return self.readout(self.channel_embeddings(x), return_attn, domain=domain)


class ConcatHead(nn.Module):
    """Concatenate every channel's embedding, then a linear or MLP readout."""

    def __init__(self, encoders, emb_dims, hidden=128, dropout=0.2,
                 freeze_encoders=True, agg="mean", adapter_rank=0, **_):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.freeze = freeze_encoders
        if freeze_encoders:
            for e in self.encoders:
                for p in e.parameters():
                    p.requires_grad_(False)
        # Precision-weighted aggregation is a property of how a subject's windows
        # are combined, not of the readout, so every head gets it. Without this
        self.unc = build_unc(emb_dims) if agg == "precision" else None
        self.adapters = build_adapters(emb_dims, adapter_rank)
        d = sum(emb_dims)
        self.head = (nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, 1))
                     if hidden else nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)))

    def channel_embeddings(self, x):
        """x: [N, 512, 14] -> list of 14 tensors [N, d_i], one per channel."""
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            zs = [e(x[:, :, i].contiguous())[0] for i, e in enumerate(self.encoders)]
        return apply_adapters(zs, getattr(self, "adapters", None))

    def win_logvar(self, zs):
        """[N] log-variance per window, clamped so weights cannot explode."""
        return self.unc(torch.cat(zs, 1)).squeeze(-1).clamp(-6.0, 6.0)

    def readout(self, zs, return_attn=False, return_feat=False):
        """Per-channel embeddings -> age. Separated from the encoders so BAG
        training can average the embeddings over a subject's windows first."""
        f = torch.cat(zs, 1)
        if return_feat:
            return f
        y = self.head(f).squeeze(-1)
        return (y, []) if return_attn else y

    def forward(self, x, return_attn=False):
        return self.readout(self.channel_embeddings(x), return_attn)


class CrossChannelAgeModel(nn.Module):
    def __init__(self, encoders, emb_dims, d_model=64, n_heads=4, n_layers=2,
                 ff_mult=2, dropout=0.1, pool="mean", head_hidden=64,
                 freeze_encoders=True, agg="mean", adapter_rank=0, **_):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.freeze = freeze_encoders
        if freeze_encoders:
            for e in self.encoders:
                for p in e.parameters():
                    p.requires_grad_(False)
        self.unc = build_unc(emb_dims) if agg == "precision" else None
        self.adapters = build_adapters(emb_dims, adapter_rank)
        # each channel has its own embedding width, so each needs its own
        # projection into teh shared attention space
        self.proj = nn.ModuleList([nn.Linear(d, d_model) for d in emb_dims])
        self.chan_emb = nn.Parameter(torch.zeros(1, len(encoders), d_model))
        nn.init.trunc_normal_(self.chan_emb, std=0.02)
        self.pool = pool
        if pool == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)
        elif pool == "attn":
            self.pool_q = nn.Linear(d_model, 1)
        self.layers = nn.ModuleList([
            ChannelAttentionLayer(d_model, n_heads, ff_mult, dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        head = []
        if head_hidden:
            head += [nn.Linear(d_model, head_hidden), nn.GELU(), nn.Dropout(dropout),
                     nn.Linear(head_hidden, 1)]
        else:
            head += [nn.Linear(d_model, 1)]
        self.head = nn.Sequential(*head)

    def channel_embeddings(self, x):
        """x: [N, 512, 14] -> list of 14 tensors [N, d_i], one per channel."""
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            zs = [enc(x[:, :, i].contiguous())[0]
                  for i, enc in enumerate(self.encoders)]
        return apply_adapters(zs, getattr(self, "adapters", None))

    def win_logvar(self, zs):
        """[N] log-variance per widnow, clamped so weights cannot explode."""
        return self.unc(torch.cat(zs, 1)).squeeze(-1).clamp(-6.0, 6.0)

    def readout(self, toks, return_attn=False, return_emb=False,
                return_feat=False):
        """Per-channel embeddings -> attention over channels -> age."""
        t = torch.stack([p(z) for p, z in zip(self.proj, toks)], dim=1)
        t = t + self.chan_emb
        if self.pool == "cls":
            t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1)
        attn = []
        for layer in self.layers:
            t, w = layer(t)
            if return_attn:
                attn.append(w)
        t = self.norm(t)
        if self.pool == "cls":
            h = t[:, 0]
        elif self.pool == "attn":
            a = torch.softmax(self.pool_q(t), dim=1)
            h = (a * t).sum(1)
        else:
            h = t.mean(1)
        if return_feat:
            return h
        y = self.head(h).squeeze(-1)
        if return_emb:
            return h
        return (y, attn) if return_attn else y

    def forward(self, x, return_attn=False, return_emb=False):
        """x: [B, 512, 14]"""
        return self.readout(self.channel_embeddings(x), return_attn, return_emb)

    def embed(self, x):
        """Pooled channel-token representation the head reads, for kNN-MAE."""
        with torch.no_grad():
            return self.forward(x, return_emb=True)


_IH = {}


def inhouse_split(subjects, ages=None):
    """Deterministic, AGE-STRATIFIED four-way split, persisted to disk."""
    import json
    uniq = sorted(set(map(str, subjects)))
    if os.path.exists(SPLIT_PATH):
        try:
            d = json.load(open(SPLIT_PATH))
            # must match what inhouse_split WRITES: fold groups only at k >= 2,
            # otherwise the cache is rejected every time and each worker rebuilds
            need = ["select", "locked", "fit", "val"] + [
                f"fold{f}_{w}" for f in range(IH_FOLDS if IH_FOLDS > 1 else 0)
                for w in ("fit", "val")]
            if (sorted(d["select"] + d["locked"]) == uniq and d.get("stratified")
                    and all(k in d for k in need)
                    and d.get("ih_folds") == IH_FOLDS):
                return {k: set(d[k]) for k in need}
        except (ValueError, KeyError):
            pass
        print("  in-house split on disk unusable here -- rebuilding", flush=True)
    if ages is None:
        raise ValueError("stratified split needs ages")
    ages_of = {str(k): float(v) for k, v in ages.items()}
    # deal age-ordered neighbour pairs into select / locked
    order = sorted(uniq, key=lambda x: (ages_of[x], x))
    rng = np.random.default_rng(C.SEED)
    sel, lock = [], []
    for i in range(0, len(order), 2):
        blk = order[i:i + 2]
        if len(blk) == 1:
            (sel if len(sel) <= len(lock) else lock).append(blk[0]); continue
        if rng.random() < 0.5:
            blk = blk[::-1]
        sel.append(blk[0]); lock.append(blk[1])
    while len(sel) > IH_SELECT_N:
        lock.append(sel.pop(len(sel) // 2))
    while len(sel) < IH_SELECT_N and lock:
        sel.append(lock.pop(len(lock) // 2))
    # sub-split select into fit / val, taking every k-th up the age order so both
    # span the full in-house range instead of splitting it young/old
    sel_sorted = sorted(sel, key=lambda x: (ages_of[x], x))
    n_val = max(1, len(sel_sorted) - IH_FIT_N)
    step = len(sel_sorted) / n_val
    val_idx = {min(len(sel_sorted) - 1, int(round(j * step + step / 2)))
               for j in range(n_val)}
    fit = [sb for i, sb in enumerate(sel_sorted) if i not in val_idx]
    val = [sb for i, sb in enumerate(sel_sorted) if i in val_idx]
    groups = {"select": sorted(sel), "locked": sorted(lock),
              "fit": sorted(fit), "val": sorted(val)}
    # k age-stratified folds: walking the age-ordered select list and taking every
    # k-th subject keeps each fold spanning teh full in-house age range, which
    for f in range(IH_FOLDS if IH_FOLDS > 1 else 0):
        groups[f"fold{f}_val"] = sorted(sel_sorted[f::IH_FOLDS])
        groups[f"fold{f}_fit"] = sorted([x for i, x in enumerate(sel_sorted)
                                         if i % IH_FOLDS != f])
    os.makedirs(CKPT, exist_ok=True)
    tmp = SPLIT_PATH + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump({**groups, "seed": int(C.SEED), "stratified": True,
                   "n_total": len(uniq), "ih_folds": IH_FOLDS,
                   "note": "locked subjects must not be scored during any search"},
                  f, indent=2)
    os.replace(tmp, SPLIT_PATH)        # atomic, so concurrent workers cannot
                                       # read a half-written file
    for nm in (["fit", "val", "locked"]
               + [f"fold{f}_val" for f in range(IH_FOLDS if IH_FOLDS > 1 else 0)]):
        a = np.array([ages_of[x] for x in groups[nm]])
        print(f"    {nm:6s} n={len(a):3d} mean {a.mean():5.1f} SD {a.std():5.2f} "
              f"chanceMAE {np.abs(a - np.median(a)).mean():5.2f}", flush=True)
    return {k: set(v) for k, v in groups.items()}


def inhouse_data(which="select"):
    """In-house eyes-closed windows for one half of the cohort."""
    key = "d_" + which
    if key not in _IH:
        os.environ.setdefault("HARMONIZED_F", "1")
        sys.path.insert(0, os.path.dirname(HERE))
        from harmonized_io import build_windows, load_dataset
        from stimulus_groups import group as _g
        _, recs = load_dataset("ours")
        ec = [x for x in recs.stimulus.unique() if _g("ours", x) == "rest_ec"]
        w = build_windows("ours", recipe="prefiltered_uv", win=512,
                          stride=IH_STRIDE, stimuli=ec, verbose=False)
        if w is None:
            _IH["d_select"] = _IH["d_locked"] = None
            return None
        subj = np.asarray(w["subject"]).astype(str)
        ag = np.asarray(w["age"], float)
        # SPLIT FIRST, DROP SECOND. inhouse_split rebuilds and OVERWRITES its
        # cached JSON whenever the subject list it is handed differs from the
        groups = inhouse_split(subj, {s_: a_ for s_, a_ in zip(subj, ag)})
        stim_all = np.asarray(w["stimulus"]).astype(str)
        if IH_DROP_STIM:
            keep_stim = ~np.isin(stim_all, IH_DROP_STIM)
            print(f"  [stimuli] dropping {list(IH_DROP_STIM)}: "
                  f"{int((~keep_stim).sum())} of {len(keep_stim)} windows "
                  f"removed", flush=True)
        else:
            keep_stim = np.ones(len(stim_all), dtype=bool)
        if IH_TEST_STIM:
            print(f"  [stimuli] TEST split restricted to {list(IH_TEST_STIM)}; "
                  f"fit/val keep the full pool", flush=True)
        for nm, grp in groups.items():
            ks = keep_stim
            if IH_TEST_STIM and nm == "locked":
                ks = keep_stim & np.isin(stim_all, IH_TEST_STIM)
            k = np.array([x in grp for x in subj]) & ks
            _IH["d_" + nm] = (w["X"][k].astype(np.float32), ag[k], subj[k])
            print(f"  in-house {nm.upper():6s}: {int(k.sum()):6d} windows, "
                  f"{len(set(subj[k])):3d} subjects", flush=True)
    return _IH[key]


def load_encoders(device):
    encs, dims = [], []
    for ch in A.CHANNELS:
        ck = torch.load(os.path.join(CKPT, f"{ch}.pt"), map_location="cpu",
                        weights_only=False)
        cfg = ck["cfg"]
        e = A.ChannelEncoderV2(
            conv_channels=cfg["conv_channels"], kernel_size=cfg["kernel_size"],
            stride=cfg["stride"], padding=cfg["padding"], dilation=cfg["dilation"],
            gru_hidden=cfg["gru_hidden"], num_layers_gru=cfg["num_layers_gru"],
            embedding_dim=cfg["embedding_dim"], cnn_dropout=cfg["cnn_dropout"],
            rnn_dropout=cfg["rnn_dropout"], activation=cfg["activation"],
            **cfg["arch_extra"])
        e.load_state_dict(ck["state_dict"])
        encs.append(e.to(device))
        dims.append(cfg["embedding_dim"])
    return encs, dims


def _subject_frame(y, p, subj):
    d = pd.DataFrame({"s": subj, "t": y, "p": p})
    return d.groupby("s").agg(t=("t", "first"), p=("p", "mean"))


def cohort_transfer(y, p, subj, datasets):
    """Recalibrate on one cohort, score on teh other, both directions."""
    d = pd.DataFrame({"s": subj, "t": y, "p": p, "d": datasets})
    g = d.groupby("s").agg(t=("t", "first"), p=("p", "mean"), d=("d", "first"))
    out = []
    uniq = [u for u in g.d.unique() if (g.d == u).sum() >= 20]
    for a_ in uniq:
        for b_ in uniq:
            if a_ == b_:
                continue
            ga, gb = g[g.d == a_], g[g.d == b_]
            if ga.t.std() < 2 or gb.t.std() < 2 or ga.p.std() < 1e-6:
                continue
            k = np.polyfit(ga.p, ga.t, 1)              # fitted on A only
            pb = np.polyval(k, gb.p)
            if np.std(pb) > 1e-9:
                out.append(float(np.corrcoef(gb.t, pb)[0, 1]))
    return float(np.mean(out)) if out else 0.0


def _ih_stats(g):
    """Metrics from a per-subject (t, p) frame, with teh degenerate guard."""
    mse = float(((g.p - g.t) ** 2).mean())
    if len(g) < 10 or g.t.std() < 2 or g.p.std() < 1e-9:
        return {"r": 0.0, "mae": float("nan"), "slope": -1.0, "mse": mse}
    return {"r": float(np.corrcoef(g.t, g.p)[0, 1]),
            "mae": float(np.abs(g.p - g.t).mean()),
            "slope": float(np.polyfit(g.t, g.p - g.t, 1)[0]), "mse": mse}


def inhouse_metrics(model, Xih, yih, sih, device, bs=256, bag=False,
                    return_frame=False):
    """Predict age for teh in-house cohort, using the same protocol as training."""
    model.eval()
    if bag:
        smap = {}
        for i, sb in enumerate(sih):
            smap.setdefault(sb, []).append(i)
        smap = {k: np.asarray(v) for k, v in smap.items()}
        keys, p = bag_predict(model, Xih, smap, device)
        yih = np.array([yih[smap[k][0]] for k in keys])
        sih = keys
        g = _subject_frame(yih, p, sih)
        return (_ih_stats(g), g) if return_frame else _ih_stats(g)
    ps = []
    with torch.no_grad():
        for i in range(0, len(Xih), bs):
            xb = torch.as_tensor(np.asarray(Xih[i:i + bs]), device=device)
            with autocast(device_type="cuda"):
                ps.append(model(xb).float().cpu().numpy())
    p = np.concatenate(ps)
    g = _subject_frame(yih, p, sih)
    return (_ih_stats(g), g) if return_frame else _ih_stats(g)


def subject_metrics(y, p, subj):
    d = pd.DataFrame({"s": subj, "t": y, "p": p})
    g = d.groupby("s").agg(t=("t", "first"), p=("p", "mean"))
    return {"subject_mae": float(np.abs(g.p - g.t).mean()),
            "slope": float(np.polyfit(g.t, g.p - g.t, 1)[0]),
            "r": float(np.corrcoef(g.t, g.p)[0, 1]),
            "sd_ratio": float(g.p.std() / g.t.std())}


def suggest(trial):
    """Architecture and hyperparameters, searched together."""
    # HEAD FIXED to the log-Euclidean SPD representation; the architecture AROUND
    # it is what this sweep investigates. attention / concat remain in teh code and
    heads = os.environ.get("AGE_HEADS", "logeuclid,attention,concat").split(",")
    head = (heads[0] if len(heads) == 1
            else trial.suggest_categorical("head", heads))
    cfg = {
        "head": head,
        # TRAINING UNIT. "widnow": every window is one observation carrying its
        # subject's age -- 98,803 rows but only 539 independent labels, so weight
        "train_unit": "bag",
        # POOL MIX -- teh thing that was never tried. Every run so far trained on
        # PUBLIC subjects only and then asked the in-house cohort to cooperate. The
        # unlearnable" from "features fitted to gelled research EEG do not encode
        # inheriting features from a different recording regime. The LOCKED half
        "pool_mix": (os.environ["AGE_POOL_MIX"] if os.environ.get("AGE_POOL_MIX")
                     else trial.suggest_categorical("pool_mix",
                                                    ["joint", "public_only"])),
        # 40 in-house against 539 public subjects is 7% of each epoch at 1:1
        # sampling, probably too little to shift the encoders. Oversampling fixes
        "ih_oversample": trial.suggest_categorical("ih_oversample", [1, 3, 8]),
        # AGE-STRATIFIED SAMPLING. With only 40 in-house subjects, drawing bags at
        # random lets some age bins dominate an epoch by chance, and the age range
        "age_strat": trial.suggest_categorical("age_strat", [True, False]),
        # AUX: unsupervised feature alignment between the cohorts (deep CORAL),
        # using in-house SELECT windows as inputs only. Weaker than joint
        "aux": trial.suggest_categorical("aux", ["none", "coral"]),
        # fine-tuning won by a wide margin last time, on other encoders; retested
        "freeze_encoders": trial.suggest_categorical("freeze_encoders", [True, False]),
        # NARROWED from [0.0, 0.60] after 112 completed trials across the joint,
        # phase-2 and phase-3 studies. The low end was spending the budget on a
        "dropout": trial.suggest_float("dropout", 0.20, 0.60),
        "head_hidden": trial.suggest_categorical("head_hidden", [0, 64, 128, 256]),
        "lr_head": trial.suggest_float("lr_head", 1e-4, 3e-3, log=True),
        "lr_encoder": trial.suggest_float("lr_encoder", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 5e-5, 6e-4, log=True),
        "loss": trial.suggest_categorical("loss", ["smoothl1", "mse", "huber"]),
        # How many times each epoch walks the training subjects. Previously this
        # was derived to match window-mode's optimiser-step count, but train_unit
        "bag_passes": trial.suggest_categorical("bag_passes", [8, 16, 24]),
        "batch_size": 128,          # unused in bag mode, kept for the window path
    }
    if cfg["aux"] == "coral":
        cfg["aux_weight"] = trial.suggest_float("aux_weight", 1e-3, 3.0, log=True)
    # ---- PHASE 2 options ------------------------------------------------
    # Encoder adaptation: frozen / full fine-tune / low-rank adapter. Replaces the
    cfg["agg"] = trial.suggest_categorical("agg", ["mean", "precision"])
    # A Gaussian likelihood is only defined when the aggregation produces a
    # variance, so it is offered inside the precision arm alone.
    if cfg["agg"] == "precision":
        cfg["loss"] = trial.suggest_categorical(
            "loss_prec", ["smoothl1", "mse", "huber", "gaussian_nll"])
    cfg["enc_adapt"] = trial.suggest_categorical(
        "enc_adapt", ["frozen", "full", "adapter"])
    cfg["freeze_encoders"] = cfg["enc_adapt"] != "full"
    cfg["adapter_rank"] = (trial.suggest_categorical("adapter_rank", [2, 4, 8, 16])
                           if cfg["enc_adapt"] == "adapter" else 0)
    # Multi-domain: train across 5 harmonised public datasets instead of 2, so the
    # alignment terms have real domain diversity to generalise from rather than a
    cfg["domains"] = trial.suggest_categorical("domains", ["2ds", "5ds"])
    if head == "logeuclid":
        # teh SPD architecture itself
        # >= 14: with spd_dim below the channel count the 14x14 Gram is
        cfg["spd_dim"] = trial.suggest_categorical("spd_dim", [16, 24, 32, 64])
        cfg["spd_norm"] = trial.suggest_categorical(
            "spd_norm", ["none", "l2", "trace", "corr"])
        # Riemannian BN (batch-level, for optimisation) and per-domain re-centering
        # (for transfer) are indpendent and both searched.
        cfg["spd_bn"] = trial.suggest_categorical("spd_bn", [True, False])
        cfg["spd_recenter"] = trial.suggest_categorical(
            "spd_recenter", ["none", "cohort"])
        cfg["spd_eps"] = trial.suggest_float("spd_eps", 1e-4, 1e-1, log=True)
        cfg["spd_with_mean"] = trial.suggest_categorical(
            "spd_with_mean", [True, False])
    if cfg["train_unit"] == "bag":
        # windows averaged per subject per step, and subjects per optimiser step
        cfg["bag_k"] = trial.suggest_categorical("bag_k", [8, 16, 32])
        cfg["bag_subjects"] = trial.suggest_categorical("bag_subjects", [8, 16, 32])
    # COST GUARD -- see COST_MAX. Prune the pathological corner of the space
    # before it trains, rather than losing a worker to it for 90 min.
    cost = (cfg["bag_passes"] * cfg.get("bag_k", 8) * cfg["ih_oversample"]
            * (32.0 / max(1, cfg.get("bag_subjects", 32))))
    if cost > COST_MAX:
        raise optuna.TrialPruned(
            f"cost {cost:.0f} > {COST_MAX:.0f} (passes x bag_k x ovs / subjects)")
    if head == "attention":
        d_model = trial.suggest_categorical("d_model", [32, 64, 96, 128])
        cfg.update({
            "d_model": d_model,
            "n_heads": trial.suggest_categorical(
                "n_heads", [h for h in (2, 4) if d_model % h == 0]),
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "ff_mult": trial.suggest_categorical("ff_mult", [1, 2, 4, 8]),
            "pool": trial.suggest_categorical("pool", ["mean", "cls", "attn"]),
        })
    return cfg


# Recorded, not optimised: a fixed criterion so val_loss stays comparable across
# trials even though `loss` (which trains the model) is searched.
VAL_CRIT = nn.SmoothL1Loss()


def subject_index(idx, subjects):
    """idx -> {subject: array of its window positions within idx}."""
    out = {}
    for i, sb in zip(idx, subjects[idx]):
        out.setdefault(sb, []).append(i)
    return {k: np.asarray(v) for k, v in out.items()}


def bag_forward(model, X, win_idx, device, k=None, rng=None):
    """One prediction per SUBJECT, from the mean of its channel embeddings."""
    if k is not None and len(win_idx) > k:
        win_idx = rng.choice(win_idx, k, replace=False)
    xb = torch.as_tensor(np.asarray(X[np.sort(win_idx)]), device=device)
    zs = model.channel_embeddings(xb)
    return model.readout([z.mean(0, keepdim=True) for z in zs])


def bag_forward_batch(model, X, widx_list, device, k=None, rng=None,
                      return_var=False, domain=None):
    """One prediction per subject, for a WHOLE BATCH of subjects at once."""
    sel, counts = [], []
    for widx in widx_list:
        w = widx
        if k is not None and len(w) > k:
            w = rng.choice(w, k, replace=False)
        w = np.sort(w)
        sel.append(w); counts.append(len(w))
    flat = np.concatenate(sel)
    xb = torch.as_tensor(np.asarray(X[flat]), device=device)
    zs = model.channel_embeddings(xb)                 # list of [sum(counts), d_i]
    sls, off = [], 0
    for n in counts:
        sls.append(slice(off, off + n)); off += n
    unc = getattr(model, "unc", None)
    if unc is None:
        pooled = [torch.stack([z[sl].mean(0) for sl in sls]) for z in zs]
        var = None
    else:
        # precision-weighted mean, the static-state Kalman update. The posterior
        # variance 1/sum(precisions) is returned so a Gaussian likelihood can use it.
        lv = model.win_logvar(zs)
        w = torch.exp(-lv)
        pooled, var = [], []
        for sl in sls:
            ww = w[sl]
            var.append(1.0 / ww.sum().clamp_min(1e-6))
        for z in zs:
            pooled.append(torch.stack([
                (z[sl] * w[sl].unsqueeze(-1)).sum(0) / w[sl].sum().clamp_min(1e-6)
                for sl in sls]))
        var = torch.stack(var)
    y = model.readout(pooled, domain=domain) if domain is not None else model.readout(pooled)
    return (y, var) if return_var else y


def bag_feats_batch(model, X, widx_list, device, k, rng):
    """Head-input features for a batch of subjects, for the CORAL term."""
    sel = []
    for widx in widx_list:
        w = widx if len(widx) <= k else rng.choice(widx, k, replace=False)
        sel.append(np.sort(w))
    counts = [len(w) for w in sel]
    xb = torch.as_tensor(np.asarray(X[np.concatenate(sel)]), device=device)
    zs = model.channel_embeddings(xb)
    sls, off = [], 0
    for n in counts:
        sls.append(slice(off, off + n)); off += n
    pooled = [torch.stack([z[sl].mean(0) for sl in sls]) for z in zs]
    return model.readout(pooled, return_feat=True)


def bag_predict(model, X, subj_map, device, order=None):
    """Evaluation: all windows per subject, one prediction each."""
    model.eval()
    keys = list(subj_map) if order is None else order
    out = []
    CH = 8                      # subjets per batched call
    with torch.no_grad():
        for i in range(0, len(keys), CH):
            blk = keys[i:i + CH]
            with autocast(device_type="cuda"):
                p = bag_forward_batch(model, X, [subj_map[s_] for s_ in blk],
                                      device, k=EVAL_BAG_K,
                                      rng=np.random.default_rng(C.SEED))
            out.append(p.float().cpu().numpy())
    return np.asarray(keys), np.concatenate(out).astype(float)


def strat_order(subs, ages_of, rng, nbin=5):
    """Subject order that walks age quantiles evenly instead of at random."""
    if len(subs) == 0:
        return np.array([], dtype=object)
    a = np.array([ages_of[s] for s in subs], dtype=float)
    q = np.quantile(a, np.linspace(0, 1, nbin + 1)[1:-1])
    bins = [[] for _ in range(nbin)]
    for sb, av in zip(subs, a):
        bins[int(np.searchsorted(q, av, side="right"))].append(sb)
    for b in bins:
        rng.shuffle(b)
    out = []
    for i in range(max(len(b) for b in bins)):
        for b in bins:
            if i < len(b):
                out.append(b[i])
    return np.array(out, dtype=object)


def coral_loss(fa, fb):
    """Deep CORAL: squared Frobenius distance between feature second moments."""
    if fa.shape[0] < 2 or fb.shape[0] < 2:
        return fa.sum() * 0.0
    d = fa.shape[1]
    ca = torch.cov(fa.T)
    cb = torch.cov(fb.T)
    return ((ca - cb) ** 2).sum() / (4.0 * d * d) + ((fa.mean(0) - fb.mean(0)) ** 2).mean()


def chance_mse(y):
    """MSE of teh best constant predictor: the variance of the ages."""
    y = np.asarray(y, dtype=float)
    return float(((y - y.mean()) ** 2).mean())


def run_trial(cfg, POOLS, va, device, X_eval, ages_eval, subj_eval, ds_eval):
    """Train once per in-house fold and average the metrics."""
    outs = []
    for f in range(IH_FOLDS):
        # k = 1 is the fixed split, not fold 0: see inhouse_split
        fk, vk = (("fit", "val") if IH_FOLDS == 1
                  else (f"fold{f}_fit", f"fold{f}_val"))
        for si in range(N_SEEDS):
            m = run_fold(cfg, POOLS, va, device, X_eval, ages_eval, subj_eval,
                         ds_eval, fk, vk, seed=C.SEED + 7919 * si)
            if m is None:
                return None
            outs.append(m)
    if len(outs) == 1:
        return outs[0]
    keys = [k for k, v in outs[0].items() if isinstance(v, (int, float))
            and not isinstance(v, bool)]
    avg = {k: float(np.mean([o[k] for o in outs if np.isfinite(o.get(k, np.nan))]
                            or [np.nan])) for k in keys}
    # spread across folds, so an unstable config is visible rather than hidden
    avg["fold_sd_inhouse_r"] = float(np.std([o["inhouse_r"] for o in outs]))
    avg["fold_sd_test_r"] = float(np.std([o["test_r"] for o in outs]))
    avg["n_folds"] = len(outs)
    avg["n_seeds"] = N_SEEDS
    # teh number that says whether this config's score means anything: the sd
    # of the mean across repeats. Compare it to the gap bewteen leaderboard
    avg["sem_test_r"] = float(np.std([o["test_r"] for o in outs], ddof=1)
                              / np.sqrt(len(outs))) if len(outs) > 1 else float("nan")
    avg["sem_inhouse_r"] = float(np.std([o["inhouse_r"] for o in outs], ddof=1)
                                 / np.sqrt(len(outs))) if len(outs) > 1 else float("nan")
    for k, v in outs[0].items():
        if k not in avg:
            avg[k] = v
    avg["feasible"] = bool(avg["pub_hinge"] == 0.0)
    return avg


def _build_sched(opt, n_epochs):
    """LR schedule for the long retrains. None in sweep mode (SCHED='none')."""
    if SCHED == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    if SCHED == "plateau":
        # stepped on in-house VAL mae, never on the hold-out
        return optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                    patience=3)
    return None


def run_fold(cfg, POOLS, va, device, X_eval, ages_eval, subj_eval, ds_eval,
             fit_key="fit", val_key="val", seed=None):
    # SEED IS AN ARGUMENT, not C.SEED. It used to be the constant C.SEED for
    # every trial, which made a config's score ONE draw from a distribution
    seed = C.SEED if seed is None else int(seed)
    torch.manual_seed(seed)
    pk = cfg.get("domains", "2ds")
    if pk not in POOLS:
        # The search space offers "5ds", which has never been a POOL_FILES key,
        # so this branch has always been taken and every "5ds" trial trained on
        print(f"    [pool] requested {pk!r} is not available -> using '2ds'",
              flush=True)
        pk = "2ds"
    cfg["domains_used"] = pk
    P = POOLS[pk]
    # Domain ids for the re-centering buffer: every public dataset gets its own,
    # in-house is teh last. Re-centering is per-DOMAIN, so with a multi-domain pool
    dom_names = sorted(set(P["datasets"].tolist()))
    DOM = {d: i for i, d in enumerate(dom_names)}
    DOM["__inhouse__"] = len(DOM)
    cfg["n_domains"] = len(DOM)
    X, ages, subjects, datasets, tr = (P["X"], P["ages"], P["subjects"],
                                       P["datasets"], P["tr"])
    encs, dims = load_encoders(device)
    if cfg["head"] == "logeuclid":
        model = LogEuclidHead(encs, dims, spd_dim=cfg["spd_dim"],
                              spd_eps=cfg["spd_eps"], spd_norm=cfg["spd_norm"],
                              spd_with_mean=cfg["spd_with_mean"],
                              hidden=cfg["head_hidden"], dropout=cfg["dropout"],
                              freeze_encoders=cfg["freeze_encoders"],
                              spd_bn=cfg.get("spd_bn", False),
                              spd_recenter=cfg.get("spd_recenter", "none"),
                              adapter_rank=cfg.get("adapter_rank", 0),
                              n_domains=cfg["n_domains"],
                              agg=cfg.get("agg", "mean")).to(device)
    elif cfg["head"] == "concat":
        model = ConcatHead(encs, dims, hidden=cfg["head_hidden"],
                           dropout=cfg["dropout"],
                           freeze_encoders=cfg["freeze_encoders"],
                           agg=cfg.get("agg", "mean"),
                           adapter_rank=cfg.get("adapter_rank", 0)).to(device)
    else:
        model = CrossChannelAgeModel(
            encs, dims, d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"], ff_mult=cfg["ff_mult"], dropout=cfg["dropout"],
            pool=cfg["pool"], head_hidden=cfg["head_hidden"],
            freeze_encoders=cfg["freeze_encoders"],
            agg=cfg.get("agg", "mean"),
            adapter_rank=cfg.get("adapter_rank", 0)).to(device)

    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("encoders.") and p.requires_grad]
    groups = [{"params": head_params, "lr": cfg["lr_head"]}]
    if not cfg["freeze_encoders"]:
        groups.append({"params": [p for p in model.encoders.parameters()],
                       "lr": cfg["lr_encoder"]})
    opt = optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    # A Gaussian likelihood needs the posterior variance, wich only exists when
    # the model actually carries an uncertainty head. Tied to the model rather
    nll = cfg["loss"] == "gaussian_nll" and getattr(model, "unc", None) is not None
    crit = {"smoothl1": nn.SmoothL1Loss(), "mse": nn.MSELoss(),
            "huber": nn.HuberLoss(delta=5.0),
            "gaussian_nll": nn.MSELoss()}[cfg["loss"]]
    scaler = GradScaler(device="cuda")
    bs = cfg["batch_size"]
    y_all = torch.as_tensor(ages, dtype=torch.float32)

    va_np = va.numpy()
    bag = cfg["train_unit"] == "bag"
    brng = np.random.default_rng(C.SEED)
    joint = cfg.get("pool_mix", "public_only") == "joint"
    if bag:
        tr_map = subject_index(tr.numpy(), subjects)
        va_map = subject_index(va_np, subj_eval)
        tr_subs = np.array(list(tr_map))
        pub_ages_of = {sb: float(ages[v[0]]) for sb, v in tr_map.items()}
        # in-house SELECT subjets, as a second labelled training source
        ih_map, ih_ages_of, ih_subs = {}, {}, np.array([], dtype=object)
        ihd0 = inhouse_data(fit_key)
        if ihd0 is not None and (joint or cfg["aux"] == "coral"):
            sv_ih = ihd0[2]
            for i, sb in enumerate(sv_ih):
                ih_map.setdefault(sb, []).append(i)
            ih_map = {k: np.asarray(v) for k, v in ih_map.items()}
            ih_ages_of = {sb: float(ihd0[1][v[0]]) for sb, v in ih_map.items()}
            ih_subs = np.array(list(ih_map), dtype=object)
        # EQUALISE OPTIMISER STEPS. One pass over 539 subjects at bag_subjects per
        # step gives ~34 steps, against ~770 for window mode at the same epoch
        bag_passes = cfg["bag_passes"]
        bag_steps = max(1, len(tr_subs) // cfg["bag_subjects"])
        cfg["bag_steps_per_epoch"] = bag_steps * bag_passes
        # windows embedded per epoch, teh honest cost measure
        cfg["windows_per_epoch"] = int(bag_passes * len(tr_subs) * cfg["bag_k"]
                                       * (1 + (cfg["ih_oversample"]
                                               if joint else 0) * 20
                                          / max(len(tr_subs), 1)))
    # chance MAE for each cohort, at SUBJECT level: the error of predicting that
    # cohort's median age for everyone. Computed once per trial.
    pub_ages = _subject_frame(ages_eval[va_np], ages_eval[va_np],
                              subj_eval[va_np]).t.to_numpy()
    pub_floor = float(np.abs(pub_ages - np.median(pub_ages)).mean())
    ihd = inhouse_data(val_key)          # validation half
    ihd_test = inhouse_data("locked")    # hold-out half -- now ALSO scored
    ihd_fit = inhouse_data(fit_key)      # training half, in-sample when joint

    def _floor(d):
        """Chance MAE: the error of predicting this subset's median age."""
        if not d:
            return 1.0
        a_ = _subject_frame(d[1], d[1], d[2]).t.to_numpy()
        return float(np.abs(a_ - np.median(a_)).mean()) or 1.0

    ih_floor = _floor(ihd)
    floors = {"fit": _floor(ihd_fit), "val": ih_floor,
              "lock": _floor(ihd_test)}
    # teh pooled floor is over the union of the three, so "all" is not just the
    # mean of the parts -- a wider age spread raises the bar it is measured against
    _all_ages = np.concatenate([_subject_frame(d[1], d[1], d[2]).t.to_numpy()
                                for d in (ihd_fit, ihd, ihd_test) if d]) \
        if any((ihd_fit, ihd, ihd_test)) else np.array([0.0])
    floors["all"] = float(np.abs(_all_ages - np.median(_all_ages)).mean()) or 1.0

    best = np.inf
    best_m = None
    n_epochs = TRACE_MAX_EPOCHS if TRACE is not None else EPOCHS
    sched = _build_sched(opt, n_epochs)
    stop_best, stop_bad = np.inf, 0
    for ep in range(n_epochs):
        model.train()
        # per-epoch training loss, for teh loss curves. Accumulated as the plain
        # mean over optimiser steps, so a bag epoch and a window epoch are both
        _ep_loss, _ep_main, _ep_aux = [], [], []
        if not bag:
            perm = tr[torch.randperm(len(tr))]
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs].numpy()
                xb = torch.as_tensor(np.asarray(X[b]), device=device)
                yb = y_all[b].to(device)
                opt.zero_grad()
                with autocast(device_type="cuda"):
                    pred = model(xb)
                loss = crit(pred.float(), yb)
                _ep_loss.append(float(loss.detach()))
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            # one observation per SUBJECT: embed bag_k of its windows, average the
            # per-channel embeddings, one prediction, one loss term
            nsub = cfg["bag_subjects"]
            strat = cfg.get("age_strat", False)

            def order(subs, ages_of):
                if len(subs) == 0:
                    return np.array([], dtype=object)
                return (strat_order(subs, ages_of, brng) if strat
                        else brng.permutation(subs))

            seq = []
            for _ in range(bag_passes):
                seq.append([("pub", sb) for sb in order(tr_subs, pub_ages_of)])
                if joint and len(ih_subs):
                    for _r in range(cfg["ih_oversample"]):
                        seq.append([("ih", sb) for sb in order(ih_subs, ih_ages_of)])
            flat = [x for blk in seq for x in blk]
            brng.shuffle(flat)        # interleave cohorts within teh epoch
            for i in range(0, len(flat), nsub):
                batch = flat[i:i + nsub]
                opt.zero_grad()
                _aux_val = 0.0      # this step's CORAL contribution, if any
                targs = []
                with autocast(device_type="cuda"):
                    # group by source so each cohort is one batched call
                    pub_b = [sb for src, sb in batch if src == "pub"]
                    ih_b = [sb for src, sb in batch if src == "ih"]
                    parts, vars_ = [], []
                    if pub_b:
                        o = bag_forward_batch(
                            model, X, [tr_map[sb] for sb in pub_b], device,
                            k=cfg["bag_k"], rng=brng, return_var=nll)
                        if nll:
                            parts.append(o[0]); vars_.append(o[1])
                        else:
                            parts.append(o)
                        targs += [ages[tr_map[sb][0]] for sb in pub_b]
                    if ih_b:
                        o = bag_forward_batch(
                            model, ihd0[0], [ih_map[sb] for sb in ih_b], device,
                            k=cfg["bag_k"], rng=brng, return_var=nll)
                        if nll:
                            parts.append(o[0]); vars_.append(o[1])
                        else:
                            parts.append(o)
                        targs += [ih_ages_of[sb] for sb in ih_b]
                    pv = torch.cat(parts)
                    tt = torch.as_tensor(np.asarray(targs, dtype=np.float32),
                                         device=device)
                    if nll:
                        # Gaussian negative log-likelihood with the aggregated
                        # posterior variance: teh model is rewarded for knowing
                        v = torch.cat(vars_).float().clamp_min(0.25)
                        loss = (0.5 * ((pv.float() - tt) ** 2 / v
                                       + torch.log(v))).mean()
                    else:
                        loss = crit(pv.float(), tt)
                    if cfg["aux"] == "coral" and len(ih_subs):
                        pb = pub_b[:8] if pub_b else []
                        ib = list(brng.choice(ih_subs, min(8, len(ih_subs)),
                                              replace=False))
                        if pb and ib:
                            fa = bag_feats_batch(model, X,
                                                 [tr_map[sb] for sb in pb], device,
                                                 cfg["bag_k"], brng)
                            fb = bag_feats_batch(model, ihd0[0],
                                                 [ih_map[sb] for sb in ib], device,
                                                 cfg["bag_k"], brng)
                            _aux = cfg["aux_weight"] * coral_loss(
                                fa.float(), fb.float())
                            _aux_val = float(_aux.detach())
                            _ep_aux.append(_aux_val)
                            loss = loss + _aux
                _tot = float(loss.detach())
                _ep_loss.append(_tot)
                _ep_main.append(_tot - _aux_val)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

        model.eval()
        if bag:
            # evaluated the way it was trained: a subject's windows averaged in
            # embedding space, one prediction per subject. Always from the shared
            sv, p = bag_predict(model, X_eval, va_map, device)
            yv = np.array([ages_eval[va_map[k][0]] for k in sv])
        else:
            preds = []
            with torch.no_grad():
                for i in range(0, len(va), 512):
                    b = va[i:i + 512].numpy()
                    xb = torch.as_tensor(np.asarray(X_eval[b]), device=device)
                    with autocast(device_type="cuda"):
                        preds.append(model(xb).float().cpu().numpy())
            p = np.concatenate(preds)
            yv, sv = ages_eval[va_np], subj_eval[va_np]
        m = subject_metrics(yv, p, sv)
        m["epoch"] = ep + 1
        dsv = (np.array([ds_eval[va_map[k][0]] for k in sv]) if bag
               else ds_eval[va_np])
        m["cross_cohort_r"] = cohort_transfer(yv, p, sv, dsv)
        DEGEN = {"r": 0.0, "mae": float("nan"), "slope": -1.0,
                 "mse": float("nan")}

        def _ih(d):
            """Metrics plus the per-subject frame, so subsets can be pooled."""
            if not d:
                return DEGEN, None
            return inhouse_metrics(model, d[0], d[1], d[2], device, bag=bag,
                                   return_frame=True)

        ihm, g_val = _ih(ihd)
        tst, g_lock = _ih(ihd_test)      # hold-out: now scored as well as reported
        fitm, g_fit = _ih(ihd_fit)       # in-sample in the joint arm
        # "all": one pooled fit over every in-house subject at once. Not teh mean
        # of teh three -- pooling re-fits r and the slope across the full age
        frames = [x for x in (g_fit, g_val, g_lock) if x is not None]
        allm = _ih_stats(pd.concat(frames)) if frames else DEGEN
        m["test_r"], m["test_mae"] = tst["r"], tst["mae"]
        m["test_slope"], m["test_mse"] = tst["slope"], tst["mse"]
        m["inhouse_r"], m["inhouse_mae"] = ihm["r"], ihm["mae"]
        m["inhouse_slope"] = ihm["slope"]
        for tag, mm in (("fit", fitm), ("all", allm)):
            m[f"ih{tag}_r"], m[f"ih{tag}_mae"] = mm["r"], mm["mae"]
            m[f"ih{tag}_slope"] = mm["slope"]
        m["subject_mae_norm"] = m["subject_mae"] / pub_floor
        m["val_loss"] = float(VAL_CRIT(torch.as_tensor(p, dtype=torch.float32),
                                       torch.as_tensor(yv, dtype=torch.float32)))
        gp = _subject_frame(yv, p, sv)
        m["pub_mse"] = float(((gp.p - gp.t) ** 2).mean())
        m["inhouse_mse"] = ihm.get("mse", float("nan"))

        # --- teh objective: MAE/chance - r + |slope|, on each cohort --------
        m["pub_term"] = (m["subject_mae"] / pub_floor - m["r"] + abs(m["slope"]))
        ih_mae = ihm.get("mae", float("nan"))
        ih_r = ihm.get("r", 0.0)
        ih_slope = ihm.get("slope", -1.0)
        m["inhouse_mae_norm"] = ih_mae / ih_floor

        def _term(mm, floor):
            """MAE/chance - r + |slope|; 3.0 for a degenerate fit, as before."""
            t = (mm.get("mae", float("nan")) / floor - mm.get("r", 0.0)
                 + abs(mm.get("slope", -1.0)))
            return t if np.isfinite(t) else 3.0

        # The in-house term is the weighted mean over the ACTIVE subsets, so the
        # score keeps its old scale whatever subset mix is switched on.
        parts = [("fit", fitm, IH_W_FIT), ("val", ihm, IH_W_VAL),
                 ("lock", tst, IH_W_LOCK), ("all", allm, IH_W_ALL)]
        wsum = sum(w for _, _, w in parts if w > 0) or 1.0
        for tag, mm, w in parts:
            m[f"ihterm_{tag}"] = _term(mm, floors[tag])
        m["inhouse_term"] = sum(m[f"ihterm_{tag}"] * w
                                for tag, _, w in parts if w > 0) / wsum
        if not np.isfinite(m["inhouse_term"]):
            m["inhouse_term"] = 3.0          # degenerate in-house fit
        # so a table built from this study can never silently present teh
        # hold-out half as indpendent
        m["locked_in_objective"] = bool(IH_W_LOCK > 0 or IH_W_ALL > 0)
        m["pub_hinge"] = (max(0.0, (R_MIN - m["r"]) / R_MIN)
                          + max(0.0, m["subject_mae"] / MAE_MAX - 1.0))
        m["feasible"] = bool(m["pub_hinge"] == 0.0)
        # COHORT-PARITY TERM. The two cohorts cannot be equalised on r: public
        # ages span sd 16.1 yr against the in-house 11.4, so the same absolute
        m["pub_mae_norm"] = m["subject_mae"] / pub_floor
        m["lock_mae_norm"] = tst.get("mae", float("nan")) / floors["lock"]
        # Parity is measured on the subset named by GAP_ON. "val" keeps the
        # locked 41 subjects OUT of teh objective entirely, so the hold-out
        _gsrc = {"val": (ihm, floors["val"]), "lock": (tst, floors["lock"]),
                 "all": (allm, floors["all"])}[GAP_ON]
        m["gap_mae_norm"] = _gsrc[0].get("mae", float("nan")) / _gsrc[1]
        gap = abs(m["gap_mae_norm"] - m["pub_mae_norm"])
        m["cohort_gap"] = gap if np.isfinite(gap) else 1.0
        m["gap_measured_on"] = GAP_ON
        m["score"] = (IH_W * m["inhouse_term"] + PUB_PEN * m["pub_hinge"]
                      + GAP_W * m["cohort_gap"])
        _improved = m["score"] < best
        if _improved:
            best, best_m = m["score"], dict(m)
        # WHICH EPOCH'S PREDICTIONS GET CAPTURED. By default the sink follows
        # the objective's best epoch. For t25 that is epoch 2 -- a model with
        if (PRED_SINK is not None and
                ((ep + 1) == PRED_AT_EPOCH if PRED_AT_EPOCH else _improved)):
            if True:
                pub = pd.DataFrame({"t": yv, "p": np.asarray(p).ravel(),
                                    "s": sv, "dataset": dsv})
                PRED_SINK.clear()
                PRED_SINK.update({"public": pub.groupby("s", as_index=False)
                                  .agg(t=("t", "first"), p=("p", "mean"),
                                       dataset=("dataset", "first")),
                                  "ih_fit": g_fit, "ih_val": g_val,
                                  "ih_locked": g_lock, "epoch": ep + 1})
                # The PUBLIC TRAINING pool, predicted the same way the public
                # validation pool is. Only captured when a sink is attached, and
                if bag and PRED_TRAIN:
                    ts, tp = bag_predict(model, X, tr_map, device)
                    PRED_SINK["public_train"] = pd.DataFrame({
                        "s": ts,
                        "t": [float(ages[tr_map[k][0]]) for k in ts],
                        "p": np.asarray(tp).ravel(),
                        "dataset": [str(datasets[tr_map[k][0]]) for k in ts]})

        # ---- long-run trace: record every epoch, schedule, maybe stop -------
        if TRACE is not None:
            TRACE.append({
                "epoch": ep + 1,
                "lr_head": opt.param_groups[0]["lr"],
                "train_loss": float(np.mean(_ep_loss)) if _ep_loss else float("nan"),
                "train_loss_main": float(np.mean(_ep_main)) if _ep_main else float("nan"),
                "train_loss_aux": float(np.mean(_ep_aux)) if _ep_aux else 0.0,
                "n_steps": len(_ep_loss),
                "pub_r": m["r"], "pub_mae": m["subject_mae"],
                "pub_slope": m["slope"],
                "ihval_r": ihm.get("r"), "ihval_mae": ihm.get("mae"),
                "ihval_slope": ihm.get("slope"),
                "ihtest_r": tst.get("r"), "ihtest_mae": tst.get("mae"),
                "ihtest_slope": tst.get("slope"),
                "ihfit_r": fitm.get("r"), "ihfit_mae": fitm.get("mae"),
                "score": m["score"], "feasible": m["feasible"],
                "cohort_gap": m["cohort_gap"],
                # chance floors (MAD-from-median MAE) carried along so a curve
                # can be drawn against the right baseline per cohort
                "floor_pub": pub_floor, "floor_ihfit": floors["fit"],
                "floor_ihval": floors["val"], "floor_ihtest": floors["lock"],
            })
            if TRACE_VERBOSE:
                _t = TRACE[-1]
                print(f"  ep {_t['epoch']:3d} loss {_t['train_loss']:.4f} "
                      f"| PUB r {_t['pub_r']:+.3f} MAE {_t['pub_mae']:5.2f} "
                      f"| IHfit {_t['ihfit_mae']:5.2f} "
                      f"| IHval r {_t['ihval_r']:+.3f} MAE {_t['ihval_mae']:5.2f} "
                      f"| IHtest r {_t['ihtest_r']:+.3f} "
                      f"MAE {_t['ihtest_mae']:5.2f}", flush=True)
            if sched is not None:
                if SCHED == "plateau":
                    sched.step(ihm.get("mae", np.inf))   # val only, never lock
                else:
                    sched.step()
            if STOP_ON != "none":
                watch = (ihm if STOP_ON == "val" else tst).get("mae", np.inf)
                if not np.isfinite(watch):
                    watch = np.inf
                if watch < stop_best - 1e-4:
                    stop_best, stop_bad = watch, 0
                else:
                    stop_bad += 1
                    if stop_bad >= PATIENCE:
                        TRACE[-1]["stopped_here"] = True
                        break
    if MODEL_SINK is not None:
        # keep a reference so teh trained model survives `del model` -- the
        # channel-interaction figure re-runs inference through it
        MODEL_SINK["model"] = model
        MODEL_SINK["cfg"] = cfg
    else:
        del model, encs
        torch.cuda.empty_cache()
    return best_m


def prepare():
    """Pools, shared public validation split and device."""
    device = torch.device("cuda")
    # Several training pools, one evaluation set. The PUBLIC validation subjects
    # always come from the 2-dataset pool, whichever pool supplied the training
    pools = {}
    for tag, fname in POOL_FILES.items():
        xp = os.path.join(HERE, fname + "_X.npy")
        mp = os.path.join(HERE, fname + "_meta.pkl")
        if not (os.path.exists(xp) and os.path.exists(mp)):
            print(f"  pool {tag!r} not built, skipping", flush=True)
            continue
        import pickle as _pk
        pools[tag] = (np.load(xp, mmap_mode="r"), _pk.load(open(mp, "rb")))
    if "2ds" not in pools:
        raise SystemExit("the 2ds pool is required")

    X, meta = pools["2ds"]
    ages, subjects, datasets = meta["age"], meta["subject"], meta["dataset"]
    subs = np.array(sorted(set(subjects)))
    rng = np.random.default_rng(C.SEED)
    va_subs = set(rng.choice(subs, size=int(len(subs) * S.VAL_FRACTION), replace=False))
    is_va = np.array([s in va_subs for s in subjects])
    va = torch.as_tensor(np.flatnonzero(is_va)[::WIN_STRIDE].copy())

    # one training index per pool, always excluding the shared validation subjects
    POOLS = {}
    for tag, (Xp, mp) in pools.items():
        sp = np.asarray(mp["subject"])
        keep = np.flatnonzero(~np.isin(sp, list(va_subs)))[::WIN_STRIDE]
        POOLS[tag] = {"X": Xp, "ages": np.asarray(mp["age"], float),
                      "subjects": sp, "datasets": np.asarray(mp["dataset"]),
                      "tr": torch.as_tensor(keep.copy())}
        nd = len(set(mp["dataset"]))
        print(f"  pool {tag:7s} {len(keep):,} train windows | "
              f"{len(set(sp[keep]))} subjects | {nd} datasets", flush=True)
    print(f"  shared public val: {len(va):,} windows | {len(va_subs)} subjects",
          flush=True)

    return POOLS, va, device, X, ages, subjects, datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--warm-from", default=None,
                    help="study to seed trial 0 from (previous, wider search)")
    args = ap.parse_args()

    if args.init:
        st = optuna.create_study(direction="minimize", study_name=STUDY,
                                 storage=get_storage(), load_if_exists=True)
        # Warm start: carry teh previous study's winner across as trial 0 so the
        # narrowed search cannot start worse than the one it replaces. Only the
        if args.warm_from and not st.trials:
            try:
                old = optuna.load_study(study_name=args.warm_from, storage=get_storage())
                done = [t for t in old.trials
                        if t.state.name == "COMPLETE" and t.value is not None]
                if done:
                    b = min(done, key=lambda t: t.value)
                    free = {"dropout", "head_hidden", "lr_head", "lr_encoder",
                            "weight_decay", "loss"}
                    seed = {k: v for k, v in b.params.items() if k in free}
                    st.enqueue_trial(seed)
                    print(f"warm-started from {args.warm_from} trial #{b.number} "
                          f"(MAE {b.value:.3f}): {seed}")
            except KeyError:
                print(f"warm-start study {args.warm_from!r} not found, skipping")
        # Build the in-house split HERE, once, in the parent process, before any
        # worker exists. The launcher runs --init in the foreground, so by the time
        print("in-house split:")
        inhouse_data("val")
        print(f"study {STUDY!r} initialised ({len(st.trials)} existing trials)"); return

    POOLS, va, device, X, ages, subjects, datasets = prepare()

    study = optuna.create_study(direction="minimize", study_name=STUDY,
                                storage=get_storage(), load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(
                                    # Per-worker seed. With one shared seed every
                                    # worker walks the same TPE path and, because
                                    seed=C.SEED + 1000 * (1 + int(os.environ.get(
                                        "AGE_WORKER", os.getpid()) ) % 97),
                                    n_startup_trials=12))

    def objective(trial):
        cfg = suggest(trial)
        try:
            m = run_trial(cfg, POOLS, va, device, X, ages, subjects, datasets)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); raise optuna.TrialPruned()
        if m is None or not np.isfinite(m["subject_mae"]):
            raise optuna.TrialPruned()
        for k, v in m.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("pool_mix_used", cfg["pool_mix"])
        trial.set_user_attr("agg_used", cfg.get("agg", "mean"))
        trial.set_user_attr("domains_used", cfg.get("domains_used", "?"))
        for k in ("bag_passes", "bag_steps_per_epoch", "windows_per_epoch"):
            if k in cfg:
                trial.set_user_attr(k, cfg[k])
        trial.set_user_attr("beats_floor", bool(m["subject_mae"] < FLOOR))
        print(f"trial {trial.number:3d} {cfg['head']:9s} "
              f"{'frozen' if cfg['freeze_encoders'] else 'finetune':8s} "
              f"{'joint' if cfg.get('pool_mix')=='joint' else 'pubonly':7s} "
              f"{('x%d' % cfg.get('ih_oversample',1)):4s} "
              f"{'strat' if cfg.get('age_strat') else 'rand ':5s} "
              f"{cfg.get('aux','none'):5s} "
              f"score {m['score']:.4f} = ih {m['inhouse_term']:.4f} "
              f"[fit {m.get('ihterm_fit', float('nan')):.2f} "
              f"val {m.get('ihterm_val', float('nan')):.2f} "
              f"lock {m.get('ihterm_lock', float('nan')):.2f} "
              f"all {m.get('ihterm_all', float('nan')):.2f}] + "
              f"pen {PUB_PEN * m['pub_hinge']:.3f}{'*' if m['feasible'] else ''} | PUB r {m['r']:.3f} "
              f"MAE {m['subject_mae']:5.2f} slope {m['slope']:+.3f} "
              f"| IHval r {m['inhouse_r']:+.3f} MAE {m['inhouse_mae']:6.2f} "
              f"sl {m['inhouse_slope']:+.3f} "
              f"| IHtest r {m['test_r']:+.3f} MAE {m['test_mae']:6.2f} "
              f"sl {m['test_slope']:+.3f} | xcoh {m['cross_cohort_r']:+.3f}",
              flush=True)
        return m["score"]

    # A trial that dies on a numerical edge case is recorded as FAILED and the
    # worker carries on; previously the exception propagated out of optimize()
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True,
                   catch=(RuntimeError, TypeError, ValueError,
                          torch.linalg.LinAlgError))
    b = study.best_trial
    u = b.user_attrs
    print(f"\nbest score {b.value:.4f} (pub {u.get('pub_term',0):.4f} + "
          f"ih {u.get('inhouse_term',0):.4f}) | r {u['r']:.3f} | "
          f"MAE {u['subject_mae']:.2f} "
          f"(floor {FLOOR}) | slope {u['slope']:+.3f} | cross-cohort "
          f"{u.get('cross_cohort_r',0):+.3f} | in-house {u.get('inhouse_r',0):+.3f}")
    print(f"  {b.params}")


if __name__ == "__main__":
    main()
