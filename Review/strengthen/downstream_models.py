"""Compare downstream model families on the frozen per-channel encoders."""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("HARMONIZED_F", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import channel_encoder_search_arch as A  # noqa: E402
import channel_encoder_search_multi as S  # noqa: E402
import config_review as C  # noqa: E402

CKPT = os.path.join(HERE, "final_encoders")
RES = os.path.join(os.path.dirname(HERE), "results")
CACHE = os.path.join(CKPT, "downstream_embeddings.npz")
FLOOR = 14.07
PCA_D = 12          # per-channel reduction before the Gram matrix


# ------------------------------------------------------------------ embedding
def build_encoder(cfg, state, device):
    e = A.ChannelEncoderV2(
        conv_channels=cfg["conv_channels"], kernel_size=cfg["kernel_size"],
        stride=cfg["stride"], padding=cfg["padding"], dilation=cfg["dilation"],
        gru_hidden=cfg["gru_hidden"], num_layers_gru=cfg["num_layers_gru"],
        embedding_dim=cfg["embedding_dim"], cnn_dropout=cfg["cnn_dropout"],
        rnn_dropout=cfg["rnn_dropout"], activation=cfg["activation"],
        **cfg["arch_extra"]).to(device)
    e.load_state_dict(state)
    e.eval()
    return e


def embed(enc, X, device, bs=1024):
    zs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(np.asarray(X[i:i + bs]), device=device)
            with torch.amp.autocast(device_type="cuda"):
                z, _ = enc(xb)
            zs.append(z.float().cpu().numpy())
    return np.concatenate(zs)


def compute():
    """Subject-level embeddings per channel for the pool and the in-house cohort."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return z["pub"].item(), z["ih"].item()
    device = torch.device("cuda")
    _, meta = S.load_pool()
    ages, subj, dsets = meta["age"], meta["subject"], meta["dataset"]

    from harmonized_io import build_windows
    # ALL in-house stimuli, not just eyes-closed. The stimulus label travels with
    # each window so the evaluation can be split three ways afterwards: per
    w = build_windows("ours", recipe="prefiltered_uv", win=512, stride=256,
                      verbose=False)
    ih_age = np.asarray(w["age"], float)
    ih_subj = np.asarray(w["subject"]).astype(str)
    ih_stim = np.asarray(w["stimulus"]).astype(str)

    pub, ih = {}, {}
    for i, ch in enumerate(A.CHANNELS):
        # skip channels whose training has not finished, so a partial encoder set
        # can be evaluated without waiting for the last worker
        if not os.path.exists(os.path.join(CKPT, f"{ch}.pt")):
            print(f"  {ch}: no checkpoint yet -- skipped", flush=True)
            continue
        ck = torch.load(os.path.join(CKPT, f"{ch}.pt"), map_location="cpu",
                        weights_only=False)
        enc = build_encoder(ck["cfg"], ck["state_dict"], device)
        Xc = S.channel_array(i)
        pub[ch] = embed(enc, Xc, device)
        ih[ch] = embed(enc, np.ascontiguousarray(w["X"][:, :, i]), device)
        del enc
        torch.cuda.empty_cache()
        print(f"  embedded {ch}", flush=True)
    P = {"Z": pub, "age": ages, "subj": subj, "ds": dsets}
    I = {"Z": ih, "age": ih_age, "subj": ih_subj, "stim": ih_stim}
    np.savez_compressed(CACHE, pub=np.array(P, dtype=object),
                        ih=np.array(I, dtype=object))
    return P, I


def subj_mean(Z, subj):
    d = pd.DataFrame(Z)
    d["s"] = subj
    return d.groupby("s").mean()


def subject_tables_masked(D, mask):
    """Subject-level tables built from a subset of windows (e.g. one stimulus)."""
    sub = {"Z": {c: Z[mask] for c, Z in D["Z"].items()},
           "age": D["age"][mask], "subj": D["subj"][mask]}
    return subject_tables(sub)


def subject_tables(D):
    """-> dict channel -> [n_subj, dim], plus aligned ages / ids / datasets."""
    out, idx = {}, None
    for ch, Z in D["Z"].items():
        M = subj_mean(Z, D["subj"])
        if idx is None:
            idx = M.index
        out[ch] = M.loc[idx].to_numpy()
    ages = pd.Series(D["age"]).groupby(pd.Series(D["subj"])).first().loc[idx].to_numpy()
    ds = (pd.Series(D["ds"]).groupby(pd.Series(D["subj"])).first().loc[idx].to_numpy()
          if "ds" in D else np.array(["ours"] * len(idx)))
    return out, ages, np.array(idx), ds


# ------------------------------------------------------------------ features
def pca_reduce(tabs, d=PCA_D, fit_idx=None):
    """Per-channel PCA to a common width so channels can form a Gram matrix.
    Fitted on fit_idx only, so the test subjects never influence the basis."""
    from sklearn.decomposition import PCA
    out = {}
    for ch, M in tabs.items():
        k = min(d, M.shape[1], len(fit_idx) - 1)
        p = PCA(n_components=k).fit(M[fit_idx])
        Z = p.transform(M)
        if Z.shape[1] < d:                       # pad so every channel matches
            Z = np.pad(Z, ((0, 0), (0, d - Z.shape[1])))
        out[ch] = Z
    return out


def feat_concat(tabs, chans):
    return np.concatenate([tabs[c] for c in chans], 1)


def feat_gram(red, chans, log_euclid=False):
    """14x14 channel Gram matrix per subject; upper triangle as features."""
    V = np.stack([red[c] for c in chans], 1)          # [n, 14, d]
    V = V / np.maximum(np.linalg.norm(V, axis=2, keepdims=True), 1e-8)
    G = np.einsum("nid,njd->nij", V, V)               # [n, 14, 14]
    if log_euclid:
        G = G + 1e-3 * np.eye(G.shape[1])[None]       # make it SPD
        w, Q = np.linalg.eigh(G)
        w = np.log(np.maximum(w, 1e-6))
        G = np.einsum("nij,nj,nkj->nik", Q, w, Q)
    iu = np.triu_indices(G.shape[1])
    return G[:, iu[0], iu[1]]


# ------------------------------------------------------------------- scoring
def fit_ridge(Xtr, ytr, Xte):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=np.logspace(-2, 4, 14)).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte)), (sc, m)


def fit_xgb(Xtr, ytr, Xte, seed=0):
    """Gradient-boosted trees on the latent space."""
    from xgboost import XGBRegressor
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4,
                     subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
                     reg_lambda=2.0, random_state=seed, n_jobs=8,
                     tree_method="hist").fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte)), (sc, m)


def fit_mlp(Xtr, ytr, Xte, seed=0):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, max_iter=600,
                     random_state=seed, early_stopping=True).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte)), (sc, m)


def metrics(y, p):
    """Pearson r, plus Spearman rho wich needs only a monotone relationship."""
    if len(y) < 5 or np.std(p) < 1e-9:
        return {"r": 0.0, "rho": 0.0,
                "mae": float(np.abs(y - np.median(y)).mean()), "slope": -1.0}
    from scipy.stats import spearmanr
    return {"r": float(np.corrcoef(y, p)[0, 1]),
            "rho": float(spearmanr(y, p).statistic),
            "mae": float(np.abs(p - y).mean()),
            "slope": float(np.polyfit(y, p - y, 1)[0])}


def cohort_transfer(y, p, ds):
    """Recalibrate on one cohort, score the other. Both directions."""
    out = []
    uniq = [u for u in np.unique(ds) if (ds == u).sum() >= 20]
    for a in uniq:
        for b in uniq:
            if a == b:
                continue
            ka, kb = ds == a, ds == b
            if np.std(y[ka]) < 2 or np.std(y[kb]) < 2 or np.std(p[ka]) < 1e-9:
                continue
            k = np.polyfit(p[ka], y[ka], 1)
            pb = np.polyval(k, p[kb])
            if np.std(pb) > 1e-9:
                out.append(float(np.corrcoef(y[kb], pb)[0, 1]))
    return float(np.mean(out)) if out else 0.0


def main():
    P, I = compute()
    tabs, ages, ids, ds = subject_tables(P)
    itabs, iages, iids, _ = subject_tables(I)
    chans = [c for c in A.CHANNELS if c in tabs]
    print(f"\npublic: {len(ages)} subjects, {len(chans)} channels | "
          f"in-house: {len(iages)} subjects\n")

    from stimulus_groups import group as _g
    I["stim_is_ec"] = np.array([_g("ours", x) == "rest_ec" for x in I["stim"]])
    print(f"in-house: {len(I['age'])} windows, {I['stim_is_ec'].sum()} eyes-closed, "
          f"{len(np.unique(I['stim']))} stimuli")
    per_stim_store = {}
    rng = np.random.default_rng(C.SEED)
    order = rng.permutation(len(ages))
    folds = np.array_split(order, 5)
    rows = []

    def ih_views(predict):
        """predict: callable(dict ch->table) -> per-subject predictions."""
        views = {}
        # eyes-closed, and every window pooled
        for name, mask in (("eyes_closed", I["stim_is_ec"]),
                           ("all_pooled", np.ones(len(I["age"]), bool))):
            t, a_, _, _ = subject_tables_masked(I, mask)
            views[name] = metrics(a_, predict(t))
        per = {}
        for st in np.unique(I["stim"]):
            m = I["stim"] == st
            if m.sum() < 100:
                continue
            t, a_, _, _ = subject_tables_masked(I, m)
            if len(a_) < 20 or np.std(a_) < 2:
                continue
            per[str(st)] = metrics(a_, predict(t))
        views["per_stimulus"] = per
        return views

    def run(name, build, fitter=fit_ridge, needs_pca=False):
        from sklearn.decomposition import PCA
        preds = np.empty(len(ages))
        fitted = []
        for te in folds:
            tr = np.setdiff1d(np.arange(len(ages)), te)
            if needs_pca:
                bases = {}
                for c in chans:
                    k = min(PCA_D, tabs[c].shape[1], len(tr) - 1)
                    bases[c] = PCA(n_components=k).fit(tabs[c][tr])
                def pad(Z):
                    return (np.pad(Z, ((0, 0), (0, PCA_D - Z.shape[1])))
                            if Z.shape[1] < PCA_D else Z)
                red = {c: pad(bases[c].transform(tabs[c])) for c in chans}
                Xall = build(red, chans)
                proj = lambda t: build({c: pad(bases[c].transform(t[c])) for c in chans}, chans)
            else:
                Xall = build(tabs, chans)
                proj = lambda t: build(t, chans)
            preds[te], model = fitter(Xall[tr], ages[tr], Xall[te])
            fitted.append((model, proj))

        def predict(t):
            ps = []
            for (sc, m), proj in fitted:
                ps.append(m.predict(sc.transform(proj(t))))
            return np.mean(ps, 0)

        mt = metrics(ages, preds)
        mt["cross_cohort_r"] = cohort_transfer(ages, preds, ds)
        v = ih_views(predict)
        mt["model"] = name
        for k in ("eyes_closed", "all_pooled"):
            mt[f"ih_{k}_r"] = v[k]["r"]; mt[f"ih_{k}_mae"] = v[k]["mae"]
            mt[f"ih_{k}_slope"] = v[k]["slope"]; mt[f"ih_{k}_rho"] = v[k]["rho"]
        per = v["per_stimulus"]
        mt["ih_n_stim"] = len(per)
        mt["ih_stim_mean_r"] = float(np.mean([x["r"] for x in per.values()])) if per else 0.0
        mt["ih_stim_best"] = max(per, key=lambda k: per[k]["r"]) if per else "-"
        mt["ih_stim_best_r"] = max((x["r"] for x in per.values()), default=0.0)
        per_stim_store[name] = per
        rows.append(mt)
        print(f"  {name:12s} r {mt['r']:.3f} rho {mt['rho']:.3f} "
              f"MAE {mt['mae']:5.2f} slope {mt['slope']:+.3f} "
              f"xcoh {mt['cross_cohort_r']:+.3f} | IH ec {mt['ih_eyes_closed_r']:+.3f} "
              f"all {mt['ih_all_pooled_r']:+.3f} | best stim {mt['ih_stim_best'][:22]} "
              f"{mt['ih_stim_best_r']:+.3f}", flush=True)

    # --- control: no cross-channel mixing at all --------------------------
    print("fitting families (5-fold, subject level):")

    def build_avg(t, c):
        return t            # marker; handled by its own fitter below

    preds = np.zeros(len(ages))
    per_ch_models = {c: [] for c in chans}
    for ch in chans:
        pc = np.empty(len(ages))
        for te in folds:
            tr = np.setdiff1d(np.arange(len(ages)), te)
            pc[te], model = fit_ridge(tabs[ch][tr], ages[tr], tabs[ch][te])
            per_ch_models[ch].append(model)
        preds += pc / len(chans)

    def predict_avg(t):
        out = 0
        for ch in chans:
            ps = [m.predict(sc.transform(t[ch])) for sc, m in per_ch_models[ch]]
            out = out + np.mean(ps, 0) / len(chans)
        return out

    mt = metrics(ages, preds)
    mt["cross_cohort_r"] = cohort_transfer(ages, preds, ds)
    v = ih_views(predict_avg)
    mt["model"] = "avg14 (no mixing)"
    for k in ("eyes_closed", "all_pooled"):
        mt[f"ih_{k}_r"] = v[k]["r"]; mt[f"ih_{k}_mae"] = v[k]["mae"]
        mt[f"ih_{k}_slope"] = v[k]["slope"]; mt[f"ih_{k}_rho"] = v[k]["rho"]
    per = v["per_stimulus"]
    mt["ih_n_stim"] = len(per)
    mt["ih_stim_mean_r"] = float(np.mean([x["r"] for x in per.values()])) if per else 0.0
    mt["ih_stim_best"] = max(per, key=lambda k: per[k]["r"]) if per else "-"
    mt["ih_stim_best_r"] = max((x["r"] for x in per.values()), default=0.0)
    per_stim_store["avg14 (no mixing)"] = per
    rows.append(mt)
    print(f"  {'avg14':12s} r {mt['r']:.3f} rho {mt['rho']:.3f} "
          f"MAE {mt['mae']:5.2f} slope {mt['slope']:+.3f} "
          f"xcoh {mt['cross_cohort_r']:+.3f} | IH ec {mt['ih_eyes_closed_r']:+.3f} "
          f"all {mt['ih_all_pooled_r']:+.3f} | best stim {mt['ih_stim_best'][:22]} "
          f"{mt['ih_stim_best_r']:+.3f}", flush=True)

    run("concat", feat_concat)
    run("concat_mlp", feat_concat, fitter=fit_mlp)
    run("concat_xgb", feat_concat, fitter=fit_xgb)
    run("gram", lambda r, c: feat_gram(r, c, False), needs_pca=True)
    run("logeuclid", lambda r, c: feat_gram(r, c, True), needs_pca=True)

    df = pd.DataFrame(rows)
    cols = ["model", "r", "rho", "mae", "slope", "cross_cohort_r",
            "ih_eyes_closed_r", "ih_eyes_closed_rho", "ih_eyes_closed_mae",
            "ih_all_pooled_r", "ih_all_pooled_rho", "ih_all_pooled_mae",
            "ih_stim_mean_r", "ih_stim_best", "ih_stim_best_r"]
    os.makedirs(RES, exist_ok=True)
    df[cols].round(4).to_csv(os.path.join(RES, "Table_downstream_models.csv"), index=False)
    print("\n" + df[cols].round(3).to_string(index=False))

    ps = []
    for mdl, per in per_stim_store.items():
        for st, m in per.items():
            ps.append({"model": mdl, "stimulus": st, **m})
    if ps:
        pdf = pd.DataFrame(ps)
        pdf.round(4).to_csv(os.path.join(RES, "Table_downstream_per_stimulus.csv"),
                            index=False)
        g = (pdf.groupby("stimulus").agg(mean_r=("r", "mean"), max_r=("r", "max"),
                                         mean_mae=("mae", "mean"))
               .sort_values("mean_r", ascending=False))
        print(f"\nin-house per stimulus, averaged over the {len(per_stim_store)} "
              f"model families (top 12 of {len(g)}):")
        print(g.head(12).round(3).to_string())
        print(f"\nstimuli with mean r > 0.10: {(g.mean_r > 0.10).sum()}/{len(g)}")

    base = df[df.model.str.startswith("avg14")].iloc[0]
    print(f"\nvs the no-mixing control (r {base.r:.3f}):")
    for t in df.itertuples():
        if t.model.startswith("avg14"):
            continue
        print(f"  {t.model:12s} dr {t.r-base.r:+.3f}  "
              f"d_IH_ec {t.ih_eyes_closed_r-base.ih_eyes_closed_r:+.3f}  "
              f"d_IH_all {t.ih_all_pooled_r-base.ih_all_pooled_r:+.3f}")


if __name__ == "__main__":
    main()
