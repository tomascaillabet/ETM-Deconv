"""Phase 3: per-sample gene expression profiles on the cross-dataset validation
pseudobulks (bulk_val), ETM-Deconv vs BayesPrism.

Outputs:
    comparison/gep_global.png, comparison/gep_pearson_by_celltype.png,
    comparison/gep_metrics.csv

Usage:
    python phase3_gep_compare.py --bulk_val bulk_val.h5ad --gt_dir gt_combined --z_standard Z_standard.npz --phase0 ./phase0 --phase2 ./phase2 --output_dir ./phase3_bulkval
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c2_model import load_phase0, seed_M_from_profiles, ETMDeconvC2, C2Encoder

METHOD_COLORS = {"Standard": "#E8884A", "ETM-Deconv": "#4C72B0"}


def pearson(x, y):
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or x[m].std() < 1e-9 or y[m].std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def mae(x, y):
    return float(np.abs(np.asarray(x).ravel() - np.asarray(y).ravel()).mean())


def to_prop(v):
    s = v.sum()
    return v / s if s > 0 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bulk_val", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--z_standard", required=True)
    ap.add_argument("--phase0", required=True)
    ap.add_argument("--phase2", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--encoder", default="encoder_best.pt")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--min_cells", type=int, default=5)
    ap.add_argument("--no_baseline", action="store_true",
                   help="drop the mean-lambda biological baseline from beta_bio (pure topic profile)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); (out / "comparison").mkdir(parents=True, exist_ok=True)

    p0 = load_phase0(args.phase0)
    my_types, my_genes = p0["types"], p0["genes"]
    K, V, T = p0["K"], p0["V"], p0["T"]

    bv = ad.read_h5ad(args.bulk_val)
    bv_genes = list(bv.var_names.astype(str))
    Xbv = np.asarray(bv.X.todense() if hasattr(bv.X, "todense") else bv.X, dtype=np.float32)
    gi = {g: i for i, g in enumerate(bv_genes)}
    cols = np.array([gi.get(g, -1) for g in my_genes])
    X = np.zeros((bv.shape[0], V), dtype=np.float32)
    X[:, cols >= 0] = Xbv[:, cols[cols >= 0]]
    sample_ids = list(bv.obs_names.astype(str))

    M_seed = seed_M_from_profiles(p0["profiles"])
    model = ETMDeconvC2(p0["alpha"], p0["rho"], p0["lambda"], M_seed).to(device)
    model.load_state_dict(torch.load(Path(args.phase2) / "c2_phase2.pt",
                                     map_location=device, weights_only=True))
    encoder = C2Encoder(V, K, T, hidden=args.hidden).to(device)
    encoder.load_state_dict(torch.load(Path(args.phase2) / args.encoder,
                                       map_location=device, weights_only=True))
    encoder.eval()
    with torch.no_grad():
        xb = torch.as_tensor(X, dtype=torch.float32, device=device)
        _, _, mu_dM, _ = encoder(xb)

    lam_base = None if args.no_baseline else model.bio_baseline

    @torch.no_grad()
    def beta_bio_sample(s_idx):
        c = F.softmax(model.M.t() + mu_dM[s_idx], dim=-1)
        logits = model._gene_logits(c)
        if lam_base is not None:
            logits = logits + lam_base
        return F.softmax(logits, dim=-1).cpu().numpy()

    gtd = np.load(Path(args.gt_dir) / "gep_per_patient.npz", allow_pickle=True)
    gep_mat = gtd["matrix"]
    gt_pid = gtd["patient_id"].astype(str)
    gt_ct = gtd["cell_type"].astype(str)
    gt_nc = gtd["n_cells"].astype(int) if "n_cells" in gtd else np.full(len(gt_pid), 9999)
    gt_genes = list(gtd["genes"].astype(str))
    row_of = {(gt_pid[i], gt_ct[i]): i for i in range(len(gt_pid))}
    nc_of = {(gt_pid[i], gt_ct[i]): gt_nc[i] for i in range(len(gt_pid))}

    zd = np.load(args.z_standard, allow_pickle=True)
    Z = zd["Z"]
    z_samp = {s: i for i, s in enumerate(zd["samples"].astype(str))}
    z_ct = {c: i for i, c in enumerate(zd["cell_types"].astype(str))}
    z_genes = list(zd["genes"].astype(str))

    common_types = [t for t in my_types
                    if t in set(gt_ct) and t in z_ct]
    common_genes = sorted(set(my_genes) & set(gt_genes) & set(z_genes))
    my_gl = {g: i for i, g in enumerate(my_genes)}
    gt_gl = {g: i for i, g in enumerate(gt_genes)}
    z_gl = {g: i for i, g in enumerate(z_genes)}
    my_idx = np.array([my_gl[g] for g in common_genes])
    gt_idx = np.array([gt_gl[g] for g in common_genes])
    z_idx = np.array([z_gl[g] for g in common_genes])
    print(f"GEP comparison: {len(common_types)} types, {len(common_genes)} common genes")

    by_real = {ct: [] for ct in common_types}
    by_c2 = {ct: [] for ct in common_types}
    by_std = {ct: [] for ct in common_types}

    for s_idx, sample in enumerate(sample_ids):
        if sample not in z_samp:
            continue
        beta = beta_bio_sample(s_idx)
        for ct in common_types:
            if (sample, ct) not in row_of:
                continue
            if nc_of.get((sample, ct), 0) < args.min_cells:
                continue
            real_v = gep_mat[row_of[(sample, ct)]][gt_idx]
            c2_v = beta[my_types.index(ct)][my_idx]
            std_v = Z[z_samp[sample], z_idx, z_ct[ct]]
            by_real[ct].append(to_prop(real_v))
            by_c2[ct].append(to_prop(c2_v))
            by_std[ct].append(to_prop(std_v))

    for ct in common_types:
        if by_real[ct]:
            by_real[ct] = np.log1p(np.concatenate(by_real[ct]) * 1e4)
            by_c2[ct] = np.log1p(np.concatenate(by_c2[ct]) * 1e4)
            by_std[ct] = np.log1p(np.concatenate(by_std[ct]) * 1e4)
        else:
            by_real[ct] = by_c2[ct] = by_std[ct] = np.array([])

    cts = [c for c in common_types if by_real[c].size > 0]
    real_g = np.concatenate([by_real[c] for c in cts])
    c2_g = np.concatenate([by_c2[c] for c in cts])
    std_g = np.concatenate([by_std[c] for c in cts])

    methods = {"Standard": std_g, "ETM-Deconv": c2_g}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    rng = np.random.default_rng(0)
    metrics_rows = []
    for ax, (name, est) in zip(axes, methods.items()):
        r = pearson(real_g, est); m = mae(real_g, est)
        metrics_rows.append({"scope": "global", "method": name, "pearson": r, "mae": m})
        N = real_g.size
        idx = rng.choice(N, 200_000, replace=False) if N > 200_000 else np.arange(N)
        ax.scatter(real_g[idx], est[idx], s=4, alpha=0.08,
                   color=METHOD_COLORS[name], edgecolors="none")
        hi = float(max(real_g.max(), est.max()))
        ax.plot([0, hi], [0, hi], "k-", lw=0.9, alpha=0.6)
        ax.set_xlabel("log1p(real × 1e4)"); ax.set_ylabel("log1p(estimado × 1e4)")
        ax.set_title(f"{name}\nPearson r = {r:.3f}   |   MAE = {m:.3f}", fontweight="bold")
    fig.suptitle("GEP real vs estimado — global (log1p) · validación cross-dataset",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "comparison/gep_global.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    ct_r_c2 = {c: pearson(by_real[c], by_c2[c]) for c in cts}
    ct_r_std = {c: pearson(by_real[c], by_std[c]) for c in cts}
    for c in cts:
        metrics_rows.append({"scope": c, "method": "ETM-Deconv",
                             "pearson": ct_r_c2[c], "mae": mae(by_real[c], by_c2[c])})
        metrics_rows.append({"scope": c, "method": "Standard",
                             "pearson": ct_r_std[c], "mae": mae(by_real[c], by_std[c])})
    order = sorted(cts, key=lambda c: ct_r_c2[c], reverse=True)
    xp = np.arange(len(order)); w = 0.4
    fig, ax = plt.subplots(figsize=(max(10, len(order) * 0.55), 5.5))
    ax.bar(xp - w/2, [ct_r_std[c] for c in order], w, label="Standard", color=METHOD_COLORS["Standard"])
    ax.bar(xp + w/2, [ct_r_c2[c] for c in order], w, label="ETM-Deconv", color=METHOD_COLORS["ETM-Deconv"])
    ax.set_xticks(xp); ax.set_xticklabels(order, rotation=90, fontsize=8)
    ax.set_ylabel("GEP Pearson r"); ax.axhline(0, color="k", lw=0.5)
    ax.set_title("GEP Pearson por tipo celular (cross-dataset)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "comparison/gep_pearson_by_celltype.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(metrics_rows).to_csv(out / "comparison/gep_metrics.csv", index=False)
    print("\n=== GEP (global, cross-dataset) ===")
    for row in metrics_rows[:2]:
        print(f"  {row['method']:<12} Pearson r = {row['pearson']:.3f}   MAE = {row['mae']:.3f}")
    print(f"Saved -> {out/'comparison/gep_global.png'}")


if __name__ == "__main__":
    main()
