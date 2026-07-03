"""Phase 3: proportions on the cross-dataset validation pseudobulks (bulk_val),
ETM-Deconv vs BayesPrism.

Outputs:
    theta.csv, comparison/proportions_global.png, comparison/metrics.csv

Usage:
    python phase3_bulkval_compare.py --bulk_val bulk_val.h5ad --gt_dir gt_combined --instaprism theta_standard.csv --phase0 ./phase0 --phase2 ./phase2 --output_dir ./phase3_bulkval
"""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from c2_model import load_phase0, seed_M_from_profiles, ETMDeconvC2, C2Encoder


def pearson(x, y):
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or x[m].std() < 1e-9 or y[m].std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def mae(x, y):
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    return float(np.abs(x - y).mean())


@torch.no_grad()
def infer_theta(model, encoder, X, device, batch_size=64):
    encoder.eval()
    outs = []
    for s in range(0, X.shape[0], batch_size):
        xb = torch.as_tensor(X[s:s+batch_size], dtype=torch.float32, device=device)
        mu_d, _, _, _ = encoder(xb)
        outs.append(F.softmax(mu_d, dim=-1).cpu().numpy())
    return np.vstack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bulk_val", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--instaprism", required=True)
    ap.add_argument("--phase0", required=True)
    ap.add_argument("--phase2", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--encoder", default="encoder_best.pt")
    ap.add_argument("--hidden", type=int, default=256)
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
    present = cols >= 0
    X[:, present] = Xbv[:, cols[present]]
    print(f"Aligned bulk_val: {bv.shape[0]} samples, {present.sum()}/{V} genes present")
    sample_ids = list(bv.obs_names.astype(str))

    M_seed = seed_M_from_profiles(p0["profiles"])
    model = ETMDeconvC2(p0["alpha"], p0["rho"], p0["lambda"], M_seed, sce2tm=p0.get("sce2tm", False), tau=p0.get("tau", 0.2), gene_baseline=p0.get("gene_baseline"), topic_gene_override=p0.get("topic_gene_override")).to(device)
    model.load_state_dict(torch.load(Path(args.phase2) / "c2_phase2.pt",
                                     map_location=device, weights_only=True))
    encoder = C2Encoder(V, K, T, hidden=args.hidden).to(device)
    encoder.load_state_dict(torch.load(Path(args.phase2) / args.encoder,
                                       map_location=device, weights_only=True))

    theta = infer_theta(model, encoder, X, device)
    theta_df = pd.DataFrame(theta, index=sample_ids, columns=my_types)
    theta_df.to_csv(out / "theta.csv")

    gt = pd.read_csv(Path(args.gt_dir) / "proportions.csv", index_col=0)
    gt.index = gt.index.astype(str)
    std = pd.read_csv(args.instaprism, index_col=0)
    std.index = std.index.astype(str)

    samples = [s for s in sample_ids if s in gt.index and s in std.index]
    common_types = [t for t in my_types if t in gt.columns and t in std.columns]
    print(f"Comparison on {len(samples)} samples × {len(common_types)} common types")

    G = gt.loc[samples, common_types].values
    methods = {
        "Standard":   std.loc[samples, common_types].values,
        "ETM-Deconv": theta_df.loc[samples, common_types].values,
    }

    ct_colors = dict(zip(common_types, sns.color_palette("husl", len(common_types))))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    fig.suptitle("Proporciones reales vs estimadas (global)", fontsize=14, fontweight="bold")
    metrics = []
    for ax, (name, P) in zip(axes, methods.items()):
        r = pearson(G, P); m = mae(G, P)
        metrics.append({"method": name, "pearson": r, "mae": m})
        for j, ct in enumerate(common_types):
            ax.scatter(G[:, j], P[:, j], s=14, color=ct_colors[ct], alpha=0.6,
                       linewidths=0, label=ct)
        ax.plot([0, 1], [0, 1], color="gray", lw=1)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Proporción real"); ax.set_ylabel("Proporción estimada")
        ax.set_title(f"{name}\nPearson r = {r:.3f}   |   MAE = {m:.3f}", fontsize=11)
    axes[-1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5),
                    title="Tipo celular", ncol=1)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "comparison/proportions_global.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(metrics).to_csv(out / "comparison/metrics.csv", index=False)
    print("\n=== Proportions (global, cross-dataset bulk_val) ===")
    for mrow in metrics:
        print(f"  {mrow['method']:<12} Pearson r = {mrow['pearson']:.3f}   MAE = {mrow['mae']:.3f}")
    print(f"\nSaved -> {out/'comparison/proportions_global.png'}")


if __name__ == "__main__":
    main()
