"""Phase 3: run the trained encoder on held-out-patient validation pseudobulks and
score proportions (theta) and per-state profiles (beta_bio) against ground truth.

Outputs:
    phase3_proportions.csv, phase3_report.json, plots/, val_pseudobulk.h5ad

Usage:
    python phase3_deconvolve.py --scrna reference.h5ad --phase0 ./phase0 --phase1 ./phase1 --phase2 ./phase2 --output_dir ./phase3
"""

import argparse
import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from c2_model import load_phase0, seed_M_from_profiles, ETMDeconvC2, C2Encoder
import build_pseudobulk_phase2 as B2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase3")


def pearson(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def infer(model, encoder, X, batch_size, device):
    """Deterministic encoder inference -> theta (N,K), beta_bio per active state."""
    encoder.eval()
    N = X.shape[0]
    thetas = []
    betas = []
    for s in range(0, N, batch_size):
        xb = torch.as_tensor(X[s:s+batch_size], dtype=torch.float32, device=device)
        mu_d, _, mu_dM, _ = encoder(xb)
        theta = F.softmax(mu_d, dim=-1)
        c = F.softmax(model.M.t().unsqueeze(0) + mu_dM, dim=-1)
        logits = model._gene_logits(c)
        beta_bio = F.softmax(logits, dim=-1)
        thetas.append(theta.cpu().numpy())
        betas.append(beta_bio.cpu().numpy())
    return np.vstack(thetas), betas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrna", required=True)
    ap.add_argument("--phase0", required=True)
    ap.add_argument("--phase1", required=True)
    ap.add_argument("--phase2", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--encoder", default="encoder_best.pt")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--build_seed", type=int, default=123)
    ap.add_argument("--n_rep_realistic", type=int, default=30)
    ap.add_argument("--n_rep_diverse", type=int, default=30)
    ap.add_argument("--n_cells", type=int, default=200)
    ap.add_argument("--min_cells", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); (out / "plots").mkdir(parents=True, exist_ok=True)
    log.info(f"Device: {device}")

    p0 = load_phase0(args.phase0)
    K, V, T = p0["K"], p0["V"], p0["T"]
    types = p0["types"]

    pb2 = ad.read_h5ad(Path(args.phase2) / "pseudobulk_mix.h5ad")
    uniq = np.array(sorted(set(pb2.obs["patient"].astype(str))))
    rng = np.random.RandomState(args.split_seed)
    n_val = max(1, int(round(len(uniq) * args.val_frac)))
    val_pat = sorted(rng.choice(uniq, size=n_val, replace=False).tolist())
    log.info(f"Held-out (encoder-unseen) patients [{len(val_pat)}]: {val_pat}")

    log.info(f"Loading scRNA {args.scrna}")
    adata = ad.read_h5ad(args.scrna)
    pb = B2.build(adata, p0["genes"], types, p0["batches"],
                  cell_type_key="cell_type", patient_key="patient_id",
                  batch_key="dataset", n_cells=args.n_cells, min_cells=args.min_cells,
                  n_rep_realistic=args.n_rep_realistic, n_rep_diverse=args.n_rep_diverse,
                  gep_min=0.02, seed=args.build_seed, keep_patients=set(val_pat))
    pb.write_h5ad(out / "val_pseudobulk.h5ad")
    A = int(pb.uns["gep_true_max_active"])
    X = np.asarray(pb.X, dtype=np.float32)
    theta_true = np.asarray(pb.obsm["theta_true"], dtype=np.float32)
    gidx = np.asarray(pb.obsm["gep_true_active_idx"], dtype=np.int64)
    gval = np.asarray(pb.obsm["gep_true_active_gep"], dtype=np.float32).reshape(-1, A, V)
    log.info(f"Validation mixtures: {X.shape[0]} samples")

    M_seed = seed_M_from_profiles(p0["profiles"])
    model = ETMDeconvC2(p0["alpha"], p0["rho"], p0["lambda"], M_seed, sce2tm=p0.get("sce2tm", False), tau=p0.get("tau", 0.2), gene_baseline=p0.get("gene_baseline"), topic_gene_override=p0.get("topic_gene_override")).to(device)
    model.load_state_dict(torch.load(Path(args.phase2) / "c2_phase2.pt",
                                     map_location=device, weights_only=True))
    encoder = C2Encoder(V, K, T, hidden=args.hidden).to(device)
    encoder.load_state_dict(torch.load(Path(args.phase2) / args.encoder,
                                       map_location=device, weights_only=True))

    theta_pred, beta_chunks = infer(model, encoder, X, args.batch_size, device)

    rows = [("proportions", "global", pearson(theta_true, theta_pred),
             float(np.abs(theta_true - theta_pred).mean()),
             float(np.sqrt(((theta_true - theta_pred) ** 2).mean())))]
    per_state = {}
    for k in range(K):
        present = theta_true[:, k] > 1e-6
        if present.sum() >= 3:
            r = pearson(theta_true[:, k], theta_pred[:, k])
            mae = float(np.abs(theta_true[:, k] - theta_pred[:, k]).mean())
            rmse = float(np.sqrt(((theta_true[:, k] - theta_pred[:, k]) ** 2).mean()))
            rows.append(("proportions", types[k], r, mae, rmse))
            per_state[types[k]] = (r, mae, int(present.sum()))

    import csv
    with open(out / "phase3_proportions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["target", "scope", "pearson", "mae", "rmse"])
        for row in rows:
            w.writerow([row[0], row[1], f"{row[2]:.6f}", f"{row[3]:.6f}", f"{row[4]:.6f}"])

    gep_records = []
    for ci, s in enumerate(range(0, X.shape[0], args.batch_size)):
        bb = beta_chunks[ci]
        for bi in range(bb.shape[0]):
            gi = s + bi
            for j in range(A):
                k = gidx[gi, j]
                if k < 0:
                    continue
                gep_records.append((types[k], pearson(bb[bi, k], gval[gi, j])))
    gep_r = np.array([r for _, r in gep_records], dtype=float)
    gep_state_mean = {}
    for name in types:
        vals = [r for n, r in gep_records if n == name and r == r]
        if vals:
            gep_state_mean[name] = float(np.mean(vals))

    report = {
        "val_patients": val_pat,
        "n_val_mixtures": int(X.shape[0]),
        "proportions": {
            "pearson_global": rows[0][2], "mae_global": rows[0][3], "rmse_global": rows[0][4],
            "pearson_perstate_mean": float(np.nanmean([v[0] for v in per_state.values()])),
            "n_states_evaluated": len(per_state),
        },
        "gep": {
            "pearson_mean": float(np.nanmean(gep_r)),
            "pearson_median": float(np.nanmedian(gep_r)),
            "n_state_samples": int(len(gep_r)),
        },
    }
    (out / "phase3_report.json").write_text(json.dumps(report, indent=2))
    log.info(f"PROPORTIONS  global r={rows[0][2]:.3f}  MAE={rows[0][3]:.4f}  "
             f"per-state mean r={report['proportions']['pearson_perstate_mean']:.3f}")
    log.info(f"GEP          mean r={report['gep']['pearson_mean']:.3f}  "
             f"median r={report['gep']['pearson_median']:.3f}")

    palette = sns.color_palette("husl", K)
    fig, ax = plt.subplots(figsize=(7, 7))
    for k in range(K):
        present = theta_true[:, k] > 1e-6
        if present.sum() >= 3:
            ax.scatter(theta_true[present, k], theta_pred[present, k], s=12,
                       color=palette[k], alpha=0.5, label=types[k], linewidths=0)
    lim = max(theta_true.max(), theta_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("true proportion"); ax.set_ylabel("predicted proportion")
    ax.set_title(f"Phase 3 (pseudobulk) — proportions\nglobal r={rows[0][2]:.3f}, "
                 f"MAE={rows[0][3]:.4f}", fontweight="bold")
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)
    fig.tight_layout(); fig.savefig(out / "plots/pred_vs_true.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    names = sorted(per_state, key=lambda n: per_state[n][0], reverse=True)
    rs = [per_state[n][0] for n in names]
    maes = [per_state[n][1] for n in names]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 9))
    a1.bar(names, rs, color="#4C72B0"); a1.axhline(0, color="k", lw=0.5)
    a1.set_ylabel("Pearson r"); a1.set_title("Per-state proportion Pearson", fontweight="bold")
    a1.tick_params(axis="x", rotation=90, labelsize=8)
    a2.bar(names, maes, color="#C44E52")
    a2.set_ylabel("MAE"); a2.set_title("Per-state proportion MAE", fontweight="bold")
    a2.tick_params(axis="x", rotation=90, labelsize=8)
    fig.tight_layout(); fig.savefig(out / "plots/per_state.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.hist(gep_r[~np.isnan(gep_r)], bins=40, color="#55A868")
    a1.axvline(np.nanmean(gep_r), color="k", ls="--",
               label=f"mean={np.nanmean(gep_r):.3f}")
    a1.set_xlabel("per-sample per-state GEP Pearson"); a1.set_ylabel("count")
    a1.set_title("GEP Pearson distribution", fontweight="bold"); a1.legend()
    gnames = sorted(gep_state_mean, key=gep_state_mean.get, reverse=True)
    a2.bar(gnames, [gep_state_mean[n] for n in gnames], color="#55A868")
    a2.set_ylabel("mean GEP Pearson"); a2.set_title("Per-state mean GEP Pearson", fontweight="bold")
    a2.tick_params(axis="x", rotation=90, labelsize=8)
    fig.tight_layout(); fig.savefig(out / "plots/gep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Saved Phase-3 outputs + plots to {out}")


if __name__ == "__main__":
    main()
