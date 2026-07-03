"""Phase 1: learn the per-cell-type topic recipes M and variability sigma_k on
pure-state pseudobulks (theta = e_k).

Outputs:
    M.npy (T,K), sigma_k.csv, c2_phase1.pt, training_log.json, phase1_report.json

Usage:
    python phase1_learn_M.py --pseudobulk pseudobulk_pure.h5ad --phase0 ./phase0 --output_dir ./phase1 --epochs 300 --lr 5e-3
"""

import argparse
import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from c2_model import load_phase0, seed_M_from_profiles, ETMDeconvC2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase1")



class PureStateDataset(Dataset):
    def __init__(self, X, state_idx, batch_idx):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.state_idx = torch.as_tensor(state_idx, dtype=torch.long)
        self.batch_idx = torch.as_tensor(batch_idx, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.state_idx[i], self.batch_idx[i]


def recon_loss(r, X, N):
    """Multinomial NLL normalised by library size; mean over batch."""
    return (-(X * torch.log(r + 1e-12)).sum(dim=-1) / N).mean()


@torch.no_grad()
def mean_cos_dist(r, X, N):
    """1 - cosine(r, empirical gene freq), mean over batch — convergence proxy."""
    x = X / N.unsqueeze(-1)
    cos = F.cosine_similarity(r, x, dim=-1)
    return (1 - cos).mean().item()



def stage1_learn_M(model, loader, M_seed, anchor_w, cfg, device):
    opt = torch.optim.Adam([model.M], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs,
                                                       eta_min=cfg.lr * 0.01)
    M_seed_t = torch.as_tensor(M_seed, dtype=torch.float32, device=device)
    anchor_t = torch.as_tensor(anchor_w, dtype=torch.float32, device=device)

    history, best, no_improve = [], float("inf"), 0
    for epoch in range(cfg.epochs):
        tot_rec = tot_cos = tot_n = 0.0
        for X, sidx, bidx in loader:
            X, sidx, bidx = X.to(device), sidx.to(device), bidx.to(device)
            N = X.sum(dim=-1).clamp(min=1)
            opt.zero_grad()
            r = model.decode_pure(sidx, bidx, use_lambda=True)
            rec = recon_loss(r, X, N)
            anchor = (anchor_t * (model.M - M_seed_t).pow(2).mean(dim=0)).sum()
            loss = rec + anchor
            loss.backward()
            opt.step()
            tot_rec += rec.item() * X.shape[0]
            tot_cos += mean_cos_dist(r, X, N) * X.shape[0]
            tot_n += X.shape[0]
        sched.step()
        rec_ep = tot_rec / tot_n
        cos_ep = tot_cos / tot_n
        history.append({"epoch": epoch, "recon": rec_ep, "cos_dist": cos_ep,
                        "lr": sched.get_last_lr()[0]})
        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            log.info(f"[S1] Epoch {epoch:4d}/{cfg.epochs} | recon={rec_ep:.4f} "
                     f"| cos_dist={cos_ep:.4f}")
        if rec_ep < best - cfg.min_delta:
            best, no_improve = rec_ep, 0
        else:
            no_improve += 1
            if cfg.patience and no_improve >= cfg.patience:
                log.info(f"[S1] Early stop at epoch {epoch}")
                break
    return history



def stage2_estimate_sigma(model, X_all, sidx_all, bidx_all, cfg, device):
    """
    Fit a per-sample dM (M_samples, T) by reconstruction with M frozen, then set
    sigma_k from the gauge-centered RMS of dM over the samples of each state.
    """
    M_samp = X_all.shape[0]
    dM = torch.zeros(M_samp, model.T, device=device, requires_grad=True)
    opt = torch.optim.Adam([dM], lr=cfg.sigma_lr)

    X_all = X_all.to(device)
    sidx_all = sidx_all.to(device)
    bidx_all = bidx_all.to(device)
    N_all = X_all.sum(dim=-1).clamp(min=1)
    Mt_cols = model.M.t()[sidx_all].detach()
    lam = model.lambda_batch[bidx_all].detach()

    bs = cfg.sigma_batch
    for step in range(cfg.sigma_steps):
        perm = torch.randperm(M_samp, device=device)
        tot = 0.0
        for s in range(0, M_samp, bs):
            sel = perm[s:s + bs]
            opt.zero_grad()
            c = F.softmax(Mt_cols[sel] + dM[sel], dim=-1)
            logits = model._gene_logits(c) + lam[sel]
            r = F.softmax(logits, dim=-1)
            rec = recon_loss(r, X_all[sel], N_all[sel])
            reg = cfg.sigma_l2 * dM[sel].pow(2).mean()
            (rec + reg).backward()
            opt.step()
            tot += rec.item()
        if step % 50 == 0 or step == cfg.sigma_steps - 1:
            log.info(f"[S2] step {step:4d}/{cfg.sigma_steps} | recon={tot/ (M_samp//bs +1):.4f}")

    with torch.no_grad():
        dM_c = dM - dM.mean(dim=-1, keepdim=True)
        dM_c = dM_c.cpu().numpy()
    sidx_np = sidx_all.cpu().numpy()

    K = model.K
    sigma = np.full(K, np.nan, dtype=np.float32)
    for k in range(K):
        mask = sidx_np == k
        if mask.sum() > 0:
            sigma[k] = float(np.sqrt(np.mean(dM_c[mask] ** 2)))
    covered = ~np.isnan(sigma)
    fallback = float(np.nanmedian(sigma)) if covered.any() else cfg.sigma_floor
    sigma[~covered] = fallback
    sigma = np.clip(sigma, cfg.sigma_floor, cfg.sigma_ceil)
    return sigma, covered



def main():
    p = argparse.ArgumentParser(description="Phase 1 — learn M and sigma_k")
    p.add_argument("--pseudobulk", required=True)
    p.add_argument("--phase0", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--anchor", type=float, default=0.01,
                   help="Base seed-anchor weight; per-state scaled by median_n/n_k.")
    p.add_argument("--anchor_max", type=float, default=1.0)
    p.add_argument("--sigma_steps", type=int, default=300)
    p.add_argument("--sigma_lr", type=float, default=5e-2)
    p.add_argument("--sigma_batch", type=int, default=256)
    p.add_argument("--sigma_l2", type=float, default=1e-4)
    p.add_argument("--sigma_floor", type=float, default=0.05)
    p.add_argument("--sigma_ceil", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    cfg = p.parse_args()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log.info(f"Device: {device}")

    p0 = load_phase0(cfg.phase0)
    pb = ad.read_h5ad(cfg.pseudobulk)
    if list(pb.uns["cell_types"]) != p0["types"]:
        raise ValueError("Pseudobulk cell_types disagree with Phase-0 ordering")
    X = np.asarray(pb.X, dtype=np.float32)
    sidx = pb.obs["state_idx"].to_numpy().astype(np.int64)
    bidx = pb.obs["batch_idx"].to_numpy().astype(np.int64)
    K = p0["K"]
    log.info(f"Pseudobulk: {X.shape[0]} samples, V={X.shape[1]}")

    n_k = np.array([(sidx == k).sum() for k in range(K)], dtype=np.float32)
    med_n = np.median(n_k[n_k > 0]) if (n_k > 0).any() else 1.0
    anchor_w = np.where(n_k > 0,
                        np.clip(cfg.anchor * med_n / np.maximum(n_k, 1.0), 0, cfg.anchor_max),
                        cfg.anchor_max).astype(np.float32)

    if p0.get("M_phase0") is not None:
        M_seed = p0["M_phase0"].astype(np.float32)
        log.info("Init M from Phase-0 learned M (M_phase0.npy)")
    else:
        M_seed = seed_M_from_profiles(p0["profiles"])
    model = ETMDeconvC2(p0["alpha"], p0["rho"], p0["lambda"], M_seed).to(device)
    for name, pa in model.named_parameters():
        pa.requires_grad = (name == "M")

    loader = DataLoader(PureStateDataset(X, sidx, bidx),
                        batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    log.info("=== Stage 1: learn M ===")
    history = stage1_learn_M(model, loader, M_seed, anchor_w, cfg, device)

    log.info("=== Stage 2: estimate sigma_k ===")
    sigma, covered = stage2_estimate_sigma(
        model,
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(sidx, dtype=torch.long),
        torch.as_tensor(bidx, dtype=torch.long),
        cfg, device,
    )
    with torch.no_grad():
        model.log_sigma_k.copy_(torch.tensor(np.log(sigma), dtype=torch.float32, device=device))

    np.save(out / "M.npy", model.M.detach().cpu().numpy())
    import csv
    with open(out / "sigma_k.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "sigma_k", "n_pseudobulk", "covered"])
        for k in range(K):
            w.writerow([p0["types"][k], f"{sigma[k]:.6f}", int(n_k[k]), bool(covered[k])])
    torch.save(model.state_dict(), out / "c2_phase1.pt")
    (out / "training_log.json").write_text(json.dumps(history, indent=2))

    report = {
        "n_pseudobulk": int(X.shape[0]),
        "states_covered": int(covered.sum()),
        "states_total": int(K),
        "states_seed_only": [p0["types"][k] for k in range(K) if not covered[k]],
        "final_recon": history[-1]["recon"] if history else None,
        "final_cos_dist": history[-1]["cos_dist"] if history else None,
        "sigma_k_mean": float(sigma.mean()),
        "sigma_k_min": float(sigma.min()),
        "sigma_k_max": float(sigma.max()),
    }
    (out / "phase1_report.json").write_text(json.dumps(report, indent=2))
    log.info(f"Saved Phase-1 outputs to {out}")
    log.info(f"  covered {report['states_covered']}/{K} states | "
             f"recon={report['final_recon']:.4f} cos_dist={report['final_cos_dist']:.4f} | "
             f"sigma_k in [{sigma.min():.3f}, {sigma.max():.3f}]")


if __name__ == "__main__":
    main()
