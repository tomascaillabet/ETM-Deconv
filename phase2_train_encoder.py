"""Phase 2: train the amortised encoder for the proportions theta and the per-sample
topic perturbation dM on mixture pseudobulks.

Outputs:
    encoder_best.pt, c2_phase2.pt, and training/validation logs (theta Pearson/MAE
    and per-sample GEP Pearson on held-out patients).

Usage:
    python phase2_train_encoder.py --pseudobulk pseudobulk_mix.h5ad --phase0 ./phase0 --phase1 ./phase1 --output_dir ./phase2 --epochs 200 --lr 1e-3 --w_gep 1.0 --w_theta 10.0
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

from c2_model import load_phase0, seed_M_from_profiles, ETMDeconvC2, C2Encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase2")



class MixtureDataset(Dataset):
    def __init__(self, X, theta, batch_idx, gep_idx, gep_val):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.theta = torch.as_tensor(theta, dtype=torch.float32)
        self.batch_idx = torch.as_tensor(batch_idx, dtype=torch.long)
        self.gep_idx = torch.as_tensor(gep_idx, dtype=torch.long)
        self.gep_val = torch.as_tensor(gep_val, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return (self.X[i], self.theta[i], self.batch_idx[i],
                self.gep_idx[i], self.gep_val[i])


class BulkDataset(Dataset):
    """Unlabelled real bulk (e.g. TCGA): only the count vector, no theta/GEP."""
    def __init__(self, X):
        self.X = torch.as_tensor(X, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i]



def kl_gaussian(mu, logvar, prior_var):
    """KL[N(mu, exp(logvar)) || N(0, prior_var)], per element (not reduced)."""
    var = logvar.exp()
    return 0.5 * (torch.log(prior_var / var) + (var + mu.pow(2)) / prior_var - 1.0)


def forward_batch(model, encoder, X, batch_idx, sample=True):
    """Encode + decode. Returns dict with r, beta_bio, theta_mean, theta_samp, KLs."""
    mu_d, logvar_d, mu_dM, logvar_dM = encoder(X)
    if sample:
        delta = mu_d + (0.5 * logvar_d).exp() * torch.randn_like(mu_d)
        dM = mu_dM + (0.5 * logvar_dM).exp() * torch.randn_like(mu_dM)
    else:
        delta, dM = mu_d, mu_dM
    theta_samp = F.softmax(delta, dim=-1)
    theta_mean = F.softmax(mu_d, dim=-1)

    Mt = model.M.t().unsqueeze(0)
    c = F.softmax(Mt + dM, dim=-1)
    logits = model._gene_logits(c)
    beta_bio = F.softmax(logits + model.bio_baseline, dim=-1)
    beta = F.softmax(logits + model.lambda_batch[batch_idx].unsqueeze(1), dim=-1)
    r = torch.einsum("bk,bkv->bv", theta_samp, beta)
    return {
        "r": r, "beta_bio": beta_bio,
        "theta_mean": theta_mean, "theta_samp": theta_samp,
        "mu_d": mu_d, "logvar_d": logvar_d,
        "mu_dM": mu_dM, "logvar_dM": logvar_dM,
    }


def forward_bulk(model, encoder, X):
    """
    Unsupervised bulk forward: reconstruct WITHOUT any lambda (beta_bio, biological
    baseline only). The bulk's only degrees of freedom are theta and dM, so no
    per-sample lambda competes for the variance.
    Returns r_bio (B,V) and the dM posterior for the KL/identifiability terms.
    """
    mu_d, logvar_d, mu_dM, logvar_dM = encoder(X)
    delta = mu_d + (0.5 * logvar_d).exp() * torch.randn_like(mu_d)
    dM = mu_dM + (0.5 * logvar_dM).exp() * torch.randn_like(mu_dM)
    theta = F.softmax(delta, dim=-1)
    Mt = model.M.t().unsqueeze(0)
    c = F.softmax(Mt + dM, dim=-1)
    logits = model._gene_logits(c)
    beta_bio = F.softmax(logits + model.bio_baseline, dim=-1)
    r = torch.einsum("bk,bkv->bv", theta, beta_bio)
    return {"r": r, "mu_d": mu_d, "logvar_d": logvar_d,
            "mu_dM": mu_dM, "logvar_dM": logvar_dM}


def gep_loss(beta_bio, theta_true, gep_idx, gep_val):
    """
    -sum_{k active} theta_true_k * sum_g gep_true_{k,g} log beta_bio_{k,g}, mean over B.
    Uses the sparse active-state storage (gep_idx (B,A), gep_val (B,A,V)).
    """
    B, A = gep_idx.shape
    ar = torch.arange(B, device=beta_bio.device)
    total = beta_bio.new_zeros(B)
    for j in range(A):
        k = gep_idx[:, j]
        valid = k >= 0
        kk = k.clamp(min=0)
        bb = beta_bio[ar, kk]
        w = theta_true[ar, kk]
        ce = -(gep_val[:, j] * torch.log(bb + 1e-12)).sum(-1)
        total = total + torch.where(valid, w * ce, torch.zeros_like(ce))
    return total.mean()



def pearson(a, b):
    a = a.ravel(); b = b.ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def evaluate(model, encoder, loader, device):
    encoder.eval()
    Ts, Ps, geps = [], [], []
    for X, theta, bidx, gidx, gval in loader:
        X, bidx = X.to(device), bidx.to(device)
        out = forward_batch(model, encoder, X, bidx, sample=False)
        Ts.append(theta.numpy())
        Ps.append(out["theta_mean"].cpu().numpy())
        bb = out["beta_bio"].cpu().numpy()
        gidx_n, gval_n = gidx.numpy(), gval.numpy()
        for b in range(X.shape[0]):
            for j in range(gidx_n.shape[1]):
                k = gidx_n[b, j]
                if k < 0:
                    continue
                geps.append(pearson(bb[b, k], gval_n[b, j]))
    T = np.vstack(Ts); P = np.vstack(Ps)
    K = T.shape[1]
    per_state = []
    for k in range(K):
        if (T[:, k] > 1e-6).sum() >= 3:
            per_state.append(pearson(T[:, k], P[:, k]))
    return {
        "theta_pearson_global": pearson(T, P),
        "theta_mae": float(np.abs(T - P).mean()),
        "theta_pearson_perstate_mean": float(np.nanmean(per_state)) if per_state else float("nan"),
        "gep_pearson_mean": float(np.nanmean(geps)) if geps else float("nan"),
    }



def main():
    p = argparse.ArgumentParser(description="Phase 2 — train theta+dM encoder")
    p.add_argument("--pseudobulk", required=True)
    p.add_argument("--phase0", required=True)
    p.add_argument("--phase1", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--w_elbo", type=float, default=1.0,
                   help="scale on the ELBO part (recon + KL); 0 = supervision-only (discriminative).")
    p.add_argument("--w_gep", type=float, default=1.0)
    p.add_argument("--w_theta", type=float, default=10.0)
    p.add_argument("--sigma_prior", type=float, default=1.0)
    p.add_argument("--kl_dM_weight", type=float, default=1.0,
                   help="extra multiplier on KL_dM; <1 relaxes the pull dM->0 (activates dM)")
    p.add_argument("--sigma_k_scale", type=float, default=1.0,
                   help="multiply phase-1 sigma_k (the ΔM prior scale); >1 widens the prior "
                        "so ΔM is not forced to 0. Attacks the ΔM-collapse at its root.")
    p.add_argument("--kl_per_read", action="store_true",
                   help="divide KL by N (reads) to match the per-read-normalised recon: proper ELBO ratio.")
    p.add_argument("--kl_warmup_frac", type=float, default=0.33)
    p.add_argument("--m_lr_scale", type=float, default=0.0,
                   help="0 = M frozen; >0 trains M at lr*scale.")
    p.add_argument("--tcga_bulk", default="",
                   help="path to prep_tcga_bulk .npz; if set, mixes recon-only bulk into training.")
    p.add_argument("--w_bulk", type=float, default=1.0,
                   help="weight on the unsupervised bulk recon (+KL) loss.")
    p.add_argument("--bulk_batch_size", type=int, default=0,
                   help="bulk minibatch size; 0 = match --batch_size (≈50/50 mix per step).")
    p.add_argument("--val_frac", type=float, default=0.2, help="patient hold-out fraction")
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    cfg = p.parse_args()

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    log.info(f"Device: {device}")

    p0 = load_phase0(cfg.phase0)
    pb = ad.read_h5ad(cfg.pseudobulk)
    if list(pb.uns["cell_types"]) != p0["types"]:
        raise ValueError("Pseudobulk cell_types disagree with Phase-0 ordering")
    K, V, T = p0["K"], p0["V"], p0["T"]
    A = int(pb.uns["gep_true_max_active"])
    X = np.asarray(pb.X, dtype=np.float32)
    theta = np.asarray(pb.obsm["theta_true"], dtype=np.float32)
    bidx = pb.obs["batch_idx"].to_numpy().astype(np.int64)
    gidx = np.asarray(pb.obsm["gep_true_active_idx"], dtype=np.int64)
    gval = np.asarray(pb.obsm["gep_true_active_gep"], dtype=np.float32).reshape(-1, A, V)
    patients = pb.obs["patient"].astype(str).to_numpy()
    log.info(f"Pseudobulk: {X.shape[0]} mixtures | A={A} active slots")

    M_seed = seed_M_from_profiles(p0["profiles"])
    model = ETMDeconvC2(p0["alpha"], p0["rho"], p0["lambda"], M_seed, sce2tm=p0.get("sce2tm", False), tau=p0.get("tau", 0.2), gene_baseline=p0.get("gene_baseline"), topic_gene_override=p0.get("topic_gene_override")).to(device)
    state = torch.load(Path(cfg.phase1) / "c2_phase1.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    log.info(f"Loaded Phase-1 M + sigma_k (sigma in [{model.sigma_k.min():.3f}, "
             f"{model.sigma_k.max():.3f}])")
    for name, pa in model.named_parameters():
        pa.requires_grad = (name == "M" and cfg.m_lr_scale > 0)
    sigma_k = model.sigma_k.detach() * cfg.sigma_k_scale
    if cfg.sigma_k_scale != 1.0:
        log.info(f"sigma_k_scale={cfg.sigma_k_scale}: widened prior to "
                 f"[{sigma_k.min():.3f}, {sigma_k.max():.3f}] (unlocks ΔM from collapse)")

    encoder = C2Encoder(V, K, T, hidden=cfg.hidden, dropout=cfg.dropout).to(device)

    uniq = np.array(sorted(set(patients)))
    rng = np.random.RandomState(cfg.seed)
    n_val = max(1, int(round(len(uniq) * cfg.val_frac)))
    val_pat = set(rng.choice(uniq, size=n_val, replace=False).tolist())
    val_mask = np.array([p in val_pat for p in patients])
    log.info(f"Patient split: {len(uniq)-n_val} train / {n_val} val patients | "
             f"{(~val_mask).sum()} train / {val_mask.sum()} val samples")

    def make_loader(mask, shuffle):
        ds = MixtureDataset(X[mask], theta[mask], bidx[mask], gidx[mask], gval[mask])
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=False)

    train_loader = make_loader(~val_mask, True)
    val_loader = make_loader(val_mask, False)

    bulk_loader = None
    if cfg.tcga_bulk:
        npz = np.load(cfg.tcga_bulk, allow_pickle=True)
        Xbulk = np.asarray(npz["counts"], dtype=np.float32)
        if "genes" in npz and list(npz["genes"]) != p0["genes"]:
            raise ValueError("TCGA bulk gene order disagrees with Phase-0 genes.json")
        if Xbulk.shape[1] != V:
            raise ValueError(f"TCGA bulk V={Xbulk.shape[1]} != model V={V}")
        bbs = cfg.bulk_batch_size or cfg.batch_size
        bulk_loader = DataLoader(BulkDataset(Xbulk), batch_size=bbs,
                                 shuffle=True, drop_last=True)
        log.info(f"TCGA bulk: {Xbulk.shape[0]} samples (recon-only, NO lambda) | "
                 f"w_bulk={cfg.w_bulk} bulk_bs={bbs}")

    params = list(encoder.parameters())
    if cfg.m_lr_scale > 0:
        opt = torch.optim.Adam([
            {"params": encoder.parameters(), "lr": cfg.lr},
            {"params": [model.M], "lr": cfg.lr * cfg.m_lr_scale},
        ])
    else:
        opt = torch.optim.Adam(params, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr*0.01)

    sigma_prior_t = torch.tensor(cfg.sigma_prior ** 2, device=device)
    sig2_k = (sigma_k ** 2).view(1, K, 1)

    history, best, no_improve, best_gep = [], -1.0, 0, -1.0
    warm_end = max(1, int(cfg.epochs * cfg.kl_warmup_frac))
    bulk_iter = iter(bulk_loader) if bulk_loader is not None else None
    for epoch in range(cfg.epochs):
        encoder.train()
        beta_t = min(1.0, epoch / warm_end)
        agg = {"rec": 0, "kld": 0, "klm": 0, "gep": 0, "ce": 0, "n": 0,
               "brec": 0, "bdM": 0, "bn": 0}
        for Xb, th, bi, gi, gv in train_loader:
            Xb, th, bi = Xb.to(device), th.to(device), bi.to(device)
            gi, gv = gi.to(device), gv.to(device)
            N = Xb.sum(-1).clamp(min=1)
            opt.zero_grad()
            o = forward_batch(model, encoder, Xb, bi, sample=True)
            rec = (-(Xb * torch.log(o["r"] + 1e-12)).sum(-1) / N).mean()
            kl_div = N if cfg.kl_per_read else torch.ones_like(N)
            kld = (kl_gaussian(o["mu_d"], o["logvar_d"], sigma_prior_t).sum(-1) / kl_div).mean()
            klm = kl_gaussian(o["mu_dM"], o["logvar_dM"], sig2_k).sum(dim=(1, 2)).mean()
            gep = gep_loss(o["beta_bio"], th, gi, gv)
            ce = -(th * torch.log(o["theta_mean"] + 1e-12)).sum(-1).mean()
            loss = (cfg.w_elbo * (rec + beta_t * (kld + cfg.kl_dM_weight * klm))
                    + cfg.w_gep * gep + cfg.w_theta * ce)

            if bulk_iter is not None:
                try:
                    Xk = next(bulk_iter)
                except StopIteration:
                    bulk_iter = iter(bulk_loader); Xk = next(bulk_iter)
                Xk = Xk.to(device)
                Nk = Xk.sum(-1).clamp(min=1)
                ob = forward_bulk(model, encoder, Xk)
                brec = (-(Xk * torch.log(ob["r"] + 1e-12)).sum(-1) / Nk).mean()
                bkl_div = Nk if cfg.kl_per_read else torch.ones_like(Nk)
                bkld = (kl_gaussian(ob["mu_d"], ob["logvar_d"], sigma_prior_t).sum(-1) / bkl_div).mean()
                bklm = kl_gaussian(ob["mu_dM"], ob["logvar_dM"], sig2_k).sum(dim=(1, 2)).mean()
                loss = loss + cfg.w_bulk * (brec + beta_t * (bkld + cfg.kl_dM_weight * bklm))
                bb = Xk.shape[0]
                agg["brec"] += brec.item()*bb
                agg["bdM"] += ob["mu_dM"].abs().mean().item()*bb
                agg["bn"] += bb

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            opt.step()
            b = Xb.shape[0]
            agg["rec"] += rec.item()*b; agg["kld"] += kld.item()*b
            agg["klm"] += klm.item()*b; agg["gep"] += gep.item()*b
            agg["ce"] += ce.item()*b; agg["n"] += b
        sched.step()
        n = agg["n"]
        val = evaluate(model, encoder, val_loader, device)
        bn = max(agg["bn"], 1)
        rec_row = {"epoch": epoch, "beta_t": beta_t,
                   "recon": agg["rec"]/n, "kl_delta": agg["kld"]/n,
                   "kl_dM": agg["klm"]/n, "gep": agg["gep"]/n, "ce": agg["ce"]/n,
                   "bulk_recon": agg["brec"]/bn, "bulk_dM_abs": agg["bdM"]/bn, **val}
        history.append(rec_row)
        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            blk = (f" | b_rec={rec_row['bulk_recon']:.3f} |dM|={rec_row['bulk_dM_abs']:.3f}"
                   if bulk_iter is not None else "")
            log.info(f"Ep {epoch:3d} | rec={rec_row['recon']:.3f} klD={rec_row['kl_delta']:.2f} "
                     f"klM={rec_row['kl_dM']:.2f} gep={rec_row['gep']:.3f} ce={rec_row['ce']:.3f}{blk} "
                     f"|| val θ_r={val['theta_pearson_global']:.3f} θ_mae={val['theta_mae']:.4f} "
                     f"GEP_r={val['gep_pearson_mean']:.3f}")
        gep_score = val["gep_pearson_mean"]
        if not np.isnan(gep_score) and gep_score > best_gep + 1e-4:
            best_gep = gep_score
            torch.save(encoder.state_dict(), out / "encoder_best_gep.pt")
        score = val["theta_pearson_global"]
        if np.isnan(score):
            score = -1.0
        if score > best + 1e-4:
            best, no_improve = score, 0
            torch.save(encoder.state_dict(), out / "encoder_best.pt")
        else:
            no_improve += 1
            if cfg.patience and no_improve >= cfg.patience:
                log.info(f"Early stop at epoch {epoch} (best val θ_pearson={best:.3f})")
                break

    torch.save(encoder.state_dict(), out / "encoder_last.pt")
    torch.save(model.state_dict(), out / "c2_phase2.pt")
    (out / "training_log.json").write_text(json.dumps(history, indent=2))
    final = history[-1]
    report = {
        "n_mixtures": int(X.shape[0]),
        "train_patients": int(len(uniq) - n_val), "val_patients": int(n_val),
        "best_val_theta_pearson": float(best),
        "best_val_gep_pearson": float(best_gep),
        "final": {k: final[k] for k in
                  ["theta_pearson_global", "theta_mae",
                   "theta_pearson_perstate_mean", "gep_pearson_mean"]},
    }
    (out / "phase2_report.json").write_text(json.dumps(report, indent=2))
    log.info(f"Saved Phase-2 outputs to {out}")
    log.info(f"  best val theta_pearson={best:.3f} | final {report['final']}")


if __name__ == "__main__":
    main()
