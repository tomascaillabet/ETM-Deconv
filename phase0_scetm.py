"""Phase 0: train scETM on scRNA-seq to learn the model's foundational components.

Outputs:
    alpha (T,L) topics, rho (L,V) gene embeddings, lambda (S,V) batch effects, the
    gene order, and the per-cell-type topic profiles that seed the M matrix, plus
    topic-gene / cell-topic matrices for visualisation.

Usage:
    python phase0_scetm.py --scrna scrna.h5ad --gene2vec gene2vec_dim_200_iter_9.txt --cell_type_key cell_type --batch_key batch --T 500 --L 200 --epochs 300 --output_dir ./phase0
"""

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase0")



@dataclass
class Phase0Config:
    scrna_path: str = ""
    gene2vec_path: str = ""
    cell_type_key: str = "cell_type"
    batch_key: str = "batch"

    T: int = 500
    L: int = 200
    hidden: int = 256
    dropout: float = 0.1

    epochs: int = 300
    lr: float = 1e-3
    batch_size: int = 256
    kl_weight: float = 1.0
    warmup_frac: float = 0.2
    normalize_kl: bool = False
    no_lambda: bool = False
    center_lambda: bool = False
    rho_path: str = ""
    rho_genes: str = ""
    random_rho: bool = False
    learn_rho: bool = False
    w_rho_prior: float = 0.0
    log_input: bool = True
    seed: int = 0
    patience: int = 30
    min_delta: float = 1e-3

    output_dir: str = "./phase0_output"



def load_gene2vec(path: str, gene_names: list, L: int) -> tuple:
    """
    Load Gene2Vec embeddings and align to gene_names.

    Returns:
        rho:   (L, V) float32 tensor, columns normalised to unit norm
        found: list[bool] of length V, True if gene had an embedding
    """
    emb = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < L + 1:
                continue
            emb[parts[0]] = np.asarray(parts[1 : L + 1], dtype=np.float32)

    V = len(gene_names)
    rho = np.zeros((L, V), dtype=np.float32)
    found = [False] * V
    for i, g in enumerate(gene_names):
        if g in emb:
            rho[:, i] = emb[g]
            found[i] = True

    n_found = sum(found)
    log.info(f"Gene2Vec: {n_found}/{V} genes found ({100 * n_found / V:.1f}%)")

    norms = np.linalg.norm(rho, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    rho = rho / norms
    return torch.tensor(rho, dtype=torch.float32), found



class SCRNADataset(Dataset):
    """
    Wraps a scRNA count matrix with cell-type and batch labels.

    Stores counts as a dense float32 tensor (load in chunks if memory-bound;
    for the typical subsampled atlas this fits comfortably).
    """

    def __init__(self, X, batch_idx, type_idx):
        self._sparse = not isinstance(X, np.ndarray)
        self.X = X
        self.batch_idx = torch.as_tensor(batch_idx, dtype=torch.long)
        self.type_idx = torch.as_tensor(type_idx, dtype=torch.long)
        self.n = self.batch_idx.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._sparse:
            row = np.asarray(self.X[i].todense()).ravel().astype(np.float32)
        else:
            row = self.X[i].astype(np.float32)
        return {
            "X": torch.from_numpy(row),
            "batch_idx": self.batch_idx[i],
            "type_idx": self.type_idx[i],
            "cell_pos": i,
        }



class TopicEncoder(nn.Module):
    """Amortised encoder producing the logistic-normal posterior over delta_c."""

    def __init__(self, V: int, T: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(V, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden, T)
        self.logvar = nn.Linear(hidden, T)

    def forward(self, x_tilde):
        h = self.backbone(x_tilde)
        return self.mu(h), self.logvar(h)



class SCETM(nn.Module):
    """
    scETM: logistic-normal topic model with Gene2Vec decoder and batch effect.
    """

    def __init__(self, V: int, T: int, L: int, n_batch: int,
                 rho: torch.Tensor, hidden: int = 256, dropout: float = 0.1,
                 learn_rho: bool = False, w_rho_prior: float = 0.0):
        super().__init__()
        self.V, self.T, self.L = V, T, L
        self._learn_rho = learn_rho
        self._w_rho_prior = w_rho_prior
        if learn_rho:
            self.rho = nn.Parameter(rho.clone())
            self.register_buffer("rho_prior", rho.clone())
        else:
            self.register_buffer("rho", rho)
        self.alpha = nn.Parameter(torch.randn(T, L) * 0.01)
        self.lambda_batch = nn.Parameter(torch.zeros(n_batch, V))
        self.encoder = TopicEncoder(V, T, hidden, dropout)

    def rho_prior_loss(self) -> torch.Tensor:
        """Per-gene squared-L2 deviation of rho from its prior (mean over genes)."""
        if not self._learn_rho:
            return torch.zeros((), device=self.alpha.device)
        return ((self.rho - self.rho_prior) ** 2).sum(0).mean()

    def topic_gene_logits(self) -> torch.Tensor:
        """beta logits per topic: (T, V) = alpha @ rho (dot-product)."""
        return self.alpha @ self.rho

    def decode(self, psi: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        r_c = softmax(psi @ alpha @ rho + lambda_s).

        Args:
            psi:       (B, T) topic loadings (simplex)
            batch_idx: (B,) batch indices
        Returns:
            r: (B, V) predicted gene proportions
        """
        topic_mix = psi @ self.alpha
        logits = topic_mix @ self.rho
        logits = logits + self.lam()[batch_idx]
        return F.softmax(logits, dim=-1)

    def lam(self) -> torch.Tensor:
        """Batch term, mean-centered across batches when _center_lambda (so it
        carries only batch-specific technical variation; the shared baseline is
        forced into alpha)."""
        if getattr(self, "_center_lambda", False):
            return self.lambda_batch - self.lambda_batch.mean(0, keepdim=True)
        return self.lambda_batch

    def forward(self, X, batch_idx, beta_kl: float = 1.0,
                normalize_kl: bool = False):
        """
        Full forward pass with the ELBO.

        Args:
            X:            (B, V) raw counts
            batch_idx:    (B,)
            beta_kl:      KL warmup weight
            normalize_kl: if True, divide KL by T so it's per-dimension
                          (keeps KL ~1-2 nats, comparable to normalized recon ~8 nats)
        Returns:
            dict with loss, recon, kl, psi (detached), mu_delta (detached)
        """
        B = X.shape[0]
        N = X.sum(dim=-1, keepdim=True).clamp(min=1)
        x_tilde = X / N
        if self._log_input:
            x_tilde = torch.log1p(x_tilde * 1e4)

        mu, logvar = self.encoder(x_tilde)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        delta = mu + std * eps
        psi = F.softmax(delta, dim=-1)

        r = self.decode(psi, batch_idx)

        recon = -(X * torch.log(r + 1e-12)).sum(dim=-1) / N.squeeze(-1)
        recon = recon.mean()

        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl = kl_per_dim.sum(dim=-1).mean()
        if normalize_kl:
            kl = kl / self.T

        loss = recon + beta_kl * kl
        return {
            "loss": loss,
            "recon": recon.detach(),
            "kl": kl.detach(),
            "psi": psi.detach(),
            "mu_delta": mu.detach(),
        }

    _log_input: bool = True



def kl_warmup(epoch: int, n_epochs: int, frac: float) -> float:
    end = max(int(n_epochs * frac), 1)
    return min(1.0, epoch / end)


def train(model: SCETM, loader: DataLoader, cfg: Phase0Config,
          device: torch.device) -> list:
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
    )
    history = []
    best = float("inf")
    no_improve = 0

    model.train()
    for epoch in range(cfg.epochs):
        beta = cfg.kl_weight * kl_warmup(epoch, cfg.epochs, cfg.warmup_frac)
        tot_loss = tot_recon = tot_kl = 0.0
        n_batches = 0
        for batch in loader:
            X = batch["X"].to(device)
            bidx = batch["batch_idx"].to(device)
            opt.zero_grad()
            out = model(X, bidx, beta_kl=beta, normalize_kl=cfg.normalize_kl)
            loss = out["loss"] + model._w_rho_prior * model.rho_prior_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += out["loss"].item()
            tot_recon += out["recon"].item()
            tot_kl += out["kl"].item()
            n_batches += 1
        sched.step()

        avg_loss = tot_loss / max(n_batches, 1)
        rec = {
            "epoch": epoch,
            "loss": avg_loss,
            "recon": tot_recon / max(n_batches, 1),
            "kl": tot_kl / max(n_batches, 1),
            "beta_kl": beta,
            "lr": sched.get_last_lr()[0],
        }
        history.append(rec)
        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            extra = ""
            if model._learn_rho:
                extra = f" | rho_drift={model.rho_prior_loss().item():.4f}"
            log.info(
                f"Epoch {epoch:4d}/{cfg.epochs} | loss={avg_loss:.4f} "
                f"| recon={rec['recon']:.4f} | kl={rec['kl']:.4f} | beta={beta:.3f}{extra}"
            )

        if beta >= 1.0:
            if avg_loss < best - cfg.min_delta:
                best = avg_loss
                no_improve = 0
            else:
                no_improve += 1
                if cfg.patience and no_improve >= cfg.patience:
                    log.info(f"Early stop at epoch {epoch} (no improvement "
                             f"for {cfg.patience} epochs)")
                    break
    return history



@torch.no_grad()
def infer_loadings(model: SCETM, loader: DataLoader, device: torch.device,
                   n_cells: int) -> np.ndarray:
    """Return psi_c (n_cells, T) using the posterior mean mu (no sampling)."""
    model.eval()
    psi_all = np.zeros((n_cells, model.T), dtype=np.float32)
    for batch in loader:
        X = batch["X"].to(device)
        pos = batch["cell_pos"].numpy()
        N = X.sum(dim=-1, keepdim=True).clamp(min=1)
        x_tilde = X / N
        if model._log_input:
            x_tilde = torch.log1p(x_tilde * 1e4)
        mu, _ = model.encoder(x_tilde)
        psi = F.softmax(mu, dim=-1).cpu().numpy()
        psi_all[pos] = psi
    return psi_all



def relabeling_diagnostics(psi: np.ndarray, type_idx: np.ndarray,
                           type_names: list) -> tuple:
    """
    For each cell, compare its annotated type with the type whose mean topic
    profile is closest (cosine similarity in topic space).

    Returns:
        diag_df: per-cell DataFrame with annotated/implied labels and prob
        profiles: (K, T) mean topic profile per annotated type (seed for M)
    """
    K = len(type_names)
    T = psi.shape[1]

    profiles = np.zeros((K, T), dtype=np.float32)
    for k in range(K):
        mask = type_idx == k
        if mask.sum() > 0:
            profiles[k] = psi[mask].mean(axis=0)

    psi_n = psi / (np.linalg.norm(psi, axis=1, keepdims=True) + 1e-12)
    prof_n = profiles / (np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-12)

    sims = psi_n @ prof_n.T
    probs = F.softmax(torch.tensor(sims) / 0.1, dim=-1).numpy()
    implied = sims.argmax(axis=1)

    diag = pd.DataFrame({
        "annotated_type": [type_names[i] for i in type_idx],
        "implied_type": [type_names[i] for i in implied],
        "implied_prob": probs[np.arange(len(implied)), implied],
        "annotated_prob": probs[np.arange(len(type_idx)), type_idx],
        "agree": (type_idx == implied),
    })
    return diag, profiles



def save_table(df: pd.DataFrame, path_base: str):
    """Save as parquet if pyarrow is available, else CSV."""
    try:
        df.to_parquet(path_base + ".parquet", index=False)
        return path_base + ".parquet"
    except Exception:
        df.to_csv(path_base + ".csv", index=False)
        return path_base + ".csv"


def save_outputs(model: SCETM, cfg: Phase0Config, genes: list,
                 batch_names: list, type_names: list,
                 psi: np.ndarray, cell_meta: pd.DataFrame,
                 history: list, g2v_found: list, device):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model.eval()

    alpha = model.alpha.detach().cpu().numpy()
    rho = model.rho.detach().cpu().numpy()
    lam = model.lam().detach().cpu().numpy()
    np.save(out / "topics_alpha.npy", alpha)
    np.save(out / "gene2vec_rho.npy", rho)
    np.save(out / "lambda_batch.npy", lam)

    with open(out / "genes.json", "w") as f:
        json.dump(genes, f)
    with open(out / "batch_names.json", "w") as f:
        json.dump(batch_names, f)
    with open(out / "cell_type_names.json", "w") as f:
        json.dump(type_names, f)
    with open(out / "gene2vec_found.json", "w") as f:
        json.dump([bool(x) for x in g2v_found], f)

    torch.save(model.encoder.state_dict(), out / "encoder_state.pt")
    torch.save(model.state_dict(), out / "scetm_full_state.pt")

    cfg_dict = asdict(cfg)
    cfg_dict.update({"V": len(genes), "n_batch": len(batch_names),
                     "n_types": len(type_names)})
    with open(out / "model_config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    diag, profiles = relabeling_diagnostics(
        psi, cell_meta["type_idx"].to_numpy(), type_names
    )
    np.save(out / "topic_profiles_per_type.npy", profiles)
    diag_path = save_table(diag, str(out / "relabeling_diagnostics"))
    agree_rate = diag["agree"].mean()
    log.info(f"Re-labeling agreement: {100 * agree_rate:.1f}% "
             f"({(~diag['agree']).sum()} cells flagged)")

    np.save(out / "cell_topic_loadings.npy", psi)
    _ = save_table(cell_meta, str(out / "cell_metadata"))

    with torch.no_grad():
        tg = F.softmax(model.topic_gene_logits(), dim=-1).cpu().numpy()
    np.save(out / "topic_gene_matrix.npy", tg)

    with open(out / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)

    manifest = {
        "core_for_phases_1_3": {
            "topics_alpha.npy": "Foundational topics alpha (T, L). FROZEN input to phases 1-3.",
            "gene2vec_rho.npy": "Gene2Vec rho (L, V). Reuse for identical gene order.",
            "lambda_batch.npy": "Batch effect lambda_s (S, V). Propagated to pseudobulk.",
            "genes.json": "Ordered list of V genes. Phases 1-3 MUST use this order.",
            "batch_names.json": "Maps batch index -> batch name.",
            "cell_type_names.json": "Maps cell-type index -> name.",
            "topic_profiles_per_type.npy": "Mean topic profile per cell type (K, T). SEED for M in phase 1.",
            "encoder_state.pt": "Phase-0 encoder weights (not reused by later phases; for reference).",
            "scetm_full_state.pt": "Full Phase-0 model state.",
            "model_config.json": "Architecture and hyperparameters, plus V, n_batch, n_types.",
        },
        "for_visualisation": {
            "cell_topic_loadings.npy": "Per-cell topic loadings psi_c (n_cells, T), posterior mean.",
            "cell_metadata.(parquet|csv)": "Per-cell metadata aligned with cell_topic_loadings rows.",
            "topic_gene_matrix.npy": "softmax(alpha @ rho) (T, V): gene distribution per topic.",
            "relabeling_diagnostics.(parquet|csv)": "Annotated vs implied label per cell (diagnostic only).",
            "gene2vec_found.json": "Per-gene bool: had a Gene2Vec embedding.",
            "training_log.json": "Loss/recon/kl curves per epoch.",
        },
        "notes": [
            "No cells were relabeled. relabeling_diagnostics is for inspection only.",
            "topic_profiles_per_type.npy seeds M: M[:,k] init from logit of profile k.",
            "psi_c uses the posterior mean (mu) without sampling, for determinism.",
        ],
    }
    with open(out / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Saved Phase-0 outputs to {out.resolve()}")
    log.info(f"Re-labeling table: {Path(diag_path).name}")



def main():
    p = argparse.ArgumentParser(description="ETM-Deconv Phase 0 (scETM on scRNA)")
    p.add_argument("--scrna", required=True)
    p.add_argument("--gene2vec", required=True)
    p.add_argument("--cell_type_key", default="cell_type")
    p.add_argument("--batch_key", default="batch")
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--L", type=int, default=200)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--warmup_frac", type=float, default=0.2)
    p.add_argument("--normalize_kl", action="store_true",
                   help="Divide KL by T to match normalized-recon scale (~8 nats).")
    p.add_argument("--no_lambda", action="store_true",
                   help="Disable batch term (lambda=0) so topics absorb the expression baseline.")
    p.add_argument("--center_lambda", action="store_true",
                   help="Mean-center lambda across batches (technical only); baseline forced into alpha.")
    p.add_argument("--rho_path", default="", help="Precomputed gene embedding .npy (L,V), e.g. scGPT.")
    p.add_argument("--rho_genes", default="", help="Gene-order json matching rho_path/random_rho columns.")
    p.add_argument("--random_rho", action="store_true", help="Random rho init (scETM: learn embeddings from scratch).")
    p.add_argument("--learn_rho", action="store_true", help="Make rho trainable (anchored to its init).")
    p.add_argument("--w_rho_prior", type=float, default=0.0, help="Weight on ||rho - rho_prior||^2 (per-gene).")
    p.add_argument("--no_log_input", action="store_true",
                   help="Disable log1p normalisation of encoder input.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--output_dir", default="./phase0_output")
    args = p.parse_args()

    cfg = Phase0Config(
        scrna_path=args.scrna, gene2vec_path=args.gene2vec,
        cell_type_key=args.cell_type_key, batch_key=args.batch_key,
        T=args.T, L=args.L, hidden=args.hidden, dropout=args.dropout,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        warmup_frac=args.warmup_frac, normalize_kl=args.normalize_kl,
        no_lambda=args.no_lambda, center_lambda=args.center_lambda,
        rho_path=args.rho_path, rho_genes=args.rho_genes, random_rho=args.random_rho,
        learn_rho=args.learn_rho, w_rho_prior=args.w_rho_prior,
        log_input=not args.no_log_input,
        seed=args.seed, patience=args.patience, output_dir=args.output_dir,
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    import anndata as ad
    import scipy.sparse as sp
    log.info(f"Loading {cfg.scrna_path}")
    adata = ad.read_h5ad(cfg.scrna_path)

    genes_all = list(adata.var_names)
    if cfg.random_rho:
        genes = json.loads(Path(cfg.rho_genes).read_text())
        gi = {g: i for i, g in enumerate(genes_all)}
        missing = [g for g in genes if g not in gi]
        if missing:
            raise ValueError(f"{len(missing)} rho_genes absent from scRNA")
        keep = [gi[g] for g in genes]
        rho_np = np.random.RandomState(cfg.seed).randn(cfg.L, len(genes)).astype(np.float32)
        rho_np /= np.linalg.norm(rho_np, axis=0, keepdims=True)
        rho = torch.tensor(rho_np, dtype=torch.float32)
        found = [True] * len(genes)
        log.info(f"random rho init (scETM-style, learnable, no prior): L={cfg.L}, {len(genes)} genes")
    elif cfg.rho_path:
        genes = json.loads(Path(cfg.rho_genes).read_text())
        rho_np = np.load(cfg.rho_path)
        assert rho_np.shape[1] == len(genes), f"rho {rho_np.shape} vs {len(genes)} genes"
        gi = {g: i for i, g in enumerate(genes_all)}
        missing = [g for g in genes if g not in gi]
        if missing:
            raise ValueError(f"{len(missing)} rho_genes absent from scRNA")
        keep = [gi[g] for g in genes]
        rho = torch.tensor(rho_np, dtype=torch.float32)
        cfg.L = rho.shape[0]
        found = (np.abs(rho_np).sum(0) > 0).tolist()
        log.info(f"Using precomputed embedding {cfg.rho_path}: L={cfg.L}, {len(genes)} genes "
                 f"({int(np.sum(found))} with nonzero embedding)")
    else:
        rho_full, found = load_gene2vec(cfg.gene2vec_path, genes_all, cfg.L)
        keep = [i for i, f in enumerate(found) if f]
        genes = [genes_all[i] for i in keep]
        log.info(f"Keeping {len(genes)}/{len(genes_all)} genes with Gene2Vec coverage")
        rho = rho_full[:, keep].contiguous()

    X = adata.X
    X = X[:, keep]
    if sp.issparse(X):
        X = X.tocsr()
    else:
        X = np.asarray(X)[:, :]

    type_series = adata.obs[cfg.cell_type_key].astype("category").cat.remove_unused_categories()
    batch_series = adata.obs[cfg.batch_key].astype("category").cat.remove_unused_categories()
    type_names = list(type_series.cat.categories)
    batch_names = list(batch_series.cat.categories)
    type_idx = type_series.cat.codes.to_numpy()
    batch_idx = batch_series.cat.codes.to_numpy()
    log.info(f"{len(type_names)} cell types, {len(batch_names)} batches, "
             f"{X.shape[0]} cells")

    ds = SCRNADataset(X, batch_idx, type_idx)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=0, drop_last=False)

    model = SCETM(V=len(genes), T=cfg.T, L=cfg.L, n_batch=len(batch_names),
                  rho=rho, hidden=cfg.hidden, dropout=cfg.dropout,
                  learn_rho=cfg.learn_rho, w_rho_prior=cfg.w_rho_prior).to(device)
    if cfg.learn_rho:
        log.info(f"learn_rho: rho trainable, anchored via w_rho_prior={cfg.w_rho_prior}")
    model._log_input = cfg.log_input
    model._center_lambda = cfg.center_lambda
    if cfg.center_lambda:
        log.info("center_lambda: lambda mean-zero across batches (technical only); baseline -> alpha")
    if cfg.no_lambda:
        model.lambda_batch.requires_grad_(False)
        log.info("no_lambda: batch term disabled (lambda frozen at 0)")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {n_params:,} trainable parameters "
             f"(T={cfg.T}, L={cfg.L}, V={len(genes)})")

    history = train(model, loader, cfg, device)

    eval_loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=0)
    psi = infer_loadings(model, eval_loader, device, n_cells=X.shape[0])

    cell_meta = pd.DataFrame({
        "cell_id": list(adata.obs_names),
        "cell_type": [type_names[i] for i in type_idx],
        "type_idx": type_idx,
        "batch": [batch_names[i] for i in batch_idx],
        "batch_idx": batch_idx,
        "library_size": np.asarray(X.sum(axis=1)).ravel(),
    })

    save_outputs(model, cfg, genes, batch_names, type_names,
                 psi, cell_meta, history, found, device)


if __name__ == "__main__":
    main()
