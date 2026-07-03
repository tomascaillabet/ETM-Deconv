"""Shared C2 model for ETM-Deconv: the simplex-mixture decoder and the two-head encoder.
Imported by phases 1-3. alpha, rho and lambda come frozen from phase 0; this module
adds the per-state topic loadings M and per-state variability sigma_k.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("c2_model")



def load_phase0(phase0_dir: str) -> dict:
    """
    Load the FROZEN Phase-0 artifacts that Phases 1-3 build on.

    Returns a dict with:
        alpha    (T, L) float32   foundational topics
        rho      (L, V) float32   Gene2Vec embeddings
        lambda   (S, V) float32   per-batch technical effect
        profiles (K, T) float32   mean topic profile per cell type (seed for M)
        genes    list[str]        ordered gene names (length V)
        batches  list[str]        ordered batch names (length S)
        types    list[str]        ordered cell-type names (length K)
        config   dict             Phase-0 model_config.json (has V, T, L, ...)
    """
    d = Path(phase0_dir)
    alpha = np.load(d / "topics_alpha.npy")
    rho = np.load(d / "gene2vec_rho.npy")
    lam = np.load(d / "lambda_batch.npy")
    profiles = np.load(d / "topic_profiles_per_type.npy")
    genes = json.loads((d / "genes.json").read_text())
    batches = json.loads((d / "batch_names.json").read_text())
    types = json.loads((d / "cell_type_names.json").read_text())
    config = json.loads((d / "model_config.json").read_text())

    T, L = alpha.shape
    S, V = lam.shape
    K = profiles.shape[0]
    assert rho.shape == (L, V), f"rho {rho.shape} != ({L},{V})"
    assert profiles.shape == (K, T), f"profiles {profiles.shape} != ({K},{T})"
    assert len(genes) == V, f"genes {len(genes)} != V {V}"
    assert len(batches) == S, f"batches {len(batches)} != S {S}"
    assert len(types) == K, f"types {len(types)} != K {K}"
    log.info(f"Phase-0 loaded: T={T}, L={L}, V={V}, K={K}, S={S}")

    M_phase0 = None
    if (d / "M_phase0.npy").exists():
        M_phase0 = np.load(d / "M_phase0.npy")
        assert M_phase0.shape == (T, K), f"M_phase0 {M_phase0.shape} != ({T},{K})"
        log.info("Phase-0 learned M found (M_phase0.npy) — use it to init M")

    sce2tm = bool(config.get("sce2tm_decoder", False))
    tau = float(config.get("tau", 0.2))
    if not sce2tm and (d / "scetm_full_state.pt").exists():
        import torch as _t
        st = _t.load(d / "scetm_full_state.pt", map_location="cpu", weights_only=True)
        sce2tm = "decoder_bn.running_mean" in st
    gene_baseline = None
    topic_gene_override = None
    if (d / "topic_gene_override.npy").exists():
        topic_gene_override = np.load(d / "topic_gene_override.npy")
        sce2tm = True
        log.info(f"topic_gene_override loaded {topic_gene_override.shape} (scE2TM library B)")
    if sce2tm:
        gbp = d / "gene_log_baseline.npy"
        if gbp.exists():
            gene_baseline = np.load(gbp)
            log.info(f"scE2TM decoder: B (tau={tau}) + gene_log_baseline")
        else:
            log.warning("scE2TM phase0 but gene_log_baseline.npy missing — GEP will be flat!")

    return {
        "alpha": alpha, "rho": rho, "lambda": lam, "profiles": profiles,
        "M_phase0": M_phase0,
        "genes": genes, "batches": batches, "types": types, "config": config,
        "T": T, "L": L, "V": V, "K": K, "S": S,
        "sce2tm": sce2tm, "tau": tau, "gene_baseline": gene_baseline, "topic_gene_override": topic_gene_override,
    }


def seed_M_from_profiles(profiles: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Seed M (T, K) so that softmax(M[:,k]) == profiles[k] (up to eps).

    logit(pi)_t = log(pi_t) - mean_t(log(pi))   (centered logit; softmax-invariant
    to the additive constant, so this exactly inverts softmax).

    Args:
        profiles: (K, T) mean topic profile per type (rows sum ~1; may have zeros).
    Returns:
        M: (T, K) float32.
    """
    K, T = profiles.shape
    M = np.zeros((T, K), dtype=np.float32)
    for k in range(K):
        p = profiles[k].astype(np.float64) + eps
        p = p / p.sum()
        logit = np.log(p) - np.log(p).mean()
        M[:, k] = logit.astype(np.float32)
    recon = np.exp(M) / np.exp(M).sum(axis=0, keepdims=True)
    err = np.abs(recon.T - profiles / profiles.sum(axis=1, keepdims=True).clip(1e-12)).max()
    log.info(f"M seed reconstruction max abs error: {err:.2e}")
    return M



class ETMDeconvC2(nn.Module):
    """
    C2 simplex-mixture model.

    Frozen (from Phase 0, registered as buffers): alpha, rho, lambda_batch.
    Trainable: M (T, K), log_sigma_k (K,).

    The decoder is the single authoritative implementation of the simplex
    mixture; Phase 1 uses the `decode_pure` fast path (theta = e_k, dM = 0),
    Phases 2-3 use `decode` (full theta, dM).
    """

    def __init__(self, alpha, rho, lambda_batch, M_init,
                 sigma_init: float = 0.3,
                 sce2tm: bool = False, tau: float = 0.2,
                 gene_baseline=None, topic_gene_override=None):
        super().__init__()
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.register_buffer("rho", torch.as_tensor(rho, dtype=torch.float32))
        self.register_buffer("lambda_batch", torch.as_tensor(lambda_batch, dtype=torch.float32))
        self.T, self.L = self.alpha.shape
        self.S, self.V = self.lambda_batch.shape
        self.K = M_init.shape[1]
        assert M_init.shape == (self.T, self.K)

        self.M = nn.Parameter(torch.as_tensor(M_init, dtype=torch.float32))
        self.log_sigma_k = nn.Parameter(
            torch.full((self.K,), float(np.log(sigma_init)), dtype=torch.float32)
        )

        self._sce2tm = bool(sce2tm)
        self._tau = float(tau)
        with torch.no_grad():
            if topic_gene_override is not None:
                B = torch.as_tensor(topic_gene_override, dtype=torch.float32)
                assert B.shape == (self.T, self.V), f"override {tuple(B.shape)} != ({self.T},{self.V})"
                self._sce2tm = True
            elif self._sce2tm:
                an = F.normalize(self.alpha, dim=1)
                gn = F.normalize(self.rho.t(), dim=1)
                B = F.softmax(-(2.0 - 2.0 * (an @ gn.t())) / self._tau, dim=0)
            else:
                B = self.alpha @ self.rho
            self.register_buffer("topic_gene", B)
        gb = torch.zeros(self.V) if gene_baseline is None else torch.as_tensor(gene_baseline, dtype=torch.float32)
        self.register_buffer("gene_baseline", gb)

        with torch.no_grad():
            if self._sce2tm:
                c0 = F.softmax(self.M.t(), dim=-1)
                cB0 = c0 @ self.topic_gene
                g_mean = cB0.mean(0)
                g_std = cB0.std(0)
                g_std = g_std.clamp(min=float(g_std.median()) * 0.3 + 1e-8)
            else:
                g_mean = torch.zeros(self.V); g_std = torch.ones(self.V)
        self.register_buffer("tg_mean", g_mean)
        self.register_buffer("tg_std", g_std)

    def _gene_logits(self, c: torch.Tensor) -> torch.Tensor:
        """c (..., T) -> per-gene logits (..., V). For scE2TM standardizes c@B per gene
        (so the level-free B's tiny type signal is learnable) then adds the per-gene
        expression baseline; for the dot-product decoder this is identity + 0."""
        z = c @ self.topic_gene
        if self._sce2tm:
            z = (z - self.tg_mean) / self.tg_std + self.gene_baseline
        return z

    @property
    def sigma_k(self) -> torch.Tensor:
        return self.log_sigma_k.exp()

    def decode_pure(self, state_idx: torch.Tensor, batch_idx: torch.Tensor,
                    use_lambda: bool = True) -> torch.Tensor:
        """
        Reconstruction for pure-state pseudobulk. With theta = e_k and dM = 0:
            c_k  = softmax(M[:, k])
            r_d  = beta_{d,k} = softmax(c_k @ alpha @ rho + lambda_s)

        Args:
            state_idx: (B,) long, the single active state per sample.
            batch_idx: (B,) long, batch index for lambda.
        Returns:
            r: (B, V) predicted gene proportions.
        """
        c = F.softmax(self.M[:, state_idx].t(), dim=-1)
        logits = self._gene_logits(c)
        if use_lambda:
            logits = logits + self.lambda_batch[batch_idx]
        return F.softmax(logits, dim=-1)

    def decode(self, theta: torch.Tensor, dM: Optional[torch.Tensor],
               batch_idx: torch.Tensor, use_lambda: bool = True,
               k_chunk: int = 0):
        """
        Full simplex mixture.

        Args:
            theta:     (B, K) proportions on the simplex.
            dM:        (B, K, T) topic perturbations, or None for dM = 0.
            batch_idx: (B,) long.
            use_lambda: include the batch effect in beta (True) or not (False,
                        for beta_bio / bulk).
            k_chunk:   if >0, compute beta in chunks of this many states to cap
                       the (B, K, V) memory peak.
        Returns:
            r:    (B, V) simplex mixture.
            beta: (B, K, V) per-state gene profiles.
        """
        B = theta.shape[0]
        Mt = self.M.t().unsqueeze(0)
        c_logit = Mt if dM is None else Mt + dM
        c = F.softmax(c_logit, dim=-1)

        lam = self.lambda_batch[batch_idx].unsqueeze(1) if use_lambda else None

        if k_chunk and k_chunk < self.K:
            r = torch.zeros(B, self.V, device=theta.device, dtype=theta.dtype)
            beta_chunks = []
            for s in range(0, self.K, k_chunk):
                e = min(s + k_chunk, self.K)
                logits = self._gene_logits(c[:, s:e])
                if lam is not None:
                    logits = logits + lam
                beta_c = F.softmax(logits, dim=-1)
                r = r + torch.einsum("bk,bkv->bv", theta[:, s:e], beta_c)
                beta_chunks.append(beta_c)
            beta = torch.cat(beta_chunks, dim=1)
            return r, beta

        logits = self._gene_logits(c)
        if lam is not None:
            logits = logits + lam
        beta = F.softmax(logits, dim=-1)
        r = torch.einsum("bk,bkv->bv", theta, beta)
        return r, beta

    @property
    def bio_baseline(self) -> torch.Tensor:
        """
        Shared expression baseline = mean of the per-batch lambda. lambda_s
        splits into a shared biological baseline (this mean, mostly housekeeping)
        and a batch-specific technical part (lambda_s - mean). The biological GEP
        keeps the baseline and drops only the technical part.
        """
        if self._sce2tm:
            return torch.zeros(self.V, device=self.lambda_batch.device)
        return self.lambda_batch.mean(0)

    def beta_bio(self, dM: Optional[torch.Tensor] = None,
                 states: Optional[torch.Tensor] = None,
                 use_baseline: bool = True) -> torch.Tensor:
        """
        Batch-corrected biological per-state gene profiles:
            beta_bio_k = softmax(_gene_logits(softmax(M[:,k] + dM_k)) + bio_baseline).
        Keeps the shared expression baseline (biology; carried by lambda_mean for the
        dot-product decoder, or by the decoder BatchNorm for scE2TM), drops the
        batch-specific technical part. use_baseline=False -> pure topic profile.
        With dM=None and states=None returns the consensus profiles (K, V).
        """
        M = self.M if states is None else self.M[:, states]
        c_logit = M.t() if dM is None else M.t() + dM
        c = F.softmax(c_logit, dim=-1)
        logits = self._gene_logits(c)
        if use_baseline:
            logits = logits + self.bio_baseline
        return F.softmax(logits, dim=-1)



class C2Encoder(nn.Module):
    """
    Two-head amortised encoder for Phases 2-3.

    Input:  X_tilde = X / N  (B, V)   [log1p optional]
    Shared: Linear(V->h)->BN->ReLU->Drop  x2
    Head delta -> (mu_delta, logvar_delta)  (B, K)
    Head dM    -> Linear(h->h)->BN->ReLU->Drop, then (mu_dM, logvar_dM)  (B, K, T)
    """

    def __init__(self, V: int, K: int, T: int, hidden: int = 256,
                 dropout: float = 0.1, log_input: bool = True):
        super().__init__()
        self.K, self.T, self.log_input = K, T, log_input
        self.backbone = nn.Sequential(
            nn.Linear(V, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.mu_delta = nn.Linear(hidden, K)
        self.logvar_delta = nn.Linear(hidden, K)
        self.dM_trunk = nn.Sequential(
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.mu_dM = nn.Linear(hidden, K * T)
        self.logvar_dM = nn.Linear(hidden, K * T)
        nn.init.zeros_(self.mu_dM.weight)
        nn.init.zeros_(self.mu_dM.bias)

    def forward(self, X):
        N = X.sum(dim=-1, keepdim=True).clamp(min=1)
        x = X / N
        if self.log_input:
            x = torch.log1p(x * 1e4)
        h = self.backbone(x)
        mu_d, logvar_d = self.mu_delta(h), self.logvar_delta(h)
        g = self.dM_trunk(h)
        mu_dM = self.mu_dM(g).view(-1, self.K, self.T)
        logvar_dM = self.logvar_dM(g).view(-1, self.K, self.T)
        return mu_d, logvar_d, mu_dM, logvar_dM
