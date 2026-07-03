# ETM-Deconv

Cellular deconvolution of bulk RNA-seq with an embedded topic model.

Given a bulk tissue sample — a mixture of many cell types in unknown proportions —
ETM-Deconv infers (i) the **cell-type proportions** and (ii) the **per-sample,
per-cell-type gene expression profiles (GEPs)**. It adapts scETM (single-cell
Embedded Topic Model) to the mixture setting: cell types are described as mixtures
of a few co-expression **topics**, and each sample is a convex (simplex) mixture of
the cell-type profiles. Because the mixture is taken on the simplex, the inferred
proportions are exactly the transcript fractions of each cell type — directly
interpretable and comparable to methods such as BayesPrism.

## Result

On a strict cross-dataset benchmark (training and validation on disjoint datasets),
ETM-Deconv outperforms BayesPrism on proportions and stays competitive on profiles:

| Method      | Proportions (Pearson r) | GEP (Pearson r) |
|-------------|-------------------------|-----------------|
| ETM-Deconv  | **0.959**               | 0.824           |
| BayesPrism  | 0.873                   | 0.870           |

## Method

scETM places genes and topics in a shared embedding space (`rho`, `alpha`) and
learns per-batch technical offsets (`lambda`) from a single-cell atlas. On top of
that frozen basis, ETM-Deconv adds a per-cell-type topic recipe `M` and a
per-sample topic perturbation `dM`. A cell type's profile is

```
beta_k = softmax( softmax(M[:,k] + dM_k) @ alpha @ rho + lambda )
```

and the sample is the convex combination `r = sum_k theta_k * beta_k`. The
likelihood is Multinomial and the model is a semi-supervised VAE trained by
amortised variational inference. Restricting the per-sample variation `dM` to the
low-dimensional topic space is what keeps the profiles biologically structured and
the problem identifiable.

## Pipeline

The model is trained in four phases; each freezes what the previous one learned and
communicates through files on disk.

| Script | Phase | Learns |
|---|---|---|
| `phase0_scetm.py` | 0 | scETM on scRNA-seq: topics `alpha`, gene embeddings `rho`, batch effects `lambda` |
| `phase1_learn_M.py` | 1 | per-cell-type topic recipes `M` and variability `sigma_k` (pure-state pseudobulks) |
| `phase2_train_encoder.py` | 2 | the amortised encoder for `theta` and `dM` (mixture pseudobulks) |
| `phase3_deconvolve.py` | 3 | inference and scoring on held-out data |

`c2_model.py` is the shared model — the simplex-mixture decoder and the two-head
encoder — imported by all phases so they cannot drift apart.
`phase3_bulkval_compare.py` and `phase3_gep_compare.py` reproduce the cross-dataset
proportion and GEP comparisons against BayesPrism.

## Requirements

Python 3.10+ with `torch`, `numpy`, `pandas`, `scipy`, `anndata`,
`scikit-learn` and `matplotlib`.

## Usage

Run the phases in order; each writes artifacts that the next one loads.

```bash
# Phase 0 — foundational scETM on the single-cell atlas
python phase0_scetm.py \
    --scrna scrna.h5ad --gene2vec gene2vec_dim_200_iter_9.txt \
    --cell_type_key cell_type --batch_key batch \
    --T 500 --L 200 --epochs 300 --output_dir ./phase0

# Phase 1 — per-cell-type topic recipes M and sigma_k
python phase1_learn_M.py \
    --pseudobulk pseudobulk_pure.h5ad --phase0 ./phase0 \
    --output_dir ./phase1 --epochs 300 --lr 5e-3

# Phase 2 — encoder for proportions and per-sample perturbation
python phase2_train_encoder.py \
    --pseudobulk pseudobulk_mix.h5ad --phase0 ./phase0 --phase1 ./phase1 \
    --output_dir ./phase2 --epochs 200 --lr 1e-3 --w_gep 1.0 --w_theta 10.0

# Phase 3 — inference and scoring on held-out patients
python phase3_deconvolve.py \
    --scrna reference.h5ad --phase0 ./phase0 --phase1 ./phase1 --phase2 ./phase2 \
    --output_dir ./phase3
```

## Report

The full write-up — model derivation, loss, ablations and results — is in the
accompanying report.
