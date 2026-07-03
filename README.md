# ETM-Deconv

## Result

On a strict cross-dataset benchmark (training and validation on disjoint datasets),
ETM-Deconv outperforms BayesPrism on proportions and stays competitive on profiles:

| Method      | Proportions (Pearson r) | GEP (Pearson r) |
|-------------|-------------------------|-----------------|
| ETM-Deconv  | **0.959**               | 0.824           |
| BayesPrism  | 0.873                   | 0.870           |

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


