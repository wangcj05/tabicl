# CLAUDE.md — TabICL

## Project overview

TabICL is a scikit-learn-compatible tabular foundation model that performs **in-context learning**: given a labelled training set and an unlabelled test set, it predicts in a single forward pass with no gradient updates. The architecture has three sequential transformer stages (column embedding → row interaction → ICL prediction) plus a full ensemble/preprocessing wrapper. Detailed architecture documentation lives in [model.md](model.md).

---

## Build and test

```bash
# install with all extras (from repo root)
pip install -e ".[all]"

# run tests
pytest tests/

# run a single tutorial script
python tutorials/classification_2D_proba.py
```

---

## Key source locations

| Path | What it contains |
|---|---|
| `src/tabicl/__init__.py` | Public API: `TabICLClassifier`, `TabICLRegressor` |
| `src/tabicl/_sklearn/classifier.py` | Scikit-learn wrapper — `fit`, `predict_proba`, embedding flags |
| `src/tabicl/_model/tabicl.py` | Core `TabICL` nn.Module — orchestrates all three stages |
| `src/tabicl/_model/embedding.py` | Stage 1: `ColEmbedding` (set-transformer per column) |
| `src/tabicl/_model/interaction.py` | Stage 2: `RowInteraction` (transformer + CLS tokens) |
| `src/tabicl/_model/learning.py` | Stage 3: `ICLearning` (ICL transformer + decoder) |
| `src/tabicl/_model/layers.py` | `MultiheadAttentionBlock`, `InducedSelfAttentionBlock`, etc. |
| `src/tabicl/_model/encoders.py` | `Encoder` (standard), `SetTransformer` (ISAB-based) |
| `src/tabicl/_model/inference.py` | Memory management: batching, GPU/CPU/disk offloading |
| `src/tabicl/_model/kv_cache.py` | `TabICLCache` — KV and repr caching |
| `tutorials/embedding_tsne_visualization.ipynb` | Embedding demo on two-moon dataset |
| `tutorials/skab_embedding_tsne.ipynb` | Embedding demo on SKAB anomaly sensor data |
| `model.md` | Full architecture reference with tensor shapes |

---

## How to extract embeddings

TabICL exposes three embedding types via optional flags on `predict_proba`. All three can be requested together in a single forward pass.

### Minimal pattern

```python
from tabicl import TabICLClassifier

clf = TabICLClassifier()
clf.fit(X_train, y_train)

result = clf.predict_proba(
    X_test,
    return_col_embedding_sample=True,       # per-column distribution fingerprint
    return_test_representations=True,        # per-row feature interactions (label-free)
    return_test_icl_representations=True,    # per-row ICL activations (label-aware)
)
proba, col_emb, row_repr, icl_repr = result
```

The return order is always: `(proba, [col_emb,] [row_repr,] [icl_repr])` — only the flags that are `True` are included, in that fixed order.

The same embeddings are also stored as fitted attributes after the call:
```python
clf.col_embedding_sample_       # (G+C, 128)
clf.test_representations_       # (n_estimators, n_test, 512)
clf.test_icl_representations_   # (n_estimators, n_test, 512)
```

---

## The three embedding types

### 1. `return_col_embedding_sample` — column identity

**Shape:** `(G+C, E)` = `(H+4, 128)` when using default `"same"` feature grouping.

One 128-dim vector per feature column, capturing the **statistical distribution fingerprint** of that column (set-transformer encoded across all training rows). Does not change between test samples.

**Critical layout of dim 0:**
- Indices `0–3`: CLS placeholder slots, all values `-100.0`, not meaningful
- Indices `4–(H+3)`: one vector per input feature, in original column order

```python
# For H=8 features: meaningful embeddings are at [4:12]
feature_embeddings = col_emb[4:]   # shape (8, 128)
```

**Extraction:** inference path takes `mean(col_embeddings[0], dim=0)` (mean over all rows of the first ensemble member's first table). Training path takes `col_embeddings[0, 0]` (single row snapshot).

**Use for:** feature similarity analysis, understanding what distributional properties the model distinguishes between columns.

---

### 2. `return_test_representations` — row-level feature interactions

**Shape:** `(n_estimators, n_test, 512)`

One 512-dim vector per test sample, produced by Stage 2 (`RowInteraction`). Encodes cross-feature interactions within each row, conditioned on column distributions from Stage 1. **No label information** — `y_train` has not been injected at this point.

**Internal structure of dim 2 (512 = 4 × 128):**
```
[CLS_0 output (128) | CLS_1 output (128) | CLS_2 output (128) | CLS_3 output (128)]
```
The 4 CLS tokens independently aggregate the full feature-group sequence for that row via cross-attention in the last RowInteraction block.

**Extraction:** `representations[:, train_size:]` sliced from the Stage 2 output `(B, T, 512)`.

**Use for:** label-free clustering, anomaly detection, understanding feature structure independent of prediction targets.

```python
# Reduce ensemble dimension before t-SNE or clustering
row_repr_mean = row_repr.mean(axis=0)   # (n_test, 512)
```

---

### 3. `return_test_icl_representations` — in-context-learning activations

**Shape:** `(n_estimators, n_test, 512)`

One 512-dim vector per test sample, extracted from Stage 3 (`ICLearning`) **after** the 12-block ICL transformer and LayerNorm, **before** the final decoder MLP. Fully label-aware — each test vector has attended to all `train_size` labelled training rows across 12 transformer layers.

**Extraction:** `src[:, train_size:]` where `src` is the post-`tf_icl`-plus-`ln` tensor in `_icl_predictions`.

**Use for:** visualising class separability as the model sees it, probing prediction confidence, t-SNE showing decision structure.

```python
# Reduce ensemble dimension before t-SNE
icl_repr_mean = icl_repr.mean(axis=0)   # (n_test, 512)
```

> **Note:** `return_test_icl_representations` is **not supported** when `kv_cache=True`. Use it only without KV cache.

---

## Embedding shape summary

| Flag | Raw shape (from `predict_proba`) | After `mean(axis=0)` |
|---|---|---|
| `return_col_embedding_sample` | `(G+C, 128)` — not stacked per estimator | `—` (already a single sample) |
| `return_test_representations` | `(n_estimators, n_test, 512)` | `(n_test, 512)` |
| `return_test_icl_representations` | `(n_estimators, n_test, 512)` | `(n_test, 512)` |

`G = H` (number of features) when `col_feature_group="same"` (default); `G+C = H+4`.

---

## Key differences between row and ICL representations

| Property | `test_representations` | `test_icl_representations` |
|---|---|---|
| Stage | After Stage 2 | After Stage 3, before decoder |
| Label-aware | No | Yes |
| Content | Feature interactions within a row | Feature interactions + ICL alignment to labelled training data |
| Next operation | 12-block ICL transformer | 2-layer MLP decoder |
| Typical t-SNE result | Clusters by feature structure | Clusters by predicted class |

---

## Common pitfalls

**Wrong axis for ensemble mean.** `test_representations` and `test_icl_representations` have shape `(n_estimators, n_test, 512)`. Always use `mean(axis=0)` to get `(n_test, 512)`. Using `mean(axis=1)` collapses the test-sample dimension and gives `(n_estimators, 512)` — only 8 points when `n_estimators=8`, which will fail t-SNE perplexity checks.

**t-SNE perplexity on column embeddings.** With few features (e.g. 8 sensors), `col_emb[4:]` has only 8 rows. Set `perplexity < 8`, e.g. `perplexity=5`.

**CLS placeholders in column embeddings.** `col_emb[0:4]` are always `-100.0`. Slice to `col_emb[4:]` before any distance-based analysis or visualisation.

**`return_test_icl_representations` + KV cache.** This combination raises `ValueError`. Disable `kv_cache` or use the standard (non-cached) `predict_proba` path.

**`col_embedding_sample_` is from the first ensemble member only.** Across multiple `predict_proba` calls it is not accumulated — it reflects the first batch's first table. For multi-run stability, prefer `test_representations_` or `test_icl_representations_` averaged over `n_estimators`.

---

## Typical embedding workflow

```python
import numpy as np
from sklearn.manifold import TSNE
from tabicl import TabICLClassifier

clf = TabICLClassifier()
clf.fit(X_train, y_train)

proba, col_emb, row_repr, icl_repr = clf.predict_proba(
    X_test,
    return_col_embedding_sample=True,
    return_test_representations=True,
    return_test_icl_representations=True,
)

# Column fingerprints (skip first 4 CLS placeholders)
feature_vecs = col_emb[4:]                    # (n_features, 128)

# Per-sample representations — average over ensemble first
row_mean = row_repr.mean(axis=0)              # (n_test, 512)
icl_mean = icl_repr.mean(axis=0)             # (n_test, 512)

# t-SNE projection
perplexity = min(30, len(y_test) - 1)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
row_2d = tsne.fit_transform(row_mean)
icl_2d = tsne.fit_transform(icl_mean)
```

---

## Codebase conventions

- All `nn.Module` submodules dispatch on `self.training`: `_train_forward` for training, `_inference_forward` for inference. Do not call the private methods directly; use `forward`.
- `skip_value = -100.0` is the sentinel used throughout `ColEmbedding` for padded CLS slots and empty feature positions. `SkippableLinear` and `InducedSelfAttentionBlock` both detect and propagate this value.
- `InferenceManager` wraps each stage's compute function and handles row-level batching and memory offloading transparently. Pass `mgr_config` to control device, AMP, and offloading.
- Public API follows scikit-learn conventions: fitted attributes end with `_` (e.g. `test_representations_`, `col_embedding_sample_`).
