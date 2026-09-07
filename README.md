# Neural Collaborative Filtering in PyTorch

A from-scratch PyTorch implementation of the models in
[He et al., *Neural Collaborative Filtering*, WWW 2017](https://arxiv.org/abs/1708.05031),
trained and evaluated on MovieLens rating data. Built as the group project for the
deep learning course in Georgia Tech's M.S. in Computer Science (OMSCS) program.

## Models

| Model | Idea |
|-------|------|
| **GMF** | Generalized Matrix Factorization — element-wise product of user/item embeddings, projected linearly to a rating. |
| **MLP** | Multi-layer perceptron over the concatenated user/item embeddings; learns high-order non-linear interactions. |
| **NMF** (Neural MF) | Element-wise product fed through an MLP prediction layer instead of a linear one. |
| **Joint NMF** | GMF and Neural MF streams combined with a learned fusion weight (α), the paper's best-performing configuration. |

All models share their embedding layers, training loop, and evaluation harness, so
architectural differences are isolated and directly comparable.

## Data

[MovieLens 100K and 20M](https://grouplens.org/datasets/movielens/), downloaded and
unzipped automatically on first run. User and item IDs are remapped to contiguous
indices for the embedding tables. For the regression setup, ratings are rescaled to
`[0, 1]`; a classification setup (BCE, without rescaling) is also supported.

## Usage

```bash
python main.py --config configs.yml
```

`configs.yml` selects everything: dataset (`100k` / `20m`), model
(`gmf` / `mlp` / `nmf` / `joint_nmf`), latent dimension, optimizer, learning rate,
weight decay, epochs, batch size, loss (`MSE` / `BCE`), and the checkpoint path.
`config_gmf.yml`, `config_nmf.yml`, and `config_joint_nmf.yml` are ready-made configs
for those variants. `tuning_experiments.ipynb` holds the hyperparameter sweeps and the
final training/evaluation runs behind the numbers below.

## Evaluation

Two families of metrics are implemented:

**Rating prediction.** Held-out test RMSE and loss, tracked per epoch. Hyperparameter
sweeps over embedding dimension and learning rate, plus per-epoch training curves, are
committed under `images/` for each model on both datasets.

**Top-N ranking.** `eval.py` scores every item for each user in batches and keeps the
best-k with a heap, then computes:

- **Hit Rate@k** — fraction of users with at least one already-rated item in the top-k
- **Precision@k** — fraction of the top-k slots filled by already-rated items

Computed over all MovieLens 100K users for the tuned per-model checkpoints:

| Model | Hit Rate@40 | Precision@40 |
|-------|-------------|--------------|
| GMF   | 0.729       | 0.055        |
| NMF   | 0.659       | 0.053        |
| MLP   | 0.526       | 0.035        |

## Notes

**Embedding dimension matters far more for the linear model than for the non-linear
ones.** GMF's test RMSE degrades sharply at small embedding dimensions while MLP and
Neural MF stay comparatively flat — non-linear interaction learning substitutes for a
wide factorization space, which is the paper's central argument and shows up directly
in these sweeps.

**Caveats worth knowing before reading the ranking numbers.** Hit rate and precision are
computed against the full rating matrix rather than the held-out split, so they measure
how well each model reconstructs known preferences rather than fresh-recommendation
quality, and the absolute values are optimistic. They're most meaningful as a
*relative* comparison between the three architectures under an identical harness.

## Repository layout

```
main.py             config parsing, model construction, training entry point
gmf.py              GMF (and the shared embedding/prediction structure)
ncf_mlp.py          MLP variant
neural_mf.py        Neural MF and joint GMF + Neural MF
train.py            training and test loops, RMSE evaluation, optimizer construction
eval.py             top-N scoring, Hit Rate@k, Precision@k
data_loader.py      MovieLens download, ID remapping, Dataset/DataLoader
configs*.yml        runnable configurations
tuning_experiments.ipynb  sweeps, final training runs, ranking evaluation
images/             hyperparameter sweeps and training curves
```
