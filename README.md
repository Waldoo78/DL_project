# PatchTST

A PyTorch implementation of **PatchTST** (Patch Time Series Transformer) for multivariate long-term time series forecasting, benchmarked against a linear (Ridge regression + seasonal-trend decomposition) baseline across 7 standard datasets.

> **Reference**: Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. ICLR 2023. [[paper]](https://openreview.net/forum?id=Jbdc0vTOcol)

## Architecture

```
Input (B, L, C)
      |
   RevIN  ────────────────────────────────────┐
      |                                       |
  Patchify  (B, C, N, P)                      |
      |                                       |
  Patch Embedding  W_p * x_p                  |
      |                                       |
  + Learnable Positional Encoding  W_pos      |
      |                                       |
  Transformer Encoder (x num_layers)          |
  ┌─────────────────────────────┐             |
  │  Multi-Head Self-Attention  │             |
  │  + Add & BatchNorm          │             |
  │  FFN (Linear → GELU → Lin)  │             |
  │  + Add & BatchNorm          │             |
  └─────────────────────────────┘             |
      |                                       |
  Prediction Head                             |
  Flatten → Linear  (B, H, C)                 |
      |                                       |
   RevIN denorm  ←────────────────────────────┘
      |
Output (B, H, C)
```

**Key design choices**: channel-independence (each channel processed separately through shared weights), BatchNorm instead of LayerNorm, learnable positional encoding.

### Key equations

| Step | Equation |
|------|----------|
| RevIN normalize | `x' = (x - mu) / sigma * gamma + beta` |
| Patching | `num_patches = (L - P) / S + 1` (with end-padding) |
| Patch embedding + position | `x_d = W_p * x_p + W_pos` |
| Self-attention | `Attn(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V` |
| Prediction head | `y = Linear(Flatten(encoder_output))` |

Where `L` = context length, `P` = patch length, `S` = patch stride, `H` = prediction horizon.

## Datasets

| Dataset | Channels | Source |
|---------|----------|--------|
| ETTh1 | 7 | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| ETTh2 | 7 | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| ETTm1 | 7 | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| ETTm2 | 7 | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| Weather | 21 | [Informer2020](https://github.com/zhouhaoyi/Informer2020) |
| Exchange | 8 | [Informer2020](https://github.com/zhouhaoyi/Informer2020) |
| Electricity | 321 | [Informer2020](https://github.com/zhouhaoyi/Informer2020) |

### Download & directory structure

Download the CSV files from the repos above and place them as follows:

```
dataset/
  ett/
    ETTh1.csv
    ETTh2.csv
    ETTm1.csv
    ETTm2.csv
  weather/
    weather.csv
  exchange_rate/
    exchange_rate.csv
  electricity/
    electricity.csv
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Usage

### Train PatchTST on all datasets

```bash
python train.py
```

Trains each experiment defined in `configs.py`. Checkpoints are saved to `checkpoints/`.

### Evaluate PatchTST

```bash
python evaluate.py
```

Loads trained checkpoints and evaluates on the test set. Results are saved to `results/patchtst.json`.

### Run the linear baseline

```bash
python baseline.py
```

Runs Ridge regression with seasonal-trend decomposition on all datasets. Results are saved to `results/baselines.json`.

### Explore results

Open `explore.ipynb` for visual comparisons: MSE heatmaps, forecast overlays, and attention heatmaps.

## Results

MSE / MAE on the test set (context length = 336):

| Dataset | Horizon | PatchTST MSE | PatchTST MAE | Baseline MSE | Baseline MAE |
|---------|---------|-------------|-------------|-------------|-------------|
| ETTh1 | 96 | 0.3710 | 0.3960 | 0.3765 | 0.3959 |
| ETTh1 | 192 | 0.4185 | 0.4240 | 0.4117 | 0.4190 |
| ETTh1 | 336 | 0.4398 | 0.4401 | 0.4367 | 0.4341 |
| ETTh2 | 96 | 0.2794 | 0.3412 | 0.2845 | 0.3437 |
| ETTh2 | 192 | 0.3790 | 0.4044 | 0.3389 | 0.3829 |
| ETTh2 | 336 | 0.3990 | 0.4194 | 0.3664 | 0.4093 |
| ETTm1 | 96 | 0.2972 | 0.3459 | 0.2875 | 0.3348 |
| ETTm1 | 192 | 0.3284 | 0.3674 | 0.3293 | 0.3588 |
| ETTm1 | 336 | 0.3702 | 0.3928 | 0.3690 | 0.3809 |
| ETTm2 | 96 | 0.1646 | 0.2558 | 0.1627 | 0.2500 |
| ETTm2 | 192 | 0.2234 | 0.2961 | 0.2179 | 0.2885 |
| ETTm2 | 336 | 0.2778 | 0.3344 | 0.2726 | 0.3249 |
| Weather | 96 | 0.1469 | 0.1974 | 0.1434 | 0.1923 |
| Weather | 192 | 0.1912 | 0.2404 | 0.1865 | 0.2350 |
| Weather | 336 | 0.2449 | 0.2829 | 0.2394 | 0.2769 |
| Exchange | 96 | 0.0959 | 0.2186 | 0.0930 | 0.2113 |
| Exchange | 192 | 0.2087 | 0.3253 | 0.1957 | 0.3095 |
| Exchange | 336 | 0.3355 | 0.4248 | 0.3603 | 0.4295 |
| Electricity | 96 | 0.1336 | 0.2254 | 0.1342 | 0.2284 |
| Electricity | 192 | 0.1514 | 0.2433 | 0.1503 | 0.2424 |
| Electricity | 336 | 0.1686 | 0.2607 | 0.1673 | 0.2594 |

The linear baseline (Ridge regression with multi-scale moving-average decomposition) is competitive with PatchTST across most datasets. PatchTST shows a clear advantage on Exchange at horizon 336, while the baseline edges ahead on several ETT configurations.

## Project structure

```
PatchTST/
  models/
    PatchTST.py       # Model: RevIN, Patchify, Encoder, Head
  configs.py           # Experiment configurations (hyperparameters per dataset)
  dataset.py           # TimeSeriesDataset (PyTorch Dataset with train/val/test splits)
  train.py             # Training loop with mixed-precision and early stopping
  evaluate.py          # Evaluate trained checkpoints on test sets
  baseline.py          # Linear baseline (Ridge + decomposition)
  explore.ipynb        # Visual analysis notebook
  requirements.txt     # Python dependencies
  results/
    patchtst.json      # PatchTST evaluation results
    baselines.json     # Baseline evaluation results
```
