import json
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from dataset import TimeSeriesDataset
from models.PatchTST import PatchTST, PatchTSTConfig
from configs import EXPERIMENTS

CHECKPOINT_DIR = "checkpoints"
OUTPUT_FILE    = "results/patchtst.json"

os.makedirs("results", exist_ok=True)

results = {}
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for exp in EXPERIMENTS:
    name = exp["name"]
    ckpt = os.path.join(CHECKPOINT_DIR, f"patchtst_{name.lower()}.pth")

    if not os.path.exists(ckpt):
        print(f"[skip] {name}: checkpoint not found")
        continue
    if not os.path.exists(exp["data_path"]):
        print(f"[skip] {name}: data not found")
        continue

    num_channels = pd.read_csv(exp["data_path"], nrows=0).shape[1] - 1
    config = PatchTSTConfig(
        num_channels=num_channels,
        context_length=exp["context_length"],
        patch_length=exp["patch_length"],
        patch_stride=exp["patch_stride"],
        d_model=exp["d_model"],
        num_heads=exp["num_heads"],
        ffn_dim=exp["ffn_dim"],
        dropout=exp["dropout"],
        prediction_length=exp["prediction_length"],
    )

    model = PatchTST(config).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    loader = DataLoader(
        TimeSeriesDataset(exp["data_path"], config.context_length, config.prediction_length, "test"),
        batch_size=128, shuffle=False,
    )

    mse_total = mae_total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse_total += nn.MSELoss()(pred, y).item()
            mae_total += nn.L1Loss()(pred, y).item()

    results[name] = {
        "MSE": round(mse_total / len(loader), 4),
        "MAE": round(mae_total / len(loader), 4),
    }
    print(f"{name:25s}  MSE: {results[name]['MSE']:.4f}  MAE: {results[name]['MAE']:.4f}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved in {OUTPUT_FILE}")
