import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from configs import EXPERIMENTS
from dataset import TimeSeriesDataset
from models.PatchTST import PatchTST, PatchTSTConfig


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            pred = model(x)
            loss = criterion(pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item()
    return total_loss / len(loader)


def run_experiment(exp, device):
    if not os.path.exists(exp["data_path"]):
        print(f"[skip] {exp['name']}: {exp['data_path']} not found")
        return None, None

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

    train_loader = DataLoader(
        TimeSeriesDataset(exp["data_path"], config.context_length, config.prediction_length, "train"),
        batch_size=exp["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        TimeSeriesDataset(exp["data_path"], config.context_length, config.prediction_length, "val"),
        batch_size=exp["batch_size"], shuffle=False,
    )
    test_loader = DataLoader(
        TimeSeriesDataset(exp["data_path"], config.context_length, config.prediction_length, "test"),
        batch_size=exp["batch_size"], shuffle=False,
    )

    model     = PatchTST(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp["learning_rate"])
    criterion = nn.MSELoss()
    scaler    = GradScaler()
    os.makedirs(os.path.dirname(exp["checkpoint_path"]), exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, exp["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss   = evaluate(model, val_loader, criterion, device)
        print(f"  Epoch {epoch:3d} | train: {train_loss:.4f} | val: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), exp["checkpoint_path"])

    model.load_state_dict(torch.load(exp["checkpoint_path"], weights_only=True))
    test_mse = evaluate(model, test_loader, criterion, device)
    test_mae = evaluate(model, test_loader, nn.L1Loss(), device)

    del model, optimizer, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()

    return test_mse, test_mae


if __name__ == "__main__":
    torch.manual_seed(2021)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}\n")

    CHECKPOINT_DIR = "checkpoints"
    for exp in EXPERIMENTS:
        exp["checkpoint_path"] = os.path.join(CHECKPOINT_DIR, os.path.basename(exp["checkpoint_path"]))

    results = {}
    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp['name']}")
        print(f"{'='*60}")
        mse, mae = run_experiment(exp, device)
        if mse is not None:
            results[exp["name"]] = (mse, mae)
            print(f">> {exp['name']} | MSE: {mse:.4f} | MAE: {mae:.4f}")

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for name, (mse, mae) in results.items():
        print(f"{name:25s} | MSE: {mse:.4f} | MAE: {mae:.4f}")
