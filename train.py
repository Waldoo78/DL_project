import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TimeSeriesDataset
from models.PatchTST import PatchTST, PatchTSTConfig


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
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


if __name__ == "__main__":
    # Config
    data_path      = "dataset/weather/weather.csv"
    checkpoint_path = "checkpoints/patchtst_weather.pth"
    epochs         = 100
    batch_size     = 128
    learning_rate  = 1e-4
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_channels = pd.read_csv(data_path, nrows=0).shape[1] - 1 
    config = PatchTSTConfig(num_channels=num_channels)

    # Datasets & DataLoaders
    train_loader = DataLoader(TimeSeriesDataset(data_path, config.context_length, config.prediction_length, "train"), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TimeSeriesDataset(data_path, config.context_length, config.prediction_length, "val"),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TimeSeriesDataset(data_path, config.context_length, config.prediction_length, "test"),  batch_size=batch_size, shuffle=False)

    # Model
    model     = PatchTST(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Load checkpoint if exists
    os.makedirs("checkpoints", exist_ok=True)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint from {checkpoint_path}")

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:3d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    # Test on best checkpoint
    model.load_state_dict(torch.load(checkpoint_path))
    test_mse = evaluate(model, test_loader, criterion, device)
    test_mae = evaluate(model, test_loader, nn.L1Loss(), device)
    print(f"\nTest MSE: {test_mse:.4f} | Test MAE: {test_mae:.4f}")
