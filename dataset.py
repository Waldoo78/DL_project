import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def _get_borders(filename: str, n: int):
    """Return (train_end, val_end, test_end) based on dataset type."""
    if "ETTh" in filename:
        return 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24
    elif "ETTm" in filename:
        return 12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4
    else:
        train_end = int(n * 0.7)
        val_end   = int(n * 0.8)
        return train_end, val_end, n


class TimeSeriesDataset(Dataset):
    def __init__(self, path: str, context_length: int, prediction_length: int, split: str = "train"):
        """
        :param path: Path to the CSV file
        :param context_length: Number of time steps as input
        :param prediction_length: Number of time steps to predict
        :param split: "train", "val" or "test"
        """
        df = pd.read_csv(path)
        df = df.drop(columns=["date"]).values.astype(np.float32)  # (T, num_channels)

        train_end, val_end, test_end = _get_borders(path, len(df))

        # StandardScaler fitted on train only
        scaler = StandardScaler()
        scaler.fit(df[:train_end])
        df = scaler.transform(df)

        # Overlap of context_length at val/test boundaries to ensure complete windows
        starts = {"train": 0,                          "val": train_end - context_length, "test": val_end - context_length}
        ends   = {"train": train_end,                  "val": val_end,                    "test": test_end}

        self.data              = torch.tensor(df[starts[split]:ends[split]])
        self.context_length    = context_length
        self.prediction_length = prediction_length

    def __len__(self):
        return len(self.data) - self.context_length - self.prediction_length + 1

    def __getitem__(self, idx):
        x = self.data[idx                       : idx + self.context_length]
        y = self.data[idx + self.context_length : idx + self.context_length + self.prediction_length]
        return x, y
