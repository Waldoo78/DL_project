import json
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

EVALUATE = {
    "ETTh1":       {"path": "dataset/ett/ETTh1.csv",                      "horizons": [96, 192, 336]},
    "ETTh2":       {"path": "dataset/ett/ETTh2.csv",                      "horizons": [96, 192, 336]},
    "ETTm1":       {"path": "dataset/ett/ETTm1.csv",                      "horizons": [96, 192, 336]},
    "ETTm2":       {"path": "dataset/ett/ETTm2.csv",                      "horizons": [96, 192, 336]},
    "Weather":     {"path": "dataset/weather/weather.csv",                "horizons": [96, 192, 336]},
    "Exchange":    {"path": "dataset/exchange_rate/exchange_rate.csv",    "horizons": [96, 192, 336]},
    "Electricity": {"path": "dataset/electricity/electricity.csv",        "horizons": [96, 192, 336]},
}
CONTEXT_LENGTH = 336
OUTPUT_FILE    = "results/baselines.json"

SEASON = {
    "ETTh1": 24, "ETTh2": 24,
    "ETTm1": 96, "ETTm2": 96,
    "Weather": 144, "Electricity": 24,
    "Exchange": None,
}


def _get_borders(path, n):
    if "ETTh" in path:
        return 12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24
    elif "ETTm" in path:
        return 12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4
    else:
        return int(n*0.7), int(n*0.8), n


def run_naive(name, path, pred_len):
    df_raw    = pd.read_csv(path)
    data      = df_raw.drop(columns=["date"]).values.astype(np.float32)
    n         = len(data)
    train_end, val_end, test_end = _get_borders(path, n)

    scaler = StandardScaler()
    scaler.fit(data[:train_end])
    data = scaler.transform(data).astype(np.float32)

    ctx    = CONTEXT_LENGTH
    n_te   = test_end - val_end - pred_len + 1
    period = SEASON.get(name)

    ctx_all = sliding_window_view(data[val_end - ctx : val_end - ctx + n_te + ctx - 1],
                                  ctx, axis=0)   # (n_te, C, ctx)
    tgt_all = sliding_window_view(data[val_end : val_end + n_te + pred_len - 1],
                                  pred_len, axis=0)   # (n_te, C, pred_len)

    # Naive
    last     = ctx_all[:, :, -1]
    yp_naive = np.repeat(last[:, :, None], pred_len, axis=2)
    mse_naive = float(np.mean((yp_naive - tgt_all) ** 2))
    mae_naive = float(np.mean(np.abs(yp_naive - tgt_all)))

    # Seasonal Naive
    if period is not None and period <= ctx:
        last_season = ctx_all[:, :, -period:]
        repeats     = (pred_len + period - 1) // period
        yp_sn       = np.tile(last_season, (1, 1, repeats))[:, :, :pred_len]
        mse_sn = float(np.mean((yp_sn - tgt_all) ** 2))
        mae_sn = float(np.mean(np.abs(yp_sn - tgt_all)))
    else:
        mse_sn, mae_sn = mse_naive, mae_naive

    return mse_naive, mae_naive, mse_sn, mae_sn


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    results = {}

    for name, cfg in EVALUATE.items():
        for pred_len in cfg["horizons"]:
            key = f"{name}_{pred_len}"
            _, _, ms, mas = run_naive(name, cfg["path"], pred_len)
            results[key] = {"MSE": round(ms, 4), "MAE": round(mas, 4)}
            print(f"{key:25s}  MSE: {ms:.4f}  MAE: {mas:.4f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved → {OUTPUT_FILE}")
