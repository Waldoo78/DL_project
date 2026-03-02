import json
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

EVALUATE = {
    "ETTh1":       {"path": "dataset/ett/ETTh1.csv",                       "horizons": [96, 192, 336, 720]},
    "ETTh2":       {"path": "dataset/ett/ETTh2.csv",                       "horizons": [96, 192, 336, 720]},
    "ETTm1":       {"path": "dataset/ett/ETTm1.csv",                       "horizons": [96, 192, 336, 720]},
    "ETTm2":       {"path": "dataset/ett/ETTm2.csv",                       "horizons": [96, 192, 336, 720]},
    "Weather":     {"path": "dataset/weather/weather.csv",                 "horizons": [96, 192, 336, 720]},
    "Exchange":    {"path": "dataset/exchange_rate/exchange_rate.csv",     "horizons": [96, 192, 336, 720]},
    "Electricity": {"path": "dataset/electricity/electricity.csv",         "horizons": [96, 192, 336, 720]},
}

OUTPUT_FILE = "results/baselines.json"
CTX = 336
MA_KERNEL = 25
MA_KERNELS = [MA_KERNEL, MA_KERNEL * 4, MA_KERNEL * 8]


def _get_borders(path, n):
    if "ETTh" in path:
        return 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24
    elif "ETTm" in path:
        return 12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4
    else:
        return int(n * 0.7), int(n * 0.8), n


def _load(path):
    df_raw = pd.read_csv(path)
    data = df_raw.drop(columns=["date"]).values.astype(np.float32)
    n = len(data)
    train_end, val_end, test_end = _get_borders(path, n)
    scaler = StandardScaler()
    scaler.fit(data[:train_end])
    data = scaler.transform(data).astype(np.float32)
    return data, train_end, val_end, test_end


def _windows(data_c, start, n_win, ctx, pred_len):
    ctx_mat = sliding_window_view(data_c[start : start + n_win + ctx - 1], ctx)
    tgt_mat = sliding_window_view(data_c[start + ctx : start + ctx + n_win + pred_len - 1], pred_len)
    return ctx_mat.astype(np.float32), tgt_mat.astype(np.float32)


def _moving_avg(mat, k):
    padded = np.concatenate([np.tile(mat[:, :1], (1, k - 1)), mat], axis=1)
    cs = np.concatenate(
        [np.zeros((mat.shape[0], 1), dtype=mat.dtype), np.cumsum(padded, axis=1)], axis=1
    )
    return (cs[:, k:] - cs[:, :-k]) / k


def run_linear_baseline(path, pred_len):
    data, train_end, val_end, test_end = _load(path)
    C = data.shape[1]
    n_tr = train_end - CTX - pred_len + 1
    n_te = test_end - val_end - pred_len + 1
    ks = sorted({min(k, CTX) for k in MA_KERNELS})

    mse_list, mae_list = [], []

    for c in range(C):
        dc = data[:, c]

        X_tr, Y_tr = _windows(dc, 0, n_tr, CTX, pred_len)
        mus_tr = X_tr.mean(axis=1, keepdims=True)
        X_tr_c = X_tr - mus_tr
        Y_tr_c = Y_tr - mus_tr
        T_tr_list = [_moving_avg(X_tr_c, k) for k in ks]
        S_tr = X_tr_c - T_tr_list[0]

        X_te, Y_te = _windows(dc, val_end - CTX, n_te, CTX, pred_len)
        mus_te = X_te.mean(axis=1, keepdims=True)
        X_te_c = X_te - mus_te
        T_te_list = [_moving_avg(X_te_c, k) for k in ks]
        S_te = X_te_c - T_te_list[0]

        TS_tr = np.hstack(T_tr_list + [S_tr])
        TS_te = np.hstack(T_te_list + [S_te])
        Y_pred_c = Ridge(alpha=1.0).fit(TS_tr, Y_tr_c).predict(TS_te)

        Y_pred = Y_pred_c + mus_te

        mse_list.append(float(np.mean((Y_pred - Y_te) ** 2)))
        mae_list.append(float(np.mean(np.abs(Y_pred - Y_te))))

    return float(np.mean(mse_list)), float(np.mean(mae_list))


if __name__ == "__main__":
    results = {}

    for name, cfg in EVALUATE.items():
        for pred_len in cfg["horizons"]:
            key = f"{name}_{pred_len}"
            mse, mae = run_linear_baseline(cfg["path"], pred_len)
            results[key] = {"MSE": round(mse, 4), "MAE": round(mae, 4)}
            print(f"{key:25s}  MSE={mse:.4f}  MAE={mae:.4f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUTPUT_FILE}")
