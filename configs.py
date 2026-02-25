_ETTh = dict(context_length=336, patch_length=16, patch_stride=8,
             d_model=16, num_heads=4, ffn_dim=128, dropout=0.2,
             batch_size=128, learning_rate=1e-4)

_ETTm = dict(context_length=336, patch_length=16, patch_stride=8,
             d_model=128, num_heads=16, ffn_dim=256, dropout=0.2,
             batch_size=128, learning_rate=1e-4)

_large = dict(context_length=336, patch_length=16, patch_stride=8,
              d_model=128, num_heads=16, ffn_dim=256, dropout=0.2,
              batch_size=128, learning_rate=1e-4)

_ILI = dict(context_length=104, patch_length=24, patch_stride=2,
            d_model=16, num_heads=4, ffn_dim=128, dropout=0.3,
            batch_size=16, learning_rate=0.0025)

def _exp(name, data_path, base, prediction_length, epochs=100):
    return dict(
        name=name,
        data_path=data_path,
        checkpoint_path=f"checkpoints/patchtst_{name.lower()}.pth",
        prediction_length=prediction_length,
        epochs=epochs,
        **base,
    )


EXPERIMENTS = []

for p in [96, 192, 336, 720]:
    EXPERIMENTS += [
        _exp(f"ETTh1_{p}", "dataset/ett/ETTh1.csv",     _ETTh,  p),
        _exp(f"ETTh2_{p}", "dataset/ett/ETTh2.csv",     _ETTh,  p),
        _exp(f"ETTm1_{p}", "dataset/ett/ETTm1.csv",     _ETTm,  p),
        _exp(f"ETTm2_{p}", "dataset/ett/ETTm2.csv",     _ETTm,  p),
        _exp(f"Weather_{p}", "dataset/weather/weather.csv", _large, p),
        _exp(f"Exchange_{p}", "dataset/exchange_rate/exchange_rate.csv", _large, p),
    ]

for p in [24, 36, 48, 60]:
    EXPERIMENTS.append(
        _exp(f"ILI_{p}", "dataset/illness/national_illness.csv", _ILI, p),
    )
