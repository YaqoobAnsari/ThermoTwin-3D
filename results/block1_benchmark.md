# Block-1 Benchmark — synthetic FEM (layered walls + thermal bridges)

- device: `cuda` · epochs: 300 · batch: 64 · seed: 1337
- FV ground-truth solver: 2.55 ms/sample

| Model | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Speedup vs FV ↑ | Params | Train (s) |
|---|---|---|---|---|---|---|---|
| fno_physics | 0.0143 | 0.0218 | 5.1% | 5.36× | 1× | 308,193 | 92 |
| fno | 0.0144 | 0.0205 | 4.9% | 5.70× | 1× | 308,193 | 89 |
| unet | 0.0167 | 0.0343 | 9.1% | 3.41× | 2× | 466,529 | 83 |
| cnn | 0.0170 | 0.0254 | 5.5% | 4.59× | 3× | 74,145 | 83 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.1168 W/m²K — the number any geometry-aware model must beat (H1).
