# Block-1 Benchmark — synthetic FEM (layered walls + thermal bridges)

- device: `cuda` · epochs: 300 · batch: 64 · seed: 1337
- FV ground-truth solver: 2.56 ms/sample

| Model | Field rel-L2 ↓ | U-value MAE ↓ (W/m²K) | U-MAPE ↓ | vs 1-D clear ↑ | Speedup vs FV ↑ | Params | Train (s) |
|---|---|---|---|---|---|---|---|
| fno | 0.0144 | 0.0205 | 4.9% | 5.70× | 0× | 308,193 | 74 |
| cnn | 0.0170 | 0.0254 | 5.5% | 4.59× | 3× | 74,145 | 68 |

Geometry-blind 1-D clear-wall baseline U-MAE: 0.1168 W/m²K — the number any geometry-aware model must beat (H1).
