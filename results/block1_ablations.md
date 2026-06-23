# Block-1 Ablations — beating the data-only FNO on U-MAE

- device: `cuda` · epochs: 300 · batch: 64 · lr: 0.001 · seeds: [1337, 1, 2]
- eval: native resolution on `data/processed/block1_val` (64 samples: 15 clear / 49 bridged)
- reference: **fno** (data-only FNO) — mean U-MAE 0.0242 W/m²K, the number to beat
- *robust win* = mean U-MAE lower than the reference by more than the pooled seed σ.

| Variant | Field rel-L2 ↓ | U-MAE ↓ (W/m²K) | U-MAE clear | U-MAE bridge | rel-L2 clear | rel-L2 bridge | Params | Train (s) | Robust win |
|---|---|---|---|---|---|---|---|---|---|
| delta_fno | 0.0131±0.0002 | 0.0105±0.0009 | 0.0017±0.0001 | 0.0133±0.0011 | 0.0005±0.0001 | 0.0170±0.0003 | 308,385 | 96 | **yes** |
| delta_fno_uloss | 0.0132±0.0002 | 0.0111±0.0005 | 0.0015±0.0003 | 0.0141±0.0008 | 0.0005±0.0001 | 0.0170±0.0003 | 308,385 | 100 | **yes** |
| fno_enriched | 0.0139±0.0004 | 0.0162±0.0017 | 0.0058±0.0027 | 0.0194±0.0014 | 0.0022±0.0003 | 0.0175±0.0006 | 308,385 | 95 | **yes** |
| ufno | 0.0136±0.0002 | 0.0196±0.0010 | 0.0032±0.0010 | 0.0247±0.0014 | 0.0015±0.0001 | 0.0174±0.0002 | 337,505 | 85 | **yes** |
| fno_uloss | 0.0157±0.0009 | 0.0200±0.0009 | 0.0072±0.0011 | 0.0240±0.0011 | 0.0032±0.0003 | 0.0195±0.0010 | 308,193 | 90 | **yes** |
| fno *(ref)* | 0.0147±0.0004 | 0.0242±0.0034 | 0.0076±0.0011 | 0.0293±0.0041 | 0.0023±0.0003 | 0.0185±0.0004 | 308,193 | 86 | — |
| fno_physics | 0.0144±0.0003 | 0.0248±0.0035 | 0.0084±0.0010 | 0.0299±0.0044 | 0.0022±0.0002 | 0.0182±0.0004 | 308,193 | 92 | no |
| fno_padded | 0.0156±0.0008 | 0.0256±0.0064 | 0.0115±0.0069 | 0.0299±0.0063 | 0.0036±0.0010 | 0.0193±0.0007 | 308,193 | 87 | no |

## Verdict per variant

- **delta_fno**: yes — robustly beats fno (ΔU-MAE +0.0137 > pooled σ 0.0035)
- **delta_fno_uloss**: yes — robustly beats fno (ΔU-MAE +0.0131 > pooled σ 0.0034)
- **fno_enriched**: yes — robustly beats fno (ΔU-MAE +0.0080 > pooled σ 0.0038)
- **ufno**: yes — robustly beats fno (ΔU-MAE +0.0046 > pooled σ 0.0035)
- **fno_uloss**: yes — robustly beats fno (ΔU-MAE +0.0042 > pooled σ 0.0035)
- **fno**: reference (U-MAE 0.0242)
- **fno_physics**: no — does not beat fno (ΔU-MAE -0.0006)
- **fno_padded**: no — does not beat fno (ΔU-MAE -0.0014)

All metrics are mean±std over the seeds. U-MAE columns split the val set by thermal-bridge presence (`n_bridges == 0` *clear* vs `>= 1` *bridge*); U-MAE is driven by the near-boundary gradient, so the bridge stratum is where geometry awareness has to pay off.
