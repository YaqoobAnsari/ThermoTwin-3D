# Block-1 OOD generalisation — does the in-distribution winner travel?

- device: `cuda` · epochs: 300 · batch: 64 · lr: 0.001 · seeds: [1337, 1, 2]
- regimes: full (train on all 256 of `data/processed/block1_train`) vs lowdata (fixed seeded 64-sample subset)
- eval: native resolution per sample, featurised with each variant's own feature set; metrics are mean±std over seeds.
- test sets: `in_dist` (64), `ood_walls` (64), `ood_films` (64), `ood_bridges` (64), `ood_res` (64).
- **U-MAE** (W/m²K) is primary; *Gen. gap* = OOD U-MAE − in-distribution U-MAE (same variant & regime; lower/negative is better).

## in-distribution (`block1_val`)

| Variant | U-MAE [full] | rel-L2 [full] | U-MAE [lowdata] | rel-L2 [lowdata] |
|---|---|---|---|---|
| delta_fno | 0.0105±0.0009 | 0.0131±0.0002 | 0.0168±0.0008 | 0.0244±0.0010 |
| ufno | 0.0196±0.0010 | 0.0136±0.0002 | 0.0376±0.0022 | 0.0206±0.0014 |
| fno_uloss | 0.0200±0.0009 | 0.0157±0.0009 | 0.0393±0.0023 | 0.0298±0.0025 |
| fno | 0.0242±0.0034 | 0.0147±0.0004 | 0.0657±0.0063 | 0.0286±0.0025 |
| fno_physics | 0.0248±0.0035 | 0.0144±0.0003 | 0.0628±0.0044 | 0.0282±0.0027 |

## OOD · `ood_walls`

| Variant | U-MAE [full] | rel-L2 [full] | Gen. gap [full] | U-MAE [lowdata] | rel-L2 [lowdata] | Gen. gap [lowdata] |
|---|---|---|---|---|---|---|
| delta_fno | 0.0680±0.0089 | 0.0542±0.0041 | +0.0575 | 0.0405±0.0141 | 0.0529±0.0012 | +0.0236 |
| ufno | 0.3026±0.0598 | 0.1353±0.0078 | +0.2830 | 0.4167±0.1167 | 0.1642±0.0200 | +0.3791 |
| fno_uloss | 0.3052±0.1410 | 0.1670±0.0075 | +0.2852 | 0.3445±0.1501 | 0.1997±0.0146 | +0.3052 |
| fno_physics | 0.3103±0.0915 | 0.1649±0.0042 | +0.2854 | 0.3554±0.0945 | 0.1965±0.0098 | +0.2926 |
| fno | 0.3228±0.1008 | 0.1662±0.0046 | +0.2986 | 0.3848±0.1055 | 0.1958±0.0091 | +0.3191 |

## OOD · `ood_films`

| Variant | U-MAE [full] | rel-L2 [full] | Gen. gap [full] | U-MAE [lowdata] | rel-L2 [lowdata] | Gen. gap [lowdata] |
|---|---|---|---|---|---|---|
| delta_fno | 0.0617±0.0054 | 0.0209±0.0004 | +0.0511 | 0.0410±0.0003 | 0.0326±0.0013 | +0.0241 |
| fno_uloss | 0.1183±0.0435 | 0.0402±0.0007 | +0.0983 | 0.1676±0.0532 | 0.0509±0.0036 | +0.1283 |
| ufno | 0.1367±0.0274 | 0.0410±0.0030 | +0.1170 | 0.1907±0.0285 | 0.0471±0.0056 | +0.1531 |
| fno | 0.1424±0.0687 | 0.0396±0.0008 | +0.1182 | 0.1881±0.0571 | 0.0510±0.0045 | +0.1225 |
| fno_physics | 0.1465±0.0764 | 0.0398±0.0008 | +0.1217 | 0.1761±0.0593 | 0.0506±0.0047 | +0.1133 |

## OOD · `ood_bridges`

| Variant | U-MAE [full] | rel-L2 [full] | Gen. gap [full] | U-MAE [lowdata] | rel-L2 [lowdata] | Gen. gap [lowdata] |
|---|---|---|---|---|---|---|
| delta_fno | 0.0491±0.0081 | 0.0435±0.0014 | +0.0386 | 0.0440±0.0024 | 0.0590±0.0011 | +0.0271 |
| fno_uloss | 0.1669±0.0080 | 0.0678±0.0011 | +0.1469 | 0.2714±0.0180 | 0.1100±0.0075 | +0.2321 |
| ufno | 0.1834±0.0231 | 0.0686±0.0015 | +0.1638 | 0.2769±0.0393 | 0.0880±0.0059 | +0.2392 |
| fno_physics | 0.2183±0.0278 | 0.0667±0.0027 | +0.1935 | 0.3334±0.0347 | 0.1113±0.0058 | +0.2706 |
| fno | 0.2205±0.0176 | 0.0664±0.0017 | +0.1963 | 0.3385±0.0326 | 0.1093±0.0064 | +0.2728 |

## OOD · `ood_res`

| Variant | U-MAE [full] | rel-L2 [full] | Gen. gap [full] | U-MAE [lowdata] | rel-L2 [lowdata] | Gen. gap [lowdata] |
|---|---|---|---|---|---|---|
| delta_fno | 0.0143±0.0008 | 0.0133±0.0003 | +0.0037 | 0.0228±0.0019 | 0.0270±0.0007 | +0.0060 |
| ufno | 0.0713±0.0111 | 0.0303±0.0036 | +0.0517 | 0.0921±0.0101 | 0.0336±0.0024 | +0.0544 |
| fno | 0.0965±0.0093 | 0.0189±0.0007 | +0.0723 | 0.1688±0.0165 | 0.0331±0.0034 | +0.1032 |
| fno_uloss | 0.0988±0.0155 | 0.0194±0.0011 | +0.0788 | 0.1588±0.0253 | 0.0345±0.0037 | +0.1195 |
| fno_physics | 0.0990±0.0091 | 0.0188±0.0007 | +0.0742 | 0.1605±0.0190 | 0.0322±0.0036 | +0.0977 |

Each cell is mean±std over seeds for a single model trained once on the named regime and scored on the named test set at native resolution. The OOD shifts move one physically meaningful axis each — unseen walls, films outside the training band, a denser/wider bridge regime, and finer discretisation — while keeping θ well-posed (temperatures are untouched, as θ is invariant to them).
