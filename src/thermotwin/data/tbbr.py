"""TBBR thermal-bridge localisation — make the detection dataset comparable.

TBBR (Thermal Bridges on Building Rooftops) is real UAV thermography with **6,927 thermal-bridge
polygon annotations** — a *detection* task, in a different format from our θ-field operator. To
bring it into the unified evaluation we score a **physics-motivated heat-loss saliency** (warm
anomaly: global warm outlier AND local contrast — the same rule used for the calibrated
ThermoScenes/TUM2TWIN thermal) against the annotated bridges. Channels per scene are
``[R, G, B, thermal, height]`` (uint8); the thermal channel drives the saliency.

Comparable metrics for the TBBR rung:
  * **precision** — fraction of flagged heat-loss pixels that fall inside an annotated bridge;
  * **bridge recall** — fraction of annotated bridges hit by ≥ 1 flagged pixel;
  * **enrichment** — precision ÷ bridge-area-fraction (how much more concentrated the saliency
    is in bridges than chance; > 1 beats random).
"""

from __future__ import annotations

import numpy as np

__all__ = ["THERMAL_CHANNEL", "bridge_masks", "saliency", "score_image"]

THERMAL_CHANNEL = 3


def bridge_masks(anns: list, height: int, width: int) -> list[np.ndarray]:
    """Rasterise each annotation's polygon(s) to a boolean mask (one per bridge instance)."""
    from PIL import Image, ImageDraw

    out = []
    for a in anns:
        img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(img)
        for poly in a.get("segmentation", []):
            if isinstance(poly, list) and len(poly) >= 6:
                draw.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], fill=1)
        out.append(np.array(img, dtype=bool))
    return out


def saliency(thermal: np.ndarray, window: int = 25, k_sigma: float = 2.0, pct: float = 97.0) -> np.ndarray:
    """Heat-loss saliency on the thermal channel (warm outlier AND local-contrast)."""
    from ..data.thermoscenes import heat_loss_anomaly

    mask, _ = heat_loss_anomaly(thermal.astype(np.float64), window=window, k_sigma=k_sigma, pct=pct)
    return mask


def score_image(channels: np.ndarray, anns: list) -> dict:
    """Localisation metrics for one TBBR scene (``channels`` ``(H,W,5)``, ``anns`` its COCO list)."""
    h, w = channels.shape[:2]
    pred = saliency(channels[..., THERMAL_CHANNEL])
    masks = bridge_masks(anns, h, w)
    gt = np.zeros((h, w), dtype=bool)
    for m in masks:
        gt |= m
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    n_pred = tp + fp
    bridge_area_frac = float(gt.mean()) or 1e-9
    precision = tp / n_pred if n_pred else 0.0
    hit = sum(1 for m in masks if (pred & m).any())
    return {
        "n_bridges": len(masks),
        "n_pred_pixels": n_pred,
        "precision": round(precision, 4),
        "bridge_recall": round(hit / len(masks), 4) if masks else 0.0,
        "enrichment": round(precision / bridge_area_frac, 3),
        "bridge_area_frac": round(bridge_area_frac, 5),
    }
