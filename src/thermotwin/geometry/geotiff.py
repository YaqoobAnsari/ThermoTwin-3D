"""Minimal stdlib GeoTIFF reader — just enough to register world coords onto a raster.

The cluster env has no ``rasterio`` / ``gdal`` / ``tifffile`` (only PIL, which reads pixels but
not the geo-transform). The TUM2TWIN thermal orthophoto we need to register CityGML footprints
onto is a baseline GeoTIFF whose georeferencing is a simple **scale + tie-point** (no rotation),
so a tiny tag parser suffices — the same self-contained-reader pattern as our CityJSON and
COLMAP-binary readers. We read two GeoTIFF tags:

* ``ModelPixelScaleTag`` (33550): ``(sx, sy, sz)`` ground units per pixel.
* ``ModelTiepointTag`` (33922): ``(i, j, k, X, Y, Z)`` — raster point ``(i, j)`` maps to world
  ``(X, Y)``. For a north-up image, world-X grows with column and world-Y *shrinks* with row.

That gives an affine world↔pixel map (:class:`GeoTransform`). Pixels themselves come from PIL.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["GeoTransform", "read_geotiff_transform", "read_geotiff"]

# TIFF field type -> byte size (only the types we touch).
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8}
_MODEL_PIXEL_SCALE = 33550
_MODEL_TIEPOINT = 33922


@dataclass(frozen=True)
class GeoTransform:
    """North-up affine map between world (CRS) coordinates and pixel (col, row)."""

    sx: float  # world units per pixel in X (column direction)
    sy: float  # world units per pixel in Y (row direction)
    x0: float  # world X at the tie-point pixel
    y0: float  # world Y at the tie-point pixel
    i0: float = 0.0  # tie-point pixel column
    j0: float = 0.0  # tie-point pixel row

    def world_to_pixel(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World ``(x, y)`` -> fractional ``(col, row)``."""
        col = self.i0 + (np.asarray(x, dtype=np.float64) - self.x0) / self.sx
        row = self.j0 + (self.y0 - np.asarray(y, dtype=np.float64)) / self.sy
        return col, row

    def pixel_to_world(self, col: np.ndarray, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fractional ``(col, row)`` -> world ``(x, y)``."""
        x = self.x0 + (np.asarray(col, dtype=np.float64) - self.i0) * self.sx
        y = self.y0 - (np.asarray(row, dtype=np.float64) - self.j0) * self.sy
        return x, y


def _read_tags(blob: bytes) -> dict[int, tuple]:
    """Parse the first IFD of a (classic) TIFF, returning ``{tag: values}``."""
    endian = "<" if blob[:2] == b"II" else ">"
    (ifd_off,) = struct.unpack(endian + "I", blob[4:8])
    (count,) = struct.unpack(endian + "H", blob[ifd_off : ifd_off + 2])
    tags: dict[int, tuple] = {}
    for i in range(count):
        e = ifd_off + 2 + i * 12
        tag, typ, n = struct.unpack(endian + "HHI", blob[e : e + 8])
        size = _TYPE_SIZE.get(typ, 1) * n
        voff = e + 8 if size <= 4 else struct.unpack(endian + "I", blob[e + 8 : e + 12])[0]
        if typ == 12:  # double
            tags[tag] = struct.unpack(endian + f"{n}d", blob[voff : voff + 8 * n])
        elif typ == 3:  # short
            tags[tag] = struct.unpack(endian + f"{n}H", blob[voff : voff + 2 * n])
    return tags


def read_geotiff_transform(path: str | Path) -> GeoTransform:
    """Read a north-up GeoTIFF's scale + tie-point into a :class:`GeoTransform`.

    Raises:
        ValueError: if the file lacks the pixel-scale / tie-point tags (not a simple
            north-up GeoTIFF — would need a full ``ModelTransformationTag`` handler).
    """
    tags = _read_tags(Path(path).read_bytes())
    scale = tags.get(_MODEL_PIXEL_SCALE)
    tie = tags.get(_MODEL_TIEPOINT)
    if not scale or not tie:
        raise ValueError(f"{path}: no ModelPixelScale/ModelTiepoint tags (not a north-up GeoTIFF)")
    i0, j0, _k0, x0, y0, _z0 = tie[:6]
    return GeoTransform(sx=float(scale[0]), sy=float(scale[1]), x0=float(x0), y0=float(y0), i0=float(i0), j0=float(j0))


def read_geotiff(path: str | Path) -> tuple[np.ndarray, GeoTransform]:
    """Read a GeoTIFF into ``(image[H, W, ...], GeoTransform)`` (pixels via PIL)."""
    from PIL import Image

    img = np.asarray(Image.open(path))
    return img, read_geotiff_transform(path)
