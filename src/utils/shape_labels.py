"""Short display labels for the analysis shape parameters."""
from __future__ import annotations

SHAPE_LABEL: dict[str, str] = {
    "original_shape_Elongation":              "Elongation",
    "original_shape_Flatness":                "Flatness",
    "original_shape_LeastAxisLength":         "LeastAxis",
    "original_shape_MajorAxisLength":         "MajorAxis",
    "original_shape_Maximum2DDiameterColumn": "Max2D-Col",
    "original_shape_Maximum2DDiameterRow":    "Max2D-Row",
    "original_shape_Maximum2DDiameterSlice":  "Max2D-Slc",
    "original_shape_Maximum3DDiameter":       "Max3D",
    "original_shape_MeshVolume":              "MeshVolume",
    "original_shape_MinorAxisLength":         "MinorAxis",
    "original_shape_Sphericity":              "Sphericity",
    "original_shape_SurfaceArea":             "SurfArea",
    "original_shape_SurfaceVolumeRatio":      "SVR",
    "rib_length":                             "rib_length",
}


def shape_label(col: str) -> str:
    """Short display label for a shape-feature column; passthrough on miss."""
    return SHAPE_LABEL.get(col, col)
