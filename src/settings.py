"""Project-wide analysis constants and figure-style configuration.

Every figure-shaping value (page widths, fonts, sizes, line weights,
export formats) lives here; no inline ``figsize=`` / ``fontsize=`` /
``dpi=`` at plot call sites. Colour palettes live in :mod:`utils.colors`;
the Plotly publication template lives in :mod:`utils.plotly_theme`.
"""
from __future__ import annotations

import os

# ── Statistics ────────────────────────────────────────────────────────────────
# Threshold used in code paths that filter/log "significant" rows; ``multipletests``
# itself uses its own default of 0.05.
FDR_DISPLAY_ALPHA: float = 0.05

# Minimum complete-case sample size required to fit a model.
MIN_N_OLS: int = 100      # adjusted (rib-level) OLS

# ── Mesh extraction (NIfTI → STL) ─────────────────────────────────────────────
MESH_TARGET_FACES_PER_RIB: int = 1_000   # quadric decimation target
MESH_TAUBIN_ITER: int = 40                # post-marching-cubes smoothing iters
MESH_TAUBIN_PASS_BAND: float = 0.1        # Taubin low-pass cutoff, range [0, 2]
MESH_N_WORKERS: int = 32                 # parallel patient workers

# Threaded workers for the SSM stage (STL loading + residual cKDTree pass)
# and for the data-ingestion JSON parse pool.
SSM_LOAD_N_WORKERS: int = 32

# ── Metadata predictor groups ────────────────────────────────────────────────
META_CONTINUOUS:  list[str] = ["age", "bmi", "body_fat_pct", "pack_years",
                                "height_cm", "weight_kg"]
META_CATEGORICAL: list[str] = ["sex", "smoking_status"]

# ── Design-matrix predictor sets (three-model architecture) ──────────────────
# Design-column names; `sex`→`is_female`, `smoking_status`→`ever_smoker` are
# derived in :func:`utils.design.add_design_columns`. Smoking is ever/never
# everywhere (no 3-level term); BMI is collinear with height+weight and so is
# unadjusted-only. ADJ = adjusted multivariable model; UNADJ = per-variable
# marginal models; PRED = the demographics→geometry generator.
ADJ_PREDICTORS:   list[str] = ["is_female", "age", "height_cm", "weight_kg",
                               "body_fat_pct", "ever_smoker", "pack_years"]
UNADJ_PREDICTORS: list[str] = ["is_female", "age", "height_cm", "weight_kg", "bmi",
                               "body_fat_pct", "ever_smoker", "pack_years"]
PRED_PREDICTORS:  list[str] = ADJ_PREDICTORS

# Per-exposure back-door adjustment sets – the machine-readable DAG behind the
# targeted (set-3) analysis. sex/age are exogenous roots (total effect = marginal;
# the other root is added only for precision); the remaining exposures are not
# point-identified under assumed latent confounding, so their estimate is a
# covariate-adjusted association. weight carries its body-size parents; bmi is
# excluded (deterministic index, not a node); pack_years is fit among ever-smokers.
DAG_ADJUSTMENT_SETS: dict[str, list[str]] = {
    "is_female":    ["age"],
    "age":          ["is_female"],
    "height_cm":    ["is_female", "age"],
    "body_fat_pct": ["is_female", "age"],
    "ever_smoker":  ["is_female", "age"],
    "pack_years":   ["is_female", "age"],
    "weight_kg":    ["is_female", "age", "height_cm", "body_fat_pct"],
}
DAG_TOTAL_EFFECT: frozenset[str] = frozenset({"is_female", "age"})

# ── PC display count ─────────────────────────────────────────────────────────
# Number of PCs shown in interactive viewer sliders and in the "showcase" figures
# (PC deformations, PC scores pair-plot). Heatmap-style figures and the Styner
# triad use the full model rank instead.
N_PCS_DISPLAY: int = 7
VIEWER_PC_SLIDER_RANGE_SD: float = 2.5   # slider half-range in units of √λ_k
VIEWER_META_QUANTILES: tuple[float, float] = (0.02, 0.98)  # metadata slider range

# ── SSM quality assessment ───────────────────────────────────────────────────
EVAL_N_SPECIFICITY_SAMPLES: int = 1_000  # random samples drawn from the model for specificity
RESIDUAL_WORST_N: int = 6                # patients shown in the per-vertex residual mosaic


# ── Figure dimensions (print-scale, mm) ───────────────────────────────────────
# Nature: 89 mm one-column / 183 mm two-column. We use 85 / 170 for round
# numbers with a 4 mm safety margin. All top-level figures share the page
# content width; sub-figures opt into ``"half"`` or ``"third"`` explicitly.
FIG_WIDTH_FULL_MM:  float = 170.0
FIG_WIDTH_HALF_MM:  float = 85.0
FIG_WIDTH_THIRD_MM: float = 56.0     # 3-up panel grid (170/3 minus gutter)

# Default per-row height; multi-row figures = N × this.
FIG_ROW_HEIGHT_MM:  float = 55.0

# Plotly ``layout.margin`` in pixels at 96 dpi – used as the *floor* the
# template's ``axis.automargin`` / ``title.automargin`` grow from. ``r=40``
# gives the colorbar helper headroom; per-helper margin bumps may grow it
# further. Bottom decorations (``annotate_n``, horizontal legend) attach to
# the container, not the plot area, so ``b=45`` does not need to host them.
MARGIN_PX = dict(l=55, r=40, t=55, b=45)


# ── Typography ────────────────────────────────────────────────────────────────
# Single sans-serif family; Kaleido/Chromium picks the first available at export.
FONT_FAMILY: str = "Inter, Helvetica, Arial, sans-serif"

# Sizes in points at print scale. Nature requires ≥ 6 pt after scaling; the
# 7 pt floor (tick/legend) keeps that under a ~15 % editor downscale.
FONT_SIZE_AXIS_PT:   int = 8
FONT_SIZE_TICK_PT:   int = 7
FONT_SIZE_TITLE_PT:  int = 9
FONT_SIZE_LEGEND_PT: int = 7
FONT_SIZE_PANEL_PT:  int = 10   # bold a / b / c panel labels
FONT_SIZE_ANNOT_PT:  int = 7    # in-figure annotations (sig stars, n-labels)


# ── Geometry (lines / markers) ───────────────────────────────────────────────
LINE_WIDTH:     float = 1.0
GRID_WIDTH:     float = 0.4
MARKER_SIZE:    int   = 4
MARKER_OPACITY: float = 0.6


# ── Export ────────────────────────────────────────────────────────────────────
# Vector formats (SVG, PDF) are resolution-independent; raster export (PNG)
# uses ``KALEIDO_SCALE`` so pixel count corresponds to ``EXPORT_RASTER_DPI``
# at print scale.
EXPORT_RASTER_DPI: int   = 600
SCREEN_DPI:        int   = 96
KALEIDO_SCALE:     float = EXPORT_RASTER_DPI / SCREEN_DPI

# Default formats from ``utils.figure_export.save_fig``. PDF is opt-in because
# Plotly's PDF export of 3D Mesh3d is poor.
EXPORT_FORMATS_DEFAULT: tuple[str, ...] = ("html", "svg", "png")

# Static 3D rendering backend for plot_pc_deformations + plot_mean_shape_gallery.
# "auto" tries PyVista offscreen first; falls back to matplotlib on OpenGL error.
# Force "matplotlib" on headless Singularity images without OSMesa/EGL.
STATIC_3D_BACKEND: str = "auto"   # "auto" | "pyvista" | "matplotlib"


# ── Determinism ──────────────────────────────────────────────────────────────
# Pin hash seed so dict insertion order in serialised Plotly layouts is stable.
os.environ.setdefault("PYTHONHASHSEED", "0")


# ── Palette versioning ───────────────────────────────────────────────────────
# Embedded in figure metadata (``utils.figure_meta``); bump on any palette change.
PALETTE_VERSION: str = "2026.05"


# ── Plotting style ────────────────────────────────────────────────────────────

def apply_publication_style() -> None:
    """Install the Plotly ``nako`` template and matching matplotlib rcParams.

    Idempotent.
    """
    from utils.plotly_theme import install_template
    install_template()

    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi":        SCREEN_DPI,
        "savefig.dpi":       EXPORT_RASTER_DPI,
        "savefig.bbox":      "tight",
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         FONT_SIZE_TICK_PT,
        "axes.titlesize":    FONT_SIZE_TITLE_PT,
        "axes.labelsize":    FONT_SIZE_AXIS_PT,
        "xtick.labelsize":   FONT_SIZE_TICK_PT,
        "ytick.labelsize":   FONT_SIZE_TICK_PT,
        "legend.fontsize":   FONT_SIZE_LEGEND_PT,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    LINE_WIDTH * 0.6,
        "lines.linewidth":   LINE_WIDTH,
    })
