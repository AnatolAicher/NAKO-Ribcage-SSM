"""Nine-panel methodology figure for one display patient.

Walks one patient through the full pipeline — raw MRI slices, segmentation
overlay, 3D segmentation, PyRadiomics descriptors on a single rib, the
four mesh-to-template registration stages, and a combined GPA + PCA panel
showing the PC1 ±2σ mean-shape envelope alongside the patient and a per-PC
sparkline score table. Wired into the ``visualizations`` stage; rendered
when a preset carries a ``methodology_figure:`` block.

Outputs ``<out_stem>.{pdf,svg,png}`` + a ``figures_manifest.json`` row.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyvista as pv
import settings as S
from data_ingestion.loaders import ANALYSIS_SHAPE_COLS, SHAPE_FEATURES
from ssm.mesh_extraction import split_rib_components
from utils.colors import cmap as palette_cmap, rib_colors
from utils.config import read_parquet
from utils.figure_export import save_fig
from utils.mesh_mpl import (
    add_mesh as add_mesh_3d,
    add_meshes as add_meshes_3d,
    pv_to_triangles,
    set_cube_bounds,
    style_axis,
)
from utils.paths import stage_dir
from utils.rib_labels import (
    anatomical_to_seg,
    vert_to_anatomical,
)
from utils.run_dir import patient_stl_dir, read_metadata

logger = logging.getLogger(__name__)


# ── Canonical 24-rib order (matches src/ssm/pca_surface.py::RIB_ORDER) ─────
RIB_LABELS: list[int] = list(range(40, 52))
RIB_SIDES:  list[str] = ["L", "R"]
RIB_ORDER:  list[tuple[int, str]] = [
    (lab, side) for lab in RIB_LABELS for side in RIB_SIDES
]


# ── Visual palette ─────────────────────────────────────────────────────────
TEMPLATE_COLOR = "#9CA3AF"   # slate-400 (template ribs in panels E/F)

# Inverse-of-side coloring for B-panel seg overlay; we reuse rib_colors() but
# clip alpha so the MRI shows through.
SEG_ALPHA = 0.55


# ── Descriptor display labels + units ──────────────────────────────────────
UNITS: dict[str, str] = {
    "original_shape_Elongation":              "-",
    "original_shape_Flatness":                "-",
    "original_shape_LeastAxisLength":         "mm",
    "original_shape_MajorAxisLength":         "mm",
    "original_shape_Maximum2DDiameterColumn": "mm",
    "original_shape_Maximum2DDiameterRow":    "mm",
    "original_shape_Maximum2DDiameterSlice":  "mm",
    "original_shape_Maximum3DDiameter":       "mm",
    "original_shape_MeshVolume":              "mm^3",
    "original_shape_MinorAxisLength":         "mm",
    "original_shape_Sphericity":              "-",
    "original_shape_SurfaceArea":             "mm^2",
    "original_shape_SurfaceVolumeRatio":      "1/mm",
    "original_shape_VoxelVolume":             "mm^3",
    "rib_length":                             "mm",
}


# ── Public entry point ─────────────────────────────────────────────────────

def render(
    run_dir: Path | str,
    data_config: dict,
    preset_cfg: dict[str, Any],
    out_stem: Path | str,
) -> None:
    """Render the nine-panel methodology figure for one display patient.

    Parameters
    ----------
    run_dir
        Top-level pipeline run directory.
    data_config
        Parsed ``data/data_config.yaml`` (must include
        ``paths.mri_nifti_base`` + ``paths.mri_nifti_pattern``).
    preset_cfg
        ``methodology_figure:`` block from the preset YAML.
    out_stem
        Output path *without* extension.
    """
    run_dir  = Path(run_dir)
    out_stem = Path(out_stem)

    pid             = int(preset_cfg["display_patient"])
    rib_cfg         = preset_cfg.get("radiomics_rib") or {"vert_level": 14, "side": "R"}
    radiomics_vert  = int(rib_cfg["vert_level"])
    radiomics_side  = str(rib_cfg["side"]).upper()
    n_pcs_table     = int(preset_cfg.get("n_pcs_for_score_table", 10))

    bundle = _load_inputs(
        run_dir, data_config, pid,
        radiomics_vert, radiomics_side,
    )

    fig = _build_figure(bundle, pid, n_pcs_table)

    def _renderer(path: Path) -> None:
        fig.savefig(path, dpi=S.EXPORT_RASTER_DPI, bbox_inches="tight", pad_inches=0)

    try:
        save_fig(
            go.Figure(),
            out_stem,
            formats=("pdf", "svg", "png"),
            title="Methodology overview",
            width_class="full",
            static_renderer=_renderer,
        )
    finally:
        plt.close(fig)
    logger.info(f"Methodology figure → {out_stem}.{{pdf,svg,png}}")


# ── Input loading ──────────────────────────────────────────────────────────

def _load_inputs(
    run_dir: Path,
    data_config: dict,
    pid: int,
    radiomics_vert: int,
    radiomics_side: str,
) -> dict[str, Any]:
    """Load and validate every artifact the figure consumes."""
    meta = read_metadata(run_dir)
    extracted_root  = Path(meta["paths"]["extracted_stl_dir"])
    registered_root = Path(meta["paths"]["registered_stl_dir"])
    ssm_pca_dir     = stage_dir(run_dir, "ssm_pca")
    ingestion_dir   = stage_dir(run_dir, "ingestion")

    # MRI + seg volumes
    mri_path = _resolve_volume_path(data_config, "mri", pid)
    seg_path = _resolve_volume_path(data_config, "seg", pid)
    if not mri_path.exists():
        raise FileNotFoundError(
            f"MRI volume not found: {mri_path}  "
            f"(check data_config.paths.mri_nifti_pattern)"
        )
    if not seg_path.exists():
        raise FileNotFoundError(
            f"Segmentation NIfTI not found: {seg_path}  "
            f"(check data_config.paths.nifti_vert_rib_pattern)"
        )
    mri_vol, mri_zooms = _load_volume(mri_path)
    seg_vol, _         = _load_volume(seg_path)

    # Side-encoded segmentation: 0=bg, lab+100=L, lab+200=R. The on-disk seg
    # uses the same label for both sides; mesh_extraction.split_rib_components
    # splits them by connected-component centroid in RAS (lower x = anatomical L).
    seg_side = np.zeros_like(seg_vol, dtype=np.int16)
    for lab in RIB_LABELS:
        res = split_rib_components(seg_vol, lab)
        seg_side[res["left_mask"]]  = lab + 100
        seg_side[res["right_mask"]] = lab + 200

    # SSM artifacts
    shapes_path = ssm_pca_dir / "shapes_registered.npz"
    if not shapes_path.exists():
        raise RuntimeError(
            f"{shapes_path} missing; rerun pipeline with stages.ssm_pca: true."
        )
    shapes_data = np.load(shapes_path, mmap_mode="r")
    patient_ids = np.asarray(shapes_data["patient_ids"])
    matches     = np.where(patient_ids == pid)[0]
    if matches.size == 0:
        first_5 = ", ".join(str(int(p)) for p in patient_ids[:5])
        raise RuntimeError(
            f"patient {pid} not found in {shapes_path.name}; "
            f"first 5 available IDs: {first_5}"
        )
    shape_idx     = int(matches[0])
    patient_shape = np.asarray(shapes_data["shapes"][shape_idx])

    faces_path = ssm_pca_dir / "template_faces.npy"
    if not faces_path.exists():
        raise RuntimeError(
            f"{faces_path} missing; rerun pipeline with stages.ssm_pca: true."
        )
    template_faces = np.load(faces_path)

    mean_path = ssm_pca_dir / "mean_shape_surface.npy"
    if not mean_path.exists():
        raise RuntimeError(f"{mean_path} missing; rerun ssm_pca stage.")
    mean_shape = np.load(mean_path)

    rib_offsets = np.load(ssm_pca_dir / "rib_offsets.npy")

    pca_npz       = np.load(ssm_pca_dir / "pca_surface.npz")
    pc_components = pca_npz["components"]                      # (K, n_pts*3)
    pc_variance   = pca_npz["explained_variance"]              # (K,)
    pc_var_ratio  = pca_npz["explained_variance_ratio"]        # (K,)

    scores_df   = pd.read_csv(ssm_pca_dir / "pc_scores_surface.csv")
    patient_row = scores_df[scores_df["patient_id"] == pid]
    if patient_row.empty:
        raise RuntimeError(
            f"patient {pid} not in pc_scores_surface.csv; rerun ssm_pca."
        )

    # STL caches
    extracted_patient_dir   = patient_stl_dir(extracted_root, pid)
    registered_patient_dir  = patient_stl_dir(registered_root, pid)
    template_id_path        = registered_root / "template_id.txt"
    if not template_id_path.exists():
        raise RuntimeError(
            f"{template_id_path} missing; rerun ssm_registration stage."
        )
    template_pid           = template_id_path.read_text().strip()
    extracted_template_dir = patient_stl_dir(extracted_root, template_pid)

    _verify_stls(extracted_patient_dir,  pid,           tag="patient extracted")
    _verify_stls(extracted_template_dir, template_pid,  tag="template extracted")
    _verify_stls(registered_patient_dir, pid,           tag="patient registered")

    # Methodology-only per-stage dumps: written by RibRegistration.scala
    # when --methodology-patient-id matches this pid. Each subdir holds
    # the same 24 per-rib STLs in their stage-specific frame.
    method_root         = registered_patient_dir / "_methodology"
    cage_patient_dir    = method_root / "cage_patient"
    per_rib_template_dir = method_root / "per_rib_template"
    gp_fit_template_dir = method_root / "gp_fit_template"
    for d, tag in (
        (cage_patient_dir,    "cage_patient (after whole-cage)"),
        (per_rib_template_dir,"per_rib_template (after per-rib ICP)"),
        (gp_fit_template_dir, "gp_fit_template (after Gaussian Process)"),
    ):
        if not d.is_dir():
            raise RuntimeError(
                f"methodology dump missing: {d}. Re-run the ssm_registration "
                f"stage with `methodology_figure.display_patient: {pid}` set "
                f"in the preset so RibRegistration dumps the per-stage STLs."
            )
        _verify_stls(d, pid, tag=tag)

    # Radiomics row
    parquet_path = ingestion_dir / "analytic_clean.parquet"
    if not parquet_path.exists():
        raise RuntimeError(
            f"{parquet_path} missing; rerun ingestion stage."
        )
    parquet_df = read_parquet(parquet_path)
    # Parquet stores side as "Left"/"Right" (loaders._parse_patient); the rest
    # of this module uses the short "L"/"R" form (matches STL filenames).
    side_long = "Right" if radiomics_side == "R" else "Left"
    mask = (
        (parquet_df["patient_id"] == pid)
        & (parquet_df["vert_level"] == radiomics_vert)
        & (parquet_df["side"] == side_long)
    )
    if not mask.any():
        rib_anat = vert_to_anatomical(radiomics_vert)
        raise RuntimeError(
            f"no radiomics row for rib {rib_anat} {radiomics_side} "
            f"(vert_level={radiomics_vert}) of patient {pid}; rerun ingestion."
        )
    radiomics_row = parquet_df[mask].iloc[0]

    return dict(
        # Volumes
        mri_vol=mri_vol, mri_zooms=mri_zooms,
        seg_vol=seg_vol, seg_side=seg_side,
        # SSM
        patient_shape=patient_shape,
        mean_shape=mean_shape, template_faces=template_faces,
        rib_offsets=rib_offsets,
        pc_components=pc_components, pc_variance=pc_variance,
        pc_var_ratio=pc_var_ratio,
        scores_df=scores_df, patient_scores=patient_row.iloc[0],
        # STLs
        extracted_patient_dir=extracted_patient_dir,
        extracted_template_dir=extracted_template_dir,
        registered_patient_dir=registered_patient_dir,
        template_pid=template_pid,
        # Methodology per-stage dumps (panels 6, 7, 8)
        cage_patient_dir=cage_patient_dir,
        per_rib_template_dir=per_rib_template_dir,
        gp_fit_template_dir=gp_fit_template_dir,
        # Radiomics
        radiomics_row=radiomics_row,
        radiomics_vert=radiomics_vert, radiomics_side=radiomics_side,
    )


def _resolve_volume_path(data_config: dict, kind: str, pid: int) -> Path:
    """Resolve MRI or segmentation NIfTI path from data_config + patient id."""
    paths_cfg = data_config["paths"]
    if kind == "mri":
        base    = paths_cfg.get("mri_nifti_base")
        pattern = paths_cfg.get("mri_nifti_pattern")
        if not base or not pattern:
            raise RuntimeError(
                "data_config.paths.mri_nifti_base + mri_nifti_pattern must be set "
                "to render the methodology figure (panel a)."
            )
    elif kind == "seg":
        base    = paths_cfg["nifti_base"]
        pattern = paths_cfg["nifti_vert_rib_pattern"]
    else:
        raise ValueError(f"unknown volume kind {kind!r}")
    block = pid // 1000
    p = pattern.replace("{base}", base).format(block=block, patient_id=pid)
    return Path(p)


def _load_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img   = nib.as_closest_canonical(nib.load(str(path)))
    data  = np.asarray(img.dataobj)
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    return data, zooms


def _verify_stls(stl_dir: Path, pid: str | int, *, tag: str) -> None:
    missing = [
        f"{pid}_rib{lab}_{side}.stl"
        for lab, side in RIB_ORDER
        if not (stl_dir / f"{pid}_rib{lab}_{side}.stl").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{tag} STLs missing in {stl_dir}: {missing[:3]}{'…' if len(missing) > 3 else ''} "
            f"({len(missing)}/{len(RIB_ORDER)} files absent)."
        )


def _centroid_bbox_slices(seg_vol: np.ndarray, labels: list[int]) -> tuple[int, int, int]:
    """Slice indices (x, y, z) through the centroid of the rib-mask bbox."""
    mask = np.isin(seg_vol, labels)
    if not mask.any():
        raise RuntimeError(
            "rib-mask bbox is empty — segmentation has no rib labels."
        )
    coords = np.argwhere(mask)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    centroid = ((lo + hi) // 2).astype(int)
    return int(centroid[0]), int(centroid[1]), int(centroid[2])


# ── Figure assembly ────────────────────────────────────────────────────────

def _build_figure(bundle: dict, pid: int, n_pcs_table: int) -> plt.Figure:
    """Lay out the A4-portrait panel mosaic and render every panel.

    Five rows: four 2-column rows (A/B, C/D, E/F, G/H) plus a final
    single-panel row (I) that spans the full page width for the merged
    GPA + PCA visualisation.
    """
    # A4 portrait: 210mm x 297mm
    fig = plt.figure(
        figsize=(210.0 / 25.4, 297.0 / 25.4),
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")

    mosaic = [
        ["A1", "A2", "A3", "B1", "B2", "B3"],
        ["C",  "C",  "C",  "D",  "D",  "D" ],
        ["E",  "E",  "E",  "F",  "F",  "F" ],
        ["G",  "G",  "G",  "H",  "H",  "H" ],
        ["I",  "I",  "I",  "I",  "I",  "I" ],
    ]
    axes = fig.subplot_mosaic(
        mosaic,
        gridspec_kw=dict(
            hspace=0.45, wspace=0.30,
            left=0.0, right=1.0, top=1.0, bottom=0.0,
        ),
    )
    for ax in axes.values():
        ax.set_facecolor("none")

    # Intensity window for the MRI underlay — once, reused across A/B panels.
    nz = bundle["mri_vol"][bundle["mri_vol"] > 0]
    if nz.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = (float(v) for v in np.percentile(nz, (0.5, 99.5)))

    slice_ix = _centroid_bbox_slices(bundle["seg_vol"], RIB_LABELS)

    _panel_a_mri(
        (axes["A1"], axes["A2"], axes["A3"]),
        bundle["mri_vol"], bundle["mri_zooms"], slice_ix, vmin, vmax,
    )
    _panel_b_seg(
        (axes["B1"], axes["B2"], axes["B3"]),
        bundle["mri_vol"], bundle["seg_side"],
        bundle["mri_zooms"], slice_ix, vmin, vmax,
    )

    _panel_c_extracted(axes["C"], bundle["extracted_patient_dir"], pid)
    _panel_d_radiomics(
        axes["D"], bundle["extracted_patient_dir"], pid,
        bundle["radiomics_vert"], bundle["radiomics_side"],
        bundle["radiomics_row"],
    )
    _panel_e_before(
        axes["E"],
        bundle["extracted_patient_dir"], pid,
        bundle["extracted_template_dir"], bundle["template_pid"],
    )
    _panel_f_whole_cage(
        axes["F"],
        bundle["cage_patient_dir"], pid,
        bundle["extracted_template_dir"], bundle["template_pid"],
    )
    _panel_g_per_rib(
        axes["G"],
        bundle["cage_patient_dir"], pid,
        bundle["per_rib_template_dir"], pid,
        bundle["radiomics_vert"], bundle["radiomics_side"],
    )
    _panel_h_gp_fit(
        axes["H"],
        bundle["cage_patient_dir"], pid,
        bundle["gp_fit_template_dir"], pid,
        bundle["radiomics_vert"], bundle["radiomics_side"],
    )
    _panel_i_gpa_pca(
        axes["I"], bundle["mean_shape"], bundle["patient_shape"],
        bundle["pc_components"], bundle["pc_variance"], bundle["pc_var_ratio"],
        bundle["template_faces"], bundle["rib_offsets"],
        bundle["scores_df"], bundle["patient_scores"],
        n_pcs_table,
    )

    panel_titles: list[tuple[list[str], str, str]] = [
        (["A1", "A2", "A3"], "A. Raw MRI",                   "patient — VIBE"),
        (["B1", "B2", "B3"], "B. Segmented MRI",             "ribs segmented"),
        (["C"],              "C. 3D mesh extraction",        "24 per-rib meshes"),
        (["D"],              "D. Descriptors",               "PyRadiomics on rib 7 R"),
        (["E"],              "E. Before alignment",          "patient + template — scanner-native frames"),
        (["F"],              "F. After whole-cage",          "rigid Procrustes — cage frame"),
        (["G"],              "G. After per-rib registration","sim. + rigid ICP — cage frame"),
        (["H"],              "H. After Gaussian-Process fit","non-rigid surface deformation — cage frame"),
        (["I"],              "I. After GPA & PCA",           "mean shape ±2σ along PC1 (viridis = |Δ| from mean, mm) + patient PC scores"),
    ]
    for keys, title, subtitle in panel_titles:
        _add_panel_card(fig, [axes[k] for k in keys])
        _add_panel_title(fig, [axes[k] for k in keys], title, subtitle)

    return fig


# ── Panels A/B: MRI + segmentation slices ─────────────────────────────────

def _panel_a_mri(
    axs: tuple[plt.Axes, plt.Axes, plt.Axes],
    vol: np.ndarray,
    zooms: np.ndarray,
    slice_ix: tuple[int, int, int],
    vmin: float,
    vmax: float,
) -> None:
    cor, sag, axi = _three_slices(vol, slice_ix)
    ratios        = _aspect_ratios(zooms)
    names         = ["Coronal", "Sagittal", "Axial"]

    for ax, img, aspect, corners, name in zip(
        axs,
        [cor, sag, axi],
        [ratios["coronal"], ratios["sagittal"], ratios["axial"]],
        [("R", "L", "S", "I"), ("A", "P", "S", "I"), ("R", "L", "A", "P")],
        names,
    ):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        # adjustable='datalim' keeps the axis box at its gridspec size and
        # flexes the data limits instead of shrinking the axis (the
        # imshow-default 'box' behaviour, which made the MRI cells appear
        # narrower than C–H).
        ax.set_aspect(aspect, adjustable="datalim")
        _strip_ticks(ax)
        _orientation_labels(ax, *corners)
        _slice_caption(ax, name)


def _panel_b_seg(
    axs: tuple[plt.Axes, plt.Axes, plt.Axes],
    mri_vol: np.ndarray,
    seg_side: np.ndarray,
    zooms: np.ndarray,
    slice_ix: tuple[int, int, int],
    vmin: float,
    vmax: float,
) -> None:
    mri_cor, mri_sag, mri_axi = _three_slices(mri_vol, slice_ix)
    seg_cor, seg_sag, seg_axi = _three_slices(seg_side, slice_ix)
    ratios                    = _aspect_ratios(zooms)
    names                     = ["Coronal", "Sagittal", "Axial"]

    overlays = [
        _seg_overlay_rgba(seg_cor),
        _seg_overlay_rgba(seg_sag),
        _seg_overlay_rgba(seg_axi),
    ]

    for ax, mri, ov, aspect, corners, name in zip(
        axs,
        [mri_cor, mri_sag, mri_axi],
        overlays,
        [ratios["coronal"], ratios["sagittal"], ratios["axial"]],
        [("R", "L", "S", "I"), ("A", "P", "S", "I"), ("R", "L", "A", "P")],
        names,
    ):
        ax.imshow(mri, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax.imshow(ov, origin="lower", interpolation="nearest")
        # See _panel_a_mri for the adjustable='datalim' rationale.
        ax.set_aspect(aspect, adjustable="datalim")
        _strip_ticks(ax)
        _orientation_labels(ax, *corners)
        _slice_caption(ax, name)


def _three_slices(vol: np.ndarray, ix: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coronal / sagittal / axial slices from a RAS volume.

    X / Y axes are flipped for radiological convention (patient-right on
    image-left for coronal & axial; anterior on image-left for sagittal).
    """
    x, y, z = ix
    coronal  = vol[::-1, y, :].T    # rows=Z (S↑), cols=flipped X (R left)
    sagittal = vol[x, ::-1, :].T    # rows=Z (S↑), cols=flipped Y (A left)
    axial    = vol[::-1, :, z].T    # rows=Y (A↑), cols=flipped X (R left)
    return coronal, sagittal, axial


def _aspect_ratios(zooms: np.ndarray) -> dict[str, float]:
    """Per-plane pixel aspect = (mm per row) / (mm per col)."""
    zx, zy, zz = (float(z) for z in zooms[:3])
    return {
        "coronal":  zz / zx,
        "sagittal": zz / zy,
        "axial":    zy / zx,
    }


def _seg_overlay_rgba(slice_2d: np.ndarray) -> np.ndarray:
    """RGBA overlay coloured per (rib level, side).

    ``slice_2d`` carries the side-encoded labels produced in
    :func:`_load_inputs` (``lab + 100`` left, ``lab + 200`` right).
    """
    h, w = slice_2d.shape
    out  = np.zeros((h, w, 4), dtype=np.float32)
    for lab in RIB_LABELS:
        for side, code in (("L", lab + 100), ("R", lab + 200)):
            mask = slice_2d == code
            if not mask.any():
                continue
            rgb = _hex_to_rgb01(rib_colors([lab], [side])[0])
            out[mask, 0] = rgb[0]
            out[mask, 1] = rgb[1]
            out[mask, 2] = rgb[2]
            out[mask, 3] = SEG_ALPHA
    return out


def _hex_to_rgb01(hex_str: str) -> tuple[float, float, float]:
    s = hex_str.lstrip("#")
    return (
        int(s[0:2], 16) / 255.0,
        int(s[2:4], 16) / 255.0,
        int(s[4:6], 16) / 255.0,
    )


def _strip_ticks(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _slice_caption(ax: plt.Axes, name: str) -> None:
    ax.text(
        0.5, -0.04, name,
        transform=ax.transAxes,
        fontsize=S.FONT_SIZE_TICK_PT,
        color="#444444",
        va="top", ha="center",
    )


def _orientation_labels(ax: plt.Axes, left: str, right: str, top: str, bottom: str) -> None:
    kw = dict(transform=ax.transAxes, color="white",
              fontsize=S.FONT_SIZE_TICK_PT, fontweight="bold",
              va="center", ha="center")
    ax.text(0.05, 0.5, left,   **kw)
    ax.text(0.95, 0.5, right,  **kw)
    ax.text(0.5, 0.95, top,    **kw)
    ax.text(0.5, 0.05, bottom, **kw)


# ── 3D rendering helpers ───────────────────────────────────────────────────
# Headless-safe: all 3D panels render via matplotlib's mpl_toolkits.mplot3d
# (Poly3DCollection). PyVista is used only for STL parsing — never for VTK
# rendering — so no OSMesa / EGL / X server is required at runtime.

def _add_3d_inset(ax: plt.Axes, *, bottom: float = 0.0) -> plt.Axes:
    """Convert a 2D mosaic cell into a 3D axis filling its area.

    ``bottom`` reserves space below the 3D axis (in fraction of the parent
    cell) for legends or captions drawn on the parent ``ax``.
    """
    ax.set_axis_off()
    return ax.inset_axes([0.0, bottom, 1.0, 1.0 - bottom], projection="3d")


def _read_patient_meshes(stl_dir: Path, pid: str | int) -> list[tuple[int, str, np.ndarray, np.ndarray]]:
    """``(lab, side, verts, faces)`` for every (40..51, L/R) STL in ``stl_dir``."""
    out = []
    for lab, side in RIB_ORDER:
        p = stl_dir / f"{pid}_rib{lab}_{side}.stl"
        verts, faces = pv_to_triangles(pv.read(str(p)))
        out.append((lab, side, verts, faces))
    return out


def _per_rib_split(
    shape: np.ndarray,
    template_faces: np.ndarray,
    rib_offsets: np.ndarray,
) -> list[tuple[int, str, np.ndarray, np.ndarray]]:
    """Slice a stacked SSM mesh into one ``(verts, faces)`` per rib.

    ``shape``      stacked ``(n_pts_total, 3)`` vertex array
    ``template_faces``   ``(n_faces, 3)`` index array over the stacked verts
    ``rib_offsets``      length-24 start offsets in canonical ``RIB_ORDER``

    Faces straddling a rib boundary (rare but possible) are dropped. Returns
    one tuple per entry of ``RIB_ORDER`` so caller can map to the per-rib
    colour palette directly.
    """
    n_pts = shape.shape[0]
    offsets = list(rib_offsets) + [n_pts]
    faces = np.asarray(template_faces, dtype=np.int64)
    ends  = np.asarray(offsets[1:], dtype=np.int64)
    v0    = np.searchsorted(ends, faces[:, 0], side="right")
    v1    = np.searchsorted(ends, faces[:, 1], side="right")
    v2    = np.searchsorted(ends, faces[:, 2], side="right")
    same  = (v0 == v1) & (v1 == v2)
    face_rib = np.where(same, v0, -1)

    out: list[tuple[int, str, np.ndarray, np.ndarray]] = []
    for ri, (lab, side) in enumerate(RIB_ORDER):
        s, e   = offsets[ri], offsets[ri + 1]
        f_sub  = faces[face_rib == ri] - s
        verts  = shape[s:e]
        out.append((lab, side, verts, f_sub))
    return out


# ── Panel C: 3D segmentation ───────────────────────────────────────────────

def _panel_c_extracted(ax: plt.Axes, stl_dir: Path, pid: int) -> None:
    ax3d = _add_3d_inset(ax)
    meshes: list[tuple[np.ndarray, np.ndarray, str]] = []
    all_verts: list[np.ndarray] = []
    for lab, side, verts, faces in _read_patient_meshes(stl_dir, pid):
        meshes.append((verts, faces, rib_colors([lab], [side])[0]))
        all_verts.append(verts)
    add_meshes_3d(ax3d, meshes)
    set_cube_bounds(ax3d, all_verts)
    style_axis(ax3d)


# ── Panel D: PyRadiomics descriptors on one rib ────────────────────────────

def _panel_d_radiomics(
    ax: plt.Axes,
    stl_dir: Path,
    pid: int,
    vert_level: int,
    side: str,
    row: pd.Series,
) -> None:
    rib_anat     = vert_to_anatomical(vert_level)
    seg_lab      = anatomical_to_seg(rib_anat)
    stl_path     = stl_dir / f"{pid}_rib{seg_lab}_{side}.stl"
    verts, faces = pv_to_triangles(pv.read(str(stl_path)))
    centroid     = verts.mean(axis=0)

    # Principal axes via PCA on vertex positions; only the first (major axis)
    # is drawn — to illustrate one descriptor in situ, not all three.
    cov         = np.cov(verts.T)
    evals, evec = np.linalg.eigh(cov)
    major_dir   = evec[:, int(np.argmax(evals))]
    major_len   = float(row["original_shape_MajorAxisLength"])

    # 1/3-left for the 3D render, 2/3-right for the descriptor table.
    ax.set_axis_off()
    ax3d = ax.inset_axes([0.0, 0.0, 0.34, 1.0], projection="3d")
    add_mesh_3d(ax3d, verts, faces,
                color=rib_colors([seg_lab], [side])[0], alpha=0.75)

    p1 = centroid - 0.5 * major_len * major_dir
    p2 = centroid + 0.5 * major_len * major_dir
    ax3d.plot(
        [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
        color="black", linewidth=0.9, linestyle=":",
    )
    # 2D-axes coords so the label always lands in the bottom of the panel
    # area regardless of the 3D camera angle (text in data coords overlapped
    # the rib in some views).
    ax3d.text2D(
        0.5, 0.06, "Major axis length",
        transform=ax3d.transAxes,
        color="black", fontsize=S.FONT_SIZE_TICK_PT,
        ha="center", va="center",
    )

    set_cube_bounds(ax3d, [verts])
    style_axis(ax3d)

    ax_tbl = ax.inset_axes([0.34, 0.0, 0.66, 1.0])
    ax_tbl.set_axis_off()
    _descriptor_table(ax_tbl, row, rib_anat, side)


def _descriptor_table(ax: plt.Axes, row: pd.Series, rib_anat: int, side: str) -> None:
    short = lambda c: c.replace("original_shape_", "")
    # All shape features plus rib_length for completeness.
    cols = SHAPE_FEATURES + ["rib_length"]
    cell_text = []
    for c in cols:
        val   = row.get(c, np.nan)
        unit  = UNITS.get(c, "")
        if pd.isna(val):
            vtxt = "—"
        elif unit in ("mm^2", "mm^3", "1/mm") or abs(val) >= 100:
            vtxt = f"{val:.1f}"
        else:
            vtxt = f"{val:.3g}"
        cell_text.append([short(c), vtxt, unit])

    title = f"Rib {rib_anat} {side} — shape descriptors"
    ax.set_title(title, fontsize=S.FONT_SIZE_TICK_PT,
                 fontweight="bold", loc="left", pad=4)
    tbl = ax.table(
        cellText=cell_text,
        colLabels=["descriptor", "value", "unit"],
        cellLoc="left",
        colLoc="left",
        loc="upper center",
        colWidths=[0.6, 0.25, 0.15],
        bbox=[0.0, 0.0, 1.0, 0.95],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(S.FONT_SIZE_TICK_PT)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.3)
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f0f0f0")


# ── Panel E: before alignment (two side-by-side viewports) ─────────────────

def _panel_e_before(
    ax: plt.Axes,
    patient_dir: Path, patient_pid: int,
    template_dir: Path, template_pid: str | int,
) -> None:
    """Patient + template, each rendered in its own scanner-native frame.

    Two side-by-side 3D viewports communicate "two independent meshes in
    different frames" — overlaying both in one viewport often makes them
    look already aligned because typical scanner conventions put them at
    similar mm coordinates.
    """
    ax.set_axis_off()
    ax_p = ax.inset_axes([0.00, 0.10, 0.48, 0.90], projection="3d")
    ax_t = ax.inset_axes([0.52, 0.10, 0.48, 0.90], projection="3d")

    meshes_p: list[tuple[np.ndarray, np.ndarray, str]] = []
    verts_p: list[np.ndarray] = []
    for lab, side, verts, faces in _read_patient_meshes(patient_dir, patient_pid):
        meshes_p.append((verts, faces, rib_colors([lab], [side])[0]))
        verts_p.append(verts)
    add_meshes_3d(ax_p, meshes_p)
    set_cube_bounds(ax_p, verts_p)
    style_axis(ax_p)

    meshes_t: list[tuple[np.ndarray, np.ndarray, str]] = []
    verts_t: list[np.ndarray] = []
    for _, _, verts, faces in _read_patient_meshes(template_dir, template_pid):
        meshes_t.append((verts, faces, TEMPLATE_COLOR))
        verts_t.append(verts)
    add_meshes_3d(ax_t, meshes_t)
    set_cube_bounds(ax_t, verts_t)
    style_axis(ax_t)

    ax.text(0.24, 0.13, "patient",  transform=ax.transAxes,
            ha="center", va="center", fontsize=S.FONT_SIZE_TICK_PT)
    ax.text(0.76, 0.13, "template", transform=ax.transAxes,
            ha="center", va="center", fontsize=S.FONT_SIZE_TICK_PT)


# ── Panel F: after whole-cage rigid alignment ──────────────────────────────

def _panel_f_whole_cage(
    ax: plt.Axes,
    cage_patient_dir: Path, patient_pid: int,
    template_extracted_dir: Path, template_pid: str | int,
) -> None:
    """Patient (in cage frame, per-rib palette) + template (grey, in its
    own = cage frame). Both meshes overlap in cage frame after the
    whole-cage rigid Procrustes pre-alignment; template is still
    template-shaped (no deformation yet).
    """
    ax3d = _add_3d_inset(ax, bottom=0.12)
    meshes: list[tuple[np.ndarray, np.ndarray, str]] = []
    alphas: list[float] = []
    all_verts: list[np.ndarray] = []
    for lab, side, verts, faces in _read_patient_meshes(cage_patient_dir, patient_pid):
        meshes.append((verts, faces, rib_colors([lab], [side])[0]))
        alphas.append(0.80)
        all_verts.append(verts)
    for _, _, verts, faces in _read_patient_meshes(template_extracted_dir, template_pid):
        meshes.append((verts, faces, TEMPLATE_COLOR))
        alphas.append(0.30)
        all_verts.append(verts)
    add_meshes_3d(ax3d, meshes, alpha=alphas)
    set_cube_bounds(ax3d, all_verts)
    style_axis(ax3d)
    _patient_plus_one_legend(
        ax, "patient (cage frame)",
        "template (raw)", TEMPLATE_COLOR,
    )


# ── Panels G/H: per-rib steps — full cage + single-rib zoom ────────────────

def _two_viewport_pair(
    ax: plt.Axes,
    cage_patient_dir: Path, patient_pid: int,
    template_dir: Path, template_pid: int,
    radiomics_vert: int, radiomics_side: str,
    *,
    legend_left_label: str,
    legend_right_label: str,
) -> None:
    """Shared layout for panels G and H: full-cage view on the left, single-rib
    zoom on the right (using the methodology-figure rib id).
    """
    ax.set_axis_off()
    ax_full = ax.inset_axes([0.00, 0.12, 0.48, 0.88], projection="3d")
    ax_rib  = ax.inset_axes([0.52, 0.12, 0.48, 0.88], projection="3d")

    # Full cage: merge all 24 patient + 24 template meshes into one
    # Poly3DCollection so matplotlib's per-face z-sort covers the whole scene.
    full_meshes: list[tuple[np.ndarray, np.ndarray, str]] = []
    full_alphas: list[float] = []
    full_verts: list[np.ndarray] = []
    for lab, side, verts, faces in _read_patient_meshes(cage_patient_dir, patient_pid):
        full_meshes.append((verts, faces, rib_colors([lab], [side])[0]))
        full_alphas.append(0.80)
        full_verts.append(verts)
    for _, _, verts, faces in _read_patient_meshes(template_dir, template_pid):
        full_meshes.append((verts, faces, TEMPLATE_COLOR))
        full_alphas.append(0.30)
        full_verts.append(verts)
    add_meshes_3d(ax_full, full_meshes, alpha=full_alphas)
    set_cube_bounds(ax_full, full_verts)
    style_axis(ax_full)

    # Single rib zoom — use the same rib identity as panel 4 (default rib 7 R).
    rib_anat = vert_to_anatomical(radiomics_vert)
    seg_lab  = anatomical_to_seg(rib_anat)
    side     = radiomics_side
    rib_label = f"rib {rib_anat} {side}"

    pat_path = cage_patient_dir / f"{patient_pid}_rib{seg_lab}_{side}.stl"
    tpl_path = template_dir / f"{template_pid}_rib{seg_lab}_{side}.stl"
    pat_v, pat_f = pv_to_triangles(pv.read(str(pat_path)))
    tpl_v, tpl_f = pv_to_triangles(pv.read(str(tpl_path)))
    add_meshes_3d(
        ax_rib,
        [
            (pat_v, pat_f, rib_colors([seg_lab], [side])[0]),
            (tpl_v, tpl_f, TEMPLATE_COLOR),
        ],
        alpha=[0.80, 0.30],
    )
    set_cube_bounds(ax_rib, [pat_v, tpl_v])
    style_axis(ax_rib)

    ax.text(0.24, 0.15, "full cage", transform=ax.transAxes,
            ha="center", va="center", fontsize=S.FONT_SIZE_TICK_PT)
    ax.text(0.76, 0.15, rib_label, transform=ax.transAxes,
            ha="center", va="center", fontsize=S.FONT_SIZE_TICK_PT)

    _patient_plus_one_legend(
        ax, legend_left_label, legend_right_label, TEMPLATE_COLOR,
    )


def _panel_g_per_rib(
    ax: plt.Axes,
    cage_patient_dir: Path, patient_pid: int,
    per_rib_template_dir: Path, template_pid: int,
    radiomics_vert: int, radiomics_side: str,
) -> None:
    """Patient (cage frame) + template after per-rib similarity + rigid ICP.
    Full-cage view on the left, single-rib zoom on the right (per-rib
    improvements are nearly invisible at full-cage scale)."""
    _two_viewport_pair(
        ax, cage_patient_dir, patient_pid,
        per_rib_template_dir, template_pid,
        radiomics_vert, radiomics_side,
        legend_left_label="patient (cage frame)",
        legend_right_label="template (per-rib aligned)",
    )


def _panel_h_gp_fit(
    ax: plt.Axes,
    cage_patient_dir: Path, patient_pid: int,
    gp_fit_template_dir: Path, template_pid: int,
    radiomics_vert: int, radiomics_side: str,
) -> None:
    """Patient (cage frame) + template after non-rigid Gaussian-Process fit."""
    _two_viewport_pair(
        ax, cage_patient_dir, patient_pid,
        gp_fit_template_dir, template_pid,
        radiomics_vert, radiomics_side,
        legend_left_label="patient (cage frame)",
        legend_right_label="template (GP-fit)",
    )


# ── Panel I: merged GPA + PCA — PC1 ±2σ meshes, patient, sparkline table ──

def _panel_i_gpa_pca(
    ax: plt.Axes,
    mean_shape: np.ndarray,
    patient_shape: np.ndarray,
    pc_components: np.ndarray,
    pc_variance: np.ndarray,
    pc_var_ratio: np.ndarray,
    template_faces: np.ndarray,
    rib_offsets: np.ndarray,
    scores_df: pd.DataFrame,
    patient_scores: pd.Series,
    n_pcs_table: int,
) -> None:
    """Triple-mesh row (mean −2σ, GPA-aligned patient, mean +2σ along PC1)
    plus the per-PC sparkline score table.

    The ±2σ meshes are coloured by per-vertex displacement magnitude from the
    mean shape using the project ``magnitude`` palette (viridis), matching
    the leading PC deformation figure. Patient uses the per-rib palette.
    """
    n_pts      = mean_shape.shape[0]
    comp_0     = pc_components[0].reshape(n_pts, 3)
    sigma_0    = float(np.sqrt(pc_variance[0]))
    mean_plus  = mean_shape + 2.0 * sigma_0 * comp_0
    mean_minus = mean_shape - 2.0 * sigma_0 * comp_0

    disp_minus = np.linalg.norm(mean_minus - mean_shape, axis=1)
    disp_plus  = np.linalg.norm(mean_plus  - mean_shape, axis=1)
    # Shared clim across both envelope shells so the viridis colour is
    # comparable left↔right.
    cmax       = float(max(disp_minus.max(), disp_plus.max(), 1e-6))
    envelope_cmap = palette_cmap("magnitude")

    ax.set_axis_off()

    # Three 3D mesh insets across the left ~67% of the panel; sparkline table
    # on the right ~30%. Sub-captions ("−2σ", "patient", "+2σ") sit at y≈0.05.
    mesh_y, mesh_h = 0.10, 0.85
    mesh_w         = 0.20
    gaps           = (0.00, 0.22, 0.44)
    ax_minus    = ax.inset_axes([gaps[0], mesh_y, mesh_w, mesh_h], projection="3d")
    ax_patient  = ax.inset_axes([gaps[1], mesh_y, mesh_w, mesh_h], projection="3d")
    ax_plus     = ax.inset_axes([gaps[2], mesh_y, mesh_w, mesh_h], projection="3d")

    add_mesh_3d(
        ax_minus, mean_minus, template_faces,
        scalars=disp_minus, cmap=envelope_cmap, clim=(0.0, cmax),
    )
    set_cube_bounds(ax_minus, [mean_minus])
    style_axis(ax_minus)

    add_mesh_3d(
        ax_plus, mean_plus, template_faces,
        scalars=disp_plus, cmap=envelope_cmap, clim=(0.0, cmax),
    )
    set_cube_bounds(ax_plus, [mean_plus])
    style_axis(ax_plus)

    patient_meshes: list[tuple[np.ndarray, np.ndarray, str]] = []
    for lab, side, verts, faces in _per_rib_split(
        patient_shape, template_faces, rib_offsets,
    ):
        if verts.size == 0 or faces.size == 0:
            continue
        patient_meshes.append((verts, faces, rib_colors([lab], [side])[0]))
    add_meshes_3d(ax_patient, patient_meshes)
    set_cube_bounds(ax_patient, [patient_shape])
    style_axis(ax_patient)

    cap_kw = dict(transform=ax.transAxes,
                  ha="center", va="center",
                  fontsize=S.FONT_SIZE_TICK_PT)
    cap_y  = 0.13
    ax.text(gaps[0] + mesh_w / 2, cap_y,
            f"−2σ along PC1 ({pc_var_ratio[0]:.1%} var.)", **cap_kw)
    ax.text(gaps[1] + mesh_w / 2, cap_y, "patient (GPA-aligned)", **cap_kw)
    ax.text(gaps[2] + mesh_w / 2, cap_y, "+2σ along PC1",        **cap_kw)

    # Sparkline table fills the right column.
    ax_tbl = ax.inset_axes([0.68, 0.00, 0.32, 1.00])
    _sparkline_score_table(
        ax_tbl, scores_df, patient_scores, pc_variance, n_pcs_table,
    )


def _sparkline_score_table(
    ax: plt.Axes,
    scores_df: pd.DataFrame,
    patient_scores: pd.Series,
    pc_variance: np.ndarray,
    n_pcs: int,
) -> None:
    ax.set_axis_off()
    pc_cols = [c for c in scores_df.columns
               if c.startswith("PC_") and c.split("_", 1)[1].isdigit()]
    pc_cols = sorted(pc_cols, key=lambda c: int(c.split("_", 1)[1]))[:n_pcs]

    # Shared x-range across rows = global cohort range for comparability.
    all_vals = scores_df[pc_cols].values
    finite   = all_vals[np.isfinite(all_vals)]
    if finite.size:
        x_lo = float(np.percentile(finite, 0.5))
        x_hi = float(np.percentile(finite, 99.5))
    else:
        x_lo, x_hi = -1.0, 1.0

    n_rows  = len(pc_cols)
    fig     = ax.figure
    x0, y0, w, h = ax.get_position().bounds
    row_h   = h / (n_rows + 1)

    # Column header
    header = fig.text(
        x0 + 0.03 * w, y0 + h - 0.5 * row_h,
        "PC",  fontsize=S.FONT_SIZE_TICK_PT, fontweight="bold",
        va="center", ha="left",
    )
    fig.text(
        x0 + 0.28 * w, y0 + h - 0.5 * row_h,
        "patient (σ)",
        fontsize=S.FONT_SIZE_TICK_PT, fontweight="bold",
        va="center", ha="center",
    )
    fig.text(
        x0 + 0.72 * w, y0 + h - 0.5 * row_h,
        "cohort distribution",
        fontsize=S.FONT_SIZE_TICK_PT, fontweight="bold",
        va="center", ha="center",
    )

    for k, col in enumerate(pc_cols):
        pc_idx = int(col.split("_", 1)[1]) - 1
        sigma  = float(np.sqrt(pc_variance[pc_idx]))
        if not np.isfinite(sigma) or sigma == 0:
            sigma = 1.0
        patient_val = float(patient_scores[col])
        patient_sig = patient_val / sigma

        y_centre = y0 + h - (k + 1.5) * row_h

        fig.text(
            x0 + 0.03 * w, y_centre,
            f"PC{pc_idx + 1}",
            fontsize=S.FONT_SIZE_TICK_PT,
            va="center", ha="left",
        )
        fig.text(
            x0 + 0.28 * w, y_centre,
            f"{patient_sig:+.2f}",
            fontsize=S.FONT_SIZE_TICK_PT,
            va="center", ha="center",
            color="#c62828" if abs(patient_sig) > 1.0 else "#222",
        )

        spark = fig.add_axes([
            x0 + 0.48 * w,
            y_centre - 0.35 * row_h,
            0.49 * w,
            0.70 * row_h,
        ])
        vals = scores_df[col].dropna().values
        if vals.size:
            spark.hist(
                vals, bins=40, range=(x_lo, x_hi),
                color="#bdbdbd", linewidth=0,
            )
        spark.axvline(patient_val, color="#c62828", linewidth=1.2)
        spark.set_xlim(x_lo, x_hi)
        spark.set_xticks([])
        spark.set_yticks([])
        for sp in spark.spines.values():
            sp.set_visible(False)


# ── Legend helper (panels E, F, G) ─────────────────────────────────────────

# Representative left+right colours from utils.colors.rib_colors, used as a
# two-square swatch in the legend to indicate "patient uses the per-rib palette".
_PATIENT_SWATCH_L = "#5091A6"   # rib_colors([46], ['L']) — mid-teal
_PATIENT_SWATCH_R = "#E53127"   # rib_colors([46], ['R']) — mid-red


def _patient_plus_one_legend(
    ax: plt.Axes,
    patient_label: str,
    other_label: str,
    other_color: str,
    *,
    y_offset: float = -0.10,
) -> None:
    """Legend for panels F, G, H: per-rib patient swatch + one other entry.

    The patient handle renders as two stacked squares (teal + red) to signal
    "left/right per-rib palette"; the other entry gets a flat patch.
    ``y_offset`` shifts the legend (panels with extra captions above the
    legend pass a more negative value).
    """
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D

    patient_handle = (
        Line2D([], [], marker="s", color=_PATIENT_SWATCH_L,
               markersize=6, markeredgecolor="none", linestyle=""),
        Line2D([], [], marker="s", color=_PATIENT_SWATCH_R,
               markersize=6, markeredgecolor="none", linestyle=""),
    )
    other_handle = mpatches.Patch(facecolor=other_color, edgecolor="none")
    ax.legend(
        handles=[patient_handle, other_handle],
        labels=[patient_label, other_label],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
        loc="lower center",
        bbox_to_anchor=(0.5, y_offset),
        ncol=2,
        frameon=False,
        fontsize=S.FONT_SIZE_TICK_PT,
        handlelength=1.4, handleheight=0.8, columnspacing=0.8,
    )


# ── Card + title helpers ──────────────────────────────────────────────────

CARD_FACE = "#F1F5F9"   # slate-50: soft cool grey-blue
CARD_EDGE = "#CBD5E1"   # slate-300
SUBTITLE_COLOR = "#6E6E6E"


def _union_bbox(axes_list: list[plt.Axes]) -> tuple[float, float, float, float]:
    bbs = [ax.get_position() for ax in axes_list]
    x0 = min(b.x0 for b in bbs)
    x1 = max(b.x1 for b in bbs)
    y0 = min(b.y0 for b in bbs)
    y1 = max(b.y1 for b in bbs)
    return x0, y0, x1, y1


def _add_panel_card(fig: plt.Figure, axes_list: list[plt.Axes]) -> None:
    x0, y0, x1, y1 = _union_bbox(axes_list)
    pad_x   = 0.008
    pad_top = 0.040   # space for title + subtitle above the axes
    pad_bot = 0.018   # space for slice labels / legend below
    card = mpatches.FancyBboxPatch(
        (x0 - pad_x, y0 - pad_bot),
        (x1 - x0) + 2 * pad_x,
        (y1 - y0) + pad_top + pad_bot,
        boxstyle="round,pad=0.0,rounding_size=0.010",
        facecolor=CARD_FACE,
        edgecolor=CARD_EDGE,
        linewidth=0.5,
        transform=fig.transFigure,
        zorder=-100,
        clip_on=False,
    )
    fig.patches.append(card)


def _add_panel_title(
    fig: plt.Figure,
    axes_list: list[plt.Axes],
    title: str,
    subtitle: str,
) -> None:
    x0, _, _, y1 = _union_bbox(axes_list)
    fig.text(
        x0, y1 + 0.027, title,
        fontsize=S.FONT_SIZE_TITLE_PT, fontweight="bold",
        va="bottom", ha="left",
    )
    fig.text(
        x0, y1 + 0.012, subtitle,
        fontsize=S.FONT_SIZE_TICK_PT, color=SUBTITLE_COLOR,
        va="bottom", ha="left",
    )


