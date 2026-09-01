"""Per-vertex registration-residual diagnostics.

Forward (registered → target) and reverse (target → registered) per-vertex
closest-point distances for every patient with all 24 registered + 24 raw ribs.

Products
--------
- ``<run>/ssm_qa_residuals/residuals_per_patient.npz`` – per-patient forward
  and reverse arrays (variable-length per rib).
- ``<run>/ssm_qa_residuals/figures/worst_patients_residuals_{forward,reverse}.png``
  – matplotlib Poly3DCollection mosaics of the worst N patients
  (rendering failures are caught; the NPZ harvest is preserved).
- ``<run>/ssm_qa_residuals/figures/residuals_distribution_{forward,reverse}.{html,svg,png}``
  – per-level mirrored Altair ridgelines over the pooled per-vertex residuals
  (:func:`ssm.plots_ssm_altair.plot_residual_distribution`).

CLI::

    python src/ssm/eval_residuals.py --run-dir DIR
        [--registered-dir DIR] [--target-dir DIR] [--worst-n 6]
        [--scope whole-cage|rib] [--rib-id rib7_R]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pyvista as pv
from joblib import Parallel, delayed
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import settings as S                               # noqa: E402
from settings import SSM_LOAD_N_WORKERS, apply_publication_style    # noqa: E402
from ssm.pca_surface import RIB_LABELS, RIB_SIDES                   # noqa: E402
from ssm.plots_ssm_altair import plot_residual_distribution        # noqa: E402
from utils import colors as C                      # noqa: E402
from utils.figure_export import save_fig           # noqa: E402
from utils.logging import get_logger               # noqa: E402
from utils.paths import stage_dir                  # noqa: E402
from utils.plotly_theme import (                                        # noqa: E402
    apply_layout, grid_spacing, place_colorbar_right,
)
from utils.rib_labels import (                                          # noqa: E402
    cli_token_from_seg,
    display_from_seg,
    parse_cli_token,
)
from utils.run_dir import patient_stl_dir                           # noqa: E402

apply_publication_style()

logger = get_logger(__name__)

# Internal rib id (file paths, dict / npz keys); on-disk seg labels 40..51.
RIB_IDS = [f"rib{lab}_{side}" for lab in RIB_LABELS for side in RIB_SIDES]
# Anatomical 1-based forms for CLI input and human-facing display.
RIB_CLI_TOKENS = [cli_token_from_seg(lab, side)
                  for lab in RIB_LABELS for side in RIB_SIDES]


def _display_from_internal(internal_rib_id: str) -> str:
    """``'rib46_R' → 'Rib 7 R'``."""
    head, side = internal_rib_id.rsplit("_", 1)
    seg = int(head.removeprefix("rib"))
    return display_from_seg(seg, side)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-vertex registration-residual diagnostics")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Top-level pipeline run dir.  Reads PCA outputs "
                        "from <run-dir>/ssm_pca/; writes residual outputs "
                        "to <run-dir>/ssm_qa_residuals/.")
    p.add_argument("--registered-dir", type=str, default=None,
                   help="Override registration source (default = "
                        "<run-dir>/metadata.json::paths.registered_stl_dir).")
    p.add_argument("--target-dir", type=str, default=None,
                   help="Target / extraction source dir.  Default: read "
                        "paths.extracted_stl_dir from <run-dir>/metadata.json.")
    p.add_argument("--worst-n", type=int, default=6,
                   help="Number of worst patients to render in the mosaic. Default 6.")
    p.add_argument("--scope", type=str, default="whole-cage",
                   choices=["whole-cage", "rib"],
                   help="Render whole 24-rib cage or a single rib identity.")
    p.add_argument("--rib-id", type=str, default=None,
                   help="Rib identity in anatomical form (e.g. 'rib7_R'; "
                        "rib number in 1..12) when --scope=rib.")
    p.add_argument("--workers", type=int, default=None, metavar="N",
                   help="Parallel workers for residual harvest. "
                        "Default: settings.SSM_LOAD_N_WORKERS (32).")
    return p.parse_args()


# ── STL loading ──────────────────────────────────────────────────────────────

def _load_stl(path: Path) -> pv.PolyData:
    return pv.read(str(path))


def _vertices(mesh: pv.PolyData) -> np.ndarray:
    return np.asarray(mesh.points, dtype=np.float64)


def _residuals_pair(reg_pts: np.ndarray, tgt_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(forward, reverse)`` per-vertex closest-point distances.

    ``forward[i]`` = distance from registered vertex ``i`` to nearest target.
    ``reverse[j]`` = distance from target vertex ``j`` to nearest registered.
    """
    fwd = cKDTree(tgt_pts).query(reg_pts)[0]
    rev = cKDTree(reg_pts).query(tgt_pts)[0]
    return fwd, rev


# ── Per-patient harvest ──────────────────────────────────────────────────────

def _harvest_patient(
    pid: int,
    registered_dir: Path,
    target_dir: Path,
    rib_ids: list[str],
) -> dict[str, dict[str, np.ndarray]] | None:
    """For one patient, load all 24 ribs and compute per-vertex residuals.

    Returns ``{rib_id: {"forward": ..., "reverse": ..., "registered": pv.PolyData,
    "target": pv.PolyData}}`` or ``None`` if any rib is missing.
    """
    out: dict[str, dict] = {}
    reg_pdir = patient_stl_dir(registered_dir, pid)
    tgt_pdir = patient_stl_dir(target_dir, pid)
    for rib_id in rib_ids:
        reg_p = reg_pdir / f"{pid}_{rib_id}.stl"
        tgt_p = tgt_pdir / f"{pid}_{rib_id}.stl"
        if not reg_p.exists() or not tgt_p.exists():
            return None
        reg_m = _load_stl(reg_p)
        tgt_m = _load_stl(tgt_p)
        # ``pv.read`` does NOT raise when the underlying vtkSTLReader
        # bails out on a malformed STL (non-finite triangle normals,
        # truncated file, etc.) – it logs a warning and returns an empty
        # PolyData. Skip the whole patient in that case so downstream
        # aggregation never sees zero-vertex meshes (would crash
        # ``np.percentile`` and cKDTree).
        if reg_m.n_points == 0 or tgt_m.n_points == 0:
            logger.warning(
                "pid=%d %s: empty mesh after pv.read "
                "(reg_pts=%d tgt_pts=%d) – skipping patient.",
                pid, rib_id, reg_m.n_points, tgt_m.n_points,
            )
            return None
        fwd, rev = _residuals_pair(_vertices(reg_m), _vertices(tgt_m))
        out[rib_id] = {
            "forward":   fwd.astype(np.float32),
            "reverse":   rev.astype(np.float32),
            "registered": reg_m,
            "target":    tgt_m,
        }
    return out


def _harvest_residuals(
    pid: int,
    registered_dir: Path,
    target_dir: Path,
    rib_ids: list[str],
) -> dict[str, dict[str, np.ndarray]] | None:
    """Like :func:`_harvest_patient` but returns residual arrays only.

    PolyData refs go out of scope on each iteration so peak memory is
    bounded; meshes are reharvested for the worst-N mosaic.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    reg_pdir = patient_stl_dir(registered_dir, pid)
    tgt_pdir = patient_stl_dir(target_dir, pid)
    for rib_id in rib_ids:
        reg_p = reg_pdir / f"{pid}_{rib_id}.stl"
        tgt_p = tgt_pdir / f"{pid}_{rib_id}.stl"
        if not reg_p.exists() or not tgt_p.exists():
            return None
        reg_m = _load_stl(reg_p)
        tgt_m = _load_stl(tgt_p)
        if reg_m.n_points == 0 or tgt_m.n_points == 0:
            logger.warning(
                "pid=%d %s: empty mesh after pv.read "
                "(reg_pts=%d tgt_pts=%d) – skipping patient.",
                pid, rib_id, reg_m.n_points, tgt_m.n_points,
            )
            return None
        fwd, rev = _residuals_pair(_vertices(reg_m), _vertices(tgt_m))
        out[rib_id] = {
            "forward": fwd.astype(np.float32),
            "reverse": rev.astype(np.float32),
        }
    return out


def _harvest_residuals_keyed(
    pid: int,
    registered_dir: Path,
    target_dir: Path,
    rib_ids: list[str],
) -> tuple[int, dict[str, dict[str, np.ndarray]] | None]:
    """``(pid, result)`` wrapper for unordered joblib consumption."""
    return pid, _harvest_residuals(pid, registered_dir, target_dir, rib_ids)


# ── Mosaic rendering (matplotlib Poly3DCollection – pure CPU) ────────────────
# PyVista's off-screen renderer needs OSMesa or EGL on headless servers; on
# Singularity containers without those, vtkOpenGLRenderWindow segfaults.
# matplotlib's mpl_toolkits.mplot3d works without any GPU/OpenGL.

def _pv_faces_to_triangles(mesh: pv.PolyData) -> np.ndarray:
    """``(n_faces, 3)`` int triangle indices from a PyVista PolyData.

    PV stores faces as a flat array ``[3, i, j, k, 3, i, j, k, ...]`` for
    triangle meshes; we slice that into a ``(n, 3)`` array.
    """
    f = np.asarray(mesh.faces, dtype=np.int64)
    if f.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    return f.reshape(-1, 4)[:, 1:]


def _render_panel_mpl(
    ax,
    patient_data: dict[str, dict],
    rib_ids: list[str],
    clim: tuple[float, float],
    title: str,
    direction: str = "forward",
) -> None:
    """Render one patient's residual-coloured cage into a 3D matplotlib axis.

    Per-vertex residuals are converted to per-face by averaging the residual
    at each triangle's three vertices. ``direction`` picks which mesh + array
    to colour: forward residuals live on registered-mesh vertices, reverse on
    target-mesh vertices.
    """
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh_key = "registered" if direction == "forward" else "target"
    norm = Normalize(vmin=clim[0], vmax=clim[1])
    cmap = C.cmap("residual")
    all_verts: list[np.ndarray] = []
    tri_blocks: list[np.ndarray] = []
    color_blocks: list[np.ndarray] = []

    for rib_id in rib_ids:
        d     = patient_data[rib_id]
        verts = np.asarray(d[mesh_key].points, dtype=np.float64)
        faces = _pv_faces_to_triangles(d[mesh_key])
        if faces.size == 0:
            continue
        face_res = d[direction][faces].mean(axis=1)
        tri_blocks.append(verts[faces])
        color_blocks.append(cmap(norm(face_res)))
        all_verts.append(verts)

    if tri_blocks:
        # Merge all 24 ribs into one Poly3DCollection so matplotlib's per-face
        # painter's sort covers the whole cage instead of stacking ribs in
        # add-order.
        triangles  = np.concatenate(tri_blocks, axis=0)
        facecolors = np.concatenate(color_blocks, axis=0)
        coll = Poly3DCollection(triangles, facecolors=facecolors,
                                edgecolors="none", linewidths=0)
        ax.add_collection3d(coll)

    if all_verts:
        all_pts = np.concatenate(all_verts, axis=0)
        lo, hi  = all_pts.min(axis=0), all_pts.max(axis=0)
        mid     = (lo + hi) / 2
        half    = (hi - lo).max() / 2 * 1.05
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=20, azim=-60)
    ax.set_axis_off()


def render_mosaic(
    out_path: Path,
    patients: list[tuple[int, dict[str, dict], float]],
    rib_ids: list[str],
    cols: int = 3,
    direction: str = "forward",
) -> None:
    """Worst-N residual mosaic – Plotly Mesh3d HTML + matplotlib PNG fallback.

    HTML uses one Plotly subplot grid of Mesh3d traces (vertex colour =
    per-vertex residual in ``direction``).  Print-quality static export
    goes through the matplotlib ``Poly3DCollection`` path (pure CPU;
    works on every headless server) so SVG / PNG don't depend on a
    working WebGL snapshot.

    ``direction='forward'`` colours the registered mesh by registered →
    target distance; ``direction='reverse'`` colours the target mesh by
    target → registered distance.
    """
    n = len(patients)
    if n == 0:
        logger.warning("No patients to render – skipping mosaic.")
        return
    rows = (n + cols - 1) // cols

    out_stem = out_path.with_suffix("") if out_path.suffix else out_path
    scope_str = "rib" if len(rib_ids) == 1 else "whole-cage"
    mesh_key = "registered" if direction == "forward" else "target"

    # Shared colour scale: 0 → 99th percentile of residuals in this direction.
    all_res = np.concatenate([
        np.concatenate([d[direction] for d in pdat.values()])
        for _, pdat, _ in patients
    ])
    clim = (0.0, float(np.percentile(all_res, 99)))

    # Build interactive Plotly mosaic.
    specs = [[{"type": "scene"}] * cols for _ in range(rows)]
    titles = [f"#{i+1} · p95={p95:.2f} mm"
              for i, (_pid, _, p95) in enumerate(patients)]
    titles += [""] * (rows * cols - len(titles))
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=titles,
        horizontal_spacing=0.02,
        vertical_spacing=grid_spacing(rows, row_height_mm=50.0, gap_mm=4.0),
    )
    cs = C.colorscale("residual")

    showed_bar = False
    for idx, (pid, pdat, p95) in enumerate(patients):
        r = idx // cols + 1
        c = idx % cols + 1
        # Concatenate this patient's ribs into one trace.
        all_v, all_f, all_res = [], [], []
        v_offset = 0
        for rib_id in rib_ids:
            d = pdat[rib_id]
            v = np.asarray(d[mesh_key].points, dtype=np.float64)
            f = _pv_faces_to_triangles(d[mesh_key])
            if f.size == 0:
                continue
            all_v.append(v)
            all_f.append(f + v_offset)
            all_res.append(d[direction])
            v_offset += v.shape[0]
        if not all_v:
            continue
        verts = np.concatenate(all_v, axis=0)
        faces = np.concatenate(all_f, axis=0)
        res   = np.concatenate(all_res, axis=0)

        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                intensity=res,
                colorscale=cs,
                cmin=clim[0], cmax=clim[1],
                showscale=not showed_bar,
                colorbar=(
                    place_colorbar_right(
                        title=f"{direction} residual (mm)", length_fraction=0.7
                    )
                    if not showed_bar else None
                ),
                lighting=dict(ambient=0.45, diffuse=0.7, specular=0.15,
                              roughness=0.85, fresnel=0.2),
                lightposition=dict(x=400, y=200, z=300),
                hovertemplate=f"#{idx+1}<br>{direction} residual=%{{intensity:.2f}} mm<extra></extra>",
                showlegend=False,
            ),
            row=r, col=c,
        )
        showed_bar = True
        scene_id = f"scene{idx + 1 if idx > 0 else ''}"
        fig.layout[scene_id].update(
            aspectmode="data",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )

    apply_layout(
        fig, width_class="full",
        height_mm=max(60.0 * rows, 80.0),
        title=(f"Worst {n} patients by 95-th percentile {direction} residual  ·  "
               f"scope={scope_str}  ·  colour scale 0 – {clim[1]:.2f} mm"),
    )

    # Print fallback via matplotlib Poly3DCollection; exported through
    # save_fig's static_renderer slot so SVG / PNG go through one entry point.
    def _render_print(path: Path) -> None:
        from matplotlib.colors import Normalize
        import matplotlib.cm as _cm

        m_fig = plt.figure(figsize=(cols * 5.0, rows * 5.5))
        m_fig.suptitle(
            f"Worst {n} patients by 95th-percentile {direction} residual  (scope={scope_str})",
            fontsize=S.FONT_SIZE_TITLE_PT + 1,
        )
        for idx, (_pid, pdat, p95) in enumerate(patients):
            ax = m_fig.add_subplot(rows, cols, idx + 1, projection="3d")
            title = f"#{idx+1}  p95={p95:.2f} mm"
            _render_panel_mpl(ax, pdat, rib_ids, clim, title, direction=direction)

        sm = _cm.ScalarMappable(cmap=C.cmap("residual"),
                                norm=Normalize(vmin=clim[0], vmax=clim[1]))
        sm.set_array([])
        cax = m_fig.add_axes([0.15, 0.04, 0.7, 0.02])
        cbar = m_fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label(f"{direction} residual (mm)  clim=[0, {clim[1]:.2f}]")
        m_fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.10,
                              wspace=0.05, hspace=0.10)
        m_fig.savefig(path, dpi=S.EXPORT_RASTER_DPI)
        plt.close(m_fig)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    save_fig(
        fig, out_stem,
        formats=("html", "svg", "png"),
        static_renderer=_render_print,
        title=f"Worst-{n} {direction} residual mosaic ({scope_str})",
        width_class="full",
    )
    logger.info("Wrote %s", out_stem)


def load_residuals_npz(
    path: Path,
) -> tuple[dict[int, dict[str, dict[str, np.ndarray]]],
           dict[str, dict[int, float]],
           list[str]]:
    """Reload ``residuals_per_patient[_<rib>].npz`` into the in-memory shape.

    Returns ``(per_patient, p95_by_pid, rib_ids)`` matching the structures
    built in :func:`main`. Per-patient dicts contain only the residual
    arrays – ``registered`` / ``target`` PolyData must be reharvested
    separately for the worst-N mosaic.
    """
    data = np.load(Path(path), allow_pickle=False)
    pids     = [int(x) for x in data["_pids"]]
    rib_ids  = [str(x) for x in data["_rib_ids"]]
    p95_fwd  = data["_p95"]
    p95_rev  = data["_p95_reverse"]

    per_patient: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for pid in pids:
        per_rib: dict[str, dict[str, np.ndarray]] = {}
        for rib_id in rib_ids:
            fwd_key = f"{pid}_{rib_id}_forward"
            rev_key = f"{pid}_{rib_id}_reverse"
            if fwd_key not in data.files or rev_key not in data.files:
                continue
            per_rib[rib_id] = {
                "forward": data[fwd_key],
                "reverse": data[rev_key],
            }
        if per_rib:
            per_patient[pid] = per_rib

    p95_by_pid: dict[str, dict[int, float]] = {
        "forward": {pid: float(v) for pid, v in zip(pids, p95_fwd)},
        "reverse": {pid: float(v) for pid, v in zip(pids, p95_rev)},
    }
    return per_patient, p95_by_pid, rib_ids


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    n_workers = SSM_LOAD_N_WORKERS if args.workers is None else int(args.workers)
    run_dir = args.run_dir
    pca_dir = stage_dir(run_dir, "ssm_pca")
    out_dir = stage_dir(run_dir, "ssm_qa_residuals")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"PCA inputs: {pca_dir}")
    logger.info(f"Residual outputs: {out_dir}")

    meta_path = run_dir / "metadata.json"
    meta: dict = (
        json.loads(meta_path.read_text()) if meta_path.exists() else {}
    )
    meta_paths = (meta.get("paths") or {})  # nested under "paths" by make_run_dir

    def _resolve_from_meta(cli_value: str | None, meta_key: str, cli_flag: str) -> Path:
        if cli_value:
            return Path(cli_value)
        src = meta_paths.get(meta_key)
        if not src:
            raise FileNotFoundError(
                f"{cli_flag} not supplied and {meta_path}::paths.{meta_key} "
                f"is missing or empty; pass {cli_flag} explicitly."
            )
        return Path(src)

    registered_dir = _resolve_from_meta(
        args.registered_dir, "registered_stl_dir", "--registered-dir")
    target_dir = _resolve_from_meta(
        args.target_dir,     "extracted_stl_dir",  "--target-dir")

    logger.info(f"Registered dir: {registered_dir}")
    logger.info(f"Target dir:     {target_dir}")

    if args.scope == "rib":
        if not args.rib_id:
            raise ValueError("--scope=rib requires --rib-id (e.g. 'rib7_R').")
        try:
            seg, side = parse_cli_token(args.rib_id)
        except ValueError as exc:
            raise ValueError(
                f"{exc}; valid forms are: {RIB_CLI_TOKENS}"
            ) from None
        rib_ids = [f"rib{seg}_{side}"]
    else:
        rib_ids = list(RIB_IDS)

    # Discover patients from the sharded layout:
    # ``<registered_dir>/<block>/<pid>/``.  Pid comes from the leaf
    # directory name; existence of every rib STL is verified later in
    # ``_harvest_patient``.
    candidate_pids: set[int] = set()
    for block_dir in registered_dir.iterdir():
        if not block_dir.is_dir() or block_dir.name.startswith("._"):
            continue
        for pid_dir in block_dir.iterdir():
            if pid_dir.is_dir() and pid_dir.name.isdigit():
                candidate_pids.add(int(pid_dir.name))
    logger.info(f"Discovered {len(candidate_pids):,} candidate patients in registered dir")

    # Parallel per-patient harvest.  STL reads (PyVista/VTK) and cKDTree
    # build/query (SciPy) all release the GIL, so threads scale on the
    # 32-core box without paying pickling cost on the PolyData objects.
    sorted_pids = sorted(candidate_pids)
    n_total = len(sorted_pids)
    t0 = time.monotonic()
    logger.info(
        f"Harvesting residuals in parallel (n_jobs={n_workers}, threads, "
        f"{n_total:,} patients)…"
    )
    # Stream results unordered so progress logs reflect actual throughput.
    # With ``return_as='generator'`` (ordered) joblib would block the consumer
    # whenever the next-in-submission-order patient hasn't finished yet – a
    # single slow NFS read on patient #1 stalls all output until it clears,
    # then a burst.  Unordered yields results as workers complete them; the
    # worker wrapper returns ``(pid, result)`` so the dict build is direct.
    results_iter = Parallel(
        n_jobs=n_workers, prefer="threads",
        return_as="generator_unordered",
    )(
        delayed(_harvest_residuals_keyed)(pid, registered_dir, target_dir, rib_ids)
        for pid in sorted_pids
    )
    progress_every = max(1, n_total // 50)  # ~50 progress lines total
    progress_interval_s = 30.0
    per_patient: dict[int, dict[str, dict]] = {}
    p95_by_pid: dict[str, dict[int, float]] = {"forward": {}, "reverse": {}}
    skipped = 0
    last_log_t = t0
    for i, (pid, data) in enumerate(results_iter, start=1):
        if data is None:
            skipped += 1
        else:
            per_patient[pid] = data
            for direction in ("forward", "reverse"):
                pooled = np.concatenate([d[direction] for d in data.values()])
                p95_by_pid[direction][pid] = float(np.percentile(pooled, 95))
        now = time.monotonic()
        if (i % progress_every == 0
                or (now - last_log_t) >= progress_interval_s
                or i == n_total):
            elapsed = now - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta_s = (n_total - i) / rate if rate > 0 else 0.0
            logger.info(
                f"  Harvest progress: {i:,}/{n_total:,} "
                f"({100 * i / n_total:.1f}%) – {rate:.2f} pat/s "
                f"– elapsed {elapsed:.0f}s – ETA {eta_s:.0f}s"
            )
            last_log_t = now
    logger.info(f"  Harvest pass done in {time.monotonic() - t0:.0f}s")
    logger.info(f"Harvested {len(per_patient):,} complete patients (skipped {skipped})")

    if not per_patient:
        raise RuntimeError("No patients with a complete set of registered+target STLs.")

    # Persist per-vertex residuals as a sparse npz (key per patient × rib × direction).
    npz_payload: dict[str, np.ndarray] = {}
    for pid, data in per_patient.items():
        for rib_id, d in data.items():
            npz_payload[f"{pid}_{rib_id}_forward"] = d["forward"]
            npz_payload[f"{pid}_{rib_id}_reverse"] = d["reverse"]
    npz_payload["_pids"] = np.array(sorted(per_patient.keys()), dtype=np.int64)
    sorted_keys = sorted(per_patient.keys())
    npz_payload["_p95"] = np.array(
        [p95_by_pid["forward"][p] for p in sorted_keys], dtype=np.float32,
    )
    npz_payload["_p95_reverse"] = np.array(
        [p95_by_pid["reverse"][p] for p in sorted_keys], dtype=np.float32,
    )
    npz_payload["_rib_ids"] = np.array(rib_ids)
    out_npz = out_dir / (
        "residuals_per_patient.npz" if args.scope == "whole-cage"
        else f"residuals_per_patient_{args.rib_id}.npz"
    )
    np.savez_compressed(out_npz, **npz_payload)
    logger.info(f"Wrote {out_npz}")

    # Worst-N mosaic + per-rib violin distribution, one of each per
    # direction.  Wrap rendering in try/except so a failure here doesn't
    # void the per-patient NPZ harvest above.  Worst-patient ranking is
    # direction-specific: a patient with bad "tents" (forward) is not
    # necessarily the same one that missed target features (reverse).
    for direction in ("forward", "reverse"):
        ranked = sorted(per_patient.keys(),
                        key=lambda p, dr=direction: -p95_by_pid[dr][p])
        worst = ranked[: args.worst_n]
        # Reharvest meshes for the worst-N only – the bulk harvest above
        # dropped PolyData to keep peak RAM bounded.
        panels: list[tuple[int, dict, float]] = []
        for pid in worst:
            full = _harvest_patient(pid, registered_dir, target_dir, rib_ids)
            if full is None:
                logger.warning(
                    f"Could not reharvest STLs for pid={pid} – skipping mosaic panel."
                )
                continue
            panels.append((pid, full, p95_by_pid[direction][pid]))
        out_png = out_dir / "figures" / (
            f"worst_patients_residuals_{direction}" if args.scope == "whole-cage"
            else f"worst_patients_residuals_{direction}_{args.rib_id}"
        )
        if not panels:
            logger.warning(
                f"No reharvestable patients for {direction} mosaic – skipping."
            )
        else:
            try:
                render_mosaic(out_png, panels, rib_ids, direction=direction)
            except Exception as e:  # Plotly/Kaleido renderer varies
                logger.error(
                    f"{direction} mosaic rendering failed ({type(e).__name__}: {e}). "
                    f"Per-patient residuals at {out_npz} are preserved."
                )

        out_dist = out_dir / "figures" / (
            f"residuals_distribution_{direction}" if args.scope == "whole-cage"
            else f"residuals_distribution_{direction}_{args.rib_id}"
        )
        try:
            plot_residual_distribution(per_patient, rib_ids, out_dist,
                                       direction=direction)
        except Exception as e:  # Plotly/Kaleido renderer varies
            logger.error(
                f"{direction} residual distribution plot failed "
                f"({type(e).__name__}: {e}); per-patient NPZ at {out_npz} preserved."
            )


if __name__ == "__main__":
    main()
