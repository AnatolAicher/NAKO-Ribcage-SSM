"""Diagnostic per-rib STL viewer: template overlay + raw vs registered.

For a chosen patient, renders three tiers of per-rib STLs in a single
interactive 3D scene with toggleable visibility:

- **Template** (purple, always overlaid): the reference patient's raw cage
  + A/B endpoint landmarks. Fixed visual anchor across patient selection.
- **Raw** (blue, selected patient): freshly extracted per-rib STLs,
  Procrustes-aligned to the registered frame.
- **Registered** (orange, selected patient): the Scalismo non-rigid
  registration output (template-in-correspondence).

Optional per-rib landmark JSONs (``<ribId>_extracted.json``,
``<ribId>_registered.json``) are overlaid when present; the canonical
pipeline does not generate them.

The HTML page is self-contained – mesh + landmark data are inlined as
JSON, so it works offline / from ``file://``.

Resolution of input directories:

- **Raw STLs**: ``--target-dir`` → ``<run-dir>/metadata.json::paths.extracted_stl_dir``.
- **Registered STLs**: ``--registered-dir`` → ``<run-dir>/metadata.json::paths.registered_stl_dir``.
- **Landmarks** (optional): ``--landmark-dir`` → ``metadata.json::paths.landmark_dir``.
- **Template pid** (optional): ``--template-pid`` → ``<registered-dir>/template_id.txt``.
- **Run dir** (output location): ``--results`` or ``NAKO_SSM_RUN_DIR`` / ``NAKO_RUN_DIR`` env.

CLI::

    python src/ssm/diagnostic_viewer.py --results <run-dir>
    python src/ssm/diagnostic_viewer.py --results <run-dir> --patient-ids 12345,67890
    python src/ssm/diagnostic_viewer.py --results <run-dir> --no-browser
    python src/ssm/diagnostic_viewer.py --results <run-dir> --max-patients 10

When installed via ``pip install -e .``, ``python -m ssm.diagnostic_viewer``
works equivalently.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pyvista as pv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from settings import FONT_FAMILY                                  # noqa: E402
from ssm.pca_surface import RIB_LABELS, RIB_SIDES, RIB_ORDER      # noqa: E402
from utils import colors as _colors                               # noqa: E402
from utils.rib_labels import display_from_seg                     # noqa: E402
from utils.run_dir import patient_stl_dir                         # noqa: E402

logger = logging.getLogger(__name__)

# Matches filenames inside a per-patient sharded dir, e.g. ``100001_rib40_L.stl``.
_RIB_FILE_RE = re.compile(r"^\d+_rib(\d+)_([LR])\.stl$")

_RAW_COLOR = "#0072B2"        # Okabe-Ito blue   – selected patient (raw)
_TPL_COLOR = "#CC79A7"        # Okabe-Ito purple – template (always shown)
_REG_COLOR = "#E69F00"        # Okabe-Ito orange – Scalismo registered
_MESH_OPACITY        = 0.55
_MESH_OPACITY_DIMMED = 0.08   # ribs not under focus

# Landmark marker size, in CSS pixels. Fixed across renders.
_LM_MARKER_SIZE = 5

# Per-rib landmark filenames; match ``ribId`` from ``RibRegistration.scala``.
_LM_EXTRACTED_SUFFIX  = "_extracted.json"
_LM_REGISTERED_SUFFIX = "_registered.json"


# ── Run-dir resolution ──────────────────────────────────────────────────────

def _default_results_dir() -> Path:
    """Resolve the run dir from ``NAKO_SSM_RUN_DIR`` / ``NAKO_RUN_DIR``."""
    explicit = os.environ.get("NAKO_SSM_RUN_DIR") or os.environ.get("NAKO_RUN_DIR")
    if explicit:
        return Path(explicit).expanduser()
    raise FileNotFoundError(
        "no run dir configured; pass --results <run-dir> or set "
        "NAKO_SSM_RUN_DIR / NAKO_RUN_DIR."
    )


def _resolve_mesh_dirs(
    results_dir: Path | None,
    target_dir: Path | None = None,
    registered_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve raw (target) and registered per-rib STL directories.

    Precedence per dir: explicit CLI arg → ``<run-dir>/metadata.json::paths.*``.
    """
    raw = (
        target_dir
        if target_dir is not None
        else _dir_from_metadata(results_dir, "extracted_stl_dir",
                                "--target-dir")
    )
    reg = (
        registered_dir
        if registered_dir is not None
        else _dir_from_metadata(results_dir, "registered_stl_dir",
                                "--registered-dir")
    )
    return raw, reg


def _resolve_template_pid(
    results_dir: Path | None,
    template_pid: int | None,
    registered_dir: Path | None,
) -> int | None:
    """Resolve the template patient ID, or ``None`` if unavailable.

    Precedence: explicit ``template_pid`` → ``<registered_dir>/template_id.txt``
    → ``metadata.json::paths.registered_stl_dir/template_id.txt``.
    """
    if template_pid is not None:
        return int(template_pid)

    candidates: list[Path] = []
    if registered_dir is not None:
        candidates.append(registered_dir / "template_id.txt")
    if results_dir is not None:
        meta_path = results_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                src = ((meta.get("paths") or {}).get("registered_stl_dir")
                       or meta.get("source_registration_dir"))
                if src:
                    candidates.append(Path(src) / "template_id.txt")
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.warning("Failed to parse %s: %s", meta_path, exc)

    for p in candidates:
        if p.is_file():
            try:
                pid = int(p.read_text().strip())
                return pid
            except (OSError, ValueError) as exc:
                logger.warning("Failed to parse %s: %s", p, exc)
                continue
    return None


def _resolve_landmark_dir(
    results_dir: Path | None,
    landmark_dir: Path | None,
) -> Path | None:
    """Resolve the per-rib landmark directory, or ``None`` if unavailable.

    Landmarks are optional; absence is silent.
    """
    if landmark_dir is not None:
        p = Path(landmark_dir).expanduser()
        if not p.is_dir():
            raise FileNotFoundError(
                f"--landmark-dir does not exist or is not a directory: {p}"
            )
        return p
    if results_dir is None:
        return None
    meta_path = results_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse %s: %s", meta_path, exc)
        return None
    src = ((meta.get("paths") or {}).get("landmark_dir")
           or meta.get("source_landmark_dir"))
    if not src:
        return None
    p = Path(src)
    if not p.is_dir():
        logger.warning("source_landmark_dir in %s points to a missing dir: %s",
                       meta_path, p)
        return None
    return p


def _dir_from_metadata(run_dir: Path | None, meta_key: str, cli_flag: str) -> Path:
    """Read ``paths.<meta_key>`` from ``<run-dir>/metadata.json``.

    Falls back to the flat top-level key for metadata.json written before the
    nested-``paths`` layout.
    """
    if run_dir is None:
        raise FileNotFoundError(
            f"No run dir resolved; pass {cli_flag} explicitly."
        )
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} does not exist; pass {cli_flag} explicitly."
        )
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Failed to parse {meta_path}: {exc}") from exc
    src = (meta.get("paths") or {}).get(meta_key) or meta.get(meta_key)
    if not src:
        raise FileNotFoundError(
            f"{meta_path}::paths.{meta_key} missing or empty; "
            f"pass {cli_flag} explicitly."
        )
    return Path(src)


def _registered_from_metadata(run_dir: Path | None) -> Path | None:
    """Best-effort read of ``source_registration_dir`` (``None`` on miss)."""
    if run_dir is None:
        return None
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse %s: %s", meta_path, exc)
        return None
    src = meta.get("source_registration_dir")
    if not src:
        return None
    return Path(src)


# ── STL loading ─────────────────────────────────────────────────────────────

def _pv_faces_to_tri(mesh: pv.PolyData) -> np.ndarray:
    """``(n_faces, 3)`` int triangle indices from PV's flat ``[3,i,j,k,3,i,j,k,...]``."""
    f = np.asarray(mesh.faces, dtype=np.int64)
    if f.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    return f.reshape(-1, 4)[:, 1:]


def _load_stl(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load an STL; return (verts (N,3) float32, faces (M,3) int32) or None on failure."""
    try:
        mesh = pv.read(str(path))
    except Exception as exc:  # VTK/PyVista readers raise varied types
        logger.warning("Failed to read %s: %s", path.name, exc)
        return None
    verts = np.asarray(mesh.points, dtype=np.float32)
    faces = _pv_faces_to_tri(mesh).astype(np.int32, copy=False)
    return verts, faces


def _load_landmarks_json(path: Path) -> list[tuple[str, float, float, float]]:
    """Parse a Scalismo-style landmarks JSON file → ``[(id, x, y, z), ...]``.

    Empty on missing file or malformed content (logged at WARNING). Accepts
    either ``coordinates`` or ``point`` keys; entries that don't parse cleanly
    are silently dropped.
    """
    if not path.exists():
        return []
    try:
        items = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse landmark JSON %s: %s", path, exc)
        return []
    if not isinstance(items, list):
        logger.warning("Landmark JSON %s is not a list (got %s)", path, type(items).__name__)
        return []
    out: list[tuple[str, float, float, float]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        coords = entry.get("coordinates") or entry.get("point")
        if not isinstance(coords, (list, tuple)) or len(coords) < 3:
            continue
        try:
            x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
        except (TypeError, ValueError):
            continue
        lid = str(entry.get("id", ""))
        out.append((lid, x, y, z))
    return out


def _discover_patients(raw_dir: Path, reg_dir: Path) -> list[int]:
    """Return sorted patient IDs that have all 24 raw AND all 24 registered STLs.

    Walks the sharded ``<root>/<block>/<pid>/<pid>_rib<lab>_<side>.stl``
    layout and tracks which (lab, side) pairs are present per patient.
    """
    def _scan(d: Path) -> dict[int, set[tuple[int, str]]]:
        found: dict[int, set[tuple[int, str]]] = {}
        if not d.exists():
            return found
        for block_dir in d.iterdir():
            if not block_dir.is_dir() or block_dir.name.startswith("._"):
                continue
            for pid_dir in block_dir.iterdir():
                if not pid_dir.is_dir() or not pid_dir.name.isdigit():
                    continue
                pid = int(pid_dir.name)
                for f in pid_dir.iterdir():
                    if f.name.startswith("._"):
                        continue
                    mm = _RIB_FILE_RE.match(f.name)
                    if not mm:
                        continue
                    lab = int(mm.group(1))
                    side = mm.group(2)
                    if (lab, side) in RIB_ORDER:
                        found.setdefault(pid, set()).add((lab, side))
        return found

    raw_pids = _scan(raw_dir)
    reg_pids = _scan(reg_dir)
    target = set(RIB_ORDER)
    complete = [
        pid for pid in raw_pids
        if pid in reg_pids
        and raw_pids[pid] == target
        and reg_pids[pid] == target
    ]
    return sorted(complete)


def _rigid_align_raw_to_reg(
    raw_meshes: list[tuple[np.ndarray, np.ndarray]],
    reg_meshes: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal Procrustes (rotation + translation, no scale) from raw → reg.

    Uses the 24 per-rib centroids as the sparse correspondence. Scale is
    excluded – body size is a study variable.
    """
    raw_c = np.stack([v.mean(axis=0) for v, _ in raw_meshes])   # (24, 3)
    reg_c = np.stack([v.mean(axis=0) for v, _ in reg_meshes])   # (24, 3)

    raw_mu = raw_c.mean(axis=0)
    reg_mu = reg_c.mean(axis=0)
    M = (reg_c - reg_mu).T @ (raw_c - raw_mu)                   # (3, 3)
    U, _, Vt = np.linalg.svd(M)
    d = np.linalg.det(Vt.T @ U.T)
    R = (Vt.T @ np.diag([1.0, 1.0, d]) @ U.T).astype(np.float32)
    t = (reg_mu - raw_mu @ R).astype(np.float32)
    return R, t


def _load_patient(
    pid: int, raw_dir: Path, reg_dir: Path,
    *, align_rigid: bool = True,
    landmark_dir: Path | None = None,
) -> dict[str, list[dict]] | None:
    """Load all 24 raw + 24 registered STLs (and optional landmarks) for one patient.

    Returns ``{"raw":  [{x, y, z, i, j, k}*24],
              "reg":  [...],
              "lm_raw":  [{x, y, z, ids}*24],
              "lm_reg":  [...]}``
    or ``None`` if any raw or registered rib mesh failed to load.

    When ``align_rigid`` is True (default), the raw cage is rigidly
    transformed (translation + rotation, **never scale**) onto the
    registered cage's frame via 24-centroid Procrustes. Raw landmarks
    follow the same transform so they stay on the (now-aligned) raw
    surface; registered landmarks are already in the registered frame.
    """
    raw_meshes: list[tuple[np.ndarray, np.ndarray]] = []
    reg_meshes: list[tuple[np.ndarray, np.ndarray]] = []
    for kind, base, bucket in (
        ("raw", raw_dir, raw_meshes),
        ("reg", reg_dir, reg_meshes),
    ):
        pdir = patient_stl_dir(base, pid)
        for lab, side in RIB_ORDER:
            stl = pdir / f"{pid}_rib{lab}_{side}.stl"
            res = _load_stl(stl)
            if res is None:
                logger.warning("Patient %d: missing %s – skipping patient", pid, stl.name)
                return None
            bucket.append(res)

    # Per-rib landmarks in two frames: raw (extracted, endpoints A/B) and
    # registered (target frame).
    raw_lms: list[np.ndarray] = []
    reg_lms: list[np.ndarray] = []
    raw_lm_ids: list[list[str]] = []
    reg_lm_ids: list[list[str]] = []
    if landmark_dir is not None:
        lm_pdir = patient_stl_dir(landmark_dir, pid)
        for lab, side in RIB_ORDER:
            rib_id = f"rib{lab}_{side}"
            ext = _load_landmarks_json(lm_pdir / f"{rib_id}{_LM_EXTRACTED_SUFFIX}")
            reg = _load_landmarks_json(lm_pdir / f"{rib_id}{_LM_REGISTERED_SUFFIX}")
            raw_lms.append(
                np.asarray([(x, y, z) for _, x, y, z in ext], dtype=np.float32).reshape(-1, 3)
            )
            reg_lms.append(
                np.asarray([(x, y, z) for _, x, y, z in reg], dtype=np.float32).reshape(-1, 3)
            )
            raw_lm_ids.append([lid for lid, *_ in ext])
            reg_lm_ids.append([lid for lid, *_ in reg])
    else:
        for _ in RIB_ORDER:
            raw_lms.append(np.empty((0, 3), dtype=np.float32))
            reg_lms.append(np.empty((0, 3), dtype=np.float32))
            raw_lm_ids.append([])
            reg_lm_ids.append([])

    if align_rigid:
        R, t = _rigid_align_raw_to_reg(raw_meshes, reg_meshes)
        # Skip the (near-)identity no-op to avoid a needless float copy.
        if (np.linalg.norm(R - np.eye(3, dtype=np.float32)) > 1e-6
                or float(np.linalg.norm(t)) > 1e-6):
            raw_meshes = [(v @ R + t, f) for v, f in raw_meshes]
            raw_lms = [
                (lm @ R + t).astype(np.float32) if lm.size else lm
                for lm in raw_lms
            ]

    out: dict[str, list[dict]] = {
        "raw":    [], "reg":    [],
        "lm_raw": [], "lm_reg": [],
    }
    for kind, meshes in (("raw", raw_meshes), ("reg", reg_meshes)):
        for verts, faces in meshes:
            v_rounded = np.round(verts, 2)   # 0.01 mm precision
            out[kind].append({
                "x": v_rounded[:, 0].tolist(),
                "y": v_rounded[:, 1].tolist(),
                "z": v_rounded[:, 2].tolist(),
                "i": faces[:, 0].tolist(),
                "j": faces[:, 1].tolist(),
                "k": faces[:, 2].tolist(),
            })
    for kind, lms, ids in (
        ("lm_raw", raw_lms, raw_lm_ids),
        ("lm_reg", reg_lms, reg_lm_ids),
    ):
        for lm, lm_ids in zip(lms, ids, strict=True):
            if lm.size:
                lm_rounded = np.round(lm, 2)
                out[kind].append({
                    "x":   lm_rounded[:, 0].tolist(),
                    "y":   lm_rounded[:, 1].tolist(),
                    "z":   lm_rounded[:, 2].tolist(),
                    "ids": list(lm_ids),
                })
            else:
                out[kind].append({"x": [], "y": [], "z": [], "ids": []})
    return out


def _load_template(
    template_pid: int, raw_dir: Path, reg_dir: Path,
    landmark_dir: Path | None,
) -> dict[str, list[dict]] | None:
    """Load the template patient's ``raw`` + ``lm_raw`` tiers for the overlay."""
    rec = _load_patient(
        template_pid, raw_dir, reg_dir,
        align_rigid=True, landmark_dir=landmark_dir,
    )
    if rec is None:
        return None
    return {"raw": rec["raw"], "lm_raw": rec["lm_raw"]}


# ── Scene-range computation ─────────────────────────────────────────────────

def _scene_range(data: dict[int, dict]) -> dict[str, list[float]]:
    """Tight axis-aligned bounding box over every loaded mesh + landmark,
    padded 5%."""
    los, his = [], []
    for pid_data in data.values():
        for kind in ("raw", "reg", "lm_raw", "lm_reg"):
            for entry in pid_data.get(kind, []):
                xs = entry["x"]; ys = entry["y"]; zs = entry["z"]
                if not xs:
                    continue
                los.append((min(xs), min(ys), min(zs)))
                his.append((max(xs), max(ys), max(zs)))
    if not los:
        return {"x": [-200, 200], "y": [-200, 200], "z": [-200, 200]}
    lo = np.min(los, axis=0)
    hi = np.max(his, axis=0)
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    return {
        "x": [float(lo[0]), float(hi[0])],
        "y": [float(lo[1]), float(hi[1])],
        "z": [float(lo[2]), float(hi[2])],
    }


# ── HTML template ───────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diagnostic Viewer – Template / Raw / Registered Rib STLs</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: __FONT_FAMILY__;
       background: #fafafa; color: #222222; }
#header { padding: 14px 20px 8px; background: #ffffff;
          border-bottom: 1px solid #e0e0e0; }
#header h2 { font-size: 17px; font-weight: 600; color: #111111; }
#header p  { font-size: 12px; color: #666; margin-top: 2px; }
#plot-wrap { height: 70vh; background: #ffffff; border-bottom: 1px solid #e0e0e0; }
#plot      { width: 100%; height: 100%; }
#controls  { background: #ffffff; padding: 12px 20px 16px; }
#bar       { display: flex; align-items: center; gap: 14px;
             flex-wrap: wrap; padding: 4px 0 10px; }
.bar-label { font-size: 12px; color: #444; font-weight: 500; }
#patient-select, #rib-focus {
  font-size: 12px; padding: 3px 6px; border-radius: 4px;
  border: 1px solid #ccc; font-family: inherit; color: #333;
  background: #fff; min-width: 140px;
}
.toggle-row { display: inline-flex; align-items: center; gap: 6px;
              font-size: 12px; color: #444; }
.swatch { display: inline-block; width: 18px; height: 10px;
          border: 1px solid #999; border-radius: 2px; }
.swatch-raw { background: __RAW_COLOR__; }
.swatch-tpl { background: __TPL_COLOR__; }
.swatch-reg { background: __REG_COLOR__; }
.dot   { display: inline-block; width: 10px; height: 10px;
         border-radius: 50%; border: 1px solid #333;
         vertical-align: middle; }
.dot-raw { background: __RAW_COLOR__; }
.dot-tpl { background: __TPL_COLOR__; }
.dot-reg { background: __REG_COLOR__; }
#status-txt { font-size: 12px; color: #555; padding-top: 6px; }
.legend { font-size: 11px; color: #777; }
.help { font-size: 11px; color: #888; padding-top: 6px; line-height: 1.45; }
#internal-banner { background: #c0392b; color: #ffffff; text-align: center;
                   font-size: 15px; font-weight: 700; letter-spacing: 0.12em;
                   text-transform: uppercase; padding: 10px 20px;
                   border-bottom: 2px solid #8b271c; }
</style>
</head>
<body>

<div id="internal-banner">INTERNAL ONLY &mdash; DO NOT PUBLISH</div>

<div id="header">
  <h2>Diagnostic Viewer &mdash; Template / Raw / Registered Rib STLs</h2>
  <p>The template patient's per-rib meshes + A/B landmarks are always
     overlaid (purple). Pick a patient from the dropdown to see their
     raw extracted (blue) and Scalismo-registered (orange) meshes on
     top of the template &nbsp;&bull;&nbsp;
     N&thinsp;=&thinsp;__N_PATIENTS__ patients embedded &nbsp;&bull;&nbsp;
     12 ribs &times; L/R = 24 identities per side</p>
</div>

<div id="plot-wrap"><div id="plot"></div></div>

<div id="controls">
  <div id="bar">
    <span class="bar-label">Patient:</span>
    <input id="patient-select" type="text" list="patient-list"
           placeholder="Patient ID"
           autocomplete="off" spellcheck="false"
           oninput="onPatientInput(this.value)">
    <datalist id="patient-list"></datalist>

    <label class="toggle-row" id="tpl-toggle-wrap" style="display:none">
      <input type="checkbox" id="show-tpl" checked onchange="onToggle()">
      <span class="swatch swatch-tpl"></span> Template
    </label>
    <label class="toggle-row">
      <input type="checkbox" id="show-raw" checked onchange="onToggle()">
      <span class="swatch swatch-raw"></span> Raw (extraction)
    </label>
    <label class="toggle-row">
      <input type="checkbox" id="show-reg" checked onchange="onToggle()">
      <span class="swatch swatch-reg"></span> Registered
    </label>
    <label class="toggle-row" id="lm-toggle-wrap" style="display:none">
      <input type="checkbox" id="show-lm" checked onchange="onToggle()">
      <span class="dot dot-tpl"></span><span class="dot dot-raw"
        style="margin-left:-2px"></span><span class="dot dot-reg"
        style="margin-left:-2px"></span>
      Landmarks
    </label>

    <span class="bar-label" style="margin-left:14px">Rib focus:</span>
    <select id="rib-focus" onchange="onFocus()"></select>
  </div>
  <div id="status-txt"></div>
  <div class="help">
    Template (purple) = reference patient, overlaid in every view. Registered
    (orange) stopping short of raw (blue) at the medial / lateral end signals
    registration truncation; raw missing material the NIfTI has signals
    extraction failure. Landmark dots (when shown) are the A (proximal /
    costovertebral) and B (distal / sternal) endpoints per rib.
  </div>
</div>

<script>
const M = __MODEL_JSON__;

const N_RIBS = M.rib_order.length;   // 24
// Trace layout (stable across renders for Plotly.restyle positional indexing):
//   [0 .. N-1]         template meshes        (mesh3d)
//   [N .. 2N-1]        raw meshes             (mesh3d, per patient)
//   [2N .. 3N-1]       registered meshes      (mesh3d, per patient)
//   [3N]               template landmarks     (cohort-wide scatter3d)
//   [3N+1]             raw landmarks          (cohort-wide scatter3d)
//   [3N+2]             registered landmarks   (cohort-wide scatter3d)
// Three pooled landmark traces (instead of 3N per-rib) keep the total
// count at 3N+3 so 3D orbit redraws stay responsive.
const TPL_MESH_BASE = 0;
const RAW_MESH_BASE = N_RIBS;
const REG_MESH_BASE = 2 * N_RIBS;
const TPL_LM_IDX    = 3 * N_RIBS;
const RAW_LM_IDX    = 3 * N_RIBS + 1;
const REG_LM_IDX    = 3 * N_RIBS + 2;
const N_TRACES      = 3 * N_RIBS + 3;
const N_MESH_SLOTS  = 3 * N_RIBS;
// renderPatient touches only these slots; template slots are pre-filled at init.
const PATIENT_MESH_FIRST = RAW_MESH_BASE;       // inclusive
const PATIENT_MESH_LAST  = REG_MESH_BASE + N_RIBS - 1;   // inclusive

const HAS_LANDMARKS = M.has_landmarks === true;
const HAS_TEMPLATE  = M.has_template === true;

let selectedPid = null;
let focusIdx = -1;   // -1 = all visible; else index into M.rib_order

function buildEmptyMeshTrace(name, color, kind) {
  return {
    type: 'mesh3d',
    x: [], y: [], z: [],
    i: [], j: [], k: [],
    color: color,
    opacity: __MESH_OPACITY__,
    flatshading: true,
    lighting: {ambient: 0.7, diffuse: 0.5, specular: 0.1, roughness: 0.9},
    hoverinfo: 'skip',
    showscale: false,
    name: name,
    legendgroup: kind,
    showlegend: false,
    visible: true,
  };
}

function buildEmptyLandmarkTrace(name, color, kind) {
  // One pooled trace per tier; hover disabled (it stalls 3D orbit redraws).
  return {
    type: 'scatter3d',
    mode: 'markers',
    x: [], y: [], z: [],
    marker: {
      size: __LM_MARKER_SIZE__,
      color: color,
      line: {color: '#222', width: 0.5},
    },
    hoverinfo: 'skip',
    name: name,
    legendgroup: kind,
    showlegend: false,
    visible: true,
  };
}

const traces = new Array(N_TRACES);
for (let r = 0; r < N_RIBS; r++) {
  traces[TPL_MESH_BASE + r] = buildEmptyMeshTrace(
    `Tpl \\u2014 ${M.rib_display[r]}`, M.tpl_color, 'tpl');
  traces[RAW_MESH_BASE + r] = buildEmptyMeshTrace(
    `Raw \\u2014 ${M.rib_display[r]}`, M.raw_color, 'raw');
  traces[REG_MESH_BASE + r] = buildEmptyMeshTrace(
    `Reg \\u2014 ${M.rib_display[r]}`, M.reg_color, 'reg');
}
traces[TPL_LM_IDX] = buildEmptyLandmarkTrace('Tpl landmarks', M.tpl_color, 'tpl');
traces[RAW_LM_IDX] = buildEmptyLandmarkTrace('Raw landmarks', M.raw_color, 'raw');
traces[REG_LM_IDX] = buildEmptyLandmarkTrace('Reg landmarks', M.reg_color, 'reg');

// Pre-fill the template-tier traces once; renderPatient never touches them.
if (M.template) {
  for (let r = 0; r < N_RIBS; r++) {
    const m = M.template.raw && M.template.raw[r];
    if (!m) continue;
    traces[TPL_MESH_BASE + r].x = m.x;
    traces[TPL_MESH_BASE + r].y = m.y;
    traces[TPL_MESH_BASE + r].z = m.z;
    traces[TPL_MESH_BASE + r].i = m.i;
    traces[TPL_MESH_BASE + r].j = m.j;
    traces[TPL_MESH_BASE + r].k = m.k;
  }
  const lmTpl = (function () {
    const x = [], y = [], z = [];
    if (M.template.lm_raw) {
      for (let r = 0; r < N_RIBS; r++) {
        const lr = M.template.lm_raw[r];
        if (!lr || !lr.x.length) continue;
        for (let k = 0; k < lr.x.length; k++) {
          x.push(lr.x[k]); y.push(lr.y[k]); z.push(lr.z[k]);
        }
      }
    }
    return {x, y, z};
  })();
  traces[TPL_LM_IDX].x = lmTpl.x;
  traces[TPL_LM_IDX].y = lmTpl.y;
  traces[TPL_LM_IDX].z = lmTpl.z;
}

const LAYOUT = {
  scene: {
    xaxis: {visible: false, range: M.scene_range.x, autorange: false},
    yaxis: {visible: false, range: M.scene_range.y, autorange: false},
    zaxis: {visible: false, range: M.scene_range.z, autorange: false},
    bgcolor: 'white',
    camera: {eye: {x: 1.5, y: -1.8, z: 0.5}, up: {x: 0, y: 0, z: -1}},
    aspectmode: 'data',
  },
  font: {family: M.font_family, color: '#222222'},
  margin: {l: 0, r: 0, b: 0, t: 0},
  paper_bgcolor: 'white',
  uirevision: 'keep',
  showlegend: false,
};
const CONFIG = {responsive: true, displayModeBar: true, displaylogo: false,
                modeBarButtonsToRemove: ['resetCameraLastSave3d']};

Plotly.newPlot('plot', traces, LAYOUT, CONFIG);

// ── Populate dropdowns ─────────────────────────────────────────────────────
{
  const dl = document.getElementById('patient-list');
  dl.innerHTML = M.patients.map(p => `<option value="${p}">`).join('');
}
{
  const sel = document.getElementById('rib-focus');
  let opts = '<option value="-1">All ribs</option>';
  for (let r = 0; r < N_RIBS; r++) {
    opts += `<option value="${r}">${M.rib_display[r]}</option>`;
  }
  sel.innerHTML = opts;
}
if (HAS_LANDMARKS) {
  document.getElementById('lm-toggle-wrap').style.display = 'inline-flex';
}
if (HAS_TEMPLATE) {
  document.getElementById('tpl-toggle-wrap').style.display = 'inline-flex';
}

// ── State updates ──────────────────────────────────────────────────────────
// Each of these functions issues exactly one Plotly.restyle. Back-to-back
// restyles on a 3D WebGL scene throttle the orbit/pan redraw queue.
function applyVisibility() {
  const tplEl   = document.getElementById('show-tpl');
  const showTpl = HAS_TEMPLATE && tplEl && tplEl.checked;
  const showRaw = document.getElementById('show-raw').checked;
  const showReg = document.getElementById('show-reg').checked;
  const lmEl    = document.getElementById('show-lm');
  const showLm  = HAS_LANDMARKS && lmEl && lmEl.checked;
  const visible = new Array(N_TRACES);
  const opacity = new Array(N_TRACES);
  for (let r = 0; r < N_RIBS; r++) {
    const focused = (focusIdx < 0) || (focusIdx === r);
    const dimmedOpacity = focused ? __MESH_OPACITY__ : __MESH_OPACITY_DIMMED__;
    visible[TPL_MESH_BASE + r] = showTpl;
    visible[RAW_MESH_BASE + r] = showRaw;
    visible[REG_MESH_BASE + r] = showReg;
    opacity[TPL_MESH_BASE + r] = dimmedOpacity;
    opacity[RAW_MESH_BASE + r] = dimmedOpacity;
    opacity[REG_MESH_BASE + r] = dimmedOpacity;
  }
  visible[TPL_LM_IDX] = showLm && showTpl;
  visible[RAW_LM_IDX] = showLm && showRaw;
  visible[REG_LM_IDX] = showLm && showReg;
  // Landmarks stay fully opaque regardless of rib focus.
  opacity[TPL_LM_IDX] = 1.0;
  opacity[RAW_LM_IDX] = 1.0;
  opacity[REG_LM_IDX] = 1.0;
  Plotly.restyle('plot', {visible, opacity});
}

function clearPlot() {
  // Clear patient-tier slots only; template-tier slots stay populated.
  const xyz = new Array(N_TRACES);
  const ijk = new Array(N_TRACES);
  for (let r = 0; r < N_RIBS; r++) {
    xyz[RAW_MESH_BASE + r] = [];
    xyz[REG_MESH_BASE + r] = [];
    ijk[RAW_MESH_BASE + r] = [];
    ijk[REG_MESH_BASE + r] = [];
  }
  xyz[RAW_LM_IDX] = [];
  xyz[REG_LM_IDX] = [];
  Plotly.restyle('plot',
    {x: xyz, y: xyz, z: xyz, i: ijk, j: ijk, k: ijk});
  selectedPid = null;
  updateStatus();
}

// Pack all 24 ribs of one tier into one (x, y, z) bundle for a pooled trace.
function packLandmarks(buckets) {
  const x = [], y = [], z = [];
  if (buckets) {
    for (let r = 0; r < N_RIBS; r++) {
      const lr = buckets[r];
      if (!lr || !lr.x.length) continue;
      for (let k = 0; k < lr.x.length; k++) {
        x.push(lr.x[k]); y.push(lr.y[k]); z.push(lr.z[k]);
      }
    }
  }
  return {x, y, z};
}

function renderPatient(pid) {
  const rec = M.data[pid];
  if (!rec) { clearPlot(); return; }
  selectedPid = pid;

  // One restyle for x/y/z + i/j/k on patient-tier slots; template slots
  // are left untouched. `undefined` entries in the restyle array are no-ops
  // (vs `null`, which trips per-attribute validation on mesh3d/scatter3d).
  const x = new Array(N_TRACES);
  const y = new Array(N_TRACES);
  const z = new Array(N_TRACES);
  const ii = new Array(N_TRACES);
  const jj = new Array(N_TRACES);
  const kk = new Array(N_TRACES);
  for (let r = 0; r < N_RIBS; r++) {
    const a = rec.raw[r], b = rec.reg[r];
    x[RAW_MESH_BASE + r] = a.x; y[RAW_MESH_BASE + r] = a.y; z[RAW_MESH_BASE + r] = a.z;
    ii[RAW_MESH_BASE + r] = a.i; jj[RAW_MESH_BASE + r] = a.j; kk[RAW_MESH_BASE + r] = a.k;
    x[REG_MESH_BASE + r] = b.x; y[REG_MESH_BASE + r] = b.y; z[REG_MESH_BASE + r] = b.z;
    ii[REG_MESH_BASE + r] = b.i; jj[REG_MESH_BASE + r] = b.j; kk[REG_MESH_BASE + r] = b.k;
  }
  const lmRaw = packLandmarks(rec.lm_raw);
  const lmReg = packLandmarks(rec.lm_reg);
  x[RAW_LM_IDX] = lmRaw.x; y[RAW_LM_IDX] = lmRaw.y; z[RAW_LM_IDX] = lmRaw.z;
  x[REG_LM_IDX] = lmReg.x; y[REG_LM_IDX] = lmReg.y; z[REG_LM_IDX] = lmReg.z;
  Plotly.restyle('plot', {x, y, z, i: ii, j: jj, k: kk});
  applyVisibility();
  updateStatus();
}

function updateStatus() {
  const el = document.getElementById('status-txt');
  // Static template counts (always shown when template is present).
  let tpl_v = 0, tpl_f = 0, lm_tpl = 0;
  if (M.template) {
    for (let r = 0; r < N_RIBS; r++) {
      const tm = M.template.raw && M.template.raw[r];
      if (tm) { tpl_v += tm.x.length; tpl_f += tm.i.length; }
      const tl = M.template.lm_raw && M.template.lm_raw[r];
      if (tl) lm_tpl += tl.x.length;
    }
  }
  const tplPrefix = HAS_TEMPLATE
    ? `Template pid ${M.template_pid}: ${tpl_v.toLocaleString()} verts / `
      + `${tpl_f.toLocaleString()} faces` + (HAS_LANDMARKS ? ` / ${lm_tpl} lm` : '')
      + ' \\u00b7 '
    : '';

  if (selectedPid == null) {
    el.textContent = tplPrefix + 'no patient selected.';
    return;
  }
  const rec = M.data[selectedPid];
  if (!rec) {
    el.textContent = tplPrefix + `patient ${selectedPid} – no data.`;
    return;
  }
  let raw_v = 0, raw_f = 0, reg_v = 0, reg_f = 0;
  let lm_raw = 0, lm_reg = 0;
  for (let r = 0; r < N_RIBS; r++) {
    raw_v += rec.raw[r].x.length;
    raw_f += rec.raw[r].i.length;
    reg_v += rec.reg[r].x.length;
    reg_f += rec.reg[r].i.length;
    if (rec.lm_raw && rec.lm_raw[r]) lm_raw += rec.lm_raw[r].x.length;
    if (rec.lm_reg && rec.lm_reg[r]) lm_reg += rec.lm_reg[r].x.length;
  }
  const focusTxt = focusIdx < 0 ? 'all ribs' : `focus: ${M.rib_display[focusIdx]}`;
  const lmTxt = HAS_LANDMARKS
    ? ` \\u00b7 landmarks: ${lm_raw} raw / ${lm_reg} reg`
    : '';
  el.textContent =
    tplPrefix
    + `Patient ${selectedPid} \\u2014 raw: ${raw_v.toLocaleString()} verts / `
    + `${raw_f.toLocaleString()} faces`
    + ` \\u00b7 registered: ${reg_v.toLocaleString()} verts / ${reg_f.toLocaleString()} faces`
    + lmTxt
    + ` \\u00b7 ${focusTxt}`;
}

function onPatientInput(value) {
  const trimmed = (value || '').trim();
  if (trimmed === '') { clearPlot(); return; }
  const pid = +trimmed;
  if (!Number.isFinite(pid) || !(pid in M.data)) {
    clearPlot();
    document.getElementById('status-txt').textContent =
      `Patient ${trimmed} not embedded in this viewer.`;
    return;
  }
  renderPatient(pid);
}

function onToggle() { applyVisibility(); }

function onFocus() {
  const v = +document.getElementById('rib-focus').value;
  focusIdx = Number.isFinite(v) ? v : -1;
  applyVisibility();
  updateStatus();
}

// Auto-load the first patient.
if (M.patients.length > 0) {
  const first = M.patients[0];
  document.getElementById('patient-select').value = first;
  renderPatient(first);
}
</script>

</body>
</html>
"""


# ── Main entry point ────────────────────────────────────────────────────────

def export_html(
    output_path: str | Path | None = None,
    results_dir: Path | None = None,
    patient_ids: list[int] | None = None,
    max_patients: int = 30,
    target_dir: Path | None = None,
    registered_dir: Path | None = None,
    landmark_dir: Path | None = None,
    template_pid: int | None = None,
    align_rigid: bool = True,
) -> Path:
    """Build the diagnostic viewer HTML and write it to disk.

    Parameters
    ----------
    output_path
        Where to write the HTML. Defaults to
        ``<results_dir>/figures/viewer_diagnostic.html``.
    results_dir
        SSM run dir.  Used to default ``output_path`` and to auto-resolve
        ``registered_dir`` / ``landmark_dir`` from
        ``<run-dir>/metadata.json`` when not supplied explicitly.
    patient_ids
        Patients to embed. If ``None``, pick the first ``max_patients``
        patients that have a complete set of raw + registered STLs.
    max_patients
        Cap on the number of patients to embed when ``patient_ids`` is None.
    target_dir
        Override for the raw (target) STL directory.  Default:
        ``paths.extracted_stl_dir`` recorded in ``<run-dir>/metadata.json``.
    registered_dir
        Override for the registered STL directory.  Default:
        ``paths.registered_stl_dir`` recorded in ``<run-dir>/metadata.json``.
    landmark_dir
        Override for the per-rib landmark directory.  Default:
        ``paths.landmark_dir`` recorded in ``<run-dir>/metadata.json``
        (only present when landmarks were generated for this run).
        When neither resolves, the viewer renders meshes only and hides
        the landmark toggle.
    align_rigid
        Rigidly transform (translate + rotate, **never scale**) the raw
        cage onto the registered cage's frame, removing the global pose
        offset.  Default: True.  Disable to see the raw vs registered
        spatial offset.  When landmarks are present, the **same** rigid
        transform is applied to the extracted-frame landmarks so they
        continue to sit on the (aligned) raw mesh.
    """
    # Only resolve a default run dir when actually needed: for output-path
    # default, or for metadata.json lookup when --target-dir / --registered-dir
    # is unset.  ``landmark_dir`` does NOT force resolution – it's optional;
    # if the run dir happens to be available, ``_resolve_landmark_dir`` will
    # opportunistically read ``paths.landmark_dir`` from its metadata, but
    # missing landmarks just mean the viewer renders meshes only.
    needs_run_dir = (
        output_path is None
        or target_dir is None
        or registered_dir is None
    )
    if results_dir is not None:
        run_dir: Path | None = results_dir
    elif needs_run_dir:
        run_dir = _default_results_dir()
    else:
        run_dir = None

    extraction_dir, reg_dir = _resolve_mesh_dirs(
        run_dir, target_dir=target_dir, registered_dir=registered_dir,
    )
    lm_dir = _resolve_landmark_dir(run_dir, landmark_dir)
    resolved_template_pid = _resolve_template_pid(
        run_dir, template_pid, reg_dir,
    )
    logger.info("Run dir       : %s", run_dir if run_dir else "(not resolved)")
    logger.info("Raw STL dir   : %s", extraction_dir)
    logger.info("Reg STL dir   : %s", reg_dir)
    logger.info("Landmark dir  : %s", lm_dir if lm_dir else "(none – meshes only)")
    logger.info("Template pid  : %s",
                resolved_template_pid if resolved_template_pid is not None
                else "(none – no template overlay)")

    available = _discover_patients(extraction_dir, reg_dir)
    logger.info("Patients with complete raw + reg sets: %d", len(available))
    if not available:
        raise RuntimeError(
            f"No patients with all 24 raw STLs in {extraction_dir} AND all 24 "
            f"registered STLs in {reg_dir}. Re-run extraction + registration first."
        )

    if patient_ids is not None:
        avail_set = set(available)
        chosen = [p for p in patient_ids if p in avail_set]
        missing = [p for p in patient_ids if p not in avail_set]
        if missing:
            logger.warning(
                "Skipping %d requested patients with incomplete STL sets: %s",
                len(missing), missing[:10],
            )
        if not chosen:
            raise RuntimeError(
                f"None of the requested patient IDs {patient_ids} have complete "
                f"STL sets in both {extraction_dir} and {reg_dir}."
            )
    else:
        chosen = available[:max_patients]
        logger.info("Embedding first %d of %d available patients", len(chosen), len(available))

    logger.info(
        "Rigid alignment: %s",
        "ON (raw → registered, translate + rotate, no scale)" if align_rigid else "OFF",
    )
    data: dict[int, dict] = {}
    for pid in chosen:
        rec = _load_patient(
            pid, extraction_dir, reg_dir,
            align_rigid=align_rigid,
            landmark_dir=lm_dir,
        )
        if rec is None:
            continue
        data[pid] = rec
    if not data:
        raise RuntimeError("Failed to load mesh data for any chosen patient.")

    # Load the template patient (separately from `data`) so the JS layer
    # can keep its trace slots constant across patient swaps.
    template_record: dict | None = None
    if resolved_template_pid is not None:
        template_record = _load_template(
            resolved_template_pid, extraction_dir, reg_dir, lm_dir,
        )
        if template_record is None:
            logger.warning(
                "Template pid %d is missing STLs in %s – template tier disabled.",
                resolved_template_pid, extraction_dir,
            )

    # Cohort-wide flag: are there ANY landmarks across ANY embedded
    # patient × rib (or in the template)? Drives the landmark UI toggle.
    has_landmarks = any(
        any(lm["x"] for lm in rec.get("lm_raw", []))
        or any(lm["x"] for lm in rec.get("lm_reg", []))
        for rec in data.values()
    ) or (
        template_record is not None
        and any(lm["x"] for lm in template_record.get("lm_raw", []))
    )
    has_template = template_record is not None
    logger.info(
        "Landmarks     : %s",
        "embedded" if has_landmarks
        else "none found (centerline JSONs missing in landmark_dir?)",
    )
    logger.info("Template tier : %s", "embedded" if has_template else "none")

    rib_display = [display_from_seg(lab, side) for lab, side in RIB_ORDER]

    model = {
        "patients":      sorted(data.keys()),
        "rib_order":     [[lab, side] for lab, side in RIB_ORDER],
        "rib_display":   rib_display,
        "data":          {str(pid): rec for pid, rec in data.items()},
        "scene_range":   _scene_range(
            {**data,
             **({-1: template_record} if template_record is not None else {})}
        ),
        "raw_color":     _RAW_COLOR,
        "tpl_color":     _TPL_COLOR,
        "reg_color":     _REG_COLOR,
        "font_family":   FONT_FAMILY,
        "has_landmarks": has_landmarks,
        "has_template":  has_template,
        "template":      template_record,
        "template_pid":  resolved_template_pid,
    }

    model_json = json.dumps(model, separators=(",", ":"))

    html = (
        _HTML_TEMPLATE
        .replace("__MODEL_JSON__",            model_json)
        .replace("__FONT_FAMILY__",           FONT_FAMILY)
        .replace("__N_PATIENTS__",            str(len(data)))
        .replace("__RAW_COLOR__",             _RAW_COLOR)
        .replace("__TPL_COLOR__",             _TPL_COLOR)
        .replace("__REG_COLOR__",             _REG_COLOR)
        .replace("__MESH_OPACITY_DIMMED__",   f"{_MESH_OPACITY_DIMMED}")
        .replace("__MESH_OPACITY__",          f"{_MESH_OPACITY}")
        .replace("__LM_MARKER_SIZE__",        f"{_LM_MARKER_SIZE}")
    )

    if output_path is None:
        assert run_dir is not None
        figs = run_dir / "figures"
        figs.mkdir(parents=True, exist_ok=True)
        output_path = figs / "viewer_diagnostic.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(
        "Diagnostic viewer written → %s (%d KB, %d patients)",
        output_path, output_path.stat().st_size // 1024, len(data),
    )
    return output_path


# ── CLI ─────────────────────────────────────────────────────────────────────

def _parse_pid_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--results", type=Path, default=None,
        help="SSM run dir (used to default --output and to auto-resolve "
             "--registered-dir from its metadata.json).",
    )
    ap.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Write the HTML to this path "
             "(default: <run_dir>/figures/viewer_diagnostic.html).",
    )
    ap.add_argument(
        "--target-dir", type=Path, default=None,
        help="Per-rib STL directory Scalismo registered against – the raw "
             "mesh-extraction output (default: paths.extracted_stl_dir from "
             "<run-dir>/metadata.json).",
    )
    ap.add_argument(
        "--registered-dir", type=Path, default=None,
        help="Registered per-rib STL directory (default: "
             "paths.registered_stl_dir from <run-dir>/metadata.json).",
    )
    ap.add_argument(
        "--landmark-dir", type=Path, default=None,
        help="Per-rib landmark directory (default: paths.landmark_dir from "
             "<run-dir>/metadata.json). Loads <ribId>_extracted.json and "
             "<ribId>_registered.json when present; absent files are silently "
             "skipped.",
    )
    ap.add_argument(
        "--template-pid", type=int, default=None,
        help="Patient ID to render as the always-overlaid template tier "
             "(default: read from <registered_dir>/template_id.txt; if that "
             "file is absent the template overlay is silently disabled).",
    )
    ap.add_argument(
        "--patient-ids", type=_parse_pid_list, default=None,
        help="Comma-separated patient IDs to embed (default: first --max-patients with complete sets).",
    )
    ap.add_argument(
        "--max-patients", type=int, default=30,
        help="Cap on number of patients to embed when --patient-ids is unset (default: 30).",
    )
    ap.add_argument(
        "--no-align", action="store_true",
        help="Do NOT rigidly align (translate + rotate, no scale) the raw cage "
             "onto the registered cage's frame.  By default the global rigid "
             "pose offset is removed via orthogonal Procrustes on 24 per-rib "
             "centroids; pass --no-align to see the raw vs registered shift.",
    )
    ap.add_argument(
        "--no-browser", action="store_true",
        help="Write the HTML without opening it in a browser.",
    )
    args = ap.parse_args()

    out = export_html(
        output_path=args.output,
        results_dir=args.results,
        patient_ids=args.patient_ids,
        max_patients=args.max_patients,
        target_dir=args.target_dir,
        registered_dir=args.registered_dir,
        landmark_dir=args.landmark_dir,
        template_pid=args.template_pid,
        align_rigid=not args.no_align,
    )
    if not args.no_browser:
        # as_uri requires an absolute path.
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
