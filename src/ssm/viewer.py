"""Self-contained interactive SSM viewer (HTML + embedded JS).

Builds ``viewer_surface.html`` with the PCA model and regression coefficients
embedded as JSON; reconstruction runs client-side. The page hosts both modes
via in-app tabs:

  - **PC explorer**       — slide leading PC scores.
  - **Metadata explorer** — slide cohort metadata (age / height / weight /
                            BMI / body fat / pack-years); regression
                            coefficients map slider deltas to PC-score updates.

The "Compare to real patient" picker renders a translucent ghost of any
cohort patient's stored PC-score reconstruction alongside the slider-driven
shape; "Load metadata into sliders" snaps the metadata sliders to that
patient's recorded values for side-by-side comparison.

Reconstruction (both modes)::

    shape_flat = mean_flat + Σ_k (pc_score_k × components_k)

Metadata explorer::

    Δpc_k = Σ_j β_kj × (x_j − x̄_j)

Two HTML files are written side-by-side:

  - ``viewer_surface_internal.html`` — full payload (patient IDs + per-patient
    metadata embedded as JSON); carries a red "INTERNAL ONLY" banner.
  - ``viewer_surface_public.html``   — no PIDs, no per-individual metadata
    combinations anywhere in the source (neither visually nor in the JSON /
    JS); safe to share alongside the manuscript.

CLI::

    python -m ssm.viewer --results PATH                       # write both + open internal
    python -m ssm.viewer --results PATH --no-browser          # write both only
    python -m ssm.viewer --results PATH --output-dir DIR      # custom output directory
    python -m ssm.viewer --results PATH --n-pcs N             # PC slider count
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from settings import (                       # noqa: E402
    FONT_FAMILY,
    N_PCS_DISPLAY,
    VIEWER_META_QUANTILES,
    VIEWER_PC_SLIDER_RANGE_SD,
)
from utils import colors as _colors           # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

_METADATA_PREDICTORS = [
    "age", "height_cm", "weight_kg", "bmi", "body_fat_pct", "pack_years",
]
_METADATA_LABELS = [
    "Age (years)", "Height (cm)", "Weight (kg)", "BMI (kg/m²)",
    "Body fat (%)", "Pack-years",
]
# Visual grouping in the Metadata Explorer panel. Each group becomes a
# `<fieldset>`; predictors absent from the regression are skipped.
_METADATA_GROUPS = [
    {"title": "Age",                    "predictors": ["age"]},
    {"title": "Height",                 "predictors": ["height_cm"]},
    {"title": "Weight, BMI & body fat", "predictors": ["weight_kg", "bmi", "body_fat_pct"]},
    {"title": "Pack-years",             "predictors": ["pack_years"]},
]


# ── Data loading ─────────────────────────────────────────────────────────────

def _require_results_dir(results_dir: Path | None) -> Path:
    if results_dir is None:
        raise ValueError(
            "viewer.load_model / viewer.export_html require an explicit "
            "results_dir (the ssm_pca/ subdir of a pipeline run)."
        )
    return Path(results_dir)


def load_model(
    results_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray | None]:
    """Load the PCA model, mean shape, face connectivity, and metadata.

    Returns
    -------
    components  : ``(n_comps, D)``    PCA eigenvectors.
    ev          : ``(n_comps,)``      explained variance.
    evr         : ``(n_comps,)``      explained variance ratio.
    mean_vec    : ``(D,)``            GPA mean shape flattened.
    mean_shape  : ``(n_pts, 3)``      mean shape as a point cloud.
    faces       : ``(n_faces, 3)``    triangle connectivity.
    scores      : DataFrame           PC scores + metadata.
    beta        : ``(n_comps, n_predictors)`` or ``None``  regression coefficients.
    """
    d = _require_results_dir(results_dir)
    pca_data   = np.load(d / "pca_surface.npz")
    components = pca_data["components"]
    ev         = pca_data["explained_variance"]
    evr        = pca_data["explained_variance_ratio"]
    mean_vec   = pca_data["mean"]
    mean_shape = np.load(d / "mean_shape_surface.npy")
    faces      = np.load(d / "template_faces.npy")
    scores     = pd.read_csv(d / "pc_scores_surface.csv")

    beta = _envelope_beta(scores, len(ev))
    return components, ev, evr, mean_vec, mean_shape, faces, scores, beta


def _envelope_beta(scores: pd.DataFrame, n_comps: int) -> np.ndarray | None:
    """Per-PC continuous-predictor slopes used only for the scene-range envelope.

    The Metadata Explorer's reconstruction runs off the geometry generator
    (``geometry_generator.npz``); these inline slopes only size the fixed plot
    axes and gate the metadata slider panel.
    """
    available = [p for p in _METADATA_PREDICTORS if p in scores.columns]
    if not available:
        return None

    pc_cols = [f"PC_{k+1}" for k in range(n_comps)]
    sub = scores.dropna(subset=available + pc_cols)
    if len(sub) < len(available) + 1:
        return None

    X = np.column_stack([np.ones(len(sub)), sub[available].to_numpy(np.float64)])
    B = np.zeros((n_comps, len(available)))
    for k in range(n_comps):
        y = sub[pc_cols[k]].to_numpy(np.float64)
        try:
            B[k, :] = np.linalg.lstsq(X, y, rcond=None)[0][1:]  # skip intercept
        except np.linalg.LinAlgError:
            pass
    return B


# ── Interactive viewer entry point ───────────────────────────────────────────

def run_viewer(
    mode: str = "pc",                     # noqa: ARG001
    results_dir: Path | None = None,
    n_pcs: int = N_PCS_DISPLAY,
    *,
    open_browser: bool = True,
) -> Path:
    """Build the internal + public HTML viewers; optionally open the internal.

    Both PC and Metadata explorers are reachable via in-app tabs; ``mode`` is
    accepted but ignored. The internal viewer is returned and opened (it is
    the richer one for interactive QA); the public viewer is written
    alongside it for downstream publication.

    Parameters
    ----------
    results_dir
        SSM PCA-outputs dir to read from.
    n_pcs
        Number of PC sliders shown in the PC explorer panel.
    open_browser
        If True (default), opens the internal HTML in the default browser.
    """
    internal_path, _ = export_both(results_dir=results_dir, n_pcs=n_pcs)
    if open_browser:
        webbrowser.open(internal_path.as_uri())
    return internal_path


# ── HTML export ──────────────────────────────────────────────────────────────

# Banner shown only on the internal viewer. The public viewer omits this
# element entirely (plain-text marker, robust against CSS hiding).
_INTERNAL_BANNER_HTML = (
    '<div id="internal-banner">INTERNAL ONLY &mdash; DO NOT PUBLISH</div>'
)

# Overlay-bar inner: internal has the patient-ID picker + load-metadata button;
# public has only the cohort sample (find-similar) controls.
_INTERNAL_OVERLAY_BAR_INNER = """\
    <span class="ob-label">Compare to real patient:</span>
    <input id="patient-select" type="text" list="patient-list"
           placeholder="Patient ID — type or pick"
           autocomplete="off" spellcheck="false"
           style="font-size:12px;padding:3px 6px;border-radius:4px;border:1px solid #ccc;font-family:inherit;color:#333;background:#fff;min-width:140px"
           oninput="onPatientInput(this.value)">
    <datalist id="patient-list"></datalist>
    <button id="load-into-sliders-btn" onclick="loadPatientIntoSliders()" disabled>
      Load metadata into sliders
    </button>
    <button id="find-similar-btn" onclick="findSimilarPatient()"
            title="Sample a cohort patient close to the current slider configuration (softmax-weighted by squared PC distance; re-click to re-roll).">
      Find similar patient
    </button>
    <span class="ghost-legend"><span class="ghost-swatch"></span>real patient</span>
    <span id="overlay-info"></span>\
"""

_PUBLIC_OVERLAY_BAR_INNER = """\
    <span class="ob-label">Synthetic sample:</span>
    <button id="find-similar-btn" onclick="findSimilarPatient()"
            title="Sample a synthetic, cohort-typical rib cage for the current metadata: the model's mean shape plus a draw from the shape variation demographics don't explain (re-click to re-roll).">
      Roll random sample
    </button>
    <span class="ghost-legend"><span class="ghost-swatch"></span>Synthetic sample</span>
    <span id="overlay-info"></span>\
"""

# Patient-related JS block. The internal version embeds patient IDs and per-
# patient metadata; the public version stubs out the picker/loader and
# rewrites findSimilarPatient to use only ``patient_pc_scores`` +
# ``patient_sex`` (no PIDs, no metadata combinations).
_INTERNAL_PATIENT_JS = """\
// ── Patient overlay (compare to real patient) ──────────────────────────────
// One <option> per patient id; the browser turns this into autocomplete.
function buildPatientPicker() {
  const dl = document.getElementById('patient-list');
  dl.innerHTML = M.patient_ids
    .map(pid => `<option value="${pid}"></option>`).join('');
}

// Mode-aware one-line summary: leading PC scores in PC mode, recorded
// metadata in metadata mode.
function updateOverlayInfo() {
  const el = document.getElementById('overlay-info');
  if (!el) return;
  if (selectedPatientIdx < 0) { el.textContent = ''; return; }
  if (curMode === 'pc') {
    const scores = M.patient_pc_scores[selectedPatientIdx] || [];
    const n = Math.min(5, scores.length);
    const parts = [];
    for (let k = 0; k < n; k++) {
      const v = scores[k];
      parts.push(`PC${k + 1} ${v >= 0 ? '+' : ''}${v.toFixed(1)}`);
    }
    el.textContent = parts.join(' \\u00b7 ');
    return;
  }
  const meta = M.patient_meta[selectedPatientIdx];
  if (!meta || !meta.length) { el.textContent = ''; return; }
  const parts = [];
  for (let j = 0; j < M.predictor_labels.length; j++) {
    const v = meta[j];
    if (v == null) continue;
    const short = M.predictor_labels[j].split(' ')[0].replace(/[:,]/g, '');
    parts.push(`${short} ${(+v).toFixed(1)}`);
  }
  el.textContent = parts.join(' \\u00b7 ');
}

// Toggle the load-button label + tooltip to match the active mode.
function updateLoadButton() {
  const btn = document.getElementById('load-into-sliders-btn');
  if (!btn) return;
  if (curMode === 'pc') {
    btn.textContent = 'Load PC scores into sliders';
    btn.title = "Snap the PC sliders to this patient's stored PC scores.";
  } else {
    btn.textContent = 'Load metadata into sliders';
    btn.title = "Snap the metadata sliders to this patient's recorded values, "
              + "so the cage shows the model's metadata-driven prediction for them.";
  }
}

// Resolve the typed input to a patient index; empty/no-match clears the
// overlay, exact match renders the ghost mesh.
function onPatientInput(value) {
  const trimmed = (value || '').trim();
  let idx = -1;
  if (trimmed !== '') {
    const pid = +trimmed;
    if (Number.isFinite(pid)) idx = M.patient_ids.indexOf(pid);
  }
  selectedPatientIdx = idx;
  document.getElementById('load-into-sliders-btn').disabled = (idx < 0);
  if (idx < 0) {
    Plotly.restyle('plot', {visible: false}, [1]);
    updateOverlayInfo();
    return;
  }
  const flat = reconstructFromScores(M.patient_pc_scores[idx]);
  const {x, y, z} = flatToXYZ(flat);
  Plotly.restyle('plot', {x: [x], y: [y], z: [z], visible: true}, [1]);
  updateOverlayInfo();
}

// ── Sex toggle (Metadata Explorer) ─────────────────────────────────────────
// Disable radios with no offset (n < 2 or column missing); hide the
// fieldset if neither sex has one.
function buildSexToggle() {
  const fs = document.getElementById('sex-toggle-group');
  if (!fs) return;
  const haveMale   = !!M.mean_pc_male;
  const haveFemale = !!M.mean_pc_female;
  if (!haveMale && !haveFemale) { fs.style.display = 'none'; return; }
  const mr = document.getElementById('sex-toggle-male');
  const fr = document.getElementById('sex-toggle-female');
  if (mr) {
    mr.disabled = !haveMale;
    if (!haveMale) mr.title = `Male offset unavailable (n = ${M.n_male})`;
  }
  if (fr) {
    fr.disabled = !haveFemale;
    if (!haveFemale) fr.title = `Female offset unavailable (n = ${M.n_female})`;
  }
}

function onSexToggle(v) {
  sexMode = v;
  if (curMode === 'metadata') {
    computeMetaPcScores();
    updatePlot();
  }
  refreshFindSimilarBtn();
}

// ── Reverse search: sample a cohort patient close to current sliders ───────
// Softmax weight exp(-d² / τ) with τ = max(median(d²), 1e-6) over the
// sex-restricted pool. Routes through onPatientInput to update the overlay.
function findSimilarPatient() {
  if (!M.patient_pc_scores || !M.patient_pc_scores.length) return;
  const pool = [];
  for (let i = 0; i < M.patient_pc_scores.length; i++) {
    if (sexMode === 'Male'   && M.patient_sex[i] !== 'Male')   continue;
    if (sexMode === 'Female' && M.patient_sex[i] !== 'Female') continue;
    if (smokeMode === 'Never' && M.patient_ever[i] !== 0) continue;
    if (smokeMode === 'Ever'  && M.patient_ever[i] !== 1) continue;
    pool.push(i);
  }
  if (!pool.length) return;
  const d2 = new Array(pool.length);
  for (let p = 0; p < pool.length; p++) {
    const ps = M.patient_pc_scores[pool[p]];
    let s = 0;
    for (let k = 0; k < N; k++) {
      const dx = pcScores[k] - ps[k];
      s += dx * dx;
    }
    d2[p] = s;
  }
  const sorted = [...d2].sort((a, b) => a - b);
  const tau = Math.max(sorted[Math.floor(sorted.length / 2)], 1e-6);
  // Subtract min(d²) before exp() to avoid underflow; the constant factor
  // cancels in the normalisation.
  const dmin = sorted[0];
  const w = d2.map(d => Math.exp(-(d - dmin) / tau));
  const total = w.reduce((a, b) => a + b, 0);
  if (!(total > 0)) return;
  let r = Math.random() * total;
  let pick = pool[0];
  for (let p = 0; p < pool.length; p++) {
    r -= w[p];
    if (r <= 0) { pick = pool[p]; break; }
  }
  const id = M.patient_ids[pick];
  const inp = document.getElementById('patient-select');
  if (inp) inp.value = id;
  onPatientInput(String(id));
}

function refreshFindSimilarBtn() {
  const btn = document.getElementById('find-similar-btn');
  if (!btn) return;
  if (!M.patient_pc_scores || !M.patient_pc_scores.length) {
    btn.style.display = 'none';
    return;
  }
  let any = false;
  for (let i = 0; i < M.patient_sex.length; i++) {
    if (sexMode === 'Male'   && M.patient_sex[i] !== 'Male')   continue;
    if (sexMode === 'Female' && M.patient_sex[i] !== 'Female') continue;
    if (smokeMode === 'Never' && M.patient_ever[i] !== 0) continue;
    if (smokeMode === 'Ever'  && M.patient_ever[i] !== 1) continue;
    any = true; break;
  }
  btn.disabled = !any;
  btn.title = any
    ? "Sample a cohort patient close to the current slider configuration "
    + "(softmax-weighted by squared PC distance; re-click to re-roll)."
    : `No ${sexMode} patients in cohort.`;
}

// Snap the active panel's sliders to the selected patient. PC mode copies
// stored PC scores; metadata mode copies recorded metadata (NaN fields
// skipped, BMI re-derived) and re-runs the regression.
function loadPatientIntoSliders() {
  if (selectedPatientIdx < 0) return;
  if (curMode === 'pc') {
    const scores = M.patient_pc_scores[selectedPatientIdx];
    if (!scores) return;
    for (let k = 0; k < N; k++) {
      const v = scores[k] || 0;
      pcScores[k] = v;
      const slider = document.getElementById(`pcs${k}`);
      const valEl  = document.getElementById(`pcv${k}`);
      if (slider) slider.value = v;
      if (valEl)  valEl.textContent = (v >= 0 ? '+' : '') + v.toFixed(1);
    }
    updatePlot();
    return;
  }
  const meta = M.patient_meta[selectedPatientIdx];
  if (!meta) return;
  for (let j = 0; j < meta.length; j++) {
    const v = meta[j];
    if (v == null) continue;
    metaVals[j] = +v;
    const slider = document.getElementById(`mts${j}`);
    const valEl  = document.getElementById(`mtv${j}`);
    if (slider) slider.value = v;
    if (valEl)  valEl.textContent = (+v).toFixed(1);
  }
  // Snap the sex toggle to this patient's recorded sex.
  const psex = M.patient_sex ? M.patient_sex[selectedPatientIdx] : null;
  if (psex === 'Male' || psex === 'Female') {
    const r = document.querySelector(
      `input[name=sex-toggle][value=${psex}]`
    );
    if (r && !r.disabled) {
      r.checked = true;
      sexMode = psex;
      refreshFindSimilarBtn();
    }
  }
  // Snap the smoking toggle to this patient's ever-smoker status.
  const pever = M.patient_ever ? M.patient_ever[selectedPatientIdx] : null;
  if (pever === 0 || pever === 1) {
    const sv = pever === 1 ? 'Ever' : 'Never';
    const sr = document.querySelector(`input[name=smoke-toggle][value=${sv}]`);
    if (sr) { sr.checked = true; smokeMode = sv; }
  }
  updateDerivedBmi();
  syncPackSlider();
  computeMetaPcScores();
  // Mirror the regression-derived PC scores into the PC sliders.
  for (let k = 0; k < N; k++) {
    const slider = document.getElementById(`pcs${k}`);
    const valEl  = document.getElementById(`pcv${k}`);
    if (slider) slider.value = pcScores[k];
    if (valEl)  valEl.textContent =
      (pcScores[k] >= 0 ? '+' : '') + pcScores[k].toFixed(1);
  }
  updateOodBanner();
  updatePlot();
}\
"""

# Public variant: no PIDs anywhere in the source. The functions that referenced
# M.patient_ids / M.patient_meta are either stubbed (when their UI is gone) or
# rewritten to operate on M.patient_pc_scores + M.patient_sex by index only.
_PUBLIC_PATIENT_JS = """\
// ── Cohort sample overlay (public viewer: no PIDs, no metadata combos) ─────
// Stub: no patient-ID autocomplete in the public viewer.
function buildPatientPicker() { /* public mode: no picker element */ }

// Last synthetic ghost sample's PC scores (null = no ghost shown).
let ghostScores = null;
// Standard-normal draw (Box–Muller) for the residual shape sampling.
function gaussRand() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// PC-mode summary only — the public viewer never surfaces per-patient
// metadata combinations.
function updateOverlayInfo() {
  const el = document.getElementById('overlay-info');
  if (!el) return;
  if (!ghostScores) { el.textContent = ''; return; }
  if (curMode === 'pc') {
    const n = Math.min(5, ghostScores.length);
    const parts = [];
    for (let k = 0; k < n; k++) {
      const v = ghostScores[k];
      parts.push(`PC${k + 1} ${v >= 0 ? '+' : ''}${v.toFixed(1)}`);
    }
    el.textContent = parts.join(' \\u00b7 ');
    return;
  }
  el.textContent = '';
}

// No load-into-sliders button in the public viewer; this function is dead
// code but the guard keeps it safe.
function updateLoadButton() {
  const btn = document.getElementById('load-into-sliders-btn');
  if (!btn) return;
  if (curMode === 'pc') {
    btn.textContent = 'Load PC scores into sliders';
    btn.title = "Snap the PC sliders to this patient's stored PC scores.";
  }
}

// Stub: no patient-ID input in the public viewer.
function onPatientInput() { /* public mode: no input element */ }

// ── Sex toggle (Metadata Explorer) ─────────────────────────────────────────
function buildSexToggle() {
  const fs = document.getElementById('sex-toggle-group');
  if (!fs) return;
  const haveMale   = !!M.mean_pc_male;
  const haveFemale = !!M.mean_pc_female;
  if (!haveMale && !haveFemale) { fs.style.display = 'none'; return; }
  const mr = document.getElementById('sex-toggle-male');
  const fr = document.getElementById('sex-toggle-female');
  if (mr) {
    mr.disabled = !haveMale;
    if (!haveMale) mr.title = `Male offset unavailable (n = ${M.n_male})`;
  }
  if (fr) {
    fr.disabled = !haveFemale;
    if (!haveFemale) fr.title = `Female offset unavailable (n = ${M.n_female})`;
  }
}

function onSexToggle(v) {
  sexMode = v;
  if (curMode === 'metadata') {
    computeMetaPcScores();
    updatePlot();
  }
  refreshFindSimilarBtn();
}

// ── Synthetic cohort-typical sample ────────────────────────────────────────
// Draws a shape from p(shape | metadata): the generator's mean PC scores for
// the current sliders/toggles plus a residual draw from the unexplained shape
// variation (resid_sd_k = sqrt((1 - R²_k)·λ_k)). No real-patient data involved.
function findSimilarPatient() {
  if (!M.resid_sd) return;
  ghostScores = new Float64Array(N);
  for (let k = 0; k < N; k++) ghostScores[k] = pcScores[k] + M.resid_sd[k] * gaussRand();
  const flat = reconstructFromScores(ghostScores);
  const {x, y, z} = flatToXYZ(flat);
  Plotly.restyle('plot', {x: [x], y: [y], z: [z], visible: true}, [1]);
  updateOverlayInfo();
}

function refreshFindSimilarBtn() {
  const btn = document.getElementById('find-similar-btn');
  if (!btn) return;
  const ok = !!M.resid_sd;
  btn.style.display = ok ? '' : 'none';
  btn.disabled = !ok;
  btn.title = "Sample a synthetic, cohort-typical rib cage for the current "
            + "metadata: the model's mean shape plus a draw from the shape "
            + "variation demographics don't explain (re-click to re-roll).";
}

// Stub: no load-into-sliders button in the public viewer.
function loadPatientIntoSliders() { /* public mode: no button */ }\
"""


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rib Cage Surface SSM — Interactive Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: __FONT_FAMILY__;
       background: #fafafa; color: #222222; }
#header { padding: 14px 20px 8px; background: #ffffff;
          border-bottom: 1px solid #e0e0e0; }
#header h2 { font-size: 17px; font-weight: 600; color: #111111; }
#header p  { font-size: 12px; color: #666; margin-top: 2px; }
#tabs { display: flex; gap: 6px; padding: 10px 20px 0; background: #ffffff; }
.tab-btn { padding: 6px 16px; border: 1px solid #ccc; border-radius: 6px 6px 0 0;
           background: #f0f0f0; cursor: pointer; font-size: 13px; font-weight: 500;
           font-family: inherit; color: #444; }
.tab-btn.active { background: #ffffff; border-bottom-color: #ffffff;
                  color: __ACCENT_COLOR__; border-color: __ACCENT_COLOR__; }
#plot-wrap { height: 58vh; background: #ffffff; border-bottom: 1px solid #e0e0e0; }
#plot      { width: 100%; height: 100%; }
#controls  { background: #ffffff; padding: 12px 20px 16px; }
.slider-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 30px; }
.slider-stack { display: flex; flex-direction: column; gap: 4px; }
.slider-row  { display: flex; align-items: center; gap: 8px; padding: 2px 0; }
.s-label { width: 160px; font-size: 12px; color: #444; white-space: nowrap;
           overflow: hidden; text-overflow: ellipsis; flex-shrink: 0; }
.s-pct   { font-size: 10px; color: #888; }
.s-derived { font-size: 10px; color: #888; font-style: italic; margin-left: 4px; }
.s-input { flex: 1; accent-color: __ACCENT_COLOR__; cursor: pointer; }
.s-input:disabled { cursor: not-allowed; opacity: 0.65; }
.s-val   { width: 52px; text-align: right; font-size: 12px;
           font-variant-numeric: tabular-nums; color: #333; }
.s-ref   { font-size: 10px; color: #aaa; width: 60px; }
.slider-group {
  border: 1px solid #d8d8d8; border-radius: 6px;
  padding: 2px 12px 6px; margin: 6px 0 0;
}
.slider-group legend {
  font-size: 11px; font-weight: 600; color: #444;
  padding: 0 6px; text-transform: uppercase; letter-spacing: 0.04em;
}
#bottom-bar { display: flex; justify-content: space-between; align-items: center;
              padding: 8px 20px 0; }
#status-txt { font-size: 12px; color: #555; }
#reset-btn  { padding: 5px 14px; font-size: 12px; border-radius: 6px;
              border: 1px solid #ccc; background: #f0f0f0; cursor: pointer;
              font-family: inherit; color: #333; }
#reset-btn:hover { background: #e0e0e0; }
#warn { font-size: 11px; color: #c55; padding: 6px 20px; background: #fff8f0;
        border-bottom: 1px solid #e0d0c0; display: none; }
#internal-banner { background: #c0392b; color: #ffffff; text-align: center;
                   font-size: 15px; font-weight: 700; letter-spacing: 0.12em;
                   text-transform: uppercase; padding: 10px 20px;
                   border-bottom: 2px solid #8b271c; }
#overlay-bar { display: flex; align-items: center; gap: 10px;
               padding: 4px 0 10px; border-bottom: 1px solid #eee;
               margin-bottom: 8px; flex-wrap: wrap; }
.ob-label { font-size: 12px; color: #444; font-weight: 500; }
#load-into-sliders-btn,
#find-similar-btn      { padding: 4px 10px; font-size: 12px; border-radius: 6px;
                         border: 1px solid #ccc; background: #f0f0f0; cursor: pointer;
                         font-family: inherit; color: #333; }
#load-into-sliders-btn:disabled,
#find-similar-btn:disabled { opacity: 0.5; cursor: not-allowed; }
#load-into-sliders-btn:not(:disabled):hover,
#find-similar-btn:not(:disabled):hover { background: #e0e0e0; }
#overlay-info { font-size: 11px; color: #777; flex: 1; min-width: 0;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ghost-legend { display: inline-flex; align-items: center; gap: 4px;
                font-size: 11px; color: #666; }
.ghost-swatch { display: inline-block; width: 18px; height: 10px;
                background: __GHOST_COLOR__; opacity: 0.45;
                border: 1px solid #999; border-radius: 2px; }
#ood-banner { font-size: 11px; padding: 6px 10px; border-radius: 5px;
              margin: 4px 0 8px; line-height: 1.35; }
#ood-banner.ood-warn   { background: #fdf6e3; border: 1px solid #d8c98c;
                         color: #6b5a16; }
#ood-banner.ood-severe { background: #fff0ee; border: 1px solid #d8a09a;
                         color: #8a2a1f; }
</style>
</head>
<body>

__INTERNAL_BANNER__

<div id="header">
  <h2>Rib Cage Surface SSM &mdash; Interactive Viewer</h2>
  <p>NAKO cohort &nbsp;&bull;&nbsp; Surface-based shape model &nbsp;&bull;&nbsp;
     GPA + PCA &nbsp;&bull;&nbsp; N&thinsp;=&thinsp;__N_PATIENTS__ patients &nbsp;&bull;&nbsp;
     __N_COMPS__ PCs &rarr; __CUM_VAR__% cumulative variance</p>
</div>

<div id="warn">__WARN_MSG__</div>

<div id="tabs">
  <button class="tab-btn active" onclick="setMode('pc')">PC Explorer</button>
  <button class="tab-btn"        onclick="setMode('metadata')">Metadata Explorer</button>
</div>

<div id="plot-wrap"><div id="plot"></div></div>

<div id="controls">
  <div id="overlay-bar">
__OVERLAY_BAR_INNER__
  </div>
  <div id="pc-panel">
    <div class="slider-grid" id="pc-sliders"></div>
  </div>
  <div id="meta-panel" style="display:none">
    <fieldset class="slider-group" id="sex-toggle-group">
      <legend>Sex</legend>
      <div class="slider-row" style="gap:14px">
        <label><input type="radio" name="sex-toggle" value="cohort" checked
               onchange="onSexToggle('cohort')"> Any (population)</label>
        <label><input type="radio" name="sex-toggle" value="Male"
               onchange="onSexToggle('Male')" id="sex-toggle-male"> Male</label>
        <label><input type="radio" name="sex-toggle" value="Female"
               onchange="onSexToggle('Female')" id="sex-toggle-female"> Female</label>
      </div>
    </fieldset>
    <fieldset class="slider-group" id="smoke-toggle-group">
      <legend>Smoking</legend>
      <div class="slider-row" style="gap:14px">
        <label><input type="radio" name="smoke-toggle" value="any" checked
               onchange="onSmokeToggle('any')"> Any (population)</label>
        <label><input type="radio" name="smoke-toggle" value="Never"
               onchange="onSmokeToggle('Never')"> Never</label>
        <label><input type="radio" name="smoke-toggle" value="Ever"
               onchange="onSmokeToggle('Ever')"> Ever</label>
      </div>
    </fieldset>
    <div id="ood-banner" style="display:none"></div>
    <div class="slider-stack" id="meta-sliders"></div>
  </div>
  <div id="bottom-bar">
    <span id="status-txt">Mean shape &mdash; drag a slider to explore</span>
    <button id="reset-btn" onclick="resetAll()">Reset to mean</button>
  </div>
</div>

<script>
// ── Embedded model data ────────────────────────────────────────────────────
const M = __MODEL_JSON__;

// Unpack the base64-packed float32 PCA basis into per-component Float32Array
// views (M.components[k] → length n_pts*3), so reconstruction code is unchanged.
(function () {
  const bin = atob(M.components_b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  const flat = new Float32Array(buf.buffer);
  const D = M.n_pts * 3;
  M.components = [];
  for (let k = 0; k < M.n_comps; k++) M.components.push(flat.subarray(k * D, (k + 1) * D));
  delete M.components_b64;
})();

// ── State ──────────────────────────────────────────────────────────────────
const N  = M.n_comps;
const NV = M.n_pts * 3;     // flattened dimension
let pcScores = new Float64Array(N);
// Slider reset values: cohort means with pack-years pinned to 0.
let metaVals = M.meta_init.slice();
let curMode  = 'pc';
// Index of the ghost-overlay patient in the cohort arrays; -1 = no overlay.
// Trace 0 = predicted (slider-driven); trace 1 = ghost (cohort sample).
let selectedPatientIdx = -1;
// Metadata-Explorer toggles feeding the geometry generator. Sex: 'cohort'
// (= Any → cohort female fraction), 'Male' (is_female 0), 'Female' (is_female 1).
// Smoking: 'any' (cohort ever fraction), 'Never' (ever 0, pack 0), 'Ever' (ever 1).
let sexMode = 'cohort';
let smokeMode = 'any';

// ── Shape reconstruction ────────────────────────────────────────────────────
function reconstruct() { return reconstructFromScores(pcScores); }

function reconstructFromScores(scoreVec) {
  const flat = M.mean_vec.slice();
  for (let k = 0; k < N; k++) {
    const s = scoreVec[k];
    if (!s || Math.abs(s) < 1e-12) continue;
    const c = M.components[k];
    for (let i = 0; i < NV; i++) flat[i] += s * c[i];
  }
  return flat;
}

function flatToXYZ(flat) {
  const n = M.n_pts;
  const x = new Array(n), y = new Array(n), z = new Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = flat[i*3]; y[i] = flat[i*3+1]; z[i] = flat[i*3+2];
  }
  return {x, y, z};
}

function computeDisplacement(flat) {
  const n = M.n_pts;
  const d = new Array(n);
  for (let i = 0; i < n; i++) {
    const dx = flat[i*3]   - M.mean_vec[i*3];
    const dy = flat[i*3+1] - M.mean_vec[i*3+1];
    const dz = flat[i*3+2] - M.mean_vec[i*3+2];
    d[i] = Math.sqrt(dx*dx + dy*dy + dz*dz);
  }
  return d;
}

// ── Plotly setup ──────────────────────────────────────────────────────────
// Ghost overlay trace; initialised with the mean shape and hidden until a
// patient is picked. Single colour, translucent, no colorbar.
function buildGhostTrace() {
  const {x, y, z} = flatToXYZ(M.mean_vec);
  return {
    type: 'mesh3d',
    x, y, z,
    i: M.faces_i, j: M.faces_j, k: M.faces_k,
    color: M.ghost_color,
    opacity: 0.35,
    flatshading: true,
    lighting: {ambient: 0.7, diffuse: 0.5, specular: 0.1, roughness: 0.9},
    hoverinfo: 'skip',
    showscale: false,
    visible: false,
    name: 'real patient',
  };
}

function buildTrace() {
  const flat = reconstruct();
  const {x, y, z} = flatToXYZ(flat);
  return {
    type: 'mesh3d',
    x, y, z,
    i: M.faces_i, j: M.faces_j, k: M.faces_k,
    intensity: computeDisplacement(flat),
    colorscale: M.displacement_colorscale,
    cmin: 0, cmax: M.clim_max,
    colorbar: {title: {text: 'Displacement (mm)',
                       font: {family: M.font_family, size: 11}},
               tickfont: {family: M.font_family, size: 10},
               thickness: 12, len: 0.6},
    flatshading: false,
    lighting: {ambient: 0.6, diffuse: 0.7, specular: 0.3, roughness: 0.5},
    lightposition: {x: 1000, y: 1000, z: 1000},
    hoverinfo: 'skip',
    showscale: true,
  };
}

const LAYOUT = {
  scene: {
    // Fixed ranges disable per-restyle auto-fit; PC1 (a global-size mode)
    // would otherwise be invisible.
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
};
const CONFIG = {responsive: true, displayModeBar: true, displaylogo: false,
                modeBarButtonsToRemove: ['resetCameraLastSave3d']};

function updatePlot() {
  const flat = reconstruct();
  const {x, y, z} = flatToXYZ(flat);
  const intensity = computeDisplacement(flat);
  Plotly.restyle('plot', {x: [x], y: [y], z: [z], intensity: [intensity]}, [0]);
  updateStatus();
}

function updateStatus() {
  const active = Array.from(pcScores).filter(v => Math.abs(v) > 1e-9).length;
  let txt = active === 0
    ? 'Mean shape \\u2014 drag a slider to explore'
    : `${active} PC${active > 1 ? 's' : ''} active`;
  if (curMode === 'metadata' && sexMode !== 'cohort') {
    txt += ` \\u00b7 ${sexMode} shift`;
  }
  document.getElementById('status-txt').textContent = txt;
}

// ── PC sliders ─────────────────────────────────────────────────────────────
function buildPcSliders() {
  const el = document.getElementById('pc-sliders');
  el.innerHTML = M.ev_show.map((ev, k) => {
    const sd   = Math.sqrt(ev);
    const pct  = (M.evr_show[k] * 100).toFixed(1);
    const step = (sd / 60).toFixed(4);
    const max  = (__SLIDER_RANGE_SD__ * sd).toFixed(3);
    return `<div class="slider-row">
      <span class="s-label">PC${k+1} <span class="s-pct">${pct}%</span></span>
      <input class="s-input" type="range" id="pcs${k}"
             min="-${max}" max="${max}" value="0" step="${step}"
             oninput="onPc(${k},+this.value)">
      <span class="s-val" id="pcv${k}">0.0</span>
    </div>`;
  }).join('');
}

function onPc(k, v) {
  pcScores[k] = v;
  document.getElementById(`pcv${k}`).textContent =
    (v >= 0 ? '+' : '') + v.toFixed(1);
  updatePlot();
}

// ── Metadata sliders ───────────────────────────────────────────────────────
function buildMetaSliders() {
  const el = document.getElementById('meta-sliders');
  if (!M.beta) {
    el.innerHTML = '<p style="color:#888;font-size:13px">Regression coefficients not available.</p>';
    return;
  }
  el.innerHTML = M.meta_groups.map(group => {
    const rows = group.indices.map(j => {
      const lbl  = M.predictor_labels[j];
      const [lo, hi] = M.meta_quantiles[j];
      const init = M.meta_init[j];   // slider opens here (cohort mean, except pack-years = 0)
      const ref  = M.meta_means[j];  // regression reference (cohort mean)
      const step = ((hi - lo) / 100).toFixed(3);
      const isBmi = (j === M.bmi_idx);
      const inputAttrs = isBmi
        ? `disabled title="Derived from height and weight: BMI = weight / (height/100)²"`
        : `oninput="onMeta(${j},+this.value)"`;
      const labelExtra = isBmi ? ' <span class="s-derived">(derived)</span>' : '';
      return `<div class="slider-row">
        <span class="s-label">${lbl}${labelExtra}</span>
        <input class="s-input" type="range" id="mts${j}"
               min="${lo.toFixed(2)}" max="${hi.toFixed(2)}"
               value="${init.toFixed(2)}" step="${step}"
               ${inputAttrs}>
        <span class="s-val" id="mtv${j}">${init.toFixed(1)}</span>
        <span class="s-ref">ref\\u2009${ref.toFixed(1)}</span>
      </div>`;
    }).join('');
    return `<fieldset class="slider-group">
      <legend>${group.title}</legend>
      ${rows}
    </fieldset>`;
  }).join('');
}

// Recompute BMI = weight / (height/100)² from the current height + weight
// sliders. Keeps the three sliders self-consistent.
function updateDerivedBmi() {
  if (M.bmi_idx < 0 || M.height_idx < 0 || M.weight_idx < 0) return;
  const h = +metaVals[M.height_idx];
  const w = +metaVals[M.weight_idx];
  if (!Number.isFinite(h) || !Number.isFinite(w) || h <= 0) return;
  const bmi = w / Math.pow(h / 100, 2);
  metaVals[M.bmi_idx] = bmi;
  const slider = document.getElementById(`mts${M.bmi_idx}`);
  const valEl  = document.getElementById(`mtv${M.bmi_idx}`);
  // The slider input clamps `.value` to [min, max]; the textual display
  // shows the actual derived BMI even if it overshoots the slider range.
  if (slider) slider.value = bmi;
  if (valEl)  valEl.textContent = bmi.toFixed(1);
}

// Value of a generator predictor from the current UI state (sliders + toggles).
function genPredictorValue(p, pIdx) {
  if (p === 'is_female')
    return sexMode === 'Male' ? 0 : sexMode === 'Female' ? 1 : M.is_female_any;
  if (p === 'ever_smoker')
    return smokeMode === 'Never' ? 0 : smokeMode === 'Ever' ? 1 : M.ever_smoker_any;
  if (p === 'pack_years')
    return smokeMode === 'Never' ? 0
         : smokeMode === 'Ever'  ? metaVals[M.gen_slider_idx[pIdx]]
         : M.pack_years_any;
  const si = M.gen_slider_idx[pIdx];
  return si >= 0 ? metaVals[si] : M.gen_offsets[pIdx];
}

// Predicted PC scores from the geometry generator:
//   ŷ_k = intercept_k + Σ_j B_kj (value_j − offset_j)
// over the CORE predictors (sex/smoking toggles + metadata sliders).
function computeMetaPcScores() {
  if (!M.gen_B) return;
  for (let k = 0; k < N; k++) {
    let s = M.gen_intercept[k];
    for (let p = 0; p < M.gen_predictors.length; p++) {
      const val = genPredictorValue(M.gen_predictors[p], p);
      s += M.gen_B[k][p] * (val - M.gen_offsets[p]);
    }
    pcScores[k] = s;
  }
}

function onMeta(j, v) {
  metaVals[j] = v;
  document.getElementById(`mtv${j}`).textContent = v.toFixed(1);
  if (j === M.height_idx || j === M.weight_idx) {
    updateDerivedBmi();
  }
  computeMetaPcScores();
  updateOodBanner();
  updatePlot();
}

// ── Smoking toggle (Metadata Explorer) ─────────────────────────────────────
// Two-part ever+pack model: Never → pack-years 0 (slider disabled); Ever →
// pack-years slider active; Any → cohort-mean pack-years (slider disabled).
function buildSmokeToggle() {
  const fs = document.getElementById('smoke-toggle-group');
  if (!fs) return;
  if (!M.gen_B || M.gen_predictors.indexOf('ever_smoker') < 0) {
    fs.style.display = 'none';
    return;
  }
  syncPackSlider();
}

// Keep the pack-years slider's value + disabled state in sync with smokeMode.
function syncPackSlider() {
  const j = M.pack_years_idx;
  if (j == null || j < 0) return;
  let disabled = true;
  let val = metaVals[j];
  if (smokeMode === 'Never')     { val = 0;                disabled = true;  }
  else if (smokeMode === 'Ever') {                         disabled = false; }
  else                           { val = M.pack_years_any; disabled = true;  }  // any
  metaVals[j] = val;
  const slider = document.getElementById(`mts${j}`);
  const valEl  = document.getElementById(`mtv${j}`);
  if (slider) { slider.disabled = disabled; slider.value = val; }
  if (valEl)  valEl.textContent = (+val).toFixed(1);
}

function onSmokeToggle(v) {
  smokeMode = v;
  syncPackSlider();
  if (curMode === 'metadata') {
    computeMetaPcScores();
    updateOodBanner();
    updatePlot();
  }
  if (typeof refreshFindSimilarBtn === 'function') refreshFindSimilarBtn();
}

// ── Out-of-distribution warning ────────────────────────────────────────────
// Mahalanobis d² of the slider state vs the cohort. Thresholds are the 95th /
// 99th percentile of the cohort's own d². Returns 0 when no payload.
function mahalanobisD2() {
  if (!M.ood_indices) return 0;
  const idx = M.ood_indices, mu = M.ood_mean, S = M.ood_inv_cov;
  const n = idx.length;
  const d = new Array(n);
  for (let i = 0; i < n; i++) d[i] = metaVals[idx[i]] - mu[i];
  let s = 0;
  for (let i = 0; i < n; i++) {
    const Si = S[i];
    for (let j = 0; j < n; j++) s += d[i] * Si[j] * d[j];
  }
  return s;
}

function updateOodBanner() {
  const el = document.getElementById('ood-banner');
  if (!el) return;
  if (curMode !== 'metadata' || !M.ood_indices) {
    el.style.display = 'none';
    return;
  }
  const d2 = mahalanobisD2();
  if (d2 < M.ood_warn_d2) {
    el.style.display = 'none';
    return;
  }
  const severe = d2 >= M.ood_severe_d2;
  el.className = severe ? 'ood-severe' : 'ood-warn';
  el.textContent = severe
    ? `Extrapolating beyond the study population — this combination is `
      + `outside ~99% of the cohort (Mahalanobis d² = ${d2.toFixed(1)}, `
      + `99th-percentile cohort d² = ${M.ood_severe_d2.toFixed(1)}).`
    : `Uncommon combination — rarer than ~95% of the cohort `
      + `(Mahalanobis d² = ${d2.toFixed(1)}, 95th-percentile cohort `
      + `d² = ${M.ood_warn_d2.toFixed(1)}).`;
  el.style.display = '';
}

__PATIENT_JS__

// ── Mode switching & reset ─────────────────────────────────────────────────
function setMode(m) {
  curMode = m;
  resetAll();
  document.getElementById('pc-panel').style.display   = m === 'pc'       ? '' : 'none';
  document.getElementById('meta-panel').style.display = m === 'metadata' ? '' : 'none';
  document.querySelectorAll('.tab-btn').forEach((b, i) =>
    b.classList.toggle('active', (i === 0) === (m === 'pc')));
  updateLoadButton();
  updateOverlayInfo();
  updateOodBanner();
}

function resetAll() {
  metaVals = M.meta_init.slice();
  sexMode = 'cohort';
  smokeMode = 'any';
  const cohortRadio = document.querySelector(
    'input[name=sex-toggle][value=cohort]'
  );
  if (cohortRadio) cohortRadio.checked = true;
  const smokeAnyRadio = document.querySelector(
    'input[name=smoke-toggle][value=any]'
  );
  if (smokeAnyRadio) smokeAnyRadio.checked = true;
  refreshFindSimilarBtn();
  document.querySelectorAll('[id^=pcs]').forEach((s, k) => {
    s.value = 0;
    const el = document.getElementById(`pcv${k}`);
    if (el) el.textContent = '0.0';
  });
  document.querySelectorAll('[id^=mts]').forEach((s, j) => {
    s.value = M.meta_init[j];
    const el = document.getElementById(`mtv${j}`);
    if (el) el.textContent = M.meta_init[j].toFixed(1);
  });
  updateDerivedBmi();
  syncPackSlider();
  // In metadata mode recompute pcScores from metaVals; in PC mode they are zero.
  if (curMode === 'metadata') {
    computeMetaPcScores();
  } else {
    pcScores.fill(0);
  }
  updatePlot();
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Trace 0 = slider-driven prediction; trace 1 = ghost overlay (hidden).
  Plotly.newPlot('plot', [buildTrace(), buildGhostTrace()], LAYOUT, CONFIG);
  buildPcSliders();
  buildMetaSliders();
  buildSexToggle();
  buildSmokeToggle();
  buildPatientPicker();
  updateDerivedBmi();
  updateStatus();
  updateLoadButton();
  refreshFindSimilarBtn();
  updateOodBanner();
  if (M.warn_msg) document.getElementById('warn').style.display = '';
});
</script>
</body>
</html>
"""


def export_html(
    output_path: str | Path | None = None,
    results_dir: Path | None = None,
    n_pcs: int = N_PCS_DISPLAY,
    *,
    public_mode: bool = False,
) -> Path:
    """Export a self-contained interactive HTML viewer.

    The HTML embeds the PCA model + face connectivity as JSON and reconstructs
    shapes client-side in JavaScript.

    ``public_mode=False`` (default) writes the internal viewer with a red
    "INTERNAL ONLY" banner, the patient-ID autocomplete picker, and the full
    per-individual metadata payload. ``public_mode=True`` writes the public
    viewer: no PIDs and no per-individual metadata combinations are embedded
    anywhere in the HTML (neither visually nor in the JSON / JS source), and
    the patient-picker / load-metadata UI is removed entirely. The "Find
    similar patient" feature still works in public mode (it operates on
    ``patient_pc_scores`` and ``patient_sex`` by index alone — both allowed
    under the single-metadata × derivative rule).
    """
    components, ev, evr, mean_vec, mean_shape, faces, scores, beta = \
        load_model(results_dir)
    n_comps = len(ev)
    n_show  = min(n_pcs, n_comps)

    available = [p for p in _METADATA_PREDICTORS if p in scores.columns]
    meta_means = scores[available].mean()
    q_lo, q_hi = VIEWER_META_QUANTILES
    meta_q = [
        [float(scores[p].quantile(q_lo)), float(scores[p].quantile(q_hi))]
        for p in available
    ]
    label_map = dict(zip(_METADATA_PREDICTORS, _METADATA_LABELS))

    # JS recomputes BMI = weight / (height/100)² whenever height or weight
    # changes, so the three sliders stay self-consistent.
    def _idx(name: str) -> int:
        return available.index(name) if name in available else -1
    bmi_idx        = _idx("bmi")
    height_idx     = _idx("height_cm")
    weight_idx     = _idx("weight_kg")
    pack_years_idx = _idx("pack_years")

    # Slider reset positions: cohort means, with pack-years pinned to 0
    # (never-smoker baseline). The regression formula still uses
    # `meta_means` for the predictor delta.
    meta_init = list(meta_means.round(2).tolist())
    if pack_years_idx >= 0:
        meta_init[pack_years_idx] = 0.0

    # Widen the derived-BMI slider range to cover worst-case combinations of
    # the empirical height/weight quantiles.
    if bmi_idx >= 0 and height_idx >= 0 and weight_idx >= 0:
        h_lo, h_hi = meta_q[height_idx]
        w_lo, w_hi = meta_q[weight_idx]
        bmi_lo = w_lo / (h_hi / 100.0) ** 2
        bmi_hi = w_hi / (h_lo / 100.0) ** 2
        meta_q[bmi_idx] = [
            float(min(meta_q[bmi_idx][0], bmi_lo)),
            float(max(meta_q[bmi_idx][1], bmi_hi)),
        ]

    # Visual grouping for the Metadata Explorer panel; skip empty groups.
    meta_groups = []
    for grp in _METADATA_GROUPS:
        idxs = [available.index(p) for p in grp["predictors"] if p in available]
        if idxs:
            meta_groups.append({"title": grp["title"], "indices": idxs})

    # ── Per-sex PC means for the Metadata-Explorer sex toggle ──────────
    # PCA centres the data, so Δsex_k = mean_PC_k|sex directly. Used as an
    # additive offset on the metadata-driven Δpc. Left None when n < 2 for
    # that sex; the JS disables the matching radio.
    pc_cols_all = [f"PC_{k+1}" for k in range(n_comps)]
    sex_offsets: dict[str, list[float] | None] = {"Male": None, "Female": None}
    sex_counts:  dict[str, int]                = {"Male": 0,    "Female": 0}
    if "sex" in scores.columns:
        for label in ("Male", "Female"):
            sub = scores[scores["sex"] == label].dropna(subset=pc_cols_all)
            sex_counts[label] = int(len(sub))
            if len(sub) >= 2:
                sex_offsets[label] = (
                    sub[pc_cols_all].mean().to_numpy().round(4).tolist()
                )
    mean_pc_male   = sex_offsets["Male"]
    mean_pc_female = sex_offsets["Female"]

    # ── Per-patient data for the "Compare to real patient" overlay ──────
    # Per-patient PC scores + metadata; reconstructed client-side as the
    # translucent ghost mesh next to the slider-driven shape.
    patient_view = (
        scores.dropna(subset=pc_cols_all)
              .sort_values("patient_id")
              .reset_index(drop=True)
    )
    patient_ids       = patient_view["patient_id"].astype(int).tolist()
    patient_pc_scores = patient_view[pc_cols_all].to_numpy(dtype=np.float64)
    patient_pc_scores = np.round(patient_pc_scores, 4).tolist()
    # Per-patient metadata in the same predictor order as `available`.
    # NaNs become None so JSON serialises them as null; the JS "Load metadata"
    # handler skips nulls.
    if available:
        meta_arr = patient_view[available].to_numpy(dtype=np.float64)
        meta_arr = np.round(meta_arr, 2)
        patient_meta = [
            [None if not np.isfinite(v) else float(v) for v in row]
            for row in meta_arr
        ]
    else:
        patient_meta = [[] for _ in patient_ids]

    # Per-patient sex (string, aligned with patient_ids).  Kept as a
    # separate array rather than encoded into patient_meta because the
    # latter is float-typed and consumed by a numeric loop in
    # loadPatientIntoSliders.  The sex toggle reads this when "Load
    # metadata into sliders" snaps to a patient, and the reverse-search
    # filter reads it when restricting candidates by sex.
    if "sex" in patient_view.columns:
        patient_sex = [
            None if pd.isna(v) else str(v)
            for v in patient_view["sex"].tolist()
        ]
    else:
        patient_sex = [None] * len(patient_ids)

    # Per-patient ever-smoker status (0/1/None), aligned with patient_ids; read
    # by the smoking toggle's "Load metadata" snap and the reverse-search filter.
    if "smoking_status" in patient_view.columns:
        patient_ever = [
            None if pd.isna(v) else (0 if str(v).lower() == "never" else 1)
            for v in patient_view["smoking_status"].tolist()
        ]
    else:
        patient_ever = [None] * len(patient_ids)

    n_patients = int(scores["patient_id"].nunique())
    warn_msg = ""
    if n_patients < 30:
        warn_msg = (
            f"Only {n_patients} patients in current dataset — regression "
            f"coefficients are unreliable. Re-run with full cohort for "
            f"meaningful metadata-driven deformations."
        )

    # ── Out-of-distribution detector for the metadata explorer ──────────
    # Each metadata slider individually spans only the in-cohort
    # [q_lo, q_hi] range, but the user can still produce *combinations*
    # the cohort never contains (e.g. tall + light + high body fat).
    # Mahalanobis distance against the cohort joint distribution; JS flags
    # implausible slider configurations. BMI is excluded — it's algebraic in
    # height + weight, which would make Σ rank-deficient.
    ood_preds = [p for p in available if p != "bmi"]
    if len(ood_preds) >= 2:
        ood_indices_in_meta = [available.index(p) for p in ood_preds]
        cohort = (
            patient_view[ood_preds]
            .dropna()
            .to_numpy(dtype=np.float64)
        )
        if cohort.shape[0] >= len(ood_preds) + 1:
            ood_mean_arr = cohort.mean(axis=0)
            cov          = np.cov(cohort, rowvar=False)
            cov         += 1e-6 * np.eye(len(ood_preds))   # ridge for stability
            inv_cov      = np.linalg.inv(cov)
            centred      = cohort - ood_mean_arr
            d2_cohort    = np.einsum("ni,ij,nj->n", centred, inv_cov, centred)
            ood_warn_d2   = float(np.quantile(d2_cohort, 0.95))
            ood_severe_d2 = float(np.quantile(d2_cohort, 0.99))
            ood_indices  = ood_indices_in_meta
            ood_mean     = ood_mean_arr.round(4).tolist()
            ood_inv_cov  = np.round(inv_cov, 6).tolist()
        else:
            ood_indices = ood_mean = ood_inv_cov = None
            ood_warn_d2 = ood_severe_d2 = None
    else:
        ood_indices = ood_mean = ood_inv_cov = None
        ood_warn_d2 = ood_severe_d2 = None

    # ── Fixed scene range ────────────────────────────────────────────────
    # Pre-compute a bounding box covering every reconstructable shape so the
    # JS can set explicit per-axis ranges; otherwise Plotly's aspectmode='data'
    # auto-fits per restyle and hides size differences.
    sd_per_pc      = np.sqrt(ev).astype(np.float64)
    pc_mode_max    = VIEWER_PC_SLIDER_RANGE_SD * sd_per_pc            # (n_comps,)
    # Per-PC absolute envelope of the sex-toggle offset.
    sex_shift_abs = np.zeros(n_comps, dtype=np.float64)
    for v in (mean_pc_male, mean_pc_female):
        if v is not None:
            sex_shift_abs = np.maximum(
                sex_shift_abs, np.abs(np.asarray(v, dtype=np.float64))
            )
    if beta is not None and len(available) > 0:
        meta_dev = np.array([
            max(abs(meta_q[j][1] - float(meta_means.iloc[j])),
                abs(float(meta_means.iloc[j]) - meta_q[j][0]))
            for j in range(len(available))
        ], dtype=np.float64)
        meta_mode_max = (
            np.abs(beta[:, :len(available)]) @ meta_dev + sex_shift_abs
        )                                                              # (n_comps,)
    else:
        meta_mode_max = sex_shift_abs.copy()
    max_score = np.maximum(pc_mode_max, meta_mode_max)                 # (n_comps,)

    n_pts        = mean_shape.shape[0]
    comps_xyz    = components.reshape(n_comps, n_pts, 3)               # (K, n_pts, 3)
    # Σ_k max_score[k] · |comp_k(v, axis)| → worst-case per-vertex displacement.
    max_dev      = (max_score[:, None, None] * np.abs(comps_xyz)).sum(axis=0)
    lo           = (mean_shape - max_dev).min(axis=0)
    hi           = (mean_shape + max_dev).max(axis=0)
    pad          = 0.05 * (hi - lo)
    lo, hi       = lo - pad, hi + pad
    scene_range  = {
        "x": [float(lo[0]), float(hi[0])],
        "y": [float(lo[1]), float(hi[1])],
        "z": [float(lo[2]), float(hi[2])],
    }
    # Colorbar range: peak per-vertex Euclidean displacement (mm) under any
    # *single* slider pushed to its extreme — narrower than the worst-case
    # sign-aligned hypercube corner used for scene_range, so single-slider
    # exploration uses the full colorscale.
    comp_norms = np.linalg.norm(comps_xyz, axis=2)                     # (K, n_pts)
    pc_peak    = float((pc_mode_max[:, None] * comp_norms).max())
    meta_peak  = 0.0
    if beta is not None and len(available) > 0:
        # Per-predictor worst-vertex norm of meta_dev[j] · Σ_k β_kj · c_k(v).
        per_pred = np.einsum("kj,kva->jva", beta[:, :len(available)], comps_xyz)
        per_pred_norm = np.linalg.norm(per_pred, axis=2)               # (n_pred, n_pts)
        meta_peak = float((meta_dev[:, None] * per_pred_norm).max())
    # Sex-toggle alone: worst-vertex norm of Σ_k mean_PC_k|sex · comp_k(v, :).
    sex_peak = 0.0
    for v in (mean_pc_male, mean_pc_female):
        if v is not None:
            arr = np.asarray(v, dtype=np.float64)
            disp = (arr[:, None, None] * comps_xyz).sum(axis=0)        # (n_pts, 3)
            sex_peak = max(sex_peak, float(np.linalg.norm(disp, axis=1).max()))
    clim_max   = max(pc_peak, meta_peak, sex_peak)

    # ── Geometry generator (demographics → PC scores) for the Metadata Explorer ──
    # ŷ_k = intercept_k + Σ_j B_kj (value_j − offset_j) over the CORE predictors;
    # sex/smoking come from the toggles, the rest from the metadata sliders.
    gen_path = Path(_require_results_dir(results_dir)) / "geometry_generator.npz"
    gen_B = gen_intercept = gen_offsets = gen_predictors = None
    gen_slider_idx: list[int] = []
    is_female_any = ever_smoker_any = pack_years_any = 0.0
    resid_sd = None
    if gen_path.exists():
        with np.load(gen_path, allow_pickle=True) as gz:
            gen_predictors = [str(p) for p in gz["predictors"]]
            gen_B          = np.round(gz["B"], 6).tolist()
            gen_intercept  = np.round(gz["intercept"], 6).tolist()
            gen_offsets    = np.round(gz["offsets"], 6).tolist()
            gen_r2         = np.asarray(gz["r2"], dtype=float)
        gen_slider_idx = [available.index(p) if p in available else -1
                          for p in gen_predictors]
        # Per-PC residual SD = sqrt((1 - R²_k) · λ_k): the unexplained shape
        # variation the public viewer draws on to synthesise a cohort-typical
        # individual at the current metadata (no real-patient data required).
        resid_sd = np.sqrt(np.clip(1.0 - gen_r2[:n_comps], 0.0, None)
                           * ev[:n_comps]).round(4).tolist()
        if "sex" in scores.columns:
            is_female_any = float((scores["sex"].astype(str).str.lower() == "female").mean())
        if "smoking_status" in scores.columns:
            ever_smoker_any = float(
                (scores["smoking_status"].astype(str).str.lower() != "never").mean()
            )
        if gen_predictors and "pack_years" in gen_predictors:
            pack_years_any = float(gen_offsets[gen_predictors.index("pack_years")])

    model_data = {
        "n_pts":             int(mean_shape.shape[0]),
        "n_comps":           n_comps,
        "mean_vec":          mean_vec.round(3).tolist(),
        # float32 basis, base64-packed (decoded client-side) to keep the file
        # small enough to host statically; ~16 MB vs ~85 MB as JSON.
        "components_b64":    base64.b64encode(components.astype("<f4").tobytes()).decode("ascii"),
        "ev":                ev.tolist(),
        "evr":               evr.tolist(),
        "ev_show":           ev[:n_show].tolist(),
        "evr_show":          evr[:n_show].tolist(),
        "faces_i":           faces[:, 0].tolist(),
        "faces_j":           faces[:, 1].tolist(),
        "faces_k":           faces[:, 2].tolist(),
        "clim_max":          clim_max,
        "scene_range":       scene_range,
        "beta":              beta.round(6).tolist() if beta is not None else None,
        "meta_means":        meta_means.round(2).tolist(),
        "meta_init":         meta_init,
        "meta_quantiles":    meta_q,
        "meta_groups":       meta_groups,
        "bmi_idx":           bmi_idx,
        "height_idx":        height_idx,
        "weight_idx":        weight_idx,
        "pack_years_idx":    pack_years_idx,
        "predictor_labels":  [label_map.get(p, p) for p in available],
        # Geometry generator: demographics → PC scores (drives the Metadata Explorer).
        "gen_B":             gen_B,
        "gen_intercept":     gen_intercept,
        "gen_offsets":       gen_offsets,
        "gen_predictors":    gen_predictors,
        "gen_slider_idx":    gen_slider_idx,
        "is_female_any":     is_female_any,
        "ever_smoker_any":   ever_smoker_any,
        "pack_years_any":    pack_years_any,
        # NOTE: every per-individual array (patient_ids / patient_meta /
        # patient_pc_scores / patient_sex / patient_ever) is added below only
        # when public_mode=False. The public viewer must contain no NAKO PIDs
        # and no per-individual data anywhere in the source; its "find similar"
        # feature instead samples synthetically from `resid_sd`.
        # Per-PC mean PC scores per sex; added on top of the metadata Δpc
        # when the sex toggle is set. None → toggle disabled (n < 2 or absent).
        "mean_pc_male":      mean_pc_male,
        "mean_pc_female":    mean_pc_female,
        "n_male":            sex_counts["Male"],
        "n_female":          sex_counts["Female"],
        # OOD detector inputs (null when cohort too small or < 2 predictors).
        "ood_indices":       ood_indices,
        "ood_mean":          ood_mean,
        "ood_inv_cov":       ood_inv_cov,
        "ood_warn_d2":       ood_warn_d2,
        "ood_severe_d2":     ood_severe_d2,
        "warn_msg":          warn_msg,
        # ── Theme — from settings.FONT_FAMILY + utils.colors.PALETTE ───────
        "font_family":            FONT_FAMILY,
        # Sequential plasma ("displacement") — non-negative.
        "displacement_colorscale": _colors.colorscale("displacement"),
        # Neutral overlay colour for the real-patient ghost mesh.
        "ghost_color":             "#5a6470",
    }
    if not public_mode:
        model_data["patient_ids"]       = patient_ids
        model_data["patient_meta"]      = patient_meta
        model_data["patient_pc_scores"] = patient_pc_scores
        model_data["patient_sex"]       = patient_sex
        model_data["patient_ever"]      = patient_ever
    else:
        model_data["resid_sd"] = resid_sd

    cum_var = f"{evr.cumsum()[-1] * 100:.1f}"
    accent_color = _colors.color("sex", "Male")

    banner_html   = "" if public_mode else _INTERNAL_BANNER_HTML
    overlay_inner = (_PUBLIC_OVERLAY_BAR_INNER if public_mode
                     else _INTERNAL_OVERLAY_BAR_INNER)
    patient_js    = (_PUBLIC_PATIENT_JS if public_mode
                     else _INTERNAL_PATIENT_JS)

    html = (
        _HTML_TEMPLATE
        .replace("__INTERNAL_BANNER__",   banner_html)
        .replace("__OVERLAY_BAR_INNER__", overlay_inner)
        .replace("__PATIENT_JS__",        patient_js)
        .replace("__MODEL_JSON__",     json.dumps(model_data, separators=(",", ":")))
        .replace("__SLIDER_RANGE_SD__", f"{VIEWER_PC_SLIDER_RANGE_SD}")
        .replace("__N_PATIENTS__",     str(n_patients))
        .replace("__N_COMPS__",        str(n_comps))
        .replace("__CUM_VAR__",        cum_var)
        .replace("__WARN_MSG__",       warn_msg)
        .replace("__FONT_FAMILY__",    FONT_FAMILY)
        .replace("__ACCENT_COLOR__",   accent_color)
        .replace("__GHOST_COLOR__",    model_data["ghost_color"])
    )

    if output_path is None:
        d = _require_results_dir(results_dir)
        figs = d / "figures"
        figs.mkdir(parents=True, exist_ok=True)
        suffix = "public" if public_mode else "internal"
        output_path = figs / f"viewer_surface_{suffix}.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(
        f"Viewer ({'public' if public_mode else 'internal'}) written → "
        f"{output_path}  ({output_path.stat().st_size // 1024} KB)"
    )
    return output_path


def export_both(
    results_dir: Path | None = None,
    output_dir: Path | None = None,
    n_pcs: int = N_PCS_DISPLAY,
) -> tuple[Path, Path]:
    """Write both the internal and public viewers; return their paths.

    Both files go into ``output_dir`` (default: ``<results_dir>/figures``).
    Filenames are fixed: ``viewer_surface_internal.html`` and
    ``viewer_surface_public.html``.
    """
    if output_dir is None:
        d = _require_results_dir(results_dir)
        output_dir = d / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    internal_path = export_html(
        output_path=output_dir / "viewer_surface_internal.html",
        results_dir=results_dir,
        n_pcs=n_pcs,
        public_mode=False,
    )
    public_path = export_html(
        output_path=output_dir / "viewer_surface_public.html",
        results_dir=results_dir,
        n_pcs=n_pcs,
        public_mode=True,
    )
    return internal_path, public_path


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--results", type=Path, required=True,
        help="SSM PCA outputs dir to read from (typically "
             "<run-dir>/ssm_pca/).",
    )
    ap.add_argument(
        "--n-pcs", type=int, default=N_PCS_DISPLAY,
        help="Number of PC sliders shown in the PC explorer panel.",
    )
    ap.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Directory to write both viewer HTMLs into "
             "(default: <run_dir>/figures/). Two files are produced: "
             "viewer_surface_internal.html and viewer_surface_public.html.",
    )
    ap.add_argument(
        "--no-browser", action="store_true",
        help="Write the HTML files without opening the internal one.",
    )
    args = ap.parse_args()

    internal_path, _ = export_both(
        results_dir=args.results,
        output_dir=args.output_dir,
        n_pcs=args.n_pcs,
    )
    if not args.no_browser:
        webbrowser.open(internal_path.resolve().as_uri())


if __name__ == "__main__":
    main()
