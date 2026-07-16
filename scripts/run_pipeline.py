"""End-to-end pipeline runner.

Orchestrates the rib SSM pipeline from raw inputs through PCA + QA +
visualizations. The SSM registration stage runs
``nako.ribs.RibRegistration`` (sbt project at ``environment/scalismo_ssm/``).

Usage::

    python scripts/run_pipeline.py presets/example.yaml

Requires sbt + JDK 17 on PATH.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ssm.scalismo import scalismo_run                                       # noqa: E402
from utils.config import load_config                                        # noqa: E402
from utils.logging import get_logger, log_stage_config                      # noqa: E402
from utils.paths import (                                                   # noqa: E402
    extracted_stl_dir as paths_extracted_stl_dir,
    registered_stl_dir as paths_registered_stl_dir,
    stage_dir,
)
from utils.run_dir import make_run_dir                                      # noqa: E402

log = get_logger("run_pipeline")


_ALLOWED_RESIDUAL_SCOPE = {"whole-cage", "rib"}

_REQUIRED_PATH_KEYS  = ("root",)
_ALL_PATH_KEYS       = _REQUIRED_PATH_KEYS
_PATH_PLACEHOLDER    = "REPLACE_ME_WITH_AN_ABSOLUTE_PATH"

_STAGE_KEYS = (
    "ingestion", "adjusted",
    "mesh_extraction",
    "ssm_registration", "ssm_pca", "radiomics_correlation", "ssm_viewer",
    "ssm_qa_metrics", "ssm_qa_residuals",
    "visualizations",
)

_TEMPLATE_ID_PLACEHOLDER = "REPLACE_ME_WITH_A_PATIENT_ID"


@dataclass
class Preset:
    name:        str
    n_patients:  int | None
    template_id: str
    stages:      dict[str, bool]
    mesh:        dict[str, Any]
    ssm:         dict[str, Any]
    qa:          dict[str, Any]
    paths:       dict[str, Path]
    methodology_figure: dict[str, Any] | None = None


def load_preset(path: Path) -> Preset:
    raw = yaml.safe_load(path.read_text())

    if "name" not in raw or not isinstance(raw["name"], str) or not raw["name"]:
        raise ValueError(f"{path}: missing required field `name`")

    tid = raw.get("template_id")
    if tid is None or not str(tid).strip():
        raise ValueError(f"{path}: missing required field `template_id`")
    if str(tid).strip() == _TEMPLATE_ID_PLACEHOLDER:
        raise ValueError(
            f"{path}: `template_id` is still the example placeholder."
        )

    n_pat = raw.get("n_patients")
    if n_pat is not None and not (isinstance(n_pat, int) and not isinstance(n_pat, bool) and n_pat > 0):
        raise ValueError(
            f"{path}: `n_patients` must be a positive integer or null; got {n_pat!r}"
        )

    stages = {k: True for k in _STAGE_KEYS}
    stages.update(raw.get("stages") or {})
    for k in stages:
        if k not in _STAGE_KEYS:
            raise ValueError(f"{path}: unknown stage `{k}`")

    mesh = raw.get("mesh") or {}
    tfpr = mesh.get("target_faces_per_rib")
    if tfpr is not None and not (isinstance(tfpr, int) and not isinstance(tfpr, bool) and tfpr > 0):
        raise ValueError(
            f"{path}: mesh.target_faces_per_rib={tfpr!r} must be a positive integer"
        )

    ssm = raw.get("ssm") or {}
    workers = ssm.get("workers")
    if workers is not None and not (isinstance(workers, int) and not isinstance(workers, bool) and workers > 0):
        raise ValueError(
            f"{path}: ssm.workers={workers!r} must be a positive integer"
        )
    vt = ssm.get("variance_threshold")
    if vt is not None and not (
        isinstance(vt, (int, float)) and not isinstance(vt, bool) and 0 < float(vt) <= 1.0
    ):
        raise ValueError(
            f"{path}: ssm.variance_threshold={vt!r} must satisfy 0 < x <= 1"
        )
    bidir = ssm.get("bidirectional")
    if bidir is not None and not isinstance(bidir, bool):
        raise ValueError(
            f"{path}: ssm.bidirectional={bidir!r} must be a boolean"
        )

    qa = raw.get("qa") or {}
    scope = qa.get("residual_scope")
    if scope is not None and scope not in _ALLOWED_RESIDUAL_SCOPE:
        raise ValueError(
            f"{path}: qa.residual_scope={scope!r} must be one of "
            f"{sorted(_ALLOWED_RESIDUAL_SCOPE)}"
        )
    if scope == "rib" and not qa.get("residual_rib_id"):
        raise ValueError(
            f"{path}: qa.residual_scope='rib' requires qa.residual_rib_id "
            f"(e.g. 'rib7_R')."
        )
    nlf = qa.get("eval_n_loo_folds")
    if nlf is not None and not (
        isinstance(nlf, int) and not isinstance(nlf, bool) and nlf >= 0
    ):
        raise ValueError(
            f"{path}: qa.eval_n_loo_folds={nlf!r} must be a non-negative int "
            f"(0 = true LOO; >=2 = K-fold)"
        )
    swc = qa.get("eval_skip_whole_cage")
    if swc is not None and not isinstance(swc, bool):
        raise ValueError(
            f"{path}: qa.eval_skip_whole_cage={swc!r} must be a boolean"
        )
    ewk = qa.get("eval_workers")
    if ewk is not None and not (
        isinstance(ewk, int) and not isinstance(ewk, bool) and ewk >= 1
    ):
        raise ValueError(
            f"{path}: qa.eval_workers={ewk!r} must be a positive int"
        )

    raw_paths = raw.get("paths") or {}
    if not isinstance(raw_paths, dict):
        raise ValueError(f"{path}: `paths` must be a mapping")
    paths: dict[str, Path] = {}
    for k in _REQUIRED_PATH_KEYS:
        if k not in raw_paths:
            raise ValueError(
                f"{path}: missing required field `paths.{k}` "
                f"(absolute path to the STL directory)"
            )
    for k, v in raw_paths.items():
        if k not in _ALL_PATH_KEYS:
            raise ValueError(
                f"{path}: unknown paths.{k} "
                f"(allowed: {list(_ALL_PATH_KEYS)})"
            )
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{path}: paths.{k} must be a non-empty string")
        if v.strip() == _PATH_PLACEHOLDER:
            raise ValueError(
                f"{path}: paths.{k} is still the placeholder "
                f"{_PATH_PLACEHOLDER!r}; replace it with an absolute path."
            )
        paths[k] = Path(v).expanduser()

    mf_raw = raw.get("methodology_figure")
    mf: dict[str, Any] | None
    if mf_raw is None:
        mf = None
    elif not isinstance(mf_raw, dict):
        raise ValueError(f"{path}: `methodology_figure` must be a mapping")
    elif mf_raw.get("display_patient") is None:
        mf = None
    else:
        dp = mf_raw["display_patient"]
        if not (isinstance(dp, int) and not isinstance(dp, bool)):
            raise ValueError(
                f"{path}: methodology_figure.display_patient must be an int; got {dp!r}"
            )
        mf = dict(mf_raw)
        mf["display_patient"] = int(dp)

    return Preset(
        name        = raw["name"],
        n_patients  = n_pat,
        template_id = str(tid).strip(),
        stages      = stages,
        mesh        = mesh,
        ssm         = ssm,
        qa          = qa,
        paths       = paths,
        methodology_figure = mf,
    )


# ── Subprocess helpers ────────────────────────────────────────────────────────

def run_streaming(script: Path, extra_args: list[str] | tuple[str, ...] = ()) -> int:
    """Spawn ``script`` as a subprocess and stream raw bytes to stdout.

    Forwards bytes (not lines) so ``\\r``-based progress updates from tqdm
    survive the subprocess boundary; stderr is merged into stdout.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [sys.executable, str(script), *extra_args],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env,
    )
    assert proc.stdout is not None
    parent_out = sys.stdout.buffer
    while True:
        chunk = proc.stdout.read1(4096)  # type: ignore[attr-defined]
        if not chunk:
            break
        parent_out.write(chunk)
        parent_out.flush()
    proc.wait()
    return proc.returncode


def _check(rc: int, label: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{label} exited with code {rc}")


# ── Stage runners ─────────────────────────────────────────────────────────────

def _stage_ingestion(preset: Preset, run_dir: Path, workers_arg: list[str]) -> None:
    args: list[str] = ["--run-dir", str(run_dir)]
    if preset.n_patients is not None:
        args += ["--n-patients", str(preset.n_patients)]
    args += workers_arg
    _check(run_streaming(REPO_ROOT / "src" / "data_ingestion" / "run_ingestion.py", args),
           "run_ingestion.py")


def _stage_adjusted(run_dir: Path) -> None:
    _check(run_streaming(REPO_ROOT / "src" / "adjusted" / "run_adjusted.py",
                         ["--run-dir", str(run_dir)]),
           "run_adjusted.py")


def _stage_mesh_extraction(
    preset: Preset, run_dir: Path, stl_per_rib: Path, workers_arg: list[str]
) -> None:
    args: list[str] = [
        "--run-dir", str(run_dir),
        "--stl-dir", str(stl_per_rib),
    ]
    if preset.n_patients is not None:
        args += ["--n-patients", str(preset.n_patients)]
    tfpr = preset.mesh.get("target_faces_per_rib")
    if tfpr is not None:
        args += ["--target-faces", str(int(tfpr))]
    args += workers_arg
    _check(run_streaming(REPO_ROOT / "src" / "ssm" / "run_mesh_extraction.py", args),
           "run_mesh_extraction.py")


def _stage_template(preset: Preset, reg_out_dir: Path, stl_per_rib: Path) -> None:
    """Write template_id.txt — required by downstream PCA stage."""
    target = reg_out_dir / "template_id.txt"
    if not stl_per_rib.exists():
        raise FileNotFoundError(
            f"Per-rib STL dir {stl_per_rib} not found. "
            f"Enable `stages.mesh_extraction` in your preset to populate it."
        )
    pids_file = stl_per_rib / "patient_ids.txt"
    if pids_file.exists():
        extracted = {l.strip() for l in pids_file.read_text().splitlines() if l.strip()}
        if extracted and preset.template_id not in extracted:
            raise ValueError(
                f"Preset template_id={preset.template_id!r} is not in "
                f"{pids_file} ({len(extracted)} extracted patients)."
            )
    reg_out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(preset.template_id)
    log.info(f"Template ID '{preset.template_id}' written to {target}")


def _stage_ssm_registration(
    preset:               Preset,
    extracted_stl_dir:    Path,
    reg_out_dir:          Path,
    extract_pids_txt:     Path,
) -> None:
    """Run rib registration via the Scalismo sbt project.

    Invokes ``runMain nako.ribs.RibRegistration`` through the
    ``scalismo_run`` helper, which wraps
    ``environment/scalismo_ssm/sbt_build.sh``.
    """
    if not extract_pids_txt.exists():
        raise FileNotFoundError(
            f"{extract_pids_txt} missing. Either run mesh extraction or copy "
            f"patient_ids.txt into the per-rib STL dir."
        )

    # Honor n_patients by writing a temp pids file.
    pids = [l.strip() for l in extract_pids_txt.read_text().splitlines() if l.strip()]
    tmp_pids_file: Path | None = None
    if preset.n_patients is not None and len(pids) > preset.n_patients:
        reg_out_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix="_pids.txt", dir=str(reg_out_dir))
        os.close(fd)
        tmp_pids_file = Path(tmp_path)
        tmp_pids_file.write_text("\n".join(pids[:preset.n_patients]))
        pids_file_for_scala = tmp_pids_file
        log.info(f"Restricting registration to {preset.n_patients} of {len(pids)} patients")
    else:
        pids_file_for_scala = extract_pids_txt
        log.info(f"Running registration on {len(pids)} patients")

    workers = int(preset.ssm.get("workers", 32))
    bidirectional = bool(preset.ssm.get("bidirectional", False))

    bidir_flag = " --bidirectional" if bidirectional else ""

    # Methodology-figure intermediates: when the preset names a display patient,
    # the Scala registration writes _methodology/{cage_patient,per_rib_template,
    # gp_fit_template}/ alongside the production output for that one patient.
    mf = preset.methodology_figure or {}
    methodology_pid = mf.get("display_patient")
    methodology_flag = (
        f" --methodology-patient-id {int(methodology_pid)}"
        if methodology_pid is not None else ""
    )

    sbt_cmd = (
        f'"runMain nako.ribs.RibRegistration'
        f' --input \\"{extracted_stl_dir}\\"'
        f' --output \\"{reg_out_dir}\\"'
        f' --template-dir \\"{extracted_stl_dir}\\"'
        f' --template-pid {preset.template_id}'
        f' --workers {workers}'
        f' --patient-ids-file \\"{pids_file_for_scala}\\"'
        f'{bidir_flag}{methodology_flag}"'
    )
    try:
        rc = scalismo_run(sbt_cmd)
    finally:
        if tmp_pids_file is not None:
            tmp_pids_file.unlink(missing_ok=True)
    _check(rc, "RibRegistration (sbt)")


def _stage_ssm_pca(preset: Preset, run_dir: Path, parquet: Path, stl_per_rib: Path,
                   reg_out_dir: Path, extract_pids_txt: Path,
                   workers_arg: list[str]) -> None:
    args = [
        "--run-dir",          str(run_dir),
        "--parquet",          str(parquet),
        "--extraction-dir",   str(stl_per_rib),
        "--registration-dir", str(reg_out_dir),
    ]
    if extract_pids_txt.exists():
        args += ["--patient-ids-file", str(extract_pids_txt)]
    vt = preset.ssm.get("variance_threshold")
    if vt is not None:
        args += ["--variance-threshold", str(float(vt))]
    args += workers_arg
    _check(run_streaming(REPO_ROOT / "src" / "ssm" / "run_ssm.py", args),
           "run_ssm.py")


def _stage_radiomics_correlation(run_dir: Path) -> None:
    args = ["--run-dir", str(run_dir)]
    _check(run_streaming(REPO_ROOT / "src" / "ssm" / "run_radiomics_correlation.py", args),
           "run_radiomics_correlation.py")


def _stage_ssm_viewer(run_dir: Path) -> None:
    from ssm.viewer import export_both
    pca_dir = stage_dir(run_dir, "ssm_pca")
    out_dir = stage_dir(run_dir, "ssm_viewer")
    out_dir.mkdir(parents=True, exist_ok=True)
    internal_path, public_path = export_both(
        results_dir=pca_dir,
        output_dir=out_dir,
    )
    log.info(f"Viewer HTML (internal): {internal_path}")
    log.info(f"Viewer HTML (public):   {public_path}")


def _stage_ssm_qa_metrics(preset: Preset, run_dir: Path) -> None:
    args = [
        "--run-dir",        str(run_dir),
        "--max-modes",      str(int(preset.qa.get("eval_max_modes", 25))),
        "--n-spec-samples", str(int(preset.qa.get("eval_n_specificity_samples", 1000))),
        "--n-loo-folds",    str(int(preset.qa.get("eval_n_loo_folds", 10))),
        "--workers",        str(int(preset.qa.get("eval_workers", 1))),
    ]
    if bool(preset.qa.get("eval_skip_whole_cage", False)):
        args.append("--skip-whole-cage")
    _check(run_streaming(REPO_ROOT / "src" / "ssm" / "eval_metrics.py", args),
           "eval_metrics.py")


def _stage_ssm_qa_residuals(
    preset:      Preset,
    run_dir:     Path,
    stl_per_rib: Path,
    reg_out_dir: Path,
    workers_arg: list[str],
) -> None:
    scope = str(preset.qa.get("residual_scope", "whole-cage"))
    args = [
        "--run-dir",        str(run_dir),
        "--target-dir",     str(stl_per_rib),
        "--registered-dir", str(reg_out_dir),
        "--worst-n",        str(int(preset.qa.get("residual_worst_n", 6))),
        "--scope",          scope,
    ]
    if scope == "rib":
        args += ["--rib-id", str(preset.qa["residual_rib_id"])]
    args += workers_arg
    _check(run_streaming(REPO_ROOT / "src" / "ssm" / "eval_residuals.py", args),
           "eval_residuals.py")


def _stage_visualizations(preset: Preset, run_dir: Path) -> None:
    """Re-render the canonical figure set into ``<run-dir>/visualizations/``."""
    args = ["--run-dir", str(run_dir)]
    if preset.methodology_figure is not None:
        args += ["--methodology-figure-json", json.dumps(preset.methodology_figure)]
    _check(run_streaming(REPO_ROOT / "scripts" / "run_visualizations.py", args),
           "run_visualizations.py")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("preset", type=Path, help="Path to a preset YAML file")
    p.add_argument(
        "--run-dir", type=Path, default=None,
        help="Reuse an existing run dir instead of creating a fresh one.",
    )
    args = p.parse_args()

    if not args.preset.exists():
        log.error(f"Preset not found: {args.preset}")
        return 2

    try:
        preset = load_preset(args.preset)
    except ValueError as exc:
        log.error(str(exc))
        return 2

    try:
        load_config()
    except FileNotFoundError as exc:
        log.error(
            f"{exc}\n"
            f"  → Copy data/data_config.example.yaml to data/data_config.yaml "
            f"and fill in the paths."
        )
        return 2

    root         = preset.paths["root"].resolve()
    stl_per_rib  = paths_extracted_stl_dir(root, preset.name)
    reg_out_dir  = paths_registered_stl_dir(root, preset.name)

    if preset.stages.get("mesh_extraction"):
        stl_per_rib.mkdir(parents=True, exist_ok=True)
    elif not stl_per_rib.is_dir():
        log.error(
            f"extracted STL dir does not exist or is not a directory: "
            f"{stl_per_rib}. Either enable stages.mesh_extraction or "
            f"point paths.root at a directory whose <name>/extracted_stl/ "
            f"already has extracted STLs."
        )
        return 2
    reg_out_dir.mkdir(parents=True, exist_ok=True)
    extract_pids_txt = stl_per_rib / "patient_ids.txt"

    if args.run_dir is not None:
        run_dir = args.run_dir.expanduser().resolve()
        if not run_dir.exists():
            log.error(f"--run-dir does not exist: {run_dir}")
            return 2
        log.info(f"Reusing existing run dir: {run_dir}")
    else:
        run_dir = make_run_dir(root, preset.name)
    parquet_path = stage_dir(run_dir, "ingestion") / "analytic_clean.parquet"

    log.info("─" * 70)
    log.info(f"Preset:                      {args.preset}  (name={preset.name})")
    log.info(f"n_patients:                  {preset.n_patients}")
    log.info(f"template_id:                 {preset.template_id}")
    log.info(f"Stages enabled:              "
             f"{[k for k, v in preset.stages.items() if v]}")
    log.info(f"Preset root:                 {root}")
    log.info(f"Extracted STL dir:           {stl_per_rib}")
    log.info(f"Registered STL dir:          {reg_out_dir}")
    log.info(f"Run dir:                     {run_dir}")
    log.info("─" * 70)

    t0 = time.monotonic()
    ran:     list[str] = []
    skipped: list[str] = []

    # `ssm.workers`: single override applied to every parallelized stage.
    # Absent → library defaults (SSM_LOAD_N_WORKERS / MESH_N_WORKERS = 32).
    _preset_workers = preset.ssm.get("workers")
    _eff_workers = int(_preset_workers) if _preset_workers is not None else 32
    workers_arg: list[str] = (
        ["--workers", str(_eff_workers)] if _preset_workers is not None else []
    )

    # Per-stage knob summary logged at the top of each stage banner.
    stage_settings: dict[str, dict] = {
        "ingestion": {
            "workers": _eff_workers,
        },
        "mesh_extraction": {
            **{k: v for k, v in {
                "target_faces_per_rib": preset.mesh.get("target_faces_per_rib"),
                "n_patients":           preset.n_patients,
            }.items() if v is not None},
            "workers": _eff_workers,
        },
        "ssm_registration": {
            "workers":       _eff_workers,
            "template_id":   preset.template_id,
            "n_patients":    preset.n_patients,
            "bidirectional": bool(preset.ssm.get("bidirectional", False)),
        },
        "ssm_pca": {
            **{k: v for k, v in {
                "variance_threshold": preset.ssm.get("variance_threshold"),
            }.items() if v is not None},
            "workers": _eff_workers,
        },
        "ssm_qa_metrics": {
            "eval_max_modes":
                int(preset.qa.get("eval_max_modes", 25)),
            "eval_n_specificity_samples":
                int(preset.qa.get("eval_n_specificity_samples", 1000)),
            "eval_n_loo_folds":
                int(preset.qa.get("eval_n_loo_folds", 10)),
            "eval_skip_whole_cage":
                bool(preset.qa.get("eval_skip_whole_cage", False)),
            "eval_workers":
                int(preset.qa.get("eval_workers", 1)),
        },
        "ssm_qa_residuals": {
            "scope":    preset.qa.get("residual_scope", "whole-cage"),
            "worst_n":  int(preset.qa.get("residual_worst_n", 6)),
            "workers":  _eff_workers,
        },
    }

    def gate(name: str, fn) -> None:
        if preset.stages.get(name):
            log.info(f"=== STAGE {name} — START ===")
            log_stage_config(log, name, stage_settings.get(name, {}))
            fn()
            log.info(f"=== STAGE {name} — DONE ===\n")
            ran.append(name)
        else:
            log.info(f"=== STAGE {name} — SKIPPED (disabled in preset) ===\n")
            skipped.append(name)

    gate("ingestion",       lambda: _stage_ingestion(preset, run_dir, workers_arg))
    gate("adjusted",        lambda: _stage_adjusted(run_dir))
    gate("mesh_extraction", lambda: _stage_mesh_extraction(preset, run_dir, stl_per_rib, workers_arg))

    if any(preset.stages.get(k) for k in
           ("ssm_registration", "ssm_pca", "ssm_viewer",
            "ssm_qa_metrics", "ssm_qa_residuals")):
        _stage_template(preset, reg_out_dir, stl_per_rib)

    gate("ssm_registration",
         lambda: _stage_ssm_registration(preset, stl_per_rib,
                                         reg_out_dir, extract_pids_txt))

    gate("ssm_pca",
         lambda: _stage_ssm_pca(preset, run_dir, parquet_path,
                                stl_per_rib, reg_out_dir, extract_pids_txt,
                                workers_arg))
    gate("radiomics_correlation",
         lambda: _stage_radiomics_correlation(run_dir))
    gate("ssm_viewer",
         lambda: _stage_ssm_viewer(run_dir))
    gate("ssm_qa_metrics",
         lambda: _stage_ssm_qa_metrics(preset, run_dir))
    gate("ssm_qa_residuals",
         lambda: _stage_ssm_qa_residuals(preset, run_dir,
                                         stl_per_rib, reg_out_dir,
                                         workers_arg))

    gate("visualizations",
         lambda: _stage_visualizations(preset, run_dir))

    elapsed = time.monotonic() - t0
    m, s = divmod(elapsed, 60)
    log.info("─" * 70)
    log.info(f"Pipeline complete — {m:.0f}m{s:.0f}s")
    log.info(f"  Ran:      {ran}")
    log.info(f"  Skipped:  {skipped}")
    log.info(f"  Run dir:  {run_dir}")
    log.info("─" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
