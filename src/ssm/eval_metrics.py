"""Styner-triad evaluation of the surface SSM (Davies / Cates).

Reads the ``(N, n_pts_total, 3)`` shape stack from ``<run>/ssm_pca/`` and
reports per-rib and whole-cage:

  - compactness     cumulative explained variance vs. mode count
  - generalisation  K-fold reconstruction error vs. mode count (default K=10;
                    ``--n-loo-folds 0`` switches to true leave-one-out)
  - specificity     random-sample-from-model NN distance vs. mode count

Outputs::

    <run>/ssm_qa_metrics/eval_styner.json
    <run>/ssm_qa_metrics/figures/eval_styner.png   1×3 triptych

Metric implementations live in :mod:`utils.shape_evaluation`.

CLI::

    python src/ssm/eval_metrics.py --run-dir DIR [--max-modes 25]
        [--n-spec-samples 1000] [--n-loo-folds 10] [--skip-whole-cage]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import settings as S                                                    # noqa: E402
from settings import apply_publication_style                            # noqa: E402
from ssm.pca_surface import RIB_LABELS, RIB_SIDES                       # noqa: E402
from utils import colors as C                                           # noqa: E402
from utils.altair_theme import make_title, width_for                    # noqa: E402
from utils.figure_export_altair import save_chart                       # noqa: E402
from utils.logging import get_logger                                    # noqa: E402
from utils.paths import stage_dir                                       # noqa: E402
from utils.rib_labels import display_from_seg                           # noqa: E402
from utils.shape_evaluation import compactness, generalisation, specificity  # noqa: E402

apply_publication_style()

logger = get_logger(__name__)

RIB_IDS = [f"rib{lab}_{side}" for lab in RIB_LABELS for side in RIB_SIDES]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSM Styner-triad evaluator")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Top-level pipeline run dir.  Reads PCA outputs "
                        "from <run-dir>/ssm_pca/; writes Styner metrics + "
                        "figures to <run-dir>/ssm_qa_metrics/.")
    p.add_argument("--max-modes", type=int, default=25,
                   help="Cap on the number of PCA modes evaluated. "
                        "Default 25. Pass 0 to evaluate up to the rank ceiling "
                        "min(N-1, D) — expensive on large cohorts.")
    p.add_argument("--n-spec-samples", type=int, default=1000,
                   help="Number of random model samples for specificity. Default 1000.")
    p.add_argument("--n-loo-folds", type=int, default=10,
                   help="Folds for K-fold generalisation. Default 10. "
                        "Pass 0 for true leave-one-out (N fits, O(N^2)).")
    p.add_argument("--skip-whole-cage", action="store_true",
                   help="Skip the whole-cage stack (per-rib only). "
                        "Halves wall time on large cohorts.")
    p.add_argument("--workers", type=int, default=1,
                   help="Process-based fold parallelism for generalisation. "
                        "Default 1 (sequential). Capped at the number of folds. "
                        "Each worker peaks ~|train|·D·8 bytes (≈7 GB at full cohort).")
    return p.parse_args()


def _pca_dir(run_dir: Path) -> Path:
    """SSM PCA outputs (upstream stage)."""
    return stage_dir(run_dir, "ssm_pca")


def _out_dir(run_dir: Path) -> Path:
    """Styner-metrics outputs (this stage)."""
    return stage_dir(run_dir, "ssm_qa_metrics")


def _eval_one_stack(
    name: str,
    shapes: np.ndarray,
    max_modes: int,
    n_spec_samples: int,
    n_folds: int = 0,
    n_workers: int = 1,
) -> dict:
    """Run all three Styner metrics on a single ``(N, n_pts, 3)`` stack."""
    n = shapes.shape[0]
    if n < 3:
        logger.warning(f"[{name}] only {n} shapes — Styner triad needs ≥ 3; skipping.")
        return {"n": int(n), "skipped": True}

    # Rank ceiling shared across the three metrics. Train size depends on
    # whether generalisation runs LOO (N-1) or K-fold (≈ N·(K-1)/K).
    D = int(shapes.shape[1] * shapes.shape[2])
    if n_folds and n_folds >= 2 and n_folds <= n:
        min_train = n - int(np.ceil(n / n_folds))
    else:
        min_train = n - 1
    rank_ceiling = max(1, min(min_train - 1, D))
    eff_max = rank_ceiling if max_modes <= 0 else max(1, min(max_modes, rank_ceiling))
    mode_counts = np.arange(1, eff_max + 1, dtype=int)

    t0 = time.monotonic()
    cmp_curve = compactness(shapes, max_modes=eff_max)
    logger.info(f"[{name}] compactness done — {time.monotonic()-t0:.1f}s")

    gen_label = "LOO" if (not n_folds or n_folds > n) else f"K={n_folds} folds"
    t0 = time.monotonic()
    gen_modes, gen_curve = generalisation(
        shapes, mode_counts=mode_counts,
        n_folds=n_folds or None, n_workers=n_workers,
    )
    logger.info(
        f"[{name}] generalisation ({gen_label}, n_workers={n_workers}) "
        f"done — {time.monotonic()-t0:.1f}s"
    )

    t0 = time.monotonic()
    spec_modes, spec_curve = specificity(
        shapes, n_samples=n_spec_samples, mode_counts=mode_counts,
    )
    logger.info(f"[{name}] specificity ({n_spec_samples} samples) done — {time.monotonic()-t0:.1f}s")

    return {
        "n":                int(n),
        "n_pts":            int(shapes.shape[1]),
        "mode_counts":      [int(m) for m in mode_counts],
        "compactness":      [float(v) for v in cmp_curve],
        "generalisation_mm": [float(v) for v in gen_curve],
        "specificity_mm":    [float(v) for v in spec_curve],
    }


def load_styner_json(path: Path) -> tuple[dict[str, dict], dict | None]:
    """Parse ``eval_styner.json`` into ``(per_rib_results, whole_cage)``."""
    payload = json.loads(Path(path).read_text())
    return payload["per_rib"], payload.get("whole_cage")


def plot_styner_triptych(
    out_path: Path,
    per_rib_results: dict[str, dict],
    whole_cage: dict | None,
) -> None:
    """1×3 triptych: compactness / generalisation / specificity.

    Per-rib curves are drawn translucently with a bold median-over-ribs
    overlay; whole-cage is a separate bold dashed line. ``out_path`` may
    be ``…/figures/eval_styner.png`` — the suffix is stripped and
    :func:`save_chart` writes HTML / SVG / PNG.
    """
    out_stem = out_path.with_suffix("") if out_path.suffix else out_path
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, str, str, bool]] = [
        ("compactness",       "Compactness",          "cumulative variance", False),
        ("generalisation_mm", "Generalisation (LOO)", "RMS error (mm)",       True),
        ("specificity_mm",    "Specificity",          "mean NN distance (mm)", True),
    ]

    rib_rows: list[dict] = []
    for rib_id, r in per_rib_results.items():
        if r.get("skipped"):
            continue
        for key, _, _, _ in panels:
            for n_modes, value in zip(r["mode_counts"], r[key]):
                rib_rows.append({
                    "metric": key,
                    "n_modes": int(n_modes),
                    "value": float(value),
                    "rib_id": str(rib_id),
                })
    rib_df = pd.DataFrame(rib_rows)
    n_ribs_eff = rib_df["rib_id"].nunique() if not rib_df.empty else 0

    median_df = (
        rib_df.groupby(["metric", "n_modes"], as_index=False)["value"].median()
        if not rib_df.empty
        else pd.DataFrame(columns=["metric", "n_modes", "value"])
    )

    cage_rows: list[dict] = []
    if whole_cage and not whole_cage.get("skipped"):
        for key, _, _, _ in panels:
            for n_modes, value in zip(whole_cage["mode_counts"], whole_cage[key]):
                cage_rows.append({
                    "metric": key,
                    "n_modes": int(n_modes),
                    "value": float(value),
                })
    cage_df = pd.DataFrame(cage_rows)

    if rib_df.empty and cage_df.empty:
        logger.warning("Styner triptych: no data to plot (all ribs skipped, no whole-cage).")
        return

    rib_color    = C.color("sex", "Male")
    cage_color   = C.color("sex", "Female")
    line_thin    = S.LINE_WIDTH * 0.6
    line_bold    = S.LINE_WIDTH * 1.6
    median_label = f"per-rib median (n={n_ribs_eff})"
    cage_label   = "whole-cage"

    color_scale = alt.Scale(
        domain=[median_label, cage_label],
        range=[rib_color, cage_color],
    )
    legend = alt.Legend(title=None, orient="right")

    def _panel(key: str, title: str, ylabel: str, zero_y: bool) -> alt.LayerChart:
        x_enc = alt.X("n_modes:Q", title="number of PCA modes")
        y_enc = alt.Y("value:Q", title=ylabel, scale=alt.Scale(zero=zero_y))

        layers: list[alt.Chart] = []
        pr = rib_df[rib_df["metric"] == key] if not rib_df.empty else rib_df
        if not pr.empty:
            layers.append(
                alt.Chart(pr)
                .mark_line(opacity=0.20, color=rib_color, strokeWidth=line_thin)
                .encode(x=x_enc, y=y_enc, detail="rib_id:N")
            )
        md = median_df[median_df["metric"] == key] if not median_df.empty else median_df
        if not md.empty:
            layers.append(
                alt.Chart(md.assign(series=median_label))
                .mark_line(strokeWidth=line_bold)
                .encode(
                    x=x_enc, y=y_enc,
                    color=alt.Color("series:N", scale=color_scale, legend=legend),
                )
            )
        cg = cage_df[cage_df["metric"] == key] if not cage_df.empty else cage_df
        if not cg.empty:
            layers.append(
                alt.Chart(cg.assign(series=cage_label))
                .mark_line(strokeWidth=line_bold, strokeDash=[6, 3])
                .encode(
                    x=x_enc, y=y_enc,
                    color=alt.Color("series:N", scale=color_scale, legend=legend),
                )
            )
        return alt.layer(*layers).properties(
            title=title,
            width=width_for("third"),
            height=180,
        )

    title_text = "Styner triad — surface SSM"
    chart = (
        alt.hconcat(*[_panel(*p) for p in panels])
        .resolve_scale(y="independent", color="shared")
        .properties(title=make_title(title_text))
    )

    save_chart(chart, out_stem,
               title="Styner triad (compactness · generalisation · specificity)",
               width_class="full")
    logger.info(f"Wrote Styner triptych → {out_stem}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    pca_dir = _pca_dir(run_dir)
    out_dir = _out_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Reading PCA outputs from: {pca_dir}")

    shapes_path  = pca_dir / "shapes_registered.npz"
    offsets_path = pca_dir / "rib_offsets.npy"
    for f in (shapes_path, offsets_path):
        if not f.exists():
            raise FileNotFoundError(f"{f} not found — run the ssm_pca stage first.")

    data    = np.load(shapes_path)
    shapes  = data["shapes"]
    pids    = data["patient_ids"]
    offsets = np.load(offsets_path)
    n_ribs  = len(offsets)
    n_total = shapes.shape[1]
    logger.info(f"Loaded shapes={shapes.shape}  patients={len(pids)}  ribs={n_ribs}")

    rib_endpoints = list(offsets) + [n_total]

    # Per-rib evaluation.  Dict keys stay in the internal seg-label form
    # (`rib40_L` … `rib51_R`) — that's the on-disk JSON contract.  The
    # display form (`Rib 1 L` …) is used only in log lines so the
    # extraction job is readable to a human.
    per_rib: dict[str, dict] = {}
    for k in range(n_ribs):
        rib_id = RIB_IDS[k] if k < len(RIB_IDS) else f"rib_{k}"
        try:
            head, side = rib_id.rsplit("_", 1)
            display_id = display_from_seg(int(head.removeprefix("rib")), side)
        except ValueError:
            display_id = rib_id
        s, e   = int(rib_endpoints[k]), int(rib_endpoints[k + 1])
        sub    = shapes[:, s:e, :]
        logger.info(f"[{display_id}] vertices={sub.shape[1]}  evaluating Styner triad…")
        per_rib[rib_id] = _eval_one_stack(
            display_id, sub,
            max_modes=args.max_modes,
            n_spec_samples=args.n_spec_samples,
            n_folds=args.n_loo_folds,
            n_workers=args.workers,
        )

    if args.skip_whole_cage:
        logger.info("[whole-cage] skipped (--skip-whole-cage).")
        whole_cage: dict | None = None
    else:
        logger.info(f"[whole-cage] vertices={shapes.shape[1]}  evaluating Styner triad…")
        whole_cage = _eval_one_stack(
            "whole-cage", shapes,
            max_modes=args.max_modes,
            n_spec_samples=args.n_spec_samples,
            n_folds=args.n_loo_folds,
            n_workers=args.workers,
        )

    eval_json = {
        "run_dir":           str(run_dir),
        "n_patients":        int(len(pids)),
        "n_ribs":            int(n_ribs),
        "max_modes":         int(args.max_modes),
        "n_spec_samples":    int(args.n_spec_samples),
        "n_loo_folds":       int(args.n_loo_folds),
        "skip_whole_cage":   bool(args.skip_whole_cage),
        "workers":           int(args.workers),
        "per_rib":           per_rib,
        "whole_cage":        whole_cage,
    }
    json_path = out_dir / "eval_styner.json"
    json_path.write_text(json.dumps(eval_json, indent=2))
    logger.info(f"Wrote {json_path}")

    plot_styner_triptych(
        out_dir / "figures" / "eval_styner.png",
        per_rib_results=per_rib,
        whole_cage=whole_cage,
    )


if __name__ == "__main__":
    main()
