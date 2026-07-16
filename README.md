# NAKO Human Ribcage Statistical Shape Model

Statistical analysis and statistical shape modelling (SSM) of the human rib
cage from a large NAKO cohort.

The pipeline has two parts that share a single ingested dataset:

1. **Tabular shape-parameter analysis** links per-rib analytic descriptors
   (length, sphericity, centerline curvature) to patient metadata (sex, age,
   BMI, body fat, smoking status, pack-years).
2. **Surface-based statistical shape model** extracts triangle meshes from
   segmentation masks, registers them in correspondence via Scalismo, builds a
   PCA shape model, and regresses the PC scores on metadata. The resulting
   model drives an interactive viewer that animates the cage as sliders move
   across the cohort distribution.

## Citing this work

TBD

## Repository layout

```
codebase/
├── README.md                                   # this file
├── LICENSE
├── pyproject.toml                              # editable install; pins read from environment/
├── data/
│   └── data_config.example.yaml                # template; copy to data_config.yaml
├── environment/
│   ├── requirements.txt                        # Python dependencies (pinned)
│   └── scalismo_ssm/                           # Scala / sbt project (Scalismo)
│       ├── build.sbt
│       ├── sbt_build.sh                        # invocation wrapper (handles mounts)
│       └── src/main/scala/nako/ribs/
│           ├── RibRegistration.scala           # per-rib non-rigid registration
│           └── TemplateBuilder.scala           # most-central template picker
├── notebooks/
│   └── template_selection.ipynb                # drives TemplateBuilder + 3D QC
├── presets/                                    # YAML run configurations
│   ├── example.yaml                            # annotated reference preset
│   ├── full_analysis.example.yaml              # full-cohort template; copy and fill in
│   └── smoke.yaml                              # 5-patient smoke-test preset
├── scripts/
│   ├── run_pipeline.py                         # end-to-end pipeline driver
│   ├── run_visualizations.py                   # supplemental figures stage
│   ├── run_all_plots.py                        # re-render figures from a run's caches
│   └── rerender_figures.py                     # copy a run dir and re-emit every figure
├── src/
│   ├── settings.py                             # centralised analysis constants
│   ├── data_ingestion/                         # JSON / metadata loaders + QC
│   ├── adjusted/                               # rib-level cluster-robust OLS + FWL
│   ├── bivariate/                              # Welch t-test + forest-plot helpers
│   ├── ssm/                                    # surface SSM (Python side)
│   ├── visualizations/                         # supplemental + methodology figures
│   └── utils/
│       ├── paths.py                            # stage → subdir mapping
│       ├── logging.py                          # shared logger + tqdm helpers
│       ├── run_dir.py                          # per-run output directory
│       ├── colors.py                           # palettes
│       └── rib_labels.py                       # rib-id conversions
└── tests/
```

`src/settings.py` holds every analysis-side constant: FDR display threshold,
the three-model design-matrix predictor sets and DAG adjustment sets,
mesh-extraction parameters, SSM evaluation sample counts, viewer slider
ranges, and the matplotlib publication-style helper. Plotting palettes live in
`src/utils/colors.py`.

`src/bivariate/` is a shared helper library, not a pipeline stage. It provides
the Welch t-test used by `src/ssm/pc_regression.py` and the forest-plot
builders used by `src/adjusted/`.

The Scalismo registration's GP kernel hyperparameters live on the Scala side
in `RibRegistration.scala::GPKernel.buildPerRib`.

## Environment setup

### Python

Python 3.13. `environment/requirements.txt` pins every dependency.

```bash
python -m venv ~/.venvs/nako_ribs
source ~/.venvs/nako_ribs/bin/activate
pip install -e .
```

The editable install (`pip install -e .`) reads the dependency pins from
`environment/requirements.txt` via `pyproject.toml` and exposes `bivariate`,
`adjusted`, `data_ingestion`, `ssm`, `utils`, and `settings` as top-level
importable modules.

### Scalismo (Scala / JVM, surface SSM only)

Required: JDK ≥ 17 and `sbt` 1.9.x. Scalismo `1.0-RC1` is pulled from Maven
Central via `build.sbt`.

```bash
cd environment/scalismo_ssm
./sbt_build.sh compile
```

`sbt_build.sh` is a thin wrapper around `sbt`. When the repository path
contains a `:` character (some cloud-mount layouts do), `sbt` mis-parses it as
a classpath separator; the wrapper detects this and rsyncs the project to a
colon-free cache directory before delegating to `sbt`. In the common case
where the path is colon-free it simply runs `sbt` in-place.

The Scalismo wrapper (`src/ssm/scalismo.py`, invoked by the `ssm_registration`
stage) resolves Java via `JAVA_HOME` then falls back to `~/jdk-17.0.13+11`;
`sbt` is resolved via `PATH` then falls back to `~/sbt/bin/sbt`. If neither
resolves the wrapper raises a `RuntimeError` that points back to this section.

### Data configuration

Copy `data/data_config.example.yaml` to `data/data_config.yaml` and fill in the
paths to your local NAKO derivatives, metadata Excel, codebook, and the
directory in which extracted / registered STLs should live:

```bash
cp data/data_config.example.yaml data/data_config.yaml
$EDITOR data/data_config.yaml
```

`data_config.yaml` is gitignored. Only the `*.example.yaml` template travels
with the repo.

## Running the pipeline

```bash
python scripts/run_pipeline.py presets/example.yaml
```

The driver reads a YAML preset that captures every per-run knob (cohort size,
template patient ID, stage on/off flags, output root) and runs every enabled
stage end-to-end. Stages run in declared order, each gated by a flag in the
preset's `stages:` block, and each writes to its own subdirectory under the run
dir, named after its key in the preset.

```
ingestion  →  adjusted
     │
     ▼
mesh_extraction  →  ssm_registration  →  ssm_pca  →  radiomics_correlation
     →  ssm_viewer  →  ssm_qa_metrics  →  ssm_qa_residuals  →  visualizations
```

Two stages dominate wall-clock time. `mesh_extraction` is a cohort-wide one-off
(NIfTI to 24 per-rib STLs per patient, tens of minutes for the full cohort);
STLs already on disk are not re-extracted, so it is typically run once per data
drop. `ssm_registration` shells out to the Scala / sbt project at
`environment/scalismo_ssm/` and takes hours on a full cohort, parallel over
patients × 24 ribs. Everything else is fast.

### Shipped presets

* [presets/example.yaml](presets/example.yaml): annotated reference with every
  knob documented inline; all stages enabled.
* [presets/full_analysis.example.yaml](presets/full_analysis.example.yaml):
  full-cohort template. Copy it, then replace the `template_id` and
  `paths.root` placeholders.
* [presets/smoke.yaml](presets/smoke.yaml): `n_patients: 5` with
  `mesh_extraction` off, used by `tests/test_pipeline_smoke.py` and for quick
  post-refactor checks.

Individual stage scripts under `src/<package>/run_*.py` (or
`scripts/run_visualizations.py`) can be invoked directly from the shell against
an existing run dir, which re-runs a single downstream stage without
recomputing upstream. `scripts/run_all_plots.py` re-renders every figure for an
existing run from its cached outputs.

### Output layout

For a preset with `name: foo` and `paths.root: /data/ribs`, the pipeline
auto-derives everything under `/data/ribs/foo/`:

```
/data/ribs/foo/
├── extracted_stl/                            per-rib STLs (mesh-extraction cache)
│   └── patient_ids.txt                       clean (anomaly-free) cohort
├── registered_stl/                           per-rib registered STLs + template_id.txt
└── results/
    ├── foo_<UTC>/                            timestamped run dir
    │   ├── metadata.json                     name, timestamp_utc, git_rev,
    │   │                                     paths.{root, extracted_stl_dir,
    │   │                                            registered_stl_dir,
    │   │                                            results_dir, run_dir}
    │   ├── ingestion/                        parquet + QC + figures
    │   ├── adjusted/                         cluster-robust OLS + FWL + figures
    │   ├── mesh_extraction/                  per-rib audit summary
    │   ├── ssm_registration/                 Scalismo diagnostics (statsOut.json)
    │   ├── ssm_pca/                          GPA + PCA + per-PC regression
    │   ├── radiomics_correlation/            PC × rib × radiomics-feature heatmaps
    │   ├── ssm_viewer/viewer_surface.html    self-contained interactive viewer
    │   ├── ssm_qa_metrics/                   Styner triad + figures/eval_styner.png
    │   ├── ssm_qa_residuals/                 per-vertex residuals + worst-N
    │   └── visualizations/                   supplemental QC figures
    └── foo_latest -> foo_<UTC>               relative symlink to the latest run
```

Per-rib STLs (registration inputs and outputs) live at
`<root>/<preset.name>/extracted_stl/` and
`<root>/<preset.name>/registered_stl/` respectively, outside the results dir so
they are shared across runs of the same preset.

## Pipeline stages

### 1. Data ingestion and QC

Loads per-patient analytic JSONs, joins with the NAKO metadata Excel, applies
the five inclusion criteria (exactly 24 rib sides; complete required metadata;
exactly two connected components per rib label 40–51; no rib touching the image
border; no split-rib flag), then writes a clean rib-level parquet plus QC
tables and figures.

Outputs (under `<run-dir>/ingestion/`):

```
analytic_clean.parquet
join_stats.json
exclusions.csv
rib_count_audit.csv
per_rib_components_audit.csv
per_rib_anomalies.csv
missingness_report.csv
normality_tests.csv
table1_metadata.csv
table1_shape.csv
figures/inclusion_flow.{html,svg,png}
figures/missingness.{html,svg,png}
figures/distribution_hist_qq.{html,svg,png}
figures/normality.{html,svg,png}
figures/corr_shape_params.{html,svg,png}
figures/corr_metadata.{html,svg,png}
```

### 2. Descriptor analysis (unadjusted and adjusted)

Rib-level analyses of each shape descriptor, z-scored within rib identity
(level × side) and pooled across ribs, with patient-clustered (HC1) standard
errors:

* **Unadjusted**: each predictor (sex, age, height, weight, BMI, body fat,
  ever-smoker, pack-years) fit marginally.
* **Adjusted**: one multivariable OLS on the CORE set (sex, age, height,
  weight, body fat, ever-smoker, pack-years; BMI excluded as collinear with
  height + weight), with per-predictor Frisch-Waugh-Lovell partial R².

Smoking is ever/never (pack-years is the within-smoker dose). BH-FDR is applied
within each layer.

Outputs (under `<run-dir>/adjusted/`):

```
descriptor_unadjusted.csv  descriptor_adjusted.csv
figures/adj_heatmap_beta.png   figures/adj_heatmap_partial_r2.png
figures/adj_forest_plots.png   figures/unadj_heatmap_beta.png
```

### 3. Mesh extraction

Loads each patient's multi-label `seg-vert-rib` NIfTI, splits each rib level
into Left / Right via connected-component analysis, runs marching cubes at
`level=0.5`, applies Taubin smoothing, decimates to the preset's
`mesh.target_faces_per_rib` target (falling back to
`settings.MESH_TARGET_FACES_PER_RIB`, currently 1000), and writes 24 STLs per
patient. Taubin alternates shrink and unshrink passes, so it
de-staircases the surface without the volume loss of Laplacian smoothing; the
audit records each mesh's volume against its source voxel volume, which makes
any residual shrinkage measurable.

Outputs (under the preset's `extracted_stl_dir`, that is
`<root>/<name>/extracted_stl/`, passed to the runner as `--stl-dir`):

```
{block}/{patient_id}/{patient_id}_rib{label}_{L|R}.stl
                                      24 STLs per patient,
                                      block = patient_id // 1000
per_rib_audit.csv                     per-(patient, rib) audit, incl. per-side
                                      voxel / mesh volume and volume %
per_rib_volume_summary.csv            mesh volume as % of source voxel volume,
                                      aggregated per rib
patient_ids.txt                       patient IDs processed
template_id.txt                       heuristic default (first patient with all
                                      24 ribs), read by the template_selection
                                      notebook; the load-bearing registration
                                      template is written separately by the
                                      driver (see below)
mesh_extraction_per_rib_log.json      run summary
```

### 4. Template selection

The registration template is the patient ID set in the preset's `template_id`
field. The pipeline driver writes it to `<registered_stl_dir>/template_id.txt`
before the SSM stages run.

The template ID itself was chosen by `nako.ribs.TemplateBuilder`
(`environment/scalismo_ssm/src/main/scala/nako/ribs/TemplateBuilder.scala`),
which picks the *most central* patient from the cohort: on a random subsample,
it computes pairwise `MeshMetrics.avgDistance` for each of the 24 rib
identities, normalises per rib by `1/(n − 1)`, and ranks patients by their
aggregate centrality score. The patient with the lowest aggregate score across
all 24 ribs is written to `template_id.txt`. Patients missing any of the 24 rib
STLs are excluded from the subsample.

`notebooks/template_selection.ipynb` drives this from Python: it shells out to
`runMain nako.ribs.TemplateBuilder --input <STL_PER_RIB> --outTxt
<STL_PER_RIB>/template_id.txt --sample 500 --seed 7`, then renders the chosen
template's 24 ribs in an interactive 3D viewer for visual QC (missing rib,
asymmetry, segmentation artefact). If the suggested template looks wrong, bump
`SEED` and re-run.

### 5. Surface SSM stages

End-to-end SSM pipeline on the per-rib triangle meshes from `mesh_extraction`
and the template selected in the preset. Each stage is gated independently in
the preset's `stages:` block.

1. **Per-rib non-rigid registration** (Scalismo, `nako.ribs.RibRegistration`):
   whole-cage Procrustes pre-alignment on rib centroids, then per (patient,
   rib): orientation-preserving principal-frame alignment (PCA eigenvector
   signs disambiguated by dot product against the template's axes, so posterior
   remains posterior) + similarity ICP (trimmed-mean) + rigid ICP polish +
   4-pass coarse-to-fine non-rigid GP-posterior ICP with three correspondence
   filters per iteration (distance, target-mesh boundary, surface-normal
   cosine). The boundary filter prevents the distal tip from slipping onto the
   shaft when the target is shorter than the template. With
   `ssm.bidirectional: true`, each iteration also adds reverse observations
   (target sampled, projected onto the deformed reference, lifted back via
   barycentric) under distance + normal filters only. Output: 24 registered
   STLs per patient sharing the template's topology.
2. **GPA + PCA + regression** (Python, `src/ssm/run_ssm.py`): concatenate the
   24 ribs into one shape vector per patient, GPA-align (rotation +
   translation, scale retained), fit randomized-SVD PCA keeping the smallest
   *k* explaining ≥ 95% cumulative variance (capped at `min(N − 1, 500)` modes;
   if even the cap falls short of 95%, all fitted modes are kept and a warning
   is logged), then regress each PC score on metadata through the three HC3
   layers (unadjusted, adjusted, targeted) plus a Welch t-test for sex, with
   BH-FDR within each layer.
3. **Interactive viewer** (`src/ssm/viewer.py`): exports a self-contained HTML
   viewer with PC-explorer and metadata-explorer modes, embedding the PCA model
   and regression weights so reconstruction runs entirely client-side in
   JavaScript.
4. **Quality assessment**: Styner triad (compactness, LOO generalisation,
   specificity), per-rib and whole-cage; plus per-vertex residual mosaics for
   the worst N patients.

Outputs are flat per stage under `<run-dir>/`:

```
metadata.json                              run metadata (paths, git rev)
ssm_pca/                                   GPA + PCA + per-PC regression
    shapes_registered.npz, mean_shape_surface.npy, gpa_rms_history.npy,
    meta_sub_surface.parquet, rib_offsets.npy, template_faces.npy,
    pca_surface.npz, pc_scores_surface.csv,
    pc_{unadjusted,adjusted,targeted}.csv, pc_ttest_surface.csv,
    geometry_generator.npz, geometry_generator_holdout.csv
    figures/{gpa_convergence,scree_surface,pc_loadings_per_rib,
             mean_shape_views,pc_deformations_surface,
             pc_regression_surface,pc_beta_vectors,pc_scores_pairs_*}
    (mean_shape_views and pc_deformations_surface are skipped unless
     template_id.txt and all 24 template rib STLs are present)
ssm_viewer/viewer_surface.html             self-contained interactive viewer
ssm_qa_metrics/eval_styner.json            Styner triad
ssm_qa_metrics/figures/eval_styner.png
ssm_qa_residuals/residuals_per_patient.npz  per-vertex registration residuals
ssm_qa_residuals/figures/worst_patients_residuals_*.png
radiomics_correlation/effects.csv          PC × rib × feature regressions
```

## Methodology

Each patient contributes 24 ribs, so rib-level observations are not
independent. Rib-level models are fitted on descriptors z-scored within rib
identity (level × side) and pooled across ribs, with HC1 cluster-robust
standard errors clustered on patient ID. PC-score regressions are patient-level
by construction, one shape vector per patient, and are not clustered.

At NAKO cohort size, statistical significance carries little information on its
own. Every model therefore reports an effect size alongside its p-value:
standardised β, Frisch-Waugh-Lovell partial R² and partial r, and Cohen's d for
the sex contrast.

Benjamini-Hochberg FDR is applied within each structurally distinct test family
(the unadjusted, adjusted and targeted layers, and the Welch t-test) rather than
across all tests pooled. Raw and FDR-adjusted p-values are both tabulated, as is
every contrast tested, regardless of significance.

Runs are scripted, seeded and version-controlled. Raw data is never modified.

## Notes on numerical reproducibility

* All sklearn calls that involve randomness use a fixed seed:
  `PCA_RANDOM_SEED = 42` for the SSM PCA (`src/ssm/pca_surface.py`), and
  `random_state=42` for the LOO / K-fold splits in
  `src/utils/shape_evaluation.py` and the distribution-plot subsampling in
  `src/data_ingestion/qc.py` and `qc_altair.py`.
* The Scalismo registration uses `Random(42)` per-patient (`Random` here is
  Scalismo's wrapper). Rigid-ICP and metric-based fits are deterministic given
  the same input meshes and parameters.
* Pack-years recoding: NAKO sentinel `7775` ("not applicable") is replaced with
  `0` for never-smokers *before* sentinel-to-NaN conversion, so never-smokers
  retain pack-years = 0 while genuine "not assessable" / "not collected"
  sentinels become NaN.

## Abbreviations

| Abbreviation | Expansion |
|---|---|
| BH-FDR     | Benjamini-Hochberg False Discovery Rate |
| BMI        | Body Mass Index (kg / m²) |
| FWL        | Frisch-Waugh-Lovell residualisation theorem |
| GP         | Gaussian Process (Scalismo deformation prior) |
| GPA        | Generalised Procrustes Analysis (translation + rotation, scale retained) |
| HC1        | Heteroskedasticity-Consistent (HC1) sandwich SE estimator |
| ICP        | Iterative Closest Point (rigid registration step) |
| L / R      | Left / Right (rib side) |
| LBFGS      | Limited-memory Broyden-Fletcher-Goldfarb-Shanno optimiser |
| LM         | Linear Model (OLS) |
| LMM        | Linear Mixed Model |
| LOO        | Leave-One-Out (cross-validation) |
| MSE        | Mean Squared Error |
| NAKO       | Nationale Kohorte (German National Cohort study) |
| NIfTI      | Neuroimaging Informatics Technology Initiative (`.nii` volume format) |
| NN         | Nearest Neighbour |
| OLS        | Ordinary Least Squares |
| PC / PCA   | Principal Component / Principal Component Analysis |
| PIR        | Posterior-Inferior-Right (centerline coordinate convention) |
| QC         | Quality Control |
| RMS        | Root Mean Square |
| SD         | Standard Deviation |
| SE         | Standard Error |
| SR / SVR   | Surface-to-Volume Ratio (rib shape feature) |
| SSM        | Statistical Shape Model |
| STL        | Stereolithography (triangle mesh file format) |
| SVD        | Singular Value Decomposition |
| UTC        | Coordinated Universal Time (run-dir timestamps) |
| VIBE       | Volumetric Interpolated Breath-hold Examination (NAKO MRI sequence) |

## Contact

Anatol Aicher
University of Zurich & University Hospital of Zurich
anatol.aicher@uzh.ch
