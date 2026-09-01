# NAKO rib-cage statistical shape model

Population-level statistical shape model of the complete 24-rib human rib cage,
built from 26,275 whole-body MRI scans of adults aged 19–74 years in the German
National Cohort (NAKO). This directory is the complete released model; it
contains no participant-level data.

Provenance: pipeline run `full_analysis-ab26a74_20260809T225556Z`, commit
`ab26a74` of
<https://github.com/AnatolAicher/NAKO-Ribcage-SSM>.
Files marked *run output* are byte-identical copies of that run's `ssm_pca/`
artefacts.

## Files

| File | Arrays (dtype, shape) | Origin |
|---|---|---|
| `pca_surface.npz` | `mean` (float32, 36141); `components` (float32, 28 × 36141); `explained_variance` (float32, 28); `explained_variance_ratio` (float32, 28); `variance_threshold` (float32 scalar, 0.95) | run output |
| `template_faces.npy` | int32, 23998 × 3 | run output |
| `rib_offsets.npy` | int64, 24 | run output |
| `mean_shape_surface.npy` | float32, 12047 × 3 – the Procrustes mean; equals `mean` reshaped, up to float32 rounding (≤ 0.002 mm) | run output |
| `geometry_generator.npz` | `B` (float64, 28 × 7); `intercept` (float64, 28); `offsets` (float64, 7); `predictors` (str, 7); `pc_cols` (str, 28); `r2` (float64, 28); `n` (int64 scalar, 26275) | run output, re-saved with the string arrays as plain unicode so it loads without `allow_pickle`; numeric arrays identical |
| `mean_shape.stl` | binary STL, 23998 triangles | derived from `mean` and `template_faces` by `ribcage_ssm.py` |
| `ribcage_ssm.py` | NumPy-only loader: reconstruction, generator, rib slicing, STL export, command line | – |

## Conventions

- **Units and frame.** Millimetres in the cohort's generalised-Procrustes frame:
  translation and rotation removed, scale retained, so surfaces carry real
  size. Axes are approximately anatomical (x left→right, y posterior→anterior,
  z inferior→superior) with the origin at the cage centroid; the mean shape
  spans 297 × 187 × 322 mm.
- **Vertex layout.** A shape is a vector of 36,141 = 12,047 × 3 values in
  row-major order (x₀ y₀ z₀ x₁ …). Vertices are grouped by rib in the order
  rib 1 left, rib 1 right, rib 2 left, …, rib 12 right (anatomical numbering;
  left and right are the participant's sides). `rib_offsets[i]` is the first
  vertex of rib `i`; each rib has about 502 vertices and 1,000 faces, and no
  face spans two ribs.
- **Correspondence.** Vertex `v` marks the same anatomical location on every
  surface (template topology), so surfaces can be compared, averaged and
  interpolated vertex by vertex.
- **Modes.** The rows of `components` are orthonormal and expressed around
  `mean`. The cohort's scores along mode `k` are centred with variance
  `explained_variance[k]` (mm²); `explained_variance_ratio` sums to 0.951 over
  the 28 retained modes.

## Formulae

Reconstruction from a score vector `s` (any length ≤ 28; trailing modes are
zero):

    shape = mean + s @ components[:len(s)]        # (36141,) → reshape(-1, 3)

A "±2 SD" deformation of mode `k` is `s[k] = ±2·√explained_variance[k]` with
every other entry zero.

Demographic generator, with `x` ordered as `predictors` = (`is_female`, `age`,
`height_cm`, `weight_kg`, `body_fat_pct`, `ever_smoker`, `pack_years`):

    s = intercept + B @ (x − offsets)

`is_female` and `ever_smoker` are 0/1 and uncentred (their `offsets` are 0);
the continuous `offsets` are the cohort means (age 48.0 years, height
173.2 cm, body mass 79.8 kg, body fat 29.9 %, pack-years 6.4). To average over
sex or smoking status, pass the cohort fractions 0.439 (female) and 0.497
(ever-smoker). `pack_years` is cumulative smoking exposure and 0 for
never-smokers. `r2` is the in-sample R² per mode (PC1 0.78, PC2 0.62, then
≤ 0.07). In 10-fold cross-validation, the per-vertex distance between a
participant's demographics-predicted surface and the reconstruction of their
own scores is 8.8 mm on average (95th percentile 12.9 mm).

## Usage

```python
from ribcage_ssm import RibCageSSM       # run inside this directory, or put it on sys.path

ssm = RibCageSSM.load()
cage = ssm.deform(mode=1, n_sigma=2)     # mean shape pushed +2 SD along PC1 → (12047, 3) in mm
cage = ssm.predict_shape(is_female=1, age=45, height_cm=165, weight_kg=68,
                         body_fat_pct=32, ever_smoker=0, pack_years=0)
verts, faces = ssm.rib(cage, "7R")       # anatomical rib 7, right side, as its own mesh
ssm.write_stl(cage, "cage.stl")
```

```bash
python ribcage_ssm.py                                   # model summary
python ribcage_ssm.py --pc 1 2 --out pc1_plus2sd.stl    # ±n SD along any mode(s), repeatable
python ribcage_ssm.py --sex female --age 45 --height 165 --weight 68 --body-fat 32 --out cage.stl
python ribcage_ssm.py --rib 7R --out rib7R_mean.stl     # one rib of the mean shape
```

Any other language: load the arrays with a NumPy-format reader, apply the two
formulae above, and use `template_faces` for connectivity.

## Citation, licence, acknowledgement

Cite the manuscript – Aicher A *et al.*, *How Sex, Age, Adiposity, and Smoking
Shape the Human Rib Cage: Evidence from 26,275 Whole-Body MRIs across the
German National Cohort (NAKO)* – and the archived release, DOI
[10.5281/zenodo.22230693](https://doi.org/10.5281/zenodo.22230693) (all versions;
`CITATION.cff` in the repository root carries the metadata). Released under the
MIT License.

We thank all participants of the German National Cohort (NAKO) and the staff
of this research initiative. Members and affiliations of the NAKO Investigator
Consortium can be accessed via www.nako.de/principal-investigators.
