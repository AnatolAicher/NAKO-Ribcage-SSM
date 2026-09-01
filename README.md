[![DOI](https://zenodo.org/badge/1240422816.svg)](https://doi.org/10.5281/zenodo.22230693)

# NAKO Human Ribcage Statistical Shape Model

A population-level statistical shape model (SSM) of the complete 24-rib human
rib cage, built from 26,275 whole-body MRI scans of adults aged 19–74 years in
the German National Cohort (NAKO). The released model – mean shape, 28
principal modes covering 95 % of the shape variance, and a demographics→shape
generator – is in [`model/`](model/); the pipeline that produced it is in
[`src/`](src/) and [`scripts/`](scripts/).

- Manuscript: Aicher A *et al.*, *How Sex, Age, Adiposity, and Smoking Shape the
  Human Rib Cage: Evidence from 26,275 Whole-Body MRIs across the German
  National Cohort (NAKO)* – preprint DOI to follow.
- Interactive manuscript, supplement and shape-model viewer: <https://anatolaicher.github.io/NAKO-Ribcage-Manuscript/>
- Archived releases on Zenodo: DOI [10.5281/zenodo.22230693](https://doi.org/10.5281/zenodo.22230693)
  (resolves to the latest version).

## Get the model

Download [`model/`](model/) (about 6 MB) or the
[Zenodo archive](https://doi.org/10.5281/zenodo.22230693). The only dependency
is NumPy.

| File | Content |
|---|---|
| `pca_surface.npz` | `mean` (36,141 = 12,047 vertices × 3), `components` (28 × 36,141, orthonormal rows), `explained_variance`, `explained_variance_ratio` |
| `template_faces.npy` | 23,998 × 3 triangle indices shared by every surface |
| `rib_offsets.npy` | first vertex of each of the 24 ribs (rib 1 left, rib 1 right, …, rib 12 right) |
| `geometry_generator.npz` | `B` (28 × 7), `intercept`, `offsets`, `predictors` – demographics → PC scores |
| `mean_shape_surface.npy` | the mean shape as 12,047 × 3 |
| `mean_shape.stl` | the mean shape as binary STL for any mesh tool |
| `ribcage_ssm.py` | dependency-free loader with the operations below, plus a command line |
| `README.md` | full format description, conventions and formulae |

```python
from ribcage_ssm import RibCageSSM       # run inside model/, or put model/ on sys.path

ssm = RibCageSSM.load()
cage = ssm.deform(mode=1, n_sigma=2)     # mean shape pushed +2 SD along PC1 → (12047, 3) in mm
cage = ssm.predict_shape(is_female=1, age=45, height_cm=165, weight_kg=68,
                         body_fat_pct=32, ever_smoker=0, pack_years=0)   # a demographic profile
verts, faces = ssm.rib(cage, "7R")       # anatomical rib 7, right side, as its own mesh
ssm.write_stl(cage, "cage.stl")
```

The same from the shell:

```bash
python model/ribcage_ssm.py                                   # model summary
python model/ribcage_ssm.py --pc 1 2 --out pc1_plus2sd.stl    # ±n SD along any mode(s)
python model/ribcage_ssm.py --sex female --age 45 --height 165 --weight 68 --body-fat 32 --out cage.stl
```

Conventions in brief: coordinates are millimetres in the cohort's Procrustes
frame, with scale retained so surfaces carry real size and axes approximately
anatomical (x left→right, y posterior→anterior, z inferior→superior); every
surface has the same 12,047 vertices in correspondence, about 502 per rib and
1,000 faces per rib; a surface is `mean + Σₖ sₖ·componentₖ`, where the cohort
scores `sₖ` have standard deviation `√explained_variance[k]` (PC1 ≈ 1,030 mm);
the generator predicts scores as `intercept + B·(x − offsets)` for `x =
(is_female, age, height_cm, weight_kg, body_fat_pct, ever_smoker,
pack_years)`, with a held-out per-vertex error of 8.8 mm (mean) and 12.9 mm
(95th percentile). Details are in [`model/README.md`](model/README.md).

The model contains no participant-level data. The interactive viewer on the
site above runs the same model in the browser.

## How the model was built

Ribs were segmented on NAKO VIBE whole-body MRI with a rib-extended SPINEPS
model, extracted as 24 per-rib surface meshes (marching cubes, Taubin
smoothing, quadric decimation), brought into dense vertex-wise correspondence
by Gaussian-process registration to a most-central template in Scalismo,
aligned by generalised Procrustes analysis (translation and rotation removed,
scale retained) and summarised by PCA. PC scores were regressed on sex, age,
height, body mass, body-fat percentage, ever-smoking and pack-years to obtain
the generator; fourteen per-rib PyRadiomics descriptors provide the anatomical
cross-walk reported in the manuscript. The stage-by-stage description is in
[`docs/pipeline.md`](docs/pipeline.md).

## Reproducing the analysis

The manuscript's numbers and figures come from run
`full_analysis-ab26a74_20260809T225556Z`, produced by commit `ab26a74` of this
repository with a preset derived from
[`presets/full_analysis.example.yaml`](presets/full_analysis.example.yaml).
[`docs/pipeline.md`](docs/pipeline.md) covers environment setup (Python 3.13,
Scala / Scalismo), data configuration, presets, every pipeline stage and its
outputs, and the numerical-reproducibility notes. NAKO participant-level data
cannot be redistributed; access is granted via
<https://www.nako.de/transferhub>.

## Citing

Please cite the manuscript (reference above) and the archived release;
[`CITATION.cff`](CITATION.cff) carries the metadata, and GitHub's
"Cite this repository" button renders it.

## License and acknowledgement

Code and model are released under the [MIT License](LICENSE).

We thank all participants of the German National Cohort (NAKO) and the staff
of this research initiative. Members and affiliations of the NAKO Investigator
Consortium can be accessed via
[www.nako.de/principal-investigators](https://www.nako.de/principal-investigators).

## Contact

Anatol Aicher · University of Zurich & University Hospital Zurich ·
anatol.aicher@uzh.ch
