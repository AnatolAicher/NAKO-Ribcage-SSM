"""NumPy-only loader for the NAKO rib-cage statistical shape model.

Reads the model files next to this module (or from a given directory) and
exposes reconstruction from principal-component scores, prediction from a
demographic profile, per-rib slicing and STL export.

Python:
    from ribcage_ssm import RibCageSSM
    ssm = RibCageSSM.load()
    cage = ssm.deform(mode=1, n_sigma=2.0)                  # (12047, 3) mm
    cage = ssm.predict_shape(is_female=1, age=45, height_cm=165, weight_kg=68,
                             body_fat_pct=32, ever_smoker=0, pack_years=0)
    verts, faces = ssm.rib(cage, "7R")
    ssm.write_stl(cage, "cage.stl")

Command line:
    python ribcage_ssm.py                                   # model summary
    python ribcage_ssm.py --pc 1 2 --out pc1_plus2sd.stl
    python ribcage_ssm.py --sex female --age 45 --height 165 --weight 68 \
        --body-fat 32 --out cage.stl
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Vertex blocks of the concatenated shape vector: rib 1 left, rib 1 right, …, rib 12 right.
RIB_LABELS: tuple[str, ...] = tuple(f"{rib}{side}" for rib in range(1, 13) for side in "LR")

# Cohort-marginal values of the binary predictors (fraction of the 26,275 modelled
# participants); pass them to average a prediction over sex or smoking status.
COHORT_FEMALE_FRACTION: float = 0.4394
COHORT_EVER_SMOKER_FRACTION: float = 0.4972

_STL_DTYPE = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])


@dataclass(frozen=True)
class RibCageSSM:
    """Mean shape, principal modes and demographic generator of the rib-cage SSM."""

    mean: np.ndarray                      # (V, 3) mm, Procrustes frame
    faces: np.ndarray                     # (F, 3) vertex indices
    rib_offsets: np.ndarray               # (24,) first vertex of each rib, RIB_LABELS order
    components: np.ndarray                # (K, 3V) orthonormal modes
    explained_variance: np.ndarray        # (K,) score variance per mode (mm²)
    explained_variance_ratio: np.ndarray  # (K,) fraction of total variance
    generator_B: np.ndarray               # (K, P)
    generator_intercept: np.ndarray       # (K,)
    generator_offsets: np.ndarray         # (P,) centring offsets (0 for binaries)
    predictors: tuple[str, ...]           # (P,) design column order

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "RibCageSSM":
        d = Path(directory) if directory is not None else Path(__file__).resolve().parent
        pca = np.load(d / "pca_surface.npz")
        gen = np.load(d / "geometry_generator.npz")
        return cls(
            mean=pca["mean"].reshape(-1, 3),
            faces=np.load(d / "template_faces.npy"),
            rib_offsets=np.load(d / "rib_offsets.npy"),
            components=pca["components"],
            explained_variance=pca["explained_variance"],
            explained_variance_ratio=pca["explained_variance_ratio"],
            generator_B=gen["B"],
            generator_intercept=gen["intercept"],
            generator_offsets=gen["offsets"],
            predictors=tuple(str(p) for p in gen["predictors"]),
        )

    @property
    def n_vertices(self) -> int:
        return int(self.mean.shape[0])

    @property
    def n_modes(self) -> int:
        return int(self.components.shape[0])

    @property
    def sigma(self) -> np.ndarray:
        """Standard deviation of the cohort scores along each mode (mm)."""
        return np.sqrt(self.explained_variance)

    def reconstruct(self, scores) -> np.ndarray:
        """Surface (V, 3) for a vector of PC scores; a short vector uses the leading modes."""
        s = np.asarray(scores, dtype=np.float64).ravel()
        if s.size > self.n_modes:
            raise ValueError(f"{s.size} scores given, model has {self.n_modes} modes")
        flat = self.mean.ravel().astype(np.float64) + s @ self.components[: s.size]
        return flat.reshape(-1, 3)

    def deform(self, mode: int, n_sigma: float) -> np.ndarray:
        """Mean shape displaced ``n_sigma`` standard deviations along one mode (1-based)."""
        if not 1 <= mode <= self.n_modes:
            raise ValueError(f"mode must be in 1..{self.n_modes}")
        scores = np.zeros(mode)
        scores[mode - 1] = n_sigma * self.sigma[mode - 1]
        return self.reconstruct(scores)

    def predict_scores(self, **profile: float) -> np.ndarray:
        """PC scores for a demographic profile; every predictor must be given."""
        missing = [p for p in self.predictors if p not in profile]
        unknown = [k for k in profile if k not in self.predictors]
        if missing or unknown:
            raise ValueError(
                f"predictors are {self.predictors}; missing {missing}, unknown {unknown}"
            )
        x = np.array([float(profile[p]) for p in self.predictors])
        return self.generator_intercept + self.generator_B @ (x - self.generator_offsets)

    def predict_shape(self, **profile: float) -> np.ndarray:
        """Surface (V, 3) predicted from a demographic profile."""
        return self.reconstruct(self.predict_scores(**profile))

    def rib(self, vertices, label: str | int) -> tuple[np.ndarray, np.ndarray]:
        """One rib of a surface as ``(vertices, faces)`` with rib-local face indices.

        ``label`` is a RIB_LABELS entry such as ``"7R"`` or a 0-based index.
        """
        i = RIB_LABELS.index(label.upper()) if isinstance(label, str) else int(label)
        bounds = np.append(self.rib_offsets, self.n_vertices)
        start, end = int(bounds[i]), int(bounds[i + 1])
        keep = (self.faces[:, 0] >= start) & (self.faces[:, 0] < end)
        return np.asarray(vertices)[start:end], self.faces[keep] - start

    def rib_of_vertex(self, index) -> np.ndarray:
        """RIB_LABELS index for each vertex index."""
        return np.searchsorted(self.rib_offsets, np.asarray(index), side="right") - 1

    def write_stl(self, vertices, path: str | Path, faces=None) -> None:
        """Write a surface as binary STL (``faces`` defaults to the full-cage connectivity)."""
        v = np.asarray(vertices, dtype=np.float32)
        f = self.faces if faces is None else np.asarray(faces)
        tri = v[f]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        rec = np.empty(len(f), dtype=_STL_DTYPE)
        rec["normal"], rec["vertices"], rec["attr"] = normal, tri, 0
        with open(path, "wb") as fh:
            fh.write(b"NAKO rib cage SSM".ljust(80, b"\0"))
            fh.write(np.array(len(f), dtype="<u4").tobytes())
            fh.write(rec.tobytes())


def summary(ssm: RibCageSSM) -> str:
    evr = ssm.explained_variance_ratio
    lines = [
        f"NAKO rib-cage SSM: {ssm.n_vertices} vertices, {len(ssm.faces)} faces, "
        f"{len(ssm.rib_offsets)} ribs, {ssm.n_modes} modes ({evr.sum():.1%} of variance)",
        "mode  sigma_mm  variance",
        *(f"{k + 1:4d}  {s:8.1f}  {r:8.1%}" for k, (s, r) in enumerate(zip(ssm.sigma, evr))),
        "generator predictors: " + ", ".join(ssm.predictors),
    ]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Reconstruct rib-cage surfaces from the NAKO shape model and write them as STL."
    )
    ap.add_argument("--model-dir", type=Path, default=None, help="directory with the model files")
    ap.add_argument("--out", type=Path, help="STL file to write; omit to print the model summary")
    ap.add_argument("--pc", nargs=2, action="append", type=float, metavar=("MODE", "N_SIGMA"),
                    help="displace the mean along MODE by N_SIGMA standard deviations (repeatable)")
    g = ap.add_argument_group("demographic profile (age, height, weight and body-fat together)")
    g.add_argument("--sex", choices=("female", "male", "cohort"), default="cohort")
    g.add_argument("--age", type=float, help="years")
    g.add_argument("--height", type=float, help="cm")
    g.add_argument("--weight", type=float, help="kg")
    g.add_argument("--body-fat", type=float, help="percent")
    g.add_argument("--smoking", choices=("never", "ever", "cohort"), default="cohort")
    g.add_argument("--pack-years", type=float,
                   help="cumulative exposure; default 0 for never-smokers, cohort mean otherwise")
    ap.add_argument("--rib", help="export a single rib, e.g. 7R")
    a = ap.parse_args(argv)

    ssm = RibCageSSM.load(a.model_dir)
    if a.out is None:
        print(summary(ssm))
        return

    profile = (a.age, a.height, a.weight, a.body_fat)
    if a.pc and any(x is not None for x in profile):
        ap.error("use either --pc or a demographic profile")
    if a.pc:
        scores = np.zeros(ssm.n_modes)
        for mode, n_sigma in a.pc:
            if not 1 <= int(mode) <= ssm.n_modes:
                ap.error(f"--pc MODE must be in 1..{ssm.n_modes}")
            scores[int(mode) - 1] += n_sigma * ssm.sigma[int(mode) - 1]
        cage = ssm.reconstruct(scores)
    elif all(x is not None for x in profile):
        is_female = {"female": 1.0, "male": 0.0, "cohort": COHORT_FEMALE_FRACTION}[a.sex]
        ever = {"never": 0.0, "ever": 1.0, "cohort": COHORT_EVER_SMOKER_FRACTION}[a.smoking]
        if a.pack_years is not None:
            pack_years = a.pack_years
        elif a.smoking == "never":
            pack_years = 0.0
        else:
            pack_years = float(ssm.generator_offsets[ssm.predictors.index("pack_years")])
        cage = ssm.predict_shape(is_female=is_female, age=a.age, height_cm=a.height,
                                 weight_kg=a.weight, body_fat_pct=a.body_fat,
                                 ever_smoker=ever, pack_years=pack_years)
    elif any(x is not None for x in profile):
        ap.error("a demographic profile needs --age, --height, --weight and --body-fat")
    else:
        cage = ssm.mean

    if a.rib:
        verts, faces = ssm.rib(cage, a.rib)
        ssm.write_stl(verts, a.out, faces=faces)
    else:
        ssm.write_stl(cage, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
