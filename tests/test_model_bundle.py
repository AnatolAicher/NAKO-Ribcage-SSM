"""The released model under model/ loads with NumPy alone and is self-consistent."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"


@pytest.fixture(scope="module")
def bundle():
    spec = importlib.util.spec_from_file_location("ribcage_ssm", MODEL_DIR / "ribcage_ssm.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ribcage_ssm"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ssm(bundle):
    return bundle.RibCageSSM.load(MODEL_DIR)


def test_bundle_shapes(ssm):
    assert ssm.mean.shape == (ssm.n_vertices, 3)
    assert ssm.components.shape == (ssm.n_modes, 3 * ssm.n_vertices)
    assert len(ssm.rib_offsets) == 24 and ssm.rib_offsets[0] == 0
    assert ssm.faces.min() == 0 and ssm.faces.max() == ssm.n_vertices - 1
    assert ssm.generator_B.shape == (ssm.n_modes, len(ssm.predictors))
    assert set(ssm.predictors) == {"is_female", "age", "height_cm", "weight_kg",
                                   "body_fat_pct", "ever_smoker", "pack_years"}


def test_modes_orthonormal(ssm):
    np.testing.assert_allclose(ssm.components @ ssm.components.T, np.eye(ssm.n_modes), atol=1e-4)
    assert 0.95 <= ssm.explained_variance_ratio.sum() <= 1.0


def test_reconstruction(ssm):
    np.testing.assert_allclose(ssm.reconstruct([]), ssm.mean, atol=1e-6)
    cage = ssm.deform(mode=1, n_sigma=2.0)
    assert np.linalg.norm(cage - ssm.mean) == pytest.approx(2.0 * ssm.sigma[0], rel=1e-5)
    with pytest.raises(ValueError):
        ssm.reconstruct(np.zeros(ssm.n_modes + 1))


def test_ribs_partition_surface(bundle, ssm):
    rib_of = [ssm.rib_of_vertex(ssm.faces[:, col]) for col in range(3)]
    assert (rib_of[0] == rib_of[1]).all() and (rib_of[0] == rib_of[2]).all()
    assert set(rib_of[0]) == set(range(24))
    verts, faces = ssm.rib(ssm.mean, "7R")
    i = bundle.RIB_LABELS.index("7R")
    assert len(verts) == ssm.rib_offsets[i + 1] - ssm.rib_offsets[i]
    assert faces.min() == 0 and faces.max() == len(verts) - 1
    np.testing.assert_array_equal(ssm.rib(ssm.mean, i)[1], faces)


def test_generator(ssm):
    cohort = dict(zip(ssm.predictors, ssm.generator_offsets.tolist()))
    np.testing.assert_allclose(ssm.predict_scores(**cohort), ssm.generator_intercept)
    female = ssm.predict_shape(**{**cohort, "is_female": 1.0})
    male = ssm.predict_shape(**{**cohort, "is_female": 0.0})
    assert np.linalg.norm(female - male) > 0
    with pytest.raises(ValueError):
        ssm.predict_scores(age=40.0)


def test_stl_roundtrip(bundle, ssm, tmp_path):
    out = tmp_path / "mean.stl"
    ssm.write_stl(ssm.mean, out)
    data = out.read_bytes()
    assert len(data) == 84 + 50 * len(ssm.faces)
    rec = np.frombuffer(data[84:], dtype=bundle._STL_DTYPE)
    np.testing.assert_allclose(rec["vertices"], ssm.mean[ssm.faces], atol=1e-6)


def test_cli(bundle, tmp_path, capsys):
    bundle.main(["--model-dir", str(MODEL_DIR)])
    assert "modes" in capsys.readouterr().out
    out = tmp_path / "cage.stl"
    bundle.main(["--model-dir", str(MODEL_DIR), "--sex", "female", "--age", "45", "--height", "165",
                 "--weight", "68", "--body-fat", "32", "--out", str(out)])
    assert out.stat().st_size == 84 + 50 * len(np.load(MODEL_DIR / "template_faces.npy"))
