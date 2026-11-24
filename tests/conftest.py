"""Tests run on the source tree (src/ on the path) and never touch the user's settings: the user folders (modules,
trust, settings) point to a temporary folder, and Qt runs offscreen."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def private_user_folders(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("XDG_DATA_HOME", str(home / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "config"))
    monkeypatch.setenv("TESSELAB_MODULE_PATH", "")
    yield


@pytest.fixture
def project(tmp_path):
    """A project with a table of three Gaussian components mixed over 20 rows (data/mix.csv)."""
    from tesselab.project import Project
    root = tmp_path / "proj"
    (root / "data").mkdir(parents=True)
    x = np.linspace(200, 400, 101)
    spectra = np.array([np.exp(-((x - c) / 20) ** 2) for c in (250, 300, 350)])
    t = np.linspace(0, 1, 20)
    weights = np.array([np.exp(-3 * t), 1 - np.exp(-3 * t), t ** 2])
    matrix = weights.T @ spectra  # 20 rows × 101 columns
    rng = np.random.default_rng(0)
    matrix = matrix + rng.normal(0, 1e-3, matrix.shape)
    with open(root / "data" / "mix.csv", "w") as f:
        f.write("wavelength," + ",".join(f"t{i}" for i in range(len(t))) + "\n")
        for i, xi in enumerate(x):
            f.write(f"{xi}," + ",".join(f"{v:.8f}" for v in matrix[:, i]) + "\n")
    return Project.create(root)
