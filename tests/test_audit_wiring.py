"""End-to-end wiring of the audit pieces on small synthetic data (not research results)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from neurotext.data import make_synthetic
from neurotext.features import feature_table
from neurotext.leakage import run_identity_seed
from neurotext.models import ENCODERS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    c = yaml.safe_load((ROOT / "configs" / "audit_synthetic.yaml").read_text())
    c["train"]["epochs"] = 2
    return c


@pytest.fixture(scope="module")
def data(cfg):
    return make_synthetic(**{k: v for k, v in cfg["data"].items() if k != "source"})


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_identity_seed_runs_for_every_encoder_with_verification(cfg, data, name):
    c = {**cfg, "model": {**cfg["model"], "name": name}}
    ft = feature_table(data.x, 100.0)
    r = run_identity_seed(data, ft.x, c, seed=0, probes=("linear",), verify=True, feature_reps=False)
    reps = r["identity"]["new_users"]
    assert set(reps) == {"trained encoder", "untrained encoder"}
    v = reps["trained encoder"]["verify"]
    assert 0.0 <= v["cosine"]["eer"] <= 1.0 and v["lda"] is not None


def test_defence_seed_accepts_fixed_test_people_and_reports_verification(cfg):
    from neurotext.defence import run_defence_seed

    c = {**cfg, "split": {**cfg["split"], "test_subjects": [0, 1, 2, 3]}}
    r = run_defence_seed(c, seed=0, probes=("linear",), verify=True)
    assert r["n_new_users"] == 4
    assert set(r["defences"]) >= {"none", "noise s=0.5", "pca k=4", "id-null m=4"}
    assert r["defences"]["none"]["identity"]["verify"]["cosine"] is not None


def test_scaling_run_seed_and_summary(cfg, data):
    sc = _load_script("scaling_curve")
    feats = feature_table(data.x, 100.0).x
    run = sc.run_seed(cfg, data, feats, seed=0, n_list=[3, 6], n_train=10, probes=("linear",), verify=True, final=False)
    assert run["n_train_people"] == 10 and set(run["results"]) == {"3", "6"}
    assert run["results"]["3"]["trained encoder"]["linear"]["chance"] == pytest.approx(1 / 3)
    summ = sc.summarise([run, run], ["linear"], [3, 6])
    assert summ["n_list"] == [3, 6] and summ["reps"]["trained encoder"]["linear"]["6"]["accuracy"] is not None


def test_scaling_final_mode_uses_features_only(cfg, data):
    sc = _load_script("scaling_curve")
    feats = feature_table(data.x, 100.0).x
    run = sc.run_seed(cfg, data, feats, seed=0, n_list=[5, 30], n_train=0, probes=("linear",), verify=True, final=True)
    assert set(run["results"]["5"]) == {"band power (all features)"} and run["n_train_people"] == 0
    assert run["results"]["30"]["band power (all features)"]["verify"]["lda"] is None  # nobody left to fit the LDA on


def test_a_pre_registered_run_refuses_to_overwrite_itself(tmp_path):
    from neurotext.audit import guard_single_run

    out = tmp_path / "final.json"
    guard_single_run(out, None)                     # first run: fine
    out.write_text("{}")
    with pytest.raises(SystemExit):
        guard_single_run(out, None)
    guard_single_run(out, "disk full")              # technical rerun: allowed, and logged
    assert "disk full" in (tmp_path / "deviations.log").read_text()
