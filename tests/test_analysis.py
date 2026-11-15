"""Tests for the core analysis code (synthetic data only; no repository data needed).

Run from the repo root with either:
    python -m pytest tests -v
    python tests/test_analysis.py          (works without pytest)
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import analysis
import sensitivity_pipelines as sp


def _labels(n_pos, n_neg):
    return np.array([1] * n_pos + [0] * n_neg)


def test_rank_auc_matches_sklearn_including_ties():
    rng = np.random.default_rng(0)
    X = np.round(rng.normal(size=(30, 8)), 1)          # rounding creates ties
    y = _labels(9, 21)
    fast = analysis._rank_auc(X, y)
    slow = [roc_auc_score(y, X[:, j]) for j in range(X.shape[1])]
    assert np.allclose(fast, slow)


def test_pure_noise_nested_cv_is_near_chance_but_apparent_is_not():
    pooled, apparent = [], []
    for s in range(12):
        rng = np.random.default_rng(100 + s)
        X = rng.normal(size=(40, 400))
        y = _labels(8, 32)
        pooled.append(analysis.nested_cv(X, y, n_genes=10, random_state=s)[1])
        apparent.append(analysis.apparent_auc(X, y, n_genes=10)[0])
    assert abs(np.mean(pooled) - 0.5) < 0.07, np.mean(pooled)
    assert np.mean(apparent) > 0.85, np.mean(apparent)


def _record_training_ids(run):
    """Run `run()` while recording which sample ids reach gene screening each time.
    Column 0 of X must hold unique sample ids."""
    calls = []
    original = analysis.select_top_genes

    def spy(X, *args, **kwargs):
        calls.append(set(X[:, 0].astype(int)))
        return original(X, *args, **kwargs)

    analysis.select_top_genes = spy
    try:
        run()
    finally:
        analysis.select_top_genes = original
    return calls


def _assert_each_sample_held_out_once(calls, n):
    assert len(calls) == 5
    all_ids = set(range(n))
    held_out = [all_ids - c for c in calls]
    assert all(len(h) > 0 for h in held_out)
    assert set().union(*held_out) == all_ids                 # everyone is tested somewhere
    assert sum(len(h) for h in held_out) == n                # and in exactly one fold


def test_screening_never_sees_the_samples_it_is_tested_on():
    n = 40
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, 60))
    X[:, 0] = np.arange(n)                                   # sample ids
    y = _labels(8, 32)
    calls = _record_training_ids(lambda: analysis.nested_cv(X, y, n_genes=5))
    _assert_each_sample_held_out_once(calls, n)
    calls = _record_training_ids(
        lambda: sp.nested_auc(sp.make_screen_pipeline(5), X, y, 5, 42))
    _assert_each_sample_held_out_once(calls, n)


def test_real_signal_is_detected_by_nested_cv():
    pooled = []
    for s in range(5):
        rng = np.random.default_rng(200 + s)
        X = rng.normal(size=(60, 300))
        y = _labels(12, 48)
        X[y == 1, :3] += 1.5                                  # 3 informative genes
        pooled.append(analysis.nested_cv(X, y, n_genes=10, random_state=s)[1])
    assert np.mean(pooled) > 0.75, np.mean(pooled)


def test_gene_selection_does_not_depend_on_column_order():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 120))
    y = _labels(10, 30)
    perm = rng.permutation(X.shape[1])
    top, _ = analysis.select_top_genes(X, y, n_top=10)
    top_perm, _ = analysis.select_top_genes(X[:, perm], y, n_top=10)
    assert set(top) == set(perm[top_perm])


def test_baseline_sensitivity_pipeline_reproduces_main_analysis():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(45, 200))
    y = _labels(9, 36)
    ref_mean, ref_pooled, _ = analysis.nested_cv(X, y, n_genes=10, random_state=42)
    ref_apparent, _ = analysis.apparent_auc(X, y, n_genes=10)
    base = sp.CONFIGS["screen_10_baseline"]
    mean_b, pooled_b = sp.nested_auc(base, X, y, 5, 42)
    assert abs(mean_b - ref_mean) < 1e-12
    assert abs(pooled_b - ref_pooled) < 1e-12
    assert abs(sp.apparent_auc(base, X, y, 42) - ref_apparent) < 1e-12


def test_ridge_pipeline_returns_probabilities_and_detects_signal():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 200))
    y = _labels(15, 45)
    X[y == 1, :5] += 1.5
    pipe = sp.CONFIGS["ridge_tuned"]
    probs = pipe(X[:45], y[:45], X[45:], 42)
    assert probs.shape == (15,) and ((probs >= 0) & (probs <= 1)).all()
    assert sp.nested_auc(pipe, X, y, 5, 42)[1] > 0.7


def test_prefilter_uses_training_rows_only():
    n = 40
    rng = np.random.default_rng(6)
    X = rng.normal(size=(n, 60))
    X[:, 0] = np.arange(n)
    y = _labels(8, 32)
    pipe = sp.make_screen_pipeline(5, prefilter="top_half_mean")
    calls = _record_training_ids(lambda: sp.nested_auc(pipe, X, y, 5, 42))
    _assert_each_sample_held_out_once(calls, n)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {error!r}")
    print(f"\n{len(tests) - failed} of {len(tests)} tests passed")
    sys.exit(1 if failed else 0)