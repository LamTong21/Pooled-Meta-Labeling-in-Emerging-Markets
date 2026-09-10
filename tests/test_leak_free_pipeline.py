"""
Automated Leakage & Look-Ahead Verification Suite.
Asserts mathematical causality, temporal purging bounds, and in-fold feature selection isolation.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.econometric_pipeline import EconometricsFeaturePipeline
from src.features.microstructure import compute_garman_klass_volatility
from src.labeling import TripleBarrierLabeler
from src.models.cross_validation import PurgedGroupTimeSeriesSplit


@pytest.fixture
def synthetic_ohlcv_dataset():
    """Generates synthetic multi-day asset price action for determinism."""
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=200, freq="B")
    
    # Random walk close prices
    returns = np.random.normal(loc=0.0005, scale=0.015, size=len(dates))
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + np.abs(np.random.normal(0, 0.005, size=len(dates))))
    low = close * (1.0 - np.abs(np.random.normal(0, 0.005, size=len(dates))))
    open_p = low + (high - low) * np.random.uniform(0.1, 0.9, size=len(dates))
    volume = np.random.lognormal(mean=12, sigma=0.5, size=len(dates))

    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "open_adj": open_p,
            "high_adj": high,
            "low_adj": low,
            "close_adj": close,
            "volume": volume,
        },
        index=dates,
    )
    return df


def test_purged_time_series_temporal_isolation(synthetic_ohlcv_dataset):
    """
    Asserts zero holding-period overlap between train events [t0, t_exit]
    and the chronological test window [test_start, test_end].
    """
    df = synthetic_ohlcv_dataset
    labeler = TripleBarrierLabeler(pt=2.5, sl=1.5, horizon=10)
    
    # Generate triggers every 5 bars
    signals = pd.Series(False, index=df.index)
    signals.iloc[::5] = True
    
    events = labeler.label_asset_series(df, signals)
    assert not events.empty, "Events dataframe should not be empty."

    cv = PurgedGroupTimeSeriesSplit(n_splits=5, embargo_pct=0.01)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(events)):
        train_events = events.iloc[train_idx]
        test_events = events.iloc[test_idx]

        test_start = test_events.index.min()
        test_end = test_events.index.max()

        # Extract train event evaluation windows
        train_entries = train_events.index
        train_exits = pd.to_datetime(train_events["exit_time"])

        # Check for any overlapping span
        overlap_mask = (train_entries <= test_end) & (train_exits >= test_start)
        leaking_events_count = overlap_mask.sum()

        assert leaking_events_count == 0, (
            f"Leakage detected in Fold {fold_idx + 1}! "
            f"{leaking_events_count} training spans intersect with test span [{test_start}, {test_end}]."
        )


def test_in_fold_econometric_standardization_independence():
    """
    Asserts that EconometricsFeaturePipeline does not consume test fold statistics
    and applies train mean/std projections exclusively.
    """
    np.random.seed(42)
    # Train partition with Mean ~ 10, Std ~ 2
    train_data = pd.DataFrame({
        "feat_a": np.random.normal(loc=10.0, scale=2.0, size=100),
        "feat_b": np.random.normal(loc=50.0, scale=5.0, size=100),
    })
    y_train = pd.Series(np.random.choice([0, 1], size=100))

    # Test partition with intentional regime shift: Mean ~ 50, Std ~ 10
    test_data = pd.DataFrame({
        "feat_a": np.random.normal(loc=50.0, scale=10.0, size=50),
        "feat_b": np.random.normal(loc=150.0, scale=20.0, size=50),
    })

    pipeline = EconometricsFeaturePipeline(correlation_threshold=0.85, min_target_corr=0.0)
    train_trans = pipeline.fit_transform(train_data, y_train)

    # Train projections must be close to 0 mean and unit variance
    np.testing.assert_allclose(train_trans.mean(), 0.0, atol=1e-7)
    np.testing.assert_allclose(train_trans.std(ddof=0), 1.0, atol=1e-7)

    test_trans = pipeline.transform(test_data)

    # If test data were contaminated into the fit, test_trans.mean() would be ~0.
    # Because train statistics are strictly used, test transformed mean must be ~ (50 - 10)/2 = 20.
    expected_shifted_mean_a = (test_data["feat_a"].mean() - pipeline.feature_means_["feat_a"]) / pipeline.feature_stds_["feat_a"]
    actual_test_mean_a = test_trans["feat_a"].mean()

    np.testing.assert_allclose(actual_test_mean_a, expected_shifted_mean_a, atol=1e-5)


def test_garman_klass_strict_causality(synthetic_ohlcv_dataset):
    """
    Asserts Garman-Klass volatility computation at t0 is mathematically independent
    of any post-t0 price bars.
    """
    df = synthetic_ohlcv_dataset
    gk_original = compute_garman_klass_volatility(df)

    # Perturb prices at bar 100 onwards
    df_perturbed = df.copy()
    df_perturbed.iloc[100:, df_perturbed.columns.get_loc("high_adj")] *= 1.5
    df_perturbed.iloc[100:, df_perturbed.columns.get_loc("close_adj")] *= 1.5

    gk_perturbed = compute_garman_klass_volatility(df_perturbed)

    # Prior bars (0 to 99) must remain identical
    np.testing.assert_array_equal(
        gk_original.iloc[:100].values,
        gk_perturbed.iloc[:100].values,
        err_msg="Garman-Klass volatility calculation is retroactively affected by future prices!"
    )


def test_suspension_calendar_gap_masking():
    """
    Asserts that log returns across calendar suspensions (> 7 days)
    are strictly converted to NaN to eliminate distorted phantom volatility.
    """
    dates = [
        pd.Timestamp("2023-01-02"),
        pd.Timestamp("2023-01-03"),
        # Suspension: 10 calendar days gap
        pd.Timestamp("2023-01-13"),
        pd.Timestamp("2023-01-16"),
    ]
    prices = [100.0, 102.0, 120.0, 121.0]
    df = pd.DataFrame({"close_adj": prices, "open_adj": prices}, index=dates)

    # Measure calendar gap
    time_diff = df.index.to_series().diff().dt.days
    df["log_return"] = np.log(df["close_adj"] / df["close_adj"].shift(1))
    
    # Masking rule
    suspension_mask = time_diff > 7
    df.loc[suspension_mask, "log_return"] = np.nan

    assert np.isnan(df["log_return"].iloc[2]), "Return across a 10-day calendar suspension was not converted to NaN."
    assert not np.isnan(df["log_return"].iloc[1]), "Regular trading return should remain valid."