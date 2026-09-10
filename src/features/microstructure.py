"""
Microstructure Feature Derivations:
Implements Garman-Klass Volatility, Institutional Flow Dynamics,
Cross-Sectional Benchmark Coupling, and Systemic Market Breadth.
"""

import numpy as np
import pandas as pd


def compute_garman_klass_volatility(df: pd.DataFrame) -> pd.Series:
    """
    Computes instantaneous Garman-Klass continuous volatility estimator:
    sigma^2 = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
    """
    log_hl = np.log(df["high_adj"] / df["low_adj"])
    log_co = np.log(df["close_adj"] / df["open_adj"])
    var_gk = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    # Clip numerical underflow artifacts
    var_gk = np.maximum(var_gk, 0.0)
    return np.sqrt(var_gk)


def compute_institutional_flow_zscore(
    volume_series: pd.Series, 
    window: int = 20
) -> pd.Series:
    """
    Normalizes continuous asset turnover via Rolling Volume Z-Score.
    Identifies high-conviction institutional accumulation.
    """
    rolling_mean = volume_series.rolling(window=window, min_periods=window // 2).mean()
    rolling_std = volume_series.rolling(window=window, min_periods=window // 2).std()
    return (volume_series - rolling_mean) / (rolling_std + 1e-8)


def compute_relative_volatility(
    asset_ret: pd.Series, 
    benchmark_ret: pd.Series, 
    window: int = 20
) -> pd.Series:
    """
    Evaluates unsystematic dispersion penalty:
    Ratio of individual asset realized volatility to systemic benchmark volatility.
    """
    asset_vol = asset_ret.rolling(window=window, min_periods=window // 2).std()
    benchmark_vol = benchmark_ret.rolling(window=window, min_periods=window // 2).std()
    return asset_vol / (benchmark_vol + 1e-8)


def extract_microstructure_features(
    df_asset: pd.DataFrame, 
    df_benchmark: pd.DataFrame
) -> pd.DataFrame:
    """
    Synthesizes the complete exogenous microstructure matrix for an individual ticker.
    """
    df = df_asset.copy()

    # 1. Garman-Klass Volatility Metric (Risk Detractor beta = -0.214)
    df["feat_gk_vol"] = compute_garman_klass_volatility(df)

    # 2. Institutional Volume Flow Z-Score (Conviction driver beta = +0.089)
    df["feat_flow_zscore"] = compute_institutional_flow_zscore(df["volume"])

    # Synchronize with Benchmark (VN-Index)
    if not df_benchmark.empty and "mkt_return" in df.columns:
        # 3. Rolling Correlation with Benchmark (Primary Alpha Driver beta = +0.156)
        df["feat_market_corr"] = (
            df["log_return"]
            .rolling(window=20, min_periods=10)
            .corr(df["mkt_return"])
            .fillna(0.0)
        )

        # 4. Relative Volatility to Benchmark (Secondary Vol Penalty beta = -0.011)
        df["feat_relative_vol"] = compute_relative_volatility(
            df["log_return"], 
            df["mkt_return"], 
            window=20
        ).fillna(1.0)
    else:
        df["feat_market_corr"] = 0.0
        df["feat_relative_vol"] = 1.0

    return df