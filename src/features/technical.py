"""
Technical Indicators & Primary Signal Generation Module.
Generates baseline heuristic execution triggers on uninterrupted isolated series.
"""

import numpy as np
import pandas as pd


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives standard technical indicators on isolated per-asset price series.
    All transformations respect strict chronological order.
    """
    out = df.copy()
    close = out["close_adj"]
    high = out["high_adj"]
    low = out["low_adj"]

    # 1. Dual Moving Averages (EMA 10 / 30)
    ema_fast = close.ewm(span=10, adjust=False).mean()
    ema_slow = close.ewm(span=30, adjust=False).mean()
    out["feat_ema_spread"] = (ema_fast - ema_slow) / (ema_slow + 1e-8)

    # 2. Relative Strength Index (RSI 14)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    out["feat_rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # 3. Donchian Breakout Bands (20-day horizon)
    out["donchian_high_20"] = high.rolling(window=20, min_periods=20).max().shift(1)
    out["donchian_low_20"] = low.rolling(window=20, min_periods=20).min().shift(1)

    # 4. Average True Range (ATR 14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["feat_atr_14"] = true_range.rolling(window=14, min_periods=14).mean() / (close + 1e-8)

    # 5. Normalized Momentum (10-day rate of change)
    out["feat_roc_10"] = (close - close.shift(10)) / (close.shift(10) + 1e-8)

    return out


def generate_primary_signals(df: pd.DataFrame) -> pd.Series:
    """
    Generates binary long execution triggers (t0 entries).
    Combines Donchian 20-day upper breakout with EMA trend confirmation.
    
    Returns:
        Boolean Series (True = Entry Trigger, False = Pass)
    """
    indicators = compute_technical_indicators(df)
    close = indicators["close_adj"]
    donchian_high = indicators["donchian_high_20"]
    ema_spread = indicators["feat_ema_spread"]

    # Rule-based trigger: Breakout above prior 20-day high under positive trend regime
    breakout = close > donchian_high
    trend_filter = ema_spread > 0.0

    primary_signals = breakout & trend_filter
    return primary_signals.fillna(False)