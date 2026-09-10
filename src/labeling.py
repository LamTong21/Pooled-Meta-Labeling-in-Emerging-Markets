"""
Labeling Module:
Implements the Dynamic Triple Barrier Method adjusted for volatility regimes
to generate binary ground-truth labels for secondary meta-classifiers.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


class TripleBarrierLabeler:
    """
    Labels trading triggers via dynamic volatility bounds:
    - Upper Barrier: pt * sigma
    - Lower Barrier: sl * sigma
    - Horizontal Barrier: h (number of bars)
    """

    def __init__(
        self,
        pt: float = 2.5,
        sl: float = 1.5,
        horizon: int = 10,
        vol_lookback: int = 20,
    ):
        self.pt = pt
        self.sl = sl
        self.horizon = horizon
        self.vol_lookback = vol_lookback

    def compute_daily_volatility(self, close_series: pd.Series) -> pd.Series:
        """Computes instantaneous daily return standard deviation."""
        log_ret = np.log(close_series / close_series.shift(1))
        vol = log_ret.rolling(window=self.vol_lookback, min_periods=max(5, self.vol_lookback // 2)).std()
        return vol.bfill()

    def label_asset_series(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
    ) -> pd.DataFrame:
        """
        Applies dynamic barriers to an individual ticker dataframe.

        Parameters:
            df: DataFrame containing at least ['open_adj', 'high_adj', 'low_adj', 'close_adj'].
            signals: Boolean Series indicating primary long trigger events (True = Entry t0).

        Returns:
            DataFrame containing labeled events and barrier timestamps.
        """
        close = df["close_adj"]
        high = df["high_adj"]
        low = df["low_adj"]
        vol = self.compute_daily_volatility(close)

        entry_dates = signals[signals].index
        records = []

        timestamps = df.index
        locs = timestamps.get_indexer(entry_dates)

        for t0_idx, t0 in zip(locs, entry_dates):
            if t0_idx < 0 or t0_idx >= len(timestamps) - 1:
                continue

            sigma = vol.iloc[t0_idx]
            if np.isnan(sigma) or sigma <= 0:
                continue

            entry_price = close.iloc[t0_idx]
            upper_barrier = entry_price * (1.0 + self.pt * sigma)
            lower_barrier = entry_price * (1.0 - self.sl * sigma)

            # Define time horizon window
            t_max_idx = min(t0_idx + self.horizon, len(timestamps) - 1)
            future_highs = high.iloc[t0_idx + 1 : t_max_idx + 1]
            future_lows = low.iloc[t0_idx + 1 : t_max_idx + 1]

            upper_hit_candidates = future_highs[future_highs >= upper_barrier].index
            lower_hit_candidates = future_lows[future_lows <= lower_barrier].index

            t_upper = upper_hit_candidates[0] if len(upper_hit_candidates) > 0 else None
            t_lower = lower_hit_candidates[0] if len(lower_hit_candidates) > 0 else None

            # Label logic: Strict upper touch before stop loss and before timeout
            label = 0
            exit_date = timestamps[t_max_idx]
            exit_reason = "timeout"

            if t_upper is not None and t_lower is not None:
                if t_upper < t_lower:
                    label = 1
                    exit_date = t_upper
                    exit_reason = "take_profit"
                else:
                    label = 0
                    exit_date = t_lower
                    exit_reason = "stop_loss"
            elif t_upper is not None:
                label = 1
                exit_date = t_upper
                exit_reason = "take_profit"
            elif t_lower is not None:
                label = 0
                exit_date = t_lower
                exit_reason = "stop_loss"

            exit_price = close.loc[exit_date]
            realized_return = (exit_price - entry_price) / entry_price

            records.append(
                {
                    "time": t0,
                    "entry_price": entry_price,
                    "exit_time": exit_date,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "realized_return": realized_return,
                    "volatility": sigma,
                    "upper_barrier": upper_barrier,
                    "lower_barrier": lower_barrier,
                    "meta_label": label,
                }
            )

        events_df = pd.DataFrame(records)
        if not events_df.empty:
            events_df.set_index("time", inplace=True)
        return events_df

    def pool_events_across_universe(
        self,
        clean_dict: Dict[str, pd.DataFrame],
        signal_generator_func,
    ) -> pd.DataFrame:
        """
        Derives primary triggers independently per ticker, applies isolated Triple
        Barriers, and merges them into a Pooled Cross-Sectional Dataset.
        """
        all_events = []

        for ticker, df in clean_dict.items():
            # Derive primary execution triggers on isolated time-series
            signals = signal_generator_func(df)
            events = self.label_asset_series(df, signals)

            if not events.empty:
                # Merge isolated asset-level features into the event bar
                feature_cols = [col for col in df.columns if col.startswith("feat_")]
                merged_events = events.join(df[feature_cols], how="inner")
                merged_events["ticker"] = ticker
                all_events.append(merged_events)

        if not all_events:
            return pd.DataFrame()

        # Concatenate and sort temporally to build the pooled training matrix
        pooled_df = pd.concat(all_events, axis=0).sort_index()
        return pooled_df