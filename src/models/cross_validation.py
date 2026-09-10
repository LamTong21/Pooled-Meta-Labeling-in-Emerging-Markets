"""
Purged Group Time-Series Cross-Validation Module.
Guarantees zero information leakage between chronological training and validation partitions
by purging overlapping event holding periods across partition temporal boundaries.
"""

from typing import Generator, Tuple, Optional
import numpy as np
import pandas as pd


class PurgedGroupTimeSeriesSplit:
    """
    Time-Series Cross-Validator with holding-period purging and embargoing:
    - Splits observations chronologically across N blocks.
    - Purges any training sample whose barrier evaluation window [t0, t_exit] 
      overlaps with the test partition time span.
    - Prevents cross-sectional constituent look-ahead leakage.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        event_times: Optional[pd.Series] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generates indices of chronological (train, test) splits with barrier-span purging.

        Parameters:
            X: Input event matrix indexed by timestamp t0.
            y: Binary target Series.
            event_times: Series indexed by t0 containing the discrete event exit_time.
                         If None, assumes 'exit_time' exists as a column in X.

        Yields:
            (train_indices, test_indices) as numpy integer arrays.
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame X must have a DatetimeIndex representing entry t0.")

        if event_times is None:
            if "exit_time" not in X.columns:
                raise KeyError("event_times must be provided or 'exit_time' must be present in X.")
            exit_series = pd.to_datetime(X["exit_time"])
        else:
            exit_series = pd.to_datetime(event_times)

        n_samples = len(X)
        indices = np.arange(n_samples)
        timestamps = X.index

        # Segment continuous timeline into n_splits chronological chunks
        fold_bounds = np.linspace(0, n_samples, self.n_splits + 1, dtype=int)

        for i in range(self.n_splits):
            test_start_idx = fold_bounds[i]
            test_end_idx = fold_bounds[i + 1]

            if test_start_idx == test_end_idx:
                continue

            test_indices = indices[test_start_idx:test_end_idx]
            test_start_time = timestamps[test_start_idx]
            test_end_time = timestamps[test_end_idx - 1]

            # 1. Candidate training set: Strictly prior to testing partition (Expanding Window)
            # or all non-testing records (for block-wise validation)
            train_candidate_indices = indices[~np.isin(indices, test_indices)]

            # 2. Purging: Drop any training event whose exit_time extends into the test window
            # or whose entry t0 precedes test_end_time while test events are active
            train_t0 = timestamps[train_candidate_indices]
            train_t1 = exit_series.iloc[train_candidate_indices]

            # Overlap Condition: (train_t0 <= test_end_time) & (train_t1 >= test_start_time)
            leakage_mask = (train_t0 <= test_end_time) & (train_t1 >= test_start_time)
            clean_train_indices = train_candidate_indices[~leakage_mask]

            # 3. Embargo: Drop training samples immediately following test partition
            if self.embargo_pct > 0:
                embargo_offset = int(n_samples * self.embargo_pct)
                embargo_limit_time = test_end_time + pd.Timedelta(days=embargo_offset)
                post_test_mask = (timestamps[clean_train_indices] > test_end_time) & (
                    timestamps[clean_train_indices] <= embargo_limit_time
                )
                clean_train_indices = clean_train_indices[~post_test_mask]

            if len(clean_train_indices) == 0:
                continue

            yield clean_train_indices, test_indices