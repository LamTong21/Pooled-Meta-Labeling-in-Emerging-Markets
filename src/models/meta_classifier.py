"""
L1-Penalized Meta-Classifier Module:
Implements regularized utility minimization, probability threshold sweeps,
and out-of-sample validation routines across Purged Cross-Validation folds.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    brier_score_loss,
)

from ..features.econometric_pipeline import EconometricsFeaturePipeline
from .cross_validation import PurgedGroupTimeSeriesSplit


class L1MetaClassifier:
    """
    Secondary Meta-Labeling Classifier utilizing L1-penalized logistic utility.
    Enforces sparsity, discards zero-importance predictors, and prevents leaf explosion.
    """

    def __init__(
        self,
        C: float = 0.1,
        max_iter: int = 1000,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = LogisticRegression(
            penalty="l1",
            C=self.C,
            solver="liblinear",
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.fitted_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits sparse L1-penalized weights on training partition."""
        self.fitted_features_ = X.columns.tolist()
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns calibrated continuous meta-probabilities P(Meta) in [0.0, 1.0]."""
        aligned_X = X[self.fitted_features_]
        return self.model.predict_proba(aligned_X)[:, 1]

    def get_feature_coefficients(self) -> pd.Series:
        """Extracts standardized beta weights across features."""
        coefs = self.model.coef_.flatten()
        return pd.Series(coefs, index=self.fitted_features_).sort_values(ascending=False)


class MetaModelValidator:
    """
    Coordinates In-Fold Econometric Transformation, Sparse L1 Model Fitting,
    and Out-Of-Sample Performance Auditing across 5 Purged Time-Series Folds.
    """

    def __init__(
        self,
        n_splits: int = 5,
        C: float = 0.1,
        correlation_threshold: float = 0.85,
    ):
        self.n_splits = n_splits
        self.C = C
        self.correlation_threshold = correlation_threshold
        self.cv = PurgedGroupTimeSeriesSplit(n_splits=self.n_splits)

    def run_cross_validation(
        self,
        pooled_events: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "meta_label",
    ) -> Dict[str, object]:
        """
        Executes strict chronological cross-validation with zero look-ahead contamination.

        Returns:
            Dictionary containing OOS probabilities, fold performance metrics, and feature weights.
        """
        X_raw = pooled_events[feature_cols]
        y_raw = pooled_events[target_col]

        oos_predictions = pd.Series(index=pooled_events.index, dtype=float)
        fold_metrics = []
        fold_coefficients = []

        fold_idx = 1
        for train_idx, test_idx in self.cv.split(pooled_events, y_raw):
            X_train, y_train = X_raw.iloc[train_idx], y_raw.iloc[train_idx]
            X_test, y_test = X_raw.iloc[test_idx], y_raw.iloc[test_idx]

            # 1. Strictly In-Fold Econometric Pipeline (Scaling + Correlation pruning)
            pipeline = EconometricsFeaturePipeline(
                correlation_threshold=self.correlation_threshold
            )
            X_train_trans = pipeline.fit_transform(X_train, y_train)
            X_test_trans = pipeline.transform(X_test)

            # 2. Fit L1-Penalized Meta-Model
            classifier = L1MetaClassifier(C=self.C)
            classifier.fit(X_train_trans, y_train)

            # 3. Generate Out-of-Sample Continuous Probabilities
            y_pred_proba = classifier.predict_proba(X_test_trans)
            oos_predictions.iloc[test_idx] = y_pred_proba

            # 4. Measure Fold Diagnostics
            auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan
            brier = brier_score_loss(y_test, y_pred_proba)
            
            fold_metrics.append({
                "fold": fold_idx,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "roc_auc": auc,
                "brier_score": brier,
            })
            
            coef_series = classifier.get_feature_coefficients()
            coef_series.name = f"fold_{fold_idx}"
            fold_coefficients.append(coef_series)

            fold_idx += 1

        metrics_df = pd.DataFrame(fold_metrics)
        coefs_df = pd.concat(fold_coefficients, axis=1).fillna(0.0)

        return {
            "oos_probabilities": oos_predictions,
            "metrics": metrics_df,
            "coefficients": coefs_df,
            "mean_auc": metrics_df["roc_auc"].mean(),
        }

    @staticmethod
    def evaluate_threshold_performance(
        pooled_events: pd.DataFrame,
        oos_probabilities: pd.Series,
        thresholds: List[float] = [0.50, 0.55, 0.60, 0.64, 0.68],
        cost_per_trade: float = 0.0030,
    ) -> pd.DataFrame:
        """
        Sweeps continuous probability cutoff thresholds P(Meta) >= P* and evaluates
        filtered Win Rates, Profit Factors, and Net Cumulative PnL after transaction friction.
        """
        records = []
        y_true = pooled_events["meta_label"]
        raw_ret = pooled_events["realized_return"]

        # Baseline: Standalone Primary Triggers (No meta-filtering)
        base_n = len(pooled_events)
        base_ret = raw_ret - cost_per_trade
        base_wins = base_ret[base_ret > 0].sum()
        base_losses = np.abs(base_ret[base_ret < 0].sum())
        base_pf = base_wins / (base_losses + 1e-8)
        base_wr = (base_ret > 0).mean() * 100.0

        records.append({
            "threshold": "Baseline (Raw)",
            "accepted_trades": base_n,
            "filter_rate_pct": 0.0,
            "win_rate_pct": base_wr,
            "profit_factor": base_pf,
            "net_cumulative_pnl": base_ret.sum() * 100.0,
        })

        # Meta-Filtered Tranches
        for th in thresholds:
            accepted_mask = oos_probabilities >= th
            filtered_ret = raw_ret[accepted_mask] - cost_per_trade
            n_trades = len(filtered_ret)

            if n_trades == 0:
                continue

            filter_rate = (1.0 - (n_trades / base_n)) * 100.0
            wins = filtered_ret[filtered_ret > 0].sum()
            losses = np.abs(filtered_ret[filtered_ret < 0].sum())
            pf = wins / (losses + 1e-8)
            wr = (filtered_ret > 0).mean() * 100.0
            net_pnl = filtered_ret.sum() * 100.0

            records.append({
                "threshold": f"P(Meta) >= {th:.2f}",
                "accepted_trades": n_trades,
                "filter_rate_pct": filter_rate,
                "win_rate_pct": wr,
                "profit_factor": pf,
                "net_cumulative_pnl": net_pnl,
            })

        return pd.DataFrame(records)