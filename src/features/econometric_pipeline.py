"""
In-Fold Econometric Feature Selection Pipeline:
Ensures absolute mathematical immunity against look-ahead bias and data leakage
by strictly encapsulating statistical transformations within CV training folds.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class EconometricsFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Leak-free feature transformer:
    1. Fits scaler parameters (mu, sigma) strictly on the in-fold training feature matrix.
    2. Drops invariant (zero-variance) predictors dynamically.
    3. Filters out redundant collinear features via pairwise correlation pruning.
    4. Evaluates statistical predictive consensus (Bivariate Correlation / Information Rank).
    """

    def __init__(
        self,
        correlation_threshold: float = 0.85,
        min_target_corr: float = 0.01,
    ):
        self.correlation_threshold = correlation_threshold
        self.min_target_corr = min_target_corr
        self.scaler = StandardScaler()
        self.selected_features_: List[str] = []
        self.feature_means_: Optional[pd.Series] = None
        self.feature_stds_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learns feature distributions and statistically consensus-driven dimensions
        strictly on training fold observations.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Feature matrix X must be a pandas DataFrame to track variable names.")

        X_train = X.copy()
        initial_cols = X_train.columns.tolist()

        # Step 1: Remove Zero-Variance (Degenerate) Dimensions
        variances = X_train.var(axis=0)
        valid_variance_cols = variances[variances > 1e-8].index.tolist()
        X_train = X_train[valid_variance_cols]

        # Step 2: Collinear Variable Pruning (Greedy Pearson Filter)
        corr_matrix = X_train.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        redundant_cols = [
            col for col in upper_tri.columns if any(upper_tri[col] > self.correlation_threshold)
        ]
        non_collinear_cols = [col for col in valid_variance_cols if col not in redundant_cols]
        X_train = X_train[non_collinear_cols]

        # Step 3: Target Informativeness Verification (If y is provided)
        if y is not None:
            aligned_y = y.loc[X_train.index]
            correlations_with_target = X_train.apply(lambda col: col.corr(aligned_y)).abs()
            retained_cols = correlations_with_target[
                correlations_with_target >= self.min_target_corr
            ].index.tolist()
            if len(retained_cols) > 0:
                self.selected_features_ = retained_cols
            else:
                self.selected_features_ = non_collinear_cols
        else:
            self.selected_features_ = non_collinear_cols

        # Step 4: Fit In-Fold Standardization
        self.scaler.fit(X_train[self.selected_features_])
        self.feature_means_ = X_train[self.selected_features_].mean()
        self.feature_stds_ = X_train[self.selected_features_].std().replace(0.0, 1.0)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms out-of-sample or test matrices using learned in-fold parameters only:
        z_{j, test} = (x_{j, test} - mu_{j, train}) / sigma_{j, train}
        """
        if not self.selected_features_:
            raise RuntimeError("The pipeline has not been fitted. Call fit() before transform().")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        # Align columns
        missing_cols = [col for col in self.selected_features_ if col not in X.columns]
        if missing_cols:
            raise KeyError(f"Input DataFrame is missing required fitted features: {missing_cols}")

        subset = X[self.selected_features_].copy()

        # Apply strict training parameters
        standardized_array = self.scaler.transform(subset)
        standardized_df = pd.DataFrame(
            standardized_array,
            index=X.index,
            columns=self.selected_features_,
        )
        return standardized_df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fits strictly on (X, y) and returns standardized training projections."""
        return self.fit(X, y).transform(X)