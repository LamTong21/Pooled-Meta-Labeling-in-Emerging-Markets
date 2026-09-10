"""
Reinforcement Learning Bridge: State Space Formulation Module.
Transforms static binary meta-filtering into a continuous confidence field
within a Markov Decision Process (MDP) to eliminate cash drag and opportunity starvation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class RLStateSpaceBuilder:
    """
    Constructs normalized continuous state tensors for Reinforcement Learning agents:
    
    State Vector S_t per constituent i:
        S_t = {
            P(Meta)_{i,t}   : Continuous calibrated meta-probability [0, 1]
            Cash_Ratio_t    : Available portfolio liquidity (Cash_t / NAV_t) [0, 1]
            sigma_{GK,t}    : Normalized Garman-Klass intraday continuous volatility
            rho_{i,Mkt,t}   : 20-day rolling correlation with VN-Index [-1, 1]
            w_{i,t-1}       : Lagged portfolio asset allocation weight [0, 1]
        }
        
    Target Action Vector A_t:
        A_t in [0.0, 1.0]^N (Continuous dynamic sizing / capital allocation)
    """

    def __init__(
        self,
        target_tickers: List[str],
        vol_clip: float = 0.05,
    ):
        self.target_tickers = sorted(target_tickers)
        self.num_assets = len(self.target_tickers)
        self.ticker_to_idx = {ticker: idx for idx, ticker in enumerate(self.target_tickers)}
        self.vol_clip = vol_clip
        
        # Dimensions per asset: [P_meta, vol_gk, corr_mkt, lagged_weight]
        # Global portfolio features: [cash_ratio]
        self.features_per_asset = 4
        self.global_features = 1
        self.state_dim = (self.num_assets * self.features_per_asset) + self.global_features

    def get_state_dimension(self) -> int:
        """Returns the flat observation space dimension for the Gym Environment."""
        return self.state_dim

    def build_discrete_step_state(
        self,
        current_date: pd.Timestamp,
        active_meta_probs: Dict[str, float],
        market_microstructure_dict: Dict[str, pd.DataFrame],
        current_weights: Dict[str, float],
        cash_ratio: float,
    ) -> np.ndarray:
        """
        Builds a single-timestamp 1D normalized state vector S_t.

        Parameters:
            current_date: Evaluation timestamp t.
            active_meta_probs: Dict mapping ticker -> continuous calibrated P(Meta)_t.
                               If no primary trigger occurs, defaults to 0.0.
            market_microstructure_dict: Dict mapping ticker -> daily DataFrame containing
                                        ['feat_gk_vol', 'feat_market_corr'].
            current_weights: Dict mapping ticker -> current portfolio weight w_{i, t-1}.
            cash_ratio: Portfolio cash ratio (Cash_t / NAV_t) in [0.0, 1.0].

        Returns:
            Flat 1D numpy array of shape (state_dim,) dtype np.float32.
        """
        state_components = []

        # 1. Global Portfolio Context: Cash Ratio
        clamped_cash_ratio = np.clip(cash_ratio, 0.0, 1.0)
        state_components.append(clamped_cash_ratio)

        # 2. Asset-Specific Features
        for ticker in self.target_tickers:
            # Feature A: Continuous Meta-Probability field P(Meta)_{i,t}
            # Un-triggered assets register 0.0 baseline confidence
            p_meta = float(active_meta_probs.get(ticker, 0.0))
            p_meta = np.clip(p_meta, 0.0, 1.0)

            # Feature B & C: Microstructure indicators
            df_asset = market_microstructure_dict.get(ticker)
            if df_asset is not None and current_date in df_asset.index:
                row = df_asset.loc[current_date]
                raw_vol = float(row.get("feat_gk_vol", 0.0))
                raw_corr = float(row.get("feat_market_corr", 0.0))
            else:
                raw_vol = 0.0
                raw_corr = 0.0

            # Normalize volatility by bounding window (scaling 0.0 to 1.0)
            norm_vol = np.clip(raw_vol / (self.vol_clip + 1e-8), 0.0, 1.0)
            # Correlation already naturally bounded in [-1.0, 1.0]
            norm_corr = np.clip(raw_corr, -1.0, 1.0)

            # Feature D: Previous allocation weight w_{i, t-1} (Friction & Turnover state)
            prev_weight = float(current_weights.get(ticker, 0.0))
            prev_weight = np.clip(prev_weight, 0.0, 1.0)

            state_components.extend([p_meta, norm_vol, norm_corr, prev_weight])

        state_vector = np.array(state_components, dtype=np.float32)
        return state_vector

    def build_historical_trajectory(
        self,
        calendar_dates: pd.DatetimeIndex,
        pooled_events: pd.DataFrame,
        oos_probabilities: pd.Series,
        market_microstructure_dict: Dict[str, pd.DataFrame],
        historical_weights_df: Optional[pd.DataFrame] = None,
        historical_cash_ratios: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Synthesizes a continuous chronological offline dataset of state vectors
        across the backtest horizon for offline RL (CQL/IQL) or environment pre-caching.
        """
        events = pooled_events.copy()
        events["meta_prob"] = oos_probabilities

        state_records = []
        for date in calendar_dates:
            # Extract concurrent triggers active on date
            active_events = events[events.index == date]
            active_meta_map = dict(zip(active_events["ticker"], active_events["meta_prob"]))

            current_cash = (
                float(historical_cash_ratios.loc[date])
                if historical_cash_ratios is not None and date in historical_cash_ratios.index
                else 1.0
            )

            current_w = (
                historical_weights_df.loc[date].to_dict()
                if historical_weights_df is not None and date in historical_weights_df.index
                else {ticker: 0.0 for ticker in self.target_tickers}
            )

            state_vec = self.build_discrete_step_state(
                current_date=date,
                active_meta_probs=active_meta_map,
                market_microstructure_dict=market_microstructure_dict,
                current_weights=current_w,
                cash_ratio=current_cash,
            )

            state_records.append(state_vec)

        df_states = pd.DataFrame(state_records, index=calendar_dates)
        return df_states

    @staticmethod
    def map_actions_to_execution(
        action_vector: np.ndarray,
        target_tickers: List[str],
        max_positions: int = 5,
        min_allocation_threshold: float = 0.05,
    ) -> Dict[str, float]:
        """
        Maps continuous RL actions A_t in [0, 1]^N to a normalized portfolio allocation:
        1. Masks sub-threshold noise allocations.
        2. Limits allocation strictly to the top-K highest conviction assets.
        3. Enforces simplex constraint: sum(w_i) <= 1.0 (remainder stays as cash buffer).
        """
        actions = np.clip(action_vector, 0.0, 1.0)
        
        # Zero out low-confidence fractional allocations
        actions[actions < min_allocation_threshold] = 0.0

        # Enforce top-K maximum open slots
        if np.count_nonzero(actions) > max_positions:
            top_indices = np.argsort(actions)[-max_positions:]
            mask = np.zeros_like(actions, dtype=bool)
            mask[top_indices] = True
            actions[~mask] = 0.0

        total_weight = np.sum(actions)
        if total_weight > 1.0:
            actions = actions / total_weight

        return {ticker: float(actions[idx]) for idx, ticker in enumerate(target_tickers)}