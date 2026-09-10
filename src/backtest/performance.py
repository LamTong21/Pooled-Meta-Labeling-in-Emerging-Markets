"""
Portfolio Performance Analyzer:
Calculates institutional performance teardowns, underwater drawdown paths,
risk-adjusted metrics, and statistical trade distribution profiles.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd


class PortfolioPerformanceAnalyzer:
    """
    Computes professional quantitative portfolio tear-sheets:
    - Underwater Drawdown Profiles (Peak-to-Trough)
    - Compounded Annualized Growth Rate (CAGR)
    - Sharpe & Sortino Ratios
    - Expectancy & Profit Factor (Gross and Net)
    """

    @staticmethod
    def compute_drawdown_series(nav_series: pd.Series) -> pd.DataFrame:
        """
        Derives high-water mark and continuous percentage underwater drawdown.
        """
        hwm = nav_series.cummax()
        drawdown = (nav_series - hwm) / hwm
        return pd.DataFrame({"nav": nav_series, "hwm": hwm, "drawdown": drawdown})

    @classmethod
    def generate_tearsheet(
        cls,
        df_nav: pd.DataFrame,
        df_trades: pd.DataFrame,
        benchmark_col: Optional[str] = "benchmark_nav",
        rf_rate: float = 0.04,  # 4% annual risk-free proxy
    ) -> Dict[str, object]:
        """
        Synthesizes end-to-end performance teardown.

        Returns:
            Dict containing summary metrics table and underwater equity profiles.
        """
        nav = df_nav["nav"]
        daily_returns = nav.pct_change().dropna()
        n_days = len(daily_returns)
        years = max(n_days / 252.0, 0.1)

        # 1. Growth & Return Metrics
        total_return = (nav.iloc[-1] - nav.iloc[0]) / nav.iloc[0]
        cagr = (1.0 + total_return) ** (1.0 / years) - 1.0

        # 2. Risk Metrics & Drawdown
        dd_df = cls.compute_drawdown_series(nav)
        max_drawdown = dd_df["drawdown"].min()

        volatility = daily_returns.std() * np.sqrt(252)
        excess_returns = daily_returns - (rf_rate / 252.0)
        sharpe = (
            (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)
            if daily_returns.std() > 0
            else 0.0
        )

        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (
            (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
            if downside_std > 0
            else 0.0
        )

        calmar = abs(cagr / max_drawdown) if max_drawdown < 0 else 0.0

        # 3. Trade Log Statistical Distribution
        if not df_trades.empty and "net_return" in df_trades.columns:
            trades = df_trades["net_return"]
            win_rate = (trades > 0).mean() * 100.0
            gross_profit = trades[trades > 0].sum()
            gross_loss = abs(trades[trades < 0].sum())
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else np.nan
            )
            total_trades = len(trades)
        else:
            win_rate, profit_factor, total_trades = np.nan, np.nan, 0

        # 4. Benchmark Alignment (VN-Index)
        benchmark_mdd = np.nan
        benchmark_return = np.nan
        if benchmark_col in df_nav.columns:
            bm_nav = df_nav[benchmark_col].dropna()
            bm_dd = cls.compute_drawdown_series(bm_nav)
            benchmark_mdd = bm_dd["drawdown"].min()
            benchmark_return = (bm_nav.iloc[-1] - bm_nav.iloc[0]) / bm_nav.iloc[0]

        summary_metrics = pd.DataFrame(
            [
                {"Metric": "Testing Horizon (Years)", "Value": f"{years:.2f}"},
                {"Metric": "Total Net Return (%)", "Value": f"{total_return * 100:.2f}%"},
                {"Metric": "CAGR (%)", "Value": f"{cagr * 100:.2f}%"},
                {"Metric": "Annualized Volatility (%)", "Value": f"{volatility * 100:.2f}%"},
                {"Metric": "Sharpe Ratio", "Value": f"{sharpe:.2f}"},
                {"Metric": "Sortino Ratio", "Value": f"{sortino:.2f}"},
                {"Metric": "Calmar Ratio", "Value": f"{calmar:.2f}"},
                {"Metric": "Strategy Max Drawdown (MDD)", "Value": f"{max_drawdown * 100:.2f}%"},
                {"Metric": "VN-Index Max Drawdown (MDD)", "Value": f"{benchmark_mdd * 100:.2f}%"},
                {"Metric": "VN-Index Net Return (%)", "Value": f"{benchmark_return * 100:.2f}%"},
                {"Metric": "Total Executed Trades", "Value": f"{total_trades}"},
                {"Metric": "Post-Friction Win Rate (%)", "Value": f"{win_rate:.2f}%"},
                {"Metric": "Post-Friction Profit Factor", "Value": f"{profit_factor:.2f}"},
                {"Metric": "Average Cash Ratio (%)", "Value": f"{df_nav['cash_ratio'].mean() * 100:.2f}%"},
            ]
        )

        return {
            "summary_table": summary_metrics,
            "drawdown_profile": dd_df,
            "metrics_dict": {
                "cagr": cagr,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "benchmark_mdd": benchmark_mdd,
                "profit_factor": profit_factor,
                "win_rate": win_rate,
            },
        }