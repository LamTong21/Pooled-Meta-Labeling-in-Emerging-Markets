"""
Portfolio Backtest & Risk Engine Package.
Simulates realistic cash-constrained execution (slot limits, slippage, taxes)
and computes institutional performance diagnostics and underwater drawdown profiles.
"""

from .execution_engine import PortfolioExecutionEngine
from .performance import PortfolioPerformanceAnalyzer

__all__ = [
    "PortfolioExecutionEngine",
    "PortfolioPerformanceAnalyzer",
]