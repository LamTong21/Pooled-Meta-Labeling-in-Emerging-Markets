"""
Portfolio Execution Engine:
Simulates cash-constrained, multi-asset portfolio execution with fixed slots,
queue management for concurrent signals, and round-trip transaction frictions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    allocated_capital: float
    shares: float
    exit_reason: str


class PortfolioExecutionEngine:
    """
    Simulates institutional execution under slot constraints and trading frictions:
    - Maximum concurrent long positions: max_positions (default 5).
    - Round-trip friction per executed trade: friction_pct (default 0.30%).
    - Capital allocation: Equal weight across available slots (1 / max_positions).
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000_000.0,
        max_positions: int = 5,
        friction_pct: float = 0.0030,
        meta_threshold: float = 0.64,
    ):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.friction_pct = friction_pct
        self.meta_threshold = meta_threshold

    def simulate(
        self,
        pooled_events: pd.DataFrame,
        oos_probabilities: pd.Series,
        price_dict: Dict[str, pd.DataFrame],
        benchmark_series: Optional[pd.Series] = None,
    ) -> Dict[str, object]:
        """
        Executes chronologically constrained portfolio simulation.

        Parameters:
            pooled_events: DataFrame of trade candidates (containing exit_time, exit_price, ticker, etc.)
            oos_probabilities: Series of calibrated continuous meta-probabilities.
            price_dict: Dict of daily adjusted price DataFrames per ticker.
            benchmark_series: Daily close Series of the market benchmark (VN-Index).

        Returns:
            Dict containing daily NAV time series, completed trade logs, and rejected events.
        """
        events = pooled_events.copy()
        events["meta_prob"] = oos_probabilities

        # 1. Filter candidates by acceptance threshold P(Meta) >= P*
        accepted_candidates = events[events["meta_prob"] >= self.meta_threshold].sort_index()

        # Build continuous daily trading calendar across the universe
        start_date = events.index.min()
        end_date = events["exit_time"].max()
        calendar_dates = pd.date_range(start=start_date, end=end_date, freq="B")

        cash = self.initial_capital
        active_positions: List[Position] = []
        trade_history: List[Dict] = []
        rejected_signals: List[Dict] = []
        daily_nav_records: List[Dict] = []

        # Convert candidate events to an efficient chronological queue
        candidate_queue = accepted_candidates.to_dict(orient="index")
        event_entry_dates = set(accepted_candidates.index)

        for current_date in calendar_dates:
            # Step A: Close active positions reaching or exceeding exit_time
            surviving_positions = []
            for pos in active_positions:
                if current_date >= pos.exit_date:
                    gross_proceeds = pos.shares * pos.exit_price
                    # Deduct round-trip transaction friction (brokerage, tax, slippage)
                    net_proceeds = gross_proceeds * (1.0 - self.friction_pct)
                    cash += net_proceeds

                    pnl = net_proceeds - pos.allocated_capital
                    ret = pnl / pos.allocated_capital

                    trade_history.append(
                        {
                            "ticker": pos.ticker,
                            "entry_date": pos.entry_date,
                            "exit_date": pos.exit_date,
                            "entry_price": pos.entry_price,
                            "exit_price": pos.exit_price,
                            "allocated_capital": pos.allocated_capital,
                            "net_proceeds": net_proceeds,
                            "net_pnl": pnl,
                            "net_return": ret,
                            "exit_reason": pos.exit_reason,
                        }
                    )
                else:
                    surviving_positions.append(pos)
            active_positions = surviving_positions

            # Step B: Open new positions from candidates triggering at current_date
            if current_date in event_entry_dates:
                daily_triggers = [
                    (idx, row)
                    for idx, row in candidate_queue.items()
                    if idx == current_date
                ]
                # If multiple triggers on same day, rank by highest meta-probability
                daily_triggers.sort(key=lambda x: x[1]["meta_prob"], reverse=True)

                for _, trigger in daily_triggers:
                    ticker = trigger["ticker"]

                    # Check position limits: max concurrent longs
                    if len(active_positions) < self.max_positions:
                        # Prevent duplicate entries in the same asset
                        if any(pos.ticker == ticker for pos in active_positions):
                            continue

                        # Fixed slot sizing: 1 / max_positions of total NAV
                        current_nav = cash + sum(
                            p.shares * price_dict[p.ticker]["close_adj"].asof(current_date)
                            if current_date in price_dict[p.ticker].index
                            else p.shares * p.entry_price
                            for p in active_positions
                        )
                        target_capital = current_nav / self.max_positions
                        allocated_capital = min(cash, target_capital)

                        if allocated_capital > 1_000_000.0:  # Minimum execution threshold
                            entry_price = trigger["entry_price"]
                            # Apply friction to the buy entry
                            effective_entry_capital = allocated_capital * (1.0 - self.friction_pct)
                            shares = effective_entry_capital / entry_price
                            cash -= allocated_capital

                            active_positions.append(
                                Position(
                                    ticker=ticker,
                                    entry_date=current_date,
                                    exit_date=pd.to_datetime(trigger["exit_time"]),
                                    entry_price=entry_price,
                                    exit_price=trigger["exit_price"],
                                    allocated_capital=allocated_capital,
                                    shares=shares,
                                    exit_reason=trigger["exit_reason"],
                                )
                            )
                    else:
                        # Slot capacity exceeded: trade starved
                        rejected_signals.append(
                            {
                                "date": current_date,
                                "ticker": ticker,
                                "meta_prob": trigger["meta_prob"],
                                "reason": "slot_exhaustion",
                            }
                        )

            # Step C: Compute Mark-to-Market Daily NAV
            equity_holdings_value = 0.0
            for pos in active_positions:
                df_asset = price_dict[pos.ticker]
                current_price = (
                    df_asset["close_adj"].asof(current_date)
                    if current_date in df_asset.index
                    else pos.entry_price
                )
                equity_holdings_value += pos.shares * current_price

            total_nav = cash + equity_holdings_value
            cash_ratio = cash / total_nav if total_nav > 0 else 1.0

            daily_nav_records.append(
                {
                    "date": current_date,
                    "nav": total_nav,
                    "cash": cash,
                    "equity_value": equity_holdings_value,
                    "cash_ratio": cash_ratio,
                    "active_slots": len(active_positions),
                }
            )

        df_nav = pd.DataFrame(daily_nav_records).set_index("date")
        df_trades = pd.DataFrame(trade_history)
        df_rejected = pd.DataFrame(rejected_signals)

        # Synchronize normalized benchmark if provided
        if benchmark_series is not None:
            aligned_mkt = benchmark_series.reindex(df_nav.index).ffill()
            df_nav["benchmark_nav"] = (
                aligned_mkt / aligned_mkt.iloc[0]
            ) * self.initial_capital

        return {
            "daily_nav": df_nav,
            "trade_log": df_trades,
            "rejected_signals": df_rejected,
        }