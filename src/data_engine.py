"""
Data Engine Module:
Handles data ingestion, physical candle geometric integrity audits,
microstructure feature derivations, and benchmark synchronization.
"""

import os
import time
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from vnstock import Quote

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MultiAssetDataPreparer:
    """
    Ingests, audits candle geometry, sanitizes non-trading anomalies,
    derives microstructure signals, and synchronizes market benchmark data.
    """

    def __init__(
        self,
        portfolio_path: str = "data/final_portfolio.csv",
        start_date: str = "2018-07-01",
        end_date: str = "2026-06-01",
        source: str = "VCI",
        cache_dir: str = "data/raw",
    ):
        self.portfolio_path = portfolio_path
        self.start_date = start_date
        self.end_date = end_date
        self.source = source
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.tickers: List[str] = self._load_tickers()

    def _load_tickers(self) -> List[str]:
        """Loads and sorts unique ticker universe from configuration CSV."""
        if not os.path.exists(self.portfolio_path):
            raise FileNotFoundError(f"Portfolio file not found at: {self.portfolio_path}")
        df = pd.read_csv(self.portfolio_path)
        if "Ticker" not in df.columns:
            raise KeyError("Portfolio CSV must contain a 'Ticker' column.")
        tickers = sorted(df["Ticker"].unique().tolist())
        logger.info("Initialized portfolio with %d constituents from %s", len(tickers), self.portfolio_path)
        return tickers

    def _fetch_history(self, symbol: str) -> pd.DataFrame:
        """
        Fetches daily OHLCV bars:
        - Symbol 'VNINDEX': Ingested via vnstock Quote API.
        - Equities: Ingested via yfinance (using ticker.VN convention).
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}.csv")
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            return df

        try:
            if symbol == "VNINDEX":
                q = Quote(symbol="VNINDEX", source=self.source)
                df = q.history(start=self.start_date, end=self.end_date, interval="1D")
                if df is not None and not df.empty:
                    df.reset_index(inplace=True)
                    df.rename(columns={col: str(col).lower() for col in df.columns}, inplace=True)
                    time_col = "time" if "time" in df.columns else "date"
                    df.rename(columns={time_col: "time"}, inplace=True)

                    required_cols = ["time", "open", "high", "low", "close", "volume"]
                    df = df[[col for col in required_cols if col in df.columns]]
                    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
                    df.sort_values(by="time", inplace=True)
                    df.set_index("time", inplace=True)
                    df.to_csv(cache_file)
                    return df
            else:
                yf_symbol = f"{symbol}.VN"
                ticker_obj = yf.Ticker(yf_symbol)
                df = ticker_obj.history(start=self.start_date, end=self.end_date, interval="1d")
                if df is not None and not df.empty:
                    df.reset_index(inplace=True)
                    df.rename(
                        columns={
                            "Date": "time",
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Volume": "volume",
                        },
                        inplace=True,
                    )
                    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
                    df.sort_values(by="time", inplace=True)
                    df.set_index("time", inplace=True)
                    df.to_csv(cache_file)
                    return df

        except Exception as err:
            logger.error("Failed fetching data for %s: %s", symbol, err)

        return pd.DataFrame()

    def _clean_and_adjust(self, df: pd.DataFrame, df_mkt: pd.DataFrame) -> pd.DataFrame:
        """
        Conducts physical geometry verification, calendar anomaly masking,
        and derives foundational econometric features.
        """
        df = df.copy()

        # 1. Deduplication & temporal ordering
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # 2. Geometry bounds audit
        max_oc = df[["open", "close"]].max(axis=1)
        min_oc = df[["open", "close"]].min(axis=1)
        df["high"] = np.maximum(df["high"], max_oc)
        df["low"] = np.minimum(df["low"], min_oc)
        df = df[(df["low"] > 0) & (df["volume"] >= 0)]

        # 3. Adjusted price calibration
        if "close_adj" not in df.columns:
            df["close_adj"] = df["close"]
            ratio = df["close_adj"] / df["close"]
            df["open_adj"] = df["open"] * ratio
            df["high_adj"] = df["high"] * ratio
            df["low_adj"] = df["low"] * ratio

        # 4. Standard log returns
        df["log_return"] = np.log(df["close_adj"] / df["close_adj"].shift(1))
        df["overnight_return"] = np.log(df["open_adj"] / df["close_adj"].shift(1))
        df["intraday_return"] = np.log(df["close_adj"] / df["open_adj"])
        df["hl_log_range"] = np.log(df["high_adj"] / df["low_adj"])

        # 5. Mask abnormal calendar gaps (Suspensions > 7 calendar days)
        time_diff = df.index.to_series().diff().dt.days
        suspension_mask = time_diff > 7
        df.loc[suspension_mask, ["log_return", "overnight_return"]] = np.nan

        # 6. Microstructure derivation: Garman-Klass continuous volatility
        log_hl = np.log(df["high_adj"] / df["low_adj"])
        log_co = np.log(df["close_adj"] / df["open_adj"])
        df["feat_gk_vol"] = np.sqrt(0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2))

        # 7. Microstructure derivation: Institutional Flow Z-Score (20-day rolling)
        vol_ma = df["volume"].rolling(window=20, min_periods=10).mean()
        vol_std = df["volume"].rolling(window=20, min_periods=10).std()
        df["feat_flow_zscore"] = (df["volume"] - vol_ma) / (vol_std + 1e-8)

        # 8. Benchmark integration & Market Correlation
        if not df_mkt.empty:
            df = df.join(df_mkt, how="left")
            mkt_price_cols = ["mkt_close", "mkt_volume", "mkt_high", "mkt_low"]
            df[mkt_price_cols] = df[mkt_price_cols].ffill()

            if "mkt_return" in df.columns:
                df["mkt_return"] = df["mkt_return"].fillna(0.0)
                # 20-day rolling correlation with benchmark
                rolling_corr = (
                    df["log_return"]
                    .rolling(window=20, min_periods=10)
                    .corr(df["mkt_return"])
                )
                df["feat_market_corr"] = rolling_corr.fillna(0.0)

                # Relative Volatility (Asset vol / Benchmark vol)
                asset_vol_20 = df["log_return"].rolling(20, min_periods=10).std()
                mkt_vol_20 = df["mkt_return"].rolling(20, min_periods=10).std()
                df["feat_relative_vol"] = asset_vol_20 / (mkt_vol_20 + 1e-8)

        return df.dropna(subset=["close_adj", "open_adj"])

    def prepare_dataset(self) -> Dict[str, pd.DataFrame]:
        """Executes ingestion, auditing, and cleaning across all target constituents."""
        logger.info("Fetching and aligning Benchmark VN-INDEX...")
        df_mkt = self._fetch_history("VNINDEX")

        if not df_mkt.empty:
            df_mkt.index = pd.to_datetime(df_mkt.index).normalize()
            df_mkt = df_mkt[~df_mkt.index.duplicated(keep="last")].sort_index()
            df_mkt = df_mkt[["close", "volume", "high", "low"]].rename(
                columns={
                    "close": "mkt_close",
                    "volume": "mkt_volume",
                    "high": "mkt_high",
                    "low": "mkt_low",
                }
            )
            df_mkt["mkt_return"] = np.log(df_mkt["mkt_close"] / df_mkt["mkt_close"].shift(1))
        else:
            logger.warning("VNINDEX benchmark unavailable. Macro indicators will be skipped.")

        clean_dict: Dict[str, pd.DataFrame] = {}
        for idx, ticker in enumerate(self.tickers, 1):
            logger.info("[%02d/%02d] Processing constituent: %s", idx, len(self.tickers), ticker)
            df_raw = self._fetch_history(ticker)
            if not df_raw.empty and len(df_raw) > 50:
                clean_df = self._clean_and_adjust(df_raw, df_mkt)
                clean_dict[ticker] = clean_df
            else:
                logger.warning("Skipping %s (Insufficient bars or empty)", ticker)
            time.sleep(0.5)

        logger.info("Data Preparation Complete: %d/%d assets operational.", len(clean_dict), len(self.tickers))
        return clean_dict