"""
Volume-Synchronized Probability of Informed Trading (VPIN).

Implementation based on Easley, López de Prado & O'Hara (2012).

Key ideas:
- Trades are bucketed by volume rather than time, so each bucket
  captures the same amount of market activity regardless of speed.
- Within each bucket, volume is split into buy- and sell-initiated
  using Bulk Volume Classification (BVC).
- VPIN is the rolling average of absolute order imbalance across
  the last N buckets.

High VPIN signals elevated probability of informed trading — a
leading indicator of adverse selection and potential flash crashes.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def bulk_volume_classification(
    prices: np.ndarray,
    volumes: np.ndarray,
    sigma: float = None,
) -> np.ndarray:
    """
    Bulk Volume Classification (BVC) from López de Prado (2012).

    Instead of classifying individual trades, BVC uses the standardized
    price change to probabilistically assign volume to buy/sell.

    For each trade:
        buy_fraction = Φ(ΔP / σ)
    where Φ is the standard normal CDF and σ is estimated from
    recent price changes.

    Returns array of signed volumes (+buy, -sell).
    """
    dp = np.diff(prices, prepend=prices[0])

    if sigma is None:
        sigma = np.std(dp[dp != 0])
        if sigma == 0 or np.isnan(sigma):
            sigma = 1e-6

    z = dp / sigma
    buy_frac = norm.cdf(z)

    buy_vol = volumes * buy_frac
    sell_vol = volumes * (1 - buy_frac)

    return buy_vol, sell_vol


def create_volume_buckets(
    trades: pd.DataFrame,
    bucket_size: int = 1000,
) -> pd.DataFrame:
    """
    Aggregate trades into volume buckets of fixed size.

    Each bucket contains approximately `bucket_size` shares.
    Trades are processed chronologically; when cumulative volume
    reaches the threshold, the bucket is closed and a new one starts.
    """
    trades = trades.sort_values("time").reset_index(drop=True)

    buy_vol, sell_vol = bulk_volume_classification(
        trades["price"].values,
        trades["size"].values,
    )

    cum_vol = np.cumsum(trades["size"].values)
    bucket_ids = (cum_vol // bucket_size).astype(int)

    df = pd.DataFrame({
        "bucket": bucket_ids,
        "buy_vol": buy_vol,
        "sell_vol": sell_vol,
        "volume": trades["size"].values,
        "time": trades["time"].values,
        "price": trades["price"].values,
    })

    buckets = df.groupby("bucket").agg(
        buy_vol=("buy_vol", "sum"),
        sell_vol=("sell_vol", "sum"),
        volume=("volume", "sum"),
        t_start=("time", "first"),
        t_end=("time", "last"),
        n_trades=("volume", "count"),
        vwap=("price", "mean"),  # approximate — proper VWAP would weight by volume
    )

    buckets["imbalance"] = np.abs(buckets["buy_vol"] - buckets["sell_vol"])
    buckets["duration"] = buckets["t_end"] - buckets["t_start"]

    return buckets


def compute_vpin(
    buckets: pd.DataFrame,
    window: int = 50,
) -> pd.Series:
    """
    Compute VPIN as a rolling average of order imbalance.

    VPIN_n = (1/N) × Σ |V_buy - V_sell|_i / V_bucket

    Higher values indicate greater probability of informed trading.
    """
    bucket_vol = buckets["volume"]
    imbalance = buckets["imbalance"]

    vpin = (imbalance / bucket_vol).rolling(window=window, min_periods=window // 2).mean()

    return vpin


def vpin_analysis(
    trades: pd.DataFrame,
    bucket_size: int = 1000,
    window: int = 50,
) -> pd.DataFrame:
    """
    End-to-end VPIN computation from raw trade data.

    Returns the bucket-level DataFrame with VPIN attached.
    """
    buckets = create_volume_buckets(trades, bucket_size)
    buckets["vpin"] = compute_vpin(buckets, window).values

    return buckets


def detect_vpin_events(
    buckets: pd.DataFrame,
    threshold_percentile: float = 95,
) -> pd.DataFrame:
    """
    Flag buckets where VPIN exceeds a threshold.

    These are candidate periods for informed trading activity.
    """
    threshold = np.nanpercentile(buckets["vpin"].dropna(), threshold_percentile)
    events = buckets[buckets["vpin"] >= threshold].copy()
    events["threshold"] = threshold

    return events
