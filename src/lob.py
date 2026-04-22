"""
LOB metrics and trade classification.

Core functions for computing microstructure measures from LOBSTER data:
midprice, spread, depth, Lee-Ready trade classification, and time-bin
aggregation for downstream analysis.
"""

import numpy as np
import pandas as pd


def compute_midprice(orderbook: pd.DataFrame) -> pd.Series:
    """Best bid-ask midpoint."""
    return (orderbook["ask_price_1"] + orderbook["bid_price_1"]) / 2


def compute_spread(orderbook: pd.DataFrame) -> pd.Series:
    """Absolute bid-ask spread."""
    return orderbook["ask_price_1"] - orderbook["bid_price_1"]


def compute_relative_spread(orderbook: pd.DataFrame) -> pd.Series:
    """Spread in basis points."""
    mid = compute_midprice(orderbook)
    return (compute_spread(orderbook) / mid) * 1e4


def compute_depth(orderbook: pd.DataFrame, levels: int = 5) -> pd.DataFrame:
    """Aggregate depth across the first `levels` price levels."""
    bid_d = sum(orderbook[f"bid_size_{i}"] for i in range(1, levels + 1)
                if f"bid_size_{i}" in orderbook.columns)
    ask_d = sum(orderbook[f"ask_size_{i}"] for i in range(1, levels + 1)
                if f"ask_size_{i}" in orderbook.columns)
    total = bid_d + ask_d
    return pd.DataFrame({
        "bid_depth": bid_d,
        "ask_depth": ask_d,
        "total_depth": total,
        "depth_imbalance": (bid_d - ask_d) / total.replace(0, np.nan),
    })


def classify_trades(
    messages: pd.DataFrame, orderbook: pd.DataFrame
) -> pd.DataFrame:
    """
    Classify executed trades as buyer- or seller-initiated using Lee-Ready.

    Quote rule: trade price vs. midprice.
    Tick rule: fallback based on price change direction.
    """
    trades = messages[messages["type"] == 4].copy()
    if trades.empty:
        return trades

    midprice = compute_midprice(orderbook)
    mid_aligned = midprice.reindex(trades.index, method="ffill")

    # quote rule
    trades["sign"] = np.where(
        trades["price"] > mid_aligned, 1,
        np.where(trades["price"] < mid_aligned, -1, 0)
    )

    # tick rule for midpoint trades
    at_mid = trades["sign"] == 0
    if at_mid.any():
        tick_sign = np.sign(trades["price"].diff()).replace(0, np.nan).ffill().fillna(1)
        trades.loc[at_mid, "sign"] = tick_sign[at_mid].astype(int)

    trades["signed_volume"] = trades["sign"] * trades["size"]
    return trades


def aggregate_intervals(
    trades: pd.DataFrame,
    messages: pd.DataFrame,
    orderbook: pd.DataFrame,
    interval: int = 300,
    t_open: float = 34_200.0,
    min_trades: int = 5,
) -> pd.DataFrame:
    """
    Aggregate trade data into fixed-length time bins.

    For each bin: net order flow, midprice change, trade count, volume.
    Bins with fewer than `min_trades` are dropped.
    """
    trades = trades.copy()
    trades["bin"] = ((trades["time"] - t_open) // interval).astype(int)

    # midprice at bin boundaries from full orderbook
    midprice = compute_midprice(orderbook)
    msg_bins = ((messages["time"] - t_open) // interval).astype(int)
    mid_df = pd.DataFrame({"mid": midprice, "bin": msg_bins})
    mid_first = mid_df.groupby("bin")["mid"].first()
    mid_last = mid_df.groupby("bin")["mid"].last()

    grouped = trades.groupby("bin").agg(
        order_flow=("signed_volume", "sum"),
        n_trades=("size", "count"),
        volume=("size", "sum"),
    )

    grouped["mid_start"] = mid_first
    grouped["mid_end"] = mid_last
    grouped["delta_p"] = grouped["mid_end"] - grouped["mid_start"]
    grouped = grouped[grouped["n_trades"] >= min_trades].copy()
    grouped["bin_time"] = t_open + grouped.index * interval

    return grouped.reset_index()
