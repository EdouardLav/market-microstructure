"""
Plotting functions for the microstructure analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path


# consistent style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 150,
})

COLORS = {
    "bid": "#0F6E56",
    "ask": "#993C1D",
    "mid": "#185FA5",
    "vpin": "#534AB7",
    "impact": "#D85A30",
    "traj_base": "#185FA5",
    "traj_adapt": "#993556",
    "neutral": "#5F5E5A",
}


def _seconds_to_time(s):
    """Convert seconds-from-midnight to HH:MM string."""
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    return f"{h:02d}:{m:02d}"


def plot_lob_summary(orderbook, messages, save_path=None):
    """Midprice and depth imbalance over the trading day."""
    from src.lob import compute_midprice, compute_depth

    mid = compute_midprice(orderbook)
    depth = compute_depth(orderbook)
    times = messages["time"].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # midprice
    ax1.plot(times, mid, linewidth=0.5, color=COLORS["mid"])
    ax1.set_ylabel("Midprice ($)")
    ax1.set_title("Intraday Midprice and Order Book Depth Imbalance")

    # depth imbalance
    imb = depth["depth_imbalance"].values
    imb_ma = pd.Series(imb).rolling(500, min_periods=50).mean()
    ax2.fill_between(times, imb, 0, alpha=0.15, color=COLORS["neutral"])
    ax2.plot(times, imb_ma, linewidth=1.5, color=COLORS["neutral"], label="500-event MA")
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_ylabel("Depth Imbalance (bid−ask)/total")
    ax2.set_xlabel("Time")
    ax2.legend(fontsize=9)

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _seconds_to_time(x)))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_kyle_regression(intervals, kyle_result, save_path=None):
    """Scatter plot of ΔP vs order flow with regression line."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        intervals["order_flow"], intervals["delta_p"],
        alpha=0.4, s=20, color=COLORS["bid"], edgecolors="none",
    )

    # regression line
    of_range = np.linspace(intervals["order_flow"].min(), intervals["order_flow"].max(), 100)
    fitted = kyle_result["alpha"] + kyle_result["lambda_"] * of_range
    ax.plot(of_range, fitted, color=COLORS["impact"], linewidth=2)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="-")

    ax.set_xlabel("Net Order Flow (signed volume)")
    ax.set_ylabel("Midprice Change ($)")
    ax.set_title(
        f"Kyle's Lambda Regression — "
        f"λ = {kyle_result['lambda_']:.2e}  "
        f"(t = {kyle_result['t_stat']:.2f}, "
        f"R² = {kyle_result['r_squared']:.3f})"
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_kyle_intraday(intraday_df, save_path=None):
    """Bar chart of Kyle's lambda across intraday windows."""
    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.bar(
        intraday_df["window"], intraday_df["lambda"],
        color=COLORS["bid"], alpha=0.7, edgecolor="white",
    )

    # highlight significant estimates
    for i, row in intraday_df.iterrows():
        if abs(row["t_stat"]) > 2:
            bars[i].set_alpha(1.0)

    ax.set_xlabel("Intraday Window")
    ax.set_ylabel("Kyle's λ")
    ax.set_title("Intraday Variation of Price Impact")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_vpin(buckets, save_path=None):
    """VPIN time series with event highlights."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    t = buckets["t_start"].values
    vpin = buckets["vpin"].values

    # VPIN
    ax1.plot(t, vpin, linewidth=0.8, color=COLORS["vpin"])
    threshold = np.nanpercentile(vpin[~np.isnan(vpin)], 95) if np.any(~np.isnan(vpin)) else 0.5
    ax1.axhline(threshold, color=COLORS["impact"], linestyle="--", linewidth=1, label=f"95th pctl ({threshold:.3f})")
    ax1.fill_between(t, vpin, threshold, where=vpin > threshold, alpha=0.3, color=COLORS["impact"])
    ax1.set_ylabel("VPIN")
    ax1.set_title("Volume-Synchronized Probability of Informed Trading")
    ax1.legend(fontsize=9)

    # bucket duration (shows volume acceleration)
    ax2.plot(t, buckets["duration"].values, linewidth=0.6, color=COLORS["neutral"], alpha=0.6)
    dur_ma = pd.Series(buckets["duration"].values).rolling(20, min_periods=5).mean()
    ax2.plot(t, dur_ma, linewidth=1.5, color=COLORS["neutral"], label="20-bucket MA")
    ax2.set_ylabel("Bucket Duration (s)")
    ax2.set_xlabel("Time")
    ax2.legend(fontsize=9)

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _seconds_to_time(x)))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_execution_comparison(base_traj, adaptive_traj, save_path=None):
    """Compare baseline and VPIN-adaptive execution trajectories."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 1])

    # trajectories
    ax1.plot(
        base_traj["time"], base_traj["holdings"],
        linewidth=2, color=COLORS["traj_base"], label="Baseline (constant γ)",
    )
    ax1.plot(
        adaptive_traj["time"], adaptive_traj["holdings"],
        linewidth=2, color=COLORS["traj_adapt"], linestyle="--",
        label="VPIN-adaptive (dynamic γ)",
    )
    ax1.set_ylabel("Remaining Position (shares)")
    ax1.set_title("Almgren-Chriss Optimal Execution: Baseline vs VPIN-Adaptive")
    ax1.legend(fontsize=10)

    # VPIN signal below
    if "vpin" in adaptive_traj.columns:
        vpin = adaptive_traj["vpin"].values
        ax2.fill_between(adaptive_traj["time"], vpin, alpha=0.4, color=COLORS["vpin"])
        ax2.plot(adaptive_traj["time"], vpin, linewidth=1, color=COLORS["vpin"])
        ax2.set_ylabel("VPIN")
        ax2.set_xlabel("Time (fraction of day)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_kyle_residuals(kyle_result, save_path=None):
    """Residual diagnostics for the Kyle regression."""
    resid = kyle_result["residuals"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # histogram
    axes[0].hist(resid, bins=30, color=COLORS["bid"], alpha=0.7, edgecolor="white", density=True)
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Residual Distribution")

    # QQ plot
    from scipy import stats
    stats.probplot(resid, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot")
    axes[1].get_lines()[0].set_color(COLORS["bid"])
    axes[1].get_lines()[1].set_color(COLORS["impact"])

    # autocorrelation
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(resid, ax=axes[2], lags=20, alpha=0.05, color=COLORS["bid"])
    axes[2].set_title("Residual ACF")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
