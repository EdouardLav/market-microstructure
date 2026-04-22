"""
Market Microstructure Analysis — Main Pipeline

Runs the full analysis chain:
  1. Load LOBSTER data (real if available, synthetic otherwise)
  2. Compute LOB metrics and classify trades
  3. Estimate Kyle's lambda (price impact)
  4. Compute VPIN (informed trading probability)
  5. Optimize execution with Almgren-Chriss
  6. Compare baseline vs VPIN-adaptive trajectories
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# project imports
import config as cfg
from src.lobster_loader import load_lobster, find_lobster_files
from src.data_generator import generate_lobster_data
from src.lob import (
    compute_midprice, compute_spread, compute_relative_spread,
    compute_depth, classify_trades, aggregate_intervals,
)
from src.kyle import estimate_kyle_lambda, kyle_lambda_intraday, kyle_lambda_rolling
from src.vpin import vpin_analysis, detect_vpin_events
from src.almgren_chriss import ACParams, optimal_trajectory, execution_cost, adaptive_execution
from src.plots import (
    plot_lob_summary, plot_kyle_regression, plot_kyle_intraday,
    plot_vpin, plot_execution_comparison, plot_kyle_residuals,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    cfg.DATA_DIR.mkdir(exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    cfg.FIGURES_DIR.mkdir(exist_ok=True)

    # ── 1. Data Loading ───────────────────────────────────────────
    # check for real LOBSTER files first
    msg_path, ob_path = find_lobster_files(cfg.DATA_DIR)

    if msg_path and ob_path:
        separator("1. Loading Real LOBSTER Data")
        messages, orderbook = load_lobster(msg_path, ob_path, n_levels=cfg.N_LEVELS)
    else:
        separator("1. Generating Synthetic LOBSTER Data")
        print("No LOBSTER files found in data/ — using synthetic data.")
        print("To use real data, place message and orderbook CSVs in data/\n")
        messages, orderbook = generate_lobster_data(
            ticker=cfg.TICKER,
            date=cfg.DATE,
            n_events=cfg.N_EVENTS,
            initial_price=cfg.INITIAL_PRICE,
            tick_size=cfg.TICK_SIZE,
            n_levels=cfg.N_LEVELS,
            seed=cfg.SEED,
        )

    # ── 2. LOB Metrics ────────────────────────────────────────────
    separator("2. Computing LOB Metrics")

    midprice = compute_midprice(orderbook)
    spread = compute_spread(orderbook)
    rel_spread = compute_relative_spread(orderbook)
    depth = compute_depth(orderbook)

    print(f"Midprice — mean: ${midprice.mean():.2f}, "
          f"std: ${midprice.std():.4f}")
    print(f"Spread — mean: {spread.mean()*100:.2f} cents, "
          f"median: {spread.median()*100:.2f} cents")
    print(f"Relative spread — mean: {rel_spread.mean():.2f} bps")
    print(f"Depth imbalance — mean: {depth['depth_imbalance'].mean():.4f}, "
          f"std: {depth['depth_imbalance'].std():.4f}")

    # classify trades
    trades = classify_trades(messages, orderbook)
    n_buys = (trades["sign"] == 1).sum()
    n_sells = (trades["sign"] == -1).sum()
    print(f"\nTrade classification (Lee-Ready):")
    print(f"  Buys: {n_buys:,}  |  Sells: {n_sells:,}  |  "
          f"Ratio: {n_buys/(n_buys+n_sells):.1%}")

    # aggregate into intervals
    intervals = aggregate_intervals(
        trades, messages, orderbook,
        interval=cfg.KYLE_INTERVAL,
        t_open=cfg.T_OPEN,
        min_trades=cfg.KYLE_MIN_TRADES,
    )
    print(f"  Aggregated into {len(intervals)} intervals "
          f"({cfg.KYLE_INTERVAL}s each)")

    # plot
    fig = plot_lob_summary(orderbook, messages, cfg.FIGURES_DIR / "lob_summary.png")
    print(f"  Saved: lob_summary.png")

    # ── 3. Kyle's Lambda ──────────────────────────────────────────
    separator("3. Estimating Kyle's Lambda")

    kyle = estimate_kyle_lambda(intervals)

    print(f"λ (Kyle's Lambda): {kyle['lambda_']:.6e}")
    print(f"  t-statistic:     {kyle['t_stat']:.3f}")
    print(f"  p-value:         {kyle['p_value']:.4e}")
    print(f"  R²:              {kyle['r_squared']:.4f}")
    print(f"  Adj. R²:         {kyle['adj_r_squared']:.4f}")
    print(f"  Observations:    {kyle['n_obs']}")

    print(f"\nOLS Summary:")
    print(kyle["model"].summary().tables[1])

    # intraday variation
    intraday = kyle_lambda_intraday(intervals, n_windows=6)
    print(f"\nIntraday λ variation:")
    print(intraday[["window", "lambda", "t_stat", "r_squared"]].to_string(index=False))

    # plots
    plot_kyle_regression(intervals, kyle, cfg.FIGURES_DIR / "kyle_regression.png")
    plot_kyle_intraday(intraday, cfg.FIGURES_DIR / "kyle_intraday.png")
    plot_kyle_residuals(kyle, cfg.FIGURES_DIR / "kyle_residuals.png")
    print(f"\n  Saved: kyle_regression.png, kyle_intraday.png, kyle_residuals.png")

    # ── 4. VPIN ───────────────────────────────────────────────────
    separator("4. Computing VPIN")

    # auto-scale bucket size: target ~2000-3000 buckets
    total_vol = trades["size"].sum()
    bucket_size = cfg.VPIN_BUCKET_SIZE or max(500, int(total_vol / 2500))
    print(f"Bucket size: {bucket_size:,} shares (total volume: {total_vol:,})")

    buckets = vpin_analysis(
        trades, bucket_size=bucket_size, window=cfg.VPIN_WINDOW,
    )
    vpin_clean = buckets["vpin"].dropna()

    print(f"Volume buckets: {len(buckets):,}")
    print(f"VPIN — mean: {vpin_clean.mean():.4f}, "
          f"std: {vpin_clean.std():.4f}, "
          f"max: {vpin_clean.max():.4f}")
    print(f"  P25: {vpin_clean.quantile(0.25):.4f}  "
          f"P50: {vpin_clean.quantile(0.50):.4f}  "
          f"P75: {vpin_clean.quantile(0.75):.4f}  "
          f"P95: {vpin_clean.quantile(0.95):.4f}")

    events = detect_vpin_events(buckets, threshold_percentile=95)
    print(f"\nHigh-VPIN events (>95th pctl): {len(events)}")
    if len(events) > 0:
        print(f"  Threshold: {events['threshold'].iloc[0]:.4f}")

    # VPIN-lambda correlation
    # map vpin buckets to kyle intervals for correlation analysis
    buckets_with_time = buckets.dropna(subset=["vpin"]).copy()
    rolling_kyle = kyle_lambda_rolling(intervals, window=15, min_periods=8)
    if len(rolling_kyle) > 10:
        # rough alignment: interpolate
        from scipy.interpolate import interp1d
        kyle_interp = interp1d(
            rolling_kyle["bin"].values,
            rolling_kyle["lambda"].values,
            bounds_error=False, fill_value="extrapolate",
        )
        bucket_bins = ((buckets_with_time["t_start"].values - cfg.T_OPEN) / cfg.KYLE_INTERVAL)
        buckets_with_time["kyle_lambda"] = kyle_interp(bucket_bins)
        corr = buckets_with_time[["vpin", "kyle_lambda"]].corr().iloc[0, 1]
        print(f"  VPIN-Lambda correlation: {corr:.3f}")

    plot_vpin(buckets, cfg.FIGURES_DIR / "vpin.png")
    print(f"\n  Saved: vpin.png")

    # ── 5. Almgren-Chriss Execution ───────────────────────────────
    separator("5. Almgren-Chriss Optimal Execution")

    # calibrate using estimated Kyle's lambda
    gamma_est = max(kyle["lambda_"], 1e-8)  # permanent impact from Kyle regression
    print(f"Using γ (permanent impact) = {gamma_est:.2e} from Kyle estimation")

    ac_params = ACParams(
        X0=cfg.AC_INITIAL_INVENTORY,
        T=1.0,
        N=cfg.AC_N_STEPS,
        sigma=cfg.AC_SIGMA,
        gamma=gamma_est,
        eta=cfg.AC_TEMP_IMPACT,
        risk_aversion=0.05,
    )

    # baseline trajectory
    avg_price = midprice.mean()
    base_traj = optimal_trajectory(ac_params)
    base_cost = execution_cost(ac_params, base_traj, price=avg_price)

    print(f"\nBaseline execution ({cfg.AC_INITIAL_INVENTORY:,} shares, "
          f"{cfg.AC_N_STEPS} intervals):")
    print(f"  Permanent impact cost: ${base_cost['permanent_cost']:,.2f}")
    print(f"  Temporary impact cost: ${base_cost['temporary_cost']:,.2f}")
    print(f"  Total expected cost:   ${base_cost['total_expected_cost']:,.2f}")
    print(f"  Timing risk (var):     ${base_cost['timing_risk']:,.2f}")
    print(f"  Cost in bps:           {base_cost['cost_bps']:.1f} bps")

    # adaptive trajectory using VPIN
    adaptive_traj = adaptive_execution(
        ac_params,
        vpin_series=buckets["vpin"].bfill().fillna(0),
        vpin_threshold=float(vpin_clean.quantile(0.50)),
        urgency_multiplier=8.0,
    )

    print(f"\nVPIN-adaptive execution:")
    adapt_cost = execution_cost(ac_params, adaptive_traj, price=avg_price)
    print(f"  Total expected cost:   ${adapt_cost['total_expected_cost']:,.2f}")
    print(f"  Timing risk (var):     ${adapt_cost['timing_risk']:,.2f}")

    # risk reduction
    risk_reduction = (base_cost["timing_risk"] - adapt_cost["timing_risk"]) / base_cost["timing_risk"]
    print(f"  Risk reduction vs baseline: {risk_reduction:.1%}")

    plot_execution_comparison(
        base_traj, adaptive_traj,
        cfg.FIGURES_DIR / "execution_comparison.png",
    )
    print(f"\n  Saved: execution_comparison.png")

    # ── 6. Summary ────────────────────────────────────────────────
    separator("Summary")

    summary = pd.DataFrame([{
        "Metric": "Kyle's Lambda (λ)",
        "Value": f"{kyle['lambda_']:.2e}",
        "Detail": f"t={kyle['t_stat']:.2f}, p={kyle['p_value']:.1e}, R²={kyle['r_squared']:.3f}",
    }, {
        "Metric": "VPIN (mean)",
        "Value": f"{vpin_clean.mean():.4f}",
        "Detail": f"95th pctl: {vpin_clean.quantile(0.95):.4f}, {len(events)} high-VPIN events",
    }, {
        "Metric": "Execution Cost",
        "Value": f"{base_cost['cost_bps']:.1f} bps",
        "Detail": f"Perm: ${base_cost['permanent_cost']:,.0f} + Temp: ${base_cost['temporary_cost']:,.0f}",
    }, {
        "Metric": "VPIN Adaptation",
        "Value": f"{risk_reduction:.1%} risk reduction",
        "Detail": "Dynamic γ adjustment based on informed trading signal",
    }])

    print(summary.to_string(index=False))

    # save summary
    summary.to_csv(cfg.OUTPUT_DIR / "summary.csv", index=False)
    buckets.to_csv(cfg.OUTPUT_DIR / "vpin_buckets.csv", index=False)
    intervals.to_csv(cfg.OUTPUT_DIR / "kyle_intervals.csv", index=False)
    intraday.to_csv(cfg.OUTPUT_DIR / "kyle_intraday.csv", index=False)
    base_traj.to_csv(cfg.OUTPUT_DIR / "trajectory_baseline.csv", index=False)
    adaptive_traj.to_csv(cfg.OUTPUT_DIR / "trajectory_adaptive.csv", index=False)

    print(f"\nAll outputs saved to {cfg.OUTPUT_DIR}/")
    print(f"All figures saved to {cfg.FIGURES_DIR}/")


if __name__ == "__main__":
    main()
