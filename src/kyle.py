"""
Kyle's Lambda estimation.

Implements the Kyle (1985) price impact regression:
    ΔP_t = α + λ × OF_t + ε_t

where OF is the net order flow (signed volume) aggregated over fixed
intervals. Lambda measures the permanent price impact per unit of
order flow — higher values indicate thinner liquidity or greater
information asymmetry.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Any


def estimate_kyle_lambda(
    intervals: pd.DataFrame,
) -> Dict[str, Any]:
    """
    OLS regression of midprice changes on net order flow.

    Parameters
    ----------
    intervals : pd.DataFrame
        Output of `aggregate_intervals` with columns [order_flow, delta_p].

    Returns
    -------
    dict with keys:
        lambda_    : float  — estimated price impact coefficient
        alpha      : float  — intercept
        t_stat     : float  — t-statistic for lambda
        p_value    : float  — p-value for lambda
        r_squared  : float  — regression R²
        n_obs      : int    — number of observations
        residuals  : array  — OLS residuals
        model      : statsmodels RegressionResultsWrapper
    """
    df = intervals.dropna(subset=["order_flow", "delta_p"]).copy()
    if len(df) < 10:
        raise ValueError(f"Too few intervals ({len(df)}) for reliable estimation")

    X = sm.add_constant(df["order_flow"].values)
    y = df["delta_p"].values

    # skip degenerate cases (zero variance in y or X)
    if np.std(y) < 1e-15 or np.std(df["order_flow"].values) < 1e-15:
        raise ValueError("Zero variance in order flow or price changes")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X).fit(cov_type="HC1")

    return {
        "lambda_": model.params[1],
        "alpha": model.params[0],
        "t_stat": model.tvalues[1],
        "p_value": model.pvalues[1],
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "n_obs": int(model.nobs),
        "residuals": model.resid,
        "model": model,
    }


def kyle_lambda_intraday(
    intervals: pd.DataFrame,
    n_windows: int = 6,
) -> pd.DataFrame:
    """
    Estimate lambda over non-overlapping intraday windows.

    Splits the trading day into `n_windows` equal segments and runs
    the Kyle regression on each. Useful for detecting how information
    asymmetry varies throughout the day (typically higher at the open).
    """
    df = intervals.dropna(subset=["order_flow", "delta_p"]).copy()
    df = df.sort_values("bin")

    bins_per_window = max(1, len(df) // n_windows)
    results = []

    for i in range(n_windows):
        start = i * bins_per_window
        end = start + bins_per_window if i < n_windows - 1 else len(df)
        chunk = df.iloc[start:end]

        if len(chunk) < 5:
            continue

        try:
            est = estimate_kyle_lambda(chunk)
            results.append({
                "window": i + 1,
                "n_obs": est["n_obs"],
                "lambda": est["lambda_"],
                "t_stat": est["t_stat"],
                "r_squared": est["r_squared"],
                "bin_start": chunk["bin"].iloc[0],
                "bin_end": chunk["bin"].iloc[-1],
            })
        except ValueError:
            continue

    return pd.DataFrame(results)


def kyle_lambda_rolling(
    intervals: pd.DataFrame,
    window: int = 20,
    min_periods: int = 10,
) -> pd.DataFrame:
    """
    Rolling-window Kyle lambda estimation.

    Runs the regression over a rolling window of `window` intervals.
    Captures the time-varying nature of price impact.
    """
    df = intervals.dropna(subset=["order_flow", "delta_p"]).copy()
    df = df.sort_values("bin").reset_index(drop=True)

    records = []
    for i in range(min_periods, len(df)):
        start = max(0, i - window)
        chunk = df.iloc[start:i + 1]

        if len(chunk) < min_periods:
            continue

        try:
            est = estimate_kyle_lambda(chunk)
            records.append({
                "idx": i,
                "bin": df.loc[i, "bin"],
                "lambda": est["lambda_"],
                "t_stat": est["t_stat"],
                "r_squared": est["r_squared"],
            })
        except ValueError:
            continue

    return pd.DataFrame(records)
