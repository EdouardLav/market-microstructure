"""
Almgren-Chriss Optimal Execution Model (2000).

Solves the optimal liquidation trajectory for a large position,
balancing two competing costs:
- Market impact (permanent + temporary): trading fast moves the price
- Timing risk: trading slow exposes you to adverse price moves

The solution depends on risk aversion (gamma), volatility (sigma),
and the impact parameters calibrated from empirical data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class ACParams:
    """Almgren-Chriss model parameters."""
    X0: float           # initial inventory (shares)
    T: float            # trading horizon (fraction of day, e.g. 1.0)
    N: int              # number of trading intervals
    sigma: float        # daily price volatility
    gamma: float        # permanent impact coefficient (Kyle's lambda)
    eta: float          # temporary impact coefficient
    risk_aversion: float  # trader's risk aversion parameter

    @property
    def tau(self) -> float:
        """Length of each trading interval."""
        return self.T / self.N


def optimal_trajectory(params: ACParams) -> pd.DataFrame:
    """
    Compute the Almgren-Chriss optimal execution trajectory.

    The optimal holdings at time step k are:

        x_k = X0 * sinh(kappa * (T - t_k)) / sinh(kappa * T)

    where kappa = sqrt(risk_aversion * sigma^2 / eta) captures
    the urgency of the trader.

    Returns DataFrame with columns: [step, time, holdings, trade_rate]
    """
    tau = params.tau

    # kappa: urgency parameter
    # higher risk_aversion -> higher kappa -> faster execution
    kappa_sq = params.risk_aversion * params.sigma**2 / (params.eta + 1e-12)
    kappa = np.sqrt(max(kappa_sq, 1e-8))

    steps = np.arange(params.N + 1)
    times = steps * tau

    # optimal holdings path
    denom = np.sinh(kappa * params.T)
    if abs(denom) < 1e-12:
        # risk-neutral case: linear liquidation
        holdings = params.X0 * (1 - times / params.T)
    else:
        holdings = params.X0 * np.sinh(kappa * (params.T - times)) / denom

    # trade sizes (shares sold per interval)
    trade_sizes = -np.diff(holdings, prepend=holdings[0])
    trade_sizes[0] = 0  # no trade at t=0

    # trade rate (shares per unit time)
    trade_rate = np.zeros_like(holdings)
    trade_rate[1:] = -np.diff(holdings) / tau

    return pd.DataFrame({
        "step": steps,
        "time": times,
        "holdings": holdings,
        "trade_size": trade_sizes,
        "trade_rate": trade_rate,
    })


def execution_cost(
    params: ACParams,
    trajectory: Optional[pd.DataFrame] = None,
    price: float = 185.0,
) -> dict:
    """
    Compute expected execution cost and variance for a given trajectory.

    Costs:
    - Permanent impact: gamma * X0^2 / 2
      (this is invariant to the trajectory — you always pay it)
    - Temporary impact: eta * sum(n_k^2 / tau)
      (depends on how aggressively you trade)
    - Timing risk: sigma^2 * sum(x_k^2 * tau)
      (depends on how long you hold the position)
    """
    if trajectory is None:
        trajectory = optimal_trajectory(params)

    tau = params.tau
    holdings = trajectory["holdings"].values

    # compute trade sizes from holdings if not provided
    if "trade_size" in trajectory.columns:
        trade_sizes = trajectory["trade_size"].values
    else:
        trade_sizes = np.zeros_like(holdings)
        trade_sizes[1:] = -np.diff(holdings)

    # permanent impact cost (path-independent)
    perm_cost = 0.5 * params.gamma * params.X0**2

    # temporary impact cost
    n_k = trade_sizes[1:]  # skip t=0
    temp_cost = params.eta * np.sum(n_k**2 / tau)

    # timing risk (variance of cost)
    # E[variance] = sigma^2 * tau * sum(x_k^2) for k=1..N
    risk = params.sigma**2 * tau * np.sum(holdings[1:]**2)

    # mean-variance objective
    total_cost = perm_cost + temp_cost
    obj = total_cost + params.risk_aversion * risk

    return {
        "permanent_cost": perm_cost,
        "temporary_cost": temp_cost,
        "total_expected_cost": total_cost,
        "timing_risk": risk,
        "objective": obj,
        "cost_bps": total_cost / (params.X0 * price) * 1e4,
    }


def compare_trajectories(
    base_params: ACParams,
    risk_aversions: list,
) -> pd.DataFrame:
    """
    Compare optimal trajectories across different risk aversion levels.

    Useful for visualizing the impact-vs-risk tradeoff.
    """
    results = []
    for ra in risk_aversions:
        p = ACParams(
            X0=base_params.X0, T=base_params.T, N=base_params.N,
            sigma=base_params.sigma, gamma=base_params.gamma,
            eta=base_params.eta, risk_aversion=ra,
        )
        traj = optimal_trajectory(p)
        cost = execution_cost(p, traj)
        results.append({
            "risk_aversion": ra,
            "temp_cost": cost["temporary_cost"],
            "timing_risk": cost["timing_risk"],
            "total_cost": cost["total_expected_cost"],
            "objective": cost["objective"],
        })

    return pd.DataFrame(results)


def adaptive_execution(
    params: ACParams,
    vpin_series: pd.Series,
    vpin_threshold: float = 0.6,
    urgency_multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Modify the execution trajectory based on real-time VPIN signals.

    When VPIN exceeds the threshold, increase risk aversion (trade faster)
    to reduce exposure to informed flow. This is the bridge between
    the VPIN analysis and the execution framework.
    """
    # resample VPIN to match trading steps
    n_buckets = len(vpin_series)
    step_to_bucket = np.linspace(0, n_buckets - 1, params.N + 1).astype(int)
    vpin_at_steps = vpin_series.iloc[step_to_bucket].values

    # build trajectory step by step
    holdings = np.zeros(params.N + 1)
    holdings[0] = params.X0

    for k in range(params.N):
        remaining_steps = params.N - k
        remaining_time = remaining_steps * params.tau

        # adjust risk aversion based on current VPIN
        ra = params.risk_aversion
        if not np.isnan(vpin_at_steps[k]) and vpin_at_steps[k] > vpin_threshold:
            ra *= urgency_multiplier

        # recompute optimal trade for remaining position
        kappa_sq = ra * params.sigma**2 / (params.eta + 1e-12)
        kappa = np.sqrt(max(kappa_sq, 1e-8))

        denom = np.sinh(kappa * remaining_time)
        if abs(denom) < 1e-12:
            # linear
            frac = 1.0 / remaining_steps
        else:
            frac = 1 - np.sinh(kappa * (remaining_time - params.tau)) / denom

        trade = holdings[k] * frac
        holdings[k + 1] = holdings[k] - trade

    return pd.DataFrame({
        "step": np.arange(params.N + 1),
        "time": np.arange(params.N + 1) * params.tau,
        "holdings": holdings,
        "vpin": vpin_at_steps,
    })
