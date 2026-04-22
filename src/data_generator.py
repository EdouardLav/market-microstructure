"""
Synthetic LOBSTER data generator.

Produces message and orderbook files that mimic the NASDAQ LOBSTER format.
Two-pass approach for speed:
  1. Pre-generate the midprice path with embedded order flow impact
  2. Build the LOB snapshots around the midprice
"""

import numpy as np
import pandas as pd


def generate_lobster_data(
    ticker="AAPL",
    date="2024-01-15",
    n_events=50_000,
    initial_price=185.00,
    tick_size=0.01,
    n_levels=10,
    seed=42,
    n_informed_episodes=5,
    t_open=34_200.0,
    t_close=57_600.0,
    impact_coeff=4e-7,
):
    """
    Generate synthetic LOBSTER-format data with price impact.

    The midprice responds to net order flow so that Kyle's lambda is
    nonzero by construction. Mean reversion prevents drift.

    Returns
    -------
    messages : pd.DataFrame
        [time, type, order_id, size, price, direction]
    orderbook : pd.DataFrame
        [ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...] for n_levels
    """
    rng = np.random.default_rng(seed)
    trading_secs = t_close - t_open

    # --- timestamps (sorted, U-shaped intensity) ---
    raw = rng.exponential(0.4, n_events)
    t_fracs = np.cumsum(raw) / np.sum(raw)
    timestamps = t_open + t_fracs * trading_secs

    # --- informed episodes ---
    ep_starts = sorted(rng.uniform(0.1, 0.85, n_informed_episodes))
    ep_durs = rng.uniform(0.02, 0.06, n_informed_episodes)
    ep_dirs = rng.choice([-1, 1], n_informed_episodes)

    is_informed = np.zeros(n_events, dtype=bool)
    informed_dir = np.zeros(n_events, dtype=int)
    for s, d, dr in zip(ep_starts, ep_durs, ep_dirs):
        mask = (t_fracs >= s) & (t_fracs <= s + d)
        is_informed |= mask
        informed_dir[mask] = dr

    # --- event types ---
    p_mkt = np.where(is_informed, 0.35, 0.20)
    p_lmt = np.where(is_informed, 0.40, 0.50)
    rolls = rng.random(n_events)
    is_execution = rolls < p_mkt
    is_submission = (rolls >= p_mkt) & (rolls < p_mkt + p_lmt)
    is_cancel = ~is_execution & ~is_submission

    # --- trade directions ---
    random_dirs = rng.choice([-1, 1], n_events)
    directions = np.where(is_informed & is_execution, informed_dir, random_dirs)

    # --- trade sizes ---
    intensity = 1.5 - 1.0 * np.sin(np.pi * t_fracs)
    intensity[is_informed] *= 2.5
    sizes = (rng.exponential(150 * intensity) + 50).astype(int)
    sizes = np.clip(sizes, 50, 2000)

    # --- midprice path with impact + mean reversion ---
    mid = np.zeros(n_events)
    mid[0] = initial_price
    kappa_mr = 10.0  # mean reversion speed
    sigma_noise = 0.06

    for i in range(1, n_events):
        dt = t_fracs[i] - t_fracs[i - 1]
        # impact from executions
        dp_impact = 0.0
        if is_execution[i]:
            dp_impact = impact_coeff * directions[i] * sizes[i]
        # mean reversion + noise
        dp_mr = kappa_mr * (initial_price - mid[i-1]) * dt
        dp_noise = sigma_noise * np.sqrt(dt) * rng.normal()
        mid[i] = mid[i-1] + dp_impact + dp_mr + dp_noise

    mid = np.round(mid, 2)

    # --- build output arrays ---
    msg_types = np.where(is_execution, 4, np.where(is_submission, 1, rng.choice([2, 3], n_events)))
    order_ids = np.arange(1, n_events + 1)

    # execution prices: at the touch
    exec_prices = np.where(
        directions == 1,
        mid + tick_size,  # buys hit the ask
        mid - tick_size,  # sells hit the bid
    )
    # limit order prices: at or away from touch
    offsets = rng.choice(n_levels, n_events, p=np.exp(-np.arange(n_levels)*0.5) / np.sum(np.exp(-np.arange(n_levels)*0.5)))
    limit_prices = np.where(
        directions == 1,
        mid - offsets * tick_size,
        mid + offsets * tick_size,
    )
    # cancel prices: near the touch
    cancel_prices = np.where(
        directions == 1,
        mid - rng.integers(1, n_levels, n_events) * tick_size,
        mid + rng.integers(1, n_levels, n_events) * tick_size,
    )

    prices = np.where(is_execution, exec_prices,
             np.where(is_submission, limit_prices, cancel_prices))
    prices = np.round(prices, 2)

    messages = pd.DataFrame({
        "time": timestamps,
        "type": msg_types,
        "order_id": order_ids,
        "size": sizes,
        "price": prices,
        "direction": directions,
    })

    # --- orderbook snapshots ---
    # for each event, build a synthetic LOB around the midprice
    ob_data = np.zeros((n_events, n_levels * 4))
    base_sizes = rng.exponential(250, (n_events, n_levels)) + 80

    for j in range(n_levels):
        # ask side
        ob_data[:, j*4] = mid + (j + 1) * tick_size       # ask price
        ob_data[:, j*4 + 1] = base_sizes[:, j]             # ask size
        # bid side
        ob_data[:, j*4 + 2] = mid - (j + 1) * tick_size   # bid price
        ob_data[:, j*4 + 3] = base_sizes[:, j] * rng.uniform(0.7, 1.3, n_events)

    ob_data = np.round(ob_data, 2)
    ob_data[:, 1::2] = np.maximum(ob_data[:, 1::2].astype(int), 10)  # min size 10

    ob_cols = []
    for j in range(1, n_levels + 1):
        ob_cols += [f"ask_price_{j}", f"ask_size_{j}", f"bid_price_{j}", f"bid_size_{j}"]

    orderbook = pd.DataFrame(ob_data, columns=ob_cols)
    # fix size columns to int
    for c in ob_cols:
        if "size" in c:
            orderbook[c] = orderbook[c].astype(int)

    n_exec = is_execution.sum()
    n_sub = is_submission.sum()
    n_canc = is_cancel.sum()
    print(f"[{ticker} {date}] {n_events:,} events — {n_exec:,} executions, "
          f"{n_sub:,} submissions, {n_canc:,} cancellations")

    return messages, orderbook
