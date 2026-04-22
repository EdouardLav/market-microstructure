"""
LOBSTER data loader.

Reads raw LOBSTER message and orderbook CSV files and converts them
to the internal DataFrame format used by the analysis pipeline.

LOBSTER format (no headers):
  Message: time, type, order_id, size, price, direction
           - prices are dollar × 10000
           - type 5 = hidden execution, type 7 = trading halt
  Orderbook: ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...
           - prices are dollar × 10000
           - empty levels filled with ±9999999999
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_lobster(
    message_path: str,
    orderbook_path: str,
    n_levels: int = 10,
) -> tuple:
    """
    Load LOBSTER message and orderbook files.

    Converts prices from integer (× 10000) to float dollars.
    Filters out trading halts (type 7) and hidden executions (type 5).
    Replaces dummy prices (±9999999999) with NaN.

    Parameters
    ----------
    message_path : str or Path
        Path to the message CSV file.
    orderbook_path : str or Path
        Path to the orderbook CSV file.
    n_levels : int
        Number of LOB levels in the data.

    Returns
    -------
    messages : pd.DataFrame
        [time, type, order_id, size, price, direction]
    orderbook : pd.DataFrame
        [ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...]
    """
    # column names
    msg_cols = ["time", "type", "order_id", "size", "price", "direction"]
    ob_cols = []
    for i in range(1, n_levels + 1):
        ob_cols += [f"ask_price_{i}", f"ask_size_{i}", f"bid_price_{i}", f"bid_size_{i}"]

    # load
    messages = pd.read_csv(message_path, header=None, names=msg_cols)
    orderbook = pd.read_csv(orderbook_path, header=None, names=ob_cols)

    # convert prices: integer × 10000 → float dollars
    messages["price"] = messages["price"] / 10_000

    for i in range(1, n_levels + 1):
        for side in ["ask", "bid"]:
            col = f"{side}_price_{i}"
            orderbook[col] = orderbook[col] / 10_000
            # replace dummy values
            orderbook.loc[orderbook[col].abs() > 900_000, col] = np.nan

    # filter trading halts
    halt_mask = messages["type"] == 7
    if halt_mask.any():
        print(f"  Removed {halt_mask.sum()} trading halt messages")
        keep = ~halt_mask
        messages = messages[keep].reset_index(drop=True)
        orderbook = orderbook[keep].reset_index(drop=True)

    # treat hidden executions (type 5) as regular executions
    messages.loc[messages["type"] == 5, "type"] = 4

    n_exec = (messages["type"] == 4).sum()
    n_sub = (messages["type"] == 1).sum()
    n_canc = messages["type"].isin([2, 3]).sum()
    ticker = Path(message_path).name.split("_")[0]
    date = "-".join(Path(message_path).name.split("_")[1:4]).replace("_", "-")

    print(f"[{ticker} — LOBSTER] {len(messages):,} events — "
          f"{n_exec:,} executions, {n_sub:,} submissions, {n_canc:,} cancellations")
    print(f"  Price range: ${messages['price'].min():.2f} – ${messages['price'].max():.2f}")
    print(f"  Time range: {messages['time'].iloc[0]:.2f}s – {messages['time'].iloc[-1]:.2f}s")

    return messages, orderbook


def find_lobster_files(data_dir: str) -> tuple:
    """
    Search for LOBSTER message and orderbook files in a directory.

    Returns (message_path, orderbook_path) or (None, None) if not found.
    """
    data_dir = Path(data_dir)
    msg_files = list(data_dir.glob("*_message_*.csv"))
    ob_files = list(data_dir.glob("*_orderbook_*.csv"))

    if msg_files and ob_files:
        # match by ticker and date
        msg = sorted(msg_files)[0]
        ob = sorted(ob_files)[0]
        return str(msg), str(ob)

    return None, None
