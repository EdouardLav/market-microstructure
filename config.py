"""
Project configuration — all tunable parameters in one place.
"""

from pathlib import Path

# Paths
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = ROOT / "figures"

# Synthetic data
TICKER = "AAPL"
DATE = "2024-01-15"
N_EVENTS = 50_000
INITIAL_PRICE = 185.00
TICK_SIZE = 0.01
N_LEVELS = 10
SEED = 42

# Trading hours (seconds from midnight, EST)
T_OPEN = 34_200    # 09:30
T_CLOSE = 57_600   # 16:00

# Kyle's lambda
KYLE_INTERVAL = 300          # 5-minute aggregation (seconds)
KYLE_MIN_TRADES = 5          # minimum trades per interval

# VPIN
VPIN_BUCKET_SIZE = None            # auto-scale (or set manually, e.g. 1000)
VPIN_WINDOW = 50             # rolling window of buckets

# Almgren-Chriss
AC_INITIAL_INVENTORY = 100_000   # shares to liquidate
AC_N_STEPS = 20                  # number of trading intervals
AC_SIGMA = 0.02                  # daily volatility
AC_TEMP_IMPACT = 1e-5             # temporary impact coefficient (eta)
