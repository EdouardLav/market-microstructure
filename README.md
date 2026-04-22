# Market Microstructure Analysis — LOBSTER Data

Empirical analysis of limit order book dynamics using NASDAQ LOBSTER Level-3 tick data (AAPL, 400K events). The project covers three interconnected topics: Kyle's lambda estimation, VPIN computation, and Almgren-Chriss optimal execution — forming a pipeline where each output calibrates the next.

## Results

| Metric | Value | Detail |
|--|--|--|
| Kyle's λ | 3.38 × 10⁻⁵ | t = 7.63, p < 10⁻¹⁴, R² = 0.45 |
| λ at open vs midday | 7.4e-5 vs 0.8e-5 | 2.6× higher impact at the open |
| Buy/Sell ratio | 48.8% / 51.2% | Net selling pressure consistent with price decline |
| VPIN (mean) | 0.094 | 95th percentile: 0.126, 123 high-toxicity events |
| VPIN-λ correlation | ρ = 0.355 | Toxic flow coincides with higher price impact |
| Execution cost | 47.2 bps | $169K permanent + $106K temporary |
| VPIN adaptation | 44% risk reduction | Dynamic γ adjustment based on toxicity signal |

## Methodology

### Kyle's Lambda

OLS regression of midprice changes on net order flow following Kyle (1985):

$$\Delta P_t = \alpha + \lambda \cdot OF_t + \varepsilon_t$$

Trade classification uses the Lee-Ready (1991) algorithm (quote rule + tick rule fallback). Trades are aggregated into 5-minute intervals (78 observations). Estimation uses HC1 (White) heteroskedasticity-robust standard errors.

**Intraday variation:** lambda is highest at the open (7.4 × 10⁻⁵) and drops during midday (0.8 × 10⁻⁵), consistent with Kyle's prediction that information asymmetry peaks at market open. Window 5 is not statistically significant (t = 0.44).

### VPIN

Trades are aggregated into volume buckets (1,140 shares each, ~2,500 buckets) rather than fixed time intervals. Within each bucket, Bulk Volume Classification (BVC) uses the standardized price change to probabilistically assign volume:

$$V_{buy} = V \cdot \Phi\left(\frac{\Delta P}{\sigma}\right)$$

VPIN is the rolling average of absolute order imbalance across 50 buckets:

$$\text{VPIN} = \frac{1}{N} \sum_{i=1}^{N} \frac{|V_{buy,i} - V_{sell,i}|}{V_i}$$

Periods where VPIN exceeds the 95th percentile (0.126) are flagged as candidate informed trading events. Easley, López de Prado & O'Hara (2012) showed VPIN was a leading indicator of the May 6, 2010 Flash Crash.

### Almgren-Chriss Optimal Execution

The optimal liquidation trajectory for 100,000 shares minimizes a mean-variance objective:

$$x_k = X_0 \cdot \frac{\sinh(\kappa(T - t_k))}{\sinh(\kappa T)}, \quad \kappa = \sqrt{\frac{\lambda_{RA} \cdot \sigma^2}{\eta}}$$

The permanent impact parameter (γ) is calibrated directly from the Kyle regression — not assumed. The adaptive variant dynamically increases risk aversion (8×) when VPIN exceeds the median, causing the trader to liquidate faster during toxic flow episodes.

| | Expected Cost | Timing Risk |
|--|--|--|
| Baseline (constant γ) | $275K | $960K |
| VPIN-adaptive (dynamic γ) | $348K | $534K |

The adaptive strategy pays more in temporary impact but cuts timing risk by 44% — equivalent to insurance against informed flow.

## Data

LOBSTER (Limit Order Book System: The Efficient Reconstructor) sample data for AAPL, June 21, 2012. 10 levels of depth, 400,391 LOB events over the full trading day (09:30–16:00 EST).

Download from [LOBSTER Sample Files](https://data.lobsterdata.com/info/DataSamples.php). Place the message and orderbook CSVs in `data/`. The pipeline auto-detects real LOBSTER files; if none are found, it falls back to a synthetic generator.

## Structure

```
main.py                    # Full analysis pipeline
config.py                  # All tunable parameters
src/
  lobster_loader.py        # LOBSTER CSV parsing (price ÷10000, halt filtering)
  data_generator.py        # Synthetic fallback (OU + Poisson + informed episodes)
  lob.py                   # Midprice, spread, depth, Lee-Ready classification
  kyle.py                  # OLS estimation (HC1), intraday windows, rolling λ
  vpin.py                  # Volume buckets, BVC, VPIN, event detection
  almgren_chriss.py        # Optimal trajectories, cost decomposition, adaptive execution
  plots.py                 # All figures
data/                      # LOBSTER CSVs (not tracked)
figures/                   # Output charts (6 PNGs)
output/                    # Output CSVs
```

## References

- Kyle, A.S. (1985). *Continuous Auctions and Insider Trading*. Econometrica, 53(6), 1315-1335.
- Lee, C.M.C. & Ready, M.J. (1991). *Inferring Trade Direction from Intraday Data*. Journal of Finance, 46(2), 733-746.
- Almgren, R. & Chriss, N. (2000). *Optimal Execution of Portfolio Transactions*. Journal of Risk, 3, 5-40.
- Easley, D., López de Prado, M.M. & O'Hara, M. (2012). *Flow Toxicity and Liquidity in a High-Frequency World*. Review of Financial Studies, 25(5), 1457-1493.
