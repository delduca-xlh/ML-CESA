# Bonus 3: Loan Pricing Model

## Overview

This module predicts credit spreads for term loans using a hybrid approach combining reduced-form credit theory with gradient boosting. It supports both rated and unrated/private companies and provides Monte Carlo simulation for resale price forecasting.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LOAN PRICING MODEL                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LOAN PRICING FORMULA:                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Interest Rate = Treasury Yield + Credit Spread           │  │
│  │                                                           │  │
│  │  Example: 4.5% Treasury + 1.75% Spread = 6.25% Rate      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  MODEL COMPONENTS:                                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. Base Spread (FRED ICE BofA indices by rating)        │  │
│  │  2. Company Adjustments (leverage, coverage, size)        │  │
│  │  3. Gradient Boosting (learn full relationship)           │  │
│  │  4. Quantile Regression (confidence intervals)            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  OUTPUTS:                                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  • Point estimate: Spread in basis points                 │  │
│  │  • 95% Confidence Interval                                │  │
│  │  • Monte Carlo resale price forecast                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
ML-CESA/
├── data/
│   ├── loan_pricing_training_data.csv           # Base training data (447 companies)
│   ├── loan_pricing_training_data_with_market.csv  # With beta & market cap
│   └── loan_pricing_model.pkl                   # Trained model
├── src/financial_planning/loan_pricing/
│   ├── price_loan.py                            # ★ Price any ticker (standalone)
│   ├── loan_pricing_model.py                    # Full model with training
│   └── fetch_market_data.py                     # Market data fetcher (beta, etc.)
```

## Usage

### Price Any Company by Ticker (Recommended)

```bash
cd ~/Documents/GitHub/ML-CESA

# Price Apple loan (5-year, 4.5% treasury)
python src/financial_planning/loan_pricing/price_loan.py AAPL

# Price Tesla loan
python src/financial_planning/loan_pricing/price_loan.py TSLA

# Custom maturity (7 years)
python src/financial_planning/loan_pricing/price_loan.py GM --maturity 7

# Custom treasury yield (5.0%)
python src/financial_planning/loan_pricing/price_loan.py MSFT --treasury 5.0

# Price as unrated/private company
python src/financial_planning/loan_pricing/price_loan.py NFLX --no-rating

# Quiet mode
python src/financial_planning/loan_pricing/price_loan.py F --quiet
```

### Output Example

```
======================================================================
LOAN PRICING REPORT: AAPL
======================================================================

Company: Apple Inc.
Data Date: 2024-09-28

┌──────────────────────────────────────────────────────┐
│  LOAN TERMS                                          │
├──────────────────────────────────────────────────────┤
│  Maturity:            5 years                        │
│  Treasury Yield:      4.50%                          │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  PRICING RESULTS                                     │
├──────────────────────────────────────────────────────┤
│  Credit Spread:         62 bps                       │
│  Interest Rate:       5.12%                          │
│  Implied Rating:         AA                          │
│  95% CI:           [45, 85] bps                      │
└──────────────────────────────────────────────────────┘

Rate Breakdown:
  Treasury Yield:      4.50%
  + Credit Spread:     0.62%
  ─────────────────────────
  = Total Rate:        5.12%

──────────────────────────────────────────────────────────────────────
RESALE PRICE FORECAST (1 Month)
──────────────────────────────────────────────────────────────────────
  Expected Price:   100.00
  90% Range:       [99.12, 100.88]

──────────────────────────────────────────────────────────────────────
KEY FINANCIAL RATIOS
──────────────────────────────────────────────────────────────────────
  Debt/Equity:                1.87
  Interest Coverage:         29.1x
  Current Ratio:              0.99
  Net Margin:                25.0%
  ROA:                       27.0%
  Debt/EBITDA:                0.97x
======================================================================
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `ticker` | Stock ticker symbol (required) | — |
| `--maturity N` | Loan maturity in years | 5 |
| `--treasury N` | Treasury yield in % | 4.5 |
| `--no-rating` | Price as unrated company | False |
| `--quiet` | Less verbose output | False |

### Python API

```python
import sys
sys.path.insert(0, 'src/financial_planning/loan_pricing')
from price_loan import price_loan, print_report

# Price Apple loan
result = price_loan('AAPL', maturity_years=5, treasury_yield=4.5)
print_report(result)

# Access results programmatically
print(f"Spread: {result.spread_bps:.0f} bps")
print(f"Interest Rate: {result.interest_rate:.2f}%")
print(f"95% CI: [{result.ci_lower:.0f}, {result.ci_upper:.0f}] bps")
print(f"Implied Rating: {result.implied_rating}")

# Monte Carlo forecast
print(f"Expected Price (1 month): {result.forecast_price:.2f}")
print(f"90% Range: [{result.forecast_price_5th:.2f}, {result.forecast_price_95th:.2f}]")
```

### Train Full Model (Optional)

```bash
cd ~/Documents/GitHub/ML-CESA

# Train and test loan pricing model (includes demo)
python src/financial_planning/loan_pricing/loan_pricing_model.py

# Fetch market data for training (beta, market cap)
python src/financial_planning/loan_pricing/fetch_market_data.py
```

## Model Details

### Features (8 Total)

| Feature | Source | Importance |
|---------|--------|------------|
| rating_num | Credit Rating | 22.1% |
| net_margin | Income Statement | 19.8% |
| interest_coverage | Income Statement | 16.5% |
| debt_to_ebitda | Balance Sheet | 8.7% |
| log_assets | Balance Sheet | 8.3% |
| debt_to_equity | Balance Sheet | 7.6% |
| roa | Financial Statements | 5.8% |
| current_ratio | Balance Sheet | 2.6% |

### Base Spreads (FRED ICE BofA)

| Rating | Base Spread (bps) | 1-Year PD |
|--------|-------------------|-----------|
| AAA | 50 | 0.01% |
| AA | 65 | 0.02% |
| A | 90 | 0.05% |
| BBB | 140 | 0.18% |
| BB | 250 | 0.80% |
| B | 400 | 3.50% |
| CCC | 900 | 12.0% |
| D | 2500 | 100% |

## Performance

### Model Accuracy

| Metric | With Rating | Without Rating |
|--------|-------------|----------------|
| R² | 0.383 | 0.212 |
| RMSE | 275 bps | 311 bps |
| MAE | 80 bps | 101 bps |

### Example Predictions

| Company Type | Rating | Spread | Rate | 95% CI |
|--------------|--------|--------|------|--------|
| Investment Grade (AAPL) | AA | 62 bps | 5.12% | [45, 85] |
| Investment Grade (MSFT) | AAA | 48 bps | 4.98% | [35, 65] |
| High Yield (F) | BB | 265 bps | 7.15% | [180, 380] |
| High Risk (unrated) | — | 350 bps | 8.00% | [220, 520] |

## Monte Carlo Simulation

### Spread Evolution Model

$$\text{Spread}_{t+\Delta t} = \text{Spread}_t \times \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t} Z\right]$$

Where:
- μ = 0 (mean-reverting spreads)
- σ = 25% (annualized volatility)
- Z ~ N(0,1)

### Price Sensitivity

$$\Delta\text{Price} \approx -\text{Duration} \times \Delta\text{Spread}$$

## Training Data

| File | Description | Records |
|------|-------------|---------|
| `loan_pricing_training_data.csv` | Base training data | 447 |
| `loan_pricing_training_data_with_market.csv` | With beta & market cap | 447 |

## Command Summary

| Task | Command |
|------|---------|
| **Price any ticker** | `python src/financial_planning/loan_pricing/price_loan.py TICKER` |
| Price with custom maturity | `python src/financial_planning/loan_pricing/price_loan.py TICKER --maturity 7` |
| Price as unrated | `python src/financial_planning/loan_pricing/price_loan.py TICKER --no-rating` |
| Train full model | `python src/financial_planning/loan_pricing/loan_pricing_model.py` |
| Fetch market data | `python src/financial_planning/loan_pricing/fetch_market_data.py` |

## Comparison with Bonus 1

| Feature | rate_ticker.py (Bonus 1) | price_loan.py (Bonus 3) |
|---------|--------------------------|-------------------------|
| Input | Stock ticker | Stock ticker |
| Output | Credit rating (AAA-D) | Credit spread (bps) |
| Confidence | Probability distribution | 95% CI |
| Extra Analysis | Fraud detection | Monte Carlo resale forecast |
| Use Case | Credit assessment | Loan pricing |

## References

- Merton, R. C. (1974). "On the Pricing of Corporate Debt."
- Duffie, D., & Singleton, K. J. (1999). "Modeling Term Structures of Defaultable Bonds."
- Jarrow, R. A., & Turnbull, S. M. (1995). "Pricing Derivatives on Financial Securities Subject to Credit Risk."
- FRED ICE BofA Corporate Bond Spread Indices.
