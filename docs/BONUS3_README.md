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
│   ├── loan_pricing_model.py                    # ★ Main pricing model
│   └── fetch_market_data.py                     # Market data fetcher (beta, etc.)
```

## Usage

### Train and Test Model

```bash
cd ~/Documents/GitHub/ML-CESA

# Train and test loan pricing model (includes demo)
python src/financial_planning/loan_pricing/loan_pricing_model.py

# Fetch market data for training (beta, market cap)
python src/financial_planning/loan_pricing/fetch_market_data.py
```

### Python API

```python
import sys
sys.path.insert(0, 'src/financial_planning/loan_pricing')
from loan_pricing_model import LoanPricingModel

# Initialize and train
model = LoanPricingModel()
model.train('data/loan_pricing_training_data.csv')

# Price a rated company loan
result = model.predict(
    rating='BBB',
    debt_to_equity=1.2,
    interest_coverage=6.0,
    current_ratio=1.5,
    net_margin=0.08,
    roa=0.06,
    debt_to_ebitda=2.5,
    total_assets=5e9,
    beta=1.1,
    market_cap=10e9,
    treasury_yield=4.5,
    maturity_years=5
)

print(f"Spread: {result['spread_bps']:.0f} bps")
print(f"Interest Rate: {result['interest_rate']:.2f}%")
print(f"95% CI: [{result['ci_low']:.0f}, {result['ci_high']:.0f}] bps")
```

### Pricing an Unrated/Private Company

```python
# Price without rating (for private companies)
result = model.predict(
    rating=None,  # No rating!
    debt_to_equity=1.2,
    interest_coverage=6.0,
    current_ratio=1.5,
    net_margin=0.08,
    roa=0.06,
    debt_to_ebitda=2.5,
    total_assets=5e9
)
```

### Monte Carlo Resale Forecast

```python
# Forecast resale price after 1 month
forecast = model.forecast_resale_price(
    initial_spread=175,
    maturity_years=5,
    forecast_months=1
)

print(f"Expected Price: {forecast['expected_price']:.2f}")
print(f"90% Range: [{forecast['price_5th']:.2f}, {forecast['price_95th']:.2f}]")
```

## Model Details

### Features (10 Total)

| Feature | Source | Importance |
|---------|--------|------------|
| rating_num | Credit Rating | 22.1% |
| net_margin | Income Statement | 19.8% |
| interest_coverage | Income Statement | 16.5% |
| debt_to_ebitda | Balance Sheet | 8.7% |
| log_assets | Balance Sheet | 8.3% |
| debt_to_equity | Balance Sheet | 7.6% |
| roa | Financial Statements | 5.8% |
| log_market_cap | Market Data | 5.4% |
| beta | Market Data | 3.2% |
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
| Rated (BBB profile) | BBB | 175 bps | 6.25% | [116, 277] |
| Unrated (same financials) | — | 175 bps | 6.25% | [122, 245] |
| High-risk | B | 401 bps | 8.51% | [146, 749] |

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
| Train & test model | `python src/financial_planning/loan_pricing/loan_pricing_model.py` |
| Fetch market data | `python src/financial_planning/loan_pricing/fetch_market_data.py` |

## References

- Merton, R. C. (1974). "On the Pricing of Corporate Debt."
- Duffie, D., & Singleton, K. J. (1999). "Modeling Term Structures of Defaultable Bonds."
- Jarrow, R. A., & Turnbull, S. M. (1995). "Pricing Derivatives on Financial Securities Subject to Credit Risk."
- FRED ICE BofA Corporate Bond Spread Indices.
