# Part 1: Balance Sheet Forecasting with XGBoost

## Overview

This module implements a balance sheet forecasting system that predicts financial statement line items while strictly enforcing accounting identities. The model uses XGBoost quantile regression for driver variable prediction and a deterministic accounting engine for derived variable computation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTO FORECAST PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Data Fetching (FMP API)                                     │
│     └── Historical quarterly financials (40Q default)           │
│                                                                  │
│  2. Feature Engineering                                          │
│     ├── Driver variables: d_t = [g_rev, ρ_cogs, ρ_opex,        │
│     │                            ρ_capex, ρ_ni]                 │
│     ├── Lag features: x_t ∈ ℝ²⁰ (4 quarters × 5 drivers)       │
│     └── Rolling statistics (4Q, 8Q windows)                     │
│                                                                  │
│  3. XGBoost Quantile Regression                                 │
│     ├── 7 quantiles: τ ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 0.90,  │
│     │                      0.95}                                 │
│     ├── Pinball loss: L_τ(y, ŷ) = τ(y-ŷ)⁺ + (1-τ)(ŷ-y)⁺       │
│     └── Gradient boosting: f_τ(x) = Σ 0.1 · h_m(x)              │
│                                                                  │
│  4. Accounting Engine (Derived Variables)                       │
│     ├── Income Statement: GP = Rev - COGS                       │
│     ├── Balance Sheet: L = A - E (by construction)              │
│     └── Cash Flow linkage: ΔCash = NI - CapEx + ΔDebt          │
│                                                                  │
│  5. Probabilistic Output                                         │
│     ├── Point estimate: median (Q_0.50)                         │
│     ├── Uncertainty: σ̂ = (Q_0.95 - Q_0.05) / 3.29              │
│     └── 90% CI: [Q_0.05, Q_0.95]                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Mathematical Formulation

### Driver Variables

We predict 5 driver variables that determine all financial statement line items:

```
d_t = [g^rev_t, ρ^cogs_t, ρ^opex_t, ρ^capex_t, ρ^ni_t]

where:
  g^rev_t    = (Revenue_t - Revenue_{t-1}) / Revenue_{t-1}    (revenue growth)
  ρ^cogs_t   = COGS_t / Revenue_t                             (COGS margin)
  ρ^opex_t   = OpEx_t / Revenue_t                             (OpEx margin)
  ρ^capex_t  = CapEx_t / Revenue_t                            (CapEx intensity)
  ρ^ni_t     = NetIncome_t / Revenue_t                        (net margin)
```

### Quantile Regression

For each driver and quantile τ ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}:

```
minimize Σᵢ L_τ(yᵢ, f_τ(xᵢ))

where L_τ(y, ŷ) = {  τ · (y - ŷ)      if y ≥ ŷ
                  { (1-τ) · (ŷ - y)   if y < ŷ
```

### Accounting Engine

The accounting engine derives all financial statement items from the 5 drivers:

**Income Statement**:
```
Revenue        = Revenue_{t-1} × (1 + g^rev_t)
COGS           = Revenue × ρ^cogs_t
Gross Profit   = Revenue - COGS
OpEx           = Revenue × ρ^opex_t
EBITDA         = Gross Profit - OpEx
Net Income     = Revenue × ρ^ni_t
```

**Balance Sheet** (identity enforced):
```
Total Assets       = f(Revenue, historical ratios)
Total Equity       = Equity_{t-1} + Net Income - Dividends
Total Liabilities  = Total Assets - Total Equity   ← ENFORCED
```

## File Structure

```
ML-CESA/
├── auto_forecast_pipeline.py              # Main entry point
├── src/financial_planning/
│   ├── models/
│   │   ├── accounting_engine.py           # Accounting identity enforcement
│   │   ├── balance_sheet_forecaster.py    # Main forecaster class
│   │   ├── forecaster_integration.py      # XGBoost integration
│   │   ├── financial_model.py             # Base financial model
│   │   ├── debt_schedule.py               # Debt modeling
│   │   ├── intermediate_tables.py         # Working capital tables
│   │   └── tax_shields.py                 # Tax calculations
│   ├── financial_statements/
│   │   ├── balance_sheet.py               # Balance sheet structure
│   │   ├── income_statement.py            # Income statement structure
│   │   ├── cash_budget.py                 # Cash flow budgeting
│   │   └── statement_builder.py           # Statement construction
│   └── utils/
│       ├── fmp_data_fetcher.py            # Financial Modeling Prep API
│       └── yahoo_finance_fetcher.py       # Yahoo Finance backup
└── outputs/xgboost_models/{ticker}/       # Output directory
```

## Usage

### Basic Usage

```bash
# Forecast for a single company
python auto_forecast_pipeline.py AAPL

# Forecast for multiple companies
python auto_forecast_pipeline.py AAPL GS GOOGL PG XOM COST NFLX
```

### Python API

```python
from src.financial_planning.models.balance_sheet_forecaster import BalanceSheetForecaster
from src.financial_planning.utils.fmp_data_fetcher import FMPDataFetcher

# Fetch data
fetcher = FMPDataFetcher(api_key="your_key")
data = fetcher.get_quarterly_data("AAPL", quarters=40)

# Run forecast
forecaster = BalanceSheetForecaster()
results = forecaster.forecast(data, test_quarters=4)
```

## Results

### Multi-Company Validation (7 Companies)

| Company | Industry | ML Driver MAPE | Accounting MAPE | Identity |
|---------|----------|----------------|-----------------|----------|
| COST | Retail | 5.52% | 6.18% | 100% ✓ |
| PG | Consumer | 7.36% | 8.90% | 100% ✓ |
| XOM | Energy | 8.79% | 12.45% | 100% ✓ |
| AAPL | Tech | 9.93% | 9.23% | 100% ✓ |
| NFLX | Streaming | 11.45% | 14.32% | 100% ✓ |
| GOOGL | Tech | 12.13% | 15.67% | 100% ✓ |
| GS | Banking | 17.23% | 18.91% | 100% ✓ |

### Key Findings

1. **Stable companies perform best**: Costco (5.52%), P&G (7.36%) with consistent margins
2. **Complex structures challenging**: Goldman Sachs (17.23%) with financial services complexity
3. **40Q window optimal**: Balances data availability with pattern relevance
4. **100% identity compliance**: By construction, A = L + E is never violated

## Output Files

```
outputs/xgboost_models/{ticker}/
├── 01_data_window_analysis.json        # Data availability analysis
├── 02_data_split.json                  # Train/test split info
├── 03_development_data.csv             # Training data
├── 03_historical_ratios.json           # Computed accounting ratios
├── 03_hyperparameter_tuning.json       # XGBoost tuning results
├── 04_final_model_config.json          # Final model configuration
├── 04_final_model_models.pkl           # Saved XGBoost models
├── 04_test_actuals.csv                 # Test set actual values
├── 05_test_ml_predictions.csv          # XGBoost predictions
├── 06_test_complete_statements.csv     # Full financial statements
├── 07_test_evaluation.json             # MAPE metrics
├── 08_future_ml_predictions.csv        # Future period predictions
├── 09_future_complete_statements.csv   # Future complete statements
└── pipeline_summary.txt                # Run summary
```

## References

- Vélez-Pareja, I. (2007). "Forecasting Financial Statements with No Plugs and No Circularity."
- Vélez-Pareja, I. (2009). "Constructing Consistent Financial Planning Models for Valuation."
- Mejía-Peláez, F., & Vélez-Pareja, I. (2011). "Analytical Solution to the Circularity Problem."
