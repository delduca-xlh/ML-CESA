# Part 1: Balance Sheet Forecasting with XGBoost

## Overview

This module implements a balance sheet forecasting system that predicts financial statement line items while strictly enforcing accounting identities. The model uses XGBoost for driver variable prediction and an accounting engine for derived variable computation.

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
│     ├── Lag features (1-4 quarters)                             │
│     ├── Rolling statistics (4Q, 8Q windows)                     │
│     └── YoY growth rates                                        │
│                                                                  │
│  3. XGBoost Prediction (Driver Variables)                       │
│     ├── sales_revenue                                           │
│     ├── cost_of_goods_sold                                      │
│     ├── overhead_expenses                                       │
│     ├── payroll_expenses                                        │
│     └── capex                                                   │
│                                                                  │
│  4. Accounting Engine (Derived Variables)                       │
│     ├── Income Statement completion                             │
│     ├── Balance Sheet derivation                                │
│     └── Identity enforcement (A = L + E)                        │
│                                                                  │
│  5. Evaluation & Output                                          │
│     ├── MAPE calculation                                        │
│     └── Complete financial statements                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
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
python auto_forecast_pipeline.py AAPL GS GOOGL PG XOM
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

## Key Components

### 1. Data Fetcher (`fmp_data_fetcher.py`)

Fetches quarterly financial data from Financial Modeling Prep API:
- Income Statement
- Balance Sheet
- Cash Flow Statement
- Key Ratios

### 2. XGBoost Forecaster (`forecaster_integration.py`)

Predicts 5 driver variables:
- `sales_revenue`: Total revenue
- `cost_of_goods_sold`: Direct costs
- `overhead_expenses`: SG&A expenses
- `payroll_expenses`: Labor costs
- `capex`: Capital expenditures

### 3. Accounting Engine (`accounting_engine.py`)

Derives all other financial statement items using accounting relationships:

```
Income Statement:
  gross_profit = revenue - COGS
  operating_income = gross_profit - operating_expenses
  net_income = operating_income - interest - taxes

Balance Sheet:
  retained_earnings += net_income - dividends
  total_equity = common_stock + retained_earnings
  total_assets = total_liabilities + total_equity  ← ENFORCED
```

## Results

### Multi-Company Validation

| Company | Industry | ML MAPE | Accounting MAPE | Identity |
|---------|----------|---------|-----------------|----------|
| PG | Consumer | 7.36% | 8.90% | 100% |
| XOM | Energy | 8.79% | 23.81% | 100% |
| AAPL | Tech | 11.67% | 9.23% | 100% |
| COST | Retail | 11.71% | 11.75% | 100% |
| GS | Banking | 12.78% | 13.37% | 100% |
| NFLX | Streaming | 20.07% | 22.29% | 100% |
| GOOGL | Tech | 26.65% | 23.98% | 100% |

### Key Findings

1. **Mature companies perform best**: PG, XOM, AAPL with stable margins
2. **High-growth challenging**: NFLX, GOOGL with volatile patterns
3. **40Q window optimal**: Balances data availability with relevance
4. **100% identity compliance**: By construction, never violated

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
