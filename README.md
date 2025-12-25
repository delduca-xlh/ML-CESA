# JP Morgan MLCOE - Balance Sheet Forecasting & LLM Financial Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive lending decision support system developed for **JP Morgan Chase - Machine Learning Center of Excellence (MLCOE)** 2026 Summer Associate position in Time Series & Reinforcement Learning.

## 📋 Project Overview

This project addresses the challenge of analyzing financial health of businesses seeking loans, enabling data-driven lending decisions. The solution combines:

| Component | Description | Key Achievement |
|-----------|-------------|-----------------|
| **Part 1** | Balance Sheet Forecasting with XGBoost | 11.67% MAPE, 100% identity compliance |
| **Part 2** | LLM Application (Ensemble + PDF Extraction) | 9/9 companies extracted, adaptive ensemble |
| **Bonus 1** | Credit Rating Model | 72.4% accuracy, Evergrande detected 12mo early |
| **Bonus 2** | Risk Warning Extraction Engine | 4/4 bankruptcy cases detected |
| **Bonus 3** | Loan Pricing Model | R²=0.383, Monte Carlo 95% CI |

## 🏗️ Project Structure

```
ML-CESA/
├── README.md                          # Main documentation
├── requirements.txt                   # Python dependencies
├── auto_forecast_pipeline.py          # Part 1: Main XGBoost forecasting pipeline
├── run_ensemble.py                    # Part 2: Ensemble model (ML + LLM)
├── run_part2.py                       # Part 2: LLM ratio integration
├── run_part2_pure_llm.py              # Part 2: Pure LLM forecasting comparison
├── print_structure.py                 # Utility: Print project structure
│
├── data/                              # Training data and models
│   ├── credit_rating_training_data.csv      # 447 companies with ratings
│   ├── credit_rating_model.pkl              # Trained credit rating model
│   ├── credit_rating_metadata.json          # Model metadata
│   ├── loan_pricing_training_data.csv       # Loan pricing data
│   ├── loan_pricing_training_data_with_market.csv  # With beta & market cap
│   ├── loan_pricing_model.pkl               # Trained pricing model
│   └── annual_reports/                      # Bankruptcy case PDFs
│       ├── evergrande/
│       │   ├── ar2020.pdf                   # Evergrande 2020 Annual Report
│       │   └── car2021.pdf                  # Evergrande 2021 Annual Report
│       ├── lehman/
│       │   └── lehman.pdf                   # Lehman Brothers 2007
│       └── enron/
│           └── EnronAnnualReport2000.pdf    # Enron 2000
│
├── docs/                              # Documentation
│   ├── COMPLETE_SYSTEM_WORKFLOW.md    # Full system workflow
│   ├── COMPLETE_WORKFLOW.md           # Pipeline documentation
│   └── FEATURE_LEVEL_WORKFLOW.md      # Feature engineering details
│
├── outputs/                           # Model outputs and results
│   ├── xgboost_models/                # Part 1: XGBoost results by company
│   │   ├── aapl/                      # Apple results
│   │   ├── gs/                        # Goldman Sachs results
│   │   ├── googl/                     # Google results
│   │   ├── pg/                        # Procter & Gamble results
│   │   ├── xom/                       # Exxon Mobil results
│   │   ├── cost/                      # Costco results
│   │   ├── nflx/                      # Netflix results
│   │   └── .../                       # Other companies
│   ├── ensemble/                      # Part 2: Ensemble results
│   │   ├── aapl/
│   │   ├── gs/
│   │   └── googl/
│   ├── part2_results/                 # Part 2: LLM integration results
│   │   ├── aapl/
│   │   ├── googl/
│   │   └── .../
│   └── part2_pure_llm/                # Pure LLM comparison results
│
└── src/
    └── financial_planning/
        ├── __init__.py
        │
        ├── models/                    # Part 1: Core forecasting models
        │   ├── __init__.py
        │   ├── accounting_engine.py         # Accounting identity enforcement
        │   ├── balance_sheet_forecaster.py  # Main forecaster class
        │   ├── forecaster_integration.py    # Model integration
        │   ├── financial_model.py           # Financial model base
        │   ├── debt_schedule.py             # Debt modeling
        │   ├── intermediate_tables.py       # Working capital tables
        │   └── tax_shields.py               # Tax calculations
        │
        ├── financial_statements/      # Financial statement components
        │   ├── __init__.py
        │   ├── balance_sheet.py             # Balance sheet structure
        │   ├── income_statement.py          # Income statement structure
        │   ├── cash_budget.py               # Cash flow budgeting
        │   └── statement_builder.py         # Statement construction
        │
        ├── utils/                     # Utilities and data fetching
        │   ├── __init__.py
        │   ├── fmp_data_fetcher.py          # Financial Modeling Prep API
        │   ├── yahoo_finance_fetcher.py     # Yahoo Finance backup
        │   ├── llm_assumption_generator.py  # Part 2: LLM ratio generation
        │   ├── pdf_extractor.py             # Part 2: PDF extraction tool
        │   └── fisher_equation.py           # Interest rate calculations
        │
        ├── core/                      # Core financial calculations
        │   ├── __init__.py
        │   ├── cash_flow.py                 # Cash flow analysis
        │   ├── circularity_solver.py        # Circular reference solver
        │   ├── cost_of_capital.py           # WACC calculations
        │   └── valuation.py                 # DCF valuation
        │
        ├── credit_rating/             # Bonus 1: Credit rating system
        │   ├── __init__.py
        │   ├── credit_rating_system.py      # Main rating system
        │   ├── ordinal_lr.py                # Ordinal logistic regression
        │   ├── fraud_detector.py            # Z-Score & M-Score
        │   ├── trainer.py                   # Model training
        │   ├── training_data.py             # Data preparation
        │   ├── fetch_training_data.py       # Data fetching
        │   ├── rating_pipeline.py           # Rating pipeline
        │   ├── train_and_save_model.py      # Model persistence
        │   ├── risk_extractor.py            # Bonus 2: Risk extraction
        │   ├── test_evergrande.py           # Evergrande test case
        │   ├── test_bankruptcy_cases.py     # Bankruptcy validation
        │   └── test_risk_extractor.py       # Risk extractor tests
        │
        └── loan_pricing/              # Bonus 3: Loan pricing
            ├── loan_pricing_model.py        # Loan spread prediction
            └── fetch_market_data.py         # Market data (beta, etc.)
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt
```

### Required Packages

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
anthropic>=0.20.0
pdfplumber>=0.10.0
pdf2image>=1.16.0
Pillow>=10.0.0
requests>=2.31.0
```

### API Keys

```bash
# Set environment variables
export FMP_API_KEY="your_fmp_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
```

## 📊 Part 1: Balance Sheet Forecasting

### Usage

```bash
# Run forecasting for a single company
python auto_forecast_pipeline.py AAPL

# Run for multiple companies
python auto_forecast_pipeline.py AAPL GS GOOGL PG XOM
```

### Results

| Company | Industry | ML MAPE | Accounting MAPE | Grade |
|---------|----------|---------|-----------------|-------|
| PG | Consumer | 7.36% | 8.90% | ⭐⭐⭐⭐⭐ |
| XOM | Energy | 8.79% | 23.81% | ⭐⭐⭐⭐⭐ |
| AAPL | Tech | 11.67% | 9.23% | ⭐⭐⭐⭐⭐ |
| COST | Retail | 11.71% | 11.75% | ⭐⭐⭐⭐⭐ |
| GS | Banking | 12.78% | 13.37% | ⭐⭐⭐⭐⭐ |
| NFLX | Streaming | 20.07% | 22.29% | ⭐⭐⭐ |
| GOOGL | Tech | 26.65% | 23.98% | ⭐⭐⭐ |

### Key Features

- **Accounting Identity Enforcement**: 100% compliance with Assets = Liabilities + Equity
- **Two-Stage Approach**: ML predicts drivers, accounting engine derives statements
- **40-Quarter Window**: Optimal historical data window for most companies
- **Execution Time**: <7 seconds per company

## 🤖 Part 2: LLM Application

### 2a-d: Ensemble Forecasting

```bash
# Run ensemble model
python run_ensemble.py AAPL
python run_ensemble.py GS
```

#### Results

| Company | Best Approach | Driver MAPE | Accounting MAPE | Overall |
|---------|---------------|-------------|-----------------|---------|
| AAPL | ML + LLM Ratios | 11.67% | 9.60% | 10.63% |
| GS | ML + Historical | 12.78% | 13.57% | 13.17% |

### 2e-i: PDF Financial Extraction

```bash
# Extract from predefined company
python -m src.financial_planning.utils.pdf_extractor GM
python -m src.financial_planning.utils.pdf_extractor MICROSOFT
python -m src.financial_planning.utils.pdf_extractor LVMH

# Extract from local PDF
python -m src.financial_planning.utils.pdf_extractor /path/to/annual_report.pdf
```

#### Supported Companies

| Company | Standard | Currency | Status |
|---------|----------|----------|--------|
| General Motors | US GAAP | USD | ✅ |
| Microsoft | US GAAP | USD | ✅ |
| Google | US GAAP | USD | ✅ |
| JPMorgan | US GAAP | USD | ✅ |
| Exxon Mobil | US GAAP | USD | ✅ |
| Alibaba | US GAAP | RMB | ✅ |
| LVMH | IFRS | EUR | ✅ |
| Tencent | IFRS | RMB | ✅ |
| Volkswagen | IFRS | EUR | ✅ |

## 🏆 Bonus 1: Credit Rating Model

### Usage

```bash
# Train model
python -m src.financial_planning.credit_rating.trainer

# Rate a company
python -m src.financial_planning.credit_rating.rating_pipeline AAPL

# Test on Evergrande
python -m src.financial_planning.credit_rating.test_evergrande
```

### Performance

- **Accuracy**: 72.4% exact match
- **Within 1 Notch**: 94.2%
- **Within 2 Notches**: 98.7%

### Validation Case Studies

| Company | Year | Model Rating | Z-Score | Outcome |
|---------|------|--------------|---------|---------|
| Evergrande | 2020 | D (100%) | 0.53 | Default Dec 2021 |
| Lehman Brothers | 2007 | D (100%) | 0.04 | Bankrupt Sep 2008 |
| Enron | 2000 | D (100%) | 1.79 | Bankrupt Dec 2001 |

## ⚠️ Bonus 2: Risk Warning Extraction

### Usage

```bash
# Test on bankruptcy cases
python -m src.financial_planning.credit_rating.test_bankruptcy_cases

# Test risk extractor
python -m src.financial_planning.credit_rating.test_risk_extractor
```

### Validation Results

| Company | Year | Risk Level | Warnings | Detection |
|---------|------|------------|----------|-----------|
| Evergrande | 2020 | CRITICAL | 15 | 12 months before default |
| Evergrande | 2021 | CRITICAL | 18 | During crisis |
| Lehman Brothers | 2007 | CRITICAL | 27 | 9 months before bankruptcy |
| Enron | 2000 | HIGH | 8 | 12 months before bankruptcy |

## 💰 Bonus 3: Loan Pricing Model

### Usage

```bash
# Train model
python -m src.financial_planning.loan_pricing.loan_pricing_model

# Fetch market data
python -m src.financial_planning.loan_pricing.fetch_market_data
```

### Performance

| Metric | With Rating | Without Rating |
|--------|-------------|----------------|
| R² | 0.383 | 0.212 |
| RMSE | 275 bps | 311 bps |
| MAE | 80 bps | 101 bps |

### Features

- **10 Input Features**: Rating, D/E, Interest Coverage, Current Ratio, Net Margin, ROA, Debt/EBITDA, Log Assets, Beta, Market Cap
- **Unrated Companies**: Separate model for private/unrated companies
- **Monte Carlo Simulation**: 95% confidence interval for resale price forecast

## 📁 Output Files Structure

### Part 1: XGBoost Models (`outputs/xgboost_models/{ticker}/`)

| File | Description |
|------|-------------|
| `01_data_window_analysis.json` | Data availability analysis |
| `02_data_split.json` | Train/test split info |
| `03_development_data.csv` | Training data |
| `03_historical_ratios.json` | Computed accounting ratios |
| `04_test_actuals.csv` | Test set actual values |
| `05_test_ml_predictions.csv` | XGBoost predictions |
| `06_test_complete_statements.csv` | Full financial statements |
| `07_test_evaluation.json` | MAPE metrics |
| `pipeline_summary.txt` | Run summary |

### Part 2: Ensemble (`outputs/ensemble/{ticker}/`)

| File | Description |
|------|-------------|
| `ensemble_results.json` | All approaches comparison |
| `ensemble_predictions.csv` | Best approach predictions |
| `ensemble_summary.csv` | Summary metrics |

## 📚 References

1. Vélez-Pareja, I. (2007). "Forecasting Financial Statements with No Plugs and No Circularity."
2. Vélez-Pareja, I. (2009). "Constructing Consistent Financial Planning Models for Valuation."
3. Alonso, M., & Dupouy, H. (2024). "Large Language Models as Financial Analysts."
4. Farr, M., et al. (2025). "AI Determinants of Success and Failure: Financial Statements."
5. Zhang, H., et al. (2025). "Financial Statement Checking Recognition System Based on LLMs."
6. Schilit, H. M. (2010). "Financial Shenanigans: How to Detect Accounting Gimmicks & Fraud."

## 👤 Author

**Lihao Xiao**
- Email: lihao@ucsb.edu
- Institution: University of California, Santa Barbara
- Position: 2026 Summer Associate Candidate - JP Morgan MLCOE

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Developed for JP Morgan Chase - Machine Learning Center of Excellence*
*2026 Summer Associate – Time Series & Reinforcement Learning*