# JP Morgan MLCOE - Balance Sheet Forecasting & LLM Financial Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive lending decision support system developed for **JP Morgan Chase - Machine Learning Center of Excellence (MLCOE)** 2026 Summer Associate position in Time Series & Reinforcement Learning.

---

## 📋 Executive Summary

Financial institutions require accurate balance sheet forecasts for credit decisions. This project presents a **hybrid ML + LLM system** that:

- **Predicts 60+ financial statement line items** from 5 driver variables
- **Guarantees accounting identity compliance** ($A = L + E$) by construction
- **Achieves 6.24% MAPE** on Apple (best-in-class result)
- **Detected Evergrande distress 12 months before default**

| Component | Description | Key Achievement |
|-----------|-------------|-----------------|
| **Part 1** | Balance Sheet Forecasting (XGBoost) | 9.93% MAPE, 100% identity compliance |
| **Part 2a-d** | LLM Ensemble Integration | 6.24% MAPE (ML+LLM), +54% NI improvement |
| **Part 2e-i** | PDF Financial Extraction | 98.5% token savings, 9/9 companies |
| **Bonus 1** | Credit Rating Model | 72.4% accuracy, 94.2% within 1 notch |
| **Bonus 2** | Risk Warning Extraction | 4/4 bankruptcy cases detected |
| **Bonus 3** | Loan Pricing Model | R²=0.383, Monte Carlo 95% CI |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LENDING DECISION SUPPORT SYSTEM                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│   │  Annual      │    │  Structured  │    │  Historical  │                  │
│   │  Report PDF  │    │  Data (API)  │    │  Financials  │                  │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
│          │                   │                   │                           │
│          ▼                   ▼                   ▼                           │
│   ┌──────────────────────────────────────────────────────────┐              │
│   │              LAYER 1: DATA INGESTION                      │              │
│   │  • PDF Extraction (98.5% token savings)                  │              │
│   │  • FMP API Integration                                   │              │
│   │  • Feature Engineering (lag, rolling, YoY)               │              │
│   └──────────────────────────┬───────────────────────────────┘              │
│                              │                                               │
│                              ▼                                               │
│   ┌──────────────────────────────────────────────────────────┐              │
│   │              LAYER 2: FORECASTING ENGINE                  │              │
│   │                                                           │              │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │              │
│   │   │  ML Only    │  │  ML + LLM   │  │  Pure LLM   │      │              │
│   │   │  (XGBoost)  │  │  (Ensemble) │  │  (Claude)   │      │              │
│   │   │  9.93%      │  │  6.24% ★    │  │  7.89%      │      │              │
│   │   └─────────────┘  └─────────────┘  └─────────────┘      │              │
│   │                                                           │              │
│   │   → Accounting Engine: Guarantees A = L + E              │              │
│   └──────────────────────────┬───────────────────────────────┘              │
│                              │                                               │
│                              ▼                                               │
│   ┌──────────────────────────────────────────────────────────┐              │
│   │              LAYER 3: CREDIT ASSESSMENT                   │              │
│   │                                                           │              │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │              │
│   │   │ Credit      │  │ Risk        │  │ Loan        │      │              │
│   │   │ Rating      │  │ Warning     │  │ Pricing     │      │              │
│   │   │ 72.4%       │  │ 4/4 cases   │  │ R²=0.383    │      │              │
│   │   └─────────────┘  └─────────────┘  └─────────────┘      │              │
│   └──────────────────────────────────────────────────────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ML-CESA/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
│
├── run_simulator.py                       # Part 1: XGBoost forecasting pipeline
├── run_ensemble.py                        # Part 2: Ensemble model (ML + LLM)
├── auto_forecast_pipeline.py              # Alternative: Full auto pipeline
├── print_structure.py                     # Utility: Print project structure
│
├── docs/                                  # Documentation
│   ├── PART1_README.md                    # Part 1 detailed documentation
│   ├── PART2_README.md                    # Part 2 detailed documentation
│   ├── BONUS1_README.md                   # Bonus 1: Credit rating
│   ├── BONUS2_README.md                   # Bonus 2: Risk extraction
│   ├── BONUS3_README.md                   # Bonus 3: Loan pricing
│   ├── main_executive_v2.pdf              # Executive Report (25 pages)
│   └── main_revised.pdf                   # Full Academic Report (73 pages)
│
├── data/                                  # Training data and models
│   ├── credit_rating_training_data.csv   # 447 companies with ratings
│   ├── credit_rating_model.pkl           # Trained credit rating model
│   ├── loan_pricing_model.pkl            # Trained pricing model
│   └── annual_reports/                   # Bankruptcy case PDFs
│       ├── evergrande/                   # Evergrande 2020-2021
│       ├── lehman/                       # Lehman Brothers 2007
│       └── enron/                        # Enron 2000
│
├── outputs/                              # Model outputs (by ticker)
│   └── {ticker}/                         # Per-company outputs
│       ├── xgboost_models/               # Part 1: XGBoost models
│       ├── ensemble/                     # Part 2: Ensemble results
│       ├── part2_results/                # LLM integration results
│       └── pdf_reports/                  # Generated PDF reports
│
└── src/financial_planning/
    ├── balance_sheet_simulator/          # Core simulator module
    │   ├── __init__.py
    │   ├── accounting_engine.py          # Accounting identity enforcement
    │   ├── quantile_simulator.py         # XGBoost quantile regression
    │   ├── rolling_validator.py          # Rolling window validation
    │   ├── ensemble_validator.py         # ML + LLM ensemble
    │   ├── llm_ensemble.py               # LLM integration layer
    │   ├── multi_year_simulator.py       # Multi-year forecasting
    │   ├── pdf_report.py                 # PDF report generation
    │   ├── statement_printer.py          # Statement formatting
    │   └── data_structures.py            # Data classes
    │
    ├── models/                           # Alternative implementations
    │   ├── financial_model.py
    │   ├── balance_sheet_forecaster.py
    │   └── forecaster_integration.py
    │
    ├── utils/                            # Utilities
    │   ├── fmp_data_fetcher.py           # Financial Modeling Prep API
    │   ├── llm_assumption_generator.py   # LLM ratio generation
    │   └── pdf_extractor.py              # PDF extraction tool
    │
    ├── credit_rating/                    # Bonus 1 & 2: Credit rating
    │   ├── __init__.py
    │   ├── credit_rating_system.py       # Main rating system
    │   ├── ordinal_lr.py                 # Ordinal logistic regression
    │   ├── fraud_detector.py             # Z-Score & M-Score
    │   ├── risk_extractor.py             # Risk warning extraction
    │   ├── trainer.py                    # Model training
    │   ├── training_data.py              # 450+ company ratings
    │   ├── rating_pipeline.py            # Rating pipeline (module)
    │   ├── rate_ticker.py                # Rate any ticker (standalone)
    │   ├── fetch_training_data.py        # Data fetching from FMP
    │   ├── train_and_save_model.py       # Model persistence
    │   ├── test_evergrande.py            # Evergrande test case
    │   ├── test_bankruptcy_cases.py      # Bankruptcy validation
    │   └── test_risk_extractor.py        # Risk extractor tests
    │
    └── loan_pricing/                     # Bonus 3: Loan pricing
        ├── loan_pricing_model.py         # Main pricing model
        └── fetch_market_data.py          # Market data fetcher
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/ML-CESA.git
cd ML-CESA

# Install dependencies
pip install -r requirements.txt
```

### API Keys

```bash
export FMP_API_KEY="your_financial_modeling_prep_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### Run Forecasting

```bash
cd ~/Documents/GitHub/ML-CESA

# Part 1: XGBoost balance sheet forecasting
python run_simulator.py AAPL

# Part 2: Ensemble with LLM integration
python run_ensemble.py AAPL
```

### Run Bonus Sections

```bash
cd ~/Documents/GitHub/ML-CESA

# Bonus 1: Credit Rating - Rate any ticker
python src/financial_planning/credit_rating/rate_ticker.py AAPL
python src/financial_planning/credit_rating/rate_ticker.py TSLA

# Bonus 1: Credit Rating - Test bankruptcy cases
python src/financial_planning/credit_rating/test_evergrande.py
python src/financial_planning/credit_rating/test_bankruptcy_cases.py

# Bonus 2: Risk Warning Extraction
python src/financial_planning/credit_rating/test_risk_extractor.py

# Bonus 3: Loan Pricing Model
python src/financial_planning/loan_pricing/loan_pricing_model.py

# PDF Extraction Tool
python src/financial_planning/utils/pdf_extractor.py GM
python src/financial_planning/utils/pdf_extractor.py MICROSOFT
python src/financial_planning/utils/pdf_extractor.py LVMH
```

---

## 📊 Part 1: Balance Sheet Forecasting

### Methodology

**Driver Variables** (predicted by XGBoost):
```
d_t = [revenue_growth, COGS_margin, OpEx_margin, CapEx_ratio, net_margin]
```

**Lag Features** (20-dimensional input):
```
x_t = [d_{t-1}, d_{t-2}, d_{t-3}, d_{t-4}] ∈ ℝ²⁰
```

**Quantile Regression** (7 quantiles):
```
τ ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}
```

### Results (7 Companies)

| Company | Industry | ML MAPE | Best Approach | Best MAPE |
|---------|----------|---------|---------------|-----------|
| AAPL | Technology | 9.93% | ML + LLM | **6.24%** |
| COST | Retail | 5.52% | ML Only | **5.52%** |
| PG | Consumer | 7.36% | ML + LLM | **7.21%** |
| GOOGL | Technology | 12.13% | ML + LLM | **9.87%** |
| NFLX | Streaming | 11.45% | Pure LLM | **8.92%** |
| GS | Banking | 17.23% | Pure LLM | **3.52%** |
| XOM | Energy | 8.79% | ML Only | **8.79%** |

**Key Achievement**: 100% accounting identity compliance across all forecasts.

---

## 🤖 Part 2: LLM Integration

### 2a-d: Ensemble Forecasting

Three approaches compared:

| Approach | Revenue Source | Margin Source | Best For |
|----------|----------------|---------------|----------|
| ML Only | XGBoost | Historical ratios | Stable companies |
| ML + LLM | XGBoost | LLM-adjusted ratios | Evolving margins |
| Pure LLM | Claude Sonnet | Claude Sonnet | Complex structures |

**Apple Rolling Validation (ML + LLM)**:

| Metric | R1 | R2 | R3 | R4 | R5 | Mean |
|--------|-----|-----|-----|-----|-----|------|
| Revenue | 4.8% | 2.4% | 8.2% | 6.6% | 0.0% | **4.4%** |
| Net Income | 22.1% | 15.5% | 5.8% | 10.1% | 2.4% | **11.2%** |
| Total Assets | 7.4% | 7.9% | 1.1% | 5.7% | 1.2% | **4.7%** |

### 2e-i: PDF Extraction

**Token Efficiency**: 98.5% reduction (603K → 9K tokens for 402-page report)

**Supported Formats**:

| Company | Standard | Pages | Tokens | Status |
|---------|----------|-------|--------|--------|
| General Motors | US GAAP | 156 | ~7,500 | ✅ |
| Shell | IFRS | 402 | ~9,000 | ✅ |
| LVMH | IFRS (EUR) | 316 | ~8,500 | ✅ |
| Microsoft | US GAAP | 88 | ~6,000 | ✅ |
| JP Morgan | US GAAP (Bank) | 312 | ~9,500 | ✅ |
| Tencent | IFRS (CNY) | 284 | ~8,000 | ✅ |

---

## 🏆 Bonus 1: Credit Rating Model

### Mathematical Formulation

**Ordinal Logistic Regression**:
```
P(Y ≤ j | X) = σ(θⱼ - β'X)
```

**Altman Z-Score**:
```
Z' = 0.717X₁ + 0.847X₂ + 3.107X₃ + 0.420X₄ + 0.998X₅
```

### Performance

| Metric | Value |
|--------|-------|
| Exact Accuracy | 72.4% |
| Within ±1 Notch | 94.2% |
| Within ±2 Notches | 98.7% |
| MAE | 0.42 notches |

### Case Studies

| Company | Year | Model Rating | Z-Score | Lead Time | Outcome |
|---------|------|--------------|---------|-----------|---------|
| Evergrande | 2020 | D (100%) | 0.53 | 12 months | Default Dec 2021 |
| Lehman Brothers | 2007 | D (100%) | 0.04 | 9 months | Bankrupt Sep 2008 |
| Enron | 2000 | D (100%) | 1.79 | 12 months | Bankrupt Dec 2001 |

---

## ⚠️ Bonus 2: Risk Warning Extraction

### Detection Categories

| Category | Severity | Example Patterns |
|----------|----------|------------------|
| Going Concern | Critical | "substantial doubt", "ability to continue" |
| Default/Covenant | Critical | "default", "covenant breach" |
| Liquidity Crisis | High | "liquidity constraints", "cash shortage" |
| Material Litigation | High | "class action", "securities litigation" |

### Validation Results

| Company | Year | Warnings | Critical | Lead Time |
|---------|------|----------|----------|-----------|
| Evergrande | 2020 | 15 | 5 | 12 months |
| Evergrande | 2021 | 18 | 8 | At crisis |
| Lehman Brothers | 2007 | 27 | 3 | 9 months |
| Enron | 2000 | 23 | 4 | 12 months |

**Result**: 4/4 bankruptcy cases correctly classified as CRITICAL.

---

## 💰 Bonus 3: Loan Pricing Model

### Methodology

**Credit Spread Model**:
```
Spread ≈ (PD × LGD) / (1 - PD)
```

**Monte Carlo Simulation** (GBM):
```
Spread_{t+Δt} = Spread_t × exp[(μ - σ²/2)Δt + σ√Δt × Z]
```

### Performance

| Metric | With Rating | Without Rating |
|--------|-------------|----------------|
| R² | 0.383 | 0.212 |
| RMSE | 275 bps | 311 bps |
| MAE | 80 bps | 101 bps |

### 95% Confidence Interval Example (BBB Loan)

| Percentile | Spread | Interest Rate | Price (1-month) |
|------------|--------|---------------|-----------------|
| 5th | 116 bps | 5.66% | 101.02 |
| 50th | 175 bps | 6.25% | 100.00 |
| 95th | 277 bps | 7.27% | 98.98 |

---

## 📚 References

1. Vélez-Pareja, I. (2007). "Forecasting Financial Statements with No Plugs and No Circularity."
2. Vélez-Pareja, I. (2009). "Constructing Consistent Financial Planning Models for Valuation."
3. Alonso, M., & Dupouy, H. (2024). "Large Language Models as Financial Analysts."
4. Schilit, H. M. (2010). "Financial Shenanigans: How to Detect Accounting Gimmicks & Fraud."
5. Altman, E. I. (1968). "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy."
6. Duffie, D., & Singleton, K. J. (1999). "Modeling Term Structures of Defaultable Bonds."

---

## 👤 Author

**Lihao Xiao**
- Email: lihao@ucsb.edu
- Institution: University of California, Santa Barbara
- Position: 2026 Summer Associate Candidate - JP Morgan MLCOE

---

## 📄 License

This project is licensed under the MIT License.

---

*Developed for JP Morgan Chase - Machine Learning Center of Excellence*  
*2026 Summer Associate – Time Series & Reinforcement Learning*
