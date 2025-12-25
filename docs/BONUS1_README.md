# Bonus 1: Credit Rating Model

## Overview

This module implements a credit rating prediction system using ordinal logistic regression, combined with fraud detection capabilities using Altman Z-Score and Beneish M-Score.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CREDIT RATING SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT: Financial Ratios from Annual Report                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  • Debt/Equity      • Interest Coverage                   │  │
│  │  • Current Ratio    • Net Margin                          │  │
│  │  • ROA              • Debt/EBITDA                         │  │
│  │  • Log(Assets)                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ORDINAL LOGISTIC REGRESSION                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  P(Rating ≤ j) = σ(θⱼ - β'X)                              │  │
│  │  Cost-sensitive learning for class imbalance              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  FRAUD DETECTION LAYER                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Altman Z-Score: Bankruptcy prediction                    │  │
│  │  Beneish M-Score: Earnings manipulation                   │  │
│  │  Red Flag Detection: 9 warning indicators                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  OUTPUT: Rating (AAA-D) + Risk Assessment                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
ML-CESA/
├── data/
│   ├── credit_rating_training_data.csv    # 447 companies with ratings
│   ├── credit_rating_model.pkl            # Trained model
│   ├── credit_rating_metadata.json        # Model metadata
│   ├── credit_rating_raw_data.json        # Raw fetched data
│   └── credit_rating_failed_tickers.txt   # Failed ticker list
├── src/financial_planning/credit_rating/
│   ├── __init__.py
│   ├── credit_rating_system.py            # Main rating system
│   ├── ordinal_lr.py                      # Ordinal logistic regression
│   ├── fraud_detector.py                  # Z-Score & M-Score
│   ├── trainer.py                         # Model training
│   ├── training_data.py                   # Data preparation
│   ├── fetch_training_data.py             # Data fetching from FMP
│   ├── rating_pipeline.py                 # Rating pipeline
│   ├── train_and_save_model.py            # Model persistence
│   ├── test_evergrande.py                 # Evergrande test case
│   └── test_bankruptcy_cases.py           # Bankruptcy validation
```

## Usage

### Training

```bash
# Fetch training data (447 companies)
python -m src.financial_planning.credit_rating.fetch_training_data

# Train the credit rating model
python -m src.financial_planning.credit_rating.trainer

# Or use the combined script
python -m src.financial_planning.credit_rating.train_and_save_model
```

### Rating a Company

```bash
# Rate a company by ticker
python -m src.financial_planning.credit_rating.rating_pipeline AAPL

# Test on Evergrande
python -m src.financial_planning.credit_rating.test_evergrande

# Test on all bankruptcy cases
python -m src.financial_planning.credit_rating.test_bankruptcy_cases
```

### Python API

```python
from src.financial_planning.credit_rating.credit_rating_system import CreditRatingSystem

# Initialize system
system = CreditRatingSystem()
system.load_model('data/credit_rating_model.pkl')

# Rate a company
result = system.rate_company(
    debt_to_equity=1.5,
    interest_coverage=5.0,
    current_ratio=1.2,
    net_margin=0.08,
    roa=0.05,
    debt_to_ebitda=3.0,
    total_assets=10e9
)

print(f"Rating: {result['rating']}")
print(f"Probability: {result['probability']:.1%}")
print(f"Z-Score: {result['z_score']:.2f}")
```

## Model Details

### Features (7 Ratios)

| Feature | Description | Importance |
|---------|-------------|------------|
| Interest Coverage | EBIT / Interest Expense | 28.5% |
| Debt/EBITDA | Total Debt / EBITDA | 18.2% |
| Net Margin | Net Income / Revenue | 14.7% |
| ROA | Return on Assets | 12.3% |
| Debt/Equity | Total Debt / Equity | 10.1% |
| Current Ratio | Current Assets / Current Liabilities | 8.9% |
| Log(Assets) | Size proxy | 7.3% |

### Training Data

- **Source**: FMP API + S&P/Moody's ratings
- **Companies**: 447
- **File**: `data/credit_rating_training_data.csv`

## Performance

| Metric | Value |
|--------|-------|
| Exact Accuracy | 72.4% |
| Within ±1 Notch | 94.2% |
| Within ±2 Notches | 98.7% |

## Fraud Detection

### Altman Z-Score

| Zone | Z-Score | Interpretation |
|------|---------|----------------|
| Safe | > 2.9 | Low bankruptcy risk |
| Grey | 1.23 - 2.9 | Uncertain |
| Distress | < 1.23 | High bankruptcy risk |

### Beneish M-Score

**Threshold**: M > -1.78 indicates likely manipulation

### Red Flags (9 Indicators)

1. Negative working capital
2. Cash crisis (cash < 5% of assets)
3. Excessive inventory (> 50% of revenue)
4. Extreme leverage (D/E > 5)
5. No interest coverage (< 1)
6. Negative equity
7. Operating losses
8. Asset quality deterioration
9. Unusual receivables growth

## Validation: Bankruptcy Cases

| Company | Year | Model Rating | Z-Score | Outcome |
|---------|------|--------------|---------|---------|
| Evergrande | 2020 | D (100%) | 0.53 | Default Dec 2021 |
| Lehman Brothers | 2007 | D (100%) | 0.04 | Bankrupt Sep 2008 |
| Enron | 2000 | D (100%) | 1.79 | Bankrupt Dec 2001 |

## References

- Altman, E. I. (1968). "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy."
- Beneish, M. D. (1999). "The Detection of Earnings Manipulation."
- Schilit, H. M. (2010). "Financial Shenanigans: How to Detect Accounting Gimmicks & Fraud."
