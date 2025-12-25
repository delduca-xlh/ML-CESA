# Part 2: LLM Application in Financial Statement Analysis

## Overview

This module applies Large Language Models (Claude Sonnet 4) to two complementary tasks:
1. **Parts 2a-d**: Ensemble forecasting combining XGBoost with LLM-generated ratio adjustments
2. **Parts 2e-i**: Automated extraction of financial data from PDF annual reports

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PART 2: LLM APPLICATION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ENSEMBLE FORECASTING (Parts 2a-d)                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Approach 1: ML + Historical Ratios                       │  │
│  │  Approach 2: ML + LLM Ratios  ← Best for AAPL            │  │
│  │  Approach 3: Pure LLM                                     │  │
│  │                                                           │  │
│  │  Selection: argmin(MAPE) across approaches                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  PDF EXTRACTION (Parts 2e-i)                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. PDF Ingestion (URL or local file)                     │  │
│  │  2. Page Detection (keyword + LLM verification)           │  │
│  │  3. Vision Extraction (LLM reads table images)            │  │
│  │  4. JSON Output (structured financial data)               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
ML-CESA/
├── run_ensemble.py                        # Ensemble model runner
├── run_part2.py                           # LLM ratio integration
├── run_part2_pure_llm.py                  # Pure LLM comparison
├── src/financial_planning/
│   └── utils/
│       ├── llm_assumption_generator.py    # LLM ratio generation
│       └── pdf_extractor.py               # PDF extraction tool
├── outputs/
│   ├── ensemble/{ticker}/                 # Ensemble results
│   ├── part2_results/{ticker}/            # LLM integration results
│   └── part2_pure_llm/{ticker}/           # Pure LLM results
```

## LLM Model

- **Model**: Claude Sonnet 4
- **API Version**: `claude-sonnet-4-20250514`
- **Temperature**: 0 (deterministic)
- **Context Window**: 200K tokens

## Part 2a-d: Ensemble Forecasting

### Usage

```bash
# Run ensemble for Apple
python run_ensemble.py AAPL

# Run ensemble for Goldman Sachs
python run_ensemble.py GS

# Run Part 2 LLM integration only
python run_part2.py AAPL

# Run pure LLM comparison
python run_part2_pure_llm.py AAPL
```

### Three Approaches

| Approach | Driver Prediction | Ratio Source | Best For |
|----------|-------------------|--------------|----------|
| ML + Historical | XGBoost | Historical averages | Stable companies |
| ML + LLM Ratios | XGBoost | LLM-adjusted ratios | Forward-looking |
| Pure LLM | LLM | LLM | Not recommended |

### Results

| Company | Best Approach | Driver MAPE | Accounting MAPE | Overall |
|---------|---------------|-------------|-----------------|---------|
| AAPL | ML + LLM Ratios | 11.67% | 9.60% | **10.63%** |
| GS | ML + Historical | 12.78% | 13.57% | **13.17%** |

### Key Findings

- **LLM adjustments are conditional**: Improved AAPL (+0.32%), worsened GS (-2.01%)
- **Pure LLM underperforms**: 40-200% higher driver MAPE than ML
- **Adaptive selection essential**: Framework chooses best approach per company

## Part 2e-i: PDF Extraction

### Usage

```bash
# Extract from predefined company
python -m src.financial_planning.utils.pdf_extractor GM
python -m src.financial_planning.utils.pdf_extractor MICROSOFT
python -m src.financial_planning.utils.pdf_extractor LVMH

# Extract from local PDF
python -m src.financial_planning.utils.pdf_extractor /path/to/annual_report.pdf

# Extract from URL
python -m src.financial_planning.utils.pdf_extractor "https://example.com/report.pdf"
```

### Predefined Companies

| Company | Ticker | Standard | Currency |
|---------|--------|----------|----------|
| General Motors | GM | US GAAP | USD |
| Microsoft | MICROSOFT | US GAAP | USD |
| Google | GOOGLE | US GAAP | USD |
| JPMorgan | JPMORGAN | US GAAP | USD |
| Exxon Mobil | EXXON | US GAAP | USD |
| Alibaba | ALIBABA | US GAAP | RMB |
| LVMH | LVMH | IFRS | EUR |
| Tencent | TENCENT | IFRS | RMB |
| Volkswagen | VOLKSWAGEN | IFRS | EUR |

### Output Format

```json
{
    "company": "General Motors",
    "period": "Year ending 2023-12-31",
    "currency": "USD",
    "unit": "millions",
    "income_statement": {
        "revenue": 171842,
        "cost_of_goods_sold": 149412,
        "gross_profit": 22430,
        "operating_income": 12235,
        "net_income": 10127
    },
    "balance_sheet": {
        "total_assets": 273064,
        "total_liabilities": 212879,
        "total_equity": 60185
    },
    "ratios": {
        "gross_margin": 0.1305,
        "net_margin": 0.0589,
        "debt_to_equity": 1.70,
        "interest_coverage": 13.43
    }
}
```

## Output Files

### Ensemble Results (`outputs/ensemble/{ticker}/`)

| File | Description |
|------|-------------|
| `ensemble_results.json` | All approaches comparison |
| `ensemble_predictions.csv` | Best approach predictions |
| `ensemble_summary.csv` | Summary metrics |

### Part 2 Results (`outputs/part2_results/{ticker}/`)

| File | Description |
|------|-------------|
| `llm_assumptions.json` | LLM-generated ratios |
| `part2_statements.csv` | Predicted statements |
| `comparison.json` | ML vs LLM comparison |
| `cfo_report.md` | CFO recommendations |

### Pure LLM Results (`outputs/part2_pure_llm/{ticker}/`)

| File | Description |
|------|-------------|
| `llm_predictions.csv` | Pure LLM predictions |
| `comparison.json` | Comparison with ML |

## References

- Alonso, M., & Dupouy, H. (2024). "Large Language Models as Financial Analysts."
- Farr, M., et al. (2025). "AI Determinants of Success and Failure: Financial Statements."
- Zhang, H., et al. (2025). "Financial Statement Checking Recognition System Based on LLMs."
