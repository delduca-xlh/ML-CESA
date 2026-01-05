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
│  │                                                           │  │
│  │  Approach 1: ML Only (XGBoost + Historical Ratios)       │  │
│  │    → Best for: Stable companies (Costco, Exxon)          │  │
│  │                                                           │  │
│  │  Approach 2: ML + LLM Ratios ★                           │  │
│  │    → Best for: Evolving margins (Apple, P&G, Alphabet)   │  │
│  │    → Apple: 9.93% → 6.24% (-37% error reduction)         │  │
│  │                                                           │  │
│  │  Approach 3: Pure LLM                                     │  │
│  │    → Best for: Complex structures (Goldman Sachs)        │  │
│  │                                                           │  │
│  │  Selection: argmin(MAPE) across rolling validation       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  PDF EXTRACTION (Parts 2e-i)                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Phase 1 (Local, 0 tokens):                               │  │
│  │    → Keyword scan with 24 patterns                        │  │
│  │    → Identify candidate pages (2-5 pages)                 │  │
│  │                                                           │  │
│  │  Phase 2 (LLM, ~9,000 tokens):                           │  │
│  │    → Vision extraction from page images                   │  │
│  │    → Structured JSON output                               │  │
│  │                                                           │  │
│  │  Result: 98.5% token savings (603K → 9K)                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## LLM Model

- **Model**: Claude Sonnet 4
- **API Version**: `claude-sonnet-4-20250514`
- **Temperature**: 0 (deterministic)
- **Context Window**: 200K tokens

## Part 2a-d: Ensemble Forecasting

### Three Approaches Compared

| Approach | Driver Source | Ratio Source | Strengths |
|----------|---------------|--------------|-----------|
| ML Only | XGBoost | Historical averages | Statistical rigor, extrapolation |
| ML + LLM | XGBoost | LLM-adjusted ratios | Forward-looking insights |
| Pure LLM | Claude Sonnet | Claude Sonnet | Business context, complex structures |

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

### Results: Apple Rolling Validation (5 Rounds)

**ML + LLM Ensemble (Best Approach)**:

| Metric | R1 | R2 | R3 | R4 | R5 | Mean |
|--------|-----|-----|-----|-----|-----|------|
| Revenue | 4.8% | 2.4% | 8.2% | 6.6% | 0.0% | **4.4%** |
| Net Income | 22.1% | 15.5% | 5.8% | 10.1% | 2.4% | **11.2%** |
| Total Assets | 7.4% | 7.9% | 1.1% | 5.7% | 1.2% | **4.7%** |
| Total Equity | 16.1% | 17.6% | 2.0% | 3.6% | 3.4% | **8.5%** |

### Multi-Company Comparison

| Company | ML Only | ML + LLM | Pure LLM | Best |
|---------|---------|----------|----------|------|
| AAPL | 9.93% | **6.24%** | 7.89% | ML + LLM |
| COST | **5.52%** | 6.18% | 8.45% | ML Only |
| PG | 7.36% | **7.21%** | 9.12% | ML + LLM |
| GOOGL | 12.13% | **9.87%** | 11.23% | ML + LLM |
| NFLX | 11.45% | 10.32% | **8.92%** | Pure LLM |
| GS | 17.23% | 12.45% | **3.52%** | Pure LLM |
| XOM | **8.79%** | 9.23% | 14.56% | ML Only |

### Key Findings

1. **LLM adjustments are conditional**: Improved Apple (+37%), P&G (+2%), worsened Costco (-12%)
2. **Pure LLM excels at complex structures**: Goldman Sachs 3.52% vs 17.23% (ML Only)
3. **Adaptive selection essential**: No single approach wins for all companies
4. **LLM adds value for margin prediction**: +54% improvement for Net Income

## Part 2e-i: PDF Extraction

### Token Efficiency

| Report | Pages | Naive Tokens | Our Method | Savings |
|--------|-------|--------------|------------|---------|
| Shell 2023 | 402 | 603,000 | ~9,000 | **98.5%** |
| GM 2023 | 156 | 234,000 | ~7,500 | **96.8%** |
| Microsoft 2023 | 88 | 132,000 | ~6,000 | **95.5%** |

### Error Types and Mitigations

| Error Type | Problem | Solution |
|------------|---------|----------|
| Notes vs. Statements | LLM selects "Notes to Financial Statements" | Exclude pages with "NOTES TO" in header |
| TOC vs. Actual | Table of Contents mentions page numbers | Title must be in first 5 lines |
| Summary vs. Complete | Financial highlights selected | Validate REVENUE/ASSETS keywords |
| Multi-Page Statements | Balance sheet spans 2-3 pages | Auto-include adjacent pages |
| Missing Statement Type | Found BS but missed IS | Search nearby pages (±3) |
| Naming Variations | Different GAAP/IFRS conventions | 24 patterns covering all standards |

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

### Supported Companies

| Company | Standard | Currency | Pages | Tokens | Status |
|---------|----------|----------|-------|--------|--------|
| General Motors | US GAAP | USD | 156 | ~7,500 | ✅ |
| Microsoft | US GAAP | USD | 88 | ~6,000 | ✅ |
| Shell | IFRS | USD | 402 | ~9,000 | ✅ |
| LVMH | IFRS | EUR | 316 | ~8,500 | ✅ |
| JP Morgan | US GAAP (Bank) | USD | 312 | ~9,500 | ✅ |
| Tencent | IFRS | CNY | 284 | ~8,000 | ✅ |
| Alibaba | US GAAP | RMB | 198 | ~7,000 | ✅ |
| Google | US GAAP | USD | 98 | ~6,500 | ✅ |
| Exxon Mobil | US GAAP | USD | 132 | ~7,000 | ✅ |

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

### Robustness Testing

- **Consistency**: 100% identical results across 5 runs (temperature=0)
- **Multi-format support**: US GAAP, IFRS, multiple currencies
- **Error handling**: Graceful fallback for missing data

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

## CFO/CEO Recommendations (Part 2d)

The system generates actionable recommendations based on forecasted financials:

**Example for Apple (ML + LLM Forecast)**:

**CFO Recommendations**:
- Maintain $90B+ annual buyback program given strong FCF generation
- Consider modest leverage increase (current interest coverage 13x vs 8x threshold)
- Monitor services margin trajectory (53.7% forecast vs 54.7% historical)

**CEO Recommendations**:
- Enhance services segment disclosure given 25%+ revenue contribution
- Quantify AI/ML R&D investment to support premium valuation
- Address 200-300bp margin compression scenario in guidance

## References

- Alonso, M., & Dupouy, H. (2024). "Large Language Models as Financial Analysts."
- Farr, M., et al. (2025). "AI Determinants of Success and Failure: Financial Statements."
- Zhang, H., et al. (2025). "Financial Statement Checking Recognition System Based on LLMs."
