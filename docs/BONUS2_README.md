# Bonus 2: Risk Warning Extraction Engine

## Overview

This module automatically extracts risk warnings and red flags from annual reports, identifying potential bankruptcy signals, qualified audit opinions, and other critical disclosures.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 RISK WARNING EXTRACTION ENGINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT: Annual Report (PDF)                                     │
│                          ↓                                       │
│  TEXT EXTRACTION                                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  • Full text extraction from all pages                    │  │
│  │  • Section identification (Audit, MD&A, Risk Factors)     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  PATTERN MATCHING                                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  8 Risk Categories × Multiple Keywords                    │  │
│  │  Severity Classification (Critical/High/Medium)           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  AUDITOR OPINION ANALYSIS                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  • Unqualified (Clean)                                    │  │
│  │  • Qualified                                              │  │
│  │  • Adverse                                                │  │
│  │  • Disclaimer of Opinion                                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  OUTPUT: Risk Assessment Report                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
ML-CESA/
├── data/
│   ├── annual_reports/                    # Bankruptcy case PDFs
│   │   ├── evergrande/
│   │   │   ├── ar2020.pdf                 # Evergrande 2020
│   │   │   └── car2021.pdf                # Evergrande 2021
│   │   ├── lehman/
│   │   │   └── lehman.pdf                 # Lehman Brothers 2007
│   │   └── enron/
│   │       └── EnronAnnualReport2000.pdf  # Enron 2000
│   └── bankruptcy_cases_cache.json        # Cached extraction results
├── src/financial_planning/credit_rating/
│   ├── risk_extractor.py                  # Main risk extraction engine
│   ├── test_risk_extractor.py             # Risk extractor tests
│   └── test_bankruptcy_cases.py           # Bankruptcy validation tests
```

## Usage

### Basic Usage

```bash
# Test on bankruptcy cases (Evergrande, Lehman, Enron)
python -m src.financial_planning.credit_rating.test_bankruptcy_cases

# Test risk extractor on specific PDF
python -m src.financial_planning.credit_rating.test_risk_extractor
```

### Python API

```python
from src.financial_planning.credit_rating.risk_extractor import RiskExtractor

# Initialize extractor
extractor = RiskExtractor()

# Extract risks from PDF
result = extractor.extract_from_pdf('data/annual_reports/evergrande/ar2020.pdf')

print(f"Risk Level: {result['overall_risk']}")
print(f"Total Warnings: {result['total_warnings']}")
print(f"Critical: {result['critical_count']}")
print(f"Going Concern: {result['going_concern']}")
```

## Risk Categories

| Category | Severity | Keywords |
|----------|----------|----------|
| Going Concern | Critical | "going concern", "substantial doubt", "ability to continue" |
| Qualified Opinion | Critical | "qualified opinion", "disclaimer", "adverse opinion" |
| Default/Covenant | Critical | "default", "covenant breach", "acceleration" |
| Liquidity | High | "liquidity risk", "cash flow concerns", "funding" |
| Litigation | High | "material litigation", "lawsuit", "legal proceedings" |
| Investigation | High | "SEC investigation", "regulatory inquiry" |
| Related Party | Medium | "related party", "affiliated transactions" |
| Control Weakness | Medium | "material weakness", "internal control deficiency" |

## Overall Risk Assessment

```python
def calculate_overall_risk(warnings):
    critical = count_by_severity(warnings, 'Critical')
    high = count_by_severity(warnings, 'High')
    
    if critical >= 2 or has_going_concern(warnings):
        return 'CRITICAL'
    elif critical >= 1 or high >= 3:
        return 'HIGH'
    elif high >= 1:
        return 'MEDIUM'
    else:
        return 'LOW'
```

## Validation Results

### Test Cases

| Company | Year | Risk Level | Warnings | Critical | Detection |
|---------|------|------------|----------|----------|-----------|
| Evergrande | 2020 | CRITICAL | 15 | 5 | 12 months before default |
| Evergrande | 2021 | CRITICAL | 18 | 8 | During crisis |
| Lehman Brothers | 2007 | CRITICAL | 27 | 3 | 9 months before bankruptcy |
| Enron | 2000 | HIGH | 8 | 1 | 12 months before bankruptcy |

**Detection Rate: 4/4 (100%)**

### Year-over-Year Comparison (Evergrande 2020 → 2021)

| Metric | 2020 | 2021 | Change |
|--------|------|------|--------|
| Total Warnings | 15 | 18 | +3 |
| Critical | 5 | 8 | +3 |
| Risk Level | CRITICAL | CRITICAL | — |
| New Issues | — | Default, Liquidation | Escalation |

## Output Format

```json
{
    "company": "Evergrande",
    "report_year": "2020",
    "overall_risk": "CRITICAL",
    "auditor_opinion": "Unqualified",
    "going_concern": true,
    "warnings": {
        "critical": 5,
        "high": 3,
        "medium": 7,
        "total": 15
    },
    "details": [
        {
            "category": "Going Concern",
            "severity": "Critical",
            "text": "...substantial doubt about the Group's ability to continue...",
            "page": 45
        }
    ]
}
```

## Key Warning Patterns

### Pattern 1: Real Estate Bubble (Evergrande)
- Excessive leverage
- Asset concentration in single sector
- Liquidity mismatch (short-term funding, long-term assets)

### Pattern 2: Financial Institution Collapse (Lehman)
- Extreme leverage (30x+)
- Funding fragility (overnight repos)
- Illiquid asset concentration

### Pattern 3: Accounting Fraud (Enron)
- Off-balance sheet vehicles (SPEs)
- Related party abuse
- Revenue manipulation schemes

## References

- Schilit, H. M. (2010). "Financial Shenanigans: How to Detect Accounting Gimmicks & Fraud."
- HKICPA (2009). "Example Auditor's Reports for SME-FRS."
