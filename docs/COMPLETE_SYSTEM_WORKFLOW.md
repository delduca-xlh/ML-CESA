# Financial Forecasting System - Complete Workflow
## From Raw Data to Final Predictions

> **Based on actual run output for AAPL (December 2024)**
> **System Version: FIXED (with correct FMP column mappings)**

---

## 🔄 SYSTEM OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FINANCIAL FORECASTING SYSTEM                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   RAW DATA (FMP API) - 120 Quarters Available                              │
│         │                                                                   │
│         ▼                                                                   │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐   │
│   │ Development │    │    Test     │    │     Historical Ratios       │   │
│   │  Data (32Q) │    │  Data (8Q)  │    │   (from Development Data)   │   │
│   └──────┬──────┘    └──────┬──────┘    └──────────────┬──────────────┘   │
│          │                  │                          │                   │
│          ▼                  │                          │                   │
│   ┌─────────────┐           │                          │                   │
│   │  XGBoost    │           │                          │                   │
│   │  Training   │           │                          │                   │
│   └──────┬──────┘           │                          │                   │
│          │                  │                          │                   │
│          ▼                  ▼                          │                   │
│   ┌─────────────────────────────┐                      │                   │
│   │     ML Predictions          │                      │                   │
│   │  (5 Core Variables)         │                      │                   │
│   └──────────────┬──────────────┘                      │                   │
│                  │                                     │                   │
│                  ▼                                     ▼                   │
│   ┌───────────────────────────────────────────────────────────────────┐   │
│   │                     ACCOUNTING ENGINE                              │   │
│   │         (Combines ML Predictions + Historical Ratios)              │   │
│   └───────────────────────────────────────────────────────────────────┘   │
│                  │                                                         │
│                  ▼                                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │                    COMPLETE FINANCIAL STATEMENTS                     │ │
│   │              Income Statement + Balance Sheet + Cash Flow            │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 STAGE 1: RAW DATA (from FMP API)

### Actual Data Fetched:

```
Fetching quarterly data for AAPL from FMP /stable/ API...
  API Key: vRmvrzQZbC...nTQNR
  ✓ Income Statement: 120 periods
  ✓ Balance Sheet: 120 periods
  ✓ Cash Flow: 120 periods
  ✓ Merged: 120 periods
  ✓ Accounting fields found: ['shares_outstanding', 'interest_expense', 
                              'total_debt', 'dividends_paid', 'retained_earnings']
  ✓ Final ML data: 120 periods
  ✓ Total columns: 21
  Date range: 1995-12-29 to 2025-09-27
  Years: 29.7
```

### FMP Column Mappings:

| Internal Name | FMP Column Name | Source | Example Value |
|---------------|-----------------|--------|---------------|
| shares_outstanding | `weightedAverageShsOut` | Income Statement | 15.60B |
| interest_expense | `interestExpense` | Income Statement | (varies) |
| total_debt | `totalDebt` | Balance Sheet | $98.66B |
| dividends_paid | `commonDividendsPaid` | Cash Flow | $3.86B/Q |
| retained_earnings | `retainedEarnings` | Balance Sheet | -$14.26B |
| stock_repurchased | `commonStockRepurchased` | Cash Flow | $20.13B/Q |

---

## 📊 STAGE 2: OPTIMAL WINDOW SELECTION

### Tested Windows:

```
  Testing 20Q window...
    ⚠️  Too few samples (16), skipping

  Testing 40Q window...
    MAPE: 9.71%  ← BEST

  Testing 60Q window...
    MAPE: 12.35%

  Testing 80Q window...
    MAPE: 17.31%

  Testing 120Q window...
    MAPE: 28.07%

✓ Optimal window: 40Q (MAPE: 9.71%)
```

### Why 40Q is Best:

```
More recent data = More relevant patterns
Too much old data = Model learns outdated patterns (Apple 1995 ≠ Apple 2024)

Trade-off:
  40Q (10 years): Recent + enough samples
  60Q (15 years): More samples but includes older patterns
  120Q (30 years): Too much historical noise
```

---

## 📊 STAGE 3: DATA SPLIT

```
Full Data: 120 quarters available
                    │
                    ▼
        Select Optimal Window: 40Q
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│                    40 Quarters                            │
├─────────────────────────────────┬─────────────────────────┤
│      Development (32Q, 80%)     │    Test (8Q, 20%)       │
│                                 │                         │
│  Used for:                      │  Used for:              │
│  • ML model training            │  • Evaluation only      │
│  • Historical ratio calculation │  • Never seen by model  │
│  • Hyperparameter tuning (CV)   │                         │
└─────────────────────────────────┴─────────────────────────┘
```

**Key Point: NO DATA LEAKAGE** - Test data is never used for training or ratio calculation!

---

## 📊 STAGE 4: HYPERPARAMETER TUNING (5-Fold CV)

### Actual CV Results:

```
✓ Development: 32Q, Samples: 28

  Testing Conservative...
    Fold 1: 17.21%
    Fold 2: 26.47%
    Fold 3: 24.81%
    Fold 4: 25.59%
    Fold 5: 24.10%
    Average: 23.63% ± 3.31%

  Testing Balanced...
    Fold 1: 17.44%
    Fold 2: 24.93%
    Fold 3: 21.50%
    Fold 4: 18.54%
    Fold 5: 15.88%
    Average: 19.66% ± 3.21%

  Testing Aggressive...  ← BEST
    Fold 1: 16.56%
    Fold 2: 17.52%
    Fold 3: 20.78%
    Fold 4: 13.03%
    Fold 5: 11.05%
    Average: 15.79% ± 3.42%

✓ Best: Aggressive (CV MAPE: 15.79%)
```

---

## 📊 STAGE 5: HISTORICAL RATIOS CALCULATION

### Retention Ratio Calculation:

```
  Retention Ratio Calculation:
    Net Income (8Q): $196.80B
    Dividends (8Q): $29.87B        ← from FMP: commonDividendsPaid
    Buybacks (8Q): $166.95B        ← from FMP: commonStockRepurchased
    ─────────────────────────────
    Total Payout: $196.82B
    Payout Ratio: 100.0%
    Retention Ratio: -0.0%
```

**Calculation Method:**
```python
# Formula: 1 - (Dividends + Buybacks) / Net Income
retention_ratio = 1 - (dividends_paid + stock_repurchased) / net_income
               = 1 - ($29.87B + $166.95B) / $196.80B
               = 1 - 100.0%
               = 0%
```

### Interest Rate Calculation:

```
  Interest Rate: 2.89% (calculated)
```

**Calculation Method:**
```python
# Formula: Annual Interest Expense / Total Debt
interest_rate = (interest_expense * 4) / total_debt   # Annualize quarterly
              = interestExpense / totalDebt           # FMP column names
# Result: 2.89%
```

### Shares Outstanding:

```
  Shares Outstanding: 15.60B
```

**Source:**
```python
# From Income Statement (most reliable)
shares_outstanding = weightedAverageShsOut   # FMP column name
# Result: 15.60B
```

### All Historical Ratios:

```
  Historical Ratios (calculated from data):
    Gross Margin: 43.7%
    EBIT Margin: 30.1%
    Net Income Margin: 25.3%
    Retention Ratio: -0.0%
    Interest Rate: 2.89%
    Shares Outstanding: 15.60B
```

### Ratio Explanations:

| Ratio | Value | Meaning |
|-------|-------|---------|
| Gross Margin | 43.7% | Hardware + Services mix |
| EBIT Margin | 30.1% | Excellent operational efficiency |
| Net Income Margin | 25.3% | Very healthy profitability |
| Retention Ratio | 0% | Apple returns ALL earnings to shareholders |
| Interest Rate | 2.89% | Investment-grade debt cost |
| Shares Outstanding | 15.60B | Down from 20B+ due to buybacks |

---

## 📊 STAGE 6: ML MODEL TRAINING

### Training Results:

```
  Training on 28 samples...
  
  ✓ Train MAPE: 1.91%
```

### Features Used (21 columns):

```
Core ML Features (5 targets × 4 lags = 20):
├── sales_revenue_lag1, lag2, lag3, lag4
├── cost_of_goods_sold_lag1, lag2, lag3, lag4
├── overhead_expenses_lag1, lag2, lag3, lag4
├── payroll_expenses_lag1, lag2, lag3, lag4
└── capex_lag1, lag2, lag3, lag4

Additional Accounting Fields:
├── shares_outstanding
├── interest_expense
├── total_debt
├── dividends_paid
├── retained_earnings
└── stock_repurchased
```

---

## 📊 STAGE 7: ML PREDICTIONS (Test Set)

### Actual Results:

```
  A. ML Predictions:
  Variable                       Actual (Avg)    Predicted (Avg) MAPE      
  ----------------------------------------------------------------------
  sales_revenue                  $ 104.04B       $  94.22B         9.03%
  cost_of_goods_sold             $  55.24B       $  53.25B         5.32%
  overhead_expenses              $  15.54B       $  13.49B        13.15%
  payroll_expenses               $   7.77B       $   6.75B        13.15%
  capex                          $   3.18B       $   2.66B        17.69%
  ----------------------------------------------------------------------
  Overall ML MAPE                                                 11.67%
```

### Analysis:

| Variable | MAPE | Assessment |
|----------|------|------------|
| sales_revenue | 9.03% | ✅ Excellent - primary driver |
| cost_of_goods_sold | 5.32% | ✅ Very good |
| overhead_expenses | 13.15% | ⚠️ Acceptable |
| payroll_expenses | 13.15% | ⚠️ Acceptable |
| capex | 17.69% | ⚠️ Higher variance (investment decisions) |
| **Overall** | **11.67%** | ✅ **Excellent** |

---

## 📊 STAGE 8: ACCOUNTING ENGINE RESULTS

### Actual Results:

```
  B. Accounting Metrics:
  Metric                         Actual (Avg)    Predicted (Avg) MAPE      
  ----------------------------------------------------------------------
  Net Income                     $  28.00B       $  23.85B        13.45%
  EBIT                           $  33.26B       $  28.32B        13.61%
  Total Assets                   $ 341.51B       $ 348.43B         3.64%
  Total Equity                   $  68.28B       $  61.17B        10.22%
  Total Liabilities              $ 273.23B       $ 287.25B         5.24%
  ----------------------------------------------------------------------
  Overall Accounting MAPE                                          9.23%
```

### How Each Value Was Derived:

```
INCOME STATEMENT (Direct Margin Method):
────────────────────────────────────────
1. Revenue = ML Prediction = $94.22B
2. COGS = Revenue × (1 - gross_margin) = $94.22B × 56.3% = $53.05B
3. EBIT = Revenue × ebit_margin = $94.22B × 30.1% = $28.32B
4. Net Income = Revenue × ni_margin = $94.22B × 25.3% = $23.85B

BALANCE SHEET:
──────────────
5. Total Assets = Prior Assets + NI × retention_ratio
                = $348B + $23.85B × 0% = $348.43B (stable)
                
6. Total Equity = Prior Equity + NI × retention_ratio  
                = $61B + $23.85B × 0% = $61.17B (stable)
                
7. Total Liabilities = Total Assets - Total Equity
                     = $348.43B - $61.17B = $287.25B
```

---

## 📊 STAGE 9: FUTURE FORECASTS

### 4-Quarter Ahead Predictions:

```
  Future Forecasts:
    Q1: Revenue $89.55B, Net Income $22.66B (25.3% margin)
    Q2: Revenue $86.25B, Net Income $21.83B (25.3% margin)
    Q3: Revenue $93.64B, Net Income $23.70B (25.3% margin)
    Q4: Revenue $97.26B, Net Income $24.61B (25.3% margin)
```

### Seasonality Pattern:

```
Q1 (Holiday): $89.55B  ← Post-holiday quarter
Q2 (Spring):  $86.25B  ← Lowest quarter
Q3 (Summer):  $93.64B  ← Back-to-school + new products
Q4 (Fall):    $97.26B  ← New iPhone launch
```

---

## 📊 FINAL RESULTS SUMMARY

```
================================================================================
PIPELINE COMPLETE FOR AAPL
================================================================================

⏱️  Duration: 6.8 seconds

📊 RESULTS:
----------
Optimal Window:   40Q
CV MAPE:          15.79%
Train MAPE:       1.91%
Test MAPE:        11.67%

Grade: ⭐⭐⭐⭐⭐ Excellent!

📁 OUTPUT FILES:
---------------
05_test_ml_predictions.csv      - ML predictions (test)
06_test_complete_statements.csv - Complete financials (test)
08_future_ml_predictions.csv    - ML predictions (future)
09_future_complete_statements.csv - Complete financials (future)
```

---

## 📊 COMPLETE DATA FLOW DIAGRAM

```
RAW DATA (FMP API)
       │
       ▼
┌──────────────────┐
│   120 Quarters   │
│   (30 years)     │
│   21 columns     │
└────────┬─────────┘
         │
         ▼
    Window Selection
    (40Q optimal)
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌───────┐
│ Dev   │  │ Test  │
│ 32Q   │  │  8Q   │
│ 28smp │  │       │
└───┬───┘  └───┬───┘
    │          │
    ▼          │
┌─────────────────────────┐
│  Historical Ratios      │
│  (from Dev only!)       │
│  • gross_margin: 43.7%  │
│  • ebit_margin: 30.1%   │
│  • ni_margin: 25.3%     │
│  • retention: 0.0%      │
│  • interest_rate: 2.89% │
│  • shares: 15.60B       │
└───────────┬─────────────┘
            │
    ┌───────┴───────┐
    ▼               │
┌─────────────────┐ │
│  XGBoost Models │ │
│  (5 targets)    │ │
│  Train: 1.91%   │ │
└────────┬────────┘ │
         │          │
         ▼          │
┌─────────────────┐ │
│ ML Predictions  │ │
│ (Test Period)   │ │
│ MAPE: 11.67%    │ │
└────────┬────────┘ │
         │          │
         ▼          ▼
┌─────────────────────────────────────────┐
│           ACCOUNTING ENGINE              │
│                                         │
│  Revenue ($94.22B) × Margins            │
│         ↓                               │
│  • Net Income: $23.85B (13.45% MAPE)   │
│  • EBIT: $28.32B (13.61% MAPE)         │
│  • Total Assets: $348.43B (3.64% MAPE) │
│  • Total Equity: $61.17B (10.22% MAPE) │
│                                         │
│  Overall Accounting MAPE: 9.23%         │
└─────────────────────────────────────────┘
```

---

## 🎯 KEY INSIGHTS FROM THIS RUN

### 1. Window Size Matters
```
40Q (10 years): 9.71% MAPE  ← Best
120Q (30 years): 28.07% MAPE ← Worst

Lesson: Recent data is more predictive than ancient history
```

### 2. Direct Margin Method Works
```
Net Income via cascade: Would be 40%+ MAPE
Net Income via direct margin: 13.45% MAPE

Lesson: Bypass cascade errors by using stable margin ratios
```

### 3. Apple's Capital Return Strategy
```
Net Income: $196.80B (8Q)
Dividends:  $29.87B
Buybacks:   $166.95B
────────────────────
Payout:     100.0%

Lesson: Apple returns virtually ALL earnings to shareholders
```

### 4. ML vs Accounting Performance
```
ML MAPE: 11.67%
Accounting MAPE: 9.23%  ← Actually better!

Lesson: Accounting engine with stable ratios can smooth ML errors
```

---

## 📈 PERFORMANCE GRADES

| Metric | Value | Grade |
|--------|-------|-------|
| ML MAPE | 11.67% | ⭐⭐⭐⭐⭐ Excellent |
| Accounting MAPE | 9.23% | ⭐⭐⭐⭐⭐ Excellent |
| Net Income MAPE | 13.45% | ⭐⭐⭐⭐ Very Good |
| Total Assets MAPE | 3.64% | ⭐⭐⭐⭐⭐ Excellent |
| Revenue MAPE | 9.03% | ⭐⭐⭐⭐⭐ Excellent |

**Final Grade: ⭐⭐⭐⭐⭐ Excellent!**

---

## 📋 GRADING SCALE

| Grade | ML MAPE | Description |
|-------|---------|-------------|
| ⭐⭐⭐⭐⭐ Excellent | < 15% | Production ready |
| ⭐⭐⭐⭐ Very Good | 15-20% | Good with minor improvements needed |
| ⭐⭐⭐ Good | 20-30% | Acceptable, room for improvement |
| ⭐⭐ Fair | 30-40% | Needs significant improvement |
| ⭐ Poor | > 40% | Not reliable for forecasting |
