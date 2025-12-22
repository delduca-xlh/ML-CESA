# 🔬 Feature-Level Workflow: Input → Output at Each Step

## 📊 Complete Feature Transformation Pipeline

---

## 🎯 **OVERVIEW DIAGRAM**

```
Raw Yahoo Data (10 columns)
         ↓
Feature Engineering (95 columns)
         ↓
LSTM Training (learns patterns)
         ↓
ML Predictions (5 columns)
         ↓
ForecastInputs (8 columns)
         ↓
Financial Statements (150+ columns)
         ↓
Final Output (Excel with validated results)
```

---

# 📋 **DETAILED STEP-BY-STEP FEATURE FLOW**

---

## **STEP 1: Yahoo Finance Data Loading**

**File**: `utils/yahoo_finance_fetcher.py`
**Method**: `fetch_company_data()` + `extract_time_series_features()`

### **Input**: 
```
Company ticker: 'AAPL'
```

### **Process**:
```python
# Fetch raw data from Yahoo Finance API
stock = yf.Ticker('AAPL')
balance_sheet = stock.balance_sheet
income_stmt = stock.income_stmt
cash_flow = stock.cashflow
```

### **Output Features** (20 periods × 10 columns):
```python
{
    'date': [2020-12-31, 2021-12-31, 2022-12-31, ...],
    
    # Balance Sheet
    'total_assets': [338516000000, 351002000000, 352755000000, ...],
    'total_liabilities': [258549000000, 287912000000, 302083000000, ...],
    'total_equity': [65339000000, 63090000000, 50672000000, ...],
    'cash': [38016000000, 34940000000, 23646000000, ...],
    'accounts_receivable': [16120000000, 26278000000, 28184000000, ...],
    'inventory': [4061000000, 6580000000, 4946000000, ...],
    'ppe': [36766000000, 39440000000, 42117000000, ...],
    'accounts_payable': [42296000000, 54763000000, 64115000000, ...],
    'short_term_debt': [13769000000, 15613000000, 21110000000, ...],
    'long_term_debt': [98667000000, 109106000000, 98959000000, ...],
    
    # Income Statement
    'revenue': [274515000000, 365817000000, 394328000000, ...],
    'cost_of_revenue': [169559000000, 212981000000, 223546000000, ...],
    'gross_profit': [104956000000, 152836000000, 170782000000, ...],
    'operating_income': [66288000000, 108949000000, 119437000000, ...],
    'net_income': [57411000000, 94680000000, 99803000000, ...],
    'ebit': [67091000000, 109207000000, 119103000000, ...],
    'interest_expense': [2873000000, 2645000000, 2931000000, ...],
    
    # Cash Flow
    'operating_cash_flow': [80674000000, 104038000000, 122151000000, ...],
    'capex': [-7309000000, -11085000000, -10708000000, ...],
    'free_cash_flow': [73365000000, 92953000000, 111443000000, ...]
}
```

**Output Shape**: `(20 periods, 20 raw features)`

---

## **STEP 2: Column Mapping & Derivation**

**File**: `balance_sheet_forecaster.py`
**Method**: `_load_from_yahoo_finance()`

### **Input**: 
Raw Yahoo Finance features (20 columns)

### **Process**:
```python
# Map Yahoo columns to expected names
column_mapping = {
    'revenue': 'sales_revenue',
    'cost_of_revenue': 'cost_of_goods_sold',
    'total_assets': 'total_assets',
    'total_liabilities': 'total_liabilities',
    'total_equity': 'total_equity',
    'net_income': 'net_income',
    'operating_income': 'ebit'
}

# Derive missing fields
df['overhead_expenses'] = df['sales_revenue'] - df['cost_of_goods_sold'] - df['ebit']
df['payroll_expenses'] = df['overhead_expenses'] * 0.5
df['capex'] = abs(df['capex'])
```

### **Output Features** (20 periods × 10 columns):
```python
{
    'date': datetime index,
    
    # Core ML targets (what we'll predict)
    'sales_revenue': [274515000000, 365817000000, 394328000000, ...],
    'cost_of_goods_sold': [169559000000, 212981000000, 223546000000, ...],
    'overhead_expenses': [38668000000, 43887000000, 51345000000, ...],  # Derived
    'payroll_expenses': [19334000000, 21943500000, 25672500000, ...],  # Derived
    'capex': [7309000000, 11085000000, 10708000000, ...],  # Cleaned
    
    # Additional context
    'total_assets': [338516000000, 351002000000, 352755000000, ...],
    'total_liabilities': [258549000000, 287912000000, 302083000000, ...],
    'total_equity': [65339000000, 63090000000, 50672000000, ...],
    'net_income': [57411000000, 94680000000, 99803000000, ...]
}
```

**Output Shape**: `(20 periods, 10 cleaned features)`

---

## **STEP 3: Feature Engineering for ML**

**File**: `balance_sheet_forecaster.py`
**Method**: `prepare_features()`

### **Input**: 
Cleaned features (10 columns)

### **Process**:
```python
# Create comprehensive feature set for LSTM

# 1. Autoregressive features (lags 1-12)
for var in ['sales_revenue', 'cost_of_goods_sold', 'overhead_expenses', 
            'payroll_expenses', 'capex']:
    for lag in range(1, 13):
        features[f'{var}_lag{lag}'] = df[var].shift(lag)

# 2. Growth rates
for var in target_variables:
    features[f'{var}_growth'] = df[var].pct_change()

# 3. Financial ratios
features['cogs_margin'] = df['cost_of_goods_sold'] / df['sales_revenue']
features['leverage_ratio'] = df['total_liabilities'] / df['total_assets']

# 4. Moving averages
for var in target_variables:
    features[f'{var}_ma3'] = df[var].rolling(3).mean()
```

### **Output Features** (8 periods × 95 columns):
```python
{
    # Autoregressive features (5 vars × 12 lags = 60 features)
    'sales_revenue_lag1': [365817000000, 394328000000, ...],
    'sales_revenue_lag2': [274515000000, 365817000000, ...],
    'sales_revenue_lag3': [...],
    # ... lag4 through lag12
    
    'cost_of_goods_sold_lag1': [212981000000, 223546000000, ...],
    'cost_of_goods_sold_lag2': [169559000000, 212981000000, ...],
    # ... lag3 through lag12
    
    'overhead_expenses_lag1': [...],
    # ... lag2 through lag12
    
    'payroll_expenses_lag1': [...],
    # ... lag2 through lag12
    
    'capex_lag1': [...],
    # ... lag2 through lag12
    
    # Growth rates (5 features)
    'sales_revenue_growth': [0.332, 0.078, 0.082, ...],
    'cost_of_goods_sold_growth': [0.256, 0.050, 0.045, ...],
    'overhead_expenses_growth': [0.135, 0.170, 0.120, ...],
    'payroll_expenses_growth': [0.135, 0.170, 0.120, ...],
    'capex_growth': [0.517, -0.034, 0.062, ...],
    
    # Financial ratios (2 features)
    'cogs_margin': [0.618, 0.582, 0.567, ...],
    'leverage_ratio': [0.764, 0.820, 0.856, ...],
    
    # Moving averages (5 vars × 3-period MA = 5 features)
    'sales_revenue_ma3': [344886666667, 378220000000, ...],
    'cost_of_goods_sold_ma3': [202028666667, 220026000000, ...],
    'overhead_expenses_ma3': [...],
    'payroll_expenses_ma3': [...],
    'capex_ma3': [...]
}

# Total: 60 + 5 + 2 + 5 = 72 features
# Plus 5 target variables = 77 columns
# After removing NaN rows: ~8 valid samples
```

### **Target Variables (y)**:
```python
y = {
    'sales_revenue': [394328000000, 413659000000, ...],
    'cost_of_goods_sold': [223546000000, 234234000000, ...],
    'overhead_expenses': [51345000000, 53789000000, ...],
    'payroll_expenses': [25672500000, 26894500000, ...],
    'capex': [10708000000, 11234000000, ...]
}
```

**Output Shape**: 
- X: `(8 samples, 72 features)`
- y: `(8 samples, 5 targets)`

---

## **STEP 4: Sequence Creation for LSTM**

**File**: `balance_sheet_forecaster.py`
**Method**: `create_sequences()`

### **Input**: 
- X: (8, 72)
- y: (8, 5)
- lookback_periods: 12

### **Process**:
```python
# Can't create sequences - need at least 12 periods!
# With only 8 samples after feature engineering, we need more data

# Real scenario with quarterly data (80 periods):
X_sequences = []
y_sequences = []

for i in range(12, 80):
    # Take last 12 periods as input
    X_sequences.append(X[i-12:i])  # Shape: (12, 72)
    y_sequences.append(y[i])        # Shape: (5,)

X_sequences = np.array(X_sequences)  # Shape: (68, 12, 72)
y_sequences = np.array(y_sequences)  # Shape: (68, 5)
```

### **Output Features**:
```python
# Each sample is now a sequence
X_sequences[0] = [
    [features_period_1],   # 72 features
    [features_period_2],   # 72 features
    # ...
    [features_period_12]   # 72 features
]
# Shape: (68 sequences, 12 timesteps, 72 features)

y_sequences[0] = [sales, cogs, overhead, payroll, capex]
# Shape: (68 sequences, 5 targets)
```

**Output Shape**: 
- X_sequences: `(68, 12, 72)` - 3D tensor for LSTM
- y_sequences: `(68, 5)`

---

## **STEP 5: Scaling**

**File**: `balance_sheet_forecaster.py`
**Method**: `train()`

### **Input**: 
- X_sequences: (68, 12, 72)
- y_sequences: (68, 5)

### **Process**:
```python
# Fit scalers on training data
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

# Reshape for scaling
X_reshaped = X_sequences.reshape(-1, 72)
X_scaled = feature_scaler.fit_transform(X_reshaped)
X_sequences_scaled = X_scaled.reshape(68, 12, 72)

y_scaled = target_scaler.fit_transform(y_sequences)
```

### **Output Features**:
```python
# All features now have mean=0, std=1
X_sequences_scaled[0][0] = [
    -0.234,  # sales_revenue_lag1 (scaled)
    1.456,   # sales_revenue_lag2 (scaled)
    -0.823,  # sales_revenue_lag3 (scaled)
    # ... 69 more scaled features
]

y_sequences_scaled[0] = [
    0.567,   # sales_revenue (scaled)
    -0.234,  # cost_of_goods_sold (scaled)
    0.891,   # overhead_expenses (scaled)
    0.234,   # payroll_expenses (scaled)
    -0.456   # capex (scaled)
]
```

**Output Shape**: Same as input, but normalized

---

## **STEP 6: LSTM Training**

**File**: `balance_sheet_forecaster.py`
**Method**: `train()` → `ml_model.fit()`

### **Input**: 
- X_train: (54, 12, 72) - 80% of 68 sequences
- y_train: (54, 5)
- X_val: (14, 12, 72) - 20% validation

### **Process**:
```python
# LSTM learns temporal patterns
# Forward pass through network:
Input: (12, 72) sequence
  ↓
LSTM Layer 1 (128 units): Learns long-term dependencies
  → Hidden states: (12, 128)
  ↓
Dropout (0.2)
  ↓
LSTM Layer 2 (64 units): Refines patterns
  → Final hidden state: (64,)
  ↓
Dropout (0.2)
  ↓
Dense Layer (32 units, ReLU)
  → Transformed: (32,)
  ↓
Output Layer (5 units, Linear)
  → Predictions: (5,)

# Backpropagation:
Loss = MSE(predicted, actual)
Update ~85,000 parameters via Adam optimizer
```

### **Output - Trained Model**:
```python
# After 50 epochs of training:
model.weights = {
    'lstm_1/kernel': array of shape (72, 512),
    'lstm_1/recurrent_kernel': array of shape (128, 512),
    'lstm_1/bias': array of shape (512,),
    'lstm_2/kernel': array of shape (128, 256),
    'lstm_2/recurrent_kernel': array of shape (64, 256),
    'lstm_2/bias': array of shape (256,),
    'dense_1/kernel': array of shape (64, 32),
    'dense_1/bias': array of shape (32,),
    'output/kernel': array of shape (32, 5),
    'output/bias': array of shape (5,)
}

# Training metrics:
final_train_loss: 0.0234
final_val_loss: 0.0456
final_val_mae: 0.1234
final_val_mape: 5.67%
```

**Output**: Trained LSTM model with ~85,000 learned parameters

---

## **STEP 7: Prediction**

**File**: `balance_sheet_forecaster.py`
**Method**: `predict_next_period()`

### **Input**: 
Last 12 periods of features (scaled)

### **Process**:
```python
# Take most recent data
recent_features = prepare_features(historical_data[-12:])
recent_scaled = feature_scaler.transform(recent_features)

# Reshape for LSTM
X_input = recent_scaled.reshape(1, 12, 72)

# Forward pass through trained LSTM
y_pred_scaled = model.predict(X_input)
# Shape: (1, 5)

# Inverse scale to get actual values
y_pred = target_scaler.inverse_transform(y_pred_scaled)
```

### **Output - ML Predictions** (1 period, 5 values):
```python
{
    'sales_revenue': 433890000000,      # Predicted!
    'cost_of_goods_sold': 245670000000, # Predicted!
    'overhead_expenses': 56789000000,   # Predicted!
    'payroll_expenses': 28394500000,    # Predicted!
    'capex': 11890000000               # Predicted!
}
```

**Output Shape**: `(5 predicted values)` for next period

---

## **STEP 8: Rolling Forecast (Multiple Periods)**

**File**: `balance_sheet_forecaster.py`
**Method**: `forecast_balance_sheet(periods=4)`

### **Input**: 
Trained model + historical data

### **Process**:
```python
all_predictions = []
current_data = historical_data.copy()

for period in [1, 2, 3, 4]:
    # Predict period i
    predictions = predict_next_period(current_data)
    all_predictions.append(predictions)
    
    # Add prediction to data for next iteration
    current_data = append(current_data, predictions)
```

### **Output - 4 Periods of Predictions**:
```python
{
    'period': [1, 2, 3, 4],
    'sales_revenue': [
        433890000000,  # Period 1
        455234000000,  # Period 2
        477123000000,  # Period 3
        499876000000   # Period 4
    ],
    'cost_of_goods_sold': [
        245670000000,
        257890000000,
        270345000000,
        283567000000
    ],
    'overhead_expenses': [
        56789000000,
        59567000000,
        62456000000,
        65478000000
    ],
    'payroll_expenses': [
        28394500000,
        29783500000,
        31228000000,
        32739000000
    ],
    'capex': [
        11890000000,
        12456000000,
        13078000000,
        13723000000
    ]
}
```

**Output Shape**: `(4 periods, 5 predictions)`

---

## **STEP 9: Convert to ForecastInputs**

**File**: `forecaster_integration.py`
**Method**: `_build_financial_statements()`

### **Input**: 
ML predictions (4 periods × 5 values)

### **Process**:
```python
# Derive additional required fields
sales = ml_predictions['sales_revenue']
cogs = ml_predictions['cost_of_goods_sold']

# Estimate unit economics
avg_revenue = mean(sales)
sales_volume = avg_revenue / 100.0  # Estimate
selling_price = 100.0               # Estimate
unit_cost = cogs / sales_volume     # Derive

# Create ForecastInputs
forecast_inputs = ForecastInputs(
    sales_revenue=[433890M, 455234M, 477123M, 499876M],
    sales_volume_units=[4338900, 4552340, 4771230, 4998760],
    selling_price=[100.0, 100.0, 100.0, 100.0],
    cost_of_goods_sold=[245670M, 257890M, 270345M, 283567M],
    unit_cost=[56.6, 56.6, 56.7, 56.7],
    overhead_expenses=[56789M, 59567M, 62456M, 65478M],
    payroll_expenses=[28394.5M, 29783.5M, 31228M, 32739M],
    capex_forecast=[11890M, 12456M, 13078M, 13723M]
)
```

### **Output - ForecastInputs** (4 periods × 8 fields):
```python
ForecastInputs(
    sales_revenue=[433890M, 455234M, 477123M, 499876M],
    sales_volume_units=[4338900, 4552340, 4771230, 4998760],
    selling_price=[100.0, 100.0, 100.0, 100.0],
    cost_of_goods_sold=[245670M, 257890M, 270345M, 283567M],
    unit_cost=[56.6, 56.6, 56.7, 56.7],
    overhead_expenses=[56789M, 59567M, 62456M, 65478M],
    payroll_expenses=[28394.5M, 29783.5M, 31228M, 32739M],
    capex_forecast=[11890M, 12456M, 13078M, 13723M]
)
```

**Output Shape**: `(4 periods, 8 input fields)`

---

## **STEP 10: Financial Model - Intermediate Tables**

**File**: `models/intermediate_tables.py`
**Method**: `build_all_tables()`

### **Input**: 
ForecastInputs (8 fields)

### **Process**:
```python
# Build forecast tables
sales_table = build_sales_forecast(sales_revenue, volumes, prices)
costs_table = build_costs_forecast(cogs, unit_costs)
inventory_table = build_inventory_forecast(...)
ar_table = build_ar_forecast(...)
ap_table = build_ap_forecast(...)
depreciation_table = build_depreciation_forecast(...)
fixed_assets_table = build_fixed_assets_forecast(capex)
```

### **Output - Intermediate Tables** (4 periods × 30 fields):
```python
{
    # Sales forecasts
    'ft_sales_revenue': [433890M, 455234M, 477123M, 499876M],
    'ft_sales_volume': [4338900, 4552340, 4771230, 4998760],
    'ft_selling_price': [100.0, 100.0, 100.0, 100.0],
    
    # Costs forecasts
    'ft_cogs': [245670M, 257890M, 270345M, 283567M],
    'ft_unit_cost': [56.6, 56.6, 56.7, 56.7],
    'ft_overhead': [56789M, 59567M, 62456M, 65478M],
    'ft_payroll': [28394.5M, 29783.5M, 31228M, 32739M],
    
    # Inventory
    'ft_inventory_beginning': [4946M, 5193M, 5453M, 5724M],
    'ft_inventory_ending': [5193M, 5453M, 5724M, 6010M],
    'ft_inventory_change': [247M, 260M, 271M, 286M],
    
    # Accounts Receivable
    'ft_ar_beginning': [28184M, 29593M, 31073M, 32597M],
    'ft_ar_ending': [29593M, 31073M, 32597M, 34188M],
    'ft_ar_change': [1409M, 1480M, 1524M, 1591M],
    'ft_collections': [404297M, 424161M, 445526M, 467279M],
    
    # Accounts Payable
    'ft_ap_beginning': [64115M, 67321M, 70687M, 74171M],
    'ft_ap_ending': [67321M, 70687M, 74171M, 77780M],
    'ft_ap_change': [3206M, 3366M, 3484M, 3609M],
    'ft_payments': [242464M, 254524M, 266861M, 279958M],
    
    # Depreciation
    'ft_depreciation': [11200M, 11648M, 12117M, 12609M],
    
    # Fixed Assets
    'ft_gross_fa_beginning': [114259M, 126149M, 138605M, 151683M],
    'ft_capex': [11890M, 12456M, 13078M, 13723M],
    'ft_gross_fa_ending': [126149M, 138605M, 151683M, 165406M],
    'ft_accumulated_dep': [72142M, 83790M, 95907M, 108516M],
    'ft_net_fa': [54007M, 54815M, 55776M, 56890M]
}
```

**Output Shape**: `(4 periods, ~30 forecast fields)`

---

## **STEP 11: Cash Budget (Vélez-Pareja 2009)**

**File**: `financial_statements/cash_budget.py`
**Method**: `calculate_cash_budget()`

### **Input**: 
Intermediate tables (30 fields)

### **Process**:
```python
# MODULE 1: Operating Cash Budget
ncb_op = collections - payments - overhead - payroll
# = [404297M - 242464M - 56789M - 28395M] = 76649M

# MODULE 2: Investment Cash Budget
ncb_inv = -capex
# = -11890M

# MODULE 3: Financing - Short Term
deficit_st = -(prev_cash + ncb_op - min_cash_required)
st_debt_new = max(deficit_st, 0)

# MODULE 4: Financing - Long Term  
deficit_lt = -ncb_inv
lt_debt_new = max(deficit_lt, 0) * leverage_ratio

# MODULE 5: Total Net Cash Budget
ncb = ncb_op + ncb_inv + financing
```

### **Output - Cash Budget** (4 periods × 25 fields):
```python
{
    # Operating
    'cb_collections': [404297M, 424161M, 445526M, 467279M],
    'cb_payments': [242464M, 254524M, 266861M, 279958M],
    'cb_overhead': [56789M, 59567M, 62456M, 65478M],
    'cb_payroll': [28395M, 29784M, 31228M, 32739M],
    'cb_ncb_operating': [76649M, 80286M, 84981M, 89104M],
    
    # Investment
    'cb_capex': [11890M, 12456M, 13078M, 13723M],
    'cb_ncb_investment': [-11890M, -12456M, -13078M, -13723M],
    
    # Financing - Short Term
    'cb_st_debt_payment': [5000M, 5250M, 5513M, 5788M],
    'cb_st_debt_new': [0M, 0M, 0M, 0M],  # No deficit
    'cb_st_debt_change': [-5000M, -5250M, -5513M, -5788M],
    
    # Financing - Long Term
    'cb_lt_debt_payment': [8000M, 8400M, 8820M, 9261M],
    'cb_lt_debt_new': [5945M, 6228M, 6539M, 6862M],  # 50% of capex
    'cb_lt_debt_change': [-2055M, -2172M, -2281M, -2399M],
    
    # Dividends
    'cb_dividends': [15000M, 15750M, 16538M, 17364M],
    
    # Total
    'cb_ncb': [42704M, 47158M, 52031M, 55344M],
    'cb_cash_beginning': [23646M, 66350M, 113508M, 165539M],
    'cb_cash_ending': [66350M, 113508M, 165539M, 220883M]
}
```

**Output Shape**: `(4 periods, ~25 cash budget fields)`

---

## **STEP 12: Income Statement**

**File**: `financial_statements/income_statement.py`
**Method**: `calculate_income_statement()`

### **Input**: 
- Intermediate tables (30 fields)
- Cash budget for debt levels (25 fields)

### **Process**:
```python
# Revenue
revenue = ft_sales_revenue

# Operating Costs
cogs = ft_cogs
gross_profit = revenue - cogs
operating_expenses = ft_overhead + ft_payroll
ebit = gross_profit - operating_expenses - depreciation

# Financial Costs (uses PREVIOUS period debt!)
interest_expense = (prev_st_debt * kd_st + prev_lt_debt * kd_lt)

# Taxes
ebt = ebit - interest_expense
taxes = ebt * tax_rate
net_income = ebt - taxes

# Components
noplat = ebit * (1 - tax_rate)
```

### **Output - Income Statement** (4 periods × 15 fields):
```python
{
    # Revenue
    'is_sales_revenue': [433890M, 455234M, 477123M, 499876M],
    
    # Operating
    'is_cogs': [245670M, 257890M, 270345M, 283567M],
    'is_gross_profit': [188220M, 197344M, 206778M, 216309M],
    'is_overhead': [56789M, 59567M, 62456M, 65478M],
    'is_payroll': [28395M, 29784M, 31228M, 32739M],
    'is_depreciation': [11200M, 11648M, 12117M, 12609M],
    'is_ebit': [91836M, 96345M, 100977M, 105483M],
    
    # Financial
    'is_interest_st': [1266M, 938M, 656M, 351M],    # Uses prev debt
    'is_interest_lt': [5940M, 5642M, 5322M, 4978M], # Uses prev debt
    'is_interest_total': [7206M, 6580M, 5978M, 5329M],
    
    # Taxes
    'is_ebt': [84630M, 89765M, 94999M, 100154M],
    'is_taxes': [29621M, 31418M, 33250M, 35054M],
    'is_net_income': [55009M, 58347M, 61749M, 65100M],
    
    # Components
    'is_noplat': [59694M, 62624M, 65635M, 68564M]
}
```

**Output Shape**: `(4 periods, ~15 income statement fields)`

---

## **STEP 13: Balance Sheet**

**File**: `financial_statements/balance_sheet.py`
**Method**: `construct_balance_sheet()`

### **Input**: 
- Intermediate tables (30 fields)
- Cash budget (25 fields)  
- Income statement (15 fields)

### **Process**:
```python
# Assets
cash = cb_cash_ending
ar = ft_ar_ending
inventory = ft_inventory_ending
current_assets = cash + ar + inventory

fixed_assets = ft_net_fa
total_assets = current_assets + fixed_assets

# Liabilities
ap = ft_ap_ending
st_debt = prev_st_debt + cb_st_debt_change
current_liabilities = ap + st_debt

lt_debt = prev_lt_debt + cb_lt_debt_change
total_liabilities = current_liabilities + lt_debt

# Equity
retained_earnings = prev_re + is_net_income - cb_dividends
total_equity = initial_equity + retained_earnings

# Verify identity
assert abs(total_assets - (total_liabilities + total_equity)) < 1e-6
```

### **Output - Balance Sheet** (4 periods × 20 fields):
```python
{
    # Current Assets
    'bs_cash': [66350M, 113508M, 165539M, 220883M],
    'bs_accounts_receivable': [29593M, 31073M, 32597M, 34188M],
    'bs_inventory': [5193M, 5453M, 5724M, 6010M],
    'bs_current_assets': [101136M, 150034M, 203860M, 261081M],
    
    # Fixed Assets
    'bs_gross_fixed_assets': [126149M, 138605M, 151683M, 165406M],
    'bs_accumulated_depreciation': [72142M, 83790M, 95907M, 108516M],
    'bs_net_fixed_assets': [54007M, 54815M, 55776M, 56890M],
    
    # Total Assets
    'bs_total_assets': [155143M, 204849M, 259636M, 317971M],
    
    # Current Liabilities
    'bs_accounts_payable': [67321M, 70687M, 74171M, 77780M],
    'bs_short_term_debt': [16110M, 10860M, 5347M, 0M],
    'bs_current_liabilities': [83431M, 81547M, 79518M, 77780M],
    
    # Long Term Debt
    'bs_long_term_debt': [96904M, 94732M, 92451M, 90052M],
    
    # Total Liabilities
    'bs_total_liabilities': [180335M, 176279M, 171969M, 167832M],
    
    # Equity
    'bs_initial_equity': [50672M, 50672M, 50672M, 50672M],
    'bs_retained_earnings': [-75864M, -22102M, 15565M, 63467M],
    'bs_total_equity': [-25192M, 28570M, 66237M, 114139M],
    
    # Verification
    'bs_total_liabilities_and_equity': [155143M, 204849M, 259636M, 317971M],
    'bs_identity_check': [0.0, 0.0, 0.0, 0.0]  # Perfect!
}
```

**Output Shape**: `(4 periods, ~20 balance sheet fields)`

---

## **STEP 14: Cash Flow Statement**

**File**: `core/cash_flow.py`
**Method**: `calculate_all_cash_flows()`

### **Input**: 
- Income statement (15 fields)
- Balance sheet changes (20 fields)
- Cash budget (25 fields)

### **Process**:
```python
# Free Cash Flow to Firm (FCF)
fcf = noplat + depreciation - capex - change_nwc

# Cash Flow to Equity (CFE)
cfe = net_income + depreciation - capex - change_nwc - debt_payment + new_debt

# Cash Flow to Debt (CFD)
cfd = interest + debt_payment - new_debt

# Tax Shields (TS)
ts = interest * tax_rate

# Capital Cash Flow (CCF)
ccf = fcf + ts

# Verify: CCF = CFE + CFD
assert abs(ccf - (cfe + cfd)) < 1e-6
```

### **Output - Cash Flows** (4 periods × 15 fields):
```python
{
    # Components
    'cf_noplat': [59694M, 62624M, 65635M, 68564M],
    'cf_depreciation': [11200M, 11648M, 12117M, 12609M],
    'cf_capex': [11890M, 12456M, 13078M, 13723M],
    'cf_change_nwc': [-1656M, -1740M, -1825M, -1914M],
    
    # Debt movements
    'cf_debt_payment': [13000M, 13650M, 14333M, 15049M],
    'cf_new_debt': [5945M, 6228M, 6539M, 6862M],
    'cf_net_debt': [-7055M, -7422M, -7794M, -8187M],
    
    # Cash flows
    'cf_fcf': [60660M, 63556M, 66499M, 69364M],
    'cf_cfe': [42704M, 47158M, 52031M, 55344M],  # = Cash change!
    'cf_cfd': [20478M, 19964M, 19446M, 18891M],
    'cf_ts': [2522M, 2303M, 2092M, 1865M],
    'cf_ccf': [63182M, 65859M, 68591M, 71229M],
    
    # Verification
    'cf_identity_check': [0.0, 0.0, 0.0, 0.0],  # CCF = CFE + CFD ✓
    'cf_cash_check': [0.0, 0.0, 0.0, 0.0]       # CFE = ΔCash ✓
}
```

**Output Shape**: `(4 periods, ~15 cash flow fields)`

---

## **STEP 15: Valuation (Optional)**

**File**: `core/valuation.py`
**Method**: `valuation_apv()`, `valuation_ccf()`, etc.

### **Input**: 
Cash flows (15 fields)

### **Process**:
```python
# APV Method
pv_fcf = sum([fcf[t] / (1 + ku)^t for t in 1..4])
pv_ts = sum([ts[t] / (1 + ku)^t for t in 1..4])  # or kd
terminal_value = fcf[4] * (1 + g) / (ku - g) / (1 + ku)^4
firm_value = pv_fcf + pv_ts + terminal_value
equity_value = firm_value - debt

# Verify with other methods (CCF, WACC, CFE)
```

### **Output - Valuation** (single row, multiple metrics):
```python
{
    # APV Method
    'val_pv_fcf': [234567M],
    'val_pv_ts': [8234M],
    'val_terminal_value': [856234M],
    'val_firm_value_apv': [1099035M],
    'val_equity_value_apv': [992131M],
    
    # CCF Method
    'val_firm_value_ccf': [1099035M],  # Should match APV
    
    # WACC Method
    'val_firm_value_wacc': [1099035M],  # Should match
    
    # CFE Method
    'val_equity_value_cfe': [992131M],  # Should match APV equity
    
    # Verification
    'val_methods_match': True,
    'val_max_difference': 0.0001  # All methods agree!
}
```

**Output Shape**: `(1 row, ~15 valuation metrics)`

---

## **STEP 16: Final Assembly & Export**

**File**: `forecaster_integration.py`
**Method**: `export_results()`

### **Input**: 
All results from Steps 1-15

### **Process**:
```python
# Combine all dataframes
final_results = pd.concat([
    ml_predictions,        # 5 columns
    intermediate_tables,   # 30 columns
    cash_budget,          # 25 columns
    income_statement,     # 15 columns
    balance_sheet,        # 20 columns
    cash_flows           # 15 columns
], axis=1)

# Export to Excel
with pd.ExcelWriter('results.xlsx') as writer:
    ml_predictions.to_excel(writer, 'ML Predictions')
    final_results.to_excel(writer, 'Financial Statements')
    cash_flows.to_excel(writer, 'Cash Flows')
    validation.to_excel(writer, 'Validation')
    valuation.to_excel(writer, 'Valuation')
```

### **Output - Excel File**:
```
results.xlsx
├─ Sheet 1: ML Predictions (4 periods × 5 cols)
├─ Sheet 2: Financial Statements (4 periods × 110 cols)
├─ Sheet 3: Cash Flows (4 periods × 15 cols)
├─ Sheet 4: Validation (identity checks)
└─ Sheet 5: Valuation (firm & equity value)
```

---

## 📊 **SUMMARY TABLE**

| Step | Input Features | Output Features | Shape Change |
|------|---------------|-----------------|--------------|
| 1. Yahoo Finance | ticker | 20 raw columns | (20, 20) |
| 2. Column Mapping | 20 columns | 10 cleaned | (20, 10) |
| 3. Feature Engineering | 10 columns | 72 features + 5 targets | (8, 77) |
| 4. Sequences | (8, 72) | (68, 12, 72) 3D tensor | Add time dim |
| 5. Scaling | (68, 12, 72) | Same, normalized | No change |
| 6. Training | (54, 12, 72) | 85K parameters | Model |
| 7. Prediction | (1, 12, 72) | 5 predictions | (1, 5) |
| 8. Rolling Forecast | Loop 4× | 4 periods × 5 values | (4, 5) |
| 9. ForecastInputs | 5 values | 8 fields | (4, 8) |
| 10. Intermediate Tables | 8 fields | 30 forecast fields | (4, 30) |
| 11. Cash Budget | 30 fields | 25 CB fields | (4, 25) |
| 12. Income Statement | 30+25 fields | 15 IS fields | (4, 15) |
| 13. Balance Sheet | 30+25+15 | 20 BS fields | (4, 20) |
| 14. Cash Flows | 15+20 | 15 CF fields | (4, 15) |
| 15. Valuation | 15 CF | 15 val metrics | (1, 15) |
| 16. Export | All | Excel file | 5 sheets |

---

## 🎯 **KEY INSIGHTS**

1. **Feature Explosion**: 10 → 72 features through engineering
2. **Time Dimension**: Flat data → 3D sequences for LSTM
3. **ML Compression**: 72 features → 5 predictions (learns patterns)
4. **Accounting Expansion**: 5 predictions → 110+ statement fields
5. **Validation**: Every step verifies consistency

The pipeline transforms **raw financial data** into **complete, validated forecasts** with guaranteed accounting consistency! 🚀
