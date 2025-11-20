# ML-CESA
2026 Machine Learning Center of Excellence Summer Associate – Time Series &amp; Reinforcement Learning Internship Project  
Participant: Lihao Xiao: a Ph.D. candidate in Statistics at the University of California, Santa Barbara. My research focuses on multi-task deep learning for large-scale time-series forecasting, combining transformer architectures, soft clustering, and similarity-based ensembles. I develop scalable models that bridge statistical theory and modern machine learning, with applications in finance, healthcare, and environmental systems. Beyond research, I’m passionate about quantitative modeling, reinforcement learning, and algorithmic trading, aiming to design intelligent systems that learn from complex, real-world data.

# Financial Planning and Valuation System

A comprehensive financial planning and valuation system based on academic research by Vélez-Pareja et al., implementing consistent financial statement forecasting without plugs or circularity.

## Features

- **No Plugs**: Consistent financial statements using double-entry principles
- **No Circularity**: Analytical solutions to circularity problems
- **Complete Cash Flow Analysis**: FCF, CFE, CCF calculations
- **Multiple Valuation Methods**: APV, WACC, CCF approaches
- **Tax Shield Management**: Flexible discount rate assumptions (Ku, Kd)
- **Comprehensive Cash Budget**: 5-module cash budget system

## Core Concepts

Based on the research papers:
1. "Constructing Consistent Financial Planning Models for Valuation" (Vélez-Pareja, 2009)
2. "Analytical Solution to the Circularity Problem" (Mejía-Peláez & Vélez-Pareja, 2011)
3. "Forecasting Financial Statements with No Plugs and No Circularity" (Vélez-Pareja, 2007)

## Installation
```bash
pip install -r requirements.txt
python setup.py install
```

## Quick Start
```python
from src.core import FinancialModel
from src.financial_statements import CashBudget, IncomeStatement, BalanceSheet

# Initialize model
model = FinancialModel(
    initial_investment=45.0,
    forecast_years=4,
    tax_rate=0.35
)

# Build financial statements
model.build_statements()

# Calculate valuation
firm_value = model.calculate_value(method='CCF', discount_rate_ts='Ku')

print(f"Firm Value: {firm_value}")
```
