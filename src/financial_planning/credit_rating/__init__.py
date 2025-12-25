"""
Credit Rating Module
====================

Complete credit rating system with fraud detection.

Quick Start:
------------
from credit_rating import rate_company

# From stock ticker
result = rate_company("GM")

# From PDF data
result = rate_company(pdf_data={
    'revenue': 250709,
    'net_income': -476,
    'total_assets': 2301221,
    ...
}, name="Evergrande")

# Access results
print(result.rating)        # 'CCC'
print(result.confidence)    # 0.72
print(result.red_flags)     # ['NEGATIVE WORKING CAPITAL', ...]
print(result.fraud_risk)    # 'CRITICAL'

Setup (One-time):
-----------------
1. Fetch training data:
   python fetch_training_data.py
   
2. Train and save model:
   python train_and_save_model.py

Files:
------
- ordinal_lr.py           : Ordinal Logistic Regression model
- fraud_detector.py       : Fraud detection (Red Flags, Altman Z, Beneish M)
- training_data.py        : 450+ companies with S&P ratings
- trainer.py              : Training pipeline with FMP API
- credit_rating_system.py : Simple user interface
- fetch_training_data.py  : Script to fetch data from FMP
- train_and_save_model.py : Script to train and save model

Author: Lihao Xiao
"""

# Core components
from .ordinal_lr import OrdinalLogisticRegression
from .fraud_detector import FraudDetector, AltmanZScore
from .training_data import LARGE_TRAINING_COMPANIES

# Training pipeline
from .trainer import (
    CreditRatingTrainer,
    FMPCreditRatingFetcher,
    FEATURE_NAMES,
    RATING_NAMES,
    RATING_MAP
)

# Simple interface
from .credit_rating_system import (
    CreditRatingSystem,
    CreditRatingResult,
    rate_company,
    detect_fraud
)

__all__ = [
    # Simple interface (most users need only these)
    'rate_company',
    'detect_fraud',
    'CreditRatingResult',
    
    # Full system
    'CreditRatingSystem',
    'CreditRatingTrainer',
    
    # Components
    'OrdinalLogisticRegression',
    'FraudDetector',
    'AltmanZScore',
    'FMPCreditRatingFetcher',
    
    # Data
    'LARGE_TRAINING_COMPANIES',
    'FEATURE_NAMES',
    'RATING_NAMES',
    'RATING_MAP'
]

__version__ = '1.0.0'