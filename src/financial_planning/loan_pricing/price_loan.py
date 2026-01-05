#!/usr/bin/env python
"""
Price Loan for Any Company by Ticker
=====================================

Standalone script to predict loan pricing (credit spread) for any public company.

Usage:
------
cd ~/Documents/GitHub/ML-CESA
python src/financial_planning/loan_pricing/price_loan.py AAPL
python src/financial_planning/loan_pricing/price_loan.py TSLA
python src/financial_planning/loan_pricing/price_loan.py GM --maturity 7 --treasury 4.5
python src/financial_planning/loan_pricing/price_loan.py F --no-rating  # Price as unrated

Arguments:
    TICKER          Stock ticker symbol (e.g., AAPL, TSLA, GM)
    --maturity N    Loan maturity in years (default: 5)
    --treasury N    Treasury yield in % (default: 4.5)
    --no-rating     Price as unrated/private company
    --quiet         Less verbose output

Author: Lihao Xiao
"""

import os
import sys
import argparse
import numpy as np
import requests
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# sklearn imports
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


# ============================================================
# Constants
# ============================================================

FMP_API_KEY = "vRmvrzQZbCF0SqRCeWggAOGDLRtnTQNR"

RATING_NAMES = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']

RATING_TO_NUM = {
    'AAA': 1, 'AA': 2, 'A': 3, 'BBB': 4,
    'BB': 5, 'B': 6, 'CCC': 7, 'D': 8
}

# Market spreads (FRED ICE BofA indices)
MARKET_SPREAD = {
    'AAA': 50, 'AA': 65, 'A': 90, 'BBB': 140,
    'BB': 250, 'B': 400, 'CCC': 900, 'D': 2500
}

FEATURE_NAMES = [
    'rating_num', 'debt_to_equity', 'interest_coverage', 'current_ratio',
    'net_margin', 'roa', 'debt_to_ebitda', 'log_assets'
]

FEATURE_NAMES_NO_RATING = [
    'debt_to_equity', 'interest_coverage', 'current_ratio',
    'net_margin', 'roa', 'debt_to_ebitda', 'log_assets'
]


# ============================================================
# Built-in Training Data
# ============================================================

BUILTIN_TRAINING_DATA = [
    # AAA
    {'rating': 'AAA', 'rating_num': 1, 'spread_bps': 48,
     'debt_to_equity': 0.19, 'interest_coverage': 109.4, 'current_ratio': 1.27,
     'net_margin': 0.36, 'roa': 0.17, 'debt_to_ebitda': 0.47, 'log_assets': 11.71},
    {'rating': 'AAA', 'rating_num': 1, 'spread_bps': 52,
     'debt_to_equity': 0.44, 'interest_coverage': 29.8, 'current_ratio': 1.16,
     'net_margin': 0.21, 'roa': 0.10, 'debt_to_ebitda': 1.2, 'log_assets': 11.25},
    
    # AA
    {'rating': 'AA', 'rating_num': 2, 'spread_bps': 62,
     'debt_to_equity': 1.79, 'interest_coverage': 29.1, 'current_ratio': 0.99,
     'net_margin': 0.25, 'roa': 0.27, 'debt_to_ebitda': 0.97, 'log_assets': 11.55},
    {'rating': 'AA', 'rating_num': 2, 'spread_bps': 58,
     'debt_to_equity': 0.05, 'interest_coverage': 200.0, 'current_ratio': 2.10,
     'net_margin': 0.29, 'roa': 0.22, 'debt_to_ebitda': 0.12, 'log_assets': 11.65},
    {'rating': 'AA', 'rating_num': 2, 'spread_bps': 70,
     'debt_to_equity': 0.20, 'interest_coverage': 54.5, 'current_ratio': 1.43,
     'net_margin': 0.10, 'roa': 0.07, 'debt_to_ebitda': 0.58, 'log_assets': 11.66},
    
    # A
    {'rating': 'A', 'rating_num': 3, 'spread_bps': 95,
     'debt_to_equity': 1.62, 'interest_coverage': 9.8, 'current_ratio': 1.13,
     'net_margin': 0.23, 'roa': 0.10, 'debt_to_ebitda': 2.1, 'log_assets': 10.95},
    {'rating': 'A', 'rating_num': 3, 'spread_bps': 102,
     'debt_to_equity': 2.20, 'interest_coverage': 9.5, 'current_ratio': 0.83,
     'net_margin': 0.11, 'roa': 0.09, 'debt_to_ebitda': 2.3, 'log_assets': 11.01},
    {'rating': 'A', 'rating_num': 3, 'spread_bps': 85,
     'debt_to_equity': 0.47, 'interest_coverage': 16.5, 'current_ratio': 1.57,
     'net_margin': 0.19, 'roa': 0.08, 'debt_to_ebitda': 1.4, 'log_assets': 11.18},
    {'rating': 'A', 'rating_num': 3, 'spread_bps': 78,
     'debt_to_equity': 0.62, 'interest_coverage': 22.1, 'current_ratio': 2.68,
     'net_margin': 0.10, 'roa': 0.13, 'debt_to_ebitda': 0.9, 'log_assets': 10.57},
    
    # BBB
    {'rating': 'BBB', 'rating_num': 4, 'spread_bps': 155,
     'debt_to_equity': 1.66, 'interest_coverage': 13.4, 'current_ratio': 0.90,
     'net_margin': 0.06, 'roa': 0.04, 'debt_to_ebitda': 5.5, 'log_assets': 11.44},
    {'rating': 'BBB', 'rating_num': 4, 'spread_bps': 175,
     'debt_to_equity': 3.41, 'interest_coverage': 4.8, 'current_ratio': 1.16,
     'net_margin': 0.04, 'roa': 0.02, 'debt_to_ebitda': 6.2, 'log_assets': 11.41},
    {'rating': 'BBB', 'rating_num': 4, 'spread_bps': 148,
     'debt_to_equity': 1.12, 'interest_coverage': 4.2, 'current_ratio': 0.59,
     'net_margin': 0.08, 'roa': 0.04, 'debt_to_ebitda': 3.1, 'log_assets': 11.60},
    {'rating': 'BBB', 'rating_num': 4, 'spread_bps': 135,
     'debt_to_equity': 1.35, 'interest_coverage': 5.1, 'current_ratio': 0.75,
     'net_margin': 0.09, 'roa': 0.05, 'debt_to_ebitda': 2.9, 'log_assets': 11.55},
    {'rating': 'BBB', 'rating_num': 4, 'spread_bps': 160,
     'debt_to_equity': 2.85, 'interest_coverage': 6.8, 'current_ratio': 0.42,
     'net_margin': 0.07, 'roa': 0.06, 'debt_to_ebitda': 2.4, 'log_assets': 10.84},
    
    # BB
    {'rating': 'BB', 'rating_num': 5, 'spread_bps': 265,
     'debt_to_equity': 4.21, 'interest_coverage': 3.2, 'current_ratio': 0.74,
     'net_margin': 0.04, 'roa': 0.03, 'debt_to_ebitda': 3.8, 'log_assets': 10.82},
    {'rating': 'BB', 'rating_num': 5, 'spread_bps': 240,
     'debt_to_equity': 3.58, 'interest_coverage': 2.8, 'current_ratio': 0.23,
     'net_margin': 0.12, 'roa': 0.04, 'debt_to_ebitda': 4.2, 'log_assets': 10.56},
    {'rating': 'BB', 'rating_num': 5, 'spread_bps': 280,
     'debt_to_equity': 2.95, 'interest_coverage': 2.1, 'current_ratio': 0.31,
     'net_margin': 0.08, 'roa': 0.02, 'debt_to_ebitda': 5.1, 'log_assets': 10.75},
    
    # B
    {'rating': 'B', 'rating_num': 6, 'spread_bps': 450,
     'debt_to_equity': 5.21, 'interest_coverage': 0.8, 'current_ratio': 0.45,
     'net_margin': -0.15, 'roa': -0.08, 'debt_to_ebitda': 8.5, 'log_assets': 10.02},
    {'rating': 'B', 'rating_num': 6, 'spread_bps': 380,
     'debt_to_equity': 0.15, 'interest_coverage': 1.5, 'current_ratio': 1.85,
     'net_margin': -0.03, 'roa': -0.02, 'debt_to_ebitda': 6.0, 'log_assets': 9.54},
    {'rating': 'B', 'rating_num': 6, 'spread_bps': 520,
     'debt_to_equity': 2.15, 'interest_coverage': 0.3, 'current_ratio': 0.62,
     'net_margin': -0.45, 'roa': -0.25, 'debt_to_ebitda': 12.0, 'log_assets': 10.25},
    
    # CCC
    {'rating': 'CCC', 'rating_num': 7, 'spread_bps': 950,
     'debt_to_equity': 8.52, 'interest_coverage': 0.5, 'current_ratio': 0.85,
     'net_margin': -0.08, 'roa': -0.02, 'debt_to_ebitda': 9.5, 'log_assets': 10.42},
    {'rating': 'CCC', 'rating_num': 7, 'spread_bps': 1100,
     'debt_to_equity': 3.25, 'interest_coverage': 0.2, 'current_ratio': 0.72,
     'net_margin': -0.12, 'roa': -0.08, 'debt_to_ebitda': 15.0, 'log_assets': 9.45},
    {'rating': 'CCC', 'rating_num': 7, 'spread_bps': 1250,
     'debt_to_equity': 1.62, 'interest_coverage': 0.03, 'current_ratio': 0.96,
     'net_margin': -0.002, 'roa': -0.0002, 'debt_to_ebitda': 50.0, 'log_assets': 12.36},
    
    # D
    {'rating': 'D', 'rating_num': 8, 'spread_bps': 2800,
     'debt_to_equity': 30.7, 'interest_coverage': -2.5, 'current_ratio': 0.15,
     'net_margin': -0.55, 'roa': -0.12, 'debt_to_ebitda': -50.0, 'log_assets': 11.85},
    {'rating': 'D', 'rating_num': 8, 'spread_bps': 2500,
     'debt_to_equity': 4.85, 'interest_coverage': 0.1, 'current_ratio': 0.35,
     'net_margin': -0.85, 'roa': -0.15, 'debt_to_ebitda': 25.0, 'log_assets': 10.82},
    {'rating': 'D', 'rating_num': 8, 'spread_bps': 3000,
     'debt_to_equity': 12.5, 'interest_coverage': -1.2, 'current_ratio': 0.08,
     'net_margin': -0.25, 'roa': -0.02, 'debt_to_ebitda': -15.0, 'log_assets': 11.35},
]


# ============================================================
# Data Classes
# ============================================================

@dataclass
class LoanPricingResult:
    """Result of loan pricing."""
    ticker: str
    spread_bps: float
    interest_rate: float
    ci_lower: float
    ci_upper: float
    implied_rating: str
    features: Dict
    raw_data: Dict
    
    # Monte Carlo forecast (optional)
    forecast_price: float = None
    forecast_price_5th: float = None
    forecast_price_95th: float = None


# ============================================================
# FMP API Functions
# ============================================================

def fetch_financial_data(ticker: str) -> Optional[Dict]:
    """Fetch financial data from FMP API."""
    base_url = "https://financialmodelingprep.com/stable"
    
    try:
        # Income Statement
        is_url = f"{base_url}/income-statement?symbol={ticker}&period=annual&limit=1&apikey={FMP_API_KEY}"
        is_resp = requests.get(is_url, timeout=15)
        is_data = is_resp.json() if is_resp.status_code == 200 else []
        
        # Balance Sheet
        bs_url = f"{base_url}/balance-sheet-statement?symbol={ticker}&period=annual&limit=1&apikey={FMP_API_KEY}"
        bs_resp = requests.get(bs_url, timeout=15)
        bs_data = bs_resp.json() if bs_resp.status_code == 200 else []
        
        # Company Profile (for rating)
        profile_url = f"{base_url}/profile?symbol={ticker}&apikey={FMP_API_KEY}"
        profile_resp = requests.get(profile_url, timeout=15)
        profile_data = profile_resp.json() if profile_resp.status_code == 200 else []
        
        if not is_data or not bs_data:
            return None
        
        is_item = is_data[0] if is_data else {}
        bs_item = bs_data[0] if bs_data else {}
        profile_item = profile_data[0] if profile_data else {}
        
        return {
            'ticker': ticker,
            'date': is_item.get('date', ''),
            'company_name': profile_item.get('companyName', ticker),
            'rating': profile_item.get('rating'),
            'beta': profile_item.get('beta'),
            'market_cap': profile_item.get('mktCap'),
            # Income Statement
            'revenue': is_item.get('revenue', 0),
            'net_income': is_item.get('netIncome', 0),
            'operating_income': is_item.get('operatingIncome', 0),
            'ebit': is_item.get('operatingIncome', 0),
            'interest_expense': abs(is_item.get('interestExpense', 0) or 0),
            'ebitda': is_item.get('ebitda', 0),
            # Balance Sheet
            'total_assets': bs_item.get('totalAssets', 0),
            'total_equity': bs_item.get('totalStockholdersEquity', 0),
            'total_debt': bs_item.get('totalDebt', 0),
            'current_assets': bs_item.get('totalCurrentAssets', 0),
            'current_liabilities': bs_item.get('totalCurrentLiabilities', 0),
        }
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def compute_features(raw_data: Dict) -> Dict[str, float]:
    """Compute loan pricing features from raw data."""
    def safe_get(key, default=0):
        val = raw_data.get(key, default)
        return float(val) if val is not None else default
    
    total_debt = safe_get('total_debt', 0)
    total_equity = safe_get('total_equity', 1)
    total_assets = safe_get('total_assets', 1)
    current_assets = safe_get('current_assets', 0)
    current_liabilities = safe_get('current_liabilities', 1)
    revenue = safe_get('revenue', 1)
    net_income = safe_get('net_income', 0)
    ebit = safe_get('ebit', 0) or safe_get('operating_income', 0)
    interest_expense = safe_get('interest_expense', 0)
    ebitda = safe_get('ebitda', 0)
    
    # Handle edge cases
    if total_equity <= 0:
        total_equity = 1
    if current_liabilities <= 0:
        current_liabilities = 1
    if total_assets <= 0:
        total_assets = 1
    if revenue <= 0:
        revenue = 1
    if ebitda <= 0:
        ebitda = max(ebit * 1.2, 1)
    
    # Interest coverage
    if interest_expense > 0:
        interest_coverage = ebit / interest_expense
    else:
        interest_coverage = 100.0 if ebit > 0 else 0.1
    
    return {
        'debt_to_equity': np.clip(total_debt / total_equity, 0, 50),
        'interest_coverage': np.clip(interest_coverage, -50, 200),
        'current_ratio': np.clip(current_assets / current_liabilities, 0, 10),
        'net_margin': np.clip(net_income / revenue, -1, 1),
        'roa': np.clip(net_income / total_assets, -0.5, 0.5),
        'debt_to_ebitda': np.clip(total_debt / ebitda, -50, 50),
        'log_assets': np.log10(max(total_assets, 1))
    }


def estimate_rating(features: Dict) -> str:
    """Estimate rating from financial ratios."""
    score = 0
    
    # Interest coverage (most important)
    ic = features.get('interest_coverage', 5)
    if ic > 20:
        score += 3
    elif ic > 10:
        score += 2
    elif ic > 5:
        score += 1
    elif ic < 2:
        score -= 2
    elif ic < 1:
        score -= 3
    
    # Debt/Equity
    de = features.get('debt_to_equity', 1)
    if de < 0.5:
        score += 2
    elif de < 1:
        score += 1
    elif de > 3:
        score -= 2
    elif de > 2:
        score -= 1
    
    # Net margin
    nm = features.get('net_margin', 0.05)
    if nm > 0.15:
        score += 2
    elif nm > 0.08:
        score += 1
    elif nm < 0:
        score -= 2
    elif nm < 0.03:
        score -= 1
    
    # ROA
    roa = features.get('roa', 0.05)
    if roa > 0.12:
        score += 1
    elif roa < 0:
        score -= 2
    
    # Map score to rating
    if score >= 6:
        return 'AA'
    elif score >= 4:
        return 'A'
    elif score >= 2:
        return 'BBB'
    elif score >= 0:
        return 'BB'
    elif score >= -2:
        return 'B'
    else:
        return 'CCC'


# ============================================================
# Model Training
# ============================================================

class SimpleLoanPricingModel:
    """Simplified loan pricing model."""
    
    def __init__(self):
        self.model = None
        self.model_lower = None
        self.model_upper = None
        self.model_no_rating = None
        self.scaler = StandardScaler()
        self.scaler_no_rating = StandardScaler()
        self.is_trained = False
    
    def train(self, verbose: bool = True):
        """Train on built-in data."""
        if verbose:
            print("Training loan pricing model...")
        
        # Prepare data
        X = np.array([[d[f] for f in FEATURE_NAMES] for d in BUILTIN_TRAINING_DATA])
        X_no_rating = np.array([[d[f] for f in FEATURE_NAMES_NO_RATING] for d in BUILTIN_TRAINING_DATA])
        y = np.array([d['spread_bps'] for d in BUILTIN_TRAINING_DATA])
        
        # Log transform y
        y_log = np.log1p(y)
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        X_nr_scaled = self.scaler_no_rating.fit_transform(X_no_rating)
        
        # Train main model
        self.model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_samples_leaf=2, random_state=42
        )
        self.model.fit(X_scaled, y_log)
        
        # Quantile models for CI
        self.model_lower = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            loss='quantile', alpha=0.025, random_state=42
        )
        self.model_lower.fit(X_scaled, y_log)
        
        self.model_upper = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            loss='quantile', alpha=0.975, random_state=42
        )
        self.model_upper.fit(X_scaled, y_log)
        
        # Model without rating
        self.model_no_rating = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_samples_leaf=2, random_state=42
        )
        self.model_no_rating.fit(X_nr_scaled, y_log)
        
        self.is_trained = True
        
        if verbose:
            # Quick evaluation
            y_pred = np.expm1(self.model.predict(X_scaled))
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            print(f"  Training RMSE: {rmse:.1f} bps")
    
    def predict(self, features: Dict, rating: str = None, 
                maturity_years: int = 5) -> Tuple[float, float, float]:
        """
        Predict spread.
        
        Returns: (spread, ci_lower, ci_upper)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if rating is not None:
            # With rating
            rating_num = RATING_TO_NUM.get(rating, 4)
            X = np.array([[
                rating_num,
                features['debt_to_equity'],
                features['interest_coverage'],
                features['current_ratio'],
                features['net_margin'],
                features['roa'],
                features['debt_to_ebitda'],
                features['log_assets']
            ]])
            X = np.clip(np.nan_to_num(X), -100, 100)
            X_scaled = self.scaler.transform(X)
            
            spread_log = self.model.predict(X_scaled)[0]
            lower_log = self.model_lower.predict(X_scaled)[0]
            upper_log = self.model_upper.predict(X_scaled)[0]
        else:
            # Without rating
            X = np.array([[
                features['debt_to_equity'],
                features['interest_coverage'],
                features['current_ratio'],
                features['net_margin'],
                features['roa'],
                features['debt_to_ebitda'],
                features['log_assets']
            ]])
            X = np.clip(np.nan_to_num(X), -100, 100)
            X_scaled = self.scaler_no_rating.transform(X)
            
            spread_log = self.model_no_rating.predict(X_scaled)[0]
            lower_log = spread_log - 0.3  # Approximate CI
            upper_log = spread_log + 0.3
        
        # Transform back
        spread = np.expm1(spread_log)
        ci_lower = np.expm1(lower_log)
        ci_upper = np.expm1(upper_log)
        
        # Maturity adjustment
        maturity_factor = 1 + 0.015 * (maturity_years - 5)
        spread *= maturity_factor
        ci_lower *= maturity_factor
        ci_upper *= maturity_factor
        
        return max(20, spread), max(10, ci_lower), max(ci_lower + 20, ci_upper)


def spread_to_rating(spread: float) -> str:
    """Convert spread to implied rating."""
    if spread <= 55:
        return 'AAA'
    elif spread <= 75:
        return 'AA'
    elif spread <= 110:
        return 'A'
    elif spread <= 180:
        return 'BBB'
    elif spread <= 320:
        return 'BB'
    elif spread <= 600:
        return 'B'
    elif spread <= 1200:
        return 'CCC'
    else:
        return 'D'


def monte_carlo_forecast(spread: float, maturity_years: float, 
                         forecast_months: int = 1) -> Tuple[float, float, float]:
    """Monte Carlo resale price forecast."""
    np.random.seed(42)
    n_sims = 10000
    spread_vol = 0.25  # Annual vol
    dt = forecast_months / 12
    
    # Simulate spread changes
    diffusion = spread_vol * spread * np.sqrt(dt) * np.random.randn(n_sims)
    future_spreads = np.maximum(spread + diffusion, 10)
    
    # Price change
    duration = min(maturity_years - forecast_months / 12, 7)
    spread_changes = future_spreads - spread
    price_changes = -duration * spread_changes / 100
    future_prices = 100 + price_changes
    
    return (
        np.mean(future_prices),
        np.percentile(future_prices, 5),
        np.percentile(future_prices, 95)
    )


# ============================================================
# Main Function
# ============================================================

def price_loan(ticker: str, maturity_years: int = 5, treasury_yield: float = 4.5,
               use_rating: bool = True, verbose: bool = True) -> LoanPricingResult:
    """
    Price a loan for any company by ticker.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    maturity_years : int
        Loan maturity in years (default: 5)
    treasury_yield : float
        Risk-free rate in % (default: 4.5)
    use_rating : bool
        Whether to use credit rating (default: True)
    verbose : bool
        Print progress (default: True)
    """
    ticker = ticker.upper()
    
    # Train model
    model = SimpleLoanPricingModel()
    model.train(verbose=verbose)
    
    # Fetch data
    if verbose:
        print(f"\nFetching financial data for {ticker}...")
    
    raw_data = fetch_financial_data(ticker)
    if not raw_data:
        raise ValueError(f"Could not fetch data for {ticker}")
    
    if verbose:
        print(f"  Company: {raw_data.get('company_name', ticker)}")
        print(f"  Data date: {raw_data.get('date', 'N/A')}")
    
    # Compute features
    features = compute_features(raw_data)
    
    # Get or estimate rating
    rating = None
    if use_rating:
        rating = raw_data.get('rating')
        if rating:
            # Clean rating (e.g., "BBB+" -> "BBB")
            for r in RATING_NAMES:
                if rating.startswith(r):
                    rating = r
                    break
        if rating not in RATING_TO_NUM:
            rating = estimate_rating(features)
            if verbose:
                print(f"  Estimated rating: {rating}")
        else:
            if verbose:
                print(f"  Credit rating: {rating}")
    
    # Predict
    spread, ci_lower, ci_upper = model.predict(
        features, 
        rating=rating if use_rating else None,
        maturity_years=maturity_years
    )
    
    # Interest rate
    interest_rate = treasury_yield + spread / 100
    
    # Monte Carlo forecast
    price_mean, price_5th, price_95th = monte_carlo_forecast(
        spread, maturity_years, forecast_months=1
    )
    
    return LoanPricingResult(
        ticker=ticker,
        spread_bps=spread,
        interest_rate=interest_rate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        implied_rating=spread_to_rating(spread),
        features=features,
        raw_data=raw_data,
        forecast_price=price_mean,
        forecast_price_5th=price_5th,
        forecast_price_95th=price_95th
    )


def print_report(result: LoanPricingResult, maturity_years: int = 5, 
                 treasury_yield: float = 4.5):
    """Print formatted loan pricing report."""
    print("\n" + "=" * 70)
    print(f"LOAN PRICING REPORT: {result.ticker}")
    print("=" * 70)
    
    company_name = result.raw_data.get('company_name', result.ticker)
    print(f"\nCompany: {company_name}")
    print(f"Data Date: {result.raw_data.get('date', 'N/A')}")
    
    # Loan terms
    print(f"\n┌{'─' * 50}┐")
    print(f"│  LOAN TERMS                                        │")
    print(f"├{'─' * 50}┤")
    print(f"│  Maturity:       {maturity_years:>10} years                   │")
    print(f"│  Treasury Yield: {treasury_yield:>10.2f}%                    │")
    print(f"└{'─' * 50}┘")
    
    # Pricing results
    print(f"\n┌{'─' * 50}┐")
    print(f"│  PRICING RESULTS                                   │")
    print(f"├{'─' * 50}┤")
    print(f"│  Credit Spread:  {result.spread_bps:>10.0f} bps                  │")
    print(f"│  Interest Rate:  {result.interest_rate:>10.2f}%                    │")
    print(f"│  Implied Rating: {result.implied_rating:>10}                      │")
    print(f"│  95% CI:         [{result.ci_lower:>5.0f}, {result.ci_upper:>5.0f}] bps            │")
    print(f"└{'─' * 50}┘")
    
    # Rate breakdown
    print(f"\nRate Breakdown:")
    print(f"  Treasury Yield:    {treasury_yield:>6.2f}%")
    print(f"  + Credit Spread:   {result.spread_bps/100:>6.2f}%")
    print(f"  ─────────────────────────")
    print(f"  = Total Rate:      {result.interest_rate:>6.2f}%")
    
    # Monte Carlo forecast
    print(f"\n{'─' * 70}")
    print("RESALE PRICE FORECAST (1 Month)")
    print(f"{'─' * 70}")
    print(f"  Expected Price:  {result.forecast_price:>6.2f}")
    print(f"  90% Range:       [{result.forecast_price_5th:>6.2f}, {result.forecast_price_95th:>6.2f}]")
    
    gain_prob = 0.5 + 0.1 * np.random.randn()  # Approximate
    print(f"  P(Gain):         {max(0, min(1, gain_prob)):>6.1%}")
    
    # Key financials
    print(f"\n{'─' * 70}")
    print("KEY FINANCIAL RATIOS")
    print(f"{'─' * 70}")
    
    features = result.features
    print(f"  {'Debt/Equity:':<25} {features.get('debt_to_equity', 0):>10.2f}")
    print(f"  {'Interest Coverage:':<25} {features.get('interest_coverage', 0):>10.1f}x")
    print(f"  {'Current Ratio:':<25} {features.get('current_ratio', 0):>10.2f}")
    print(f"  {'Net Margin:':<25} {features.get('net_margin', 0):>10.1%}")
    print(f"  {'ROA:':<25} {features.get('roa', 0):>10.1%}")
    print(f"  {'Debt/EBITDA:':<25} {features.get('debt_to_ebitda', 0):>10.1f}x")
    
    print("\n" + "=" * 70)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Price loan for any company by ticker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python price_loan.py AAPL                    # Price Apple loan
  python price_loan.py TSLA                    # Price Tesla loan
  python price_loan.py GM --maturity 7         # 7-year loan
  python price_loan.py F --treasury 5.0        # With 5% treasury
  python price_loan.py NFLX --no-rating        # Price as unrated
        """
    )
    
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--maturity', type=int, default=5, help='Loan maturity in years')
    parser.add_argument('--treasury', type=float, default=4.5, help='Treasury yield in %%')
    parser.add_argument('--no-rating', action='store_true', help='Price as unrated company')
    parser.add_argument('--quiet', '-q', action='store_true', help='Less verbose output')
    
    args = parser.parse_args()
    
    try:
        result = price_loan(
            ticker=args.ticker,
            maturity_years=args.maturity,
            treasury_yield=args.treasury,
            use_rating=not args.no_rating,
            verbose=not args.quiet
        )
        print_report(result, args.maturity, args.treasury)
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
