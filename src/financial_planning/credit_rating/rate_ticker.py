#!/usr/bin/env python
"""
Rate Any Company by Ticker Symbol
==================================

Standalone script to predict credit rating for any public company.

Usage:
------
cd ~/Documents/GitHub/ML-CESA
python src/financial_planning/credit_rating/rate_ticker.py AAPL
python src/financial_planning/credit_rating/rate_ticker.py TSLA
python src/financial_planning/credit_rating/rate_ticker.py GM --no-fraud

Arguments:
    TICKER      Stock ticker symbol (e.g., AAPL, TSLA, GM)
    --no-fraud  Skip fraud detection (faster)
    --quiet     Less verbose output

Author: Lihao Xiao
"""

import os
import sys
import argparse
import numpy as np
import requests
from typing import Dict, Optional

# Add current directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import local modules
from ordinal_lr import OrdinalLogisticRegression
from fraud_detector import FraudDetector, AltmanZScore


# ============================================================
# Constants
# ============================================================

RATING_NAMES = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']

FEATURE_NAMES = [
    'debt_to_equity',
    'interest_coverage',
    'current_ratio',
    'net_margin',
    'roa',
    'debt_to_ebitda',
    'log_assets'
]

# FMP API Key (free tier)
FMP_API_KEY = "vRmvrzQZbCF0SqRCeWggAOGDLRtnTQNR"


# ============================================================
# Built-in Training Data (26 companies)
# ============================================================

BUILTIN_TRAINING_DATA = [
    # AAA
    {'ticker': 'MSFT', 'name': 'Microsoft', 'rating_num': 0,
     'debt_to_equity': 0.19, 'interest_coverage': 109.4, 'current_ratio': 1.27,
     'net_margin': 0.36, 'roa': 0.17, 'debt_to_ebitda': 0.47, 'log_assets': 11.71},
    {'ticker': 'JNJ', 'name': 'Johnson & Johnson', 'rating_num': 0,
     'debt_to_equity': 0.44, 'interest_coverage': 29.8, 'current_ratio': 1.16,
     'net_margin': 0.21, 'roa': 0.10, 'debt_to_ebitda': 1.2, 'log_assets': 11.25},
    
    # AA
    {'ticker': 'AAPL', 'name': 'Apple', 'rating_num': 1,
     'debt_to_equity': 1.79, 'interest_coverage': 29.1, 'current_ratio': 0.99,
     'net_margin': 0.25, 'roa': 0.27, 'debt_to_ebitda': 0.97, 'log_assets': 11.55},
    {'ticker': 'GOOGL', 'name': 'Alphabet', 'rating_num': 1,
     'debt_to_equity': 0.05, 'interest_coverage': 200.0, 'current_ratio': 2.10,
     'net_margin': 0.29, 'roa': 0.22, 'debt_to_ebitda': 0.12, 'log_assets': 11.65},
    {'ticker': 'XOM', 'name': 'Exxon Mobil', 'rating_num': 1,
     'debt_to_equity': 0.20, 'interest_coverage': 54.5, 'current_ratio': 1.43,
     'net_margin': 0.10, 'roa': 0.07, 'debt_to_ebitda': 0.58, 'log_assets': 11.66},
    
    # A
    {'ticker': 'KO', 'name': 'Coca-Cola', 'rating_num': 2,
     'debt_to_equity': 1.62, 'interest_coverage': 9.8, 'current_ratio': 1.13,
     'net_margin': 0.23, 'roa': 0.10, 'debt_to_ebitda': 2.1, 'log_assets': 10.95},
    {'ticker': 'PEP', 'name': 'PepsiCo', 'rating_num': 2,
     'debt_to_equity': 2.20, 'interest_coverage': 9.5, 'current_ratio': 0.83,
     'net_margin': 0.11, 'roa': 0.09, 'debt_to_ebitda': 2.3, 'log_assets': 11.01},
    {'ticker': 'INTC', 'name': 'Intel', 'rating_num': 2,
     'debt_to_equity': 0.47, 'interest_coverage': 16.5, 'current_ratio': 1.57,
     'net_margin': 0.19, 'roa': 0.08, 'debt_to_ebitda': 1.4, 'log_assets': 11.18},
    {'ticker': 'NKE', 'name': 'Nike', 'rating_num': 2,
     'debt_to_equity': 0.62, 'interest_coverage': 22.1, 'current_ratio': 2.68,
     'net_margin': 0.10, 'roa': 0.13, 'debt_to_ebitda': 0.9, 'log_assets': 10.57},
    
    # BBB
    {'ticker': 'GM', 'name': 'General Motors', 'rating_num': 3,
     'debt_to_equity': 1.66, 'interest_coverage': 13.4, 'current_ratio': 0.90,
     'net_margin': 0.06, 'roa': 0.04, 'debt_to_ebitda': 5.5, 'log_assets': 11.44},
    {'ticker': 'F', 'name': 'Ford', 'rating_num': 3,
     'debt_to_equity': 3.41, 'interest_coverage': 4.8, 'current_ratio': 1.16,
     'net_margin': 0.04, 'roa': 0.02, 'debt_to_ebitda': 6.2, 'log_assets': 11.41},
    {'ticker': 'T', 'name': 'AT&T', 'rating_num': 3,
     'debt_to_equity': 1.12, 'interest_coverage': 4.2, 'current_ratio': 0.59,
     'net_margin': 0.08, 'roa': 0.04, 'debt_to_ebitda': 3.1, 'log_assets': 11.60},
    {'ticker': 'VZ', 'name': 'Verizon', 'rating_num': 3,
     'debt_to_equity': 1.35, 'interest_coverage': 5.1, 'current_ratio': 0.75,
     'net_margin': 0.09, 'roa': 0.05, 'debt_to_ebitda': 2.9, 'log_assets': 11.55},
    {'ticker': 'DAL', 'name': 'Delta Airlines', 'rating_num': 3,
     'debt_to_equity': 2.85, 'interest_coverage': 6.8, 'current_ratio': 0.42,
     'net_margin': 0.07, 'roa': 0.06, 'debt_to_ebitda': 2.4, 'log_assets': 10.84},
    
    # BB
    {'ticker': 'UAL', 'name': 'United Airlines', 'rating_num': 4,
     'debt_to_equity': 4.21, 'interest_coverage': 3.2, 'current_ratio': 0.74,
     'net_margin': 0.04, 'roa': 0.03, 'debt_to_ebitda': 3.8, 'log_assets': 10.82},
    {'ticker': 'RCL', 'name': 'Royal Caribbean', 'rating_num': 4,
     'debt_to_equity': 3.58, 'interest_coverage': 2.8, 'current_ratio': 0.23,
     'net_margin': 0.12, 'roa': 0.04, 'debt_to_ebitda': 4.2, 'log_assets': 10.56},
    {'ticker': 'CCL', 'name': 'Carnival', 'rating_num': 4,
     'debt_to_equity': 2.95, 'interest_coverage': 2.1, 'current_ratio': 0.31,
     'net_margin': 0.08, 'roa': 0.02, 'debt_to_ebitda': 5.1, 'log_assets': 10.75},
    
    # B
    {'ticker': 'AMC', 'name': 'AMC Entertainment', 'rating_num': 5,
     'debt_to_equity': 5.21, 'interest_coverage': 0.8, 'current_ratio': 0.45,
     'net_margin': -0.15, 'roa': -0.08, 'debt_to_ebitda': 8.5, 'log_assets': 10.02},
    {'ticker': 'GME', 'name': 'GameStop', 'rating_num': 5,
     'debt_to_equity': 0.15, 'interest_coverage': 1.5, 'current_ratio': 1.85,
     'net_margin': -0.03, 'roa': -0.02, 'debt_to_ebitda': 6.0, 'log_assets': 9.54},
    {'ticker': 'WEWORK', 'name': 'WeWork', 'rating_num': 5,
     'debt_to_equity': 2.15, 'interest_coverage': 0.3, 'current_ratio': 0.62,
     'net_margin': -0.45, 'roa': -0.25, 'debt_to_ebitda': 12.0, 'log_assets': 10.25},
    
    # CCC
    {'ticker': 'HTZ', 'name': 'Hertz', 'rating_num': 6,
     'debt_to_equity': 8.52, 'interest_coverage': 0.5, 'current_ratio': 0.85,
     'net_margin': -0.08, 'roa': -0.02, 'debt_to_ebitda': 9.5, 'log_assets': 10.42},
    {'ticker': 'REV', 'name': 'Revlon', 'rating_num': 6,
     'debt_to_equity': 3.25, 'interest_coverage': 0.2, 'current_ratio': 0.72,
     'net_margin': -0.12, 'roa': -0.08, 'debt_to_ebitda': 15.0, 'log_assets': 9.45},
    {'ticker': 'EGRNF', 'name': 'Evergrande', 'rating_num': 6,
     'debt_to_equity': 1.62, 'interest_coverage': 0.03, 'current_ratio': 0.96,
     'net_margin': -0.002, 'roa': -0.0002, 'debt_to_ebitda': 50.0, 'log_assets': 12.36},
    
    # D
    {'ticker': 'LEH', 'name': 'Lehman Brothers', 'rating_num': 7,
     'debt_to_equity': 30.7, 'interest_coverage': -2.5, 'current_ratio': 0.15,
     'net_margin': -0.55, 'roa': -0.12, 'debt_to_ebitda': -50.0, 'log_assets': 11.85},
    {'ticker': 'ENRN', 'name': 'Enron', 'rating_num': 7,
     'debt_to_equity': 4.85, 'interest_coverage': 0.1, 'current_ratio': 0.35,
     'net_margin': -0.85, 'roa': -0.15, 'debt_to_ebitda': 25.0, 'log_assets': 10.82},
    {'ticker': 'SIVB', 'name': 'SVB Financial', 'rating_num': 7,
     'debt_to_equity': 12.5, 'interest_coverage': -1.2, 'current_ratio': 0.08,
     'net_margin': -0.25, 'roa': -0.02, 'debt_to_ebitda': -15.0, 'log_assets': 11.35},
]


# ============================================================
# FMP API Functions
# ============================================================

def fetch_financial_data(ticker: str, api_key: str = FMP_API_KEY) -> Optional[Dict]:
    """Fetch financial data from FMP API"""
    base_url = "https://financialmodelingprep.com/stable"
    
    try:
        # Income Statement
        is_url = f"{base_url}/income-statement?symbol={ticker}&period=annual&limit=1&apikey={api_key}"
        is_resp = requests.get(is_url, timeout=15)
        is_data = is_resp.json() if is_resp.status_code == 200 else []
        
        # Balance Sheet
        bs_url = f"{base_url}/balance-sheet-statement?symbol={ticker}&period=annual&limit=1&apikey={api_key}"
        bs_resp = requests.get(bs_url, timeout=15)
        bs_data = bs_resp.json() if bs_resp.status_code == 200 else []
        
        # Cash Flow
        cf_url = f"{base_url}/cash-flow-statement?symbol={ticker}&period=annual&limit=1&apikey={api_key}"
        cf_resp = requests.get(cf_url, timeout=15)
        cf_data = cf_resp.json() if cf_resp.status_code == 200 else []
        
        if not is_data or not bs_data:
            return None
        
        is_item = is_data[0] if is_data else {}
        bs_item = bs_data[0] if bs_data else {}
        cf_item = cf_data[0] if cf_data else {}
        
        return {
            'ticker': ticker,
            'date': is_item.get('date', ''),
            # Income Statement
            'revenue': is_item.get('revenue', 0),
            'net_income': is_item.get('netIncome', 0),
            'operating_income': is_item.get('operatingIncome', 0),
            'ebit': is_item.get('operatingIncome', 0),
            'interest_expense': abs(is_item.get('interestExpense', 0) or 0),
            'ebitda': is_item.get('ebitda', 0),
            'gross_profit': is_item.get('grossProfit', 0),
            # Balance Sheet
            'total_assets': bs_item.get('totalAssets', 0),
            'total_liabilities': bs_item.get('totalLiabilities', 0),
            'total_equity': bs_item.get('totalStockholdersEquity', 0),
            'total_debt': bs_item.get('totalDebt', 0),
            'current_assets': bs_item.get('totalCurrentAssets', 0),
            'current_liabilities': bs_item.get('totalCurrentLiabilities', 0),
            'cash': bs_item.get('cashAndCashEquivalents', 0),
            'inventory': bs_item.get('inventory', 0),
            'retained_earnings': bs_item.get('retainedEarnings', 0),
            # Cash Flow
            'operating_cash_flow': cf_item.get('operatingCashFlow', 0),
            'depreciation': cf_item.get('depreciationAndAmortization', 0),
        }
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def compute_features(raw_data: Dict) -> Dict[str, float]:
    """Compute credit rating features from raw financial data"""
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
    
    # Calculate EBITDA if not available
    if ebitda <= 0:
        depreciation = safe_get('depreciation', 0)
        ebitda = ebit + depreciation if ebit > 0 else revenue * 0.15
    
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
        ebitda = 1
    
    # Interest coverage
    if interest_expense > 0:
        interest_coverage = ebit / interest_expense
    else:
        interest_coverage = 100.0 if ebit > 0 else 0.1
    
    interest_coverage = np.clip(interest_coverage, -50, 200)
    
    return {
        'debt_to_equity': np.clip(total_debt / total_equity, 0, 50),
        'interest_coverage': interest_coverage,
        'current_ratio': np.clip(current_assets / current_liabilities, 0, 10),
        'net_margin': np.clip(net_income / revenue, -1, 1),
        'roa': np.clip(net_income / total_assets, -0.5, 0.5),
        'debt_to_ebitda': np.clip(total_debt / ebitda, -50, 50),
        'log_assets': np.log10(max(total_assets, 1))
    }


# ============================================================
# Model Training
# ============================================================

def train_model(verbose: bool = True) -> OrdinalLogisticRegression:
    """Train the credit rating model using built-in data"""
    if verbose:
        print("Training credit rating model...")
    
    # Prepare training data
    X = np.array([[d[f] for f in FEATURE_NAMES] for d in BUILTIN_TRAINING_DATA])
    y = np.array([d['rating_num'] for d in BUILTIN_TRAINING_DATA])
    
    # Train model
    model = OrdinalLogisticRegression(
        n_classes=8,
        alpha=1.0,
        balance=True,
        regularization=0.01
    )
    model.fit(X, y)
    
    if verbose:
        scores = model.score(X, y)
        print(f"  Training accuracy: {scores['accuracy']:.1%}")
        print(f"  MAE: {scores['mae']:.2f} notches")
    
    return model


# ============================================================
# Rating Functions
# ============================================================

def rate_ticker(ticker: str, check_fraud: bool = True, verbose: bool = True) -> Dict:
    """
    Rate a company by ticker symbol
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'TSLA')
    check_fraud : bool
        Whether to run fraud detection
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    Dict with rating results
    """
    ticker = ticker.upper()
    
    # 1. Train model
    model = train_model(verbose=verbose)
    
    # 2. Fetch financial data
    if verbose:
        print(f"\nFetching financial data for {ticker}...")
    
    raw_data = fetch_financial_data(ticker)
    if not raw_data:
        raise ValueError(f"Could not fetch data for {ticker}. Check if the ticker is valid.")
    
    if verbose:
        print(f"  Data date: {raw_data.get('date', 'N/A')}")
    
    # 3. Compute features
    features = compute_features(raw_data)
    
    # 4. Predict
    X = np.array([[features[f] for f in FEATURE_NAMES]])
    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    rating = RATING_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    
    result = {
        'ticker': ticker,
        'rating': rating,
        'rating_numeric': pred_idx,
        'confidence': confidence,
        'investment_grade': pred_idx <= 3,
        'probabilities': {
            RATING_NAMES[i]: float(p)
            for i, p in enumerate(probs) if p > 0.01
        },
        'features': features,
        'raw_data': raw_data
    }
    
    # 5. Fraud detection
    if check_fraud:
        result['red_flags'] = FraudDetector.check_red_flags(raw_data)
        result['altman_z'] = AltmanZScore.calculate(raw_data)
        
        # Overall fraud risk
        severe_flags = sum(1 for f in result['red_flags'] 
                         if 'NEGATIVE' in f or 'CRISIS' in f or 'CONCERN' in f)
        if severe_flags >= 3:
            result['fraud_risk'] = "CRITICAL"
        elif severe_flags >= 1:
            result['fraud_risk'] = "HIGH"
        elif len(result['red_flags']) >= 2:
            result['fraud_risk'] = "MODERATE"
        else:
            result['fraud_risk'] = "LOW"
    
    return result


def print_report(result: Dict):
    """Print formatted credit rating report"""
    ticker = result['ticker']
    rating = result['rating']
    confidence = result['confidence']
    ig = "Investment Grade" if result['investment_grade'] else "Speculative Grade"
    
    print("\n" + "=" * 70)
    print(f"CREDIT RATING REPORT: {ticker}")
    print("=" * 70)
    
    # Rating box
    print(f"\n┌{'─' * 50}┐")
    print(f"│  Rating:         {rating:>15}                  │")
    print(f"│  Confidence:     {confidence:>15.1%}                  │")
    print(f"│  Grade:          {ig:>28}   │")
    print(f"└{'─' * 50}┘")
    
    # Probability distribution
    print(f"\nProbability Distribution:")
    for r, p in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
        bar_len = int(p * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  {r:<4} [{bar}] {p:>6.1%}")
    
    # Key financial ratios
    features = result['features']
    print(f"\nKey Financial Ratios:")
    print(f"  {'Debt/Equity:':<25} {features.get('debt_to_equity', 0):>10.2f}")
    print(f"  {'Interest Coverage:':<25} {features.get('interest_coverage', 0):>10.1f}x")
    print(f"  {'Current Ratio:':<25} {features.get('current_ratio', 0):>10.2f}")
    print(f"  {'Net Margin:':<25} {features.get('net_margin', 0):>10.1%}")
    print(f"  {'ROA:':<25} {features.get('roa', 0):>10.1%}")
    print(f"  {'Debt/EBITDA:':<25} {features.get('debt_to_ebitda', 0):>10.1f}x")
    print(f"  {'Log(Total Assets):':<25} {features.get('log_assets', 0):>10.2f}")
    
    # Fraud detection
    if 'red_flags' in result:
        print(f"\n{'─' * 70}")
        print("FRAUD DETECTION & WARNING SIGNALS")
        print(f"{'─' * 70}")
        
        # Overall risk
        risk_emoji = {
            'LOW': '🟢',
            'MODERATE': '🟡',
            'HIGH': '🔴',
            'CRITICAL': '⛔'
        }
        print(f"\nOverall Risk: {risk_emoji.get(result.get('fraud_risk', 'LOW'), '?')} {result.get('fraud_risk', 'LOW')}")
        
        # Altman Z-Score
        if result.get('altman_z'):
            z = result['altman_z']
            zone_emoji = {'SAFE': '🟢', 'GREY': '🟡', 'DISTRESS': '🔴'}
            print(f"\nAltman Z-Score: {z['z_score']:.2f} ({zone_emoji.get(z['zone'], '')} {z['zone']})")
            print(f"  {z['risk_assessment']}")
        
        # Red flags
        if result.get('red_flags'):
            print(f"\n⚠️  {len(result['red_flags'])} Warning Signal(s) Detected:")
            for flag in result['red_flags']:
                print(f"  • {flag}")
        else:
            print(f"\n✅ No major warning signals detected")
    
    print("\n" + "=" * 70)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Rate any company by ticker symbol',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rate_ticker.py AAPL          # Rate Apple
  python rate_ticker.py TSLA          # Rate Tesla
  python rate_ticker.py GM --no-fraud # Rate GM without fraud detection
  python rate_ticker.py F --quiet     # Rate Ford with minimal output
        """
    )
    
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL, TSLA, GM)')
    parser.add_argument('--no-fraud', action='store_true', help='Skip fraud detection')
    parser.add_argument('--quiet', '-q', action='store_true', help='Less verbose output')
    
    args = parser.parse_args()
    
    try:
        result = rate_ticker(
            ticker=args.ticker,
            check_fraud=not args.no_fraud,
            verbose=not args.quiet
        )
        print_report(result)
        
        # Return code based on rating
        # 0 = Investment Grade, 1 = Speculative Grade
        return 0 if result['investment_grade'] else 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
