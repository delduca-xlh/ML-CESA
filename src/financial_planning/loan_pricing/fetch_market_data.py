#!/usr/bin/env python3
"""
Fetch Market Data for Loan Pricing Model
=========================================

Adds market-based features to improve spread prediction:
- Beta (stock volatility vs market)
- Market cap

Data Source: FMP API (stable endpoints)

Usage:
    python src/financial_planning/loan_pricing/fetch_market_data.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

# FMP API
FMP_API_KEY = "vRmvrzQZbCF0SqRCeWggAOGDLRtnTQNR"


def find_project_root():
    """Find project root directory."""
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.exists(os.path.join(current, 'data')):
            return current
        current = os.path.dirname(current)
    return os.getcwd()


def fetch_company_profile(ticker: str, debug: bool = False) -> dict:
    """
    Fetch company profile from FMP using NEW stable endpoint.
    
    Returns: beta, market_cap
    """
    # NEW stable endpoint (not /api/v3/)
    url = f"https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={FMP_API_KEY}"
    
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        
        if debug:
            print(f"  DEBUG {ticker}: status={resp.status_code}, data={str(data)[:300]}")
        
        if data and isinstance(data, list) and len(data) > 0:
            profile = data[0]
            return {
                'beta': profile.get('beta'),
                'market_cap': profile.get('mktCap') or profile.get('marketCap'),
            }
        elif data and isinstance(data, dict):
            if 'Error Message' in data:
                if debug:
                    print(f"  API Error: {data['Error Message']}")
                return None
            return {
                'beta': data.get('beta'),
                'market_cap': data.get('mktCap') or data.get('marketCap'),
            }
    except Exception as e:
        if debug:
            print(f"  Exception {ticker}: {e}")
    
    return None


def fetch_market_data_for_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch market data for all tickers in dataframe.
    """
    print("\nFetching market data from FMP...")
    
    # Test first ticker with debug
    print("\n--- Testing API with first 3 tickers ---")
    for ticker in df['ticker'].head(3):
        fetch_company_profile(ticker, debug=True)
    print("--- End test ---\n")
    
    betas = []
    market_caps = []
    
    tickers = df['ticker'].tolist()
    total = len(tickers)
    success_count = 0
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{total} (success: {success_count})")
        
        # Fetch profile
        profile = fetch_company_profile(ticker)
        
        if profile and profile.get('beta') is not None:
            betas.append(profile.get('beta'))
            market_caps.append(profile.get('market_cap'))
            success_count += 1
        else:
            betas.append(None)
            market_caps.append(None)
        
        # Rate limiting
        time.sleep(0.1)
    
    df['beta'] = betas
    df['market_cap'] = market_caps
    
    # Convert to numeric and calculate log
    df['market_cap'] = pd.to_numeric(df['market_cap'], errors='coerce')
    df['beta'] = pd.to_numeric(df['beta'], errors='coerce')
    df['log_market_cap'] = np.log(df['market_cap'].clip(lower=1e6) / 1e6)
    
    print(f"\nTotal success: {success_count}/{total}")
    
    return df


def main():
    print("=" * 70)
    print("FETCHING MARKET DATA FOR LOAN PRICING")
    print("=" * 70)
    
    project_root = find_project_root()
    
    # Load existing training data
    csv_path = os.path.join(project_root, 'data', 'loan_pricing_training_data.csv')
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        print("   Run loan_pricing_model.py first!")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} companies")
    
    # Fetch market data
    df = fetch_market_data_for_all(df)
    
    # Summary
    print("\n" + "=" * 70)
    print("MARKET DATA SUMMARY")
    print("=" * 70)
    
    beta_valid = df['beta'].notna().sum()
    mc_valid = df['market_cap'].notna().sum()
    
    print(f"\nBeta:")
    print(f"  Valid: {beta_valid}/{len(df)} ({beta_valid/len(df)*100:.1f}%)")
    if beta_valid > 0:
        print(f"  Mean:  {df['beta'].mean():.2f}")
        print(f"  Range: [{df['beta'].min():.2f}, {df['beta'].max():.2f}]")
    
    print(f"\nMarket Cap:")
    print(f"  Valid: {mc_valid}/{len(df)} ({mc_valid/len(df)*100:.1f}%)")
    if mc_valid > 0:
        print(f"  Mean:  ${df['market_cap'].mean()/1e9:.1f}B")
    
    # Save
    output_path = os.path.join(project_root, 'data', 'loan_pricing_training_data_with_market.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    # Show sample
    print("\nSample data:")
    cols = ['ticker', 'rating', 'spread_bps', 'beta', 'log_market_cap']
    print(df[cols].head(15).to_string())


if __name__ == "__main__":
    main()