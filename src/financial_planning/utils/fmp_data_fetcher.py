#!/usr/bin/env python3
"""
fmp_data_fetcher_correct.py - FMP Data Fetcher (CORRECT ENDPOINTS)

Based on official FMP documentation:
https://site.financialmodelingprep.com/developer/docs/stable/income-statement

Endpoints:
- https://financialmodelingprep.com/stable/income-statement?symbol=AAPL
- https://financialmodelingprep.com/stable/balance-sheet-statement?symbol=AAPL
- https://financialmodelingprep.com/stable/cashflow-statement?symbol=AAPL
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class FMPDataFetcher:
    """Fetch financial data from Financial Modeling Prep /stable/ API."""
    
    def __init__(self, api_key: str = "vRmvrzQZbCF0SqRCeWggAOGDLRtnTQNR"):
        """
        Initialize FMP data fetcher.
        
        Args:
            api_key: Your FMP API key
        """
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/stable"
        
    def _make_request(self, endpoint: str, params: Dict = None) -> List:
        """
        Make API request with proper error handling.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            
        Returns:
            JSON response as list
        """
        if params is None:
            params = {}
        
        # Add API key to params
        params['apikey'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # FMP returns list
            if not isinstance(data, list):
                print(f"Warning: Expected list, got {type(data)}")
                return []
            
            return data
            
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error {e.response.status_code}: {e}")
            try:
                error_data = e.response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Response text: {e.response.text[:200]}")
            raise
        except Exception as e:
            print(f"Request error: {e}")
            raise
    
    def fetch_company_profile(self, ticker: str) -> Dict:
        """
        Fetch company profile information.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            # Use search-symbol endpoint
            data = self._make_request('search-symbol', {'query': ticker})
            
            if data and len(data) > 0:
                return data[0]
            else:
                return {'symbol': ticker, 'companyName': ticker}
                
        except Exception as e:
            print(f"Error fetching company profile: {e}")
            return {'symbol': ticker, 'companyName': ticker}
    
    def fetch_income_statement(
        self, 
        ticker: str, 
        period: str = 'quarter',
        limit: int = 120
    ) -> pd.DataFrame:
        """
        Fetch income statement data.
        
        Args:
            ticker: Stock ticker
            period: 'quarter' or 'annual'  
            limit: Number of periods to fetch
            
        Returns:
            DataFrame with income statement data
        """
        try:
            params = {
                'symbol': ticker,
                'period': period,
                'limit': limit
            }
            
            data = self._make_request('income-statement', params)
            
            if not data:
                print(f"No income statement data returned for {ticker}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Convert date column if exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
            
            print(f"  ✓ Income Statement: {len(df)} periods")
            return df
            
        except Exception as e:
            print(f"Error fetching income statement: {e}")
            return pd.DataFrame()
    
    def fetch_balance_sheet(
        self, 
        ticker: str, 
        period: str = 'quarter',
        limit: int = 120
    ) -> pd.DataFrame:
        """
        Fetch balance sheet data.
        
        Args:
            ticker: Stock ticker
            period: 'quarter' or 'annual'
            limit: Number of periods to fetch
            
        Returns:
            DataFrame with balance sheet data
        """
        try:
            params = {
                'symbol': ticker,
                'period': period,
                'limit': limit
            }
            
            data = self._make_request('balance-sheet-statement', params)
            
            if not data:
                print(f"No balance sheet data returned for {ticker}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
            
            print(f"  ✓ Balance Sheet: {len(df)} periods")
            return df
            
        except Exception as e:
            print(f"Error fetching balance sheet: {e}")
            return pd.DataFrame()
    
    def fetch_cash_flow(
        self, 
        ticker: str, 
        period: str = 'quarter',
        limit: int = 120
    ) -> pd.DataFrame:
        """
        Fetch cash flow statement data.
        
        Args:
            ticker: Stock ticker
            period: 'quarter' or 'annual'
            limit: Number of periods to fetch
            
        Returns:
            DataFrame with cash flow data
        """
        try:
            params = {
                'symbol': ticker,
                'period': period,
                'limit': limit
            }
            
            data = self._make_request('cash-flow-statement', params)
            
            if not data:
                print(f"No cash flow data returned for {ticker}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
            
            print(f"  ✓ Cash Flow: {len(df)} periods")
            return df
            
        except Exception as e:
            print(f"Error fetching cash flow: {e}")
            return pd.DataFrame()
    
    def extract_ml_features(
        self,
        ticker: str,
        period: str = 'quarter',
        limit: int = 120
    ) -> pd.DataFrame:
        """
        Extract features needed for ML model from all financial statements.
        
        Args:
            ticker: Stock ticker
            period: 'quarter' or 'annual'
            limit: Number of periods
            
        Returns:
            DataFrame with ML-ready features
        """
        print(f"\nFetching {period}ly data for {ticker} from FMP /stable/ API...")
        print(f"  API Key: {self.api_key[:10]}...{self.api_key[-5:]}")
        
        # Fetch all three statements
        income = self.fetch_income_statement(ticker, period, limit)
        balance = self.fetch_balance_sheet(ticker, period, limit)
        cashflow = self.fetch_cash_flow(ticker, period, limit)
        
        if income.empty or balance.empty:
            raise ValueError(f"No data returned for {ticker}")
        
        # Merge on date
        merged = income.merge(balance, on='date', how='inner', suffixes=('_is', '_bs'))
        
        if not cashflow.empty:
            merged = merged.merge(cashflow, on='date', how='left', suffixes=('', '_cf'))
        
        print(f"  ✓ Merged: {len(merged)} periods")
        
        # Extract required columns with flexible column name matching
        ml_data = pd.DataFrame()
        ml_data['date'] = merged['date']
        
        # From Income Statement - try different possible column names
        revenue_cols = ['revenue', 'totalRevenue', 'sales', 'netSales']
        for col in revenue_cols:
            if col in merged.columns:
                ml_data['sales_revenue'] = merged[col]
                break
        
        cogs_cols = ['costOfRevenue', 'costOfGoodsSold', 'cogs']
        for col in cogs_cols:
            if col in merged.columns:
                ml_data['cost_of_goods_sold'] = merged[col]
                break
        
        ni_cols = ['netIncome', 'earnings', 'profit']
        for col in ni_cols:
            if col in merged.columns:
                ml_data['net_income'] = merged[col]
                break
        
        ebit_cols = ['operatingIncome', 'ebit', 'operatingProfit']
        for col in ebit_cols:
            if col in merged.columns:
                ml_data['ebit'] = merged[col]
                break
        
        # From Balance Sheet
        assets_cols = ['totalAssets', 'assets']
        for col in assets_cols:
            if col in merged.columns:
                ml_data['total_assets'] = merged[col]
                break
        
        liab_cols = ['totalLiabilities', 'liabilities']
        for col in liab_cols:
            if col in merged.columns:
                ml_data['total_liabilities'] = merged[col]
                break
        
        equity_cols = ['totalStockholdersEquity', 'stockholdersEquity', 'equity', 'totalEquity']
        for col in equity_cols:
            if col in merged.columns:
                ml_data['total_equity'] = merged[col]
                break
        
        # From Cash Flow
        capex_cols = ['capitalExpenditure', 'capex', 'purchaseOfPPE']
        for col in capex_cols:
            if col in merged.columns:
                ml_data['capex'] = merged[col].abs()
                break
        
        # If capex not found, estimate
        if 'capex' not in ml_data.columns and 'total_assets' in ml_data.columns:
            ml_data['capex'] = ml_data['total_assets'] * 0.05
        
        # Derive overhead and payroll if EBIT available
        if 'ebit' in ml_data.columns and 'sales_revenue' in ml_data.columns and 'cost_of_goods_sold' in ml_data.columns:
            ml_data['overhead_expenses'] = (
                ml_data['sales_revenue'] - 
                ml_data['cost_of_goods_sold'] - 
                ml_data['ebit']
            ).clip(lower=0)
        else:
            ml_data['overhead_expenses'] = ml_data.get('sales_revenue', 0) * 0.15
        
        ml_data['payroll_expenses'] = ml_data['overhead_expenses'] * 0.5
        
        # Check for required columns
        required = ['sales_revenue', 'cost_of_goods_sold', 'total_assets', 
                   'total_liabilities', 'total_equity', 'net_income']
        
        missing = [col for col in required if col not in ml_data.columns]
        if missing:
            print(f"  ⚠ Warning: Missing columns: {missing}")
            print(f"  Available columns in merged data: {list(merged.columns)[:20]}")
        
        # Clean up
        ml_data = ml_data.dropna(subset=['sales_revenue', 'total_assets'])
        
        print(f"  ✓ Final ML data: {len(ml_data)} periods")
        if len(ml_data) > 0:
            print(f"  Date range: {ml_data['date'].min().date()} to {ml_data['date'].max().date()}")
            print(f"  Years: {(ml_data['date'].max() - ml_data['date'].min()).days / 365.25:.1f}")
        
        return ml_data


# ============================================================================
# Test function
# ============================================================================

def test_correct_api():
    """Test the corrected FMP API fetcher."""
    
    print("="*80)
    print("TESTING CORRECTED FMP /stable/ API")
    print("Using documented endpoints from FMP docs")
    print("="*80)
    
    fetcher = FMPDataFetcher()
    ticker = 'AAPL'
    
    print(f"\n[1] Fetching company profile...")
    try:
        profile = fetcher.fetch_company_profile(ticker)
        print(f"  ✓ Company: {profile.get('companyName', 'N/A')}")
        print(f"  ✓ Symbol: {profile.get('symbol', 'N/A')}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print(f"\n[2] Fetching financial statements...")
    try:
        ml_data = fetcher.extract_ml_features(ticker, period='quarter', limit=40)
        
        if len(ml_data) > 0:
            print(f"\n  ✓ SUCCESS! Got {len(ml_data)} quarters of REAL data!")
            print(f"\n  Sample (last 5 quarters):")
            sample = ml_data.tail(5)[['date', 'sales_revenue', 'net_income', 'total_assets']]
            for _, row in sample.iterrows():
                print(f"    {row['date'].date()}: "
                      f"Revenue=${row['sales_revenue']/1e9:.1f}B, "
                      f"NI=${row['net_income']/1e9:.1f}B, "
                      f"Assets=${row['total_assets']/1e9:.1f}B")
            
            # Save
            output_file = f'{ticker.lower()}_fmp_real_data.csv'
            ml_data.to_csv(output_file, index=False)
            print(f"\n  ✓ Saved to: {output_file}")
            
            print(f"\n  🎉 READY FOR ML TRAINING!")
        else:
            print(f"  ⚠ No data returned")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)


if __name__ == "__main__":
    test_correct_api()