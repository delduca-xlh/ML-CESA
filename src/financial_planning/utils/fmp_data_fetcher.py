#!/usr/bin/env python3
"""
fmp_data_fetcher.py - FMP Data Fetcher (FIXED VERSION)

Added fields for accounting engine:
- sharesOutstanding / weightedAverageShsOut
- interestExpense
- totalDebt
- dividendsPaid
- retainedEarnings
- depreciationAndAmortization
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
        """
        if params is None:
            params = {}
        
        params['apikey'] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
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
        """Fetch company profile information."""
        try:
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
        """Fetch income statement data."""
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
        """Fetch balance sheet data."""
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
        """Fetch cash flow statement data."""
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
    
    def _get_column(self, df: pd.DataFrame, possible_names: List[str], default=None):
        """Get column value trying multiple possible names."""
        for name in possible_names:
            if name in df.columns:
                return df[name]
        return default
    
    def extract_ml_features(
        self,
        ticker: str,
        period: str = 'quarter',
        limit: int = 120
    ) -> pd.DataFrame:
        """
        Extract features needed for ML model from all financial statements.
        
        FIXED: Now includes all fields needed for accounting engine:
        - shares outstanding
        - interest expense
        - total debt
        - dividends paid
        - retained earnings
        - depreciation
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
        
        # ================================================================
        # INCOME STATEMENT FIELDS
        # ================================================================
        
        # Revenue
        revenue_cols = ['revenue', 'totalRevenue', 'sales', 'netSales']
        for col in revenue_cols:
            if col in merged.columns:
                ml_data['sales_revenue'] = merged[col]
                break
        
        # COGS
        cogs_cols = ['costOfRevenue', 'costOfGoodsSold', 'cogs']
        for col in cogs_cols:
            if col in merged.columns:
                ml_data['cost_of_goods_sold'] = merged[col]
                break
        
        # Net Income
        ni_cols = ['netIncome', 'netIncome_is', 'earnings', 'profit']
        for col in ni_cols:
            if col in merged.columns:
                ml_data['net_income'] = merged[col]
                break
        
        # EBIT
        ebit_cols = ['operatingIncome', 'ebit', 'operatingProfit']
        for col in ebit_cols:
            if col in merged.columns:
                ml_data['ebit'] = merged[col]
                break
        
        # EBT (Earnings Before Tax)
        ebt_cols = ['incomeBeforeTax', 'ebt', 'pretaxIncome']
        for col in ebt_cols:
            if col in merged.columns:
                ml_data['ebt'] = merged[col]
                break
        
        # NEW: Interest Expense
        interest_cols = ['interestExpense', 'interestExpense_is', 'interestAndDebtExpense']
        for col in interest_cols:
            if col in merged.columns:
                ml_data['interest_expense'] = merged[col].abs()
                break
        
        # NEW: Depreciation & Amortization
        depr_cols = ['depreciationAndAmortization', 'depreciation', 'depreciationAndAmortization_is']
        for col in depr_cols:
            if col in merged.columns:
                ml_data['depreciation'] = merged[col].abs()
                break
        
        # NEW: Shares Outstanding (from Income Statement)
        shares_cols = ['weightedAverageShsOut', 'weightedAverageShsOutDil', 
                       'sharesOutstanding', 'commonStockSharesOutstanding']
        for col in shares_cols:
            if col in merged.columns:
                ml_data['shares_outstanding'] = merged[col]
                break
        
        # ================================================================
        # BALANCE SHEET FIELDS
        # ================================================================
        
        # Total Assets
        assets_cols = ['totalAssets', 'totalAssets_bs', 'assets']
        for col in assets_cols:
            if col in merged.columns:
                ml_data['total_assets'] = merged[col]
                break
        
        # Total Liabilities
        liab_cols = ['totalLiabilities', 'totalLiabilities_bs', 'liabilities']
        for col in liab_cols:
            if col in merged.columns:
                ml_data['total_liabilities'] = merged[col]
                break
        
        # Total Equity
        equity_cols = ['totalStockholdersEquity', 'stockholdersEquity', 
                       'totalEquity', 'equity', 'totalStockholdersEquity_bs']
        for col in equity_cols:
            if col in merged.columns:
                ml_data['total_equity'] = merged[col]
                break
        
        # NEW: Total Debt
        debt_cols = ['totalDebt', 'totalDebt_bs', 'longTermDebt', 
                     'shortLongTermDebtTotal', 'netDebt']
        for col in debt_cols:
            if col in merged.columns:
                ml_data['total_debt'] = merged[col]
                break
        
        # Fallback: estimate debt from liabilities
        if 'total_debt' not in ml_data.columns and 'total_liabilities' in ml_data.columns:
            ml_data['total_debt'] = ml_data['total_liabilities'] * 0.5
        
        # NEW: Retained Earnings
        re_cols = ['retainedEarnings', 'retainedEarnings_bs', 
                   'accumulatedRetainedEarningsDeficit']
        for col in re_cols:
            if col in merged.columns:
                ml_data['retained_earnings'] = merged[col]
                break
        
        # NEW: Common Stock
        cs_cols = ['commonStock', 'commonStock_bs', 'commonStockValue']
        for col in cs_cols:
            if col in merged.columns:
                ml_data['common_stock'] = merged[col]
                break
        
        # ================================================================
        # CASH FLOW FIELDS
        # ================================================================
        
        # CapEx
        capex_cols = ['capitalExpenditure', 'capitalExpenditure_cf', 
                      'capex', 'purchaseOfPPE']
        for col in capex_cols:
            if col in merged.columns:
                ml_data['capex'] = merged[col].abs()
                break
        
        # If capex not found, estimate
        if 'capex' not in ml_data.columns and 'total_assets' in ml_data.columns:
            ml_data['capex'] = ml_data['total_assets'] * 0.03
        
        # NEW: Dividends Paid (FMP uses commonDividendsPaid or netDividendsPaid)
        div_cols = ['commonDividendsPaid', 'netDividendsPaid', 'dividendsPaid', 
                    'dividendsPaid_cf', 'paymentOfDividends']
        for col in div_cols:
            if col in merged.columns:
                ml_data['dividends_paid'] = merged[col].abs()
                break
        
        # NEW: Stock Repurchases (Buybacks) - FMP uses commonStockRepurchased
        buyback_cols = ['commonStockRepurchased', 'netCommonStockIssuance',
                        'repurchaseOfCommonStock', 'stockRepurchased', 'buybackOfShares']
        for col in buyback_cols:
            if col in merged.columns:
                # Repurchases are negative in FMP, we want positive value
                val = merged[col]
                # If values are negative (outflows), take absolute value
                ml_data['stock_repurchased'] = val.abs()
                break
        
        # Operating Cash Flow
        ocf_cols = ['operatingCashFlow', 'netCashProvidedByOperatingActivities',
                    'cashFlowFromOperations']
        for col in ocf_cols:
            if col in merged.columns:
                ml_data['operating_cash_flow'] = merged[col]
                break
        
        # ================================================================
        # DERIVED FIELDS
        # ================================================================
        
        # Overhead expenses (derived from EBIT if available)
        if 'ebit' in ml_data.columns and 'sales_revenue' in ml_data.columns and 'cost_of_goods_sold' in ml_data.columns:
            ml_data['overhead_expenses'] = (
                ml_data['sales_revenue'] - 
                ml_data['cost_of_goods_sold'] - 
                ml_data['ebit']
            ).clip(lower=0)
        else:
            ml_data['overhead_expenses'] = ml_data.get('sales_revenue', 0) * 0.15
        
        ml_data['payroll_expenses'] = ml_data['overhead_expenses'] * 0.5
        
        # ================================================================
        # VALIDATION & CLEANUP
        # ================================================================
        
        # Check for required columns
        required = ['sales_revenue', 'cost_of_goods_sold', 'total_assets', 
                   'total_liabilities', 'total_equity', 'net_income']
        
        missing = [col for col in required if col not in ml_data.columns]
        if missing:
            print(f"  ⚠ Warning: Missing required columns: {missing}")
        
        # Check for new accounting engine columns
        accounting_cols = ['shares_outstanding', 'interest_expense', 'total_debt',
                          'dividends_paid', 'retained_earnings']
        available_acc = [col for col in accounting_cols if col in ml_data.columns]
        missing_acc = [col for col in accounting_cols if col not in ml_data.columns]
        
        print(f"  ✓ Accounting fields found: {available_acc}")
        if missing_acc:
            print(f"  ⚠ Accounting fields missing: {missing_acc}")
        
        # Clean up
        ml_data = ml_data.dropna(subset=['sales_revenue', 'total_assets'])
        
        print(f"  ✓ Final ML data: {len(ml_data)} periods")
        print(f"  ✓ Total columns: {len(ml_data.columns)}")
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
    print("TESTING FMP DATA FETCHER (FIXED VERSION)")
    print("="*80)
    
    fetcher = FMPDataFetcher()
    ticker = 'AAPL'
    
    print(f"\n[1] Fetching financial statements...")
    try:
        ml_data = fetcher.extract_ml_features(ticker, period='quarter', limit=40)
        
        if len(ml_data) > 0:
            print(f"\n  ✓ SUCCESS! Got {len(ml_data)} quarters")
            
            print(f"\n  Columns available ({len(ml_data.columns)}):")
            for col in sorted(ml_data.columns):
                val = ml_data[col].iloc[-1]
                if col == 'date':
                    print(f"    {col}: {val}")
                elif abs(val) > 1e9:
                    print(f"    {col}: ${val/1e9:.2f}B")
                elif abs(val) > 1e6:
                    print(f"    {col}: ${val/1e6:.2f}M")
                else:
                    print(f"    {col}: {val:.2f}")
            
            # Check accounting engine fields specifically
            print(f"\n  KEY ACCOUNTING ENGINE FIELDS:")
            checks = {
                'shares_outstanding': '~15B for AAPL',
                'interest_expense': '~$1B/quarter',
                'total_debt': '~$100B',
                'dividends_paid': '~$4B/quarter',
                'retained_earnings': '~$5B',
            }
            for field, expected in checks.items():
                if field in ml_data.columns:
                    val = ml_data[field].iloc[-1]
                    if abs(val) > 1e9:
                        print(f"    ✓ {field}: ${val/1e9:.2f}B (expected: {expected})")
                    else:
                        print(f"    ✓ {field}: {val:.2f} (expected: {expected})")
                else:
                    print(f"    ✗ {field}: NOT FOUND (expected: {expected})")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)


if __name__ == "__main__":
    test_correct_api()
