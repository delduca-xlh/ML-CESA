"""
Yahoo Finance Data Fetcher

Utility to fetch historical financial statement data from Yahoo Finance
for training the balance sheet forecasting model.

Based on: https://rfachrizal.medium.com/how-to-obtain-financial-statements-from-stocks-using-yfinance-87c432b803b8
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


class YahooFinanceDataFetcher:
    """Fetch and process financial data from Yahoo Finance."""
    
    def __init__(self):
        """Initialize data fetcher."""
        self.cache = {}
    
    def fetch_company_data(
        self,
        ticker: str,
        include_info: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all financial data for a company.
        
        Args:
            ticker: Stock ticker symbol
            include_info: Whether to include company info
            
        Returns:
            Dictionary with all financial data
        """
        print(f"Fetching data for {ticker}...")
        
        try:
            stock = yf.Ticker(ticker)
            
            data = {
                'ticker': ticker,
                'balance_sheet': stock.balance_sheet,
                'income_statement': stock.income_stmt,
                'cash_flow': stock.cashflow,
                'quarterly_balance_sheet': stock.quarterly_balance_sheet,
                'quarterly_income': stock.quarterly_income_stmt,
                'quarterly_cashflow': stock.quarterly_cashflow,
            }
            
            if include_info:
                data['info'] = stock.info
            
            # Cache the data
            self.cache[ticker] = data
            
            return data
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return {}
    
    def fetch_multiple_companies(
        self,
        tickers: List[str]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch data for multiple companies.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping tickers to their data
        """
        all_data = {}
        
        for ticker in tickers:
            data = self.fetch_company_data(ticker)
            if data:
                all_data[ticker] = data
        
        return all_data
    
    def extract_time_series_features(
        self,
        ticker: str,
        frequency: str = 'annual'
    ) -> pd.DataFrame:
        """
        Extract time series features for modeling.
        
        Args:
            ticker: Stock ticker
            frequency: 'annual' or 'quarterly'
            
        Returns:
            DataFrame with time series features
        """
        if ticker not in self.cache:
            self.fetch_company_data(ticker)
        
        data = self.cache[ticker]
        
        # Select frequency
        if frequency == 'quarterly':
            bs = data.get('quarterly_balance_sheet', pd.DataFrame())
            is_stmt = data.get('quarterly_income', pd.DataFrame())
            cf = data.get('quarterly_cashflow', pd.DataFrame())
        else:
            bs = data.get('balance_sheet', pd.DataFrame())
            is_stmt = data.get('income_statement', pd.DataFrame())
            cf = data.get('cash_flow', pd.DataFrame())
        
        if bs.empty or is_stmt.empty:
            return pd.DataFrame()
        
        # Extract key features
        features = []
        
        for col in bs.columns:
            period_features = {
                'ticker': ticker,
                'date': col,
                'period_type': frequency
            }
            
            # Balance Sheet items
            period_features['total_assets'] = self._safe_get(bs, 'Total Assets', col)
            period_features['current_assets'] = self._safe_get(bs, 'Current Assets', col)
            period_features['cash'] = self._safe_get(bs, 'Cash And Cash Equivalents', col)
            period_features['accounts_receivable'] = self._safe_get(bs, 'Accounts Receivable', col)
            period_features['inventory'] = self._safe_get(bs, 'Inventory', col)
            period_features['ppe'] = self._safe_get(bs, 'Net PPE', col)
            
            period_features['total_liabilities'] = self._safe_get(bs, 'Total Liabilities Net Minority Interest', col)
            period_features['current_liabilities'] = self._safe_get(bs, 'Current Liabilities', col)
            period_features['accounts_payable'] = self._safe_get(bs, 'Accounts Payable', col)
            period_features['short_term_debt'] = self._safe_get(bs, 'Current Debt', col)
            period_features['long_term_debt'] = self._safe_get(bs, 'Long Term Debt', col)
            
            period_features['total_equity'] = self._safe_get(bs, 'Total Equity Gross Minority Interest', col)
            period_features['retained_earnings'] = self._safe_get(bs, 'Retained Earnings', col)
            
            # Income Statement items (if available for this period)
            if col in is_stmt.columns:
                period_features['revenue'] = self._safe_get(is_stmt, 'Total Revenue', col)
                period_features['cost_of_revenue'] = self._safe_get(is_stmt, 'Cost Of Revenue', col)
                period_features['gross_profit'] = self._safe_get(is_stmt, 'Gross Profit', col)
                period_features['operating_income'] = self._safe_get(is_stmt, 'Operating Income', col)
                period_features['ebit'] = self._safe_get(is_stmt, 'EBIT', col)
                period_features['net_income'] = self._safe_get(is_stmt, 'Net Income', col)
                period_features['interest_expense'] = self._safe_get(is_stmt, 'Interest Expense', col)
            
            # Cash Flow items (if available)
            if not cf.empty and col in cf.columns:
                period_features['operating_cash_flow'] = self._safe_get(cf, 'Operating Cash Flow', col)
                period_features['capex'] = self._safe_get(cf, 'Capital Expenditure', col)
                period_features['free_cash_flow'] = self._safe_get(cf, 'Free Cash Flow', col)
            
            # Calculate derived features
            period_features = self._calculate_derived_features(period_features)
            
            features.append(period_features)
        
        df = pd.DataFrame(features)
        df = df.sort_values('date')
        
        return df
    
    def _safe_get(
        self,
        df: pd.DataFrame,
        row_name: str,
        col_name: any
    ) -> Optional[float]:
        """Safely get a value from DataFrame."""
        try:
            if row_name in df.index and col_name in df.columns:
                value = df.loc[row_name, col_name]
                if pd.notna(value):
                    return float(value)
        except:
            pass
        return None
    
    def _calculate_derived_features(
        self,
        features: Dict[str, Optional[float]]
    ) -> Dict[str, Optional[float]]:
        """Calculate derived financial features."""
        
        # Working capital
        if features.get('current_assets') and features.get('current_liabilities'):
            features['working_capital'] = (
                features['current_assets'] - features['current_liabilities']
            )
        
        # Current ratio
        if features.get('current_assets') and features.get('current_liabilities'):
            if features['current_liabilities'] > 0:
                features['current_ratio'] = (
                    features['current_assets'] / features['current_liabilities']
                )
        
        # Quick ratio
        if (features.get('current_assets') and features.get('inventory') and
            features.get('current_liabilities')):
            if features['current_liabilities'] > 0:
                quick_assets = features['current_assets'] - features['inventory']
                features['quick_ratio'] = quick_assets / features['current_liabilities']
        
        # Debt ratios
        total_debt = (
            (features.get('short_term_debt') or 0) +
            (features.get('long_term_debt') or 0)
        )
        features['total_debt'] = total_debt
        
        if features.get('total_equity') and features['total_equity'] > 0:
            features['debt_to_equity'] = total_debt / features['total_equity']
        
        if features.get('total_assets') and features['total_assets'] > 0:
            features['debt_to_assets'] = total_debt / features['total_assets']
        
        # Profitability ratios
        if features.get('net_income') and features.get('total_assets'):
            if features['total_assets'] > 0:
                features['roa'] = features['net_income'] / features['total_assets']
        
        if features.get('net_income') and features.get('total_equity'):
            if features['total_equity'] > 0:
                features['roe'] = features['net_income'] / features['total_equity']
        
        if features.get('net_income') and features.get('revenue'):
            if features['revenue'] > 0:
                features['net_margin'] = features['net_income'] / features['revenue']
        
        # Asset turnover
        if features.get('revenue') and features.get('total_assets'):
            if features['total_assets'] > 0:
                features['asset_turnover'] = features['revenue'] / features['total_assets']
        
        return features
    
    def prepare_ml_dataset(
        self,
        tickers: List[str],
        frequency: str = 'annual',
        include_lags: int = 4
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare dataset for machine learning.
        
        Args:
            tickers: List of tickers
            frequency: 'annual' or 'quarterly'
            include_lags: Number of lagged periods to include
            
        Returns:
            Tuple of (features DataFrame, targets DataFrame)
        """
        all_features = []
        
        for ticker in tickers:
            df = self.extract_time_series_features(ticker, frequency)
            if not df.empty:
                all_features.append(df)
        
        if not all_features:
            return pd.DataFrame(), pd.DataFrame()
        
        # Combine all companies
        combined = pd.concat(all_features, ignore_index=True)
        
        # Add lagged features
        combined = self._add_lagged_features(combined, include_lags)
        
        # Split features and targets
        target_cols = [
            'total_assets', 'total_liabilities', 'total_equity',
            'revenue', 'net_income'
        ]
        
        feature_cols = [col for col in combined.columns if col not in target_cols + ['ticker', 'date', 'period_type']]
        
        X = combined[feature_cols]
        y = combined[target_cols]
        
        return X, y
    
    def _add_lagged_features(
        self,
        df: pd.DataFrame,
        n_lags: int
    ) -> pd.DataFrame:
        """Add lagged features for time series modeling."""
        
        df = df.sort_values(['ticker', 'date'])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['date']]
        
        for col in numeric_cols:
            for lag in range(1, n_lags + 1):
                df[f'{col}_lag{lag}'] = df.groupby('ticker')[col].shift(lag)
        
        # Drop rows with NaN lags
        df = df.dropna()
        
        return df
    
    def export_to_csv(
        self,
        ticker: str,
        output_dir: str = '.'
    ):
        """Export company data to CSV files."""
        if ticker not in self.cache:
            self.fetch_company_data(ticker)
        
        data = self.cache[ticker]
        
        # Export each statement
        for key, df in data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                filename = f"{output_dir}/{ticker}_{key}.csv"
                df.to_csv(filename)
                print(f"Exported {filename}")
    
    def export_to_excel(
        self,
        ticker: str,
        output_file: str
    ):
        """Export all company data to a single Excel file."""
        if ticker not in self.cache:
            self.fetch_company_data(ticker)
        
        data = self.cache[ticker]
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for key, df in data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    sheet_name = key.replace('_', ' ').title()[:31]  # Excel limit
                    df.to_excel(writer, sheet_name=sheet_name)
            
            # Add info sheet if available
            if 'info' in data and isinstance(data['info'], dict):
                info_df = pd.DataFrame([data['info']]).T
                info_df.columns = ['Value']
                info_df.to_excel(writer, sheet_name='Company Info')
        
        print(f"Exported {output_file}")


def example_usage():
    """Example usage of the data fetcher."""
    
    print("Yahoo Finance Data Fetcher Example")
    print("=" * 60)
    
    fetcher = YahooFinanceDataFetcher()
    
    # Example 1: Fetch single company
    print("\nExample 1: Fetching Apple (AAPL) data...")
    apple_data = fetcher.fetch_company_data('AAPL')
    
    if apple_data:
        print(f"  Balance Sheet shape: {apple_data['balance_sheet'].shape}")
        print(f"  Income Statement shape: {apple_data['income_statement'].shape}")
        print(f"  Cash Flow shape: {apple_data['cash_flow'].shape}")
    
    # Example 2: Extract time series features
    print("\nExample 2: Extracting time series features...")
    features = fetcher.extract_time_series_features('AAPL')
    print(f"  Features shape: {features.shape}")
    print(f"  Columns: {list(features.columns)}")
    
    # Example 3: Prepare ML dataset
    print("\nExample 3: Preparing ML dataset...")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    X, y = fetcher.prepare_ml_dataset(tickers, frequency='annual', include_lags=3)
    
    if not X.empty:
        print(f"  Features shape: {X.shape}")
        print(f"  Targets shape: {y.shape}")
        print(f"  Sample features: {list(X.columns[:10])}")
    
    # Example 4: Export to Excel
    print("\nExample 4: Exporting to Excel...")
    fetcher.export_to_excel('AAPL', 'apple_financials.xlsx')


if __name__ == "__main__":
    example_usage()
