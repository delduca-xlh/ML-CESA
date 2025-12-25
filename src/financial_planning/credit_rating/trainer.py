"""
Credit Rating Model Trainer
============================

Complete training and testing pipeline for credit rating model.

Training Data Source: Financial Modeling Prep (FMP) API
- Stable, standardized data
- Covers 1000+ public companies
- Consistent field names

Testing Data Source: FMP API or PDF extraction
- Flexible input for new companies

Workflow:
1. Fetch training data from FMP for companies with known ratings
2. Compute financial ratios
3. Train Ordinal Logistic Regression model
4. Save model for later use
5. Test on new companies

Author: Lihao Xiao
"""

import numpy as np
import pandas as pd
import requests
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from datetime import datetime

# Import the model
try:
    from .ordinal_lr import OrdinalLogisticRegression
    from .fraud_detector import FraudDetector, AltmanZScore
except ImportError:
    from ordinal_lr import OrdinalLogisticRegression
    from fraud_detector import FraudDetector, AltmanZScore


# ============================================================
# Rating Constants
# ============================================================

RATING_NAMES = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']

# S&P Rating to numeric mapping (8 classes)
RATING_MAP = {
    'AAA': 0,
    'AA+': 1, 'AA': 1, 'AA-': 1,
    'A+': 2, 'A': 2, 'A-': 2,
    'BBB+': 3, 'BBB': 3, 'BBB-': 3,
    'BB+': 4, 'BB': 4, 'BB-': 4,
    'B+': 5, 'B': 5, 'B-': 5,
    'CCC+': 6, 'CCC': 6, 'CCC-': 6, 'CC': 6, 'C': 6,
    'D': 7, 'SD': 7, 'NR': None
}

# Feature names for the model
FEATURE_NAMES = [
    'debt_to_equity',
    'interest_coverage',
    'current_ratio',
    'net_margin',
    'roa',
    'debt_to_ebitda',
    'log_assets'
]


# ============================================================
# Training Data: Companies with Known Ratings
# ============================================================

# Import large training data (450+ companies)
try:
    from .training_data import LARGE_TRAINING_COMPANIES
    TRAINING_COMPANIES = LARGE_TRAINING_COMPANIES
except ImportError:
    from training_data import LARGE_TRAINING_COMPANIES
    TRAINING_COMPANIES = LARGE_TRAINING_COMPANIES


# ============================================================
# FMP Data Fetcher for Credit Rating
# ============================================================

class FMPCreditRatingFetcher:
    """
    Fetch financial data from FMP API and compute credit rating features.
    
    This integrates with your existing fmp_data_fetcher.py
    """
    
    def __init__(self, api_key: str = "vRmvrzQZbCF0SqRCeWggAOGDLRtnTQNR"):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/stable"
        
    def _make_request(self, endpoint: str, params: Dict = None) -> List:
        """Make API request"""
        if params is None:
            params = {}
        params['apikey'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            print(f"  API Error: {e}")
            return []
    
    def fetch_financial_data(self, ticker: str, period: str = 'annual') -> Dict:
        """
        Fetch all financial data needed for credit rating
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        period : str
            'annual' or 'quarter'
            
        Returns:
        --------
        Dict with raw financial data
        """
        # Fetch income statement
        is_data = self._make_request('income-statement', {
            'symbol': ticker,
            'period': period,
            'limit': 1
        })
        
        # Fetch balance sheet
        bs_data = self._make_request('balance-sheet-statement', {
            'symbol': ticker,
            'period': period,
            'limit': 1
        })
        
        # Fetch cash flow
        cf_data = self._make_request('cash-flow-statement', {
            'symbol': ticker,
            'period': period,
            'limit': 1
        })
        
        if not is_data or not bs_data:
            return {}
        
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
            'ebit': is_item.get('operatingIncome', 0),  # EBIT ≈ Operating Income
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
    
    def compute_credit_features(self, raw_data: Dict) -> Optional[Dict]:
        """
        Compute credit rating features from raw financial data
        
        Parameters:
        -----------
        raw_data : Dict
            Raw financial data from fetch_financial_data()
            
        Returns:
        --------
        Dict with 7 features for credit rating model
        """
        if not raw_data:
            return None
        
        # Extract values with safe defaults
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
        
        # Handle edge cases to avoid division by zero or extreme values
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
        
        # Calculate interest coverage
        if interest_expense > 0:
            interest_coverage = ebit / interest_expense
        else:
            interest_coverage = 100.0 if ebit > 0 else 0.1
        
        # Clip extreme values
        interest_coverage = np.clip(interest_coverage, -50, 200)
        
        features = {
            'debt_to_equity': np.clip(total_debt / total_equity, 0, 50),
            'interest_coverage': interest_coverage,
            'current_ratio': np.clip(current_assets / current_liabilities, 0, 10),
            'net_margin': np.clip(net_income / revenue, -1, 1),
            'roa': np.clip(net_income / total_assets, -0.5, 0.5),
            'debt_to_ebitda': np.clip(total_debt / ebitda, -50, 50),
            'log_assets': np.log10(max(total_assets, 1))
        }
        
        return features
    
    def fetch_company_features(self, ticker: str, period: str = 'annual') -> Optional[Dict]:
        """
        Fetch and compute features for a single company
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        period : str
            'annual' or 'quarter'
            
        Returns:
        --------
        Dict with features and metadata
        """
        raw_data = self.fetch_financial_data(ticker, period)
        if not raw_data:
            return None
        
        features = self.compute_credit_features(raw_data)
        if not features:
            return None
        
        features['ticker'] = ticker
        features['date'] = raw_data.get('date', '')
        features['raw_data'] = raw_data  # Keep raw data for reference
        
        return features


# ============================================================
# Credit Rating Training Pipeline
# ============================================================

class CreditRatingTrainer:
    """
    Complete training pipeline for credit rating model
    
    Usage:
    ------
    trainer = CreditRatingTrainer()
    
    # Collect training data from FMP
    trainer.collect_training_data()
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save('credit_rating_model.pkl')
    
    # Predict on new company
    result = trainer.predict('TSLA')
    """
    
    def __init__(self, api_key: str = None):
        self.fetcher = FMPCreditRatingFetcher(api_key) if api_key else FMPCreditRatingFetcher()
        self.model = None
        self.training_data = []
        self.X = None
        self.y = None
        self.is_trained = False
        
    def collect_training_data(self, 
                              companies: List[Dict] = None,
                              period: str = 'annual',
                              verbose: bool = True) -> pd.DataFrame:
        """
        Collect training data from FMP for companies with known ratings
        
        Parameters:
        -----------
        companies : List[Dict], optional
            List of {'ticker': str, 'rating': str, 'name': str}
            If None, uses TRAINING_COMPANIES
        period : str
            'annual' or 'quarter'
        verbose : bool
            Print progress
            
        Returns:
        --------
        DataFrame with training data
        """
        if companies is None:
            companies = TRAINING_COMPANIES
        
        if verbose:
            print("=" * 60)
            print("Collecting Training Data from FMP API")
            print("=" * 60)
            print(f"Companies to fetch: {len(companies)}")
        
        successful = 0
        failed = []
        
        for i, company in enumerate(companies):
            ticker = company['ticker']
            rating = company['rating']
            name = company.get('name', ticker)
            
            # Convert rating to numeric
            rating_num = RATING_MAP.get(rating)
            if rating_num is None:
                if verbose:
                    print(f"  [{i+1}/{len(companies)}] {ticker}: Skipped (no rating)")
                continue
            
            # Fetch features
            features = self.fetcher.fetch_company_features(ticker, period)
            
            if features:
                features['rating'] = rating
                features['rating_num'] = rating_num
                features['name'] = name
                self.training_data.append(features)
                successful += 1
                if verbose:
                    print(f"  [{i+1}/{len(companies)}] {ticker} ({rating}): OK")
            else:
                failed.append(ticker)
                if verbose:
                    print(f"  [{i+1}/{len(companies)}] {ticker}: FAILED")
        
        if verbose:
            print(f"\nResults: {successful}/{len(companies)} companies fetched")
            if failed:
                print(f"Failed: {failed}")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.training_data)
        
        if verbose and len(df) > 0:
            print(f"\nClass distribution:")
            for i, name in enumerate(RATING_NAMES):
                count = sum(df['rating_num'] == i)
                if count > 0:
                    print(f"  {name}: {count} ({count/len(df)*100:.1f}%)")
        
        return df
    
    def prepare_training_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert training data to numpy arrays"""
        if not self.training_data:
            raise ValueError("No training data. Call collect_training_data() first.")
        
        X_list = []
        y_list = []
        
        for item in self.training_data:
            features = [item[f] for f in FEATURE_NAMES]
            X_list.append(features)
            y_list.append(item['rating_num'])
        
        self.X = np.array(X_list)
        self.y = np.array(y_list)
        
        return self.X, self.y
    
    def train(self, 
              alpha: float = 1.0,
              balance: bool = True,
              regularization: float = 0.01,
              verbose: bool = True) -> 'CreditRatingTrainer':
        """
        Train the Ordinal Logistic Regression model
        
        Parameters:
        -----------
        alpha : float
            Cost function exponent
        balance : bool
            Use sample weights for class imbalance
        regularization : float
            L2 regularization strength
        verbose : bool
            Print training progress
        """
        if self.X is None:
            self.prepare_training_arrays()
        
        if verbose:
            print("\n" + "=" * 60)
            print("Training Ordinal Logistic Regression Model")
            print("=" * 60)
            print(f"Samples: {len(self.y)}, Features: {self.X.shape[1]}")
        
        # Create and train model
        self.model = OrdinalLogisticRegression(
            n_classes=8,
            alpha=alpha,
            balance=balance,
            regularization=regularization
        )
        
        self.model.fit(self.X, self.y)
        self.is_trained = True
        
        # Evaluate
        if verbose:
            scores = self.model.score(self.X, self.y)
            print(f"\nTraining Performance:")
            print(f"  Accuracy: {scores['accuracy']:.1%}")
            print(f"  MAE: {scores['mae']:.2f} notches")
            print(f"  Within +/-1 notch: {scores['within_1_notch']:.1%}")
            print(f"  Within +/-2 notches: {scores['within_2_notches']:.1%}")
            
            # Feature importance
            print(f"\nFeature Importance:")
            importance = self.model.get_feature_importance(FEATURE_NAMES)
            for name, imp in importance.items():
                print(f"  {name}: {imp:.3f}")
        
        return self
    
    def predict(self, 
                ticker: str = None,
                raw_data: Dict = None,
                period: str = 'annual') -> Dict:
        """
        Predict credit rating for a company
        
        Parameters:
        -----------
        ticker : str, optional
            Stock ticker (fetches data from FMP)
        raw_data : Dict, optional
            Raw financial data (from PDF extraction)
        period : str
            'annual' or 'quarter' (for FMP fetch)
            
        Returns:
        --------
        Dict with rating prediction
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get features
        if ticker:
            features = self.fetcher.fetch_company_features(ticker, period)
            if not features:
                raise ValueError(f"Could not fetch data for {ticker}")
            company_name = ticker
        elif raw_data:
            features = self.fetcher.compute_credit_features(raw_data)
            if not features:
                raise ValueError("Could not compute features from raw data")
            company_name = raw_data.get('name', 'Company')
        else:
            raise ValueError("Provide either ticker or raw_data")
        
        # Prepare feature array
        X = np.array([[features[f] for f in FEATURE_NAMES]])
        
        # Predict
        probs = self.model.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        rating = RATING_NAMES[pred_idx]
        confidence = probs[pred_idx]
        
        return {
            'company': company_name,
            'rating': rating,
            'rating_numeric': pred_idx,
            'confidence': float(confidence),
            'investment_grade': pred_idx <= 3,
            'probabilities': {
                RATING_NAMES[i]: float(p)
                for i, p in enumerate(probs) if p > 0.01
            },
            'features': features
        }
    
    def predict_from_pdf(self, pdf_data: Dict, company_name: str = "Company") -> Dict:
        """
        Predict rating from PDF extracted data
        
        Parameters:
        -----------
        pdf_data : Dict
            Financial data extracted from PDF
            Expected keys: total_debt, total_equity, total_assets,
                          current_assets, current_liabilities, revenue,
                          net_income, operating_income, interest_expense, ebitda
        company_name : str
            Company name for display
        """
        pdf_data['name'] = company_name
        return self.predict(raw_data=pdf_data)
    
    def print_report(self, result: Dict):
        """Print formatted rating report"""
        print("\n" + "=" * 60)
        print(f"CREDIT RATING REPORT: {result['company']}")
        print("=" * 60)
        
        rating = result['rating']
        confidence = result['confidence']
        ig = "Investment Grade" if result['investment_grade'] else "Speculative Grade"
        
        print(f"\n+{'-' * 40}+")
        print(f"|  Rating:      {rating:>12}             |")
        print(f"|  Confidence:  {confidence:>12.1%}             |")
        print(f"|  Grade:       {ig:>20}   |")
        print(f"+{'-' * 40}+")
        
        print(f"\nProbability Distribution:")
        for r, p in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
            bar_len = int(p * 30)
            bar = "#" * bar_len + "." * (30 - bar_len)
            print(f"  {r:<4} [{bar}] {p:.1%}")
        
        print(f"\nKey Ratios:")
        features = result.get('features', {})
        print(f"  Debt/Equity:       {features.get('debt_to_equity', 0):>10.2f}")
        print(f"  Interest Coverage: {features.get('interest_coverage', 0):>10.1f}x")
        print(f"  Current Ratio:     {features.get('current_ratio', 0):>10.2f}")
        print(f"  Net Margin:        {features.get('net_margin', 0):>10.1%}")
        print(f"  ROA:               {features.get('roa', 0):>10.1%}")
        print(f"  Debt/EBITDA:       {features.get('debt_to_ebitda', 0):>10.1f}x")
        
        print("=" * 60)
    
    def save(self, filepath: str):
        """Save trained model and training data"""
        import pickle
        
        data = {
            'model': self.model,
            'training_data': self.training_data,
            'X': self.X,
            'y': self.y,
            'feature_names': FEATURE_NAMES,
            'rating_names': RATING_NAMES,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained model"""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.training_data = data.get('training_data', [])
        self.X = data.get('X')
        self.y = data.get('y')
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
        print(f"Training samples: {len(self.training_data)}")
    
    def save_training_data_csv(self, filepath: str):
        """Save training data to CSV for inspection"""
        if not self.training_data:
            raise ValueError("No training data to save")
        
        df = pd.DataFrame(self.training_data)
        
        # Select relevant columns
        cols = ['ticker', 'name', 'rating', 'rating_num', 'date'] + FEATURE_NAMES
        cols = [c for c in cols if c in df.columns]
        
        df[cols].to_csv(filepath, index=False)
        print(f"Training data saved to {filepath}")
    
    def load_training_data_csv(self, filepath: str):
        """Load training data from CSV"""
        df = pd.read_csv(filepath)
        
        self.training_data = df.to_dict('records')
        print(f"Loaded {len(self.training_data)} companies from {filepath}")
        
        self.prepare_training_arrays()


# ============================================================
# Convenience Functions
# ============================================================

def quick_train_and_predict(ticker: str, api_key: str = None) -> Dict:
    """
    Quick function to train model and predict for a single company
    
    Usage:
    ------
    result = quick_train_and_predict('TSLA')
    print(result['rating'])
    """
    trainer = CreditRatingTrainer(api_key)
    
    # Use built-in training data (faster)
    print("Using built-in training data...")
    trainer.training_data = _get_builtin_training_data()
    trainer.prepare_training_arrays()
    trainer.train(verbose=False)
    
    # Predict
    result = trainer.predict(ticker=ticker)
    trainer.print_report(result)
    
    return result


def _get_builtin_training_data() -> List[Dict]:
    """Get built-in training data (no API calls needed)"""
    return [
        # AAA
        {'ticker': 'MSFT', 'name': 'Microsoft', 'rating': 'AAA', 'rating_num': 0,
         'debt_to_equity': 0.19, 'interest_coverage': 109.4, 'current_ratio': 1.27,
         'net_margin': 0.36, 'roa': 0.17, 'debt_to_ebitda': 0.47, 'log_assets': 11.71},
        {'ticker': 'JNJ', 'name': 'Johnson & Johnson', 'rating': 'AAA', 'rating_num': 0,
         'debt_to_equity': 0.44, 'interest_coverage': 29.8, 'current_ratio': 1.16,
         'net_margin': 0.21, 'roa': 0.10, 'debt_to_ebitda': 1.2, 'log_assets': 11.25},
        
        # AA
        {'ticker': 'AAPL', 'name': 'Apple', 'rating': 'AA+', 'rating_num': 1,
         'debt_to_equity': 1.79, 'interest_coverage': 29.1, 'current_ratio': 0.99,
         'net_margin': 0.25, 'roa': 0.27, 'debt_to_ebitda': 0.97, 'log_assets': 11.55},
        {'ticker': 'GOOGL', 'name': 'Alphabet', 'rating': 'AA+', 'rating_num': 1,
         'debt_to_equity': 0.05, 'interest_coverage': 200.0, 'current_ratio': 2.10,
         'net_margin': 0.29, 'roa': 0.22, 'debt_to_ebitda': 0.12, 'log_assets': 11.65},
        {'ticker': 'XOM', 'name': 'Exxon Mobil', 'rating': 'AA-', 'rating_num': 1,
         'debt_to_equity': 0.20, 'interest_coverage': 54.5, 'current_ratio': 1.43,
         'net_margin': 0.10, 'roa': 0.07, 'debt_to_ebitda': 0.58, 'log_assets': 11.66},
        
        # A
        {'ticker': 'KO', 'name': 'Coca-Cola', 'rating': 'A+', 'rating_num': 2,
         'debt_to_equity': 1.62, 'interest_coverage': 9.8, 'current_ratio': 1.13,
         'net_margin': 0.23, 'roa': 0.10, 'debt_to_ebitda': 2.1, 'log_assets': 10.95},
        {'ticker': 'PEP', 'name': 'PepsiCo', 'rating': 'A+', 'rating_num': 2,
         'debt_to_equity': 2.20, 'interest_coverage': 9.5, 'current_ratio': 0.83,
         'net_margin': 0.11, 'roa': 0.09, 'debt_to_ebitda': 2.3, 'log_assets': 11.01},
        {'ticker': 'INTC', 'name': 'Intel', 'rating': 'A', 'rating_num': 2,
         'debt_to_equity': 0.47, 'interest_coverage': 16.5, 'current_ratio': 1.57,
         'net_margin': 0.19, 'roa': 0.08, 'debt_to_ebitda': 1.4, 'log_assets': 11.18},
        {'ticker': 'NKE', 'name': 'Nike', 'rating': 'A+', 'rating_num': 2,
         'debt_to_equity': 0.62, 'interest_coverage': 22.1, 'current_ratio': 2.68,
         'net_margin': 0.10, 'roa': 0.13, 'debt_to_ebitda': 0.9, 'log_assets': 10.57},
        
        # BBB
        {'ticker': 'GM', 'name': 'General Motors', 'rating': 'BBB', 'rating_num': 3,
         'debt_to_equity': 1.66, 'interest_coverage': 13.4, 'current_ratio': 0.90,
         'net_margin': 0.06, 'roa': 0.04, 'debt_to_ebitda': 5.5, 'log_assets': 11.44},
        {'ticker': 'F', 'name': 'Ford', 'rating': 'BBB-', 'rating_num': 3,
         'debt_to_equity': 3.41, 'interest_coverage': 4.8, 'current_ratio': 1.16,
         'net_margin': 0.04, 'roa': 0.02, 'debt_to_ebitda': 6.2, 'log_assets': 11.41},
        {'ticker': 'T', 'name': 'AT&T', 'rating': 'BBB', 'rating_num': 3,
         'debt_to_equity': 1.12, 'interest_coverage': 4.2, 'current_ratio': 0.59,
         'net_margin': 0.08, 'roa': 0.04, 'debt_to_ebitda': 3.1, 'log_assets': 11.60},
        {'ticker': 'VZ', 'name': 'Verizon', 'rating': 'BBB+', 'rating_num': 3,
         'debt_to_equity': 1.35, 'interest_coverage': 5.1, 'current_ratio': 0.75,
         'net_margin': 0.09, 'roa': 0.05, 'debt_to_ebitda': 2.9, 'log_assets': 11.55},
        {'ticker': 'DAL', 'name': 'Delta Airlines', 'rating': 'BBB-', 'rating_num': 3,
         'debt_to_equity': 2.85, 'interest_coverage': 6.8, 'current_ratio': 0.42,
         'net_margin': 0.07, 'roa': 0.06, 'debt_to_ebitda': 2.4, 'log_assets': 10.84},
        
        # BB
        {'ticker': 'UAL', 'name': 'United Airlines', 'rating': 'BB', 'rating_num': 4,
         'debt_to_equity': 4.21, 'interest_coverage': 3.2, 'current_ratio': 0.74,
         'net_margin': 0.04, 'roa': 0.03, 'debt_to_ebitda': 3.8, 'log_assets': 10.82},
        {'ticker': 'RCL', 'name': 'Royal Caribbean', 'rating': 'BB+', 'rating_num': 4,
         'debt_to_equity': 3.58, 'interest_coverage': 2.8, 'current_ratio': 0.23,
         'net_margin': 0.12, 'roa': 0.04, 'debt_to_ebitda': 4.2, 'log_assets': 10.56},
        {'ticker': 'CCL', 'name': 'Carnival', 'rating': 'BB-', 'rating_num': 4,
         'debt_to_equity': 2.95, 'interest_coverage': 2.1, 'current_ratio': 0.31,
         'net_margin': 0.08, 'roa': 0.02, 'debt_to_ebitda': 5.1, 'log_assets': 10.75},
        
        # B
        {'ticker': 'AMC', 'name': 'AMC Entertainment', 'rating': 'CCC+', 'rating_num': 5,
         'debt_to_equity': 5.21, 'interest_coverage': 0.8, 'current_ratio': 0.45,
         'net_margin': -0.15, 'roa': -0.08, 'debt_to_ebitda': 8.5, 'log_assets': 10.02},
        {'ticker': 'GME', 'name': 'GameStop', 'rating': 'B', 'rating_num': 5,
         'debt_to_equity': 0.15, 'interest_coverage': 1.5, 'current_ratio': 1.85,
         'net_margin': -0.03, 'roa': -0.02, 'debt_to_ebitda': 6.0, 'log_assets': 9.54},
        {'ticker': 'WEWORK', 'name': 'WeWork', 'rating': 'B-', 'rating_num': 5,
         'debt_to_equity': 2.15, 'interest_coverage': 0.3, 'current_ratio': 0.62,
         'net_margin': -0.45, 'roa': -0.25, 'debt_to_ebitda': 12.0, 'log_assets': 10.25},
        
        # CCC
        {'ticker': 'HTZ', 'name': 'Hertz', 'rating': 'CCC', 'rating_num': 6,
         'debt_to_equity': 8.52, 'interest_coverage': 0.5, 'current_ratio': 0.85,
         'net_margin': -0.08, 'roa': -0.02, 'debt_to_ebitda': 9.5, 'log_assets': 10.42},
        {'ticker': 'REV', 'name': 'Revlon', 'rating': 'CCC', 'rating_num': 6,
         'debt_to_equity': 3.25, 'interest_coverage': 0.2, 'current_ratio': 0.72,
         'net_margin': -0.12, 'roa': -0.08, 'debt_to_ebitda': 15.0, 'log_assets': 9.45},
        {'ticker': 'EGRNF', 'name': 'Evergrande', 'rating': 'CCC', 'rating_num': 6,
         'debt_to_equity': 1.62, 'interest_coverage': 0.03, 'current_ratio': 0.96,
         'net_margin': -0.002, 'roa': -0.0002, 'debt_to_ebitda': 50.0, 'log_assets': 12.36},
        
        # D
        {'ticker': 'LEH', 'name': 'Lehman Brothers', 'rating': 'D', 'rating_num': 7,
         'debt_to_equity': 30.7, 'interest_coverage': -2.5, 'current_ratio': 0.15,
         'net_margin': -0.55, 'roa': -0.12, 'debt_to_ebitda': -50.0, 'log_assets': 11.85},
        {'ticker': 'ENRN', 'name': 'Enron', 'rating': 'D', 'rating_num': 7,
         'debt_to_equity': 4.85, 'interest_coverage': 0.1, 'current_ratio': 0.35,
         'net_margin': -0.85, 'roa': -0.15, 'debt_to_ebitda': 25.0, 'log_assets': 10.82},
        {'ticker': 'SIVB', 'name': 'SVB Financial', 'rating': 'D', 'rating_num': 7,
         'debt_to_equity': 12.5, 'interest_coverage': -1.2, 'current_ratio': 0.08,
         'net_margin': -0.25, 'roa': -0.02, 'debt_to_ebitda': -15.0, 'log_assets': 11.35},
    ]


# ============================================================
# Main - Example Usage
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Credit Rating Model - Complete Training & Testing Pipeline")
    print("=" * 70)
    
    # Create trainer
    trainer = CreditRatingTrainer()
    
    # Option 1: Use built-in training data (fast, no API needed)
    print("\n[1] Using built-in training data...")
    trainer.training_data = _get_builtin_training_data()
    trainer.prepare_training_arrays()
    
    # Option 2: Fetch from FMP (uncomment to use)
    # print("\n[1] Collecting training data from FMP API...")
    # trainer.collect_training_data()
    
    # Train model
    print("\n[2] Training model...")
    trainer.train()
    
    # Test predictions
    print("\n[3] Testing predictions...")
    
    # Test from built-in ticker (using FMP)
    # result = trainer.predict(ticker='TSLA')
    # trainer.print_report(result)
    
    # Test from raw data (simulating PDF extraction)
    print("\n--- Testing with raw data (PDF extraction simulation) ---")
    
    gm_pdf_data = {
        'name': 'General Motors',
        'total_debt': 120000,
        'total_equity': 72000,
        'total_assets': 273064,
        'current_assets': 90000,
        'current_liabilities': 100000,
        'revenue': 171842,
        'net_income': 10127,
        'operating_income': 14000,
        'interest_expense': 1044,
        'ebitda': 18000
    }
    
    result = trainer.predict_from_pdf(gm_pdf_data, "General Motors (from PDF)")
    trainer.print_report(result)
    
    # Test Evergrande
    evergrande_pdf = {
        'name': 'Evergrande',
        'total_debt': 571800,
        'total_equity': 352084,
        'total_assets': 2301221,
        'current_assets': 1656577,
        'current_liabilities': 1733840,
        'revenue': 250709,
        'net_income': -476,
        'operating_income': 1500,
        'interest_expense': 57800,
        'ebitda': 28500
    }
    
    result = trainer.predict_from_pdf(evergrande_pdf, "Evergrande (from PDF)")
    trainer.print_report(result)
    
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)