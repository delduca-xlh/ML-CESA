"""
Credit Rating System
====================

Simple interface for credit rating prediction and fraud detection.

For users: Just call rate_company() with a ticker or PDF data.

Usage:
------
# From ticker (uses FMP API)
from credit_rating import rate_company
result = rate_company("TSLA")

# From PDF data
result = rate_company(pdf_data={"revenue": ..., "total_assets": ...}, name="Evergrande")

# With fraud detection
result = rate_company("GM", check_fraud=True)

Author: Lihao Xiao
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from dataclasses import dataclass

# Import components
try:
    from .ordinal_lr import OrdinalLogisticRegression
    from .fraud_detector import FraudDetector, AltmanZScore
    from .trainer import FMPCreditRatingFetcher, RATING_MAP, FEATURE_NAMES
except ImportError:
    from ordinal_lr import OrdinalLogisticRegression
    from fraud_detector import FraudDetector, AltmanZScore
    from trainer import FMPCreditRatingFetcher, RATING_MAP, FEATURE_NAMES

# Rating names
RATING_NAMES = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']


@dataclass
class CreditRatingResult:
    """Credit rating prediction result"""
    company: str
    rating: str
    rating_numeric: int
    confidence: float
    investment_grade: bool
    probabilities: Dict[str, float]
    features: Dict[str, float]
    
    # Fraud detection results (optional)
    red_flags: list = None
    altman_z: Dict = None
    beneish_m: Dict = None
    fraud_risk: str = None
    
    def __repr__(self):
        ig = "Investment Grade" if self.investment_grade else "Speculative Grade"
        return f"CreditRating({self.company}: {self.rating}, {self.confidence:.1%}, {ig})"
    
    def to_dict(self):
        return {
            'company': self.company,
            'rating': self.rating,
            'rating_numeric': self.rating_numeric,
            'confidence': self.confidence,
            'investment_grade': self.investment_grade,
            'probabilities': self.probabilities,
            'features': self.features,
            'red_flags': self.red_flags,
            'altman_z': self.altman_z,
            'fraud_risk': self.fraud_risk
        }


class CreditRatingSystem:
    """
    Complete Credit Rating System
    
    Features:
    - Load pre-trained model
    - Predict from ticker (FMP API) or raw data (PDF)
    - Fraud detection (Red Flags, Altman Z-Score, Beneish M-Score)
    
    Usage:
    ------
    system = CreditRatingSystem()
    system.load_model('data/credit_rating_model.pkl')
    
    # Or train from saved data
    system.train_from_csv('data/credit_rating_training_data.csv')
    
    # Predict
    result = system.predict(ticker='GM')
    result = system.predict(raw_data={...}, name='Evergrande')
    """
    
    def __init__(self, api_key: str = None):
        self.model = None
        self.fetcher = FMPCreditRatingFetcher(api_key) if api_key else FMPCreditRatingFetcher()
        self.is_loaded = False
        
    def load_model(self, filepath: str):
        """Load pre-trained model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.is_loaded = True
        
        print(f"Model loaded from {filepath}")
        if 'training_samples' in data:
            print(f"Training samples: {data['training_samples']}")
    
    def save_model(self, filepath: str, training_samples: int = 0):
        """Save trained model to file"""
        data = {
            'model': self.model,
            'feature_names': FEATURE_NAMES,
            'rating_names': RATING_NAMES,
            'training_samples': training_samples
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Model saved to {filepath}")
    
    def train_from_csv(self, filepath: str, verbose: bool = True):
        """
        Train model from pre-saved CSV data
        
        Parameters:
        -----------
        filepath : str
            Path to training data CSV (from fetch_training_data.py)
        """
        if verbose:
            print(f"Loading training data from {filepath}...")
        
        df = pd.read_csv(filepath)
        
        # Extract features and labels
        X = df[FEATURE_NAMES].values
        y = df['rating_num'].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        if verbose:
            print(f"Training samples: {len(y)}")
            print(f"Features: {len(FEATURE_NAMES)}")
            print("\nRating distribution:")
            for i, name in enumerate(RATING_NAMES):
                count = sum(y == i)
                if count > 0:
                    print(f"  {name}: {count} ({count/len(y)*100:.1f}%)")
        
        # Train model
        self.model = OrdinalLogisticRegression(
            n_classes=8,
            alpha=1.0,
            balance=True,
            regularization=0.01
        )
        
        self.model.fit(X, y)
        self.is_loaded = True
        
        # Evaluate
        if verbose:
            scores = self.model.score(X, y)
            print(f"\nTraining Performance:")
            print(f"  Accuracy: {scores['accuracy']:.1%}")
            print(f"  MAE: {scores['mae']:.2f} notches")
            print(f"  Within +/-1 notch: {scores['within_1_notch']:.1%}")
            print(f"  Within +/-2 notches: {scores['within_2_notches']:.1%}")
            
            print(f"\nFeature Importance:")
            importance = self.model.get_feature_importance(FEATURE_NAMES)
            for name, imp in list(importance.items())[:5]:
                print(f"  {name}: {imp:.3f}")
        
        return len(y)
    
    def _compute_features(self, raw_data: Dict) -> Dict[str, float]:
        """Compute credit rating features from raw financial data"""
        return self.fetcher.compute_credit_features(raw_data)
    
    def predict(self, 
                ticker: str = None,
                raw_data: Dict = None,
                name: str = None,
                check_fraud: bool = True,
                period: str = 'annual') -> CreditRatingResult:
        """
        Predict credit rating for a company
        
        Parameters:
        -----------
        ticker : str, optional
            Stock ticker (fetches data from FMP API)
        raw_data : Dict, optional
            Raw financial data (from PDF extraction)
        name : str, optional
            Company name (for display)
        check_fraud : bool
            Whether to run fraud detection
        period : str
            'annual' or 'quarter' (for FMP fetch)
            
        Returns:
        --------
        CreditRatingResult with rating and fraud detection
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() or train_from_csv() first.")
        
        # Get features
        if ticker:
            features_dict = self.fetcher.fetch_company_features(ticker, period)
            if not features_dict:
                raise ValueError(f"Could not fetch data for {ticker}")
            company_name = name or ticker
            raw = features_dict.get('raw_data', {})
        elif raw_data:
            features_dict = self._compute_features(raw_data)
            company_name = name or "Company"
            raw = raw_data
        else:
            raise ValueError("Provide either ticker or raw_data")
        
        # Prepare feature array
        X = np.array([[features_dict.get(f, 0) for f in FEATURE_NAMES]])
        X = np.nan_to_num(X, nan=0.0)
        
        # Predict
        probs = self.model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        rating = RATING_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        
        # Create result
        result = CreditRatingResult(
            company=company_name,
            rating=rating,
            rating_numeric=pred_idx,
            confidence=confidence,
            investment_grade=(pred_idx <= 3),
            probabilities={
                RATING_NAMES[i]: float(p)
                for i, p in enumerate(probs) if p > 0.01
            },
            features={f: features_dict.get(f, 0) for f in FEATURE_NAMES}
        )
        
        # Fraud detection
        if check_fraud:
            result.red_flags = FraudDetector.check_red_flags(raw)
            result.altman_z = AltmanZScore.calculate(raw)
            
            # Overall fraud risk
            severe_flags = sum(1 for f in result.red_flags if 'NEGATIVE' in f or 'CRISIS' in f or 'CONCERN' in f)
            if severe_flags >= 3:
                result.fraud_risk = "CRITICAL"
            elif severe_flags >= 1:
                result.fraud_risk = "HIGH"
            elif len(result.red_flags) >= 2:
                result.fraud_risk = "MODERATE"
            else:
                result.fraud_risk = "LOW"
        
        return result
    
    def print_report(self, result: CreditRatingResult, detailed: bool = True):
        """Print formatted credit rating report"""
        print("\n" + "=" * 70)
        print(f"CREDIT RATING REPORT: {result.company}")
        print("=" * 70)
        
        # Rating box
        ig = "Investment Grade" if result.investment_grade else "Speculative Grade"
        print(f"\n┌{'─' * 45}┐")
        print(f"│  Rating:       {result.rating:>12}                   │")
        print(f"│  Confidence:   {result.confidence:>12.1%}                   │")
        print(f"│  Grade:        {ig:>24}   │")
        print(f"└{'─' * 45}┘")
        
        # Probability distribution
        print(f"\nProbability Distribution:")
        for r, p in sorted(result.probabilities.items(), key=lambda x: -x[1]):
            bar_len = int(p * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"  {r:<4} [{bar}] {p:>6.1%}")
        
        if detailed:
            # Key financial ratios
            print(f"\nKey Financial Ratios:")
            print(f"  {'Debt/Equity:':<25} {result.features.get('debt_to_equity', 0):>10.2f}")
            print(f"  {'Interest Coverage:':<25} {result.features.get('interest_coverage', 0):>10.1f}x")
            print(f"  {'Current Ratio:':<25} {result.features.get('current_ratio', 0):>10.2f}")
            print(f"  {'Net Margin:':<25} {result.features.get('net_margin', 0):>10.1%}")
            print(f"  {'ROA:':<25} {result.features.get('roa', 0):>10.1%}")
            print(f"  {'Debt/EBITDA:':<25} {result.features.get('debt_to_ebitda', 0):>10.1f}x")
        
        # Fraud detection
        if result.red_flags is not None:
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
            print(f"\nOverall Risk: {risk_emoji.get(result.fraud_risk, '?')} {result.fraud_risk}")
            
            # Altman Z-Score
            if result.altman_z:
                z = result.altman_z
                zone_emoji = {'SAFE': '🟢', 'GREY': '🟡', 'DISTRESS': '🔴'}
                print(f"\nAltman Z-Score: {z['z_score']:.2f} ({zone_emoji.get(z['zone'], '')} {z['zone']})")
                print(f"  {z['risk_assessment']}")
            
            # Red flags
            if result.red_flags:
                print(f"\n⚠️  {len(result.red_flags)} Warning Signal(s) Detected:")
                for flag in result.red_flags:
                    print(f"  • {flag}")
            else:
                print(f"\n✅ No major warning signals detected")
        
        print("\n" + "=" * 70)


# ============================================================
# Simple Interface Functions
# ============================================================

# Global system instance
_system = None

def _get_system():
    """Get or create global system instance"""
    global _system
    if _system is None:
        _system = CreditRatingSystem()
        
        # Try to load pre-trained model
        model_paths = [
            'data/credit_rating_model.pkl',
            '../data/credit_rating_model.pkl',
            'credit_rating_model.pkl'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                _system.load_model(path)
                return _system
        
        # Try to train from CSV
        csv_paths = [
            'data/credit_rating_training_data.csv',
            '../data/credit_rating_training_data.csv'
        ]
        
        for path in csv_paths:
            if os.path.exists(path):
                _system.train_from_csv(path)
                return _system
        
        # Use built-in data as fallback
        print("No pre-trained model found. Using built-in training data...")
        _train_with_builtin_data(_system)
    
    return _system


def _train_with_builtin_data(system):
    """Train with built-in data (26 companies)"""
    from .trainer import _get_builtin_training_data
    
    data = _get_builtin_training_data()
    
    X = np.array([[d[f] for f in FEATURE_NAMES] for d in data])
    y = np.array([d['rating_num'] for d in data])
    
    system.model = OrdinalLogisticRegression(n_classes=8, alpha=1.0, balance=True)
    system.model.fit(X, y)
    system.is_loaded = True
    
    print(f"Trained on {len(data)} built-in samples")


def rate_company(
    ticker: str = None,
    pdf_data: Dict = None,
    name: str = None,
    check_fraud: bool = True,
    print_report: bool = True
) -> CreditRatingResult:
    """
    Rate a company's credit - Simple interface
    
    Parameters:
    -----------
    ticker : str, optional
        Stock ticker symbol (e.g., 'TSLA', 'GM')
        Uses FMP API to fetch financial data
    pdf_data : Dict, optional
        Raw financial data from PDF extraction
        Required keys: revenue, total_assets, total_debt, etc.
    name : str, optional
        Company name (for display)
    check_fraud : bool
        Whether to run fraud detection (default: True)
    print_report : bool
        Whether to print formatted report (default: True)
        
    Returns:
    --------
    CreditRatingResult with rating and fraud detection
    
    Examples:
    ---------
    # From ticker
    result = rate_company("GM")
    
    # From PDF data
    result = rate_company(pdf_data={
        'revenue': 250709,
        'net_income': -476,
        'total_assets': 2301221,
        'total_equity': 352084,
        'total_debt': 571800,
        'current_assets': 1656577,
        'current_liabilities': 1733840,
        'operating_income': 1500,
        'interest_expense': 57800,
        'ebitda': 28500
    }, name="Evergrande")
    
    # Just get result without printing
    result = rate_company("AAPL", print_report=False)
    print(result.rating)  # 'AA'
    """
    system = _get_system()
    
    result = system.predict(
        ticker=ticker,
        raw_data=pdf_data,
        name=name,
        check_fraud=check_fraud
    )
    
    if print_report:
        system.print_report(result)
    
    return result


def detect_fraud(
    ticker: str = None,
    pdf_data: Dict = None,
    name: str = None,
    print_report: bool = True
) -> Dict:
    """
    Run fraud detection on a company
    
    Parameters:
    -----------
    ticker : str, optional
        Stock ticker symbol
    pdf_data : Dict, optional
        Raw financial data from PDF
    name : str, optional
        Company name
    print_report : bool
        Whether to print report
        
    Returns:
    --------
    Dict with fraud detection results
    """
    system = _get_system()
    
    # Get raw data
    if ticker:
        features = system.fetcher.fetch_company_features(ticker)
        if not features:
            raise ValueError(f"Could not fetch data for {ticker}")
        raw = features.get('raw_data', {})
        company_name = name or ticker
    elif pdf_data:
        raw = pdf_data
        company_name = name or "Company"
    else:
        raise ValueError("Provide either ticker or pdf_data")
    
    # Run fraud detection
    red_flags = FraudDetector.check_red_flags(raw)
    altman_z = AltmanZScore.calculate(raw)
    
    # Risk level
    severe_flags = sum(1 for f in red_flags if 'NEGATIVE' in f or 'CRISIS' in f or 'CONCERN' in f)
    if severe_flags >= 3:
        risk = "CRITICAL"
    elif severe_flags >= 1:
        risk = "HIGH"
    elif len(red_flags) >= 2:
        risk = "MODERATE"
    else:
        risk = "LOW"
    
    result = {
        'company': company_name,
        'risk_level': risk,
        'red_flags': red_flags,
        'altman_z': altman_z
    }
    
    if print_report:
        print(FraudDetector.generate_report(raw, company_name))
    
    return result