"""
Credit Rating Pipeline
======================

Complete credit rating workflow integrating:
1. Ordinal Logistic Regression
2. Cost-Sensitive Learning
3. Fraud Detection
4. Cross-Validation

Author: Lihao Xiao
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold

from .ordinal_lr import OrdinalLogisticRegression
from .fraud_detector import FraudDetector, AltmanZScore


# Rating mapping
RATING_NAMES = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']
RATING_TO_NUM = {name: i for i, name in enumerate(RATING_NAMES)}

# Default features for credit rating
DEFAULT_FEATURES = [
    'debt_to_equity',
    'interest_coverage', 
    'current_ratio',
    'net_margin',
    'roa',
    'debt_to_ebitda',
    'log_assets'
]


class CreditRatingPipeline:
    """
    Complete Credit Rating Pipeline
    
    Usage:
    ------
    # 1. Create and train
    pipeline = CreditRatingPipeline()
    pipeline.train()  # Use built-in data
    # or
    pipeline.train(X, y)  # Use your own data
    
    # 2. Predict
    result = pipeline.predict_company({
        'debt_to_equity': 1.66,
        'interest_coverage': 13.4,
        ...
    })
    
    # 3. Print report
    pipeline.print_report("GM", result)
    """
    
    def __init__(self, n_classes: int = 8):
        self.n_classes = n_classes
        self.model = None
        self.feature_names = DEFAULT_FEATURES
        self.is_trained = False
        
    def _get_sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Built-in sample training data"""
        data = [
            # [debt_to_equity, interest_coverage, current_ratio, net_margin, roa, debt_to_ebitda, log_assets, rating]
            # AAA
            [0.19, 109.4, 1.27, 0.36, 0.17, 0.47, 11.71, 0],  # Microsoft
            [0.44, 29.8, 1.16, 0.21, 0.10, 1.2, 11.25, 0],    # J&J
            # AA
            [1.79, 29.1, 0.99, 0.25, 0.27, 0.97, 11.55, 1],   # Apple
            [0.05, 200.0, 2.10, 0.29, 0.22, 0.12, 11.65, 1],  # Google
            [0.20, 54.5, 1.43, 0.10, 0.07, 0.58, 11.66, 1],   # Exxon
            # A
            [1.62, 9.8, 1.13, 0.23, 0.10, 2.1, 10.95, 2],     # Coca-Cola
            [2.20, 9.5, 0.83, 0.11, 0.09, 2.3, 11.01, 2],     # PepsiCo
            [0.47, 16.5, 1.57, 0.19, 0.08, 1.4, 11.18, 2],    # Intel
            [0.62, 22.1, 2.68, 0.10, 0.13, 0.9, 10.57, 2],    # Nike
            # BBB
            [1.66, 13.4, 0.90, 0.06, 0.04, 5.5, 11.44, 3],    # GM
            [3.41, 4.8, 1.16, 0.04, 0.02, 6.2, 11.41, 3],     # Ford
            [1.12, 4.2, 0.59, 0.08, 0.04, 3.1, 11.60, 3],     # AT&T
            [1.35, 5.1, 0.75, 0.09, 0.05, 2.9, 11.55, 3],     # Verizon
            [2.85, 6.8, 0.42, 0.07, 0.06, 2.4, 10.84, 3],     # Delta
            # BB
            [4.21, 3.2, 0.74, 0.04, 0.03, 3.8, 10.82, 4],     # United Airlines
            [3.58, 2.8, 0.23, 0.12, 0.04, 4.2, 10.56, 4],     # Royal Caribbean
            [2.95, 2.1, 0.31, 0.08, 0.02, 5.1, 10.75, 4],     # Carnival
            # B
            [5.21, 0.8, 0.45, -0.15, -0.08, 8.5, 10.02, 5],   # AMC
            [0.15, 1.5, 1.85, -0.03, -0.02, 6.0, 9.54, 5],    # GameStop
            [2.15, 0.3, 0.62, -0.45, -0.25, 12.0, 10.25, 5],  # WeWork
            # CCC
            [8.52, 0.5, 0.85, -0.08, -0.02, 9.5, 10.42, 6],   # Hertz
            [3.25, 0.2, 0.72, -0.12, -0.08, 15.0, 9.45, 6],   # Revlon
            [1.62, 0.03, 0.96, -0.002, -0.0002, 50.0, 12.36, 6],  # Evergrande
            # D
            [30.7, -2.5, 0.15, -0.55, -0.12, -50.0, 11.85, 7],  # Lehman Brothers
            [4.85, 0.1, 0.35, -0.85, -0.15, 25.0, 10.82, 7],    # Enron
            [12.5, -1.2, 0.08, -0.25, -0.02, -15.0, 11.35, 7],  # SVB
        ]
        arr = np.array(data)
        return arr[:, :-1], arr[:, -1].astype(int)
    
    def train(self, X: np.ndarray = None, y: np.ndarray = None,
              alpha: float = 1.0, balance: bool = True,
              regularization: float = 0.01, verbose: bool = True) -> 'CreditRatingPipeline':
        """
        Train the model
        
        Parameters:
        -----------
        X : np.ndarray, optional
            Feature matrix. If not provided, uses built-in data.
        y : np.ndarray, optional
            Labels. If not provided, uses built-in data.
        alpha : float
            Cost function exponent for cost-sensitive learning
        balance : bool
            Whether to use sample weights for class imbalance
        regularization : float
            L2 regularization strength
        verbose : bool
            Whether to print training progress
        """
        if X is None or y is None:
            X, y = self._get_sample_data()
        
        if verbose:
            print("=" * 60)
            print("Training Credit Rating Model (Ordinal LR + Cost-Sensitive)")
            print("=" * 60)
            print(f"\nSamples: {len(y)}, Features: {X.shape[1]}")
            print("Class distribution:")
            for i, name in enumerate(RATING_NAMES):
                count = sum(y == i)
                if count > 0:
                    print(f"  {name}: {count} ({count/len(y)*100:.1f}%)")
        
        # Create model
        self.model = OrdinalLogisticRegression(
            n_classes=self.n_classes,
            alpha=alpha,
            balance=balance,
            regularization=regularization
        )
        
        # Train
        self.model.fit(X, y)
        self.is_trained = True
        
        # Evaluate
        if verbose:
            scores = self.model.score(X, y)
            print(f"\nTraining performance:")
            print(f"  Accuracy: {scores['accuracy']:.1%}")
            print(f"  MAE: {scores['mae']:.2f} notches")
            print(f"  Within +/-1 notch: {scores['within_1_notch']:.1%}")
            print(f"  Within +/-2 notches: {scores['within_2_notches']:.1%}")
        
        return self
    
    def cross_validate(self, X: np.ndarray = None, y: np.ndarray = None,
                       n_folds: int = 5) -> Dict[str, float]:
        """Perform k-fold cross-validation"""
        if X is None or y is None:
            X, y = self._get_sample_data()
        
        print(f"\n{n_folds}-Fold Cross-Validation...")
        
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        accuracies, maes = [], []
        
        for train_idx, val_idx in kf.split(X, y):
            model = OrdinalLogisticRegression(n_classes=self.n_classes)
            model.fit(X[train_idx], y[train_idx])
            scores = model.score(X[val_idx], y[val_idx])
            accuracies.append(scores['accuracy'])
            maes.append(scores['mae'])
        
        results = {
            'cv_accuracy': f"{np.mean(accuracies):.1%} +/- {np.std(accuracies):.1%}",
            'cv_mae': f"{np.mean(maes):.2f} +/- {np.std(maes):.2f}"
        }
        
        print(f"  CV Accuracy: {results['cv_accuracy']}")
        print(f"  CV MAE: {results['cv_mae']}")
        
        return results
    
    def predict_company(self, financials: Dict) -> Dict:
        """
        Predict credit rating for a single company
        
        Parameters:
        -----------
        financials : dict
            Dictionary containing financial metrics
        
        Returns:
        --------
        dict with rating, confidence, probabilities, etc.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features
        X = np.array([[
            financials.get('debt_to_equity', 1.0),
            financials.get('interest_coverage', 5.0),
            financials.get('current_ratio', 1.0),
            financials.get('net_margin', 0.05),
            financials.get('roa', 0.05),
            financials.get('debt_to_ebitda', 3.0),
            financials.get('log_assets', 10.0)
        ]])
        
        # Predict
        probs = self.model.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        rating = RATING_NAMES[pred_idx]
        confidence = probs[pred_idx]
        
        # Investment grade status
        investment_grade = pred_idx <= 3  # AAA, AA, A, BBB
        
        return {
            'rating': rating,
            'rating_numeric': pred_idx,
            'confidence': float(confidence),
            'investment_grade': investment_grade,
            'probabilities': {
                RATING_NAMES[i]: float(p) 
                for i, p in enumerate(probs) if p > 0.01
            }
        }
    
    def predict_with_fraud_check(self, financials: Dict, company_name: str = "Company") -> Dict:
        """Predict rating with fraud detection"""
        result = self.predict_company(financials)
        
        # Add Altman Z-Score
        result['altman_z'] = AltmanZScore.calculate(financials)
        
        # Add red flags
        result['red_flags'] = FraudDetector.check_red_flags(financials)
        
        return result
    
    def print_report(self, company_name: str, result: Dict):
        """Print formatted rating report"""
        print("\n" + "=" * 60)
        print(f"CREDIT RATING REPORT: {company_name}")
        print("=" * 60)
        
        rating = result['rating']
        confidence = result['confidence']
        ig = "Investment Grade" if result['investment_grade'] else "Speculative Grade"
        
        print(f"\n+{'-' * 35}+")
        print(f"|  Rating:     {rating:>10}            |")
        print(f"|  Confidence: {confidence:>10.1%}            |")
        print(f"|  Grade:      {ig:>18} |")
        print(f"+{'-' * 35}+")
        
        print(f"\nProbability Distribution:")
        for r, p in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
            bar_len = int(p * 30)
            bar = "#" * bar_len + "." * (30 - bar_len)
            print(f"  {r:<4} [{bar}] {p:.1%}")
        
        # If there are red flags
        if 'red_flags' in result and result['red_flags']:
            print(f"\nRed Flags ({len(result['red_flags'])}):")
            for flag in result['red_flags']:
                print(f"  - {flag}")
        
        print("=" * 60)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        return self.model.get_feature_importance(self.feature_names)
    
    def save_model(self, filepath: str):
        """Save model to file (requires joblib)"""
        import joblib
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'n_classes': self.n_classes
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        import joblib
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.n_classes = data['n_classes']
        self.is_trained = True
        print(f"Model loaded from {filepath}")


# ============================================================
# Convenience Functions
# ============================================================

def rate_company(financials: Dict, verbose: bool = True) -> Dict:
    """
    Quick rating function
    
    Usage:
    ------
    from credit_rating import rate_company
    
    result = rate_company({
        'debt_to_equity': 1.66,
        'interest_coverage': 13.4,
        'current_ratio': 0.90,
        'net_margin': 0.06,
        'roa': 0.04,
        'debt_to_ebitda': 5.5,
        'log_assets': 11.44
    })
    print(result['rating'])  # 'BBB'
    """
    pipeline = CreditRatingPipeline()
    pipeline.train(verbose=verbose)
    return pipeline.predict_company(financials)