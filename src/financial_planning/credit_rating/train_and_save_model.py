"""
Train and Save Credit Rating Model
===================================

This script trains the credit rating model from saved training data
and saves the model for production use.

Run after fetch_training_data.py to create the model file.

Usage:
------
python train_and_save_model.py

Input:
------
- data/credit_rating_training_data.csv

Output:
------
- data/credit_rating_model.pkl

Author: Lihao Xiao
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ordinal_lr import OrdinalLogisticRegression


FEATURE_NAMES = [
    'debt_to_equity',
    'interest_coverage',
    'current_ratio',
    'net_margin',
    'roa',
    'debt_to_ebitda',
    'log_assets'
]

RATING_NAMES = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']


def train_and_save_model(
    data_path: str = "data/credit_rating_training_data.csv",
    model_path: str = "data/credit_rating_model.pkl",
    alpha: float = 1.0,
    balance: bool = True,
    regularization: float = 0.01
):
    """
    Train credit rating model and save to file.
    
    Parameters:
    -----------
    data_path : str
        Path to training data CSV
    model_path : str
        Path to save model
    alpha : float
        Cost function exponent
    balance : bool
        Use sample weights for class imbalance
    regularization : float
        L2 regularization strength
    """
    print("=" * 70)
    print("Training Credit Rating Model")
    print("=" * 70)
    
    # Load data
    if not os.path.exists(data_path):
        print(f"ERROR: Training data not found at {data_path}")
        print("Run fetch_training_data.py first to collect data.")
        return
    
    print(f"\nLoading training data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Total samples: {len(df)}")
    
    # Check for required columns
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return
    
    # Extract features and labels
    X = df[FEATURE_NAMES].values
    y = df['rating_num'].values
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    
    # Print distribution
    print("\nRating distribution:")
    for i, name in enumerate(RATING_NAMES):
        count = sum(y == i)
        if count > 0:
            pct = count / len(y) * 100
            bar = "#" * int(pct / 2)
            print(f"  {name:>4}: {count:>4} ({pct:>5.1f}%) {bar}")
    
    # Train model
    print("\nTraining Ordinal Logistic Regression...")
    print(f"  Alpha: {alpha}")
    print(f"  Balance: {balance}")
    print(f"  Regularization: {regularization}")
    
    model = OrdinalLogisticRegression(
        n_classes=8,
        alpha=alpha,
        balance=balance,
        regularization=regularization
    )
    
    model.fit(X, y)
    
    # Evaluate
    scores = model.score(X, y)
    print(f"\nTraining Performance:")
    print(f"  Accuracy:            {scores['accuracy']:>10.1%}")
    print(f"  MAE:                 {scores['mae']:>10.2f} notches")
    print(f"  Within +/-1 notch:   {scores['within_1_notch']:>10.1%}")
    print(f"  Within +/-2 notches: {scores['within_2_notches']:>10.1%}")
    
    # Feature importance
    print(f"\nFeature Importance (coefficient magnitude):")
    importance = model.get_feature_importance(FEATURE_NAMES)
    for name, imp in importance.items():
        direction = "↑ worse rating" if model.coef[FEATURE_NAMES.index(name)] > 0 else "↓ better rating"
        print(f"  {name:<20}: {imp:>8.3f}  ({direction})")
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_names': FEATURE_NAMES,
        'rating_names': RATING_NAMES,
        'training_samples': len(y),
        'training_accuracy': scores['accuracy'],
        'training_mae': scores['mae'],
        'created': datetime.now().isoformat(),
        'parameters': {
            'alpha': alpha,
            'balance': balance,
            'regularization': regularization
        }
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {model_path}")
    
    # Test predictions
    print("\n" + "=" * 70)
    print("Sample Predictions (Training Data)")
    print("=" * 70)
    
    y_pred = model.predict(X)
    
    # Show some examples
    sample_idx = [0, len(df)//4, len(df)//2, 3*len(df)//4, -1]
    
    print(f"\n{'Company':<20} {'Actual':<8} {'Predicted':<10} {'Correct':<8}")
    print("-" * 50)
    
    for idx in sample_idx:
        if idx < 0:
            idx = len(df) + idx
        row = df.iloc[idx]
        actual = RATING_NAMES[int(row['rating_num'])]
        predicted = RATING_NAMES[y_pred[idx]]
        correct = "✓" if actual == predicted else f"(off by {abs(int(row['rating_num']) - y_pred[idx])})"
        print(f"{row['ticker']:<20} {actual:<8} {predicted:<10} {correct}")
    
    print("\n" + "=" * 70)
    print("DONE! Model is ready for use.")
    print("=" * 70)
    print("\nUsage:")
    print("  from credit_rating import rate_company")
    print("  result = rate_company('TSLA')")
    print("  result = rate_company(pdf_data={...}, name='Evergrande')")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and save credit rating model')
    parser.add_argument('--data', default='data/credit_rating_training_data.csv',
                        help='Path to training data CSV')
    parser.add_argument('--output', default='data/credit_rating_model.pkl',
                        help='Path to save model')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Cost function exponent')
    parser.add_argument('--no-balance', action='store_true',
                        help='Disable sample weighting')
    parser.add_argument('--reg', type=float, default=0.01,
                        help='L2 regularization')
    
    args = parser.parse_args()
    
    train_and_save_model(
        data_path=args.data,
        model_path=args.output,
        alpha=args.alpha,
        balance=not args.no_balance,
        regularization=args.reg
    )