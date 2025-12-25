"""
Ordinal Logistic Regression for Credit Rating
==============================================

Mathematical Form:
P(Y <= j | X) = sigmoid(theta_j - beta'X)

Advantages:
1. Utilizes rating order information (AAA > AA > A > ...)
2. Fewer parameters (p + K-1), less prone to overfitting
3. Reasonable prediction distribution (unimodal)

Author: Lihao Xiao
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import Counter
from typing import Dict, List, Tuple, Optional


class OrdinalLogisticRegression:
    """
    Ordinal Logistic Regression with Cost-Sensitive Learning
    
    Parameters:
    -----------
    n_classes : int
        Number of rating categories (default 8: AAA, AA, A, BBB, BB, B, CCC, D)
    alpha : float
        Cost function exponent C(y,j) = |y-j|^alpha
    balance : bool
        Whether to use sample weights for class imbalance
    regularization : float
        L2 regularization strength
    """
    
    def __init__(self, 
                 n_classes: int = 8,
                 alpha: float = 1.0,
                 balance: bool = True,
                 regularization: float = 0.01):
        self.n_classes = n_classes
        self.alpha = alpha
        self.balance = balance
        self.regularization = regularization
        
        self.thresholds = None  # theta_j
        self.coef = None        # beta
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Logistic function with numerical stability."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute balanced sample weights: w_i = n / (K * n_class_i)"""
        counter = Counter(y)
        n_samples = len(y)
        n_classes = len(counter)
        
        weights = np.array([
            n_samples / (n_classes * counter[yi]) 
            for yi in y
        ])
        return weights / weights.sum() * len(weights)
    
    def _cost_matrix(self) -> np.ndarray:
        """Build cost matrix C[i,j] = |i-j|^alpha"""
        n = self.n_classes
        C = np.abs(np.arange(n).reshape(-1, 1) - np.arange(n).reshape(1, -1))
        return np.power(C.astype(float), self.alpha)
    
    def _cumulative_probs(self, X: np.ndarray) -> np.ndarray:
        """Compute cumulative probabilities P(Y <= j)"""
        linear = X @ self.coef
        cum_probs = np.zeros((X.shape[0], self.n_classes))
        for j in range(self.n_classes - 1):
            cum_probs[:, j] = self._sigmoid(self.thresholds[j] - linear)
        cum_probs[:, -1] = 1.0
        return cum_probs
    
    def _class_probs(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities P(Y = j) = P(Y <= j) - P(Y <= j-1)"""
        cum_probs = self._cumulative_probs(X)
        probs = np.zeros_like(cum_probs)
        probs[:, 0] = cum_probs[:, 0]
        for j in range(1, self.n_classes):
            probs[:, j] = cum_probs[:, j] - cum_probs[:, j-1]
        return np.clip(probs, 1e-10, 1.0)
    
    def _loss_function(self, params: np.ndarray, X: np.ndarray, 
                       y: np.ndarray, weights: np.ndarray) -> float:
        """
        Cost-Sensitive loss function with L2 regularization
        
        L = sum_i w_i * sum_j C(y_i, j) * P(j | X_i) + lambda * ||beta||^2
        """
        self.thresholds = params[:self.n_classes-1]
        self.coef = params[self.n_classes-1:]
        
        probs = self._class_probs(X)
        C = self._cost_matrix()
        
        # Cost-sensitive loss
        total_loss = 0
        for i in range(len(y)):
            costs = C[y[i], :]
            expected_cost = np.sum(probs[i, :] * costs)
            total_loss += weights[i] * expected_cost
        
        # L2 regularization
        reg_loss = self.regularization * np.sum(self.coef ** 2)
        
        return total_loss + reg_loss
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OrdinalLogisticRegression':
        """
        Train the model
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,), values 0 to n_classes-1
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.clip(X_scaled, -10, 10)
        
        # Sample weights
        weights = self._compute_sample_weights(y) if self.balance else np.ones(len(y))
        
        n_features = X_scaled.shape[1]
        
        # Initialize parameters
        initial_thresholds = np.linspace(-2, 2, self.n_classes - 1)
        initial_coef = np.zeros(n_features)
        initial_params = np.concatenate([initial_thresholds, initial_coef])
        
        # Optimize
        result = minimize(
            self._loss_function,
            initial_params,
            args=(X_scaled, y, weights),
            method='L-BFGS-B',
            options={'maxiter': 2000}
        )
        
        self.thresholds = result.x[:self.n_classes-1]
        self.coef = result.x[self.n_classes-1:]
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        X_scaled = np.clip(X_scaled, -10, 10)
        return self._class_probs(X_scaled)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        return np.argmax(self.predict_proba(X), axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics
        
        Returns:
        --------
        dict with accuracy, mae, within_1_notch, within_2_notches
        """
        y_pred = self.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'mae': np.mean(np.abs(y - y_pred)),
            'within_1_notch': np.mean(np.abs(y - y_pred) <= 1),
            'within_2_notches': np.mean(np.abs(y - y_pred) <= 2)
        }
    
    def get_feature_importance(self, feature_names: List[str] = None) -> Dict[str, float]:
        """Get feature importance (absolute coefficient values)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.coef))]
        
        return dict(sorted(
            zip(feature_names, np.abs(self.coef)),
            key=lambda x: -x[1]
        ))