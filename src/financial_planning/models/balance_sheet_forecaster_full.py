"""
Balance Sheet Forecasting with XGBoost - ENHANCED VERSION

Added features:
1. Original 5 target variables × 4 lags = 20 features
2. + Financial ratios (gross_margin, ni_margin, etc.) = +5 features
3. + YoY growth rates = +5 features  
4. + Seasonality (quarter dummies) = +3 features
5. + Momentum indicators = +3 features

Total: 36 features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings
from sklearn.preprocessing import StandardScaler
import pickle
import json

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    raise ImportError("XGBoost required. Install with: pip install xgboost")


@dataclass
class ForecastConfig:
    """Configuration for XGBoost forecasting model."""
    
    # Data preparation
    lookback_periods: int = 4
    
    # XGBoost hyperparameters - MORE CONSERVATIVE for more features
    n_estimators: int = 200       # Reduced from 300
    max_depth: int = 5            # Reduced from 8 to prevent overfitting
    learning_rate: float = 0.05   # Reduced from 0.1
    subsample: float = 0.8        # Reduced from 0.9
    colsample_bytree: float = 0.8 # Reduced from 0.9
    reg_alpha: float = 0.1        # L1 regularization (was 0.0)
    reg_lambda: float = 1.0       # L2 regularization

    min_child_weight: int = 5     # Increased from 3
    gamma: float = 0.2            # Increased from 0.1
            
    # Random state
    random_state: int = 42
    
    # Features to predict (5 core variables)
    target_variables: List[str] = field(default_factory=lambda: [
        'sales_revenue',
        'cost_of_goods_sold',
        'overhead_expenses',
        'payroll_expenses',
        'capex'
    ])
    
    # NEW: Toggle for enhanced features
    use_enhanced_features: bool = True


@dataclass
class ForecastResults:
    """Results from balance sheet forecasting."""
    predictions: Dict[str, np.ndarray]
    accuracy_metrics: Dict[str, float]
    model_metadata: Dict[str, Any]


class BalanceSheetForecaster:
    """
    XGBoost-based Balance Sheet Forecaster - ENHANCED VERSION
    
    Features:
    - 20 base features (5 variables × 4 lags)
    - 16 enhanced features (ratios, growth, seasonality, momentum)
    """
    
    def __init__(
        self,
        company_ticker: str,
        config: Optional[ForecastConfig] = None
    ):
        """Initialize forecaster."""
        self.company_ticker = company_ticker
        self.config = config or ForecastConfig()
        
        # XGBoost models (one per target variable)
        self.models: Dict[str, XGBRegressor] = {}
        
        # Data
        self.historical_data: Optional[pd.DataFrame] = None
        
        # Feature names for interpretability
        self.feature_names: List[str] = []
        
        # Metadata
        self.metadata = {
            'company_ticker': company_ticker,
            'model_type': 'xgboost_enhanced',
            'trained': False
        }
    
    def prepare_features(
        self,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare ENHANCED features from historical data.
        
        Features:
        1. Lagged values (20 features): 5 vars × 4 lags
        2. Financial ratios (5 features): gross_margin, ni_margin, op_margin, asset_turnover, debt_ratio
        3. YoY growth rates (5 features): for each target variable
        4. Seasonality (3 features): Q1, Q2, Q3 dummies (Q4 is reference)
        5. Momentum (3 features): rev_momentum, rev_volatility, capex_trend
        
        Total: 36 features when enhanced=True, 20 when enhanced=False
        """
        features = []
        targets = []
        self.feature_names = []
        
        # Need at least 5 quarters for YoY calculations
        start_idx = max(self.config.lookback_periods, 5)
        
        for i in range(start_idx, len(data)):
            feature_row = []
            
            # ================================================================
            # 1. LAGGED FEATURES (Original - 20 features)
            # ================================================================
            for lag in range(1, self.config.lookback_periods + 1):
                for col in self.config.target_variables:
                    if col in data.columns:
                        feature_row.append(data.iloc[i-lag][col])
                    else:
                        feature_row.append(0.0)
                    
                    if i == start_idx:  # Only record names once
                        self.feature_names.append(f"{col}_lag{lag}")
            
            # ================================================================
            # ENHANCED FEATURES (only if enabled)
            # ================================================================
            if self.config.use_enhanced_features:
                
                # ============================================================
                # 2. FINANCIAL RATIOS (5 features)
                # ============================================================
                prev = data.iloc[i-1]
                
                # Gross Margin = (Revenue - COGS) / Revenue
                if 'sales_revenue' in data.columns and prev['sales_revenue'] > 0:
                    if 'cost_of_goods_sold' in data.columns:
                        gross_margin = (prev['sales_revenue'] - prev['cost_of_goods_sold']) / prev['sales_revenue']
                    else:
                        gross_margin = 0.4  # default
                else:
                    gross_margin = 0.4
                feature_row.append(gross_margin)
                
                # Net Income Margin = NI / Revenue
                if 'net_income' in data.columns and 'sales_revenue' in data.columns and prev['sales_revenue'] > 0:
                    ni_margin = prev['net_income'] / prev['sales_revenue']
                else:
                    ni_margin = 0.2
                feature_row.append(ni_margin)
                
                # Operating Margin = EBIT / Revenue
                if 'ebit' in data.columns and 'sales_revenue' in data.columns and prev['sales_revenue'] > 0:
                    op_margin = prev['ebit'] / prev['sales_revenue']
                elif 'operating_income' in data.columns and 'sales_revenue' in data.columns and prev['sales_revenue'] > 0:
                    op_margin = prev['operating_income'] / prev['sales_revenue']
                else:
                    op_margin = 0.25
                feature_row.append(op_margin)
                
                # Asset Turnover = Revenue / Total Assets
                if 'total_assets' in data.columns and prev['total_assets'] > 0:
                    asset_turnover = prev['sales_revenue'] / prev['total_assets'] if 'sales_revenue' in data.columns else 0.3
                else:
                    asset_turnover = 0.3
                feature_row.append(asset_turnover)
                
                # Debt Ratio = Total Liabilities / Total Assets
                if 'total_assets' in data.columns and 'total_liabilities' in data.columns and prev['total_assets'] > 0:
                    debt_ratio = prev['total_liabilities'] / prev['total_assets']
                else:
                    debt_ratio = 0.5
                feature_row.append(debt_ratio)
                
                if i == start_idx:
                    self.feature_names.extend(['gross_margin', 'ni_margin', 'op_margin', 
                                               'asset_turnover', 'debt_ratio'])
                
                # ============================================================
                # 3. YoY GROWTH RATES (5 features)
                # ============================================================
                for col in self.config.target_variables:
                    if col in data.columns and i >= 5:
                        current_val = data.iloc[i-1][col]
                        yoy_val = data.iloc[i-5][col]  # 4 quarters ago = 1 year
                        if yoy_val != 0:
                            yoy_growth = (current_val - yoy_val) / abs(yoy_val)
                            # Clip extreme values to avoid outliers
                            yoy_growth = np.clip(yoy_growth, -1.0, 2.0)
                        else:
                            yoy_growth = 0.0
                    else:
                        yoy_growth = 0.0
                    feature_row.append(yoy_growth)
                    
                    if i == start_idx:
                        self.feature_names.append(f"{col}_yoy_growth")
                
                # ============================================================
                # 4. SEASONALITY - Quarter Dummies (3 features)
                # ============================================================
                if 'date' in data.columns:
                    try:
                        quarter = pd.to_datetime(data.iloc[i]['date']).quarter
                    except:
                        quarter = ((i % 4) + 1)
                else:
                    # Estimate quarter from position
                    quarter = ((i % 4) + 1)
                
                q1 = 1 if quarter == 1 else 0
                q2 = 1 if quarter == 2 else 0
                q3 = 1 if quarter == 3 else 0
                # Q4 is reference (all zeros)
                feature_row.extend([q1, q2, q3])
                
                if i == start_idx:
                    self.feature_names.extend(['is_Q1', 'is_Q2', 'is_Q3'])
                
                # ============================================================
                # 5. MOMENTUM INDICATORS (3 features)
                # ============================================================
                # Revenue momentum (3Q moving average growth)
                if 'sales_revenue' in data.columns and i >= 4:
                    ma3_current = data.iloc[i-3:i]['sales_revenue'].mean()
                    ma3_prev = data.iloc[i-4:i-1]['sales_revenue'].mean()
                    if ma3_prev > 0:
                        rev_momentum = (ma3_current - ma3_prev) / ma3_prev
                    else:
                        rev_momentum = 0.0
                else:
                    rev_momentum = 0.0
                feature_row.append(rev_momentum)
                
                # Revenue volatility (rolling std / mean)
                if 'sales_revenue' in data.columns and i >= 4:
                    rev_std = data.iloc[i-4:i]['sales_revenue'].std()
                    rev_mean = data.iloc[i-4:i]['sales_revenue'].mean()
                    if rev_mean > 0:
                        rev_volatility = rev_std / rev_mean
                    else:
                        rev_volatility = 0.0
                else:
                    rev_volatility = 0.0
                feature_row.append(rev_volatility)
                
                # Capex intensity trend
                if 'capex' in data.columns and 'sales_revenue' in data.columns and i >= 2:
                    capex_intensity_now = data.iloc[i-1]['capex'] / data.iloc[i-1]['sales_revenue'] if data.iloc[i-1]['sales_revenue'] > 0 else 0
                    capex_intensity_prev = data.iloc[i-2]['capex'] / data.iloc[i-2]['sales_revenue'] if data.iloc[i-2]['sales_revenue'] > 0 else 0
                    capex_trend = capex_intensity_now - capex_intensity_prev
                else:
                    capex_trend = 0.0
                feature_row.append(capex_trend)
                
                if i == start_idx:
                    self.feature_names.extend(['rev_momentum', 'rev_volatility', 'capex_trend'])
            
            features.append(feature_row)
            
            # Target values (same as before)
            target_row = []
            for col in self.config.target_variables:
                if col in data.columns:
                    target_row.append(data.iloc[i][col])
                else:
                    target_row.append(0.0)
            
            targets.append(target_row)
        
        X = np.array(features)
        y = np.array(targets)
        
        # Handle any NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return X, y
    
    def train(
        self,
        test_size: float = 0.2,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Train XGBoost models.
        
        Args:
            test_size: Fraction of data to use for testing
            verbose: Verbosity level (0=silent, 1=progress)
            
        Returns:
            metrics: Dictionary of performance metrics
        """
        if self.historical_data is None:
            raise ValueError("Must load historical data first")
        
        if verbose:
            print("="*80)
            print(f"TRAINING XGBOOST FORECASTER FOR {self.company_ticker}")
            if self.config.use_enhanced_features:
                print("MODE: ENHANCED (36 features)")
            else:
                print("MODE: BASIC (20 features)")
            print("="*80)
        
        # Prepare features
        X, y = self.prepare_features(self.historical_data)
        
        # Split data
        test_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:test_idx], X[test_idx:]
        y_train, y_test = y[:test_idx], y[test_idx:]
        
        if verbose:
            print(f"\nData split:")
            print(f"  Train: {len(X_train)} samples")
            print(f"  Test:  {len(X_test)} samples")
            print(f"  Features: {X.shape[1]}")
            print(f"  Feature/Sample ratio: {X.shape[1]/len(X_train):.2f}")
            if X.shape[1]/len(X_train) > 0.5:
                print(f"  ⚠️  Warning: High feature/sample ratio")
            print(f"\nTraining {len(self.config.target_variables)} XGBoost models...")
        
        # Train models
        self.models = {}
        train_predictions = {}
        test_predictions = {}
        
        for i, var in enumerate(self.config.target_variables):
            model = XGBRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                min_child_weight=self.config.min_child_weight,
                gamma=self.config.gamma,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train[:, i])
            self.models[var] = model
            
            # Predictions
            train_predictions[var] = model.predict(X_train)
            test_predictions[var] = model.predict(X_test)
        
        # Calculate metrics
        train_mapes = {}
        test_mapes = {}
        
        for i, var in enumerate(self.config.target_variables):
            # Train MAPE
            train_mape = np.mean(np.abs((y_train[:, i] - train_predictions[var]) / y_train[:, i])) * 100
            train_mapes[var] = train_mape
            
            # Test MAPE
            test_mape = np.mean(np.abs((y_test[:, i] - test_predictions[var]) / y_test[:, i])) * 100
            test_mapes[var] = test_mape
            
            if verbose:
                print(f"  {var}: Train MAPE = {train_mape:.2f}%, Test MAPE = {test_mape:.2f}%")
        
        overall_train_mape = np.mean(list(train_mapes.values()))
        overall_test_mape = np.mean(list(test_mapes.values()))
        
        if verbose:
            print(f"\nOverall Train MAPE: {overall_train_mape:.2f}%")
            print(f"Overall Test MAPE:  {overall_test_mape:.2f}%")
        
        # Update metadata
        self.metadata['trained'] = True
        self.metadata['train_samples'] = len(X_train)
        self.metadata['test_samples'] = len(X_test)
        self.metadata['n_features'] = X.shape[1]
        self.metadata['feature_names'] = self.feature_names
        
        metrics = {
            'overall_mape': overall_test_mape,
            'train_mape': overall_train_mape,
            'test_mape': overall_test_mape,
            'individual_mapes': test_mapes,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X.shape[1]
        }
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance for all models."""
        if not self.models:
            raise ValueError("Must train model first")
        
        importance_data = []
        for var, model in self.models.items():
            importances = model.feature_importances_
            for i, imp in enumerate(importances):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                importance_data.append({
                    'target': var,
                    'feature': feature_name,
                    'importance': imp
                })
        
        df = pd.DataFrame(importance_data)
        
        # Aggregate importance across all targets
        agg_importance = df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        return agg_importance
    
    def predict(
        self,
        periods: int = 4
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions for future periods.
        """
        if not self.models:
            raise ValueError("Must train model first")
        
        if self.historical_data is None:
            raise ValueError("No historical data loaded")
        
        predictions = {}
        current_data = self.historical_data.copy()
        
        for period in range(periods):
            # Get features for next prediction
            X, _ = self.prepare_features(current_data)
            
            if len(X) == 0:
                break
                
            last_features = X[-1:, :]
            
            # Predict all variables
            period_preds = {}
            for var in self.config.target_variables:
                pred = self.models[var].predict(last_features)[0]
                period_preds[var] = pred
                
                if var not in predictions:
                    predictions[var] = []
                predictions[var].append(pred)
            
            # Add prediction to data for next iteration
            new_row = current_data.iloc[-1:].copy()
            for var, val in period_preds.items():
                if var in new_row.columns:
                    new_row[var] = val
            current_data = pd.concat([current_data, new_row], ignore_index=True)
        
        # Convert to arrays
        return {k: np.array(v) for k, v in predictions.items()}
    
    def save_model(self, filepath: str):
        """Save trained models."""
        if not self.models:
            raise ValueError("No trained models to save")
        
        with open(f"{filepath}_models.pkl", 'wb') as f:
            pickle.dump(self.models, f)
        
        with open(f"{filepath}_config.json", 'w') as f:
            config_dict = {
                'lookback_periods': self.config.lookback_periods,
                'n_estimators': self.config.n_estimators,
                'max_depth': self.config.max_depth,
                'learning_rate': self.config.learning_rate,
                'target_variables': self.config.target_variables,
                'use_enhanced_features': self.config.use_enhanced_features
            }
            json.dump(config_dict, f, indent=2)
        
        with open(f"{filepath}_metadata.json", 'w') as f:
            meta_copy = self.metadata.copy()
            if 'feature_names' in meta_copy:
                meta_copy['feature_names'] = list(meta_copy['feature_names'])
            json.dump(meta_copy, f, indent=2)
    
    def load_model(self, filepath: str):
        """Load trained models."""
        with open(f"{filepath}_models.pkl", 'rb') as f:
            self.models = pickle.load(f)
        
        with open(f"{filepath}_metadata.json", 'r') as f:
            self.metadata = json.load(f)


# ============================================================================
# HELPER FUNCTIONS (for compatibility with auto_forecast_pipeline.py)
# ============================================================================

def train_xgboost_models(X_train, y_train, config, target_variables):
    """Train XGBoost models - for pipeline compatibility."""
    models = {}
    
    for i, var in enumerate(target_variables):
        model = XGBRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=getattr(config, 'subsample', 0.8),
            colsample_bytree=getattr(config, 'colsample_bytree', 0.8),
            reg_alpha=getattr(config, 'reg_alpha', 0.1),
            reg_lambda=getattr(config, 'reg_lambda', 1.0),
            min_child_weight=getattr(config, 'min_child_weight', 5),
            gamma=getattr(config, 'gamma', 0.2),
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train[:, i])
        models[var] = model
    
    return models


def predict_with_models(models, X, target_variables):
    """Predict using trained models."""
    predictions = {}
    for var in target_variables:
        predictions[var] = models[var].predict(X)
    return predictions


def calculate_mape(y_true, y_pred_dict, target_variables):
    """Calculate MAPE for each variable."""
    mapes = {}
    for i, var in enumerate(target_variables):
        mape = np.mean(np.abs((y_true[:, i] - y_pred_dict[var]) / y_true[:, i])) * 100
        mapes[var] = mape
    return mapes
