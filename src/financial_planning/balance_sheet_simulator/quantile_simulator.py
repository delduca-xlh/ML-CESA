"""
quantile_simulator.py

XGBoost Quantile Regression for Probabilistic Balance Sheet Forecasting.
Outputs probability DISTRIBUTION, not just point estimate.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from .data_structures import QuantileForecast, CompleteFinancialStatements
from .accounting_engine import AccountingEngine


class QuantileSimulator:
    """XGBoost Quantile Regression for Time Series."""
    
    QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    # Keep simple: 5 core drivers only (seasonality features caused overfitting)
    DRIVERS = ['revenue_growth', 'cogs_margin', 'opex_margin', 'capex_ratio', 'net_margin']
    
    def __init__(self, seq_length: int = 4):
        self.seq_length = seq_length
        self.models = {}
        self.scalers = {}
        self.is_fitted = False
        self.accounting_engine = None
        self.available_drivers = []
        
    def prepare_drivers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert raw financials to driver ratios."""
        df = data.copy()
        col_map = {
            'sales_revenue': 'revenue',
            'cost_of_goods_sold': 'cogs',
            'overhead_expenses': 'opex',
            'totalRevenue': 'revenue',
            'costOfRevenue': 'cogs',
            'operatingExpenses': 'opex',
        }
        df = df.rename(columns=col_map)
        
        drivers = pd.DataFrame(index=df.index)
        
        # Core drivers only - keep it simple to avoid overfitting
        if 'revenue' in df.columns:
            drivers['revenue_growth'] = df['revenue'].pct_change()
        if 'cogs' in df.columns and 'revenue' in df.columns:
            drivers['cogs_margin'] = df['cogs'] / df['revenue']
        if 'opex' in df.columns and 'revenue' in df.columns:
            drivers['opex_margin'] = df['opex'] / df['revenue']
        if 'capex' in df.columns and 'revenue' in df.columns:
            drivers['capex_ratio'] = df['capex'].abs() / df['revenue']
        elif 'capitalExpenditure' in df.columns and 'revenue' in df.columns:
            drivers['capex_ratio'] = df['capitalExpenditure'].abs() / df['revenue']
        
        # Net margin - directly predict net income / revenue
        net_income_col = None
        for col in ['net_income', 'netIncome']:
            if col in df.columns:
                net_income_col = col
                break
        if net_income_col and 'revenue' in df.columns:
            drivers['net_margin'] = df[net_income_col] / df['revenue']
        
        return drivers.dropna()
    
    def create_lag_features(self, drivers: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Create X = [D_{t-1}, ..., D_{t-k}], y = D_t"""
        available_drivers = [d for d in self.DRIVERS if d in drivers.columns]
        n_samples = len(drivers) - self.seq_length
        n_drivers = len(available_drivers)
        
        X = np.zeros((n_samples, self.seq_length * n_drivers))
        y = np.zeros((n_samples, n_drivers))
        values = drivers[available_drivers].values
        
        for i in range(n_samples):
            X[i] = values[i:i + self.seq_length].flatten()
            y[i] = values[i + self.seq_length]
        
        return X, y, available_drivers
    
    def fit(self, data: pd.DataFrame, verbose: bool = True) -> Dict:
        """Train XGBoost Quantile models."""
        self.accounting_engine = AccountingEngine(data)
        drivers = self.prepare_drivers(data)
        X, y, available_drivers = self.create_lag_features(drivers)
        self.available_drivers = available_drivers
        
        if len(X) < 10:
            raise ValueError(f"Not enough training samples: {len(X)}. Need at least 10.")
        
        if verbose:
            print(f"\n{'='*80}")
            print("QUANTILE SIMULATOR - TRAINING")
            print(f"{'='*80}")
            print(f"Training samples: {len(X)}")
            print(f"Drivers: {available_drivers}")
            print(f"Quantiles: {self.QUANTILES}")
        
        for i, driver in enumerate(available_drivers):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[driver] = scaler
            self.models[driver] = {}
            
            for q in self.QUANTILES:
                model = GradientBoostingRegressor(
                    loss='quantile', alpha=q, n_estimators=100,
                    max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42
                )
                model.fit(X_scaled, y[:, i])
                self.models[driver][q] = model
        
        self.last_drivers = drivers[available_drivers].values[-self.seq_length:]
        self.last_data = data.iloc[-1]
        self.is_fitted = True
        
        if verbose:
            print("Training complete.")
        
        return {'n_samples': len(X), 'drivers': available_drivers}
    
    def predict_distribution(self, recent_drivers: np.ndarray) -> Dict[str, QuantileForecast]:
        """Predict probability distribution for each driver."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        X = recent_drivers[-self.seq_length:].flatten().reshape(1, -1)
        forecasts = {}
        
        for driver in self.available_drivers:
            X_scaled = self.scalers[driver].transform(X)
            preds = {q: self.models[driver][q].predict(X_scaled)[0] for q in self.QUANTILES}
            forecasts[driver] = QuantileForecast(
                variable=driver,
                q05=preds[0.05], q10=preds[0.10], q25=preds[0.25],
                q50=preds[0.50], q75=preds[0.75], q90=preds[0.90], q95=preds[0.95]
            )
        
        return forecasts
    
    def create_prior_statements(self, row: pd.Series) -> CompleteFinancialStatements:
        """Create CompleteFinancialStatements from a data row."""
        stmt = CompleteFinancialStatements(period="Prior")
        
        # Comprehensive field mappings for FMP data
        mappings = {
            # Income Statement
            'revenue': ['revenue', 'sales_revenue', 'totalRevenue'],
            'cogs': ['cogs', 'cost_of_goods_sold', 'costOfRevenue'],
            'gross_profit': ['gross_profit', 'grossProfit'],
            'opex': ['opex', 'operatingExpenses'],
            'net_income': ['net_income', 'netIncome'],
            'ebitda': ['ebitda'],
            'ebit': ['ebit', 'operatingIncome'],
            'depreciation': ['depreciation', 'depreciationAndAmortization'],
            'interest_expense': ['interest_expense', 'interestExpense'],
            
            # Current Assets
            'cash': ['cash', 'cash_and_cash_equivalents', 'cashAndCashEquivalents'],
            'short_term_investments': ['short_term_investments', 'shortTermInvestments'],
            'accounts_receivable': ['accounts_receivable', 'accountsReceivable', 'netReceivables'],
            'inventory': ['inventory'],
            'prepaid_expenses': ['prepaid_expenses', 'prepaidExpenses', 'otherCurrentAssets'],
            'other_current_assets': ['other_current_assets', 'otherCurrentAssets'],
            'total_current_assets': ['total_current_assets', 'totalCurrentAssets'],
            
            # Non-Current Assets
            'ppe_net': ['ppe_net', 'property_plant_equipment_net', 'propertyPlantEquipmentNet'],
            'ppe_gross': ['ppe_gross', 'property_plant_equipment_gross'],
            'accumulated_depreciation': ['accumulated_depreciation', 'accumulatedDepreciation'],
            'goodwill': ['goodwill'],
            'intangible_assets': ['intangible_assets', 'intangibleAssets', 'goodwillAndIntangibleAssets'],
            'long_term_investments': ['long_term_investments', 'longTermInvestments'],
            'other_noncurrent_assets': ['other_noncurrent_assets', 'otherNonCurrentAssets'],
            'total_noncurrent_assets': ['total_noncurrent_assets', 'totalNonCurrentAssets'],
            'total_assets': ['total_assets', 'totalAssets'],
            
            # Current Liabilities
            'accounts_payable': ['accounts_payable', 'accountsPayable', 'accountPayables'],
            'accrued_expenses': ['accrued_expenses', 'accruedExpenses', 'accruedLiabilities'],
            'short_term_debt': ['short_term_debt', 'shortTermDebt'],
            'deferred_revenue': ['deferred_revenue', 'deferredRevenue'],
            'other_current_liabilities': ['other_current_liabilities', 'otherCurrentLiabilities'],
            'total_current_liabilities': ['total_current_liabilities', 'totalCurrentLiabilities'],
            
            # Non-Current Liabilities
            'long_term_debt': ['long_term_debt', 'longTermDebt', 'total_debt', 'totalDebt'],
            'deferred_tax_liabilities': ['deferred_tax_liabilities', 'deferredTaxLiabilitiesNonCurrent'],
            'other_noncurrent_liabilities': ['other_noncurrent_liabilities', 'otherNonCurrentLiabilities'],
            'total_noncurrent_liabilities': ['total_noncurrent_liabilities', 'totalNonCurrentLiabilities'],
            'total_liabilities': ['total_liabilities', 'totalLiabilities'],
            
            # Equity
            'common_stock': ['common_stock', 'commonStock'],
            'additional_paid_in_capital': ['additional_paid_in_capital', 'additionalPaidInCapital', 'capitalStock'],
            'retained_earnings': ['retained_earnings', 'retainedEarnings'],
            'treasury_stock': ['treasury_stock', 'treasuryStock'],
            'aoci': ['aoci', 'accumulatedOtherComprehensiveIncome', 'accumulatedOtherComprehensiveIncomeLoss'],
            'total_equity': ['total_equity', 'totalEquity', 'totalStockholdersEquity'],
        }
        
        for attr, possible_names in mappings.items():
            for name in possible_names:
                if name in row.index:
                    val = row[name]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        setattr(stmt, attr, val)
                    break
        
        # Derive missing values from totals if available
        if stmt.ppe_gross == 0 and stmt.ppe_net > 0:
            stmt.ppe_gross = stmt.ppe_net * 1.5
            stmt.accumulated_depreciation = stmt.ppe_gross - stmt.ppe_net
        
        # Estimate other non-current assets from total if we have it
        if stmt.total_noncurrent_assets > 0 and stmt.other_noncurrent_assets == 0:
            known_nca = stmt.ppe_net + stmt.goodwill + stmt.intangible_assets + stmt.long_term_investments
            stmt.other_noncurrent_assets = max(0, stmt.total_noncurrent_assets - known_nca)
        
        # Estimate other current assets from total if we have it
        if stmt.total_current_assets > 0 and stmt.other_current_assets == 0:
            known_ca = stmt.cash + stmt.short_term_investments + stmt.accounts_receivable + stmt.inventory + stmt.prepaid_expenses
            stmt.other_current_assets = max(0, stmt.total_current_assets - known_ca)
        
        return stmt
    
    def simulate_scenarios(self, n_scenarios: int = 100, n_periods: int = 4) -> Dict:
        """Simulate multiple possible futures."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        all_scenarios = []
        
        for s in range(n_scenarios):
            scenario_path = []
            current_drivers = self.last_drivers.copy()
            prior = self.create_prior_statements(self.last_data)
            
            for t in range(n_periods):
                forecasts = self.predict_distribution(current_drivers)
                
                driver_values = {}
                sampled_drivers = []
                for driver in self.available_drivers:
                    sampled = forecasts[driver].sample(1)[0]
                    driver_values[driver] = sampled
                    sampled_drivers.append(sampled)
                
                driver_values.setdefault('revenue_growth', 0.02)
                driver_values.setdefault('cogs_margin', 0.60)
                driver_values.setdefault('opex_margin', 0.15)
                driver_values.setdefault('capex_ratio', 0.03)
                driver_values.setdefault('net_margin', 0.15)
                
                statements = self.accounting_engine.derive_statements(
                    drivers=driver_values, prior=prior, period=f"Q+{t+1}"
                )
                scenario_path.append(statements)
                prior = statements
                current_drivers = np.vstack([current_drivers[1:], sampled_drivers])
            
            all_scenarios.append(scenario_path)
        
        # Calculate statistics
        key_vars = ['revenue', 'net_income', 'total_assets', 'total_equity', 'cash']
        stats = {var: {'mean': [], 'std': [], 'p5': [], 'p95': []} for var in key_vars}
        
        for t in range(n_periods):
            for var in key_vars:
                values = [getattr(all_scenarios[s][t], var, 0) for s in range(n_scenarios)]
                stats[var]['mean'].append(np.mean(values))
                stats[var]['std'].append(np.std(values))
                stats[var]['p5'].append(np.percentile(values, 5))
                stats[var]['p95'].append(np.percentile(values, 95))
        
        return {'all_scenarios': all_scenarios, 'stats': stats, 'n_scenarios': n_scenarios, 'n_periods': n_periods}
