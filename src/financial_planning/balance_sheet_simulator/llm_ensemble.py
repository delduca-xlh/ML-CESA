"""
llm_ensemble.py

LLM Ensemble for Balance Sheet Forecasting.
Compares and combines ML predictions with LLM predictions.

Approaches:
1. ML Only (XGBoost Quantile Regression) - reuses Q1 results if available
2. ML + LLM Ratios (ML drivers + LLM-adjusted ratios)
3. Pure LLM (LLM predicts everything directly)
4. Ensemble (select best approach based on validation)

API Key Security:
- Set ANTHROPIC_API_KEY environment variable, OR
- Pass --api-key via command line
- NEVER hardcode API keys in source code
"""

import os
import json
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .quantile_simulator import QuantileSimulator
from .data_structures import CompleteFinancialStatements
from .accounting_engine import AccountingEngine


@dataclass
class LLMPrediction:
    """LLM prediction result."""
    drivers: Dict[str, float]
    reasoning: str
    source: str  # 'LLM', 'Fallback'
    confidence: float


@dataclass
class EnsembleResult:
    """Result of ensemble comparison."""
    best_approach: str
    ml_mape: float
    llm_mape: float
    ensemble_mape: float
    ml_predictions: Dict
    llm_predictions: Dict
    selected_predictions: Dict
    reasoning: str


def load_q1_results(ticker: str, output_dir: str = "outputs") -> Optional[Dict]:
    """
    Load existing Q1 (ML Only) results if available.
    
    Args:
        ticker: Stock ticker
        output_dir: Base output directory
        
    Returns:
        Dict with Q1 results or None if not found
    """
    # Check multiple possible locations
    possible_paths = [
        Path(output_dir) / f"{ticker.upper()}_statements",
        Path(output_dir) / f"{ticker.lower()}_statements", 
        Path(f"{ticker.upper()}_statements"),
        Path(f"{ticker.lower()}_statements"),
        Path(output_dir) / "xgboost_models" / ticker.lower(),
    ]
    
    for base_path in possible_paths:
        # Look for saved predictions JSON
        json_path = base_path / "ml_predictions.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                print(f"  ✓ Loaded Q1 results from: {json_path}")
                return data
            except Exception as e:
                print(f"  ⚠ Failed to load {json_path}: {e}")
                
        # Look for CSV predictions
        csv_path = base_path / "predictions.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                print(f"  ✓ Loaded Q1 results from: {csv_path}")
                return {'dataframe': df, 'source': str(csv_path)}
            except Exception as e:
                print(f"  ⚠ Failed to load {csv_path}: {e}")
    
    return None


def save_q1_results(ticker: str, results: Dict, output_dir: str = "outputs"):
    """
    Save Q1 (ML Only) results for reuse.
    
    Args:
        ticker: Stock ticker
        results: Results dict to save
        output_dir: Base output directory
    """
    base_path = Path(output_dir) / f"{ticker.upper()}_statements"
    base_path.mkdir(parents=True, exist_ok=True)
    
    json_path = base_path / "ml_predictions.json"
    
    # Convert statements to serializable format
    serializable = {
        'ticker': ticker,
        'timestamp': datetime.now().isoformat(),
        'source': 'ML Only (XGBoost Quantile)',
    }
    
    if 'statements' in results:
        serializable['predictions'] = []
        for stmt in results['statements']:
            serializable['predictions'].append({
                'period': stmt.period,
                'revenue': stmt.revenue,
                'net_income': stmt.net_income,
                'total_assets': stmt.total_assets,
                'total_equity': stmt.total_equity,
                'cash': stmt.cash,
            })
    
    try:
        with open(json_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"  ✓ Saved Q1 results to: {json_path}")
    except Exception as e:
        print(f"  ⚠ Failed to save Q1 results: {e}")


class LLMForecaster:
    """LLM-based financial forecasting."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize LLM forecaster.
        
        Args:
            api_key: Anthropic API key. 
                     Priority: 1) explicit arg, 2) ANTHROPIC_API_KEY env var
                     
        Security Note:
            NEVER hardcode API keys. Use environment variables:
            $ export ANTHROPIC_API_KEY="your-key-here"
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.client = None
        
        if not self.api_key:
            print("  ⚠ No API key found. Set ANTHROPIC_API_KEY environment variable.")
            print("    Example: export ANTHROPIC_API_KEY='sk-ant-...'")
            print("    Or pass --api-key argument")
        
    def _init_client(self):
        """Initialize Anthropic client."""
        if self.client is None:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")
    
    def predict_drivers(self, 
                       ticker: str,
                       historical_data: pd.DataFrame,
                       n_periods: int = 4) -> LLMPrediction:
        """
        LLM predicts driver ratios for forecasting.
        
        Args:
            ticker: Stock ticker symbol
            historical_data: Historical financial data
            n_periods: Number of quarters to forecast
            
        Returns:
            LLMPrediction with driver values
        """
        # Calculate historical averages for prompt
        recent = historical_data.tail(8) if len(historical_data) >= 8 else historical_data
        
        # Extract key metrics
        revenue_col = self._find_column(recent, ['revenue', 'sales_revenue', 'totalRevenue'])
        cogs_col = self._find_column(recent, ['cogs', 'cost_of_goods_sold', 'costOfRevenue'])
        opex_col = self._find_column(recent, ['opex', 'operatingExpenses'])
        ni_col = self._find_column(recent, ['net_income', 'netIncome'])
        capex_col = self._find_column(recent, ['capex', 'capitalExpenditure'])
        
        avg_revenue = recent[revenue_col].mean() if revenue_col else 0
        avg_cogs = recent[cogs_col].mean() if cogs_col else 0
        avg_opex = recent[opex_col].mean() if opex_col else 0
        avg_ni = recent[ni_col].mean() if ni_col else 0
        avg_capex = abs(recent[capex_col].mean()) if capex_col else 0
        
        # Calculate historical ratios
        gross_margin = (avg_revenue - avg_cogs) / avg_revenue if avg_revenue > 0 else 0.4
        opex_ratio = avg_opex / avg_revenue if avg_revenue > 0 else 0.15
        net_margin = avg_ni / avg_revenue if avg_revenue > 0 else 0.1
        capex_ratio = avg_capex / avg_revenue if avg_revenue > 0 else 0.05
        
        # Calculate revenue growth trend
        if revenue_col and len(recent) >= 4:
            rev_growth = recent[revenue_col].pct_change().mean()
        else:
            rev_growth = 0.02
        
        prompt = f"""You are a senior financial analyst predicting key financial ratios for {ticker}.

### Historical Data (Last {len(recent)} Quarters):
- Average Revenue: ${avg_revenue/1e9:.2f}B
- Gross Margin: {gross_margin:.1%}
- Operating Expense Ratio: {opex_ratio:.1%}
- Net Margin: {net_margin:.1%}
- CapEx/Revenue: {capex_ratio:.1%}
- Historical Revenue Growth: {rev_growth:.1%} per quarter

### Recent Revenue Trend (in $B):
{self._format_trend(recent, revenue_col)}

### Task:
Predict the following driver ratios for the next {n_periods} quarters.
Consider: industry trends, company-specific factors, seasonality, and your knowledge of {ticker}.

### Respond ONLY with this JSON (no other text):
{{
    "revenue_growth": 0.XX,
    "cogs_margin": 0.XX,
    "opex_margin": 0.XX,
    "capex_ratio": 0.XX,
    "net_margin": 0.XX,
    "confidence": 0.X,
    "reasoning": "brief explanation"
}}

Note:
- revenue_growth: Expected quarterly revenue growth rate (e.g., 0.02 = 2%)
- cogs_margin: Cost of goods sold / Revenue (e.g., 0.60 = 60%)
- opex_margin: Operating expenses / Revenue
- capex_ratio: Capital expenditure / Revenue
- net_margin: Net income / Revenue
- confidence: Your confidence in these predictions (0.0 to 1.0)
"""

        try:
            self._init_client()
            
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text.strip()
            
            # Parse JSON response
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                return LLMPrediction(
                    drivers={
                        'revenue_growth': result.get('revenue_growth', rev_growth),
                        'cogs_margin': result.get('cogs_margin', 1 - gross_margin),
                        'opex_margin': result.get('opex_margin', opex_ratio),
                        'capex_ratio': result.get('capex_ratio', capex_ratio),
                        'net_margin': result.get('net_margin', net_margin),
                    },
                    reasoning=result.get('reasoning', ''),
                    source='LLM',
                    confidence=result.get('confidence', 0.7)
                )
                
        except Exception as e:
            print(f"  ⚠ LLM driver prediction failed: {e}")
        
        # Fallback to historical averages
        return LLMPrediction(
            drivers={
                'revenue_growth': rev_growth,
                'cogs_margin': 1 - gross_margin,
                'opex_margin': opex_ratio,
                'capex_ratio': capex_ratio,
                'net_margin': net_margin,
            },
            reasoning='Fallback to historical averages',
            source='Fallback',
            confidence=0.5
        )
    
    def predict_complete_statements(self,
                                    ticker: str,
                                    historical_data: pd.DataFrame,
                                    n_periods: int = 4) -> Dict:
        """
        LLM directly predicts complete financial values.
        
        Args:
            ticker: Stock ticker
            historical_data: Historical data
            n_periods: Quarters to forecast
            
        Returns:
            Dict with predicted values for each period
        """
        recent = historical_data.tail(8) if len(historical_data) >= 8 else historical_data
        
        # Get key metrics
        revenue_col = self._find_column(recent, ['revenue', 'sales_revenue', 'totalRevenue'])
        ni_col = self._find_column(recent, ['net_income', 'netIncome'])
        assets_col = self._find_column(recent, ['total_assets', 'totalAssets'])
        equity_col = self._find_column(recent, ['total_equity', 'totalEquity', 'totalStockholdersEquity'])
        
        avg_revenue = recent[revenue_col].mean() if revenue_col else 100e9
        avg_ni = recent[ni_col].mean() if ni_col else 10e9
        avg_assets = recent[assets_col].mean() if assets_col else 300e9
        avg_equity = recent[equity_col].mean() if equity_col else 100e9
        
        prompt = f"""You are a senior financial analyst forecasting {n_periods} quarters for {ticker}.

### Historical Averages (Last {len(recent)} Quarters):
- Revenue: ${avg_revenue/1e9:.2f}B
- Net Income: ${avg_ni/1e9:.2f}B
- Total Assets: ${avg_assets/1e9:.2f}B
- Total Equity: ${avg_equity/1e9:.2f}B

### Recent Revenue Trend:
{self._format_trend(recent, revenue_col)}

### Task:
Predict the next {n_periods} quarters. Consider seasonality and trends.

### Respond ONLY with this JSON (values in BILLIONS):
{{
    "predictions": [
        {{"quarter": 1, "revenue": XX.X, "net_income": X.X, "total_assets": XXX.X, "total_equity": XX.X}},
        {{"quarter": 2, "revenue": XX.X, "net_income": X.X, "total_assets": XXX.X, "total_equity": XX.X}},
        {{"quarter": 3, "revenue": XX.X, "net_income": X.X, "total_assets": XXX.X, "total_equity": XX.X}},
        {{"quarter": 4, "revenue": XX.X, "net_income": X.X, "total_assets": XXX.X, "total_equity": XX.X}}
    ],
    "reasoning": "brief explanation"
}}
"""

        try:
            self._init_client()
            
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text.strip()
            match = re.search(r'\{[\s\S]*\}', text)
            
            if match:
                result = json.loads(match.group())
                predictions = result.get('predictions', [])[:n_periods]
                
                if len(predictions) >= n_periods:
                    return {
                        'revenue': np.array([p['revenue'] * 1e9 for p in predictions]),
                        'net_income': np.array([p['net_income'] * 1e9 for p in predictions]),
                        'total_assets': np.array([p['total_assets'] * 1e9 for p in predictions]),
                        'total_equity': np.array([p['total_equity'] * 1e9 for p in predictions]),
                        'reasoning': result.get('reasoning', ''),
                        'source': 'LLM'
                    }
                    
        except Exception as e:
            print(f"  ⚠ LLM complete prediction failed: {e}")
        
        # Fallback
        return {
            'revenue': np.full(n_periods, avg_revenue),
            'net_income': np.full(n_periods, avg_ni),
            'total_assets': np.full(n_periods, avg_assets),
            'total_equity': np.full(n_periods, avg_equity),
            'reasoning': 'Fallback to historical averages',
            'source': 'Fallback'
        }
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column name."""
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _format_trend(self, df: pd.DataFrame, col: str) -> str:
        """Format revenue trend for prompt."""
        if col is None or col not in df.columns:
            return "N/A"
        values = df[col].values[-4:] if len(df) >= 4 else df[col].values
        return ', '.join([f'Q{i+1}: ${v/1e9:.1f}B' for i, v in enumerate(values)])


class EnsembleForecaster:
    """
    Ensemble forecaster combining ML and LLM predictions.
    
    Approaches:
    1. ML Only: XGBoost quantile regression (reuses Q1 results if available)
    2. ML + LLM: ML drivers with LLM-adjusted ratios
    3. Pure LLM: LLM predicts everything
    4. Auto-Ensemble: Select best based on validation
    
    Features:
    - Reuses Q1 (ML Only) results if already computed
    - Ensures consistency between Q1 and Q2 ML predictions
    """
    
    def __init__(self, api_key: str = None, output_dir: str = "outputs"):
        """
        Initialize ensemble forecaster.
        
        Args:
            api_key: Anthropic API key (from env var or explicit)
            output_dir: Directory for saving/loading results
        """
        self.llm_forecaster = LLMForecaster(api_key)
        self.ml_simulator = None
        self.output_dir = output_dir
        self.q1_results = None  # Cached Q1 results
        
    def fit(self, data: pd.DataFrame, ticker: str = None, verbose: bool = True, 
            reuse_q1: bool = True):
        """
        Fit the ML model on historical data.
        
        Args:
            data: Historical financial data
            ticker: Stock ticker (for loading cached results)
            verbose: Print progress
            reuse_q1: If True, try to load existing Q1 results
        """
        self.data = data
        self.ticker = ticker
        
        # Try to load existing Q1 results
        if reuse_q1 and ticker:
            if verbose:
                print(f"\n  Checking for existing Q1 results...")
            self.q1_results = load_q1_results(ticker, self.output_dir)
            
            if self.q1_results:
                if verbose:
                    print(f"  ✓ Found cached Q1 results - will reuse for consistency")
        
        # Always fit ML simulator (needed for ML+LLM approach)
        self.ml_simulator = QuantileSimulator(seq_length=4)
        self.ml_simulator.fit(data, verbose=verbose)
        
    def forecast_all_approaches(self,
                                ticker: str,
                                n_periods: int = 4,
                                verbose: bool = True,
                                save_ml: bool = True) -> Dict:
        """
        Generate forecasts using all approaches.
        
        Args:
            ticker: Stock ticker
            n_periods: Quarters to forecast
            verbose: Print progress
            save_ml: Save ML results for future reuse
        
        Returns:
            Dict with predictions from each approach
        """
        if self.ml_simulator is None:
            raise ValueError("Must call fit() first")
            
        results = {}
        
        # Approach 1: ML Only (reuse if available)
        if verbose:
            print("\n[1] ML Only (XGBoost Quantile)")
            
        if self.q1_results and 'predictions' in self.q1_results:
            # Reuse cached results
            if verbose:
                print("    → Reusing cached Q1 results for consistency")
            ml_predictions = self._load_cached_ml_predictions()
        else:
            # Generate new predictions
            if verbose:
                print("    → Generating new ML predictions")
            ml_predictions = self._forecast_ml(n_periods)
            
            # Save for future reuse
            if save_ml:
                save_q1_results(ticker, ml_predictions, self.output_dir)
        
        results['ML Only'] = ml_predictions
        if verbose and 'statements' in ml_predictions:
            print(f"    Revenue Q1: ${ml_predictions['statements'][0].revenue/1e9:.2f}B")
        
        # Approach 2: ML + LLM Ratios
        if verbose:
            print("\n[2] ML + LLM Ratios")
        
        if self.llm_forecaster.api_key:
            llm_pred = self.llm_forecaster.predict_drivers(ticker, self.data, n_periods)
            if verbose:
                print(f"    LLM Source: {llm_pred.source}")
                print(f"    LLM Confidence: {llm_pred.confidence:.0%}")
            ml_llm_predictions = self._forecast_ml_with_llm_ratios(llm_pred, n_periods)
            results['ML + LLM'] = {**ml_llm_predictions, 'llm_reasoning': llm_pred.reasoning}
            if verbose:
                print(f"    Revenue Q1: ${ml_llm_predictions['statements'][0].revenue/1e9:.2f}B")
        else:
            if verbose:
                print("    ⚠ Skipped (no API key)")
            results['ML + LLM'] = None
        
        # Approach 3: Pure LLM
        if verbose:
            print("\n[3] Pure LLM")
        
        if self.llm_forecaster.api_key:
            pure_llm = self.llm_forecaster.predict_complete_statements(ticker, self.data, n_periods)
            results['Pure LLM'] = pure_llm
            if verbose:
                print(f"    LLM Source: {pure_llm['source']}")
                print(f"    Revenue Q1: ${pure_llm['revenue'][0]/1e9:.2f}B")
        else:
            if verbose:
                print("    ⚠ Skipped (no API key)")
            results['Pure LLM'] = None
        
        return results
    
    def _load_cached_ml_predictions(self) -> Dict:
        """Load ML predictions from cache."""
        if not self.q1_results or 'predictions' not in self.q1_results:
            return self._forecast_ml(4)
        
        # Reconstruct statements from cached data
        statements = []
        for pred in self.q1_results['predictions']:
            stmt = CompleteFinancialStatements(period=pred.get('period', 'Q+?'))
            stmt.revenue = pred.get('revenue', 0)
            stmt.net_income = pred.get('net_income', 0)
            stmt.total_assets = pred.get('total_assets', 0)
            stmt.total_equity = pred.get('total_equity', 0)
            stmt.cash = pred.get('cash', 0)
            statements.append(stmt)
        
        return {
            'statements': statements,
            'source': 'ML (cached)'
        }
    
    def _forecast_ml(self, n_periods: int) -> Dict:
        """Generate ML-only forecasts."""
        statements = []
        current_drivers = self.ml_simulator.last_drivers.copy()
        prior = self.ml_simulator.create_prior_statements(self.ml_simulator.last_data)
        
        for t in range(n_periods):
            forecasts = self.ml_simulator.predict_distribution(current_drivers)
            
            # Use median (q50) for point prediction
            driver_values = {d: forecasts[d].q50 for d in self.ml_simulator.available_drivers}
            driver_values.setdefault('revenue_growth', 0.02)
            driver_values.setdefault('cogs_margin', 0.60)
            driver_values.setdefault('opex_margin', 0.15)
            driver_values.setdefault('capex_ratio', 0.03)
            driver_values.setdefault('net_margin', 0.15)
            
            stmt = self.ml_simulator.accounting_engine.derive_statements(
                drivers=driver_values, prior=prior, period=f"Q+{t+1}"
            )
            statements.append(stmt)
            prior = stmt
            
            # Update drivers for next period
            new_drivers = [driver_values[d] for d in self.ml_simulator.available_drivers]
            current_drivers = np.vstack([current_drivers[1:], new_drivers])
        
        return {
            'statements': statements,
            'source': 'ML'
        }
    
    def _forecast_ml_with_llm_ratios(self, llm_pred: LLMPrediction, n_periods: int) -> Dict:
        """Generate forecasts using ML drivers but LLM-adjusted ratios."""
        statements = []
        current_drivers = self.ml_simulator.last_drivers.copy()
        prior = self.ml_simulator.create_prior_statements(self.ml_simulator.last_data)
        
        for t in range(n_periods):
            # Get ML distribution
            forecasts = self.ml_simulator.predict_distribution(current_drivers)
            
            # Use ML for growth, LLM for margins
            driver_values = {
                'revenue_growth': forecasts['revenue_growth'].q50 if 'revenue_growth' in forecasts else llm_pred.drivers['revenue_growth'],
                'cogs_margin': llm_pred.drivers['cogs_margin'],
                'opex_margin': llm_pred.drivers['opex_margin'],
                'capex_ratio': llm_pred.drivers['capex_ratio'],
                'net_margin': llm_pred.drivers['net_margin'],
            }
            
            stmt = self.ml_simulator.accounting_engine.derive_statements(
                drivers=driver_values, prior=prior, period=f"Q+{t+1}"
            )
            statements.append(stmt)
            prior = stmt
            
            new_drivers = [driver_values.get(d, 0) for d in self.ml_simulator.available_drivers]
            current_drivers = np.vstack([current_drivers[1:], new_drivers])
        
        return {
            'statements': statements,
            'source': 'ML + LLM'
        }
    
    def evaluate_approaches(self,
                           predictions: Dict,
                           actuals: pd.DataFrame,
                           verbose: bool = True) -> Dict:
        """
        Evaluate all approaches against actual data.
        
        Args:
            predictions: Dict from forecast_all_approaches()
            actuals: Actual financial data for the forecast periods
            
        Returns:
            Dict with MAPE for each approach and best approach selection
        """
        results = {}
        
        metrics = ['revenue', 'net_income', 'total_assets', 'total_equity']
        
        for approach_name, approach_data in predictions.items():
            # Skip None approaches (LLM disabled)
            if approach_data is None:
                continue
                
            mapes = {}
            
            for metric in metrics:
                if approach_name == 'Pure LLM':
                    pred_values = approach_data.get(metric, np.array([]))
                    if len(pred_values) == 0:
                        continue
                else:
                    if 'statements' not in approach_data or not approach_data['statements']:
                        continue
                    pred_values = np.array([getattr(s, metric, 0) for s in approach_data['statements']])
                
                # Get actual values
                actual_col = self._find_actual_column(actuals, metric)
                if actual_col:
                    actual_values = actuals[actual_col].values[:len(pred_values)]
                    mape = self._calculate_mape(actual_values, pred_values)
                    mapes[metric] = mape
            
            avg_mape = np.mean(list(mapes.values())) if mapes else float('inf')
            results[approach_name] = {
                'mapes': mapes,
                'avg_mape': avg_mape
            }
        
        # Find best approach (only from available)
        if not results:
            return {'results': {}, 'best_approach': None, 'best_mape': float('inf')}
            
        best_approach = min(results, key=lambda x: results[x]['avg_mape'])
        
        if verbose:
            print("\n" + "="*70)
            print("ENSEMBLE COMPARISON")
            print("="*70)
            print(f"\n{'Approach':<20} {'Revenue':<12} {'Net Income':<12} {'Assets':<12} {'Equity':<12} {'Avg MAPE':<10}")
            print("-"*78)
            
            for name, res in results.items():
                mapes = res['mapes']
                print(f"{name:<20} {mapes.get('revenue', 0):>8.2f}%    {mapes.get('net_income', 0):>8.2f}%    {mapes.get('total_assets', 0):>8.2f}%    {mapes.get('total_equity', 0):>8.2f}%    {res['avg_mape']:>6.2f}%")
            
            print("-"*78)
            print(f"BEST: {best_approach} (MAPE: {results[best_approach]['avg_mape']:.2f}%)")
        
        return {
            'results': results,
            'best_approach': best_approach,
            'best_mape': results[best_approach]['avg_mape']
        }
    
    def _find_actual_column(self, df: pd.DataFrame, metric: str) -> Optional[str]:
        """Find column name for metric in actual data."""
        mappings = {
            'revenue': ['revenue', 'sales_revenue', 'totalRevenue'],
            'net_income': ['net_income', 'netIncome'],
            'total_assets': ['total_assets', 'totalAssets'],
            'total_equity': ['total_equity', 'totalEquity', 'totalStockholdersEquity'],
        }
        for col in mappings.get(metric, [metric]):
            if col in df.columns:
                return col
        return None
    
    def _calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        mask = actual != 0
        if mask.sum() == 0:
            return 0.0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def run_ensemble_forecast(data: pd.DataFrame,
                         ticker: str,
                         n_periods: int = 4,
                         validation_data: pd.DataFrame = None,
                         verbose: bool = True,
                         api_key: str = None,
                         reuse_q1: bool = True,
                         output_dir: str = "outputs") -> Dict:
    """
    Run complete ensemble forecast.
    
    Args:
        data: Historical training data
        ticker: Stock ticker
        n_periods: Quarters to forecast
        validation_data: Optional data for validation
        verbose: Print output
        api_key: Anthropic API key (from env var ANTHROPIC_API_KEY or explicit)
        reuse_q1: If True, reuse existing Q1 ML results for consistency
        output_dir: Directory for saving/loading cached results
        
    Returns:
        Dict with all predictions and evaluation results
        
    Note:
        API key should be set via:
        1. ANTHROPIC_API_KEY environment variable (recommended)
        2. --api-key command line argument
        3. Explicit api_key parameter
        
        NEVER hardcode API keys in source code.
    """
    if verbose:
        print("\n" + "="*70)
        print(f"ENSEMBLE FORECAST: {ticker}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        if api_key:
            print(f"API Key: ****...{api_key[-4:] if len(api_key) > 4 else '****'}")
        else:
            print("API Key: Not set (LLM features disabled)")
    
    # Initialize and fit (with Q1 reuse)
    ensemble = EnsembleForecaster(api_key=api_key, output_dir=output_dir)
    ensemble.fit(data, ticker=ticker, verbose=verbose, reuse_q1=reuse_q1)
    
    # Generate all forecasts
    if verbose:
        print("\n" + "-"*70)
        print("GENERATING FORECASTS")
        print("-"*70)
    
    predictions = ensemble.forecast_all_approaches(ticker, n_periods, verbose=verbose)
    
    # Evaluate if validation data provided
    evaluation = None
    if validation_data is not None and len(validation_data) > 0:
        if verbose:
            print("\n" + "-"*70)
            print("EVALUATING APPROACHES")
            print("-"*70)
        evaluation = ensemble.evaluate_approaches(predictions, validation_data, verbose=verbose)
    
    # Summary
    if verbose:
        print("\n" + "="*70)
        print("FORECAST SUMMARY")
        print("="*70)
        
        for approach_name, approach_data in predictions.items():
            print(f"\n{approach_name}:")
            if approach_name == 'Pure LLM':
                for i in range(min(n_periods, 4)):
                    print(f"  Q{i+1}: Revenue=${approach_data['revenue'][i]/1e9:.2f}B, "
                          f"NI=${approach_data['net_income'][i]/1e9:.2f}B")
            else:
                for i, stmt in enumerate(approach_data['statements'][:4]):
                    print(f"  Q{i+1}: Revenue=${stmt.revenue/1e9:.2f}B, "
                          f"NI=${stmt.net_income/1e9:.2f}B")
    
    return {
        'predictions': predictions,
        'evaluation': evaluation,
        'ticker': ticker,
        'n_periods': n_periods,
        'timestamp': datetime.now().isoformat()
    }
