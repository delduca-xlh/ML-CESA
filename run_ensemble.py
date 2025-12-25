#!/usr/bin/env python3
"""
run_ensemble.py - Ensemble Model: Best Complete Approach

Compare complete approaches and select the one with lowest overall MAPE.
This ensures accounting formulas remain consistent.

Approaches:
1. ML + Historical Ratios (Part 1)
2. ML + LLM Ratios (Part 2)
3. Pure LLM (direct prediction)

Usage:
    python run_ensemble.py AAPL
"""

import sys
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from financial_planning.models.accounting_engine import AccountingEngine

import os
ANTHROPIC_API_KEY = os.environ.get(
    "ANTHROPIC_API_KEY",
    "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"
)


def check_and_run_part1(ticker: str) -> bool:
    """Ensure Part 1 has been run."""
    part1_dir = Path(f'outputs/xgboost_models/{ticker.lower()}')
    required = ['03_development_data.csv', '04_test_actuals.csv', 
                '05_test_ml_predictions.csv', '06_test_complete_statements.csv',
                '03_historical_ratios.json']
    
    missing = [f for f in required if not (part1_dir / f).exists()]
    
    if not missing:
        return True
    
    print(f"  Missing Part 1 files: {missing}")
    print(f"  Running Part 1...")
    import subprocess
    subprocess.run([sys.executable, 'auto_forecast_pipeline.py', ticker])
    
    missing = [f for f in required if not (part1_dir / f).exists()]
    return len(missing) == 0


def generate_llm_ratios(ticker: str, development_data: pd.DataFrame) -> dict:
    """Generate LLM-based ratios."""
    recent = development_data.tail(8)
    
    avg_revenue = recent['sales_revenue'].mean()
    avg_cogs = recent['cost_of_goods_sold'].mean()
    gross_margin = (avg_revenue - avg_cogs) / avg_revenue
    
    avg_ni = recent['net_income'].mean() if 'net_income' in recent.columns else 0
    ni_margin = avg_ni / avg_revenue if avg_revenue > 0 else 0.25
    
    avg_ebit = recent['ebit'].mean() if 'ebit' in recent.columns else 0
    ebit_margin = avg_ebit / avg_revenue if avg_revenue > 0 else 0.30
    
    total_ni = recent['net_income'].sum() if 'net_income' in recent.columns else 0
    total_div = abs(recent['dividends_paid'].sum()) if 'dividends_paid' in recent.columns else 0
    total_buyback = abs(recent['stock_repurchased'].sum()) if 'stock_repurchased' in recent.columns else 0
    retention = 1 - (total_div + total_buyback) / total_ni if total_ni > 0 else 0
    
    prompt = f"""You are a financial analyst predicting ratios for {ticker}.

Historical (Last 8Q):
- Gross Margin: {gross_margin:.1%}
- Net Income Margin: {ni_margin:.1%}  
- EBIT Margin: {ebit_margin:.1%}
- Retention Ratio: {retention:.1%}

Predict ratios for next 8 quarters. Consider trends and your knowledge of {ticker}.

Respond ONLY with JSON:
{{
    "gross_margin": 0.XX,
    "net_income_margin": 0.XX,
    "ebit_margin": 0.XX,
    "retention_ratio": X.XX,
    "reasoning": "brief"
}}"""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = response.content[0].text.strip()
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return {
                'gross_margin': result.get('gross_margin', gross_margin),
                'avg_net_income_margin': result.get('net_income_margin', ni_margin),
                'avg_ebit_margin': result.get('ebit_margin', ebit_margin),
                'retention_ratio': result.get('retention_ratio', retention),
                'reasoning': result.get('reasoning', ''),
                'source': 'LLM'
            }
    except Exception as e:
        print(f"  LLM ratio error: {e}")
    
    return {
        'gross_margin': gross_margin,
        'avg_net_income_margin': ni_margin,
        'avg_ebit_margin': ebit_margin,
        'retention_ratio': retention,
        'reasoning': 'Fallback to historical',
        'source': 'Fallback'
    }


def generate_pure_llm_predictions(ticker: str, development_data: pd.DataFrame, periods: int) -> dict:
    """LLM directly predicts all values."""
    recent = development_data.tail(8)
    
    avg_revenue = recent['sales_revenue'].mean()
    avg_cogs = recent['cost_of_goods_sold'].mean()
    avg_ni = recent['net_income'].mean() if 'net_income' in recent.columns else 0
    avg_ebit = recent['ebit'].mean() if 'ebit' in recent.columns else 0
    avg_assets = recent['total_assets'].mean() if 'total_assets' in recent.columns else 0
    avg_equity = recent['total_equity'].mean() if 'total_equity' in recent.columns else 0
    
    prompt = f"""Forecast {periods} quarters for {ticker}.

Historical Averages (8Q):
- Revenue: ${avg_revenue/1e9:.2f}B, COGS: ${avg_cogs/1e9:.2f}B
- Net Income: ${avg_ni/1e9:.2f}B, EBIT: ${avg_ebit/1e9:.2f}B
- Total Assets: ${avg_assets/1e9:.2f}B, Equity: ${avg_equity/1e9:.2f}B

Revenue Trend: {', '.join([f'{v/1e9:.1f}' for v in recent['sales_revenue'].values[-4:]])}B

Respond ONLY with JSON (values in BILLIONS):
{{
    "predictions": [
        {{"q":1, "revenue":XX, "cogs":XX, "overhead":X, "payroll":X, "capex":X, "net_income":XX, "ebit":XX, "total_assets":XXX, "total_equity":XX}},
        // ... for all {periods} quarters
    ],
    "reasoning": "brief"
}}"""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = response.content[0].text.strip()
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            result = json.loads(match.group())
            preds = result.get('predictions', [])[:periods]
            
            if len(preds) >= periods:
                return {
                    'sales_revenue': np.array([p['revenue'] * 1e9 for p in preds]),
                    'cost_of_goods_sold': np.array([p['cogs'] * 1e9 for p in preds]),
                    'overhead_expenses': np.array([p['overhead'] * 1e9 for p in preds]),
                    'payroll_expenses': np.array([p['payroll'] * 1e9 for p in preds]),
                    'capex': np.array([p['capex'] * 1e9 for p in preds]),
                    'net_income': np.array([p['net_income'] * 1e9 for p in preds]),
                    'ebit': np.array([p['ebit'] * 1e9 for p in preds]),
                    'total_assets': np.array([p['total_assets'] * 1e9 for p in preds]),
                    'total_equity': np.array([p['total_equity'] * 1e9 for p in preds]),
                    'reasoning': result.get('reasoning', ''),
                    'source': 'Pure LLM'
                }
    except Exception as e:
        print(f"  Pure LLM error: {e}")
    
    # Fallback
    return {
        'sales_revenue': np.full(periods, avg_revenue),
        'cost_of_goods_sold': np.full(periods, avg_cogs),
        'overhead_expenses': np.full(periods, recent.get('overhead_expenses', pd.Series([10e9])).mean()),
        'payroll_expenses': np.full(periods, recent.get('payroll_expenses', pd.Series([5e9])).mean()),
        'capex': np.full(periods, recent.get('capex', pd.Series([3e9])).mean()),
        'net_income': np.full(periods, avg_ni),
        'ebit': np.full(periods, avg_ebit),
        'total_assets': np.full(periods, avg_assets),
        'total_equity': np.full(periods, avg_equity),
        'reasoning': 'Fallback to historical',
        'source': 'Fallback'
    }


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate MAPE."""
    mask = actual != 0
    if mask.sum() == 0:
        return float('inf')
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def run_ensemble(ticker: str):
    """Run ensemble model - select best complete approach."""
    
    print("=" * 80)
    print(f"ENSEMBLE MODEL: {ticker}")
    print(f"Select Best Complete Approach (Accounting Consistent)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Ensure Part 1 exists
    print("\n[1] Loading Part 1 data...")
    if not check_and_run_part1(ticker):
        print("  ✗ Part 1 failed")
        return
    
    part1_dir = Path(f'outputs/xgboost_models/{ticker.lower()}')
    
    # Load data
    development_data = pd.read_csv(part1_dir / '03_development_data.csv')
    test_actuals = pd.read_csv(part1_dir / '04_test_actuals.csv')
    ml_predictions = pd.read_csv(part1_dir / '05_test_ml_predictions.csv')
    ml_hist_statements = pd.read_csv(part1_dir / '06_test_complete_statements.csv')
    
    periods = len(test_actuals)
    print(f"  ✓ Development: {len(development_data)}Q, Test: {periods}Q")
    
    # Generate LLM ratios and build ML + LLM statements
    print("\n[2] Building Approach 2: ML + LLM Ratios...")
    llm_ratios = generate_llm_ratios(ticker, development_data)
    print(f"  LLM Ratios: GM={llm_ratios['gross_margin']:.1%}, NIM={llm_ratios['avg_net_income_margin']:.1%}, Retention={llm_ratios['retention_ratio']:.1%}")
    
    ml_preds_dict = {col: ml_predictions[col].values for col in ml_predictions.columns}
    engine = AccountingEngine(development_data.tail(20))
    engine.set_assumptions(llm_ratios)
    ml_llm_statements = engine.build_complete_statements(predictions=ml_preds_dict, periods=periods)
    print(f"  ✓ ML + LLM Ratios statements built")
    
    # Generate Pure LLM predictions
    print("\n[3] Building Approach 3: Pure LLM...")
    pure_llm = generate_pure_llm_predictions(ticker, development_data, periods)
    print(f"  ✓ Pure LLM: {pure_llm.get('source', 'Unknown')}")
    
    # Define metrics to compare
    driver_vars = ['sales_revenue', 'cost_of_goods_sold', 'overhead_expenses', 'payroll_expenses', 'capex']
    acct_vars = [
        ('net_income', 'is_net_income'),
        ('ebit', 'is_ebit'),
        ('total_assets', 'bs_total_assets'),
        ('total_equity', 'bs_total_equity'),
    ]
    
    # Build approaches dictionary
    approaches = {
        'ML + Historical': {
            'drivers': {var: ml_predictions[var].values for var in driver_vars},
            'accounting': {var: ml_hist_statements[col].values[:periods] for var, col in acct_vars},
            'statements': ml_hist_statements,
        },
        'ML + LLM Ratios': {
            'drivers': {var: ml_predictions[var].values for var in driver_vars},
            'accounting': {var: ml_llm_statements[col].values[:periods] for var, col in acct_vars},
            'statements': ml_llm_statements,
        },
        'Pure LLM': {
            'drivers': {var: pure_llm[var] for var in driver_vars},
            'accounting': {var: pure_llm[var] for var, _ in acct_vars},
            'statements': None,
        },
    }
    
    # Calculate MAPEs for each approach
    print("\n" + "=" * 80)
    print("[4] COMPARING APPROACHES")
    print("=" * 80)
    
    approach_results = {}
    
    for approach_name, approach_data in approaches.items():
        driver_mapes = {}
        acct_mapes = {}
        
        for var in driver_vars:
            if var in test_actuals.columns:
                actual = test_actuals[var].values
                pred = approach_data['drivers'][var][:len(actual)]
                driver_mapes[var] = calculate_mape(actual, pred)
        
        for var, _ in acct_vars:
            if var in test_actuals.columns:
                actual = test_actuals[var].values
                pred = approach_data['accounting'][var][:len(actual)]
                acct_mapes[var] = calculate_mape(actual, pred)
        
        avg_driver = np.mean(list(driver_mapes.values()))
        avg_acct = np.mean(list(acct_mapes.values()))
        overall = (avg_driver + avg_acct) / 2
        
        approach_results[approach_name] = {
            'driver_mapes': driver_mapes,
            'acct_mapes': acct_mapes,
            'avg_driver': avg_driver,
            'avg_acct': avg_acct,
            'overall': overall,
        }
    
    # Print comparison table
    print(f"\n  {'Approach':<20s} {'Driver MAPE':<15s} {'Acct MAPE':<15s} {'Overall':<12s}")
    print(f"  " + "-" * 62)
    
    for name, result in approach_results.items():
        print(f"  {name:<20s} {result['avg_driver']:>8.2f}%      {result['avg_acct']:>8.2f}%      {result['overall']:>8.2f}%")
    
    # Find best approach
    best_approach = min(approach_results, key=lambda x: approach_results[x]['overall'])
    best_result = approach_results[best_approach]
    
    print(f"  " + "-" * 62)
    print(f"  BEST: {best_approach} (Overall MAPE: {best_result['overall']:.2f}%)")
    
    # Detailed breakdown of best approach
    print("\n" + "=" * 80)
    print(f"[5] BEST APPROACH: {best_approach}")
    print("=" * 80)
    
    print(f"\n  A. Driver Predictions:")
    print(f"  {'Variable':<25s} {'Actual (Avg)':<15s} {'Predicted':<15s} {'MAPE':<10s}")
    print(f"  " + "-" * 65)
    
    best_drivers = approaches[best_approach]['drivers']
    for var in driver_vars:
        if var in test_actuals.columns:
            actual_avg = np.mean(test_actuals[var].values)
            pred_avg = np.mean(best_drivers[var])
            mape = best_result['driver_mapes'][var]
            print(f"  {var:<25s} ${actual_avg/1e9:>7.2f}B       ${pred_avg/1e9:>7.2f}B       {mape:>6.2f}%")
    
    print(f"  " + "-" * 65)
    print(f"  {'Overall Driver MAPE':<25s} {'':15s} {'':15s} {best_result['avg_driver']:>6.2f}%")
    
    print(f"\n  B. Accounting Metrics:")
    print(f"  {'Metric':<25s} {'Actual (Avg)':<15s} {'Predicted':<15s} {'MAPE':<10s}")
    print(f"  " + "-" * 65)
    
    best_acct = approaches[best_approach]['accounting']
    for var, _ in acct_vars:
        if var in test_actuals.columns:
            actual_avg = np.mean(test_actuals[var].values)
            pred_avg = np.mean(best_acct[var])
            mape = best_result['acct_mapes'][var]
            print(f"  {var:<25s} ${actual_avg/1e9:>7.2f}B       ${pred_avg/1e9:>7.2f}B       {mape:>6.2f}%")
    
    print(f"  " + "-" * 65)
    print(f"  {'Overall Accounting MAPE':<25s} {'':15s} {'':15s} {best_result['avg_acct']:>6.2f}%")
    
    # Final summary
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           ENSEMBLE FINAL RESULTS                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Ticker: {ticker:<10s}           Test Periods: {periods} quarters                   ║
║                                                                              ║
║   ┌─────────────────────────────────────────────────────────────────────┐   ║
║   │  APPROACH COMPARISON                                                │   ║
║   ├─────────────────────────────────────────────────────────────────────┤   ║
║   │    ML + Historical:    {approach_results['ML + Historical']['overall']:>6.2f}%                                    │   ║
║   │    ML + LLM Ratios:    {approach_results['ML + LLM Ratios']['overall']:>6.2f}%                                    │   ║
║   │    Pure LLM:           {approach_results['Pure LLM']['overall']:>6.2f}%                                    │   ║
║   └─────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
║   ┌─────────────────────────────────────────────────────────────────────┐   ║
║   │  WINNER: {best_approach:<20s}                                    │   ║
║   ├─────────────────────────────────────────────────────────────────────┤   ║
║   │    Driver MAPE:        {best_result['avg_driver']:>6.2f}%                                    │   ║
║   │    Accounting MAPE:    {best_result['avg_acct']:>6.2f}%                                    │   ║
║   │    ─────────────────────────────────                                │   ║
║   │    OVERALL MAPE:       {best_result['overall']:>6.2f}%                                    │   ║
║   └─────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Grade
    if best_result['overall'] < 10:
        grade = "⭐⭐⭐⭐⭐ Excellent"
    elif best_result['overall'] < 15:
        grade = "⭐⭐⭐⭐ Very Good"
    elif best_result['overall'] < 20:
        grade = "⭐⭐⭐ Good"
    else:
        grade = "⭐⭐ Fair"
    
    print(f"  Grade: {grade}")
    
    # Save results
    print("\n[6] Saving results...")
    output_dir = Path(f'outputs/ensemble/{ticker.lower()}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best approach predictions
    best_preds_df = pd.DataFrame({
        **{var: best_drivers[var] for var in driver_vars},
        **{var: best_acct[var] for var, _ in acct_vars},
    })
    best_preds_df.to_csv(output_dir / 'ensemble_predictions.csv', index=False)
    
    # Save summary
    summary_rows = []
    for var in driver_vars:
        if var in test_actuals.columns:
            summary_rows.append({
                'variable': var,
                'type': 'driver',
                'actual_avg': np.mean(test_actuals[var].values),
                'predicted_avg': np.mean(best_drivers[var]),
                'mape': best_result['driver_mapes'][var],
            })
    for var, _ in acct_vars:
        if var in test_actuals.columns:
            summary_rows.append({
                'variable': var,
                'type': 'accounting',
                'actual_avg': np.mean(test_actuals[var].values),
                'predicted_avg': np.mean(best_acct[var]),
                'mape': best_result['acct_mapes'][var],
            })
    
    pd.DataFrame(summary_rows).to_csv(output_dir / 'ensemble_summary.csv', index=False)
    
    # Save detailed JSON
    with open(output_dir / 'ensemble_results.json', 'w') as f:
        json.dump({
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'test_periods': periods,
            'best_approach': best_approach,
            'approach_comparison': {
                name: {
                    'driver_mape': res['avg_driver'],
                    'acct_mape': res['avg_acct'],
                    'overall_mape': res['overall'],
                }
                for name, res in approach_results.items()
            },
            'best_approach_details': {
                'driver_mapes': best_result['driver_mapes'],
                'acct_mapes': best_result['acct_mapes'],
                'avg_driver_mape': best_result['avg_driver'],
                'avg_acct_mape': best_result['avg_acct'],
                'overall_mape': best_result['overall'],
            },
            'grade': grade,
            'llm_ratios': {k: v for k, v in llm_ratios.items() if k != 'reasoning'},
            'llm_reasoning': llm_ratios.get('reasoning', ''),
        }, f, indent=2)
    
    print(f"  ✓ Results saved to: {output_dir}/")
    
    print("\n" + "=" * 80)
    print("ENSEMBLE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_ensemble.py TICKER")
        print("Example: python run_ensemble.py AAPL")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    run_ensemble(ticker)