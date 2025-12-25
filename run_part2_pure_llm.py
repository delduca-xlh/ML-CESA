#!/usr/bin/env python3
"""
run_part2_pure_llm.py - Pure LLM Forecasting vs ML

Compare:
- Part 1 (ML): Already saved in outputs/xgboost_models/{ticker}/
- Part 2 (LLM): Direct prediction of ALL values

Usage:
    python run_part2_pure_llm.py AAPL
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

import os
ANTHROPIC_API_KEY = os.environ.get(
    "ANTHROPIC_API_KEY",
    "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"
)


def check_part1_files(ticker: str):
    """Check if required Part 1 files exist."""
    part1_dir = Path(f'outputs/xgboost_models/{ticker.lower()}')
    
    required_files = [
        '03_development_data.csv',
        '04_test_actuals.csv',
        '05_test_ml_predictions.csv',
        '06_test_complete_statements.csv',
    ]
    
    missing = [f for f in required_files if not (part1_dir / f).exists()]
    return len(missing) == 0, missing


def run_part1_if_needed(ticker: str) -> bool:
    """Run Part 1 if files are missing."""
    files_ok, missing = check_part1_files(ticker)
    
    if files_ok:
        return True
    
    print(f"  Missing Part 1 files: {missing}")
    print(f"  Running Part 1 automatically...")
    print("-" * 60)
    
    import subprocess
    result = subprocess.run([sys.executable, 'auto_forecast_pipeline.py', ticker])
    
    print("-" * 60)
    
    files_ok, _ = check_part1_files(ticker)
    return files_ok


def llm_predict_all(ticker: str, development_data: pd.DataFrame, periods: int) -> dict:
    """
    LLM directly predicts ALL values: drivers + accounting metrics.
    """
    
    recent = development_data.tail(8)
    
    # Calculate summary stats for prompt
    avg_revenue = recent['sales_revenue'].mean()
    avg_cogs = recent['cost_of_goods_sold'].mean()
    avg_ni = recent['net_income'].mean() if 'net_income' in recent.columns else 0
    avg_ebit = recent['ebit'].mean() if 'ebit' in recent.columns else 0
    avg_assets = recent['total_assets'].mean() if 'total_assets' in recent.columns else 0
    avg_equity = recent['total_equity'].mean() if 'total_equity' in recent.columns else 0
    
    prompt = f"""You are a financial analyst forecasting {periods} quarters for {ticker}.

### Historical Data (Last 8 Quarters Averages):
- Revenue: ${avg_revenue/1e9:.2f}B
- COGS: ${avg_cogs/1e9:.2f}B  
- Gross Margin: {((avg_revenue - avg_cogs) / avg_revenue * 100):.1f}%
- Net Income: ${avg_ni/1e9:.2f}B
- EBIT: ${avg_ebit/1e9:.2f}B
- Total Assets: ${avg_assets/1e9:.2f}B
- Total Equity: ${avg_equity/1e9:.2f}B

### Historical Quarterly Revenue Trend (in $B):
{', '.join([f'Q{i+1}: {v/1e9:.1f}' for i, v in enumerate(recent['sales_revenue'].values)])}

### Task:
Predict the next {periods} quarters. Consider:
1. Seasonality (Q1 = holiday quarter for tech)
2. Recent trends
3. Your knowledge of {ticker}'s business outlook

### Respond with ONLY this JSON (all values in BILLIONS):
{{
    "predictions": [
        {{
            "quarter": 1,
            "revenue": XX.X,
            "cogs": XX.X,
            "overhead": X.X,
            "payroll": X.X,
            "capex": X.X,
            "net_income": XX.X,
            "ebit": XX.X,
            "total_assets": XXX.X,
            "total_equity": XX.X
        }},
        // ... repeat for all {periods} quarters
    ],
    "reasoning": "Brief explanation"
}}"""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text.strip()
        
        # Parse JSON
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
            predictions = result.get('predictions', [])
            
            if len(predictions) >= periods:
                preds = predictions[:periods]
                return {
                    # Drivers
                    'sales_revenue': np.array([p['revenue'] * 1e9 for p in preds]),
                    'cost_of_goods_sold': np.array([p['cogs'] * 1e9 for p in preds]),
                    'overhead_expenses': np.array([p['overhead'] * 1e9 for p in preds]),
                    'payroll_expenses': np.array([p['payroll'] * 1e9 for p in preds]),
                    'capex': np.array([p['capex'] * 1e9 for p in preds]),
                    # Accounting
                    'net_income': np.array([p['net_income'] * 1e9 for p in preds]),
                    'ebit': np.array([p['ebit'] * 1e9 for p in preds]),
                    'total_assets': np.array([p['total_assets'] * 1e9 for p in preds]),
                    'total_equity': np.array([p['total_equity'] * 1e9 for p in preds]),
                    'reasoning': result.get('reasoning', ''),
                    'source': 'LLM'
                }
        
        print(f"  ⚠ Failed to parse LLM response")
        
    except Exception as e:
        print(f"  ⚠ LLM Error: {e}")
    
    # Fallback to historical averages
    return {
        'sales_revenue': np.full(periods, avg_revenue),
        'cost_of_goods_sold': np.full(periods, avg_cogs),
        'overhead_expenses': np.full(periods, recent['overhead_expenses'].mean() if 'overhead_expenses' in recent.columns else 10e9),
        'payroll_expenses': np.full(periods, recent['payroll_expenses'].mean() if 'payroll_expenses' in recent.columns else 5e9),
        'capex': np.full(periods, recent['capex'].mean() if 'capex' in recent.columns else 3e9),
        'net_income': np.full(periods, avg_ni),
        'ebit': np.full(periods, avg_ebit),
        'total_assets': np.full(periods, avg_assets),
        'total_equity': np.full(periods, avg_equity),
        'reasoning': 'Fallback to historical averages',
        'source': 'Fallback'
    }


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate MAPE."""
    mask = actual != 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def run_comparison(ticker: str):
    """Run ML vs Pure LLM comparison."""
    
    print("=" * 80)
    print(f"ML vs PURE LLM COMPARISON: {ticker}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Check Part 1
    print("\n[1] Loading Part 1 (ML) results...")
    if not run_part1_if_needed(ticker):
        print("  ✗ Cannot proceed without Part 1")
        return
    
    part1_dir = Path(f'outputs/xgboost_models/{ticker.lower()}')
    
    # Load all Part 1 data
    development_data = pd.read_csv(part1_dir / '03_development_data.csv')
    test_actuals = pd.read_csv(part1_dir / '04_test_actuals.csv')
    ml_predictions = pd.read_csv(part1_dir / '05_test_ml_predictions.csv')
    ml_statements = pd.read_csv(part1_dir / '06_test_complete_statements.csv')
    
    periods = len(test_actuals)
    print(f"  ✓ Development data: {len(development_data)}Q")
    print(f"  ✓ Test periods: {periods}Q")
    print(f"  ✓ ML predictions loaded")
    
    # Generate LLM predictions
    print("\n[2] Generating Pure LLM predictions...")
    print(f"  (LLM only sees development data - no data leakage)")
    llm_preds = llm_predict_all(ticker, development_data, periods)
    print(f"  ✓ Source: {llm_preds.get('source', 'Unknown')}")
    
    if llm_preds.get('reasoning'):
        reasoning = llm_preds['reasoning']
        if len(reasoning) > 100:
            reasoning = reasoning[:100] + "..."
        print(f"  Reasoning: {reasoning}")
    
    # Compare Drivers
    print("\n" + "=" * 80)
    print("[3] DRIVER PREDICTIONS COMPARISON")
    print("=" * 80)
    
    driver_vars = [
        ('sales_revenue', 'Revenue'),
        ('cost_of_goods_sold', 'COGS'),
        ('overhead_expenses', 'Overhead'),
        ('payroll_expenses', 'Payroll'),
        ('capex', 'CapEx'),
    ]
    
    print(f"\n  {'Variable':<15s} {'Actual':<12s} {'ML':<12s} {'ML MAPE':<10s} {'LLM':<12s} {'LLM MAPE':<10s} {'Winner':<8s}")
    print(f"  " + "-" * 85)
    
    ml_driver_mapes = {}
    llm_driver_mapes = {}
    
    for var, name in driver_vars:
        if var in test_actuals.columns and var in ml_predictions.columns:
            actual = test_actuals[var].values
            ml_pred = ml_predictions[var].values
            llm_pred = llm_preds[var]
            
            ml_mape = calculate_mape(actual, ml_pred)
            llm_mape = calculate_mape(actual, llm_pred)
            
            ml_driver_mapes[name] = ml_mape
            llm_driver_mapes[name] = llm_mape
            
            winner = "ML" if ml_mape < llm_mape else "LLM" if llm_mape < ml_mape else "Tie"
            
            print(f"  {name:<15s} ${np.mean(actual)/1e9:>6.2f}B   ${np.mean(ml_pred)/1e9:>6.2f}B   {ml_mape:>7.2f}%   ${np.mean(llm_pred)/1e9:>6.2f}B   {llm_mape:>7.2f}%   {winner}")
    
    ml_driver_avg = np.mean(list(ml_driver_mapes.values()))
    llm_driver_avg = np.mean(list(llm_driver_mapes.values()))
    driver_winner = "ML" if ml_driver_avg < llm_driver_avg else "LLM"
    
    print(f"  " + "-" * 85)
    print(f"  {'AVERAGE':<15s} {'':12s} {'':12s} {ml_driver_avg:>7.2f}%   {'':12s} {llm_driver_avg:>7.2f}%   {driver_winner}")
    
    # Compare Accounting Metrics
    print("\n" + "=" * 80)
    print("[4] ACCOUNTING METRICS COMPARISON")
    print("=" * 80)
    
    accounting_vars = [
        ('net_income', 'is_net_income', 'Net Income'),
        ('ebit', 'is_ebit', 'EBIT'),
        ('total_assets', 'bs_total_assets', 'Total Assets'),
        ('total_equity', 'bs_total_equity', 'Total Equity'),
    ]
    
    print(f"\n  {'Metric':<15s} {'Actual':<12s} {'ML':<12s} {'ML MAPE':<10s} {'LLM':<12s} {'LLM MAPE':<10s} {'Winner':<8s}")
    print(f"  " + "-" * 85)
    
    ml_acc_mapes = {}
    llm_acc_mapes = {}
    
    for actual_col, ml_col, name in accounting_vars:
        if actual_col in test_actuals.columns:
            actual = test_actuals[actual_col].values
            ml_pred = ml_statements[ml_col].values[:len(actual)] if ml_col in ml_statements.columns else np.zeros(len(actual))
            llm_pred = llm_preds.get(actual_col, np.zeros(len(actual)))
            
            ml_mape = calculate_mape(actual, ml_pred)
            llm_mape = calculate_mape(actual, llm_pred)
            
            ml_acc_mapes[name] = ml_mape
            llm_acc_mapes[name] = llm_mape
            
            winner = "ML" if ml_mape < llm_mape else "LLM" if llm_mape < ml_mape else "Tie"
            
            print(f"  {name:<15s} ${np.mean(actual)/1e9:>6.2f}B   ${np.mean(ml_pred)/1e9:>6.2f}B   {ml_mape:>7.2f}%   ${np.mean(llm_pred)/1e9:>6.2f}B   {llm_mape:>7.2f}%   {winner}")
    
    ml_acc_avg = np.mean(list(ml_acc_mapes.values()))
    llm_acc_avg = np.mean(list(llm_acc_mapes.values()))
    acc_winner = "ML" if ml_acc_avg < llm_acc_avg else "LLM"
    
    print(f"  " + "-" * 85)
    print(f"  {'AVERAGE':<15s} {'':12s} {'':12s} {ml_acc_avg:>7.2f}%   {'':12s} {llm_acc_avg:>7.2f}%   {acc_winner}")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    overall_winner = 'ML' if ml_driver_avg + ml_acc_avg < llm_driver_avg + llm_acc_avg else 'LLM'
    
    print(f"""
┌─────────────────────────────────────────────────────────────┐
│                    ML vs Pure LLM Results                   │
├─────────────────────────────────────────────────────────────┤
│  DRIVER PREDICTIONS (5 variables):                          │
│    ML MAPE:        {ml_driver_avg:>6.2f}%                                  │
│    LLM MAPE:       {llm_driver_avg:>6.2f}%                                  │
│    Winner:         {driver_winner:<6s} (by {abs(ml_driver_avg - llm_driver_avg):.2f}%)                        │
├─────────────────────────────────────────────────────────────┤
│  ACCOUNTING METRICS (4 variables):                          │
│    ML MAPE:        {ml_acc_avg:>6.2f}%                                  │
│    LLM MAPE:       {llm_acc_avg:>6.2f}%                                  │
│    Winner:         {acc_winner:<6s} (by {abs(ml_acc_avg - llm_acc_avg):.2f}%)                        │
├─────────────────────────────────────────────────────────────┤
│  OVERALL WINNER:   {overall_winner:<6s}                                      │
└─────────────────────────────────────────────────────────────┘
""")
    
    # Save results
    output_dir = Path(f'outputs/part2_pure_llm/{ticker.lower()}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LLM predictions
    llm_df = pd.DataFrame({
        'sales_revenue': llm_preds['sales_revenue'],
        'cost_of_goods_sold': llm_preds['cost_of_goods_sold'],
        'overhead_expenses': llm_preds['overhead_expenses'],
        'payroll_expenses': llm_preds['payroll_expenses'],
        'capex': llm_preds['capex'],
        'net_income': llm_preds['net_income'],
        'ebit': llm_preds['ebit'],
        'total_assets': llm_preds['total_assets'],
        'total_equity': llm_preds['total_equity'],
    })
    llm_df.to_csv(output_dir / 'llm_predictions.csv', index=False)
    
    with open(output_dir / 'comparison.json', 'w') as f:
        json.dump({
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'driver_mapes': {'ml': ml_driver_mapes, 'llm': llm_driver_mapes},
            'accounting_mapes': {'ml': ml_acc_mapes, 'llm': llm_acc_mapes},
            'summary': {
                'ml_driver_avg': ml_driver_avg,
                'llm_driver_avg': llm_driver_avg,
                'ml_accounting_avg': ml_acc_avg,
                'llm_accounting_avg': llm_acc_avg,
                'driver_winner': driver_winner,
                'accounting_winner': acc_winner,
                'overall_winner': overall_winner,
            },
            'llm_reasoning': llm_preds.get('reasoning', ''),
        }, f, indent=2)
    
    print(f"  ✓ Results saved to: {output_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_part2_pure_llm.py TICKER")
        print("Example: python run_part2_pure_llm.py AAPL")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    run_comparison(ticker)
