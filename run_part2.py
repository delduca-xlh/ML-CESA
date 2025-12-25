#!/usr/bin/env python3
"""
run_part2.py - Simple Part 2 Runner

This script runs Part 2 analysis on top of Part 1 results.
It's minimal and uses the existing Part 1 infrastructure.

Usage:
    python run_part2.py AAPL

Requirements:
    - Part 1 must be run first: python auto_forecast_pipeline.py AAPL
    - Or this script will run Part 1 automatically
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from financial_planning.models.balance_sheet_forecaster import BalanceSheetForecaster, ForecastConfig
from financial_planning.utils.fmp_data_fetcher import FMPDataFetcher
from financial_planning.models.accounting_engine import AccountingEngine

# Import LLM generator (place in utils folder)
try:
    from financial_planning.utils.llm_assumption_generator import generate_assumptions, generate_cfo_report, print_assumptions
except ImportError:
    # Fallback: import from outputs if not in utils yet
    sys.path.insert(0, str(project_root / 'outputs'))
    from llm_assumption_generator import generate_assumptions, generate_cfo_report, print_assumptions


def load_part1_results(ticker: str):
    """Load Part 1 results from saved files."""
    output_dir = Path(f'outputs/xgboost_models/{ticker.lower()}')
    
    if not output_dir.exists():
        return None
    
    results = {}
    
    # Load ML predictions
    pred_file = output_dir / '05_test_ml_predictions.csv'
    if pred_file.exists():
        results['test_predictions'] = pd.read_csv(pred_file)
    
    # Load test statements
    stmt_file = output_dir / '06_test_complete_statements.csv'
    if stmt_file.exists():
        results['test_statements'] = pd.read_csv(stmt_file)
    
    # Load evaluation
    eval_file = output_dir / '07_test_evaluation.json'
    if eval_file.exists():
        with open(eval_file) as f:
            results['evaluation'] = json.load(f)
    
    return results if results else None


def check_part1_files(ticker: str) -> bool:
    """Check if all required Part 1 files exist."""
    part1_dir = Path(f'outputs/xgboost_models/{ticker.lower()}')
    
    required_files = [
        '02_data_split.json',
        '03_development_data.csv',      # For LLM context (no data leakage)
        '03_historical_ratios.json',
        '04_test_actuals.csv',
        '05_test_ml_predictions.csv',
        '06_test_complete_statements.csv',
        '07_test_evaluation.json',
    ]
    
    missing = []
    for f in required_files:
        if not (part1_dir / f).exists():
            missing.append(f)
    
    return len(missing) == 0, missing


def run_part1_if_needed(ticker: str) -> bool:
    """Run Part 1 if files are missing. Returns True if successful."""
    files_ok, missing = check_part1_files(ticker)
    
    if files_ok:
        return True
    
    print(f"  Missing Part 1 files: {missing}")
    print(f"  Running Part 1 automatically...")
    print("-" * 60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'auto_forecast_pipeline.py', ticker],
        capture_output=False  # Show output in real-time
    )
    
    print("-" * 60)
    
    if result.returncode != 0:
        print(f"  ✗ Part 1 failed!")
        return False
    
    # Verify files now exist
    files_ok, still_missing = check_part1_files(ticker)
    if not files_ok:
        print(f"  ✗ Part 1 completed but files still missing: {still_missing}")
        return False
    
    print(f"  ✓ Part 1 completed successfully")
    return True


def run_part2(ticker: str):
    """Run Part 2 analysis."""
    
    print("=" * 80)
    print(f"PART 2: LLM-ENHANCED FORECASTING FOR {ticker}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Check for Part 1 results and run if needed
    print("\n[1] Checking Part 1 results...")
    
    if not run_part1_if_needed(ticker):
        print("  Cannot proceed without Part 1 results.")
        return
    
    # Load Part 1 results
    part1_results = load_part1_results(ticker)
    if part1_results is None:
        print("  ✗ Failed to load Part 1 results")
        return
    
    print(f"  ✓ Part 1 results loaded")
    print(f"  Part 1 ML MAPE: {part1_results['evaluation'].get('test_mape', 'N/A'):.2f}%")
    
    # Load saved data from Part 1
    print("\n[2] Loading Part 1 saved data...")
    
    part1_dir = Path(f'outputs/xgboost_models/{ticker.lower()}')
    
    # Load historical ratios from Part 1
    with open(part1_dir / '03_historical_ratios.json') as f:
        part1_ratios = json.load(f)
    print(f"  ✓ Loaded Part 1 historical ratios")
    print(f"    Gross Margin: {part1_ratios['gross_margin']:.2%}")
    print(f"    Net Income Margin: {part1_ratios['avg_net_income_margin']:.2%}")
    print(f"    Retention Ratio: {part1_ratios['retention_ratio']:.2%}")
    
    # Load test actuals from Part 1
    test_actuals = pd.read_csv(part1_dir / '04_test_actuals.csv')
    print(f"  ✓ Loaded test actuals: {len(test_actuals)} periods")
    
    # Generate LLM assumptions
    print("\n[3] Generating LLM assumptions...")
    
    # IMPORTANT: Use ONLY development data (training period) to prevent data leakage
    # LLM should NOT see test period data
    dev_data_file = part1_dir / '03_development_data.csv'
    if dev_data_file.exists():
        llm_context_data = pd.read_csv(dev_data_file)
        print(f"  ✓ Using Part 1 development data: {len(llm_context_data)}Q")
        print(f"  (No data leakage - LLM only sees training period)")
    else:
        print(f"  ✗ Development data not found. Re-run Part 1.")
        return
    
    llm_assumptions = generate_assumptions(ticker, llm_context_data)
    print_assumptions(llm_assumptions, "LLM Assumptions")
    
    # Show Part 1 ratios for comparison
    print(f"\n  Part 1 Historical Ratios (for comparison):")
    print(f"  " + "-" * 50)
    print(f"  Gross Margin:      {part1_ratios['gross_margin']:.2%}")
    print(f"  Net Income Margin: {part1_ratios['avg_net_income_margin']:.2%}")
    print(f"  EBIT Margin:       {part1_ratios['avg_ebit_margin']:.2%}")
    print(f"  CapEx/Revenue:     {part1_ratios['capex_to_revenue']:.2%}")
    print(f"  Retention Ratio:   {part1_ratios['retention_ratio']:.2%}")
    
    # Build Part 2 statements
    print("\n[4] Building Part 2 statements with LLM assumptions...")
    
    # Convert test predictions to dict format
    test_preds = part1_results['test_predictions']
    predictions_dict = {col: test_preds[col].values for col in test_preds.columns}
    
    # Create engine with test_actuals as base (for balance sheet starting values)
    engine = AccountingEngine(test_actuals)
    engine.set_assumptions(llm_assumptions)
    
    # Build statements
    part2_statements = engine.build_complete_statements(
        predictions=predictions_dict,
        periods=len(test_preds)
    )
    
    # Calculate Part 2 MAPE
    print("\n[5] Calculating Part 2 MAPE...")
    
    # Load Part 1 complete statements
    part1_statements_file = part1_dir / '06_test_complete_statements.csv'
    if part1_statements_file.exists():
        part1_statements = pd.read_csv(part1_statements_file)
    else:
        print(f"  ✗ Part 1 statements not found!")
        return
    
    # Use test_actuals already loaded above
    actual_test = test_actuals
    print(f"  Using {len(actual_test)} test periods for comparison")
    
    key_metrics = [
        ('net_income', 'is_net_income', 'Net Income'),
        ('ebit', 'is_ebit', 'EBIT'),
        ('total_assets', 'bs_total_assets', 'Total Assets'),
        ('total_equity', 'bs_total_equity', 'Total Equity'),
    ]
    
    # Calculate MAPE for BOTH Part 1 and Part 2 using same actuals
    print(f"\n  {'Metric':<18s} {'Actual':<12s} {'Part1':<12s} {'P1 MAPE':<10s} {'Part2':<12s} {'P2 MAPE':<10s}")
    print(f"  " + "-" * 74)
    
    part1_mapes = {}
    part2_mapes = {}
    
    for actual_col, pred_col, display_name in key_metrics:
        if actual_col in actual_test.columns and pred_col in part2_statements.columns:
            actual_vals = actual_test[actual_col].values
            part1_pred_vals = part1_statements[pred_col].values[:len(actual_vals)]
            part2_pred_vals = part2_statements[pred_col].values[:len(actual_vals)]
            
            valid_mask = actual_vals != 0
            if valid_mask.sum() > 0:
                mape1 = np.mean(np.abs((actual_vals[valid_mask] - part1_pred_vals[valid_mask]) / actual_vals[valid_mask])) * 100
                mape2 = np.mean(np.abs((actual_vals[valid_mask] - part2_pred_vals[valid_mask]) / actual_vals[valid_mask])) * 100
                
                part1_mapes[display_name] = mape1
                part2_mapes[display_name] = mape2
                
                actual_avg = np.mean(actual_vals)
                p1_avg = np.mean(part1_pred_vals)
                p2_avg = np.mean(part2_pred_vals)
                
                print(f"  {display_name:<18s} ${actual_avg/1e9:>6.2f}B   ${p1_avg/1e9:>6.2f}B   {mape1:>7.2f}%   ${p2_avg/1e9:>6.2f}B   {mape2:>7.2f}%")
    
    part1_overall = np.mean(list(part1_mapes.values())) if part1_mapes else 0
    part2_overall = np.mean(list(part2_mapes.values())) if part2_mapes else 0
    
    print(f"  " + "-" * 74)
    print(f"  {'Overall MAPE':<18s} {'':12s} {'':12s} {part1_overall:>7.2f}%   {'':12s} {part2_overall:>7.2f}%")
    
    # Compare with Part 1
    print("\n[6] Comparison: Part 1 vs Part 2...")
    diff = part2_overall - part1_overall
    
    print(f"\n  Part 1 (Historical) Accounting MAPE: {part1_overall:.2f}%")
    print(f"  Part 2 (LLM)        Accounting MAPE: {part2_overall:.2f}%")
    print(f"  Difference:                          {diff:+.2f}%")
    
    if diff < 0:
        print(f"\n  ✓ Part 2 (LLM) is BETTER by {-diff:.2f}%")
        winner = "Part 2 (LLM)"
    else:
        print(f"\n  ✗ Part 1 (Historical) is BETTER by {diff:.2f}%")
        winner = "Part 1 (Historical)"
    
    # Generate CFO Report
    print("\n[7] Generating CFO Report...")
    cfo_report = generate_cfo_report(ticker, llm_context_data, part2_statements, llm_assumptions)
    
    print("\n" + "=" * 80)
    print("CFO RECOMMENDATION REPORT")
    print("=" * 80)
    print(cfo_report)
    
    # Save results
    print("\n[8] Saving Part 2 results...")
    part2_output_dir = Path(f'outputs/part2_results/{ticker.lower()}')
    part2_output_dir.mkdir(parents=True, exist_ok=True)
    
    part2_statements.to_csv(part2_output_dir / 'part2_statements.csv', index=False)
    
    with open(part2_output_dir / 'llm_assumptions.json', 'w') as f:
        json.dump(llm_assumptions, f, indent=2)
    
    with open(part2_output_dir / 'cfo_report.md', 'w') as f:
        f.write(f"# CFO Report: {ticker}\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(cfo_report)
    
    with open(part2_output_dir / 'comparison.json', 'w') as f:
        json.dump({
            'part1_mapes': part1_mapes,
            'part2_mapes': part2_mapes,
            'part1_overall': part1_overall,
            'part2_overall': part2_overall,
            'winner': winner,
            'llm_source': llm_assumptions.get('source', 'Unknown')
        }, f, indent=2)
    
    print(f"  ✓ Saved to: {part2_output_dir}/")
    
    # Summary
    print("\n" + "=" * 80)
    print("PART 2 COMPLETE")
    print("=" * 80)
    print(f"""
Ticker: {ticker}

Part 1 (Historical Assumptions): Accounting MAPE = {part1_overall:.2f}%
Part 2 (LLM Assumptions):        Accounting MAPE = {part2_overall:.2f}%

LLM Source: {llm_assumptions.get('source', 'Unknown')}
Winner: {winner}

Output Files:
  - {part2_output_dir}/part2_statements.csv
  - {part2_output_dir}/llm_assumptions.json
  - {part2_output_dir}/cfo_report.md
  - {part2_output_dir}/comparison.json
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_part2.py TICKER")
        print("Example: python run_part2.py AAPL")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    run_part2(ticker)