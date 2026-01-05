"""
rolling_validator.py

Rolling Validation for Balance Sheet Simulator.
Train on [1, t], predict t+1, compare to actual.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .data_structures import QuantileForecast
from .quantile_simulator import QuantileSimulator
from .statement_printer import print_complete_statements
from .pdf_report import generate_statement_pdf, setup_output_folder, HAS_REPORTLAB


def run_rolling_validation(data: pd.DataFrame,
                           min_train_periods: int = 20,
                           n_test_periods: Optional[int] = None,
                           seq_length: int = 4,
                           verbose: bool = True,
                           ticker: str = "SAMPLE",
                           save_pdf: bool = True) -> List[Dict]:
    """
    Run rolling validation.
    
    For each period t from min_train to T-1:
        1. Train model on periods [1, t]
        2. Predict for period t+1
        3. Compare to actual t+1
        4. Save full statements to PDF (if reportlab installed)
    """
    
    if verbose:
        print("\n" + "="*70)
        print("ROLLING VALIDATION")
        print("="*70)
        print(f"Data periods: {len(data)} | Min train: {min_train_periods}")
    
    # Setup output folder for PDFs (only if reportlab available)
    output_folder = None
    if save_pdf and HAS_REPORTLAB:
        output_folder = setup_output_folder(ticker)
        if verbose:
            print(f"PDF output: {output_folder}")
    
    results = []
    max_test = len(data) - min_train_periods - 1
    if n_test_periods is None:
        n_test_periods = max_test
    else:
        n_test_periods = min(n_test_periods, max_test)
    
    if n_test_periods <= 0:
        print(f"Error: Not enough data. Have {len(data)}, need at least {min_train_periods + 2}.")
        return []
    
    start_t = len(data) - n_test_periods - 1
    
    if verbose:
        print(f"Validation rounds: {n_test_periods}")
        print("="*70)
    
    for round_num, t in enumerate(range(start_t, len(data) - 1)):
        if verbose:
            print(f"\nRound {round_num+1}/{n_test_periods}: Training...", end=" ", flush=True)
        
        # Train on data up to t
        train_data = data.iloc[:t+1]
        
        # Initialize and train
        simulator = QuantileSimulator(seq_length=seq_length)
        simulator.fit(train_data, verbose=False)
        
        if verbose:
            print("Predicting...", end=" ", flush=True)
        
        # Predict
        forecasts = simulator.predict_distribution(simulator.last_drivers)
        
        if verbose:
            print("Done.")
        
        # Get median predictions
        driver_values = {d: forecasts[d].q50 for d in simulator.available_drivers}
        driver_values.setdefault('revenue_growth', 0.02)
        driver_values.setdefault('cogs_margin', 0.60)
        driver_values.setdefault('opex_margin', 0.15)
        driver_values.setdefault('capex_ratio', 0.03)
        driver_values.setdefault('net_margin', 0.15)
        
        # Derive statements
        prior = simulator.create_prior_statements(train_data.iloc[-1])
        predicted = simulator.accounting_engine.derive_statements(
            drivers=driver_values, prior=prior, period=f"Period {t+2}"
        )
        
        # Get actual
        actual = data.iloc[t+1]
        
        # Console output (compact)
        if verbose:
            print_complete_statements(predicted, actual, f"Period {t+2}", prior=prior, compact=True)
        
        # Save PDF (full statements)
        if save_pdf and output_folder:
            pdf_path = os.path.join(output_folder, f"round_{round_num+1:02d}_period_{t+2}.pdf")
            try:
                generate_statement_pdf(predicted, actual, f"Period {t+2}", pdf_path, ticker)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not generate PDF: {e}")
        
        # Calculate errors
        errors = {}
        key_items = [
            ('revenue', ['revenue', 'sales_revenue', 'totalRevenue']),
            ('net_income', ['net_income', 'netIncome']),
            ('total_assets', ['total_assets', 'totalAssets']),
            ('total_equity', ['total_equity', 'totalEquity']),
            ('cash', ['cash', 'cash_and_cash_equivalents']),
        ]
        
        for key, possible_names in key_items:
            pred_val = getattr(predicted, key, 0)
            actual_val = None
            for name in possible_names:
                if name in actual.index:
                    val = actual[name]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        actual_val = val
                        break
            if actual_val and actual_val != 0:
                errors[key] = abs(pred_val - actual_val) / abs(actual_val) * 100
        
        results.append({
            'round': round_num + 1,
            'period': t + 2,
            'errors': errors,
            'identity_holds': predicted.all_identities_hold,
        })
    
    # Print summary
    if verbose:
        print_validation_summary(results, output_folder)
    
    return results


def print_validation_summary(results: List[Dict], output_folder: Optional[str] = None):
    """Print validation summary."""
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    all_errors = {}
    for r in results:
        for key, val in r.get('errors', {}).items():
            if key not in all_errors:
                all_errors[key] = []
            all_errors[key].append(val)
    
    print(f"\n{'Variable':<20} {'Mean MAPE':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-"*60)
    
    overall = []
    for var, errors in all_errors.items():
        if errors:
            mean_err = np.mean(errors)
            overall.append(mean_err)
            print(f"{var:<20} {mean_err:>8.2f}% {np.std(errors):>8.2f}% "
                  f"{np.min(errors):>8.2f}% {np.max(errors):>8.2f}%")
    
    print("-"*60)
    if overall:
        avg = np.mean(overall)
        print(f"{'OVERALL':<20} {avg:>8.2f}%")
    
    identity_pass = sum(1 for r in results if r.get('identity_holds', False))
    print(f"\nAccounting Identity: {identity_pass}/{len(results)} passed")
    
    if overall:
        avg = np.mean(overall)
        if avg < 15:
            print("✓ GOOD: Reasonable forecasts")
        elif avg < 25:
            print("○ FAIR: Moderate error")
        else:
            print("✗ NEEDS IMPROVEMENT: High error")
    
    if output_folder:
        print(f"\n📁 Full statements saved to: {output_folder}")
