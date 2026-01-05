"""
multi_year_simulator.py

Multi-Year Balance Sheet Simulation.
Forecast complete balance sheets multiple years into the future.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .quantile_simulator import QuantileSimulator
from .statement_printer import print_complete_statements, fmt_currency
from .pdf_report import generate_statement_pdf, setup_output_folder, HAS_REPORTLAB


def simulate_multi_year(data: pd.DataFrame,
                        n_years: int = 1,
                        n_scenarios: int = 100,
                        seq_length: int = 4,
                        verbose: bool = True,
                        ticker: str = "SAMPLE",
                        save_pdf: bool = True) -> Dict:
    """
    Simulate multiple years of complete financial statements.
    
    Args:
        data: Historical financial data
        n_years: Number of years to forecast
        n_scenarios: Number of scenario paths
        seq_length: Sequence length for model
        verbose: Print output
        ticker: Stock ticker for PDF naming
        save_pdf: Whether to save PDFs
    """
    
    if verbose:
        print("\n" + "="*70)
        print(f"MULTI-YEAR SIMULATION")
        print("="*70)
        print(f"Horizon: {n_years} years ({n_years * 4} quarters)")
        print(f"Scenarios: {n_scenarios}")
    
    # Setup output folder for PDFs
    output_folder = None
    if save_pdf and HAS_REPORTLAB:
        output_folder = setup_output_folder(ticker)
        if verbose:
            print(f"PDF output: {output_folder}")
    
    simulator = QuantileSimulator(seq_length=seq_length)
    simulator.fit(data, verbose=verbose)
    
    n_periods = n_years * 4
    all_scenarios = []
    
    if verbose:
        print(f"\nGenerating {n_scenarios} scenarios...")
    
    for s in range(n_scenarios):
        if verbose and (s + 1) % 20 == 0:
            print(f"  Scenario {s+1}/{n_scenarios}")
        
        scenario_path = []
        current_drivers = simulator.last_drivers.copy()
        prior = simulator.create_prior_statements(simulator.last_data)
        
        for t in range(n_periods):
            forecasts = simulator.predict_distribution(current_drivers)
            
            driver_values = {}
            sampled_drivers = []
            for driver in simulator.available_drivers:
                sampled = forecasts[driver].sample(1)[0]
                driver_values[driver] = sampled
                sampled_drivers.append(sampled)
            
            driver_values.setdefault('revenue_growth', 0.02)
            driver_values.setdefault('cogs_margin', 0.60)
            driver_values.setdefault('opex_margin', 0.15)
            driver_values.setdefault('capex_ratio', 0.03)
            driver_values.setdefault('net_margin', 0.15)
            
            statements = simulator.accounting_engine.derive_statements(
                drivers=driver_values, prior=prior, period=f"Q+{t+1}"
            )
            scenario_path.append(statements)
            prior = statements
            current_drivers = np.vstack([current_drivers[1:], sampled_drivers])
        
        all_scenarios.append(scenario_path)
    
    # Statistics
    key_vars = ['revenue', 'net_income', 'total_assets', 'total_equity', 'cash']
    stats = {var: {'mean': [], 'std': [], 'p5': [], 'p95': []} for var in key_vars}
    
    for t in range(n_periods):
        for var in key_vars:
            values = [getattr(all_scenarios[s][t], var, 0) for s in range(n_scenarios)]
            stats[var]['mean'].append(np.mean(values))
            stats[var]['std'].append(np.std(values))
            stats[var]['p5'].append(np.percentile(values, 5))
            stats[var]['p95'].append(np.percentile(values, 95))
    
    if verbose:
        print_multi_year_summary(stats, n_years, key_vars)
        
        # Find median scenario
        final_assets = [getattr(all_scenarios[s][-1], 'total_assets', 0) for s in range(n_scenarios)]
        median_idx = np.argsort(final_assets)[n_scenarios // 2]
        median_scenario = all_scenarios[median_idx]
        
        # Print and save each quarter of median scenario
        print(f"\n{'='*70}")
        print("MEDIAN SCENARIO - QUARTERLY FORECASTS")
        print(f"{'='*70}")
        
        prior_stmt = simulator.create_prior_statements(simulator.last_data)
        
        for t in range(n_periods):
            stmt = median_scenario[t]
            
            # Console output with accounting verification
            print(f"\n{'─'*70}")
            print(f"  Q+{t+1} FORECAST")
            print(f"{'─'*70}")
            
            # Key metrics
            print(f"  {'Revenue:':<20} {fmt_currency(stmt.revenue)}")
            print(f"  {'Net Income:':<20} {fmt_currency(stmt.net_income)}")
            print(f"  {'Total Assets:':<20} {fmt_currency(stmt.total_assets)}")
            print(f"  {'Total Equity:':<20} {fmt_currency(stmt.total_equity)}")
            
            # Accounting identity verification
            print(f"\n  ACCOUNTING IDENTITIES:")
            verify_accounting_identities(stmt, prior_stmt, verbose=True)
            
            # Save PDF (use same function as rolling validation for full details)
            if output_folder:
                pdf_path = os.path.join(output_folder, f"forecast_Q{t+1}.pdf")
                try:
                    generate_statement_pdf(stmt, None, f"Q+{t+1} Forecast", pdf_path, ticker)
                    print(f"  📄 Saved: forecast_Q{t+1}.pdf")
                except Exception as e:
                    print(f"  Warning: Could not generate PDF: {e}")
            
            prior_stmt = stmt
        
        if output_folder:
            print(f"\n📁 All forecasts saved to: {output_folder}")
    
    return {'all_scenarios': all_scenarios, 'stats': stats, 'n_scenarios': n_scenarios, 'n_years': n_years}


def verify_accounting_identities(stmt, prior, verbose: bool = False) -> Dict[str, bool]:
    """Verify all accounting identities hold."""
    results = {}
    
    # 1. Balance Sheet Identity: A = L + E
    a_eq_l_plus_e = abs(stmt.total_assets - stmt.total_liabilities - stmt.total_equity) < 1
    results['A = L + E'] = a_eq_l_plus_e
    if verbose:
        diff = stmt.total_assets - stmt.total_liabilities - stmt.total_equity
        status = "✓" if a_eq_l_plus_e else "✗"
        print(f"    {status} A = L + E: {fmt_currency(stmt.total_assets)} = {fmt_currency(stmt.total_liabilities)} + {fmt_currency(stmt.total_equity)} (diff: {fmt_currency(diff)})")
    
    # 2. Income Statement: GP = Rev - COGS
    gp_check = abs(stmt.gross_profit - (stmt.revenue - stmt.cogs)) < 1
    results['GP = Rev - COGS'] = gp_check
    if verbose:
        status = "✓" if gp_check else "✗"
        print(f"    {status} GP = Rev - COGS: {fmt_currency(stmt.gross_profit)} = {fmt_currency(stmt.revenue)} - {fmt_currency(stmt.cogs)}")
    
    # 3. EBITDA = GP - OpEx
    ebitda_check = abs(stmt.ebitda - (stmt.gross_profit - stmt.opex)) < 1
    results['EBITDA = GP - OpEx'] = ebitda_check
    if verbose:
        status = "✓" if ebitda_check else "✗"
        print(f"    {status} EBITDA = GP - OpEx: {fmt_currency(stmt.ebitda)} = {fmt_currency(stmt.gross_profit)} - {fmt_currency(stmt.opex)}")
    
    # 4. Net Income = EBT - Tax
    ni_check = abs(stmt.net_income - (stmt.ebt - stmt.income_tax)) < 1
    results['NI = EBT - Tax'] = ni_check
    if verbose:
        status = "✓" if ni_check else "✗"
        print(f"    {status} NI = EBT - Tax: {fmt_currency(stmt.net_income)} = {fmt_currency(stmt.ebt)} - {fmt_currency(stmt.income_tax)}")
    
    # 5. Cash Flow: End Cash = Begin + CFO + CFI + CFF
    if prior is not None:
        begin_cash = prior.cash
        cfo = stmt.cf_operating
        cfi = stmt.cf_investing
        cff = stmt.cf_financing
        expected_cash = begin_cash + cfo + cfi + cff
        cash_check = abs(stmt.cash - expected_cash) < 1e6  # Allow $1M tolerance
        results['Cash Flow'] = cash_check
        if verbose:
            status = "✓" if cash_check else "✗"
            print(f"    {status} Cash: {fmt_currency(begin_cash)} + {fmt_currency(cfo)} + {fmt_currency(cfi)} + {fmt_currency(cff)} = {fmt_currency(expected_cash)} (actual: {fmt_currency(stmt.cash)})")
    
    # 6. RE = Prior RE + NI - Div - Buybacks
    if prior is not None:
        expected_re = prior.retained_earnings + stmt.net_income - stmt.cf_dividends - stmt.cf_buybacks
        re_check = abs(stmt.retained_earnings - expected_re) < 1e6
        results['RE Rollforward'] = re_check
        if verbose:
            status = "✓" if re_check else "✗"
            print(f"    {status} RE: {fmt_currency(prior.retained_earnings)} + {fmt_currency(stmt.net_income)} - {fmt_currency(stmt.cf_dividends)} - {fmt_currency(stmt.cf_buybacks)} = {fmt_currency(expected_re)} (actual: {fmt_currency(stmt.retained_earnings)})")
    
    return results


def print_multi_year_summary(stats: Dict, n_years: int, key_vars: List[str]):
    """Print multi-year forecast summary."""
    
    print(f"\n{'='*70}")
    print("FORECAST SUMMARY BY YEAR END")
    print(f"{'='*70}")
    
    for year in range(n_years):
        q = (year + 1) * 4 - 1
        print(f"\n{'─'*70}")
        print(f"YEAR {year + 1} END (Q+{q+1})")
        print(f"{'─'*70}")
        print(f"{'Variable':<15} {'Mean':>15} {'Std':>12} {'90% CI':>25}")
        print(f"{'─'*70}")
        
        for var in key_vars:
            mean_val = stats[var]['mean'][q]
            std_val = stats[var]['std'][q]
            p5 = stats[var]['p5'][q]
            p95 = stats[var]['p95'][q]
            ci = f"[{fmt_currency(p5)}, {fmt_currency(p95)}]"
            print(f"{var:<15} {fmt_currency(mean_val):>15} {fmt_currency(std_val):>12} {ci:>25}")


def create_sample_data(n_quarters: int = 60) -> pd.DataFrame:
    """Create sample financial data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(end='2024-09-30', periods=n_quarters, freq='Q')
    
    base_revenue = 50e9
    revenue = [base_revenue]
    for i in range(1, n_quarters):
        growth = np.random.normal(0.02, 0.03)
        revenue.append(revenue[-1] * (1 + growth))
    revenue = np.array(revenue)
    
    cogs = revenue * np.clip(np.random.normal(0.60, 0.02, n_quarters), 0.4, 0.8)
    opex = revenue * np.clip(np.random.normal(0.15, 0.02, n_quarters), 0.05, 0.30)
    capex = revenue * np.clip(np.random.normal(0.03, 0.01, n_quarters), 0.01, 0.10)
    
    depreciation = revenue * 0.03
    net_income = (revenue - cogs - opex - depreciation) * 0.79
    
    total_assets = revenue * np.random.normal(3.5, 0.2, n_quarters)
    total_equity = total_assets * np.clip(np.random.normal(0.4, 0.05, n_quarters), 0.2, 0.6)
    total_liabilities = total_assets - total_equity
    
    cash = revenue * np.clip(np.random.normal(0.15, 0.03, n_quarters), 0.05, 0.30)
    accounts_receivable = revenue * (45/365)
    inventory = cogs * (60/365)
    accounts_payable = cogs * (45/365)
    
    ppe_net = total_assets * 0.3
    long_term_debt = total_liabilities * 0.5
    retained_earnings = total_equity * 0.6
    common_stock = total_equity * 0.4
    
    return pd.DataFrame({
        'revenue': revenue, 'cogs': cogs, 'opex': opex, 'capex': capex,
        'depreciation': depreciation, 'net_income': net_income,
        'total_assets': total_assets, 'total_equity': total_equity, 'total_liabilities': total_liabilities,
        'cash': cash, 'accounts_receivable': accounts_receivable, 'inventory': inventory,
        'accounts_payable': accounts_payable, 'ppe_net': ppe_net, 'long_term_debt': long_term_debt,
        'retained_earnings': retained_earnings, 'common_stock': common_stock,
    }, index=dates)
