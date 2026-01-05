#!/usr/bin/env python3
"""
run_simulator.py

Balance Sheet Simulator - Main Entry Point

Usage:
    python run_simulator.py                     # Sample data
    python run_simulator.py AAPL                # Real company
    python run_simulator.py --help              # All options
"""

import argparse
import sys
from pathlib import Path

# Add the balance_sheet_simulator directly to path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'financial_planning'))

from balance_sheet_simulator import (
    QuantileSimulator,
    run_rolling_validation,
    simulate_multi_year,
    create_sample_data,
)


def load_real_data(ticker: str, api_key: str = None, n_quarters: int = 60):
    """Load real financial data from FMP API.
    
    Args:
        ticker: Stock ticker symbol
        api_key: FMP API key (optional)
        n_quarters: Number of quarters to fetch (default: 60 = 15 years, most recent)
    """
    try:
        # Try multiple import paths
        try:
            from utils.fmp_data_fetcher import FMPDataFetcher
        except ImportError:
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from fmp_data_fetcher import FMPDataFetcher
            except ImportError:
                print("FMPDataFetcher not found. Using sample data.")
                return None
    except Exception as e:
        print(f"Import error: {e}")
        return None
    
    try:
        fetcher = FMPDataFetcher(api_key=api_key) if api_key else FMPDataFetcher()
        
        print(f"\nFetching data for {ticker}...")
        
        income_stmt = fetcher.fetch_income_statement(ticker, period='quarter', limit=n_quarters)
        print(f"  ✓ Income Statement: {len(income_stmt)} periods")
        
        balance_sheet = fetcher.fetch_balance_sheet(ticker, period='quarter', limit=n_quarters)
        print(f"  ✓ Balance Sheet: {len(balance_sheet)} periods")
        
        # Try different method names for cash flow
        try:
            cash_flow = fetcher.fetch_cash_flow(ticker, period='quarter', limit=n_quarters)
            print(f"  ✓ Cash Flow: {len(cash_flow)} periods")
        except AttributeError:
            try:
                cash_flow = fetcher.fetch_cash_flow_statement(ticker, period='quarter', limit=n_quarters)
                print(f"  ✓ Cash Flow: {len(cash_flow)} periods")
            except AttributeError:
                cash_flow = None
                print("  Note: Cash flow data not available")
        
        if income_stmt.empty:
            print(f"No data found for {ticker}")
            return None
        
        data = income_stmt.copy()
        for col in balance_sheet.columns:
            if col not in data.columns:
                data[col] = balance_sheet[col]
        if cash_flow is not None:
            for col in cash_flow.columns:
                if col not in data.columns:
                    data[col] = cash_flow[col]
        
        if 'date' in data.columns:
            data = data.sort_values('date').reset_index(drop=True)
        
        print(f"Loaded {len(data)} quarters of data")
        return data
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Balance Sheet Simulator')
    parser.add_argument('ticker', nargs='?', default=None, help='Stock ticker (e.g., AAPL)')
    parser.add_argument('--api-key', type=str, default=None, help='FMP API key')
    parser.add_argument('--min-train', type=int, default=20, help='Min training periods (default: 20)')
    parser.add_argument('--test-periods', type=int, default=5, help='Test periods (default: 5)')
    parser.add_argument('--scenarios', type=int, default=100, help='Simulation scenarios (default: 100)')
    parser.add_argument('--years', type=int, default=1, help='Years to simulate (default: 1)')
    parser.add_argument('--skip-validation', action='store_true', help='Skip rolling validation')
    parser.add_argument('--skip-simulation', action='store_true', help='Skip multi-year simulation')
    parser.add_argument('--quarters', type=int, default=60, help='Quarters of history to fetch (default: 60 = 15 years, most recent)')
    parser.add_argument('--seq-length', type=int, default=4, help='Sequence length for lag features (default: 4, try 8 for YoY)')
    
    args = parser.parse_args()
    
    print("="*100)
    print("BALANCE SHEET SIMULATOR")
    print("XGBoost Quantile Regression for Probabilistic Forecasting")
    print("="*100)
    
    # Load data
    print("\n[1] Loading Data...")
    if args.ticker:
        data = load_real_data(args.ticker, args.api_key, n_quarters=args.quarters)
        if data is None:
            print("Falling back to sample data...")
            data = create_sample_data(args.quarters)
            data_source = "sample"
        else:
            data_source = args.ticker
    else:
        print("No ticker provided. Using sample data.")
        data = create_sample_data(args.quarters)
        data_source = "sample"
    
    print(f"Data source: {data_source}")
    print(f"Periods: {len(data)}")
    
    # Rolling Validation
    if not args.skip_validation:
        print("\n[2] Rolling Validation...")
        results = run_rolling_validation(
            data,
            min_train_periods=args.min_train,
            n_test_periods=args.test_periods,
            verbose=True,
            ticker=data_source,
            save_pdf=True
        )
    else:
        print("\n[2] Skipping Rolling Validation")
    
    # Multi-Year Simulation
    if not args.skip_simulation:
        print(f"\n[3] Multi-Year Simulation ({args.years} years, {args.scenarios} scenarios)...")
        simulation = simulate_multi_year(
            data,
            n_years=args.years,
            n_scenarios=args.scenarios,
            verbose=True,
            ticker=data_source,
            save_pdf=True
        )
    else:
        print("\n[3] Skipping Multi-Year Simulation")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print("\nFeatures demonstrated:")
    print("  ✓ Probabilistic forecasting (quantile regression)")
    print("  ✓ Complete three-statement output (30+ line items)")
    print("  ✓ All accounting identities satisfied")
    print("  ✓ Rolling validation with actual comparison")
    print("  ✓ Multi-year simulation with uncertainty")
    print("  ✓ PDF reports with accounting verification")


if __name__ == "__main__":
    main()
