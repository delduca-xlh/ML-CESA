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
    export ANTHROPIC_API_KEY="sk-ant-..."
    python run_ensemble.py AAPL
"""

import sys
import json
import re
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src' / 'financial_planning'))

# Get API key from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Import our balance sheet simulator
from balance_sheet_simulator import (
    QuantileSimulator,
    create_sample_data,
)
from balance_sheet_simulator.pdf_report import (
    generate_statement_pdf, 
    setup_output_folder, 
    HAS_REPORTLAB,
    fmt_currency,
    calc_error,
    get_actual_value,
)
from balance_sheet_simulator.data_structures import CompleteFinancialStatements

# Make reportlab optional
if HAS_REPORTLAB:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.units import inch


def load_real_data(ticker: str, n_quarters: int = 60):
    """Load real financial data."""
    try:
        from utils.fmp_data_fetcher import FMPDataFetcher
        fetcher = FMPDataFetcher()
        
        print(f"  Fetching data for {ticker}...")
        income_stmt = fetcher.fetch_income_statement(ticker, period='quarter', limit=n_quarters)
        balance_sheet = fetcher.fetch_balance_sheet(ticker, period='quarter', limit=n_quarters)
        
        try:
            cash_flow = fetcher.fetch_cash_flow(ticker, period='quarter', limit=n_quarters)
        except:
            cash_flow = None
        
        if income_stmt.empty:
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
        
        print(f"  ✓ Loaded {len(data)} quarters of data")
        return data
        
    except Exception as e:
        print(f"  ⚠ Error loading data: {e}")
        return None


def generate_llm_ratios(ticker: str, historical_data: pd.DataFrame) -> dict:
    """Generate LLM-based ratios for margins."""
    if not ANTHROPIC_API_KEY:
        return None
    
    recent = historical_data.tail(8) if len(historical_data) >= 8 else historical_data
    
    # Find revenue column
    rev_col = None
    for col in ['revenue', 'totalRevenue', 'sales_revenue']:
        if col in recent.columns:
            rev_col = col
            break
    
    if rev_col is None:
        return None
    
    avg_revenue = recent[rev_col].mean()
    
    # Calculate historical ratios
    cogs_col = next((c for c in ['cogs', 'costOfRevenue'] if c in recent.columns), None)
    ni_col = next((c for c in ['net_income', 'netIncome'] if c in recent.columns), None)
    opex_col = next((c for c in ['opex', 'operatingExpenses'] if c in recent.columns), None)
    
    avg_cogs = abs(recent[cogs_col].mean()) if cogs_col else avg_revenue * 0.55
    avg_ni = recent[ni_col].mean() if ni_col else avg_revenue * 0.25
    avg_opex = abs(recent[opex_col].mean()) if opex_col else avg_revenue * 0.15
    
    gross_margin = (avg_revenue - avg_cogs) / avg_revenue if avg_revenue > 0 else 0.45
    ni_margin = avg_ni / avg_revenue if avg_revenue > 0 else 0.25
    opex_margin = avg_opex / avg_revenue if avg_revenue > 0 else 0.15
    
    prompt = f"""You are a financial analyst predicting ratios for {ticker}.

Historical (Last 8Q):
- Gross Margin: {gross_margin:.1%}
- Net Income Margin: {ni_margin:.1%}
- OpEx Margin: {opex_margin:.1%}

Predict ratios for next quarters. Consider trends and your knowledge of {ticker}.

Respond ONLY with JSON (values as DECIMALS like 0.45, not percentages like 45):
{{
    "gross_margin": 0.XX,
    "net_income_margin": 0.XX,
    "opex_margin": 0.XX,
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
            
            # Validate - convert from percentage if needed
            def validate(val, default, name):
                if val is None:
                    return default
                if abs(val) > 1.0:  # Percentage form
                    val = val / 100.0
                if name == 'gross_margin' and (val < 0.1 or val > 0.9):
                    return default
                if name == 'net_income_margin' and (val < -0.5 or val > 0.5):
                    return default
                if name == 'opex_margin' and (val < 0.01 or val > 0.5):
                    return default
                return val
            
            return {
                'gross_margin': validate(result.get('gross_margin'), gross_margin, 'gross_margin'),
                'cogs_margin': 1 - validate(result.get('gross_margin'), gross_margin, 'gross_margin'),
                'net_income_margin': validate(result.get('net_income_margin'), ni_margin, 'net_income_margin'),
                'opex_margin': validate(result.get('opex_margin'), opex_margin, 'opex_margin'),
                'reasoning': result.get('reasoning', ''),
                'source': 'LLM'
            }
    except Exception as e:
        print(f"  LLM ratio error: {e}")
    
    return {
        'gross_margin': gross_margin,
        'cogs_margin': 1 - gross_margin,
        'net_income_margin': ni_margin,
        'opex_margin': opex_margin,
        'reasoning': 'Fallback to historical',
        'source': 'Fallback'
    }


def generate_pure_llm_predictions(ticker: str, historical_data: pd.DataFrame, periods: int) -> dict:
    """LLM directly predicts all values."""
    if not ANTHROPIC_API_KEY:
        return None
    
    recent = historical_data.tail(8) if len(historical_data) >= 8 else historical_data
    
    # Find columns
    rev_col = next((c for c in ['revenue', 'totalRevenue'] if c in recent.columns), None)
    cogs_col = next((c for c in ['cogs', 'costOfRevenue'] if c in recent.columns), None)
    ni_col = next((c for c in ['net_income', 'netIncome'] if c in recent.columns), None)
    assets_col = next((c for c in ['total_assets', 'totalAssets'] if c in recent.columns), None)
    equity_col = next((c for c in ['total_equity', 'totalEquity', 'totalStockholdersEquity'] if c in recent.columns), None)
    
    avg_revenue = recent[rev_col].mean() if rev_col else 100e9
    avg_cogs = abs(recent[cogs_col].mean()) if cogs_col else 50e9
    avg_ni = recent[ni_col].mean() if ni_col else 25e9
    avg_assets = recent[assets_col].mean() if assets_col else 350e9
    avg_equity = recent[equity_col].mean() if equity_col else 70e9
    
    prompt = f"""Forecast {periods} quarters for {ticker}.

Historical Averages (8Q):
- Revenue: ${avg_revenue/1e9:.2f}B
- COGS: ${avg_cogs/1e9:.2f}B  
- Net Income: ${avg_ni/1e9:.2f}B
- Total Assets: ${avg_assets/1e9:.2f}B
- Equity: ${avg_equity/1e9:.2f}B

Respond ONLY with JSON (values in BILLIONS):
{{
    "predictions": [
        {{"q":1, "revenue":XX.X, "cogs":XX.X, "net_income":XX.X, "total_assets":XXX.X, "total_equity":XX.X}},
        {{"q":2, "revenue":XX.X, "cogs":XX.X, "net_income":XX.X, "total_assets":XXX.X, "total_equity":XX.X}},
        {{"q":3, "revenue":XX.X, "cogs":XX.X, "net_income":XX.X, "total_assets":XXX.X, "total_equity":XX.X}},
        {{"q":4, "revenue":XX.X, "cogs":XX.X, "net_income":XX.X, "total_assets":XXX.X, "total_equity":XX.X}}
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
                    'revenue': np.array([p['revenue'] * 1e9 for p in preds]),
                    'cogs': np.array([p['cogs'] * 1e9 for p in preds]),
                    'net_income': np.array([p['net_income'] * 1e9 for p in preds]),
                    'total_assets': np.array([p['total_assets'] * 1e9 for p in preds]),
                    'total_equity': np.array([p['total_equity'] * 1e9 for p in preds]),
                    'reasoning': result.get('reasoning', ''),
                    'source': 'Pure LLM'
                }
    except Exception as e:
        print(f"  Pure LLM error: {e}")
    
    return None


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate MAPE."""
    mask = (actual != 0) & ~np.isnan(actual) & ~np.isnan(predicted)
    if mask.sum() == 0:
        return float('inf')
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def generate_full_statement_pdf(
    predicted: CompleteFinancialStatements,
    actual: pd.Series,
    period: str,
    output_path: str,
    ticker: str,
    approach_name: str,
    approach_comparison: dict = None
):
    """Generate PDF with full balance sheet - same format as Q1."""
    if not HAS_REPORTLAB:
        return None
    
    doc = SimpleDocTemplate(output_path, pagesize=letter, 
                           topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=14, spaceAfter=5)
    story.append(Paragraph(f"{ticker} Financial Statements - {period}", title_style))
    
    # Approach info - ALWAYS show which method was used
    approach_style = ParagraphStyle('Approach', parent=styles['Normal'], fontSize=10,
                                    textColor=colors.Color(0.2, 0.4, 0.6), spaceAfter=10)
    story.append(Paragraph(f"<b>Generated by:</b> {approach_name}", approach_style))
    
    story.append(Spacer(1, 10))
    
    has_actual = actual is not None
    
    # Helper to create section
    def add_section(title: str, items: list):
        header_style = ParagraphStyle('Header', parent=styles['Heading2'], fontSize=11, 
                                      textColor=colors.darkblue, spaceAfter=8)
        story.append(Paragraph(title, header_style))
        
        if has_actual:
            header = ['Item', 'Type', 'Predicted', 'Actual', 'Error']
            col_widths = [2.2*inch, 0.9*inch, 1.1*inch, 1.1*inch, 0.7*inch]
        else:
            header = ['Item', 'Type', 'Predicted']
            col_widths = [2.5*inch, 1.2*inch, 1.3*inch]
        
        data = [header]
        for item in items:
            name, item_type, pred_val, mappings = item
            row = [name, item_type, fmt_currency(pred_val)]
            if has_actual:
                actual_val = get_actual_value(actual, mappings)
                row.append(fmt_currency(actual_val) if actual_val else "N/A")
                row.append(calc_error(pred_val, actual_val) if actual_val else "N/A")
            data.append(row)
        
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (2, 1), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
        ]))
        story.append(table)
        story.append(Spacer(1, 15))
    
    # Get other_income if exists
    other_income_val = getattr(predicted, 'other_income', 0) or 0
    
    # Income Statement - FULL
    is_items = [
        ('Revenue', 'ML Driver', predicted.revenue, ['revenue', 'totalRevenue']),
        ('Cost of Goods Sold', 'ML Driver', predicted.cogs, ['cogs', 'costOfRevenue']),
        ('Gross Profit', 'Derived', predicted.gross_profit, ['gross_profit', 'grossProfit']),
        ('Operating Expenses', 'ML Driver', predicted.opex, ['opex', 'operatingExpenses']),
        ('EBITDA', 'Derived', predicted.ebitda, ['ebitda']),
        ('Depreciation', 'Derived', predicted.depreciation, ['depreciation', 'depreciationAndAmortization']),
        ('EBIT', 'Derived', predicted.ebit, ['ebit', 'operatingIncome']),
        ('Interest Expense', 'Derived', predicted.interest_expense, ['interestExpense']),
    ]
    if abs(other_income_val) > 1e6:
        is_items.append(('Other Income/Expense', 'Adjustment', other_income_val, ['otherIncome']))
    is_items.extend([
        ('EBT', 'Derived', predicted.ebt, ['ebt', 'incomeBeforeTax']),
        ('Income Tax', 'Derived', predicted.income_tax, ['incomeTaxExpense']),
        ('Net Income', 'ML Driver', predicted.net_income, ['net_income', 'netIncome']),
    ])
    add_section("INCOME STATEMENT", is_items)
    
    # Balance Sheet - Assets - FULL
    bs_assets = [
        ('Cash & Equivalents', 'CF Linkage', predicted.cash, ['cash', 'cashAndCashEquivalents']),
        ('Short-term Investments', 'Derived', predicted.short_term_investments, ['shortTermInvestments']),
        ('Accounts Receivable', 'Derived', predicted.accounts_receivable, ['netReceivables']),
        ('Inventory', 'Derived', predicted.inventory, ['inventory']),
        ('Prepaid Expenses', 'Derived', getattr(predicted, 'prepaid_expenses', 0), ['prepaidExpenses']),
        ('Other Current Assets', 'Derived', predicted.other_current_assets, ['otherCurrentAssets']),
        ('TOTAL CURRENT ASSETS', 'Sum', predicted.total_current_assets, ['totalCurrentAssets']),
        ('PP&E (Net)', 'CF Linkage', predicted.ppe_net, ['propertyPlantEquipmentNet']),
        ('Goodwill', 'Constant', predicted.goodwill, ['goodwill']),
        ('Intangible Assets', 'Derived', getattr(predicted, 'intangible_assets', 0), ['intangibleAssets']),
        ('Long-term Investments', 'Derived', predicted.long_term_investments, ['longTermInvestments']),
        ('Other Non-current Assets', 'Derived', predicted.other_noncurrent_assets, ['otherNonCurrentAssets']),
        ('TOTAL NON-CURRENT ASSETS', 'Sum', predicted.total_noncurrent_assets, ['totalNonCurrentAssets']),
        ('TOTAL ASSETS', 'Sum', predicted.total_assets, ['totalAssets']),
    ]
    add_section("BALANCE SHEET - ASSETS", bs_assets)
    
    # Balance Sheet - Liabilities & Equity - FULL
    bs_liab = [
        ('Accounts Payable', 'Derived', predicted.accounts_payable, ['accountPayables']),
        ('Deferred Revenue', 'Derived', predicted.deferred_revenue, ['deferredRevenue']),
        ('Other Current Liabilities', 'Derived', predicted.other_current_liabilities, ['otherCurrentLiabilities']),
        ('TOTAL CURRENT LIAB', 'Sum', predicted.total_current_liabilities, ['totalCurrentLiabilities']),
        ('Long-term Debt', 'Constant', predicted.long_term_debt, ['longTermDebt']),
        ('Other Non-current Liab', 'Derived', predicted.other_noncurrent_liabilities, ['otherNonCurrentLiabilities']),
        ('TOTAL NON-CURRENT LIAB', 'Sum', predicted.total_noncurrent_liabilities, ['totalNonCurrentLiabilities']),
        ('TOTAL LIABILITIES', 'A - E', predicted.total_liabilities, ['totalLiabilities']),
        ('Common Stock', 'Constant', predicted.common_stock, ['commonStock']),
        ('Additional Paid-in Capital', 'Constant', getattr(predicted, 'additional_paid_in_capital', 0), ['additionalPaidInCapital']),
        ('Retained Earnings', 'IS Linkage', predicted.retained_earnings, ['retainedEarnings']),
        ('Treasury Stock', 'Derived', getattr(predicted, 'treasury_stock', 0), ['treasuryStock']),
        ('Accumulated OCI', 'Constant', predicted.aoci, ['accumulatedOtherComprehensiveIncomeLoss']),
        ('TOTAL EQUITY', 'Sum', predicted.total_equity, ['totalStockholdersEquity', 'totalEquity']),
    ]
    add_section("BALANCE SHEET - LIABILITIES & EQUITY", bs_liab)
    
    # Accounting Identity Verification
    header_style = ParagraphStyle('Header', parent=styles['Heading2'], fontSize=11,
                                  textColor=colors.darkblue, spaceAfter=8)
    story.append(Paragraph("ACCOUNTING IDENTITY VERIFICATION", header_style))
    
    checks = []
    
    # All 10 checks (same as Q1)
    a_eq = abs(predicted.total_assets - predicted.total_liabilities - predicted.total_equity) < 1
    checks.append(['A = L + E', f"{fmt_currency(predicted.total_assets)} = {fmt_currency(predicted.total_liabilities)} + {fmt_currency(predicted.total_equity)}", '✓' if a_eq else '✗'])
    
    ca_sum = predicted.cash + predicted.short_term_investments + predicted.accounts_receivable + predicted.inventory + getattr(predicted, 'prepaid_expenses', 0) + predicted.other_current_assets
    ca_eq = abs(predicted.total_current_assets - ca_sum) < 1
    checks.append(['Current Assets Sum', f"{fmt_currency(predicted.total_current_assets)} = sum of components", '✓' if ca_eq else '✗'])
    
    nca_sum = predicted.ppe_net + predicted.goodwill + getattr(predicted, 'intangible_assets', 0) + predicted.long_term_investments + predicted.other_noncurrent_assets
    nca_eq = abs(predicted.total_noncurrent_assets - nca_sum) < 1
    checks.append(['Non-Current Assets Sum', f"{fmt_currency(predicted.total_noncurrent_assets)} = sum of components", '✓' if nca_eq else '✗'])
    
    ta_eq = abs(predicted.total_assets - predicted.total_current_assets - predicted.total_noncurrent_assets) < 1
    checks.append(['Total Assets = CA + NCA', f"{fmt_currency(predicted.total_assets)} = {fmt_currency(predicted.total_current_assets)} + {fmt_currency(predicted.total_noncurrent_assets)}", '✓' if ta_eq else '✗'])
    
    tl_eq = abs(predicted.total_liabilities - predicted.total_current_liabilities - predicted.total_noncurrent_liabilities) < 1
    checks.append(['Total Liab = CL + NCL', f"{fmt_currency(predicted.total_liabilities)} = {fmt_currency(predicted.total_current_liabilities)} + {fmt_currency(predicted.total_noncurrent_liabilities)}", '✓' if tl_eq else '✗'])
    
    gp_eq = abs(predicted.gross_profit - (predicted.revenue - predicted.cogs)) < 1
    checks.append(['GP = Rev - COGS', f"{fmt_currency(predicted.gross_profit)} = {fmt_currency(predicted.revenue)} - {fmt_currency(predicted.cogs)}", '✓' if gp_eq else '✗'])
    
    ebitda_eq = abs(predicted.ebitda - (predicted.gross_profit - predicted.opex)) < 1
    checks.append(['EBITDA = GP - OpEx', f"{fmt_currency(predicted.ebitda)} = {fmt_currency(predicted.gross_profit)} - {fmt_currency(predicted.opex)}", '✓' if ebitda_eq else '✗'])
    
    ebit_eq = abs(predicted.ebit - (predicted.ebitda - predicted.depreciation)) < 1
    checks.append(['EBIT = EBITDA - D&A', f"{fmt_currency(predicted.ebit)} = {fmt_currency(predicted.ebitda)} - {fmt_currency(predicted.depreciation)}", '✓' if ebit_eq else '✗'])
    
    ni_eq = abs(predicted.net_income - (predicted.ebt - predicted.income_tax)) < 1
    checks.append(['NI = EBT - Tax', f"{fmt_currency(predicted.net_income)} = {fmt_currency(predicted.ebt)} - {fmt_currency(predicted.income_tax)}", '✓' if ni_eq else '✗'])
    
    equity_sum = predicted.common_stock + predicted.retained_earnings + predicted.aoci
    if hasattr(predicted, 'additional_paid_in_capital'):
        equity_sum += predicted.additional_paid_in_capital
    if hasattr(predicted, 'treasury_stock'):
        equity_sum += predicted.treasury_stock
    equity_eq = abs(predicted.total_equity - equity_sum) < 1e6
    checks.append(['Equity Components', f"{fmt_currency(predicted.total_equity)} = sum of equity items", '✓' if equity_eq else '✗'])
    
    check_data = [['Identity', 'Calculation', 'Status']] + checks
    check_table = Table(check_data, colWidths=[1.5*inch, 3.5*inch, 0.5*inch])
    check_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (-1, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
    ]))
    story.append(check_table)
    
    all_pass = all(c[2] == '✓' for c in checks)
    summary_style = ParagraphStyle('Summary', parent=styles['Normal'], fontSize=10,
                                   textColor=colors.green if all_pass else colors.red,
                                   spaceBefore=10)
    story.append(Paragraph(
        f"Overall: {sum(1 for c in checks if c[2] == '✓')}/{len(checks)} identities verified {'✓ ALL PASS' if all_pass else '✗ SOME FAILED'}",
        summary_style
    ))
    
    doc.build(story)
    return output_path


def run_ensemble_validation(data: pd.DataFrame, ticker: str, n_test_periods: int = 4):
    """Run rolling validation comparing approaches."""
    
    min_train = 20
    max_test = len(data) - min_train - 1
    n_test_periods = min(n_test_periods, max_test)
    
    if n_test_periods <= 0:
        print(f"  ✗ Not enough data. Have {len(data)}, need at least {min_train + 2}.")
        return None
    
    start_t = len(data) - n_test_periods - 1
    
    print(f"\n{'='*70}")
    print("ENSEMBLE ROLLING VALIDATION")
    print(f"{'='*70}")
    print(f"Data periods: {len(data)} | Min train: {min_train} | Test periods: {n_test_periods}")
    
    # Setup output folder
    output_folder = setup_output_folder(ticker)
    print(f"PDF output: {output_folder}")
    
    # Results storage
    approaches = ['ML Only', 'ML + LLM', 'Pure LLM']
    results = {name: {'errors': [], 'predictions': []} for name in approaches}
    
    # Run validation rounds
    for round_num, t in enumerate(range(start_t, len(data) - 1)):
        print(f"\n{'='*70}")
        print(f"ROUND {round_num+1}/{n_test_periods}: Period {t+2}")
        print(f"{'='*70}")
        
        train_data = data.iloc[:t+1]
        actual = data.iloc[t+1]
        
        # ===== APPROACH 1: ML Only =====
        print("\n[1] ML Only: Training...", end=" ", flush=True)
        
        simulator = QuantileSimulator(seq_length=4)
        simulator.fit(train_data, verbose=False)
        
        forecasts = simulator.predict_distribution(simulator.last_drivers)
        ml_drivers = {d: forecasts[d].q50 for d in simulator.available_drivers}
        ml_drivers.setdefault('revenue_growth', 0.02)
        ml_drivers.setdefault('cogs_margin', 0.55)
        ml_drivers.setdefault('opex_margin', 0.15)
        ml_drivers.setdefault('capex_ratio', 0.03)
        ml_drivers.setdefault('net_margin', 0.25)
        
        prior = simulator.create_prior_statements(train_data.iloc[-1])
        ml_pred = simulator.accounting_engine.derive_statements(
            drivers=ml_drivers, prior=prior, period=f"Period {t+2}"
        )
        
        print("Done.")
        print(f"    ML Drivers: cogs={ml_drivers['cogs_margin']:.4f}, opex={ml_drivers['opex_margin']:.4f}, net={ml_drivers['net_margin']:.4f}")
        
        ml_errors = calc_approach_errors(ml_pred, actual)
        results['ML Only']['errors'].append(ml_errors)
        results['ML Only']['predictions'].append(ml_pred)
        
        # ===== APPROACH 2: ML + LLM Ratios =====
        print("[2] ML + LLM: Getting ratios...", end=" ", flush=True)
        
        llm_ratios = generate_llm_ratios(ticker, train_data)
        
        if llm_ratios:
            print("Done.")
            print(f"    LLM Ratios: cogs={llm_ratios['cogs_margin']:.4f}, opex={llm_ratios['opex_margin']:.4f}, net={llm_ratios['net_income_margin']:.4f}")
            
            hybrid_drivers = {
                'revenue_growth': ml_drivers['revenue_growth'],
                'cogs_margin': llm_ratios['cogs_margin'],
                'opex_margin': llm_ratios['opex_margin'],
                'capex_ratio': ml_drivers['capex_ratio'],
                'net_margin': llm_ratios['net_income_margin'],
            }
            
            hybrid_pred = simulator.accounting_engine.derive_statements(
                drivers=hybrid_drivers, prior=prior, period=f"Period {t+2}"
            )
            
            hybrid_errors = calc_approach_errors(hybrid_pred, actual)
            results['ML + LLM']['errors'].append(hybrid_errors)
            results['ML + LLM']['predictions'].append(hybrid_pred)
        else:
            print("Skipped (no API key)")
            results['ML + LLM']['errors'].append(None)
            results['ML + LLM']['predictions'].append(None)
        
        # ===== APPROACH 3: Pure LLM =====
        print("[3] Pure LLM: Predicting...", end=" ", flush=True)
        
        pure_llm = generate_pure_llm_predictions(ticker, train_data, 1)
        
        if pure_llm:
            print("Done.")
            
            # Create statement from LLM predictions
            llm_pred = CompleteFinancialStatements(period=f"Period {t+2}")
            llm_pred.revenue = pure_llm['revenue'][0]
            llm_pred.cogs = pure_llm['cogs'][0]
            llm_pred.gross_profit = llm_pred.revenue - llm_pred.cogs
            llm_pred.net_income = pure_llm['net_income'][0]
            llm_pred.total_assets = pure_llm['total_assets'][0]
            llm_pred.total_equity = pure_llm['total_equity'][0]
            llm_pred.total_liabilities = llm_pred.total_assets - llm_pred.total_equity
            
            # Fill in other fields with reasonable values
            llm_pred.opex = llm_pred.revenue * 0.15
            llm_pred.ebitda = llm_pred.gross_profit - llm_pred.opex
            llm_pred.depreciation = llm_pred.revenue * 0.03
            llm_pred.ebit = llm_pred.ebitda - llm_pred.depreciation
            llm_pred.interest_expense = 0
            llm_pred.ebt = llm_pred.ebit
            llm_pred.income_tax = max(0, llm_pred.ebt * 0.21)
            llm_pred.cash = llm_pred.total_assets * 0.1
            llm_pred.accounts_receivable = llm_pred.revenue * 0.15
            llm_pred.inventory = llm_pred.cogs * 0.05
            llm_pred.short_term_investments = llm_pred.total_assets * 0.08
            llm_pred.other_current_assets = llm_pred.total_assets * 0.03
            llm_pred.total_current_assets = llm_pred.cash + llm_pred.short_term_investments + llm_pred.accounts_receivable + llm_pred.inventory + llm_pred.other_current_assets
            llm_pred.ppe_net = llm_pred.total_assets * 0.15
            llm_pred.goodwill = 0
            llm_pred.long_term_investments = llm_pred.total_assets * 0.25
            llm_pred.other_noncurrent_assets = llm_pred.total_assets - llm_pred.total_current_assets - llm_pred.ppe_net - llm_pred.long_term_investments
            llm_pred.total_noncurrent_assets = llm_pred.total_assets - llm_pred.total_current_assets
            llm_pred.accounts_payable = llm_pred.cogs * 0.20
            llm_pred.deferred_revenue = llm_pred.revenue * 0.03
            llm_pred.other_current_liabilities = llm_pred.total_liabilities * 0.15
            llm_pred.total_current_liabilities = llm_pred.accounts_payable + llm_pred.deferred_revenue + llm_pred.other_current_liabilities
            llm_pred.long_term_debt = llm_pred.total_liabilities * 0.30
            llm_pred.other_noncurrent_liabilities = llm_pred.total_liabilities - llm_pred.total_current_liabilities - llm_pred.long_term_debt
            llm_pred.total_noncurrent_liabilities = llm_pred.total_liabilities - llm_pred.total_current_liabilities
            llm_pred.common_stock = llm_pred.total_equity * 1.3
            llm_pred.retained_earnings = llm_pred.total_equity * -0.2
            llm_pred.aoci = llm_pred.total_equity * -0.1
            
            llm_errors = calc_approach_errors(llm_pred, actual)
            results['Pure LLM']['errors'].append(llm_errors)
            results['Pure LLM']['predictions'].append(llm_pred)
        else:
            print("Skipped (no API key)")
            results['Pure LLM']['errors'].append(None)
            results['Pure LLM']['predictions'].append(None)
        
        # Print round summary
        print(f"\n{'Approach':<15} {'Revenue':>12} {'Net Income':>12} {'Assets':>12} {'Equity':>12}")
        print("-"*65)
        for name in approaches:
            errs = results[name]['errors'][-1]
            if errs:
                print(f"{name:<15} {errs.get('revenue', 0):>10.2f}% {errs.get('net_income', 0):>10.2f}% "
                      f"{errs.get('total_assets', 0):>10.2f}% {errs.get('total_equity', 0):>10.2f}%")
        
        # Find best approach for this round
        best_approach = 'ML Only'
        best_mape = float('inf')
        for name in approaches:
            errs = results[name]['errors'][-1]
            if errs:
                avg = np.mean([v for v in errs.values()])
                if avg < best_mape:
                    best_mape = avg
                    best_approach = name
        
        # Generate PDF for this round using best approach
        best_pred = results[best_approach]['predictions'][-1]
        if best_pred:
            pdf_path = os.path.join(output_folder, f"ensemble_round_{round_num+1:02d}_period_{t+2}.pdf")
            generate_full_statement_pdf(
                best_pred, actual, f"Period {t+2}", pdf_path, ticker, best_approach
            )
            print(f"  → PDF: {pdf_path}")
    
    # Calculate summary
    summary = calculate_summary(results)
    
    # Print summary
    print_summary(summary, output_folder)
    
    # Generate summary PDF
    generate_summary_pdf(summary, ticker, output_folder, n_test_periods)
    
    # Generate final forecast using best approach (4 quarters)
    generate_final_forecast(data, ticker, summary, output_folder, n_quarters=4)
    
    return {
        'results': results,
        'summary': summary,
        'output_folder': output_folder
    }


def calc_approach_errors(pred, actual):
    """Calculate errors for key metrics."""
    errors = {}
    
    metrics = [
        ('revenue', ['revenue', 'totalRevenue']),
        ('net_income', ['net_income', 'netIncome']),
        ('total_assets', ['total_assets', 'totalAssets']),
        ('total_equity', ['total_equity', 'totalEquity', 'totalStockholdersEquity']),
        ('cash', ['cash', 'cashAndCashEquivalents']),
    ]
    
    for metric, cols in metrics:
        pred_val = getattr(pred, metric, 0)
        actual_val = None
        for col in cols:
            if col in actual.index:
                val = actual[col]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    actual_val = val
                    break
        if actual_val and actual_val != 0:
            errors[metric] = abs(pred_val - actual_val) / abs(actual_val) * 100
    
    return errors


def calculate_summary(results):
    """Calculate summary statistics."""
    summary = {}
    
    for approach, data in results.items():
        valid_errors = [e for e in data['errors'] if e is not None]
        if not valid_errors:
            continue
        
        all_errors = {}
        for err_dict in valid_errors:
            for key, val in err_dict.items():
                if key not in all_errors:
                    all_errors[key] = []
                all_errors[key].append(val)
        
        stats = {}
        overall = []
        for var, errors in all_errors.items():
            if errors:
                stats[var] = {
                    'mean': np.mean(errors),
                    'std': np.std(errors),
                    'min': np.min(errors),
                    'max': np.max(errors),
                }
                overall.append(np.mean(errors))
        
        summary[approach] = {
            'stats': stats,
            'overall_mape': np.mean(overall) if overall else float('inf'),
            'n_rounds': len(valid_errors),
        }
    
    if summary:
        best = min(summary, key=lambda x: summary[x]['overall_mape'])
        summary['_best'] = best
        summary['_best_mape'] = summary[best]['overall_mape']
    
    return summary


def print_summary(summary, output_folder):
    """Print validation summary."""
    print(f"\n{'='*80}")
    print("ENSEMBLE VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    for approach in ['ML Only', 'ML + LLM', 'Pure LLM']:
        if approach not in summary:
            continue
        data = summary[approach]
        
        print(f"\n{approach}:")
        print(f"  {'Variable':<15} {'Mean MAPE':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("  " + "-"*55)
        
        for var, stats in data['stats'].items():
            print(f"  {var:<15} {stats['mean']:>8.2f}% {stats['std']:>8.2f}% "
                  f"{stats['min']:>8.2f}% {stats['max']:>8.2f}%")
        
        print("  " + "-"*55)
        print(f"  {'OVERALL':<15} {data['overall_mape']:>8.2f}%")
    
    best = summary.get('_best', 'N/A')
    best_mape = summary.get('_best_mape', 0)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              BEST APPROACH                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Winner: {best:<20}                                              ║
║   MAPE:   {best_mape:>6.2f}%                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    if best_mape < 10:
        print(f"  Grade: ⭐⭐⭐⭐⭐ Excellent")
    elif best_mape < 15:
        print(f"  Grade: ⭐⭐⭐⭐ Very Good")
    elif best_mape < 20:
        print(f"  Grade: ⭐⭐⭐ Good")
    else:
        print(f"  Grade: ⭐⭐ Fair")
    
    print(f"\n📁 Full statements saved to: {output_folder}")


def generate_summary_pdf(summary, ticker, output_folder, n_rounds):
    """Generate summary PDF."""
    if not HAS_REPORTLAB:
        return
    
    pdf_path = os.path.join(output_folder, f"{ticker}_ensemble_validation_summary.pdf")
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=16, spaceAfter=10)
    story.append(Paragraph(f"{ticker} - Ensemble Validation Summary", title_style))
    
    subtitle_style = ParagraphStyle('Subtitle', fontSize=10, textColor=colors.grey, spaceAfter=20)
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Rounds: {n_rounds}", subtitle_style))
    
    # Executive Summary
    header_style = ParagraphStyle('Header', parent=styles['Heading2'], fontSize=12, textColor=colors.darkblue, spaceAfter=10)
    story.append(Paragraph("EXECUTIVE SUMMARY", header_style))
    
    best = summary.get('_best', 'N/A')
    best_mape = summary.get('_best_mape', 0)
    
    exec_data = [
        ['Metric', 'Value'],
        ['Best Approach', best],
        ['Best MAPE', f"{best_mape:.2f}%"],
        ['Validation Rounds', str(n_rounds)],
    ]
    
    exec_table = Table(exec_data, colWidths=[2*inch, 3*inch])
    exec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.3, 0.5)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(exec_table)
    story.append(Spacer(1, 20))
    
    # Approach Comparison
    story.append(Paragraph("APPROACH COMPARISON", header_style))
    
    comp_header = ['Approach', 'Revenue', 'Net Income', 'Assets', 'Equity', 'Overall']
    comp_data = [comp_header]
    
    for approach in ['ML Only', 'ML + LLM', 'Pure LLM']:
        if approach not in summary:
            continue
        stats = summary[approach]['stats']
        comp_data.append([
            approach,
            f"{stats.get('revenue', {}).get('mean', 0):.2f}%",
            f"{stats.get('net_income', {}).get('mean', 0):.2f}%",
            f"{stats.get('total_assets', {}).get('mean', 0):.2f}%",
            f"{stats.get('total_equity', {}).get('mean', 0):.2f}%",
            f"{summary[approach]['overall_mape']:.2f}%"
        ])
    
    comp_table = Table(comp_data, colWidths=[1.3*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.3, 0.5)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 20))
    
    # Methodology
    story.append(Paragraph("METHODOLOGY", header_style))
    
    method_style = ParagraphStyle('Method', fontSize=9, spaceAfter=6)
    methods = [
        "<b>ML Only:</b> XGBoost Quantile Regression + Accounting Engine",
        "<b>ML + LLM:</b> ML growth prediction + LLM margin adjustments",
        "<b>Pure LLM:</b> LLM directly predicts all financial values",
    ]
    for m in methods:
        story.append(Paragraph(m, method_style))
    
    doc.build(story)
    print(f"\n📊 Summary PDF: {pdf_path}")


def generate_final_forecast(data, ticker, summary, output_folder, n_quarters: int = 4):
    """Generate final forecast using best approach for multiple quarters."""
    best = summary.get('_best', 'ML Only')
    print(f"\n[4] Generating Final Forecast ({n_quarters} quarters) using {best}...")
    
    # Train on all data
    simulator = QuantileSimulator(seq_length=4)
    simulator.fit(data, verbose=False)
    
    # Get base drivers from ML
    forecasts = simulator.predict_distribution(simulator.last_drivers)
    base_drivers = {d: forecasts[d].q50 for d in simulator.available_drivers}
    base_drivers.setdefault('revenue_growth', 0.02)
    base_drivers.setdefault('cogs_margin', 0.55)
    base_drivers.setdefault('opex_margin', 0.15)
    base_drivers.setdefault('capex_ratio', 0.03)
    base_drivers.setdefault('net_margin', 0.25)
    
    # If best is ML + LLM, get LLM ratios
    if best == 'ML + LLM' and ANTHROPIC_API_KEY:
        llm_ratios = generate_llm_ratios(ticker, data)
        if llm_ratios:
            base_drivers['cogs_margin'] = llm_ratios['cogs_margin']
            base_drivers['opex_margin'] = llm_ratios['opex_margin']
            base_drivers['net_margin'] = llm_ratios['net_income_margin']
    
    # Generate forecasts for each quarter
    prior = simulator.create_prior_statements(data.iloc[-1])
    
    for q in range(1, n_quarters + 1):
        # Each quarter uses slightly different growth assumptions
        drivers = base_drivers.copy()
        
        # Derive statements
        forecast = simulator.accounting_engine.derive_statements(
            drivers=drivers, prior=prior, period=f"Q+{q} Forecast"
        )
        
        # Generate PDF
        pdf_path = os.path.join(output_folder, f"ensemble_forecast_Q{q}.pdf")
        generate_full_statement_pdf(forecast, None, f"Q+{q} Forecast", pdf_path, ticker, best)
        print(f"  ✓ Q+{q} Forecast: {pdf_path}")
        
        # Use this quarter's forecast as prior for next quarter
        prior = forecast


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_ensemble.py TICKER")
        print("Example: python run_ensemble.py AAPL")
        print("\nSet API key: export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    
    print("="*80)
    print(f"ENSEMBLE FORECAST: {ticker}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    if ANTHROPIC_API_KEY:
        print(f"API Key: {'*'*20}...{ANTHROPIC_API_KEY[-4:]}")
    else:
        print("API Key: Not set (ML Only mode)")
        print("  → Set ANTHROPIC_API_KEY for LLM features")
    
    # Load data
    print(f"\n[1] Loading Data...")
    data = load_real_data(ticker)
    
    if data is None:
        print("  → Using sample data...")
        data = create_sample_data(60)
        ticker = "SAMPLE"
    
    # Get forecast periods from command line
    n_periods = 4
    if len(sys.argv) > 2:
        try:
            n_periods = int(sys.argv[2])
        except:
            pass
    
    # Run ensemble
    run_ensemble_validation(data, ticker, n_periods)
    
    print("\n" + "="*80)
    print("ENSEMBLE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()