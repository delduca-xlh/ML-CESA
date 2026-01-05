"""
statement_printer.py

Utility functions for printing complete financial statements.
"""

import numpy as np
import pandas as pd
from typing import Optional

from .data_structures import CompleteFinancialStatements


def fmt_currency(val: float) -> str:
    """Format currency value."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if abs(val) >= 1e9:
        return f"${val/1e9:,.2f}B"
    elif abs(val) >= 1e6:
        return f"${val/1e6:,.1f}M"
    elif abs(val) >= 1e3:
        return f"${val/1e3:,.1f}K"
    else:
        return f"${val:,.0f}"


def calc_error(pred: float, actual: float) -> str:
    """Calculate percentage error."""
    if actual is None or actual == 0 or (isinstance(actual, float) and np.isnan(actual)):
        return "N/A"
    err = abs(pred - actual) / abs(actual) * 100
    return f"{err:.1f}%"


def print_complete_statements(predicted: CompleteFinancialStatements,
                               actual: Optional[pd.Series] = None,
                               period: str = "",
                               prior: Optional[CompleteFinancialStatements] = None,
                               compact: bool = True):
    """Print complete three financial statements with comparison to actual.
    
    Args:
        compact: If True, only show key metrics and summary. If False, show full details.
    """
    
    def get_actual(key: str, mappings: list = None) -> Optional[float]:
        if actual is None:
            return None
        names = mappings if mappings else [key]
        for name in names:
            if name in actual.index:
                val = actual[name]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    return val
        return None
    
    has_actual = actual is not None
    has_prior = prior is not None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMPACT MODE: Only show key metrics
    # ═══════════════════════════════════════════════════════════════════════════
    if compact:
        print(f"\n{'─'*70}")
        print(f"  {period}")
        print(f"{'─'*70}")
        
        key_metrics = [
            ('Revenue', predicted.revenue, ['revenue', 'sales_revenue', 'totalRevenue']),
            ('Net Income', predicted.net_income, ['net_income', 'netIncome']),
            ('Total Assets', predicted.total_assets, ['total_assets', 'totalAssets']),
            ('Total Equity', predicted.total_equity, ['total_equity', 'totalEquity', 'totalStockholdersEquity']),
        ]
        
        if has_actual:
            print(f"  {'Metric':<16} {'Predicted':>12} {'Actual':>12} {'Error':>8}")
            print(f"  {'─'*50}")
            for name, pred_val, mappings in key_metrics:
                actual_val = get_actual(mappings[0], mappings)
                if actual_val is not None:
                    err = abs(pred_val - actual_val) / abs(actual_val) * 100
                    err_str = f"{err:.1f}%"
                    status = "✓" if err < 10 else "△" if err < 20 else "✗"
                else:
                    err_str = "N/A"
                    status = ""
                print(f"  {name:<16} {fmt_currency(pred_val):>12} {fmt_currency(actual_val) if actual_val else 'N/A':>12} {err_str:>6} {status}")
        else:
            print(f"  {'Metric':<16} {'Predicted':>12}")
            print(f"  {'─'*28}")
            for name, pred_val, mappings in key_metrics:
                print(f"  {name:<16} {fmt_currency(pred_val):>12}")
        return
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FULL MODE: Show complete statements (original behavior)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'━'*100}")
    print(f"{'COMPLETE FINANCIAL STATEMENTS - ' + period:^100}")
    print(f"{'━'*100}")
    
    if has_actual:
        header = f"{'Item':<40} {'Type':<12} {'Predicted':>16} {'Actual':>16} {'Error':>12}"
    else:
        header = f"{'Item':<40} {'Type':<12} {'Predicted':>16}"
    
    # Income Statement
    print(f"\n{'─'*100}")
    print(f"{'INCOME STATEMENT':^100}")
    print(f"{'─'*100}")
    print(header)
    print(f"{'─'*100}")
    
    # Get other_income if it exists
    other_income_val = getattr(predicted, 'other_income', 0) or 0
    
    is_items = [
        ('Revenue', 'ML Driver', predicted.revenue, ['revenue', 'sales_revenue', 'totalRevenue']),
        ('Cost of Goods Sold', 'ML Driver', predicted.cogs, ['cogs', 'cost_of_goods_sold', 'costOfRevenue']),
        ('Gross Profit', 'Derived', predicted.gross_profit, ['gross_profit', 'grossProfit']),
        ('Operating Expenses', 'ML Driver', predicted.opex, ['opex', 'overhead_expenses', 'operatingExpenses']),
        ('EBITDA', 'Derived', predicted.ebitda, ['ebitda']),
        ('Depreciation', 'Derived', predicted.depreciation, ['depreciation', 'depreciationAndAmortization']),
        ('EBIT', 'Derived', predicted.ebit, ['ebit', 'operatingIncome']),
        ('Interest Expense', 'Derived', predicted.interest_expense, ['interest_expense', 'interestExpense']),
    ]
    
    # Add Other Income row if significant
    if abs(other_income_val) > 1e6:  # Only show if > $1M
        is_items.append(('Other Income/Expense', 'Adjustment', other_income_val, ['other_income', 'otherIncome']))
    
    is_items.extend([
        ('EBT', 'Derived', predicted.ebt, ['ebt', 'incomeBeforeTax']),
        ('Income Tax', 'Derived', predicted.income_tax, ['income_tax', 'incomeTaxExpense']),
        ('Net Income', 'ML Driver', predicted.net_income, ['net_income', 'netIncome']),  # Now ML-driven via net_margin
    ])
    
    for name, typ, pred_val, keys in is_items:
        marker = "●" if "ML" in typ else "○"
        actual_val = get_actual(keys[0], keys)
        if has_actual and actual_val is not None:
            err = calc_error(pred_val, actual_val)
            print(f"{marker} {name:<38} {typ:<12} {fmt_currency(pred_val):>16} {fmt_currency(actual_val):>16} {err:>12}")
        else:
            print(f"{marker} {name:<38} {typ:<12} {fmt_currency(pred_val):>16}")
    
    # Balance Sheet - Assets
    print(f"\n{'─'*100}")
    print(f"{'BALANCE SHEET - ASSETS':^100}")
    print(f"{'─'*100}")
    print(header)
    print(f"{'─'*100}")
    
    print("CURRENT ASSETS")
    asset_items = [
        ('  Cash & Equivalents', 'CF Linkage', predicted.cash, ['cash', 'cashAndCashEquivalents']),
        ('  Short-term Investments', 'Derived', predicted.short_term_investments, ['shortTermInvestments', 'short_term_investments']),
        ('  Accounts Receivable', 'Derived', predicted.accounts_receivable, ['accounts_receivable', 'accountsReceivable', 'netReceivables']),
        ('  Inventory', 'Derived', predicted.inventory, ['inventory']),
        ('  Other Current Assets', 'Derived', predicted.other_current_assets, ['otherCurrentAssets', 'other_current_assets']),
        ('TOTAL CURRENT ASSETS', 'Sum', predicted.total_current_assets, ['total_current_assets', 'totalCurrentAssets']),
    ]
    
    for name, typ, pred_val, keys in asset_items:
        marker = "★" if "TOTAL" in name else "○"
        actual_val = get_actual(keys[0], keys)
        if has_actual and actual_val is not None:
            err = calc_error(pred_val, actual_val)
            print(f"{marker} {name:<38} {typ:<12} {fmt_currency(pred_val):>16} {fmt_currency(actual_val):>16} {err:>12}")
        else:
            print(f"{marker} {name:<38} {typ:<12} {fmt_currency(pred_val):>16}")
    
    print("\nNON-CURRENT ASSETS")
    nca_items = [
        ('  PP&E (Net)', 'CF Linkage', predicted.ppe_net, ['ppe_net', 'propertyPlantEquipmentNet']),
        ('  Goodwill', 'Constant', predicted.goodwill, ['goodwill']),
        ('  Intangible Assets', 'Derived', predicted.intangible_assets, ['intangibleAssets', 'goodwillAndIntangibleAssets']),
        ('  Long-term Investments', 'Derived', predicted.long_term_investments, ['longTermInvestments', 'long_term_investments']),
        ('  Other Non-current Assets', 'Derived', predicted.other_noncurrent_assets, ['otherNonCurrentAssets', 'other_noncurrent_assets']),
        ('TOTAL NON-CURRENT ASSETS', 'Sum', predicted.total_noncurrent_assets, ['total_noncurrent_assets', 'totalNonCurrentAssets']),
    ]
    
    for name, typ, pred_val, keys in nca_items:
        marker = "★" if "TOTAL" in name else "○"
        actual_val = get_actual(keys[0], keys)
        if has_actual and actual_val is not None:
            err = calc_error(pred_val, actual_val)
            print(f"{marker} {name:<38} {typ:<12} {fmt_currency(pred_val):>16} {fmt_currency(actual_val):>16} {err:>12}")
        else:
            print(f"{marker} {name:<38} {typ:<12} {fmt_currency(pred_val):>16}")
    
    print(f"\n{'─'*60}")
    actual_ta = get_actual('total_assets', ['total_assets', 'totalAssets'])
    if has_actual and actual_ta:
        err = calc_error(predicted.total_assets, actual_ta)
        print(f"★ {'TOTAL ASSETS':<38} {'Sum':<12} {fmt_currency(predicted.total_assets):>16} {fmt_currency(actual_ta):>16} {err:>12}")
    else:
        print(f"★ {'TOTAL ASSETS':<38} {'Sum':<12} {fmt_currency(predicted.total_assets):>16}")
    
    # Balance Sheet - Liabilities & Equity
    print(f"\n{'─'*100}")
    print(f"{'BALANCE SHEET - LIABILITIES & EQUITY':^100}")
    print(f"{'─'*100}")
    print(header)
    print(f"{'─'*100}")
    
    print("CURRENT LIABILITIES")
    cl_items = [
        ('  Accounts Payable', 'Derived', predicted.accounts_payable, ['accounts_payable', 'accountsPayable', 'accountPayables']),
        ('  Accrued Expenses', 'Derived', predicted.accrued_expenses, ['accruedExpenses', 'accrued_expenses']),
        ('  Deferred Revenue', 'Derived', predicted.deferred_revenue, ['deferredRevenue', 'deferred_revenue']),
        ('  Other Current Liabilities', 'Derived', predicted.other_current_liabilities, ['otherCurrentLiabilities', 'other_current_liabilities']),
        ('TOTAL CURRENT LIABILITIES', 'Sum', predicted.total_current_liabilities, ['total_current_liabilities', 'totalCurrentLiabilities']),
    ]
    
    for name, typ, pred_val, keys in cl_items:
        marker = "★" if "TOTAL" in name else "○"
        actual_val = get_actual(keys[0], keys)
        if has_actual and actual_val is not None:
            err = calc_error(pred_val, actual_val)
            print(f"{marker} {name:<38} {typ:<12} {fmt_currency(pred_val):>16} {fmt_currency(actual_val):>16} {err:>12}")
        else:
            print(f"{marker} {name:<38} {typ:<12} {fmt_currency(pred_val):>16}")
    
    print("\nNON-CURRENT LIABILITIES")
    ncl_items = [
        ('  Long-term Debt', 'Constant', predicted.long_term_debt, ['long_term_debt', 'longTermDebt']),
        ('  Deferred Tax Liabilities', 'Derived', predicted.deferred_tax_liabilities, ['deferredTaxLiabilitiesNonCurrent', 'deferred_tax_liabilities']),
        ('  Other Non-current Liab.', 'Derived', predicted.other_noncurrent_liabilities, ['otherNonCurrentLiabilities', 'other_noncurrent_liabilities']),
        ('TOTAL NON-CURRENT LIABILITIES', 'Sum', predicted.total_noncurrent_liabilities, ['total_noncurrent_liabilities', 'totalNonCurrentLiabilities']),
    ]
    
    for name, typ, pred_val, keys in ncl_items:
        marker = "★" if "TOTAL" in name else "○"
        actual_val = get_actual(keys[0], keys)
        if has_actual and actual_val is not None:
            err = calc_error(pred_val, actual_val)
            print(f"{marker} {name:<38} {typ:<12} {fmt_currency(pred_val):>16} {fmt_currency(actual_val):>16} {err:>12}")
        else:
            print(f"{marker} {name:<38} {typ:<12} {fmt_currency(pred_val):>16}")
    
    actual_tl = get_actual('total_liabilities', ['total_liabilities', 'totalLiabilities'])
    if has_actual and actual_tl:
        err = calc_error(predicted.total_liabilities, actual_tl)
        print(f"\n★ {'TOTAL LIABILITIES':<38} {'A - E':<12} {fmt_currency(predicted.total_liabilities):>16} {fmt_currency(actual_tl):>16} {err:>12}")
    else:
        print(f"\n★ {'TOTAL LIABILITIES':<38} {'A - E':<12} {fmt_currency(predicted.total_liabilities):>16}")
    
    print("\nSHAREHOLDERS' EQUITY")
    eq_items = [
        ('  Common Stock', 'Constant', predicted.common_stock, ['common_stock', 'commonStock']),
        ('  Retained Earnings', 'IS Linkage', predicted.retained_earnings, ['retained_earnings', 'retainedEarnings']),
        ('  Treasury Stock', 'Derived', predicted.treasury_stock, ['treasury_stock', 'treasuryStock']),
        ('  Accumulated OCI', 'Constant', predicted.aoci, ['aoci', 'accumulatedOtherComprehensiveIncomeLoss']),
    ]
    
    for name, typ, pred_val, keys in eq_items:
        actual_val = get_actual(keys[0], keys)
        if has_actual and actual_val is not None:
            err = calc_error(pred_val, actual_val)
            print(f"○ {name:<38} {typ:<12} {fmt_currency(pred_val):>16} {fmt_currency(actual_val):>16} {err:>12}")
        else:
            print(f"○ {name:<38} {typ:<12} {fmt_currency(pred_val):>16}")
    
    actual_te = get_actual('total_equity', ['total_equity', 'totalEquity', 'totalStockholdersEquity'])
    if has_actual and actual_te:
        err = calc_error(predicted.total_equity, actual_te)
        print(f"\n★ {'TOTAL EQUITY':<38} {'Sum':<12} {fmt_currency(predicted.total_equity):>16} {fmt_currency(actual_te):>16} {err:>12}")
    else:
        print(f"\n★ {'TOTAL EQUITY':<38} {'Sum':<12} {fmt_currency(predicted.total_equity):>16}")
    
    # Identity Check - Comprehensive
    print(f"\n{'─'*100}")
    print(f"{'ACCOUNTING IDENTITY & LINKAGE VERIFICATION':^100}")
    print(f"{'─'*100}")
    
    all_pass = True
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 1. INCOME STATEMENT LINKAGES
    # ═══════════════════════════════════════════════════════════════════════════════
    print(f"\n{'[1] INCOME STATEMENT LINKAGES':─<80}")
    
    # Gross Profit = Revenue - COGS
    gp_calc = predicted.revenue - predicted.cogs
    gp_diff = abs(gp_calc - predicted.gross_profit)
    gp_pass = gp_diff < 1
    all_pass = all_pass and gp_pass
    print(f"  Revenue - COGS = Gross Profit")
    print(f"    {fmt_currency(predicted.revenue)} - {fmt_currency(predicted.cogs)} = {fmt_currency(gp_calc)}")
    print(f"    Reported Gross Profit: {fmt_currency(predicted.gross_profit)}  {'✓' if gp_pass else '✗'}")
    
    # EBITDA = Gross Profit - OpEx
    ebitda_calc = predicted.gross_profit - predicted.opex
    ebitda_diff = abs(ebitda_calc - predicted.ebitda)
    ebitda_pass = ebitda_diff < 1
    all_pass = all_pass and ebitda_pass
    print(f"\n  Gross Profit - OpEx = EBITDA")
    print(f"    {fmt_currency(predicted.gross_profit)} - {fmt_currency(predicted.opex)} = {fmt_currency(ebitda_calc)}")
    print(f"    Reported EBITDA: {fmt_currency(predicted.ebitda)}  {'✓' if ebitda_pass else '✗'}")
    
    # EBIT = EBITDA - Depreciation
    ebit_calc = predicted.ebitda - predicted.depreciation
    ebit_diff = abs(ebit_calc - predicted.ebit)
    ebit_pass = ebit_diff < 1
    all_pass = all_pass and ebit_pass
    print(f"\n  EBITDA - Depreciation = EBIT")
    print(f"    {fmt_currency(predicted.ebitda)} - {fmt_currency(predicted.depreciation)} = {fmt_currency(ebit_calc)}")
    print(f"    Reported EBIT: {fmt_currency(predicted.ebit)}  {'✓' if ebit_pass else '✗'}")
    
    # EBT = EBIT - Interest Expense + Interest Income + Other Income
    other_income = getattr(predicted, 'other_income', 0) or 0
    ebt_calc = predicted.ebit - predicted.interest_expense + predicted.interest_income + other_income
    ebt_diff = abs(ebt_calc - predicted.ebt)
    ebt_pass = ebt_diff < 1
    all_pass = all_pass and ebt_pass
    print(f"\n  EBIT - Interest Exp + Interest Inc + Other = EBT")
    if abs(other_income) > 1:
        print(f"    {fmt_currency(predicted.ebit)} - {fmt_currency(predicted.interest_expense)} + {fmt_currency(predicted.interest_income)} + {fmt_currency(other_income)} = {fmt_currency(ebt_calc)}")
    else:
        print(f"    {fmt_currency(predicted.ebit)} - {fmt_currency(predicted.interest_expense)} + {fmt_currency(predicted.interest_income)} = {fmt_currency(ebt_calc)}")
    print(f"    Reported EBT: {fmt_currency(predicted.ebt)}  {'✓' if ebt_pass else '✗'}")
    
    # Net Income = EBT - Tax
    ni_calc = predicted.ebt - predicted.income_tax
    ni_diff = abs(ni_calc - predicted.net_income)
    ni_pass = ni_diff < 1
    all_pass = all_pass and ni_pass
    print(f"\n  EBT - Income Tax = Net Income")
    print(f"    {fmt_currency(predicted.ebt)} - {fmt_currency(predicted.income_tax)} = {fmt_currency(ni_calc)}")
    print(f"    Reported Net Income: {fmt_currency(predicted.net_income)}  {'✓' if ni_pass else '✗'}")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 2. BALANCE SHEET COMPONENT SUMS
    # ═══════════════════════════════════════════════════════════════════════════════
    print(f"\n{'[2] BALANCE SHEET COMPONENT SUMS':─<80}")
    
    # Total Current Assets
    ca_calc = (predicted.cash + predicted.short_term_investments + predicted.accounts_receivable + 
               predicted.inventory + predicted.prepaid_expenses + predicted.other_current_assets)
    ca_diff = abs(ca_calc - predicted.total_current_assets)
    ca_pass = ca_diff < 1
    all_pass = all_pass and ca_pass
    print(f"  Current Assets Sum:")
    print(f"    Cash + STI + AR + Inventory + Prepaid + Other CA")
    print(f"    = {fmt_currency(predicted.cash)} + {fmt_currency(predicted.short_term_investments)} + {fmt_currency(predicted.accounts_receivable)}")
    print(f"      + {fmt_currency(predicted.inventory)} + {fmt_currency(predicted.prepaid_expenses)} + {fmt_currency(predicted.other_current_assets)}")
    print(f"    = {fmt_currency(ca_calc)}")
    print(f"    Reported Total Current Assets: {fmt_currency(predicted.total_current_assets)}  {'✓' if ca_pass else '✗'}")
    
    # Total Non-Current Assets
    nca_calc = (predicted.ppe_net + predicted.goodwill + predicted.intangible_assets + 
                predicted.long_term_investments + predicted.other_noncurrent_assets)
    nca_diff = abs(nca_calc - predicted.total_noncurrent_assets)
    nca_pass = nca_diff < 1
    all_pass = all_pass and nca_pass
    print(f"\n  Non-Current Assets Sum:")
    print(f"    PPE + Goodwill + Intangibles + LTI + Other NCA")
    print(f"    = {fmt_currency(predicted.ppe_net)} + {fmt_currency(predicted.goodwill)} + {fmt_currency(predicted.intangible_assets)}")
    print(f"      + {fmt_currency(predicted.long_term_investments)} + {fmt_currency(predicted.other_noncurrent_assets)}")
    print(f"    = {fmt_currency(nca_calc)}")
    print(f"    Reported Total Non-Current Assets: {fmt_currency(predicted.total_noncurrent_assets)}  {'✓' if nca_pass else '✗'}")
    
    # Total Assets
    ta_calc = predicted.total_current_assets + predicted.total_noncurrent_assets
    ta_diff = abs(ta_calc - predicted.total_assets)
    ta_pass = ta_diff < 1
    all_pass = all_pass and ta_pass
    print(f"\n  Total Assets = Current + Non-Current")
    print(f"    {fmt_currency(predicted.total_current_assets)} + {fmt_currency(predicted.total_noncurrent_assets)} = {fmt_currency(ta_calc)}")
    print(f"    Reported Total Assets: {fmt_currency(predicted.total_assets)}  {'✓' if ta_pass else '✗'}")
    
    # Total Current Liabilities
    cl_calc = (predicted.accounts_payable + predicted.accrued_expenses + predicted.deferred_revenue +
               predicted.other_current_liabilities + predicted.short_term_debt + predicted.current_portion_ltd)
    cl_diff = abs(cl_calc - predicted.total_current_liabilities)
    cl_pass = cl_diff < 1
    all_pass = all_pass and cl_pass
    print(f"\n  Current Liabilities Sum:")
    print(f"    AP + Accrued + Deferred Rev + Other CL + STD + Current LTD")
    print(f"    = {fmt_currency(predicted.accounts_payable)} + {fmt_currency(predicted.accrued_expenses)} + {fmt_currency(predicted.deferred_revenue)}")
    print(f"      + {fmt_currency(predicted.other_current_liabilities)} + {fmt_currency(predicted.short_term_debt)} + {fmt_currency(predicted.current_portion_ltd)}")
    print(f"    = {fmt_currency(cl_calc)}")
    print(f"    Reported Total Current Liabilities: {fmt_currency(predicted.total_current_liabilities)}  {'✓' if cl_pass else '✗'}")
    
    # Total Equity
    eq_calc = (predicted.common_stock + predicted.additional_paid_in_capital + 
               predicted.retained_earnings + predicted.treasury_stock + predicted.aoci)
    eq_diff = abs(eq_calc - predicted.total_equity)
    eq_pass = eq_diff < 1
    all_pass = all_pass and eq_pass
    print(f"\n  Equity Sum:")
    print(f"    Common Stock + APIC + Retained Earnings + Treasury + AOCI")
    print(f"    = {fmt_currency(predicted.common_stock)} + {fmt_currency(predicted.additional_paid_in_capital)} + {fmt_currency(predicted.retained_earnings)}")
    print(f"      + {fmt_currency(predicted.treasury_stock)} + {fmt_currency(predicted.aoci)}")
    print(f"    = {fmt_currency(eq_calc)}")
    print(f"    Reported Total Equity: {fmt_currency(predicted.total_equity)}  {'✓' if eq_pass else '✗'}")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 3. FUNDAMENTAL BALANCE SHEET IDENTITY
    # ═══════════════════════════════════════════════════════════════════════════════
    print(f"\n{'[3] FUNDAMENTAL BALANCE SHEET IDENTITY':─<80}")
    
    a, l, e = predicted.total_assets, predicted.total_liabilities, predicted.total_equity
    bs_diff = abs(a - l - e)
    bs_pass = bs_diff < 1
    all_pass = all_pass and bs_pass
    print(f"  Assets = Liabilities + Equity")
    print(f"    {fmt_currency(a)} = {fmt_currency(l)} + {fmt_currency(e)}")
    print(f"    {fmt_currency(a)} = {fmt_currency(l + e)}  {'✓' if bs_pass else '✗'}")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 4. CASH FLOW STATEMENT LINKAGES
    # ═══════════════════════════════════════════════════════════════════════════════
    print(f"\n{'[4] CASH FLOW STATEMENT LINKAGES':─<80}")
    
    # CFO = NI + D&A + ΔWC
    delta_wc = (predicted.cf_change_receivables + predicted.cf_change_inventory + 
                predicted.cf_change_payables + predicted.cf_change_other)
    cfo_calc = predicted.cf_net_income + predicted.cf_depreciation + delta_wc
    cfo_diff = abs(cfo_calc - predicted.cf_operating)
    cfo_pass = cfo_diff < 1
    all_pass = all_pass and cfo_pass
    print(f"  CFO = Net Income + D&A + ΔWorking Capital")
    print(f"    Net Income:     {fmt_currency(predicted.cf_net_income)}")
    print(f"    + Depreciation: {fmt_currency(predicted.cf_depreciation)}")
    print(f"    + Δ Receivables:{fmt_currency(predicted.cf_change_receivables)}")
    print(f"    + Δ Inventory:  {fmt_currency(predicted.cf_change_inventory)}")
    print(f"    + Δ Payables:   {fmt_currency(predicted.cf_change_payables)}")
    print(f"    + Δ Other:      {fmt_currency(predicted.cf_change_other)}")
    print(f"    = CFO:          {fmt_currency(cfo_calc)}")
    print(f"    Reported CFO: {fmt_currency(predicted.cf_operating)}  {'✓' if cfo_pass else '✗'}")
    
    # CFI = CapEx + Acquisitions + Investments
    cfi_calc = predicted.cf_capex + predicted.cf_acquisitions + predicted.cf_investments
    cfi_diff = abs(cfi_calc - predicted.cf_investing)
    cfi_pass = cfi_diff < 1
    all_pass = all_pass and cfi_pass
    print(f"\n  CFI = CapEx + Acquisitions + Investments")
    print(f"    {fmt_currency(predicted.cf_capex)} + {fmt_currency(predicted.cf_acquisitions)} + {fmt_currency(predicted.cf_investments)} = {fmt_currency(cfi_calc)}")
    print(f"    Reported CFI: {fmt_currency(predicted.cf_investing)}  {'✓' if cfi_pass else '✗'}")
    
    # CFF = Debt + Dividends + Buybacks + Stock
    cff_calc = (predicted.cf_debt_issued + predicted.cf_debt_repaid + 
                predicted.cf_dividends + predicted.cf_buybacks + predicted.cf_stock_issued)
    cff_diff = abs(cff_calc - predicted.cf_financing)
    cff_pass = cff_diff < 1
    all_pass = all_pass and cff_pass
    print(f"\n  CFF = Debt Issued + Debt Repaid + Dividends + Buybacks + Stock Issued")
    print(f"    {fmt_currency(predicted.cf_debt_issued)} + {fmt_currency(predicted.cf_debt_repaid)} + {fmt_currency(predicted.cf_dividends)}")
    print(f"    + {fmt_currency(predicted.cf_buybacks)} + {fmt_currency(predicted.cf_stock_issued)} = {fmt_currency(cff_calc)}")
    print(f"    Reported CFF: {fmt_currency(predicted.cf_financing)}  {'✓' if cff_pass else '✗'}")
    
    # Net Change = CFO + CFI + CFF
    net_change_calc = predicted.cf_operating + predicted.cf_investing + predicted.cf_financing
    nc_diff = abs(net_change_calc - predicted.cf_net_change)
    nc_pass = nc_diff < 1
    all_pass = all_pass and nc_pass
    print(f"\n  Net Change in Cash = CFO + CFI + CFF")
    print(f"    {fmt_currency(predicted.cf_operating)} + {fmt_currency(predicted.cf_investing)} + {fmt_currency(predicted.cf_financing)} = {fmt_currency(net_change_calc)}")
    print(f"    Reported Net Change: {fmt_currency(predicted.cf_net_change)}  {'✓' if nc_pass else '✗'}")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 5. CASH FLOW TO BALANCE SHEET LINKAGE
    # ═══════════════════════════════════════════════════════════════════════════════
    print(f"\n{'[5] CASH FLOW → BALANCE SHEET LINKAGE':─<80}")
    
    # Ending Cash = Beginning Cash + Net Change
    ending_cash_calc = predicted.cf_beginning_cash + predicted.cf_net_change
    ec_diff = abs(ending_cash_calc - predicted.cf_ending_cash)
    ec_pass = ec_diff < 1
    all_pass = all_pass and ec_pass
    print(f"  Ending Cash = Beginning Cash + Net Change")
    print(f"    {fmt_currency(predicted.cf_beginning_cash)} + {fmt_currency(predicted.cf_net_change)} = {fmt_currency(ending_cash_calc)}")
    print(f"    Reported Ending Cash (CF): {fmt_currency(predicted.cf_ending_cash)}  {'✓' if ec_pass else '✗'}")
    
    # CF Ending Cash = BS Cash
    cash_match_diff = abs(predicted.cf_ending_cash - predicted.cash)
    cash_match_pass = cash_match_diff < 1
    all_pass = all_pass and cash_match_pass
    print(f"\n  CF Ending Cash = Balance Sheet Cash")
    print(f"    CF Ending Cash: {fmt_currency(predicted.cf_ending_cash)}")
    print(f"    BS Cash:        {fmt_currency(predicted.cash)}  {'✓' if cash_match_pass else '✗'}")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 6. PERIOD-OVER-PERIOD LINKAGES (requires prior period data)
    # ═══════════════════════════════════════════════════════════════════════════════
    re_pass = True
    ppe_pass = True
    ts_pass = True
    
    if has_prior:
        print(f"\n{'[6] PERIOD-OVER-PERIOD LINKAGES':─<80}")
        
        # Detect which method the company uses
        uses_treasury_stock = prior.treasury_stock != 0
        
        if uses_treasury_stock:
            # Treasury Stock Method: Buybacks go to treasury stock, not RE
            re_calc = prior.retained_earnings + predicted.net_income + predicted.cf_dividends
            ts_calc = prior.treasury_stock + predicted.cf_buybacks
            
            print(f"  [Treasury Stock Method Detected]")
            print(f"\n  Retained Earnings: RE_t = RE_{{t-1}} + Net Income - Dividends")
            print(f"    Prior RE:         {fmt_currency(prior.retained_earnings)}")
            print(f"    + Net Income:     {fmt_currency(predicted.net_income)}")
            print(f"    + Dividends (neg):{fmt_currency(predicted.cf_dividends)}")
            print(f"    = Calculated RE:  {fmt_currency(re_calc)}")
            print(f"    Reported RE:      {fmt_currency(predicted.retained_earnings)}  {'✓' if abs(re_calc - predicted.retained_earnings) < 1 else '✗'}")
            
            print(f"\n  Treasury Stock: TS_t = TS_{{t-1}} - Buybacks")
            print(f"    Prior TS:         {fmt_currency(prior.treasury_stock)}")
            print(f"    + Buybacks (neg): {fmt_currency(predicted.cf_buybacks)}")
            print(f"    = Calculated TS:  {fmt_currency(ts_calc)}")
            print(f"    Reported TS:      {fmt_currency(predicted.treasury_stock)}  {'✓' if abs(ts_calc - predicted.treasury_stock) < 1 else '✗'}")
            
            re_pass = abs(re_calc - predicted.retained_earnings) < 1
            ts_pass = abs(ts_calc - predicted.treasury_stock) < 1
        else:
            # Retirement Method: Buybacks reduce RE directly
            re_calc = prior.retained_earnings + predicted.net_income + predicted.cf_dividends + predicted.cf_buybacks
            
            print(f"  [Share Retirement Method Detected]")
            print(f"\n  Retained Earnings: RE_t = RE_{{t-1}} + Net Income - Dividends - Buybacks")
            print(f"    Prior RE:         {fmt_currency(prior.retained_earnings)}")
            print(f"    + Net Income:     {fmt_currency(predicted.net_income)}")
            print(f"    + Dividends (neg):{fmt_currency(predicted.cf_dividends)}")
            print(f"    + Buybacks (neg): {fmt_currency(predicted.cf_buybacks)}")
            print(f"    = Calculated RE:  {fmt_currency(re_calc)}")
            print(f"    Reported RE:      {fmt_currency(predicted.retained_earnings)}  {'✓' if abs(re_calc - predicted.retained_earnings) < 1 else '✗'}")
            
            re_pass = abs(re_calc - predicted.retained_earnings) < 1
        
        all_pass = all_pass and re_pass and ts_pass
        
        # PPE: PPE_t = PPE_{t-1} + CapEx - Depreciation
        ppe_calc = prior.ppe_net - predicted.cf_capex - predicted.depreciation
        # Note: cf_capex is negative
        ppe_diff = abs(ppe_calc - predicted.ppe_net)
        ppe_pass = ppe_diff < 1
        all_pass = all_pass and ppe_pass
        print(f"\n  PP&E: PPE_t = PPE_{{t-1}} + CapEx - Depreciation")
        print(f"    Prior PPE:        {fmt_currency(prior.ppe_net)}")
        print(f"    + CapEx (neg):    {fmt_currency(predicted.cf_capex)}")
        print(f"    - Depreciation:   {fmt_currency(-predicted.depreciation)}")
        print(f"    = Calculated PPE: {fmt_currency(ppe_calc)}")
        print(f"    Reported PPE:     {fmt_currency(predicted.ppe_net)}  {'✓' if ppe_pass else '✗'}")
    else:
        print(f"\n{'[6] PERIOD-OVER-PERIOD LINKAGES':─<80}")
        print(f"  (Requires prior period data - skipped)")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*100}")
    print(f"{'IDENTITY VERIFICATION SUMMARY':^100}")
    print(f"{'═'*100}")
    
    checks = [
        ("Income Statement Linkages", gp_pass and ebitda_pass and ebit_pass and ebt_pass and ni_pass),
        ("Balance Sheet Component Sums", ca_pass and nca_pass and ta_pass and cl_pass and eq_pass),
        ("Fundamental Identity (A=L+E)", bs_pass),
        ("Cash Flow Statement Linkages", cfo_pass and cfi_pass and cff_pass and nc_pass),
        ("CF → BS Cash Linkage", ec_pass and cash_match_pass),
        ("Period-over-Period (RE, PPE)", re_pass and ppe_pass if has_prior else True),
    ]
    
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:<50} {status}")
    
    print(f"\n  {'ALL IDENTITIES:':<50} {'✓ ALL PASS' if all_pass else '✗ SOME FAILED'}")
    print(f"{'━'*100}")
