"""
pdf_report.py

Generate PDF reports for financial statement predictions.
"""

import os
from typing import Optional
import pandas as pd
import numpy as np

# Make reportlab optional
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.units import inch
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

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


def get_actual_value(actual: pd.Series, mappings: list) -> Optional[float]:
    """Get actual value from series using multiple possible column names."""
    if actual is None:
        return None
    for name in mappings:
        if name in actual.index:
            val = actual[name]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                return val
    return None


def generate_statement_pdf(
    predicted: CompleteFinancialStatements,
    actual: Optional[pd.Series],
    period: str,
    output_path: str,
    ticker: str = ""
) -> Optional[str]:
    """Generate PDF report for a single period's financial statements.
    
    Returns output_path if successful, None if reportlab not available.
    """
    if not HAS_REPORTLAB:
        return None
    
    doc = SimpleDocTemplate(output_path, pagesize=letter, 
                           topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=16, spaceAfter=20)
    story.append(Paragraph(f"{ticker} Financial Statements - {period}", title_style))
    story.append(Spacer(1, 10))
    
    has_actual = actual is not None
    
    # Helper to create section
    def add_section(title: str, items: list):
        # Section header
        header_style = ParagraphStyle('Header', parent=styles['Heading2'], fontSize=12, 
                                      textColor=colors.darkblue, spaceAfter=8)
        story.append(Paragraph(title, header_style))
        
        # Table header
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
    
    # Income Statement
    is_items = [
        ('Revenue', 'ML Driver', predicted.revenue, ['revenue', 'totalRevenue']),
        ('Cost of Goods Sold', 'ML Driver', predicted.cogs, ['cogs', 'costOfRevenue']),
        ('Gross Profit', 'Derived', predicted.gross_profit, ['gross_profit', 'grossProfit']),
        ('Operating Expenses', 'ML Driver', predicted.opex, ['opex', 'operatingExpenses']),
        ('EBITDA', 'Derived', predicted.ebitda, ['ebitda']),
        ('Depreciation', 'Derived', predicted.depreciation, ['depreciation']),
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
    
    # Balance Sheet - Assets
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
    
    # Balance Sheet - Liabilities & Equity
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
    
    # Accounting Identity Verification (All checks)
    header_style = ParagraphStyle('Header', parent=styles['Heading2'], fontSize=12,
                                  textColor=colors.darkblue, spaceAfter=8)
    story.append(Paragraph("ACCOUNTING IDENTITY VERIFICATION", header_style))
    
    checks = []
    
    # 1. Balance Sheet: A = L + E
    a_eq = abs(predicted.total_assets - predicted.total_liabilities - predicted.total_equity) < 1
    checks.append([
        'A = L + E',
        f"{fmt_currency(predicted.total_assets)} = {fmt_currency(predicted.total_liabilities)} + {fmt_currency(predicted.total_equity)}",
        '✓' if a_eq else '✗'
    ])
    
    # 2. Current Assets Sum
    ca_sum = predicted.cash + predicted.short_term_investments + predicted.accounts_receivable + predicted.inventory + getattr(predicted, 'prepaid_expenses', 0) + predicted.other_current_assets
    ca_eq = abs(predicted.total_current_assets - ca_sum) < 1
    checks.append([
        'Current Assets Sum',
        f"{fmt_currency(predicted.total_current_assets)} = sum of components",
        '✓' if ca_eq else '✗'
    ])
    
    # 3. Non-Current Assets Sum
    nca_sum = predicted.ppe_net + predicted.goodwill + getattr(predicted, 'intangible_assets', 0) + predicted.long_term_investments + predicted.other_noncurrent_assets
    nca_eq = abs(predicted.total_noncurrent_assets - nca_sum) < 1
    checks.append([
        'Non-Current Assets Sum',
        f"{fmt_currency(predicted.total_noncurrent_assets)} = sum of components",
        '✓' if nca_eq else '✗'
    ])
    
    # 4. Total Assets = CA + NCA
    ta_eq = abs(predicted.total_assets - predicted.total_current_assets - predicted.total_noncurrent_assets) < 1
    checks.append([
        'Total Assets = CA + NCA',
        f"{fmt_currency(predicted.total_assets)} = {fmt_currency(predicted.total_current_assets)} + {fmt_currency(predicted.total_noncurrent_assets)}",
        '✓' if ta_eq else '✗'
    ])
    
    # 5. Total Liabilities = CL + NCL
    tl_eq = abs(predicted.total_liabilities - predicted.total_current_liabilities - predicted.total_noncurrent_liabilities) < 1
    checks.append([
        'Total Liab = CL + NCL',
        f"{fmt_currency(predicted.total_liabilities)} = {fmt_currency(predicted.total_current_liabilities)} + {fmt_currency(predicted.total_noncurrent_liabilities)}",
        '✓' if tl_eq else '✗'
    ])
    
    # 6. Gross Profit = Rev - COGS
    gp_eq = abs(predicted.gross_profit - (predicted.revenue - predicted.cogs)) < 1
    checks.append([
        'GP = Rev - COGS',
        f"{fmt_currency(predicted.gross_profit)} = {fmt_currency(predicted.revenue)} - {fmt_currency(predicted.cogs)}",
        '✓' if gp_eq else '✗'
    ])
    
    # 7. EBITDA = GP - OpEx
    ebitda_eq = abs(predicted.ebitda - (predicted.gross_profit - predicted.opex)) < 1
    checks.append([
        'EBITDA = GP - OpEx',
        f"{fmt_currency(predicted.ebitda)} = {fmt_currency(predicted.gross_profit)} - {fmt_currency(predicted.opex)}",
        '✓' if ebitda_eq else '✗'
    ])
    
    # 8. EBIT = EBITDA - D&A
    ebit_eq = abs(predicted.ebit - (predicted.ebitda - predicted.depreciation)) < 1
    checks.append([
        'EBIT = EBITDA - D&A',
        f"{fmt_currency(predicted.ebit)} = {fmt_currency(predicted.ebitda)} - {fmt_currency(predicted.depreciation)}",
        '✓' if ebit_eq else '✗'
    ])
    
    # 9. Net Income = EBT - Tax
    ni_eq = abs(predicted.net_income - (predicted.ebt - predicted.income_tax)) < 1
    checks.append([
        'NI = EBT - Tax',
        f"{fmt_currency(predicted.net_income)} = {fmt_currency(predicted.ebt)} - {fmt_currency(predicted.income_tax)}",
        '✓' if ni_eq else '✗'
    ])
    
    # 10. Equity Components
    equity_sum = predicted.common_stock + predicted.retained_earnings + predicted.aoci
    if hasattr(predicted, 'additional_paid_in_capital'):
        equity_sum += predicted.additional_paid_in_capital
    if hasattr(predicted, 'treasury_stock'):
        equity_sum += predicted.treasury_stock  # Usually negative
    equity_eq = abs(predicted.total_equity - equity_sum) < 1e6
    checks.append([
        'Equity Components',
        f"{fmt_currency(predicted.total_equity)} = sum of equity items",
        '✓' if equity_eq else '✗'
    ])
    
    # Create verification table
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
    
    # Summary
    all_pass = all(c[2] == '✓' for c in checks)
    summary_style = ParagraphStyle('Summary', parent=styles['Normal'], fontSize=10,
                                   textColor=colors.green if all_pass else colors.red,
                                   spaceBefore=10)
    story.append(Paragraph(
        f"Overall: {sum(1 for c in checks if c[2] == '✓')}/{len(checks)} identities verified {'✓ ALL PASS' if all_pass else '✗ SOME FAILED'}",
        summary_style
    ))
    
    # Build PDF
    doc.build(story)
    return output_path


def setup_output_folder(ticker: str, base_path: str = None) -> str:
    """Create output folder for ticker if it doesn't exist."""
    if base_path is None:
        # Try to use 'outputs' folder in current directory, fallback to cwd
        outputs_dir = os.path.join(os.getcwd(), "outputs")
        if os.path.exists(outputs_dir):
            base_path = outputs_dir
        else:
            base_path = os.getcwd()
    folder_path = os.path.join(base_path, f"{ticker}_statements")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def generate_ensemble_comparison_pdf(
    predictions: dict,
    evaluation: dict,
    ticker: str,
    output_path: str,
    actual_data: pd.DataFrame = None,
    n_periods: int = 4
) -> Optional[str]:
    """
    Generate comprehensive PDF report with ML vs LLM comparison.
    
    Args:
        predictions: Dict with predictions from all approaches
        evaluation: Dict with MAPE results for each approach
        ticker: Stock ticker
        output_path: Output PDF path
        actual_data: Actual data for comparison (optional)
        n_periods: Number of forecast periods
        
    Returns:
        output_path if successful, None otherwise
    """
    if not HAS_REPORTLAB:
        print("  ⚠ reportlab not installed. Run: pip install reportlab")
        return None
    
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                           topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # ====== PAGE 1: Title & Summary ======
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=18, 
                                 spaceAfter=10, textColor=colors.darkblue)
    story.append(Paragraph(f"{ticker} - Ensemble Forecast Report", title_style))
    
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=10,
                                    textColor=colors.grey, spaceAfter=20)
    from datetime import datetime
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    
    # Executive Summary Box
    header_style = ParagraphStyle('Header', parent=styles['Heading2'], fontSize=14,
                                  textColor=colors.darkblue, spaceAfter=10, spaceBefore=15)
    story.append(Paragraph("EXECUTIVE SUMMARY", header_style))
    
    if evaluation and 'best_approach' in evaluation:
        best = evaluation['best_approach']
        best_mape = evaluation['best_mape']
        
        # Grade determination
        if best_mape < 10:
            grade = "⭐⭐⭐⭐⭐ Excellent"
            grade_color = colors.green
        elif best_mape < 15:
            grade = "⭐⭐⭐⭐ Very Good"
            grade_color = colors.Color(0.4, 0.7, 0.2)
        elif best_mape < 20:
            grade = "⭐⭐⭐ Good"
            grade_color = colors.orange
        else:
            grade = "⭐⭐ Fair"
            grade_color = colors.red
        
        summary_data = [
            ['Metric', 'Value'],
            ['Best Approach', best],
            ['Best MAPE', f"{best_mape:.2f}%"],
            ['Grade', grade],
            ['Forecast Periods', f"{n_periods} quarters"],
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.3, 0.5)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(summary_table)
    
    story.append(Spacer(1, 20))
    
    # ====== PAGE 2: Approach Comparison ======
    story.append(Paragraph("APPROACH COMPARISON", header_style))
    
    if evaluation and 'results' in evaluation:
        # Create comparison table
        comp_header = ['Approach', 'Revenue', 'Net Income', 'Total Assets', 'Total Equity', 'Avg MAPE']
        comp_data = [comp_header]
        
        for approach_name, res in evaluation['results'].items():
            mapes = res.get('mapes', {})
            row = [
                approach_name,
                f"{mapes.get('revenue', 0):.2f}%",
                f"{mapes.get('net_income', 0):.2f}%",
                f"{mapes.get('total_assets', 0):.2f}%",
                f"{mapes.get('total_equity', 0):.2f}%",
                f"{res.get('avg_mape', 0):.2f}%"
            ]
            comp_data.append(row)
        
        comp_table = Table(comp_data, colWidths=[1.3*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.3, 0.5)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        # Highlight best approach
        if evaluation.get('best_approach'):
            for i, row in enumerate(comp_data[1:], 1):
                if row[0] == evaluation['best_approach']:
                    comp_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, i), (-1, i), colors.Color(0.9, 1.0, 0.9)),
                        ('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold'),
                    ]))
        
        story.append(comp_table)
    
    story.append(Spacer(1, 20))
    
    # ====== Methodology Description ======
    story.append(Paragraph("METHODOLOGY", header_style))
    
    method_style = ParagraphStyle('Method', parent=styles['Normal'], fontSize=9,
                                  spaceAfter=8, leftIndent=20)
    
    methods = [
        ("<b>1. ML Only (XGBoost Quantile Regression)</b>", 
         "Pure machine learning approach using XGBoost to predict 5 key financial drivers "
         "(revenue growth, COGS margin, OpEx margin, CapEx ratio, net margin). "
         "Complete financial statements derived via accounting engine."),
        
        ("<b>2. ML + LLM Ratios (Hybrid)</b>",
         "Combines ML-predicted growth rates with LLM-adjusted margin ratios. "
         "ML captures short-term dynamics while LLM provides industry knowledge adjustment."),
        
        ("<b>3. Pure LLM (Claude)</b>",
         "LLM directly predicts absolute financial values based on historical patterns "
         "and industry context. Provides reasoning for predictions."),
    ]
    
    for title, desc in methods:
        story.append(Paragraph(title, method_style))
        story.append(Paragraph(desc, method_style))
    
    story.append(Spacer(1, 20))
    
    # ====== PAGE 3+: Detailed Forecasts ======
    story.append(Paragraph("DETAILED QUARTERLY FORECASTS", header_style))
    
    # ML Only forecasts (complete statements)
    if 'ML Only' in predictions and predictions['ML Only'] is not None:
        ml_data = predictions['ML Only']
        if 'statements' in ml_data and ml_data['statements']:
            story.append(Paragraph("<b>ML Only Forecasts:</b>", method_style))
            
            ml_header = ['Period', 'Revenue', 'Net Income', 'Total Assets', 'Total Equity']
            ml_rows = [ml_header]
            
            for stmt in ml_data['statements'][:n_periods]:
                ml_rows.append([
                    stmt.period,
                    fmt_currency(stmt.revenue),
                    fmt_currency(stmt.net_income),
                    fmt_currency(stmt.total_assets),
                    fmt_currency(stmt.total_equity),
                ])
            
            ml_table = Table(ml_rows, colWidths=[1*inch, 1.3*inch, 1.3*inch, 1.3*inch, 1.3*inch])
            ml_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(ml_table)
            story.append(Spacer(1, 15))
    
    # ML + LLM forecasts
    if 'ML + LLM' in predictions and predictions['ML + LLM'] is not None:
        hybrid_data = predictions['ML + LLM']
        if 'statements' in hybrid_data and hybrid_data['statements']:
            story.append(Paragraph("<b>ML + LLM Hybrid Forecasts:</b>", method_style))
            
            hybrid_header = ['Period', 'Revenue', 'Net Income', 'Total Assets', 'Total Equity']
            hybrid_rows = [hybrid_header]
            
            for stmt in hybrid_data['statements'][:n_periods]:
                hybrid_rows.append([
                    stmt.period,
                    fmt_currency(stmt.revenue),
                    fmt_currency(stmt.net_income),
                    fmt_currency(stmt.total_assets),
                    fmt_currency(stmt.total_equity),
                ])
            
            hybrid_table = Table(hybrid_rows, colWidths=[1*inch, 1.3*inch, 1.3*inch, 1.3*inch, 1.3*inch])
            hybrid_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(hybrid_table)
            
            # Add LLM reasoning if available
            if 'llm_reasoning' in hybrid_data and hybrid_data['llm_reasoning']:
                story.append(Spacer(1, 5))
                reasoning_style = ParagraphStyle('Reasoning', parent=styles['Normal'], 
                                                  fontSize=8, textColor=colors.grey, leftIndent=20)
                story.append(Paragraph(f"<i>LLM Reasoning: {hybrid_data['llm_reasoning']}</i>", reasoning_style))
            story.append(Spacer(1, 15))
    
    # Pure LLM forecasts
    if 'Pure LLM' in predictions and predictions['Pure LLM'] is not None:
        llm_data = predictions['Pure LLM']
        if 'revenue' in llm_data:
            story.append(Paragraph("<b>Pure LLM Forecasts:</b>", method_style))
            
            llm_header = ['Period', 'Revenue', 'Net Income', 'Total Assets', 'Total Equity']
            llm_rows = [llm_header]
            
            for i in range(min(n_periods, len(llm_data.get('revenue', [])))):
                llm_rows.append([
                    f"Q+{i+1}",
                    fmt_currency(llm_data['revenue'][i]),
                    fmt_currency(llm_data['net_income'][i]),
                    fmt_currency(llm_data['total_assets'][i]),
                    fmt_currency(llm_data['total_equity'][i]),
                ])
            
            llm_table = Table(llm_rows, colWidths=[1*inch, 1.3*inch, 1.3*inch, 1.3*inch, 1.3*inch])
            llm_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(llm_table)
            
            # Add source info
            if 'source' in llm_data:
                story.append(Spacer(1, 5))
                source_style = ParagraphStyle('Source', parent=styles['Normal'], 
                                              fontSize=8, textColor=colors.grey, leftIndent=20)
                story.append(Paragraph(f"<i>Source: {llm_data['source']}</i>", source_style))
    
    story.append(Spacer(1, 20))
    
    # ====== Final: Complete Balance Sheet (Best Approach) ======
    if evaluation and 'best_approach' in evaluation:
        best_approach = evaluation['best_approach']
        best_data = predictions.get(best_approach)
        
        if best_data and 'statements' in best_data and best_data['statements']:
            stmt = best_data['statements'][0]  # First forecast period
            
            story.append(Paragraph(f"COMPLETE BALANCE SHEET - {best_approach} (Q+1)", header_style))
            
            # Create full balance sheet table
            actual = actual_data.iloc[0] if actual_data is not None and len(actual_data) > 0 else None
            has_actual = actual is not None
            
            def add_bs_section(title: str, items: list):
                section_style = ParagraphStyle('Section', parent=styles['Heading3'], fontSize=11,
                                               textColor=colors.Color(0.3, 0.3, 0.5), spaceAfter=5, spaceBefore=10)
                story.append(Paragraph(title, section_style))
                
                if has_actual:
                    header = ['Item', 'Predicted', 'Actual', 'Error']
                    col_widths = [2.5*inch, 1.2*inch, 1.2*inch, 0.8*inch]
                else:
                    header = ['Item', 'Predicted']
                    col_widths = [3*inch, 1.5*inch]
                
                data = [header]
                for item in items:
                    name, pred_val, mappings = item
                    row = [name, fmt_currency(pred_val)]
                    if has_actual:
                        actual_val = get_actual_value(actual, mappings)
                        row.append(fmt_currency(actual_val) if actual_val else "N/A")
                        row.append(calc_error(pred_val, actual_val) if actual_val else "N/A")
                    data.append(row)
                
                table = Table(data, colWidths=col_widths)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.97, 0.97, 0.97)]),
                ]))
                story.append(table)
            
            # Income Statement
            is_items = [
                ('Revenue', stmt.revenue, ['revenue', 'totalRevenue']),
                ('COGS', stmt.cogs, ['cogs', 'costOfRevenue']),
                ('Gross Profit', stmt.gross_profit, ['grossProfit']),
                ('Operating Expenses', stmt.opex, ['operatingExpenses']),
                ('EBITDA', stmt.ebitda, ['ebitda']),
                ('EBIT', stmt.ebit, ['operatingIncome']),
                ('Net Income', stmt.net_income, ['netIncome']),
            ]
            add_bs_section("Income Statement", is_items)
            
            # Balance Sheet - Assets
            asset_items = [
                ('Cash & Equivalents', stmt.cash, ['cashAndCashEquivalents']),
                ('Total Current Assets', stmt.total_current_assets, ['totalCurrentAssets']),
                ('PP&E (Net)', stmt.ppe_net, ['propertyPlantEquipmentNet']),
                ('Total Non-Current Assets', stmt.total_noncurrent_assets, ['totalNonCurrentAssets']),
                ('TOTAL ASSETS', stmt.total_assets, ['totalAssets']),
            ]
            add_bs_section("Balance Sheet - Assets", asset_items)
            
            # Balance Sheet - Liabilities & Equity
            liab_items = [
                ('Total Current Liabilities', stmt.total_current_liabilities, ['totalCurrentLiabilities']),
                ('Long-term Debt', stmt.long_term_debt, ['longTermDebt']),
                ('Total Non-Current Liabilities', stmt.total_noncurrent_liabilities, ['totalNonCurrentLiabilities']),
                ('TOTAL LIABILITIES', stmt.total_liabilities, ['totalLiabilities']),
                ('Retained Earnings', stmt.retained_earnings, ['retainedEarnings']),
                ('TOTAL EQUITY', stmt.total_equity, ['totalEquity']),
            ]
            add_bs_section("Balance Sheet - Liabilities & Equity", liab_items)
            
            # Accounting Identity Check
            story.append(Spacer(1, 15))
            story.append(Paragraph("ACCOUNTING IDENTITY VERIFICATION", header_style))
            
            a_eq = abs(stmt.total_assets - stmt.total_liabilities - stmt.total_equity) < 1
            check_data = [
                ['Identity', 'Calculation', 'Status'],
                ['A = L + E', 
                 f"{fmt_currency(stmt.total_assets)} = {fmt_currency(stmt.total_liabilities)} + {fmt_currency(stmt.total_equity)}",
                 '✓ PASS' if a_eq else '✗ FAIL'],
            ]
            
            check_table = Table(check_data, colWidths=[1.5*inch, 3.5*inch, 0.8*inch])
            check_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (-1, 1), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('TEXTCOLOR', (-1, 1), (-1, -1), colors.green if a_eq else colors.red),
            ]))
            story.append(check_table)
    
    # ====== Footer ======
    story.append(Spacer(1, 30))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8,
                                  textColor=colors.grey, alignment=1)
    story.append(Paragraph("Generated by Balance Sheet Simulator - XGBoost Quantile + LLM Ensemble", footer_style))
    story.append(Paragraph("JP Morgan MLCOE 2026 Summer Associate Project", footer_style))
    
    # Build PDF
    try:
        doc.build(story)
        return output_path
    except Exception as e:
        print(f"  ⚠ Failed to generate PDF: {e}")
        return None
