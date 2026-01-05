"""
ensemble_validator.py

Rolling Validation for Ensemble Forecasting (Q2).
Same structure as Q1 rolling_validator, but compares ML vs LLM approaches.

For each validation round:
    1. Train on [1, t]
    2. Predict t+1 using all approaches (ML Only, ML+LLM, Pure LLM)
    3. Compare to actual t+1
    4. Save PDF with full statements
    5. Track MAPE for each approach
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

from .data_structures import CompleteFinancialStatements
from .quantile_simulator import QuantileSimulator
from .statement_printer import print_complete_statements
from .pdf_report import generate_statement_pdf, setup_output_folder, HAS_REPORTLAB, fmt_currency, calc_error, get_actual_value

# Make reportlab optional
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.units import inch
except ImportError:
    pass


class LLMForecasterSimple:
    """Simplified LLM forecaster for validation (uses historical averages as proxy)."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = None
        
    def _init_client(self):
        if self.client is None and self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                pass
    
    def predict_ratios(self, data: pd.DataFrame) -> Dict[str, float]:
        """Predict financial ratios using LLM or fallback to historical."""
        recent = data.tail(8) if len(data) >= 8 else data
        
        # Find columns
        def find_col(df, names):
            for n in names:
                if n in df.columns:
                    return n
            return None
        
        rev_col = find_col(recent, ['revenue', 'totalRevenue', 'sales_revenue'])
        cogs_col = find_col(recent, ['cogs', 'costOfRevenue', 'cost_of_goods_sold'])
        opex_col = find_col(recent, ['opex', 'operatingExpenses'])
        ni_col = find_col(recent, ['net_income', 'netIncome'])
        capex_col = find_col(recent, ['capex', 'capitalExpenditure'])
        
        avg_rev = recent[rev_col].mean() if rev_col else 1
        avg_cogs = recent[cogs_col].mean() if cogs_col else 0
        avg_opex = recent[opex_col].mean() if opex_col else 0
        avg_ni = recent[ni_col].mean() if ni_col else 0
        avg_capex = abs(recent[capex_col].mean()) if capex_col else 0
        
        # Calculate ratios
        cogs_margin = avg_cogs / avg_rev if avg_rev > 0 else 0.6
        opex_margin = avg_opex / avg_rev if avg_rev > 0 else 0.15
        net_margin = avg_ni / avg_rev if avg_rev > 0 else 0.1
        capex_ratio = avg_capex / avg_rev if avg_rev > 0 else 0.03
        
        # Revenue growth
        if rev_col and len(recent) >= 4:
            rev_growth = recent[rev_col].pct_change().mean()
            if np.isnan(rev_growth):
                rev_growth = 0.02
        else:
            rev_growth = 0.02
        
        # If we have API, try to get LLM adjustment
        if self.api_key:
            try:
                self._init_client()
                if self.client:
                    # Simple prompt for ratio adjustment
                    prompt = f"""Based on these historical averages for a company:
- Revenue growth: {rev_growth*100:.1f}%
- COGS margin: {cogs_margin*100:.1f}%
- OpEx margin: {opex_margin*100:.1f}%
- Net margin: {net_margin*100:.1f}%

Provide slight adjustments (+/- 10%) if you think the trend should change.
Return ONLY a JSON object with these exact keys:
{{"revenue_growth": X.XX, "cogs_margin": X.XX, "opex_margin": X.XX, "net_margin": X.XX, "capex_ratio": X.XX}}"""
                    
                    response = self.client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=200,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    import json
                    import re
                    text = response.content[0].text
                    match = re.search(r'\{[^{}]*\}', text)
                    if match:
                        result = json.loads(match.group())
                        return {
                            'revenue_growth': result.get('revenue_growth', rev_growth),
                            'cogs_margin': result.get('cogs_margin', cogs_margin),
                            'opex_margin': result.get('opex_margin', opex_margin),
                            'net_margin': result.get('net_margin', net_margin),
                            'capex_ratio': result.get('capex_ratio', capex_ratio),
                        }
            except Exception as e:
                pass  # Fall through to return historical
        
        return {
            'revenue_growth': rev_growth,
            'cogs_margin': cogs_margin,
            'opex_margin': opex_margin,
            'net_margin': net_margin,
            'capex_ratio': capex_ratio,
        }


def run_ensemble_validation(data: pd.DataFrame,
                           min_train_periods: int = 20,
                           n_test_periods: Optional[int] = None,
                           seq_length: int = 4,
                           verbose: bool = True,
                           ticker: str = "SAMPLE",
                           save_pdf: bool = True,
                           api_key: str = None) -> Dict:
    """
    Run rolling validation comparing ML vs LLM approaches.
    
    For each period t from min_train to T-1:
        1. Train ML model on [1, t]
        2. Get LLM predictions
        3. Predict t+1 using:
           - ML Only
           - ML + LLM Ratios
           - Pure LLM (if API available)
        4. Compare to actual t+1
        5. Save PDF with all approaches
    
    Returns:
        Dict with results for each approach
    """
    
    if verbose:
        print("\n" + "="*70)
        print("ENSEMBLE ROLLING VALIDATION")
        print("="*70)
        print(f"Data periods: {len(data)} | Min train: {min_train_periods}")
        if api_key:
            print(f"API Key: ****...{api_key[-4:]}")
        else:
            print("API Key: Not set (ML Only mode)")
    
    # Setup output folder
    output_folder = None
    if save_pdf and HAS_REPORTLAB:
        output_folder = setup_output_folder(ticker)
        if verbose:
            print(f"PDF output: {output_folder}")
    
    # Calculate validation rounds
    max_test = len(data) - min_train_periods - 1
    if n_test_periods is None:
        n_test_periods = min(5, max_test)  # Default to 5 rounds
    else:
        n_test_periods = min(n_test_periods, max_test)
    
    if n_test_periods <= 0:
        print(f"Error: Not enough data. Have {len(data)}, need at least {min_train_periods + 2}.")
        return {}
    
    start_t = len(data) - n_test_periods - 1
    
    if verbose:
        print(f"Validation rounds: {n_test_periods}")
        print("="*70)
    
    # Initialize LLM forecaster
    llm_forecaster = LLMForecasterSimple(api_key=api_key)
    
    # Results storage
    results = {
        'ML Only': {'errors': [], 'predictions': []},
        'ML + LLM': {'errors': [], 'predictions': []},
    }
    if api_key:
        results['Pure LLM'] = {'errors': [], 'predictions': []}
    
    # Run validation rounds
    for round_num, t in enumerate(range(start_t, len(data) - 1)):
        if verbose:
            print(f"\n{'='*70}")
            print(f"ROUND {round_num+1}/{n_test_periods}: Period {t+2}")
            print(f"{'='*70}")
        
        # Train data
        train_data = data.iloc[:t+1]
        actual = data.iloc[t+1]
        
        # ===== APPROACH 1: ML Only =====
        if verbose:
            print("\n[1] ML Only: Training...", end=" ", flush=True)
        
        simulator = QuantileSimulator(seq_length=seq_length)
        simulator.fit(train_data, verbose=False)
        
        forecasts = simulator.predict_distribution(simulator.last_drivers)
        ml_drivers = {d: forecasts[d].q50 for d in simulator.available_drivers}
        ml_drivers.setdefault('revenue_growth', 0.02)
        ml_drivers.setdefault('cogs_margin', 0.60)
        ml_drivers.setdefault('opex_margin', 0.15)
        ml_drivers.setdefault('capex_ratio', 0.03)
        ml_drivers.setdefault('net_margin', 0.15)
        
        prior = simulator.create_prior_statements(train_data.iloc[-1])
        ml_pred = simulator.accounting_engine.derive_statements(
            drivers=ml_drivers, prior=prior, period=f"Period {t+2}"
        )
        
        if verbose:
            print("Done.")
        
        # Calculate ML errors
        ml_errors = calculate_errors(ml_pred, actual)
        results['ML Only']['errors'].append(ml_errors)
        results['ML Only']['predictions'].append(ml_pred)
        
        # ===== APPROACH 2: ML + LLM Ratios =====
        if verbose:
            print("[2] ML + LLM: Getting ratios...", end=" ", flush=True)
        
        llm_ratios = llm_forecaster.predict_ratios(train_data)
        
        # Hybrid: ML growth + LLM margins
        hybrid_drivers = {
            'revenue_growth': ml_drivers['revenue_growth'],  # ML
            'cogs_margin': llm_ratios['cogs_margin'],        # LLM
            'opex_margin': llm_ratios['opex_margin'],        # LLM
            'capex_ratio': llm_ratios['capex_ratio'],        # LLM
            'net_margin': llm_ratios['net_margin'],          # LLM
        }
        
        hybrid_pred = simulator.accounting_engine.derive_statements(
            drivers=hybrid_drivers, prior=prior, period=f"Period {t+2}"
        )
        
        if verbose:
            print("Done.")
        
        hybrid_errors = calculate_errors(hybrid_pred, actual)
        results['ML + LLM']['errors'].append(hybrid_errors)
        results['ML + LLM']['predictions'].append(hybrid_pred)
        
        # ===== APPROACH 3: Pure LLM (if API available) =====
        if api_key and 'Pure LLM' in results:
            if verbose:
                print("[3] Pure LLM: Predicting...", end=" ", flush=True)
            
            # Use LLM ratios for everything
            llm_pred = simulator.accounting_engine.derive_statements(
                drivers=llm_ratios, prior=prior, period=f"Period {t+2}"
            )
            
            if verbose:
                print("Done.")
            
            llm_errors = calculate_errors(llm_pred, actual)
            results['Pure LLM']['errors'].append(llm_errors)
            results['Pure LLM']['predictions'].append(llm_pred)
        
        # Print round summary
        if verbose:
            print(f"\n{'Approach':<15} {'Revenue':>12} {'Net Income':>12} {'Assets':>12} {'Equity':>12}")
            print("-"*65)
            for approach, data_dict in results.items():
                if data_dict['errors']:
                    errs = data_dict['errors'][-1]
                    print(f"{approach:<15} {errs.get('revenue', 0):>10.2f}% {errs.get('net_income', 0):>10.2f}% "
                          f"{errs.get('total_assets', 0):>10.2f}% {errs.get('total_equity', 0):>10.2f}%")
        
        # Save PDF for this round
        if save_pdf and output_folder:
            pdf_path = os.path.join(output_folder, f"ensemble_round_{round_num+1:02d}_period_{t+2}.pdf")
            try:
                generate_ensemble_round_pdf(
                    round_num=round_num+1,
                    period=t+2,
                    results=results,
                    actual=actual,
                    ticker=ticker,
                    output_path=pdf_path
                )
                if verbose:
                    print(f"  → PDF: {pdf_path}")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not generate PDF: {e}")
    
    # Calculate summary statistics
    summary = calculate_validation_summary(results)
    
    # Print summary
    if verbose:
        print_ensemble_summary(summary, output_folder)
    
    # Generate final comparison PDF
    if save_pdf and output_folder:
        final_pdf = os.path.join(output_folder, f"{ticker}_ensemble_summary.pdf")
        try:
            generate_ensemble_summary_pdf(summary, ticker, final_pdf, n_test_periods)
            if verbose:
                print(f"\n📊 Summary PDF: {final_pdf}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not generate summary PDF: {e}")
    
    return {
        'results': results,
        'summary': summary,
        'n_rounds': n_test_periods,
        'output_folder': output_folder
    }


def calculate_errors(pred: CompleteFinancialStatements, actual: pd.Series) -> Dict[str, float]:
    """Calculate percentage errors for key metrics."""
    errors = {}
    key_items = [
        ('revenue', ['revenue', 'sales_revenue', 'totalRevenue']),
        ('net_income', ['net_income', 'netIncome']),
        ('total_assets', ['total_assets', 'totalAssets']),
        ('total_equity', ['total_equity', 'totalEquity', 'totalStockholdersEquity']),
        ('cash', ['cash', 'cashAndCashEquivalents']),
    ]
    
    for key, possible_names in key_items:
        pred_val = getattr(pred, key, 0)
        actual_val = None
        for name in possible_names:
            if name in actual.index:
                val = actual[name]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    actual_val = val
                    break
        if actual_val and actual_val != 0:
            errors[key] = abs(pred_val - actual_val) / abs(actual_val) * 100
    
    return errors


def calculate_validation_summary(results: Dict) -> Dict:
    """Calculate summary statistics for all approaches."""
    summary = {}
    
    for approach, data in results.items():
        all_errors = {}
        for err_dict in data['errors']:
            for key, val in err_dict.items():
                if key not in all_errors:
                    all_errors[key] = []
                all_errors[key].append(val)
        
        # Calculate stats
        stats = {}
        overall_mapes = []
        for var, errors in all_errors.items():
            if errors:
                stats[var] = {
                    'mean': np.mean(errors),
                    'std': np.std(errors),
                    'min': np.min(errors),
                    'max': np.max(errors),
                }
                overall_mapes.append(np.mean(errors))
        
        summary[approach] = {
            'stats': stats,
            'overall_mape': np.mean(overall_mapes) if overall_mapes else float('inf'),
            'n_rounds': len(data['errors']),
        }
    
    # Determine best approach
    best_approach = min(summary, key=lambda x: summary[x]['overall_mape'])
    summary['_best'] = best_approach
    summary['_best_mape'] = summary[best_approach]['overall_mape']
    
    return summary


def print_ensemble_summary(summary: Dict, output_folder: Optional[str] = None):
    """Print validation summary."""
    
    print("\n" + "="*80)
    print("ENSEMBLE VALIDATION SUMMARY")
    print("="*80)
    
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
    
    # Best approach
    print("\n" + "="*80)
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
    
    # Grade
    if best_mape < 10:
        grade = "⭐⭐⭐⭐⭐ Excellent"
    elif best_mape < 15:
        grade = "⭐⭐⭐⭐ Very Good"
    elif best_mape < 20:
        grade = "⭐⭐⭐ Good"
    else:
        grade = "⭐⭐ Fair"
    print(f"Grade: {grade}")
    
    if output_folder:
        print(f"\n📁 Full statements saved to: {output_folder}")


def generate_ensemble_round_pdf(round_num: int, period: int, results: Dict,
                                actual: pd.Series, ticker: str, output_path: str):
    """Generate PDF for a single validation round with all approaches."""
    if not HAS_REPORTLAB:
        return
    
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                           topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=14, spaceAfter=10)
    header_style = ParagraphStyle('Header', parent=styles['Heading2'], fontSize=11,
                                  textColor=colors.darkblue, spaceAfter=8)
    
    # Title
    story.append(Paragraph(f"{ticker} Ensemble Validation - Round {round_num}, Period {period}", title_style))
    story.append(Spacer(1, 10))
    
    # Approach comparison summary
    story.append(Paragraph("APPROACH COMPARISON", header_style))
    
    comp_header = ['Approach', 'Revenue Error', 'Net Income Error', 'Assets Error', 'Equity Error', 'Avg MAPE']
    comp_data = [comp_header]
    
    for approach in ['ML Only', 'ML + LLM', 'Pure LLM']:
        if approach not in results or not results[approach]['errors']:
            continue
        errs = results[approach]['errors'][-1]  # Latest round
        avg = np.mean([v for v in errs.values()])
        comp_data.append([
            approach,
            f"{errs.get('revenue', 0):.2f}%",
            f"{errs.get('net_income', 0):.2f}%",
            f"{errs.get('total_assets', 0):.2f}%",
            f"{errs.get('total_equity', 0):.2f}%",
            f"{avg:.2f}%"
        ])
    
    comp_table = Table(comp_data, colWidths=[1.2*inch, 1*inch, 1.1*inch, 0.9*inch, 0.9*inch, 0.9*inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.3, 0.5)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 15))
    
    # For each approach, show full statements
    for approach in ['ML Only', 'ML + LLM']:
        if approach not in results or not results[approach]['predictions']:
            continue
        
        pred = results[approach]['predictions'][-1]
        
        story.append(Paragraph(f"{approach.upper()} - FULL STATEMENTS", header_style))
        
        # Income Statement
        is_data = [['Item', 'Predicted', 'Actual', 'Error']]
        is_items = [
            ('Revenue', pred.revenue, ['revenue', 'totalRevenue']),
            ('COGS', pred.cogs, ['cogs', 'costOfRevenue']),
            ('Gross Profit', pred.gross_profit, ['grossProfit']),
            ('OpEx', pred.opex, ['operatingExpenses']),
            ('EBITDA', pred.ebitda, ['ebitda']),
            ('Net Income', pred.net_income, ['netIncome']),
        ]
        for name, pred_val, mappings in is_items:
            actual_val = get_actual_value(actual, mappings)
            is_data.append([
                name,
                fmt_currency(pred_val),
                fmt_currency(actual_val) if actual_val else "N/A",
                calc_error(pred_val, actual_val) if actual_val else "N/A"
            ])
        
        is_table = Table(is_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 0.8*inch])
        is_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(is_table)
        story.append(Spacer(1, 8))
        
        # Balance Sheet
        bs_data = [['Item', 'Predicted', 'Actual', 'Error']]
        bs_items = [
            ('Cash', pred.cash, ['cashAndCashEquivalents']),
            ('Total Current Assets', pred.total_current_assets, ['totalCurrentAssets']),
            ('Total Assets', pred.total_assets, ['totalAssets']),
            ('Total Liabilities', pred.total_liabilities, ['totalLiabilities']),
            ('Total Equity', pred.total_equity, ['totalEquity', 'totalStockholdersEquity']),
        ]
        for name, pred_val, mappings in bs_items:
            actual_val = get_actual_value(actual, mappings)
            bs_data.append([
                name,
                fmt_currency(pred_val),
                fmt_currency(actual_val) if actual_val else "N/A",
                calc_error(pred_val, actual_val) if actual_val else "N/A"
            ])
        
        bs_table = Table(bs_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 0.8*inch])
        bs_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(bs_table)
        story.append(Spacer(1, 8))
        
        # Accounting identity
        a_eq = abs(pred.total_assets - pred.total_liabilities - pred.total_equity) < 1
        identity_style = ParagraphStyle('Identity', fontSize=8, 
                                        textColor=colors.green if a_eq else colors.red)
        story.append(Paragraph(
            f"A = L + E: {fmt_currency(pred.total_assets)} = {fmt_currency(pred.total_liabilities)} + {fmt_currency(pred.total_equity)} {'✓' if a_eq else '✗'}",
            identity_style
        ))
        story.append(Spacer(1, 10))
    
    doc.build(story)


def generate_ensemble_summary_pdf(summary: Dict, ticker: str, output_path: str, n_rounds: int):
    """Generate final summary PDF with all approach comparisons."""
    if not HAS_REPORTLAB:
        return
    
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                           topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=18, 
                                 spaceAfter=10, textColor=colors.darkblue)
    story.append(Paragraph(f"{ticker} - Ensemble Validation Summary", title_style))
    
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=10,
                                    textColor=colors.grey, spaceAfter=20)
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Rounds: {n_rounds}", subtitle_style))
    
    # Executive Summary
    header_style = ParagraphStyle('Header', parent=styles['Heading2'], fontSize=14,
                                  textColor=colors.darkblue, spaceAfter=10)
    story.append(Paragraph("EXECUTIVE SUMMARY", header_style))
    
    best = summary.get('_best', 'N/A')
    best_mape = summary.get('_best_mape', 0)
    
    if best_mape < 10:
        grade = "Excellent"
        grade_color = colors.green
    elif best_mape < 15:
        grade = "Very Good"
        grade_color = colors.Color(0.4, 0.7, 0.2)
    elif best_mape < 20:
        grade = "Good"
        grade_color = colors.orange
    else:
        grade = "Fair"
        grade_color = colors.red
    
    exec_data = [
        ['Metric', 'Value'],
        ['Best Approach', best],
        ['Best MAPE', f"{best_mape:.2f}%"],
        ['Grade', grade],
        ['Validation Rounds', str(n_rounds)],
    ]
    
    exec_table = Table(exec_data, colWidths=[2*inch, 3*inch])
    exec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.3, 0.5)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
    ]))
    story.append(exec_table)
    story.append(Spacer(1, 20))
    
    # Approach Comparison
    story.append(Paragraph("APPROACH COMPARISON", header_style))
    
    comp_header = ['Approach', 'Revenue', 'Net Income', 'Total Assets', 'Total Equity', 'Overall MAPE']
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
    
    # Highlight best
    for i, row in enumerate(comp_data[1:], 1):
        if row[0] == best:
            comp_table.setStyle(TableStyle([
                ('BACKGROUND', (0, i), (-1, i), colors.Color(0.9, 1.0, 0.9)),
                ('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold'),
            ]))
    
    story.append(comp_table)
    story.append(Spacer(1, 20))
    
    # Detailed stats for each approach
    story.append(Paragraph("DETAILED STATISTICS BY APPROACH", header_style))
    
    for approach in ['ML Only', 'ML + LLM', 'Pure LLM']:
        if approach not in summary:
            continue
        
        story.append(Paragraph(f"<b>{approach}</b>", styles['Normal']))
        
        detail_header = ['Variable', 'Mean MAPE', 'Std', 'Min', 'Max']
        detail_data = [detail_header]
        
        for var, stats in summary[approach]['stats'].items():
            detail_data.append([
                var,
                f"{stats['mean']:.2f}%",
                f"{stats['std']:.2f}%",
                f"{stats['min']:.2f}%",
                f"{stats['max']:.2f}%"
            ])
        
        # Add overall
        detail_data.append([
            'OVERALL',
            f"{summary[approach]['overall_mape']:.2f}%",
            '-', '-', '-'
        ])
        
        detail_table = Table(detail_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        detail_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(detail_table)
        story.append(Spacer(1, 10))
    
    # Methodology
    story.append(Spacer(1, 10))
    story.append(Paragraph("METHODOLOGY", header_style))
    
    method_style = ParagraphStyle('Method', parent=styles['Normal'], fontSize=9, spaceAfter=6)
    
    methods = [
        ("<b>ML Only:</b> XGBoost Quantile Regression predicting 5 key drivers (revenue growth, COGS margin, OpEx margin, CapEx ratio, net margin). Complete statements derived via Accounting Engine."),
        ("<b>ML + LLM:</b> Hybrid approach using ML for growth prediction and LLM for margin adjustments based on industry knowledge."),
        ("<b>Pure LLM:</b> LLM provides all ratio predictions based on historical patterns and domain expertise."),
    ]
    
    for m in methods:
        story.append(Paragraph(m, method_style))
    
    # Footer
    story.append(Spacer(1, 30))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8,
                                  textColor=colors.grey, alignment=1)
    story.append(Paragraph("Generated by Balance Sheet Simulator - JP Morgan MLCOE 2026", footer_style))
    
    doc.build(story)
