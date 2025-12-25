#!/usr/bin/env python3
"""
accounting_engine.py - Part 1 + Part 2 Compatible Version

Supports both:
- Part 1: Historical ratio calculation (default behavior)
- Part 2: LLM assumption override via set_assumptions()

Fixes based on actual FMP API data:
1. Retention Ratio: Use 1 - (dividends + buybacks) / net_income
2. Shares Outstanding: Use weightedAverageShsOut (already works)
3. Interest Rate: Handle companies with 0 interest expense (like Apple)

FMP Column Names (verified):
- commonDividendsPaid (Cash Flow)
- commonStockRepurchased (Cash Flow)  
- weightedAverageShsOut (Income Statement)
- interestExpense (Income Statement) - may be 0!
- totalDebt (Balance Sheet)
- retainedEarnings (Balance Sheet)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HistoricalRatios:
    """Historical financial ratios and actual values."""
    gross_margin: float
    overhead_to_revenue: float
    payroll_to_revenue: float
    tax_rate: float
    interest_coverage: float
    cash_to_revenue: float
    ar_days: float
    inventory_days: float
    ap_days: float
    asset_turnover: float
    debt_to_equity: float
    current_ratio: float
    ocf_to_ni: float
    capex_to_revenue: float
    dividend_payout: float
    # Use totals directly
    avg_total_assets: float = 0.0
    avg_total_liabilities: float = 0.0
    avg_total_equity: float = 0.0
    avg_common_stock: float = 0.0
    avg_revenue: float = 0.0
    avg_depreciation_rate: float = 0.025
    # Margin fields
    avg_net_income_margin: float = 0.20
    avg_ebit_margin: float = 0.25
    retention_ratio: float = 0.30
    avg_interest_rate: float = 0.02
    accrued_to_revenue: float = 0.05
    last_shares_outstanding: float = 1_000_000_000


class AccountingEngine:
    """
    Accounting Engine - Part 1 + Part 2 Compatible
    
    Builds complete financial statements from ML predictions.
    
    Part 1 Usage (default - historical ratios):
        engine = AccountingEngine(historical_data)
        statements = engine.build_complete_statements(predictions)
    
    Part 2 Usage (LLM assumptions):
        engine = AccountingEngine(historical_data)
        engine.set_assumptions(llm_assumptions)  # Override ratios
        statements = engine.build_complete_statements(predictions)
    """
    
    def __init__(self, historical_data: pd.DataFrame):
        """Initialize with historical data."""
        self.historical_data = historical_data
        self.ratios = self._calculate_historical_ratios()
    
    # ================================================================
    # PART 2: LLM ASSUMPTION OVERRIDE
    # ================================================================
    
    def set_assumptions(self, assumptions: Dict) -> None:
        """
        Override historical ratios with external assumptions (e.g., from LLM).
        
        This method enables Part 2 to inject LLM-generated assumptions
        without modifying the core accounting logic.
        
        Args:
            assumptions: Dictionary with keys:
                - gross_margin: float (e.g., 0.46)
                - avg_net_income_margin: float (or net_income_margin)
                - avg_ebit_margin: float (or ebit_margin)
                - capex_to_revenue: float (or capex_ratio)
                - retention_ratio: float
                - reasoning: str (optional, for logging)
        
        Example:
            engine.set_assumptions({
                "gross_margin": 0.46,
                "avg_net_income_margin": 0.26,
                "avg_ebit_margin": 0.31,
                "capex_to_revenue": 0.03,
                "retention_ratio": 0.0,
                "reasoning": "Based on LLM analysis..."
            })
        """
        # Map of assumption keys to ratio attributes (with aliases)
        key_mapping = {
            'gross_margin': 'gross_margin',
            'avg_net_income_margin': 'avg_net_income_margin',
            'net_income_margin': 'avg_net_income_margin',  # alias
            'avg_ebit_margin': 'avg_ebit_margin',
            'ebit_margin': 'avg_ebit_margin',  # alias
            'capex_to_revenue': 'capex_to_revenue',
            'capex_ratio': 'capex_to_revenue',  # alias
            'retention_ratio': 'retention_ratio',
            'avg_interest_rate': 'avg_interest_rate',
            'interest_rate': 'avg_interest_rate',  # alias
            'tax_rate': 'tax_rate',
        }
        
        print("\n" + "=" * 60)
        print("APPLYING LLM ASSUMPTIONS (Part 2)")
        print("=" * 60)
        
        for key, value in assumptions.items():
            if key in key_mapping and isinstance(value, (int, float)):
                attr_name = key_mapping[key]
                old_value = getattr(self.ratios, attr_name, None)
                setattr(self.ratios, attr_name, float(value))
                if old_value is not None:
                    print(f"  {attr_name}: {old_value:.2%} -> {value:.2%}")
        
        if 'reasoning' in assumptions:
            reasoning = assumptions['reasoning']
            if len(reasoning) > 100:
                reasoning = reasoning[:100] + "..."
            print(f"\n  Reasoning: {reasoning}")
        
        print("=" * 60)
    
    # ================================================================
    # PART 1: HISTORICAL RATIO CALCULATION (unchanged)
    # ================================================================
    
    def _calculate_historical_ratios(self) -> HistoricalRatios:
        """Calculate all ratios from historical data."""
        df = self.historical_data
        recent = df.tail(8)
        
        # Basic calculations
        revenue = recent['sales_revenue'].mean() if 'sales_revenue' in recent.columns else 0
        
        if 'cost_of_goods_sold' in recent.columns and revenue > 0:
            gross_margin = (revenue - recent['cost_of_goods_sold'].mean()) / revenue
        else:
            gross_margin = 0.35
        
        overhead_to_revenue = recent['overhead_expenses'].mean() / revenue if revenue > 0 and 'overhead_expenses' in recent.columns else 0.15
        payroll_to_revenue = recent['payroll_expenses'].mean() / revenue if revenue > 0 and 'payroll_expenses' in recent.columns else 0.10
        
        # Tax rate
        if 'ebt' in recent.columns and 'net_income' in recent.columns:
            ebt = recent['ebt'].mean()
            ni = recent['net_income'].mean()
            if ebt > 0:
                tax_rate = (ebt - ni) / ebt
                tax_rate = max(0.0, min(tax_rate, 0.40))
            else:
                tax_rate = 0.21
        else:
            tax_rate = 0.21
        
        # Working capital ratios
        cash_to_revenue = recent['cash'].mean() / revenue if revenue > 0 and 'cash' in recent.columns else 0.10
        
        ar_days = 45
        if 'accounts_receivable' in recent.columns and revenue > 0:
            ar_days = (recent['accounts_receivable'].mean() / revenue) * 90
        
        inventory_days = 30
        if 'inventory' in recent.columns and 'cost_of_goods_sold' in recent.columns:
            cogs = recent['cost_of_goods_sold'].mean()
            if cogs > 0:
                inventory_days = (recent['inventory'].mean() / cogs) * 90
        
        ap_days = 45
        if 'accounts_payable' in recent.columns and 'cost_of_goods_sold' in recent.columns:
            cogs = recent['cost_of_goods_sold'].mean()
            if cogs > 0:
                ap_days = (recent['accounts_payable'].mean() / cogs) * 90
        
        # Asset/liability ratios
        avg_total_assets = recent['total_assets'].mean() if 'total_assets' in recent.columns else 0
        avg_total_liabilities = recent['total_liabilities'].mean() if 'total_liabilities' in recent.columns else 0
        avg_total_equity = recent['total_equity'].mean() if 'total_equity' in recent.columns else 0
        
        asset_turnover = (revenue * 4) / avg_total_assets if avg_total_assets > 0 else 0.5
        debt_to_equity = avg_total_liabilities / avg_total_equity if avg_total_equity > 0 else 1.0
        current_ratio = 1.5
        
        # Common stock
        if 'common_stock' in recent.columns:
            avg_common_stock = recent['common_stock'].mean()
        else:
            avg_common_stock = avg_total_equity * 0.5
        
        # Depreciation rate
        if 'depreciation' in recent.columns and revenue > 0:
            avg_depreciation_rate = recent['depreciation'].mean() / revenue
        else:
            avg_depreciation_rate = 0.025
        
        # OCF to NI
        if 'operating_cash_flow' in recent.columns and 'net_income' in recent.columns:
            ni = recent['net_income'].mean()
            if ni > 0:
                ocf_to_ni = recent['operating_cash_flow'].mean() / ni
            else:
                ocf_to_ni = 1.3
        else:
            ocf_to_ni = 1.3
        
        capex_to_revenue = recent['capex'].mean() / revenue if revenue > 0 and 'capex' in recent.columns else 0.05
        dividend_payout = 0.0
        interest_coverage = 10.0
        
        # ================================================================
        # MARGINS (direct calculation)
        # ================================================================
        if 'net_income' in recent.columns:
            avg_net_income = recent['net_income'].mean()
            avg_net_income_margin = avg_net_income / revenue if revenue > 0 else 0.20
        else:
            avg_net_income_margin = 0.20
        
        if 'ebit' in recent.columns:
            avg_ebit = recent['ebit'].mean()
            avg_ebit_margin = avg_ebit / revenue if revenue > 0 else 0.25
        else:
            avg_ebit_margin = 0.25
        
        # ================================================================
        # FIX 1: RETENTION RATIO
        # Formula: 1 - (Dividends + Buybacks) / Net Income
        # ================================================================
        retention_ratio = 0.30  # default
        
        total_net_income = df['net_income'].tail(8).sum() if 'net_income' in df.columns else 0
        
        if total_net_income > 0:
            # Get dividends paid
            total_dividends = 0
            if 'dividends_paid' in df.columns:
                total_dividends = df['dividends_paid'].tail(8).sum()
            
            # Get stock repurchases (buybacks)
            total_buybacks = 0
            if 'stock_repurchased' in df.columns:
                total_buybacks = df['stock_repurchased'].tail(8).sum()
            
            # Total payout
            total_payout = total_dividends + total_buybacks
            payout_ratio = total_payout / total_net_income
            
            # Retention = 1 - Payout Ratio
            retention_ratio = 1 - payout_ratio
            
            # Can be negative for companies that pay out more than they earn
            # Clip to reasonable range
            retention_ratio = max(-1.0, min(retention_ratio, 1.0))
            
            print(f"\n  Retention Ratio Calculation:")
            print(f"    Net Income (8Q): ${total_net_income/1e9:.2f}B")
            print(f"    Dividends (8Q): ${total_dividends/1e9:.2f}B")
            print(f"    Buybacks (8Q): ${total_buybacks/1e9:.2f}B")
            print(f"    Payout Ratio: {payout_ratio*100:.1f}%")
            print(f"    Retention Ratio: {retention_ratio*100:.1f}%")
        else:
            # Fallback to retained earnings method
            if 'retained_earnings' in df.columns and len(df) >= 8:
                re_start = df['retained_earnings'].iloc[-8]
                re_end = df['retained_earnings'].iloc[-1]
                re_change = re_end - re_start
                
                if total_net_income != 0:
                    retention_ratio = re_change / abs(total_net_income)
                    retention_ratio = max(-1.0, min(retention_ratio, 1.0))
                    print(f"\n  Retention (from RE change): {retention_ratio*100:.1f}%")
        
        # ================================================================
        # FIX 2: INTEREST RATE
        # Some companies (like Apple) have 0 or minimal interest expense
        # ================================================================
        avg_interest_rate = 0.02  # default
        
        if 'interest_expense' in recent.columns:
            avg_interest = recent['interest_expense'].mean()
            
            # Get debt
            avg_debt = 0
            if 'total_debt' in recent.columns:
                avg_debt = recent['total_debt'].mean()
            elif 'total_liabilities' in recent.columns:
                avg_debt = recent['total_liabilities'].mean() * 0.4  # estimate
            
            if avg_debt > 0 and avg_interest > 0:
                # Annualize quarterly interest
                avg_interest_rate = (avg_interest * 4) / avg_debt
                avg_interest_rate = max(0.001, min(avg_interest_rate, 0.15))
                print(f"    Interest Rate: {avg_interest_rate*100:.2f}% (calculated)")
            elif avg_interest == 0:
                # Company has no net interest expense (like Apple)
                # Use a minimal rate
                avg_interest_rate = 0.005  # 0.5%
                print(f"    Interest Rate: {avg_interest_rate*100:.2f}% (minimal - company has no net interest expense)")
            else:
                print(f"    Interest Rate: {avg_interest_rate*100:.2f}% (fallback)")
        
        # ================================================================
        # FIX 3: SHARES OUTSTANDING
        # ================================================================
        last_shares_outstanding = 1_000_000_000  # default 1B
        
        if 'shares_outstanding' in df.columns:
            last_shares_outstanding = df['shares_outstanding'].iloc[-1]
            if last_shares_outstanding <= 0:
                last_shares_outstanding = 1_000_000_000
        
        # Accrued expenses
        accrued_to_revenue = 0.05
        if 'accrued_expenses' in recent.columns:
            accrued_to_revenue = recent['accrued_expenses'].mean() / revenue if revenue > 0 else 0.05
        
        # Final output
        print(f"\n  Historical Ratios (Part 1):")
        print(f"    Gross Margin: {gross_margin*100:.1f}%")
        print(f"    EBIT Margin: {avg_ebit_margin*100:.1f}%")
        print(f"    Net Income Margin: {avg_net_income_margin*100:.1f}%")
        print(f"    Retention Ratio: {retention_ratio*100:.1f}%")
        print(f"    Interest Rate: {avg_interest_rate*100:.2f}%")
        print(f"    Shares Outstanding: {last_shares_outstanding/1e9:.2f}B")
        
        return HistoricalRatios(
            gross_margin=gross_margin,
            overhead_to_revenue=overhead_to_revenue,
            payroll_to_revenue=payroll_to_revenue,
            tax_rate=tax_rate,
            interest_coverage=interest_coverage,
            cash_to_revenue=cash_to_revenue,
            ar_days=ar_days,
            inventory_days=inventory_days,
            ap_days=ap_days,
            asset_turnover=asset_turnover,
            debt_to_equity=debt_to_equity,
            current_ratio=current_ratio,
            ocf_to_ni=ocf_to_ni,
            capex_to_revenue=capex_to_revenue,
            dividend_payout=dividend_payout,
            avg_total_assets=avg_total_assets,
            avg_total_liabilities=avg_total_liabilities,
            avg_total_equity=avg_total_equity,
            avg_common_stock=avg_common_stock,
            avg_revenue=revenue,
            avg_depreciation_rate=avg_depreciation_rate,
            avg_net_income_margin=avg_net_income_margin,
            avg_ebit_margin=avg_ebit_margin,
            retention_ratio=retention_ratio,
            avg_interest_rate=avg_interest_rate,
            accrued_to_revenue=accrued_to_revenue,
            last_shares_outstanding=last_shares_outstanding
        )
    
    # ================================================================
    # BUILD COMPLETE STATEMENTS (unchanged)
    # ================================================================
    
    def build_complete_statements(
        self,
        predictions: Dict[str, np.ndarray],
        periods: int = 8
    ) -> pd.DataFrame:
        """Build complete financial statements from ML predictions."""
        
        statements = []
        
        # Get last historical period for rolling forward
        last_hist = self.historical_data.iloc[-1].copy()
        
        for period in range(periods):
            # Get ML predictions for this period
            revenue = predictions['sales_revenue'][period] if period < len(predictions.get('sales_revenue', [])) else last_hist.get('sales_revenue', 0)
            cogs = predictions['cost_of_goods_sold'][period] if period < len(predictions.get('cost_of_goods_sold', [])) else last_hist.get('cost_of_goods_sold', 0)
            overhead = predictions['overhead_expenses'][period] if period < len(predictions.get('overhead_expenses', [])) else last_hist.get('overhead_expenses', 0)
            payroll = predictions['payroll_expenses'][period] if period < len(predictions.get('payroll_expenses', [])) else last_hist.get('payroll_expenses', 0)
            capex = predictions['capex'][period] if period < len(predictions.get('capex', [])) else last_hist.get('capex', 0)
            
            # Build Income Statement
            income_stmt = self._build_income_statement(revenue, cogs, overhead, payroll, last_hist)
            
            # Build Balance Sheet
            balance_sheet = self._build_balance_sheet(revenue, capex, income_stmt, last_hist, period)
            
            # Build Cash Flow Statement
            cash_flow = self._build_cash_flow(income_stmt, balance_sheet, capex, last_hist)
            
            # Combine all statements
            period_data = {
                'period': period + 1,
                **income_stmt,
                **balance_sheet,
                **cash_flow
            }
            
            statements.append(period_data)
            last_hist = pd.Series(period_data)
        
        return pd.DataFrame(statements)
    
    def _build_income_statement(
        self,
        revenue: float,
        cogs: float,
        overhead: float,
        payroll: float,
        last_period: pd.Series
    ) -> Dict[str, float]:
        """Build income statement using historical ratios."""
        
        # Gross Profit
        adjusted_cogs = revenue * (1 - self.ratios.gross_margin)
        gross_profit = revenue - adjusted_cogs
        
        # Operating expenses
        adjusted_overhead = revenue * self.ratios.overhead_to_revenue
        adjusted_payroll = revenue * self.ratios.payroll_to_revenue
        operating_expenses = adjusted_overhead + adjusted_payroll
        
        # EBITDA
        ebitda = gross_profit - operating_expenses
        
        # Depreciation
        depreciation = revenue * self.ratios.avg_depreciation_rate
        
        # EBIT - USE DIRECT MARGIN
        ebit = revenue * self.ratios.avg_ebit_margin
        
        # Interest Expense
        if 'bs_total_liabilities' in last_period and last_period['bs_total_liabilities'] > 0:
            interest_expense = last_period['bs_total_liabilities'] * self.ratios.avg_interest_rate / 4
        else:
            interest_expense = abs(ebit) * 0.01  # minimal
        
        interest_expense = min(interest_expense, abs(ebit) * 0.20)
        ebt = ebit - interest_expense
        
        # NET INCOME - USE DIRECT MARGIN
        net_income = revenue * self.ratios.avg_net_income_margin
        
        # Back-calculate tax
        if ebt > 0:
            tax_expense = ebt - net_income
            effective_tax_rate = tax_expense / ebt if ebt > 0 else 0
        else:
            tax_expense = 0
            effective_tax_rate = 0
        
        net_margin = net_income / revenue if revenue > 0 else 0
        
        # Shares
        if 'shares_outstanding' in last_period and last_period['shares_outstanding'] > 0:
            shares = last_period['shares_outstanding']
        else:
            shares = self.ratios.last_shares_outstanding
        
        eps = net_income / shares if shares > 0 else 0
        
        return {
            'is_revenue': revenue,
            'is_cogs': adjusted_cogs,
            'is_gross_profit': gross_profit,
            'is_gross_margin': self.ratios.gross_margin,
            'is_operating_expenses': operating_expenses,
            'is_ebitda': ebitda,
            'is_depreciation': depreciation,
            'is_ebit': ebit,
            'is_interest_expense': interest_expense,
            'is_ebt': ebt,
            'is_tax_expense': tax_expense,
            'is_net_income': net_income,
            'is_net_margin': net_margin,
            'is_eps': eps,
            'shares_outstanding': shares
        }
    
    def _build_balance_sheet(
        self,
        revenue: float,
        capex: float,
        income_stmt: Dict[str, float],
        last_period: pd.Series,
        period_num: int
    ) -> Dict[str, float]:
        """Build balance sheet using historical ratios."""
        
        # For companies with negative retention (like Apple with buybacks),
        # assets and equity will shrink over time. This is realistic.
        
        # Total Assets
        if 'bs_total_assets' in last_period and last_period['bs_total_assets'] > 0:
            prior_assets = last_period['bs_total_assets']
            # Assets change by retained earnings (can be negative)
            asset_change = income_stmt['is_net_income'] * max(self.ratios.retention_ratio, 0)
            total_assets = prior_assets + asset_change
        else:
            total_assets = self.ratios.avg_total_assets
        
        # Current assets
        cash = revenue * self.ratios.cash_to_revenue
        accounts_receivable = (revenue / 90) * self.ratios.ar_days
        cogs = income_stmt['is_cogs']
        inventory = (cogs / 90) * self.ratios.inventory_days
        current_assets = cash + accounts_receivable + inventory
        
        # Non-current
        non_current_assets = total_assets - current_assets
        ppe = non_current_assets * 0.7
        goodwill = non_current_assets * 0.3
        
        # Liabilities - relatively stable
        if 'bs_total_liabilities' in last_period and last_period['bs_total_liabilities'] > 0:
            total_liabilities = last_period['bs_total_liabilities']
        else:
            total_liabilities = self.ratios.avg_total_liabilities
        
        # Current liabilities
        accounts_payable = (cogs / 90) * self.ratios.ap_days
        accrued_expenses = revenue * self.ratios.accrued_to_revenue
        current_liabilities = accounts_payable + accrued_expenses
        total_debt = total_liabilities - current_liabilities
        
        # Equity - changes with retention
        if 'bs_total_equity' in last_period and last_period['bs_total_equity'] > 0:
            prior_equity = last_period['bs_total_equity']
            # For buyback companies, equity decreases
            equity_change = income_stmt['is_net_income'] * self.ratios.retention_ratio
            total_equity = prior_equity + equity_change
            # Don't let equity go negative
            total_equity = max(total_equity, prior_equity * 0.5)
        else:
            total_equity = total_assets - total_liabilities
        
        # Split equity
        if 'bs_common_stock' in last_period and last_period['bs_common_stock'] > 0:
            common_stock = last_period['bs_common_stock']
            retained_earnings = total_equity - common_stock
        else:
            common_stock = self.ratios.avg_common_stock
            retained_earnings = total_equity - common_stock
        
        return {
            'bs_cash': cash,
            'bs_accounts_receivable': accounts_receivable,
            'bs_inventory': inventory,
            'bs_current_assets': current_assets,
            'bs_ppe': ppe,
            'bs_goodwill': goodwill,
            'bs_total_assets': total_assets,
            'bs_accounts_payable': accounts_payable,
            'bs_accrued_expenses': accrued_expenses,
            'bs_current_liabilities': current_liabilities,
            'bs_total_debt': total_debt,
            'bs_total_liabilities': total_liabilities,
            'bs_common_stock': common_stock,
            'bs_retained_earnings': retained_earnings,
            'bs_total_equity': total_equity
        }
    
    def _build_cash_flow(
        self,
        income_stmt: Dict[str, float],
        balance_sheet: Dict[str, float],
        capex: float,
        last_period: pd.Series
    ) -> Dict[str, float]:
        """Build cash flow statement."""
        
        net_income = income_stmt['is_net_income']
        depreciation = income_stmt['is_depreciation']
        
        # Working capital changes
        delta_ar = balance_sheet['bs_accounts_receivable'] - last_period.get('bs_accounts_receivable', balance_sheet['bs_accounts_receivable'])
        delta_inv = balance_sheet['bs_inventory'] - last_period.get('bs_inventory', balance_sheet['bs_inventory'])
        delta_ap = balance_sheet['bs_accounts_payable'] - last_period.get('bs_accounts_payable', balance_sheet['bs_accounts_payable'])
        
        operating_cf = net_income + depreciation - delta_ar - delta_inv + delta_ap
        investing_cf = -abs(capex)
        
        # Financing
        delta_debt = balance_sheet['bs_total_debt'] - last_period.get('bs_total_debt', balance_sheet['bs_total_debt'])
        financing_cf = delta_debt
        
        net_change = operating_cf + investing_cf + financing_cf
        free_cash_flow = operating_cf - abs(capex)
        
        return {
            'cf_operating': operating_cf,
            'cf_investing': investing_cf,
            'cf_financing': financing_cf,
            'cf_net_change': net_change,
            'cf_free_cash_flow': free_cash_flow,
            'cf_capex': -abs(capex)
        }
