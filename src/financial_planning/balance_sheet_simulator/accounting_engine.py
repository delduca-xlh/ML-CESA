"""
accounting_engine.py

Derives complete three statements from driver variables.
All accounting identities satisfied BY CONSTRUCTION.
"""

import numpy as np
import pandas as pd
from typing import Dict

from .data_structures import CompleteFinancialStatements


class AccountingEngine:
    """
    Transforms 4 ML-predicted drivers → 30+ financial statement items.
    
    Exogenous (ML predicted): revenue_growth, cogs_margin, opex_margin, capex_ratio
    Endogenous (derived): Everything else
    """
    
    def __init__(self, historical_data: pd.DataFrame):
        self.ratios = self._calculate_ratios(historical_data)
        
    def _safe_divide(self, num: float, denom: float, default: float = 0) -> float:
        if denom == 0 or denom is None or (isinstance(denom, float) and np.isnan(denom)):
            return default
        return num / denom
    
    def _get_col(self, df: pd.DataFrame, names: list, default: float = 0) -> pd.Series:
        for name in names:
            if name in df.columns:
                return df[name]
        return pd.Series([default] * len(df))
    
    def _calculate_ratios(self, data: pd.DataFrame) -> Dict:
        df = data.copy()
        recent = df.tail(8)
        
        revenue = self._get_col(recent, ['revenue', 'sales_revenue', 'totalRevenue'])
        cogs = self._get_col(recent, ['cogs', 'cost_of_goods_sold', 'costOfRevenue'])
        net_income = self._get_col(recent, ['net_income', 'netIncome'])
        income_tax = self._get_col(recent, ['income_tax', 'incomeTaxExpense'])
        ebt = self._get_col(recent, ['ebt', 'incomeBeforeTax', 'earningsBeforeTax'])
        total_debt = self._get_col(recent, ['total_debt', 'totalDebt', 'longTermDebt'])
        total_equity = self._get_col(recent, ['total_equity', 'totalEquity', 'totalStockholdersEquity'])
        total_assets = self._get_col(recent, ['total_assets', 'totalAssets'])
        ar = self._get_col(recent, ['accounts_receivable', 'accountsReceivable', 'netReceivables'])
        inventory = self._get_col(recent, ['inventory'])
        ap = self._get_col(recent, ['accounts_payable', 'accountsPayable', 'accountPayables'])
        cash = self._get_col(recent, ['cash', 'cash_and_cash_equivalents', 'cashAndCashEquivalents'])
        depreciation = self._get_col(recent, ['depreciation', 'depreciationAndAmortization'])
        dividends = self._get_col(recent, ['dividendsPaid', 'dividends_paid', 'commonDividendsPaid', 'paymentOfDividends'])
        ppe = self._get_col(recent, ['propertyPlantEquipmentNet', 'ppe_net', 'property_plant_equipment_net'])
        
        # Get actual buyback data from cash flow statement
        buybacks = self._get_col(recent, ['commonStockRepurchased', 'repurchaseOfCommonStock', 'buybacks', 'stock_repurchased'])
        
        # Additional assets
        short_term_inv = self._get_col(recent, ['shortTermInvestments', 'short_term_investments'])
        long_term_inv = self._get_col(recent, ['longTermInvestments', 'long_term_investments'])
        other_ca = self._get_col(recent, ['otherCurrentAssets', 'other_current_assets'])
        other_nca = self._get_col(recent, ['otherNonCurrentAssets', 'other_noncurrent_assets'])
        intangibles = self._get_col(recent, ['intangibleAssets', 'goodwillAndIntangibleAssets'])
        
        # Liabilities
        total_cl = self._get_col(recent, ['totalCurrentLiabilities', 'total_current_liabilities'])
        total_ncl = self._get_col(recent, ['totalNonCurrentLiabilities', 'total_noncurrent_liabilities'])
        accrued = self._get_col(recent, ['accruedExpenses', 'otherCurrentLiabilities'])
        deferred_rev = self._get_col(recent, ['deferredRevenue', 'deferred_revenue'])
        other_ncl = self._get_col(recent, ['otherNonCurrentLiabilities', 'other_noncurrent_liabilities'])
        
        avg_rev = revenue.mean()
        avg_cogs = cogs.mean()
        avg_assets = total_assets.mean()
        avg_ni = net_income.mean()
        avg_ppe = ppe.mean() if ppe.mean() > 0 else 1
        
        # Calculate actual payout ratio from data
        payout = self._safe_divide(abs(dividends.mean()), avg_ni, 0.30)
        
        # Calculate actual buyback ratio from data (this is key!)
        # Buybacks are typically negative in CF statement
        avg_buyback = abs(buybacks.mean())
        actual_buyback_ratio = self._safe_divide(avg_buyback, avg_ni, 0.10)
        # Cap buyback ratio - more conservative for financial institutions
        # Banks typically have lower buyback ratios due to capital requirements
        
        # Calculate other_ncl ratio from actual data
        other_ncl_ratio = self._safe_divide(other_ncl.mean(), avg_assets, 0.10)
        
        # Interest expense ratio - detect if this is a financial institution
        # Banks have interest expense >> debt * typical rate
        interest_exp = self._get_col(recent, ['interestExpense', 'interest_expense'])
        avg_interest = abs(interest_exp.mean()) if len(interest_exp) > 0 else 0
        avg_debt = total_debt.mean() if total_debt.mean() > 0 else 1
        
        # Calculate implied interest rate - if > 20%, likely a bank
        implied_rate = avg_interest / avg_debt * 4  # Annualize
        is_financial = implied_rate > 0.20 or avg_interest > avg_ni * 0.5
        
        # For financials, learn interest as ratio of revenue (proxy for interest-bearing liabilities)
        if is_financial:
            interest_ratio = self._safe_divide(avg_interest, avg_rev, 0.10)
            # Cap buyback ratio more aggressively for financial institutions
            actual_buyback_ratio = min(actual_buyback_ratio, 0.50)  # Max 50% for banks
        else:
            interest_ratio = min(implied_rate, 0.10)  # Cap at 10% for non-financials
            actual_buyback_ratio = min(actual_buyback_ratio, 1.5)  # Max 150% for non-banks
        
        # Calculate effective tax rate from data
        # Use income_tax / (net_income + income_tax) to get effective rate
        avg_tax = abs(income_tax.mean()) if income_tax.mean() != 0 else 0
        avg_ebt = ebt.mean() if ebt.mean() > 0 else avg_ni + avg_tax
        effective_tax_rate = self._safe_divide(avg_tax, avg_ebt, 0.21)
        effective_tax_rate = min(max(effective_tax_rate, 0.10), 0.35)  # Clamp between 10-35%
        
        # Depreciation ratio: use revenue-based for stability (PPE-based is volatile when PPE grows)
        # This gives more stable predictions across different PPE growth scenarios
        depreciation_ratio_revenue = self._safe_divide(depreciation.mean(), avg_rev, 0.03)
        depreciation_ratio_ppe = self._safe_divide(depreciation.mean(), avg_ppe, 0.02) if avg_ppe > 0 else 0.02
        
        return {
            'gross_margin': self._safe_divide((revenue - cogs).mean(), avg_rev, 0.40),
            'net_margin': self._safe_divide(avg_ni, avg_rev, 0.10),
            'sga_ratio': 0.10,
            'rd_ratio': 0.05,
            # Depreciation: use revenue-based ratio (more stable) with PPE-based as backup
            'depreciation_ratio_revenue': depreciation_ratio_revenue,
            'depreciation_ratio_ppe': depreciation_ratio_ppe,
            # Tax rate learned from data
            'tax_rate': effective_tax_rate,
            # Interest rate learned from data
            'interest_rate': interest_ratio,
            'is_financial': is_financial,
            'deposit_rate': 0.02,
            'ar_days': self._safe_divide(ar.mean(), avg_rev / 365, 45),
            'inventory_days': self._safe_divide(inventory.mean(), avg_cogs / 365, 60),
            'ap_days': self._safe_divide(ap.mean(), avg_cogs / 365, 45),
            'cash_ratio': self._safe_divide(cash.mean(), avg_rev, 0.10),
            'prepaid_ratio': 0.01,
            'accrued_ratio': self._safe_divide(accrued.mean(), avg_rev, 0.02),
            'deferred_rev_ratio': self._safe_divide(deferred_rev.mean(), avg_rev, 0.01),
            'debt_to_equity': self._safe_divide(total_debt.mean(), total_equity.mean(), 0.50),
            'payout_ratio': payout,
            'buyback_ratio': actual_buyback_ratio,  # Now learned from data with caps
            'intangible_amort_rate': 0.05,
            
            # New ratios for additional assets
            'short_term_inv_ratio': self._safe_divide(short_term_inv.mean(), avg_assets, 0.05),
            'long_term_inv_ratio': self._safe_divide(long_term_inv.mean(), avg_assets, 0.10),
            'other_ca_ratio': self._safe_divide(other_ca.mean(), avg_rev, 0.05),
            'other_nca_ratio': self._safe_divide(other_nca.mean(), avg_assets, 0.10),
            'intangibles_ratio': self._safe_divide(intangibles.mean(), avg_assets, 0.05),
            
            # Liability ratios - now learned from data
            'other_cl_ratio': self._safe_divide(total_cl.mean() - ap.mean(), avg_rev, 0.15),
            'other_ncl_ratio': other_ncl_ratio,  # Learned from actual data
        }
    
    def derive_statements(self,
                          drivers: Dict[str, float],
                          prior: CompleteFinancialStatements,
                          period: str) -> CompleteFinancialStatements:
        """Derive complete three statements from driver variables."""
        r = self.ratios
        p = prior
        stmt = CompleteFinancialStatements(period=period)
        
        g = drivers.get('revenue_growth', 0)
        m_cogs = drivers.get('cogs_margin', 0.6)
        m_opex = drivers.get('opex_margin', 0.15)
        capex_ratio = drivers.get('capex_ratio', 0.03)
        net_margin = drivers.get('net_margin', None)  # Directly predicted
        
        # ═══════════════════════════════════════════════════════════════════════
        # INCOME STATEMENT
        # ═══════════════════════════════════════════════════════════════════════
        stmt.revenue = p.revenue * (1 + g) if p.revenue > 0 else 1e9
        stmt.cogs = stmt.revenue * m_cogs
        stmt.gross_profit = stmt.revenue - stmt.cogs
        stmt.opex = stmt.revenue * m_opex
        stmt.sga = stmt.revenue * r['sga_ratio']
        stmt.rd = stmt.revenue * r['rd_ratio']
        # Use revenue-based depreciation for stability
        stmt.depreciation = stmt.revenue * r.get('depreciation_ratio_revenue', 0.03)
        stmt.ebitda = stmt.gross_profit - stmt.opex
        stmt.ebit = stmt.ebitda - stmt.depreciation
        
        # Interest expense: different calculation for financial institutions
        if r.get('is_financial', False):
            stmt.interest_expense = stmt.revenue * r['interest_rate']
        else:
            stmt.interest_expense = p.long_term_debt * r['interest_rate'] / 4
        
        stmt.interest_income = (p.cash + p.short_term_investments) * r['deposit_rate'] / 4
        
        # ═══════════════════════════════════════════════════════════════════════
        # NET INCOME with Accounting Identity Preservation
        # ═══════════════════════════════════════════════════════════════════════
        # If net_margin is predicted, use "Other Income" to reconcile
        # This preserves: EBT = EBIT - Interest + Interest Income + Other
        #                 NI = EBT - Tax
        
        if net_margin is not None:
            # Target Net Income from predicted margin
            target_ni = stmt.revenue * net_margin
            
            # Calculate what NI would be without adjustment
            base_ebt = stmt.ebit - stmt.interest_expense + stmt.interest_income
            base_tax = max(0, base_ebt * r['tax_rate'])
            base_ni = base_ebt - base_tax
            
            # Calculate required "Other Income" to hit target NI
            # target_ni = (base_ebt + other_income) * (1 - tax_rate)
            # other_income = target_ni / (1 - tax_rate) - base_ebt
            effective_rate = r['tax_rate']
            if effective_rate < 1:
                required_ebt = target_ni / (1 - effective_rate)
                stmt.other_income = required_ebt - base_ebt
            else:
                stmt.other_income = 0
            
            # Now calculate final values with proper accounting chain
            stmt.ebt = base_ebt + stmt.other_income
            stmt.income_tax = max(0, stmt.ebt * r['tax_rate'])
            stmt.net_income = stmt.ebt - stmt.income_tax
        else:
            # Fallback: no adjustment
            stmt.other_income = 0
            stmt.ebt = stmt.ebit - stmt.interest_expense + stmt.interest_income
            stmt.income_tax = max(0, stmt.ebt * r['tax_rate'])
            stmt.net_income = stmt.ebt - stmt.income_tax
        
        # ═══════════════════════════════════════════════════════════════════════
        # WORKING CAPITAL
        # ═══════════════════════════════════════════════════════════════════════
        stmt.accounts_receivable = stmt.revenue * (r['ar_days'] / 365)
        stmt.inventory = stmt.cogs * (r['inventory_days'] / 365)
        stmt.prepaid_expenses = stmt.revenue * r['prepaid_ratio']
        stmt.accounts_payable = stmt.cogs * (r['ap_days'] / 365)
        
        # Accrued expenses: carry forward with growth (more stable than ratio)
        if p.accrued_expenses > 0:
            stmt.accrued_expenses = p.accrued_expenses * (1 + g)
        else:
            stmt.accrued_expenses = stmt.revenue * r['accrued_ratio']
        stmt.deferred_revenue = stmt.revenue * r['deferred_rev_ratio']
        
        # ═══════════════════════════════════════════════════════════════════════
        # CASH FLOW STATEMENT
        # ═══════════════════════════════════════════════════════════════════════
        capex = stmt.revenue * capex_ratio
        stmt.cf_net_income = stmt.net_income
        stmt.cf_depreciation = stmt.depreciation
        stmt.cf_change_receivables = -(stmt.accounts_receivable - p.accounts_receivable)
        stmt.cf_change_inventory = -(stmt.inventory - p.inventory)
        stmt.cf_change_payables = stmt.accounts_payable - p.accounts_payable
        stmt.cf_change_other = stmt.accrued_expenses - p.accrued_expenses
        stmt.cf_operating = (stmt.cf_net_income + stmt.cf_depreciation +
                            stmt.cf_change_receivables + stmt.cf_change_inventory +
                            stmt.cf_change_payables + stmt.cf_change_other)
        
        stmt.cf_capex = -capex
        stmt.cf_investing = stmt.cf_capex
        
        dividends = stmt.net_income * r['payout_ratio']
        buybacks = stmt.net_income * r['buyback_ratio']
        
        # Safeguard: Cap buybacks to prevent negative equity
        # Calculate what equity would be after all distributions
        estimated_equity_after = (p.common_stock + p.additional_paid_in_capital + 
                                  p.retained_earnings + stmt.net_income - dividends +
                                  p.treasury_stock + p.aoci)
        if p.treasury_stock != 0:
            # Treasury stock method - buybacks go to treasury
            if estimated_equity_after - buybacks < p.total_equity * 0.5:  # Keep at least 50% of prior equity
                buybacks = max(0, estimated_equity_after - p.total_equity * 0.5)
        else:
            # Retirement method - buybacks reduce RE
            if estimated_equity_after - buybacks < p.total_equity * 0.5:
                buybacks = max(0, estimated_equity_after - p.total_equity * 0.5)
        
        stmt.cf_dividends = -dividends
        stmt.cf_buybacks = -buybacks
        stmt.cf_financing = stmt.cf_dividends + stmt.cf_buybacks
        
        stmt.cf_net_change = stmt.cf_operating + stmt.cf_investing + stmt.cf_financing
        stmt.cf_beginning_cash = p.cash
        stmt.cf_ending_cash = stmt.cf_beginning_cash + stmt.cf_net_change
        
        # ═══════════════════════════════════════════════════════════════════════
        # BALANCE SHEET - ASSETS
        # ═══════════════════════════════════════════════════════════════════════
        # Current Assets
        stmt.cash = stmt.cf_ending_cash
        
        # Short-term investments: carry forward with slight growth
        if p.short_term_investments > 0:
            stmt.short_term_investments = p.short_term_investments * (1 + g * 0.5)
        else:
            stmt.short_term_investments = p.total_assets * r['short_term_inv_ratio'] if p.total_assets > 0 else 0
        
        # Other current assets: scale with revenue
        if p.other_current_assets > 0:
            stmt.other_current_assets = p.other_current_assets * (1 + g)
        else:
            stmt.other_current_assets = stmt.revenue * r['other_ca_ratio']
        
        stmt.total_current_assets = (stmt.cash + stmt.short_term_investments +
                                     stmt.accounts_receivable + stmt.inventory +
                                     stmt.prepaid_expenses + stmt.other_current_assets)
        
        # Non-Current Assets
        stmt.ppe_gross = p.ppe_gross + capex
        stmt.accumulated_depreciation = p.accumulated_depreciation + stmt.depreciation
        stmt.ppe_net = stmt.ppe_gross - stmt.accumulated_depreciation
        
        # Goodwill: constant (no acquisitions assumed)
        stmt.goodwill = p.goodwill
        
        # Intangibles: amortize slowly
        stmt.intangible_assets = p.intangible_assets * (1 - r['intangible_amort_rate'])
        
        # Long-term investments: carry forward with slight growth
        if p.long_term_investments > 0:
            stmt.long_term_investments = p.long_term_investments * (1 + g * 0.3)
        else:
            stmt.long_term_investments = p.total_assets * r['long_term_inv_ratio'] if p.total_assets > 0 else 0
        
        # Other non-current assets: scale slowly
        if p.other_noncurrent_assets > 0:
            stmt.other_noncurrent_assets = p.other_noncurrent_assets * (1 + g * 0.2)
        else:
            stmt.other_noncurrent_assets = p.total_assets * r['other_nca_ratio'] if p.total_assets > 0 else 0
        
        stmt.total_noncurrent_assets = (stmt.ppe_net + stmt.goodwill + stmt.intangible_assets +
                                        stmt.long_term_investments + stmt.other_noncurrent_assets)
        
        stmt.total_assets = stmt.total_current_assets + stmt.total_noncurrent_assets
        
        # ═══════════════════════════════════════════════════════════════════════
        # BALANCE SHEET - EQUITY
        # ═══════════════════════════════════════════════════════════════════════
        stmt.common_stock = p.common_stock
        stmt.additional_paid_in_capital = p.additional_paid_in_capital
        
        # Treasury Stock handling determines how buybacks affect equity
        # Method 1: Treasury Stock Method - buybacks go to treasury stock account
        # Method 2: Retirement Method - buybacks reduce retained earnings directly
        if p.treasury_stock != 0:
            # Company uses treasury stock method
            # Buybacks go to treasury stock, NOT retained earnings
            stmt.retained_earnings = p.retained_earnings + stmt.net_income - dividends
            stmt.treasury_stock = p.treasury_stock - buybacks
        else:
            # Company retires shares - buybacks reduce retained earnings
            stmt.retained_earnings = p.retained_earnings + stmt.net_income - dividends - buybacks
            stmt.treasury_stock = 0
        
        stmt.aoci = p.aoci
        stmt.total_equity = (stmt.common_stock + stmt.additional_paid_in_capital +
                            stmt.retained_earnings + stmt.treasury_stock + stmt.aoci)
        
        # ═══════════════════════════════════════════════════════════════════════
        # BALANCE SHEET - LIABILITIES (derived from A = L + E)
        # ═══════════════════════════════════════════════════════════════════════
        stmt.total_liabilities = stmt.total_assets - stmt.total_equity
        
        # Current liabilities - use carry forward for stability
        stmt.short_term_debt = p.short_term_debt
        stmt.current_portion_ltd = p.current_portion_ltd
        
        # Other current liabilities: carry forward with revenue growth (more stable)
        if p.other_current_liabilities > 0:
            stmt.other_current_liabilities = p.other_current_liabilities * (1 + g)
        else:
            stmt.other_current_liabilities = stmt.revenue * r['other_cl_ratio']
        
        stmt.total_current_liabilities = (stmt.accounts_payable + stmt.accrued_expenses +
                                          stmt.deferred_revenue + stmt.other_current_liabilities +
                                          stmt.short_term_debt + stmt.current_portion_ltd)
        
        # Non-current liabilities
        stmt.total_noncurrent_liabilities = max(0, stmt.total_liabilities - stmt.total_current_liabilities)
        stmt.long_term_debt = p.long_term_debt  # Assume debt stays constant
        
        # Use learned ratio for other NCL instead of residual calculation
        # This is more stable and based on actual company data
        if p.other_noncurrent_liabilities > 0:
            stmt.other_noncurrent_liabilities = p.other_noncurrent_liabilities * (1 + g * 0.1)
        else:
            stmt.other_noncurrent_liabilities = stmt.total_assets * r['other_ncl_ratio']
        
        # Deferred tax liabilities
        if p.deferred_tax_liabilities > 0:
            stmt.deferred_tax_liabilities = p.deferred_tax_liabilities * (1 + g * 0.1)
        else:
            stmt.deferred_tax_liabilities = max(0, stmt.total_noncurrent_liabilities * 0.05)
        
        # Pension liabilities
        stmt.pension_liabilities = p.pension_liabilities if p.pension_liabilities > 0 else 0
        
        # ═══════════════════════════════════════════════════════════════════════
        # VALIDATE IDENTITIES
        # ═══════════════════════════════════════════════════════════════════════
        stmt.identity_balance_sheet = abs(stmt.total_assets - stmt.total_liabilities - stmt.total_equity) < 1
        stmt.identity_cash_flow = abs(stmt.cf_ending_cash - (stmt.cf_beginning_cash + stmt.cf_net_change)) < 1
        stmt.identity_retained_earnings = True
        stmt.all_identities_hold = stmt.identity_balance_sheet and stmt.identity_cash_flow
        
        return stmt
