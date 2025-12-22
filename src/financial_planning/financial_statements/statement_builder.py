# src/financial_planning/financial_statements/statement_builder.py
"""
Statement Builder

Orchestrates the construction of all financial statements
in the correct sequence to avoid circularity.

Sequence:
1. Cash Budget for period t-1 determines debt/investment for period t
2. Income Statement for period t uses debt from period t-1
3. Balance Sheet for period t uses results from CB and IS
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from .balance_sheet import BalanceSheet
from .income_statement import IncomeStatement
from .cash_budget import CashBudget


@dataclass
class StatementInputs:
    """Input data for constructing financial statements."""
    
    # ============================================
    # REQUIRED FIELDS (no defaults) - MUST BE FIRST
    # ============================================
    
    # Sales and revenue
    sales_revenue: float
    
    # Operating expenses
    cost_of_goods_sold: float
    administrative_expenses: float
    sales_expenses: float
    depreciation: float
    
    # Cash flows
    sales_inflows: float
    purchases_outflows: float
    
    # Investment
    investment_in_fixed_assets: float
    net_fixed_assets: float
    
    # Working capital
    accounts_receivable: float
    inventory: float
    accounts_payable: float
    
    # Parameters
    tax_rate: float
    minimum_cash_required: float
    
    # ============================================
    # OPTIONAL FIELDS (with defaults) - MUST BE LAST
    # ============================================
    
    # Sales and revenue (optional)
    other_income: float = 0.0
    
    # Working capital (optional)
    advance_payments_paid: float = 0.0
    advance_payments_received: float = 0.0
    
    # Financing
    equity_investment: float = 0.0
    dividend_payout_ratio: float = 0.0
    
    # Parameters (optional)
    st_investment_return_rate: float = 0.0
    debt_financing_ratio: float = 0.7
    
    # Previous period data
    previous_cumulated_ncb: float = 0.0
    previous_st_investment: float = 0.0
    previous_retained_earnings: float = 0.0
    previous_st_debt: float = 0.0
    previous_lt_debt: float = 0.0

class StatementBuilder:
    """
    Build financial statements without circularity or plugs.
    
    This class orchestrates the construction of all three financial
    statements in the correct sequence.
    """
    
    def __init__(self):
        """Initialize statement builder."""
        self.balance_sheet = BalanceSheet()
        self.income_statement = IncomeStatement()
        self.cash_budget = CashBudget()
        
        self.history = {
            'balance_sheets': [],
            'income_statements': [],
            'cash_budgets': []
        }
    
    def build_statements(
        self,
        inputs: StatementInputs,
        st_debt_schedule: Dict[str, float],
        lt_debt_schedule: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Build all three financial statements for one period.
        
        Args:
            inputs: All input data for the period
            st_debt_schedule: Short-term debt schedule
                {'principal': X, 'interest': Y}
            lt_debt_schedule: Long-term debt schedule
                {'principal': X, 'interest': Y}
            
        Returns:
            Dictionary containing all three statements
        """
        # Step 1: Build Cash Budget
        # This determines financing needs
        cb_result = self._build_cash_budget(
            inputs,
            st_debt_schedule,
            lt_debt_schedule
        )
        
        # Step 2: Build Income Statement
        # Uses interest from debt schedule (based on previous period's debt)
        is_result = self._build_income_statement(
            inputs,
            st_debt_schedule['interest'],
            lt_debt_schedule['interest'],
            cb_result['st_investment_return']
        )
        
        # Step 3: Build Balance Sheet
        # Uses results from CB and IS
        bs_result = self._build_balance_sheet(
            inputs,
            cb_result,
            is_result
        )
        
        # Store in history
        self.history['cash_budgets'].append(cb_result)
        self.history['income_statements'].append(is_result)
        self.history['balance_sheets'].append(bs_result)
        
        # Validate that balance sheet balances
        is_balanced, imbalance = self.balance_sheet.validate()
        if not is_balanced:
            raise ValueError(
                f"Balance sheet does not balance. Imbalance: {imbalance}"
            )
        
        return {
            'cash_budget': cb_result,
            'income_statement': is_result,
            'balance_sheet': bs_result
        }
    
    def _build_cash_budget(
        self,
        inputs: StatementInputs,
        st_debt_schedule: Dict[str, float],
        lt_debt_schedule: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Build cash budget for the period.
        
        Args:
            inputs: Statement inputs
            st_debt_schedule: Short-term debt payments
            lt_debt_schedule: Long-term debt payments
            
        Returns:
            Cash budget results
        """
        cb = self.cash_budget
        
        # Module 1: Operating Activities
        operating_ncb = cb.construct_module_1_operating(
            sales_inflows=inputs.sales_inflows,
            purchases_outflows=inputs.purchases_outflows,
            administrative_expenses=inputs.administrative_expenses,
            sales_expenses=inputs.sales_expenses,
            tax_payments=0.0,  # Taxes are calculated after IS is built
            other_operating_inflows=inputs.other_income
        )
        
        # Module 2: Investment in Assets
        investing_ncb = cb.construct_module_2_investing(
            investment_in_fixed_assets=inputs.investment_in_fixed_assets
        )
        
        # Module 3: External Financing (Debt)
        debt_financing = cb.construct_module_3_external_financing(
            previous_cumulated_ncb=inputs.previous_cumulated_ncb,
            minimum_cash_required=inputs.minimum_cash_required,
            st_principal_payment=st_debt_schedule['principal'],
            st_interest_payment=st_debt_schedule['interest'],
            lt_principal_payment=lt_debt_schedule['principal'],
            lt_interest_payment=lt_debt_schedule['interest'],
            debt_financing_ratio=inputs.debt_financing_ratio
        )
        
        # Module 4: Equity Financing
        # Calculate dividends based on previous period's net income
        # (We don't have current period's NI yet)
        equity_ncb = cb.construct_module_4_equity_financing(
            equity_investment=inputs.equity_investment,
            dividend_payment=0.0,  # Will be calculated from previous IS
            stock_repurchase=0.0
        )
        
        # Module 5: Discretionary Transactions
        has_new_financing = (
            debt_financing['st_loan'] > 0 or
            debt_financing['lt_loan'] > 0 or
            inputs.equity_investment > 0
        )
        
        discretionary = cb.construct_module_5_discretionary(
            previous_cumulated_ncb=inputs.previous_cumulated_ncb,
            minimum_cash_required=inputs.minimum_cash_required,
            previous_st_investment=inputs.previous_st_investment,
            st_investment_return_rate=inputs.st_investment_return_rate,
            has_new_debt_or_equity=has_new_financing
        )
        
        # Calculate cumulated NCB
        cumulated_ncb = cb.calculate_cumulated_ncb(
            inputs.previous_cumulated_ncb
        )
        
        # Combine all results
        result = cb.to_dict()
        result.update(debt_financing)
        result.update(discretionary)
        result['cumulated_ncb'] = cumulated_ncb
        
        return result
    
    def _build_income_statement(
        self,
        inputs: StatementInputs,
        st_interest_expense: float,
        lt_interest_expense: float,
        interest_income: float
    ) -> Dict[str, float]:
        """
        Build income statement for the period.
        
        Args:
            inputs: Statement inputs
            st_interest_expense: Short-term debt interest
            lt_interest_expense: Long-term debt interest
            interest_income: Interest from short-term investments
            
        Returns:
            Income statement results
        """
        is_result = self.income_statement.construct(
            sales_revenue=inputs.sales_revenue,
            cost_of_goods_sold=inputs.cost_of_goods_sold,
            administrative_expenses=inputs.administrative_expenses,
            sales_expenses=inputs.sales_expenses,
            depreciation=inputs.depreciation,
            interest_expense=st_interest_expense + lt_interest_expense,
            interest_income=interest_income,
            tax_rate=inputs.tax_rate,
            other_income=inputs.other_income
        )
        
        # Calculate dividends and retained earnings
        dividends, retained_earnings = self.income_statement.calculate_dividends(
            payout_ratio=inputs.dividend_payout_ratio,
            previous_retained_earnings=inputs.previous_retained_earnings
        )
        
        is_result['dividends'] = dividends
        is_result['retained_earnings'] = retained_earnings
        
        return is_result
    
    def _build_balance_sheet(
        self,
        inputs: StatementInputs,
        cb_result: Dict[str, float],
        is_result: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Build balance sheet for the period.
        
        Args:
            inputs: Statement inputs
            cb_result: Cash budget results
            is_result: Income statement results
            
        Returns:
            Balance sheet results
        """
        # Calculate total debt
        total_st_debt = inputs.previous_st_debt + cb_result['st_loan']
        total_lt_debt = inputs.previous_lt_debt + cb_result['lt_loan']
        
        # Calculate total equity investment
        total_equity_investment = inputs.equity_investment
        
        bs_result = self.balance_sheet.construct_from_components(
            cash=cb_result['cumulated_ncb'],
            accounts_receivable=inputs.accounts_receivable,
            inventory=inputs.inventory,
            advance_payments_paid=inputs.advance_payments_paid,
            short_term_investments=cb_result.get('new_st_investment', 0.0),
            net_fixed_assets=inputs.net_fixed_assets,
            accounts_payable=inputs.accounts_payable,
            advance_payments_received=inputs.advance_payments_received,
            short_term_debt=total_st_debt,
            long_term_debt=total_lt_debt,
            equity_investment=total_equity_investment,
            retained_earnings=is_result['retained_earnings'],
            current_year_net_income=is_result['net_income']
        )
        
        return bs_result
    
    def get_historical_statements(self) -> pd.DataFrame:
        """
        Get all historical statements as a DataFrame.
        
        Returns:
            DataFrame with all periods
        """
        if not self.history['balance_sheets']:
            return pd.DataFrame()
        
        # Combine all periods
        all_periods = []
        
        for i, (bs, is_stmt, cb) in enumerate(zip(
            self.history['balance_sheets'],
            self.history['income_statements'],
            self.history['cash_budgets']
        )):
            period_data = {'period': i}
            period_data.update({f'bs_{k}': v for k, v in bs.items()})
            period_data.update({f'is_{k}': v for k, v in is_stmt.items()})
            period_data.update({f'cb_{k}': v for k, v in cb.items()})
            all_periods.append(period_data)
        
        return pd.DataFrame(all_periods)
    
    def validate_all_statements(self) -> Tuple[bool, List[str]]:
        """
        Validate all constructed statements.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        all_errors = []
        
        # Validate balance sheet
        bs_valid, bs_errors = self.balance_sheet.validate()
        if not bs_valid:
            all_errors.extend([f"BS: {e}" for e in bs_errors])
        
        # Validate income statement
        is_valid, is_errors = self.income_statement.validate()
        if not is_valid:
            all_errors.extend([f"IS: {e}" for e in is_errors])
        
        # Validate cash budget
        cb_valid, cb_errors = self.cash_budget.validate()
        if not cb_valid:
            all_errors.extend([f"CB: {e}" for e in cb_errors])
        
        return len(all_errors) == 0, all_errors