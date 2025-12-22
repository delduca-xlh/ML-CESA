# src/financial_planning/financial_statements/income_statement.py
"""
Income Statement Construction

Implements income statement construction that integrates with
the cash budget and balance sheet without circularity.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class IncomeStatementData:
    """Data structure for income statement components."""
    # Revenues
    sales_revenue: float = 0.0
    other_income: float = 0.0
    
    # Operating expenses
    cost_of_goods_sold: float = 0.0
    administrative_expenses: float = 0.0
    sales_expenses: float = 0.0
    depreciation: float = 0.0
    
    # Financial items
    interest_expense: float = 0.0
    interest_income: float = 0.0
    
    # Tax
    tax_rate: float = 0.0
    
    # Results
    gross_profit: float = 0.0
    ebit: float = 0.0
    ebt: float = 0.0
    tax_expense: float = 0.0
    net_income: float = 0.0


class IncomeStatement:
    """
    Income Statement construction without circularity.
    
    The Income Statement is constructed using data from:
    - Cash Budget (for interest items)
    - Intermediate tables (for operating items)
    - Debt schedules (for interest expense)
    
    Key principle: Interest expense is based on PREVIOUS period's debt,
    avoiding circularity.
    """
    
    def __init__(self):
        """Initialize income statement."""
        self.data = IncomeStatementData()
        self.history = []
    
    def construct(
        self,
        sales_revenue: float,
        cost_of_goods_sold: float,
        administrative_expenses: float,
        sales_expenses: float,
        depreciation: float,
        interest_expense: float,
        interest_income: float,
        tax_rate: float,
        other_income: float = 0.0
    ) -> Dict[str, float]:
        """
        Construct income statement from components.
        
        Args:
            sales_revenue: Total sales revenue
            cost_of_goods_sold: COGS
            administrative_expenses: Administrative expenses
            sales_expenses: Sales expenses
            depreciation: Depreciation expense
            interest_expense: Interest paid on debt (from previous period)
            interest_income: Interest earned on investments
            tax_rate: Corporate tax rate
            other_income: Other operating income
            
        Returns:
            Dictionary with complete income statement
        """
        # Calculate gross profit
        gross_profit = sales_revenue - cost_of_goods_sold
        
        # Calculate EBIT
        total_operating_expenses = (
            administrative_expenses +
            sales_expenses +
            depreciation
        )
        
        ebit = gross_profit - total_operating_expenses + other_income
        
        # Calculate EBT
        net_financial_expense = interest_expense - interest_income
        ebt = ebit - net_financial_expense
        
        # Calculate tax expense
        # Tax is only on positive income
        taxable_income = max(ebt, 0)
        tax_expense = taxable_income * tax_rate
        
        # Calculate net income
        net_income = ebt - tax_expense
        
        # Store data
        self.data = IncomeStatementData(
            sales_revenue=sales_revenue,
            other_income=other_income,
            cost_of_goods_sold=cost_of_goods_sold,
            administrative_expenses=administrative_expenses,
            sales_expenses=sales_expenses,
            depreciation=depreciation,
            interest_expense=interest_expense,
            interest_income=interest_income,
            tax_rate=tax_rate,
            gross_profit=gross_profit,
            ebit=ebit,
            ebt=ebt,
            tax_expense=tax_expense,
            net_income=net_income
        )
        
        return self.to_dict()
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert income statement to dictionary format.
        
        Returns:
            Dictionary with all income statement items
        """
        return {
            # Revenues
            'sales_revenue': self.data.sales_revenue,
            'other_income': self.data.other_income,
            'total_revenue': self.data.sales_revenue + self.data.other_income,
            
            # Cost of Sales
            'cost_of_goods_sold': self.data.cost_of_goods_sold,
            'gross_profit': self.data.gross_profit,
            'gross_margin': (
                self.data.gross_profit / self.data.sales_revenue 
                if self.data.sales_revenue != 0 else 0.0
            ),
            
            # Operating Expenses
            'administrative_expenses': self.data.administrative_expenses,
            'sales_expenses': self.data.sales_expenses,
            'depreciation': self.data.depreciation,
            'total_operating_expenses': (
                self.data.administrative_expenses +
                self.data.sales_expenses +
                self.data.depreciation
            ),
            
            # EBIT
            'ebit': self.data.ebit,
            'ebit_margin': (
                self.data.ebit / self.data.sales_revenue 
                if self.data.sales_revenue != 0 else 0.0
            ),
            
            # Financial Items
            'interest_expense': self.data.interest_expense,
            'interest_income': self.data.interest_income,
            'net_financial_expense': (
                self.data.interest_expense - self.data.interest_income
            ),
            
            # EBT
            'ebt': self.data.ebt,
            
            # Tax
            'tax_rate': self.data.tax_rate,
            'tax_expense': self.data.tax_expense,
            'effective_tax_rate': (
                self.data.tax_expense / self.data.ebt 
                if self.data.ebt > 0 else 0.0
            ),
            
            # Net Income
            'net_income': self.data.net_income,
            'net_margin': (
                self.data.net_income / self.data.sales_revenue 
                if self.data.sales_revenue != 0 else 0.0
            )
        }
    
    def calculate_noplat(self) -> float:
        """
        Calculate NOPLAT (Net Operating Profit Less Adjusted Taxes).
        
        NOPLAT = EBIT * (1 - Tax Rate)
        
        Returns:
            NOPLAT value
        """
        return self.data.ebit * (1 - self.data.tax_rate)
    
    def calculate_ebitda(self) -> float:
        """
        Calculate EBITDA.
        
        EBITDA = EBIT + Depreciation
        
        Returns:
            EBITDA value
        """
        return self.data.ebit + self.data.depreciation
    
    def calculate_tax_shield(self) -> float:
        """
        Calculate tax shield from interest expense.
        
        Based on Vélez-Pareja (2008): "Return to Basics: Are You 
        Properly Calculating Tax Shields?"
        
        TS = Tax Rate * min(EBIT, Interest Expense)
        
        Returns:
            Tax shield value
        """
        # Tax shield is limited by EBIT
        deductible_interest = min(
            max(self.data.ebit, 0),
            self.data.interest_expense
        )
        
        return self.data.tax_rate * deductible_interest
    
    def calculate_dividends(
        self,
        payout_ratio: float,
        previous_retained_earnings: float = 0.0
    ) -> tuple[float, float]:
        """
        Calculate dividends and retained earnings.
        
        Args:
            payout_ratio: Dividend payout ratio (0 to 1)
            previous_retained_earnings: Retained earnings from previous period
            
        Returns:
            Tuple of (dividends, new_retained_earnings)
        """
        # Dividends are only paid on positive net income
        distributable_income = max(self.data.net_income, 0)
        dividends = distributable_income * payout_ratio
        
        # Update retained earnings
        retained_earnings = (
            previous_retained_earnings +
            self.data.net_income -
            dividends
        )
        
        return dividends, retained_earnings
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert income statement to DataFrame.
        
        Returns:
            DataFrame with income statement
        """
        is_dict = self.to_dict()
        return pd.DataFrame([is_dict])
    
    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate income statement for common errors.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check for negative revenue
        if self.data.sales_revenue < 0:
            errors.append("Sales revenue is negative")
        
        # Check gross profit calculation
        expected_gross_profit = (
            self.data.sales_revenue - self.data.cost_of_goods_sold
        )
        if abs(expected_gross_profit - self.data.gross_profit) > 1e-6:
            errors.append("Gross profit calculation error")
        
        # Check EBIT calculation
        expected_ebit = (
            self.data.gross_profit -
            self.data.administrative_expenses -
            self.data.sales_expenses -
            self.data.depreciation +
            self.data.other_income
        )
        if abs(expected_ebit - self.data.ebit) > 1e-6:
            errors.append("EBIT calculation error")
        
        # Check EBT calculation
        expected_ebt = (
            self.data.ebit -
            self.data.interest_expense +
            self.data.interest_income
        )
        if abs(expected_ebt - self.data.ebt) > 1e-6:
            errors.append("EBT calculation error")
        
        # Check tax rate
        if not 0 <= self.data.tax_rate <= 1:
            errors.append("Tax rate must be between 0 and 1")
        
        # Check net income calculation
        expected_net_income = self.data.ebt - self.data.tax_expense
        if abs(expected_net_income - self.data.net_income) > 1e-6:
            errors.append("Net income calculation error")
        
        return len(errors) == 0, errors