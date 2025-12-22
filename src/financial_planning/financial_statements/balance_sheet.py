# src/financial_planning/financial_statements/balance_sheet.py
"""
Balance Sheet Construction

Implements balance sheet construction following the double-entry
principle without using plugs.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class BalanceSheetData:
    """Data structure for balance sheet components."""
    # Assets
    cash: float = 0.0
    accounts_receivable: float = 0.0
    inventory: float = 0.0
    advance_payments_paid: float = 0.0
    short_term_investments: float = 0.0
    net_fixed_assets: float = 0.0
    
    # Liabilities
    accounts_payable: float = 0.0
    advance_payments_received: float = 0.0
    short_term_debt: float = 0.0
    long_term_debt: float = 0.0
    
    # Equity
    equity_investment: float = 0.0
    retained_earnings: float = 0.0
    current_year_net_income: float = 0.0


class BalanceSheet:
    """
    Balance Sheet construction without plugs.
    
    Based on the methodology from Vélez-Pareja (2007):
    "Forecasting Financial Statements with No plugs and No Circularity"
    """
    
    def __init__(self):
        """Initialize balance sheet."""
        self.data = BalanceSheetData()
        self.history = []
    
    def construct_from_components(
        self,
        cash: float,
        accounts_receivable: float,
        inventory: float,
        advance_payments_paid: float,
        short_term_investments: float,
        net_fixed_assets: float,
        accounts_payable: float,
        advance_payments_received: float,
        short_term_debt: float,
        long_term_debt: float,
        equity_investment: float,
        retained_earnings: float,
        current_year_net_income: float
    ) -> Dict[str, float]:
        """
        Construct balance sheet from components.
        
        Args:
            All balance sheet line items
            
        Returns:
            Dictionary with balance sheet structure
        """
        self.data = BalanceSheetData(
            cash=cash,
            accounts_receivable=accounts_receivable,
            inventory=inventory,
            advance_payments_paid=advance_payments_paid,
            short_term_investments=short_term_investments,
            net_fixed_assets=net_fixed_assets,
            accounts_payable=accounts_payable,
            advance_payments_received=advance_payments_received,
            short_term_debt=short_term_debt,
            long_term_debt=long_term_debt,
            equity_investment=equity_investment,
            retained_earnings=retained_earnings,
            current_year_net_income=current_year_net_income
        )
        
        return self.to_dict()
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert balance sheet to dictionary format.
        
        Returns:
            Dictionary with all balance sheet items
        """
        # Assets
        current_assets = (
            self.data.cash +
            self.data.accounts_receivable +
            self.data.inventory +
            self.data.advance_payments_paid +
            self.data.short_term_investments
        )
        
        total_assets = current_assets + self.data.net_fixed_assets
        
        # Liabilities
        current_liabilities = (
            self.data.accounts_payable +
            self.data.advance_payments_received +
            self.data.short_term_debt
        )
        
        total_liabilities = current_liabilities + self.data.long_term_debt
        
        # Equity
        total_equity = (
            self.data.equity_investment +
            self.data.retained_earnings +
            self.data.current_year_net_income
        )
        
        # Check balance
        total_liabilities_and_equity = total_liabilities + total_equity
        balance_check = total_assets - total_liabilities_and_equity
        
        return {
            # Assets
            'cash': self.data.cash,
            'accounts_receivable': self.data.accounts_receivable,
            'inventory': self.data.inventory,
            'advance_payments_paid': self.data.advance_payments_paid,
            'short_term_investments': self.data.short_term_investments,
            'current_assets': current_assets,
            'net_fixed_assets': self.data.net_fixed_assets,
            'total_assets': total_assets,
            
            # Liabilities
            'accounts_payable': self.data.accounts_payable,
            'advance_payments_received': self.data.advance_payments_received,
            'short_term_debt': self.data.short_term_debt,
            'current_liabilities': current_liabilities,
            'long_term_debt': self.data.long_term_debt,
            'total_liabilities': total_liabilities,
            
            # Equity
            'equity_investment': self.data.equity_investment,
            'retained_earnings': self.data.retained_earnings,
            'current_year_net_income': self.data.current_year_net_income,
            'total_equity': total_equity,
            
            # Totals
            'total_liabilities_and_equity': total_liabilities_and_equity,
            'balance_check': balance_check
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert balance sheet to DataFrame.
        
        Returns:
            DataFrame with balance sheet
        """
        bs_dict = self.to_dict()
        return pd.DataFrame([bs_dict])
    
    def validate(self, tolerance: float = 1e-6) -> tuple[bool, float]:
        """
        Validate that balance sheet balances.
        
        Args:
            tolerance: Maximum acceptable imbalance
            
        Returns:
            Tuple of (is_balanced, imbalance_amount)
        """
        bs_dict = self.to_dict()
        imbalance = bs_dict['balance_check']
        is_balanced = abs(imbalance) < tolerance
        
        return is_balanced, imbalance
    
    def get_working_capital(self) -> float:
        """
        Calculate working capital.
        
        Returns:
            Working capital (Current Assets - Current Liabilities)
        """
        bs_dict = self.to_dict()
        return bs_dict['current_assets'] - bs_dict['current_liabilities']
    
    def get_leverage_ratio(self) -> float:
        """
        Calculate leverage ratio (D/V).
        
        Returns:
            Debt to value ratio
        """
        bs_dict = self.to_dict()
        total_debt = bs_dict['short_term_debt'] + bs_dict['long_term_debt']
        total_value = total_debt + bs_dict['total_equity']
        
        if total_value == 0:
            return 0.0
        
        return total_debt / total_value
    
    def get_debt_to_equity_ratio(self) -> float:
        """
        Calculate debt to equity ratio (D/E).
        
        Returns:
            Debt to equity ratio
        """
        bs_dict = self.to_dict()
        total_debt = bs_dict['short_term_debt'] + bs_dict['long_term_debt']
        total_equity = bs_dict['total_equity']
        
        if total_equity == 0:
            return float('inf') if total_debt > 0 else 0.0
        
        return total_debt / total_equity