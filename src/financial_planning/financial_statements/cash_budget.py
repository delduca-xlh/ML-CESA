# src/financial_planning/financial_statements/cash_budget.py
"""
Cash Budget Construction

The Cash Budget is the core of the financial planning model.
It determines debt needs and cash excess without circularity.

Based on Vélez-Pareja (2009): "Constructing Consistent Financial 
Planning Models for Valuation"
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class ModuleType(Enum):
    """Cash budget module types."""
    OPERATING = "operating"
    INVESTING = "investing"
    EXTERNAL_FINANCING = "external_financing"
    EQUITY_FINANCING = "equity_financing"
    DISCRETIONARY = "discretionary"


@dataclass
class CashBudgetModule:
    """Data structure for a cash budget module."""
    name: str
    inflows: float = 0.0
    outflows: float = 0.0
    net_cash_balance: float = 0.0


class CashBudget:
    """
    Cash Budget construction without plugs or circularity.
    
    The Cash Budget has 5 modules:
    1. Operating activities
    2. Investment in assets
    3. External financing (debt)
    4. Equity financing (owners)
    5. Discretionary transactions (short-term investments)
    
    Critical formulas:
    - Short-term debt covers operating deficits
    - Long-term debt covers investment deficits
    - Excess cash is invested in marketable securities
    """
    
    def __init__(self):
        """Initialize cash budget."""
        self.modules = {
            ModuleType.OPERATING: CashBudgetModule("Operating Activities"),
            ModuleType.INVESTING: CashBudgetModule("Investment in Assets"),
            ModuleType.EXTERNAL_FINANCING: CashBudgetModule("External Financing"),
            ModuleType.EQUITY_FINANCING: CashBudgetModule("Equity Financing"),
            ModuleType.DISCRETIONARY: CashBudgetModule("Discretionary Transactions")
        }
        self.cumulated_ncb = 0.0
        self.minimum_cash_required = 0.0
    
    def construct_module_1_operating(
        self,
        sales_inflows: float,
        purchases_outflows: float,
        administrative_expenses: float,
        sales_expenses: float,
        tax_payments: float,
        other_operating_inflows: float = 0.0,
        other_operating_outflows: float = 0.0
    ) -> float:
        """
        Module 1: Operating Activities.
        
        This module calculates the net cash balance from operations.
        
        Args:
            sales_inflows: Cash received from sales
            purchases_outflows: Cash paid for purchases
            administrative_expenses: Admin expenses paid
            sales_expenses: Sales expenses paid
            tax_payments: Income taxes paid
            other_operating_inflows: Other operating cash inflows
            other_operating_outflows: Other operating cash outflows
            
        Returns:
            Operating net cash balance
        """
        module = self.modules[ModuleType.OPERATING]
        
        module.inflows = sales_inflows + other_operating_inflows
        
        module.outflows = (
            purchases_outflows +
            administrative_expenses +
            sales_expenses +
            tax_payments +
            other_operating_outflows
        )
        
        module.net_cash_balance = module.inflows - module.outflows
        
        return module.net_cash_balance
    
    def construct_module_2_investing(
        self,
        investment_in_fixed_assets: float,
        proceeds_from_asset_sales: float = 0.0
    ) -> float:
        """
        Module 2: Investment in Assets.
        
        Args:
            investment_in_fixed_assets: Capital expenditures
            proceeds_from_asset_sales: Cash from selling assets
            
        Returns:
            Investing net cash balance
        """
        module = self.modules[ModuleType.INVESTING]
        
        module.inflows = proceeds_from_asset_sales
        module.outflows = investment_in_fixed_assets
        module.net_cash_balance = module.inflows - module.outflows
        
        return module.net_cash_balance
    
    def calculate_ncb_after_capex(self) -> float:
        """
        Calculate NCB after capital expenditures.
        
        This is used to determine long-term financing needs.
        
        Returns:
            NCB after investing activities
        """
        operating_ncb = self.modules[ModuleType.OPERATING].net_cash_balance
        investing_ncb = self.modules[ModuleType.INVESTING].net_cash_balance
        
        return operating_ncb + investing_ncb
    
    def construct_module_3_external_financing(
        self,
        previous_cumulated_ncb: float,
        minimum_cash_required: float,
        st_principal_payment: float,
        st_interest_payment: float,
        lt_principal_payment: float,
        lt_interest_payment: float,
        debt_financing_ratio: float = 0.7
    ) -> Dict[str, float]:
        """
        Module 3: External Financing (Debt).
        
        Critical formulas (from Vélez-Pareja 2009):
        
        ST Loan = -(Cumulated NCB + Operating NCB - ST Debt Payment - Min Cash)
        LT Loan = -(NCB after Capex + ST Loan - LT Debt Payment - Min Cash) * Debt%
        
        Args:
            previous_cumulated_ncb: Cumulated NCB from previous period
            minimum_cash_required: Target minimum cash balance
            st_principal_payment: Short-term debt principal payment
            st_interest_payment: Short-term debt interest payment
            lt_principal_payment: Long-term debt principal payment
            lt_interest_payment: Long-term debt interest payment
            debt_financing_ratio: Fraction of deficit financed by debt (vs equity)
            
        Returns:
            Dictionary with debt financing details
        """
        module = self.modules[ModuleType.EXTERNAL_FINANCING]
        operating_ncb = self.modules[ModuleType.OPERATING].net_cash_balance
        
        # Calculate short-term debt need (for operating deficit)
        st_debt_need = -(
            previous_cumulated_ncb +
            operating_ncb -
            st_principal_payment -
            st_interest_payment -
            minimum_cash_required
        )
        
        st_loan = max(st_debt_need, 0)  # Only borrow if deficit exists
        
        # Calculate long-term debt need (for capital investment deficit)
        ncb_after_capex = self.calculate_ncb_after_capex()
        
        lt_debt_need = -(
            previous_cumulated_ncb +
            ncb_after_capex +
            st_loan -
            st_principal_payment -
            st_interest_payment -
            lt_principal_payment -
            lt_interest_payment -
            minimum_cash_required
        )
        
        lt_loan = max(lt_debt_need * debt_financing_ratio, 0)
        
        # Calculate total debt service
        total_debt_payment = (
            st_principal_payment +
            st_interest_payment +
            lt_principal_payment +
            lt_interest_payment
        )
        
        # Module cash flows
        module.inflows = st_loan + lt_loan
        module.outflows = total_debt_payment
        module.net_cash_balance = module.inflows - module.outflows
        
        return {
            'st_loan': st_loan,
            'lt_loan': lt_loan,
            'st_principal_payment': st_principal_payment,
            'st_interest_payment': st_interest_payment,
            'lt_principal_payment': lt_principal_payment,
            'lt_interest_payment': lt_interest_payment,
            'total_debt_payment': total_debt_payment,
            'net_cash_balance': module.net_cash_balance
        }
    
    def construct_module_4_equity_financing(
        self,
        equity_investment: float,
        dividend_payment: float,
        stock_repurchase: float = 0.0
    ) -> float:
        """
        Module 4: Equity Financing (Transactions with Owners).
        
        Args:
            equity_investment: New equity invested
            dividend_payment: Dividends paid to shareholders
            stock_repurchase: Stock buybacks
            
        Returns:
            Equity financing net cash balance
        """
        module = self.modules[ModuleType.EQUITY_FINANCING]
        
        module.inflows = equity_investment
        module.outflows = dividend_payment + stock_repurchase
        module.net_cash_balance = module.inflows - module.outflows
        
        return module.net_cash_balance
    
    def calculate_ncb_after_financing(
        self,
        previous_cumulated_ncb: float
    ) -> float:
        """
        Calculate NCB after all financing activities.
        
        Args:
            previous_cumulated_ncb: Cumulated NCB from previous period
            
        Returns:
            NCB after financing
        """
        operating_ncb = self.modules[ModuleType.OPERATING].net_cash_balance
        investing_ncb = self.modules[ModuleType.INVESTING].net_cash_balance
        external_fin_ncb = self.modules[ModuleType.EXTERNAL_FINANCING].net_cash_balance
        equity_fin_ncb = self.modules[ModuleType.EQUITY_FINANCING].net_cash_balance
        
        return (
            previous_cumulated_ncb +
            operating_ncb +
            investing_ncb +
            external_fin_ncb +
            equity_fin_ncb
        )
    
    def construct_module_5_discretionary(
        self,
        previous_cumulated_ncb: float,
        minimum_cash_required: float,
        previous_st_investment: float,
        st_investment_return_rate: float,
        has_new_debt_or_equity: bool
    ) -> Dict[str, float]:
        """
        Module 5: Discretionary Transactions (Short-term Investments).
        
        Critical formula:
        ST Investment = Cumulated NCB + NCB after financing + 
                       ST Investment Return - Min Cash
        
        Only invest if there is no new debt or equity in the period.
        
        Args:
            previous_cumulated_ncb: Cumulated NCB from previous period
            minimum_cash_required: Target minimum cash balance
            previous_st_investment: ST investment from previous period
            st_investment_return_rate: Return rate on ST investments
            has_new_debt_or_equity: Whether new financing occurred
            
        Returns:
            Dictionary with investment details
        """
        module = self.modules[ModuleType.DISCRETIONARY]
        
        # Calculate return on previous investment
        st_investment_return = previous_st_investment * st_investment_return_rate
        
        # Redeem previous investment
        st_investment_redemption = previous_st_investment
        
        # Total inflows
        module.inflows = st_investment_redemption + st_investment_return
        
        # Calculate cash available for investment
        ncb_after_financing = self.calculate_ncb_after_financing(
            previous_cumulated_ncb
        )
        
        cash_available = (
            previous_cumulated_ncb +
            ncb_after_financing +
            module.inflows -
            minimum_cash_required
        )
        
        # Only invest if no new financing and positive excess cash
        if has_new_debt_or_equity or cash_available <= 0:
            new_st_investment = 0.0
        else:
            new_st_investment = cash_available
        
        module.outflows = new_st_investment
        module.net_cash_balance = module.inflows - module.outflows
        
        return {
            'st_investment_redemption': st_investment_redemption,
            'st_investment_return': st_investment_return,
            'new_st_investment': new_st_investment,
            'net_cash_balance': module.net_cash_balance
        }
    
    def calculate_period_ncb(self) -> float:
        """
        Calculate net cash balance for the period.
        
        Returns:
            Total NCB for the period
        """
        total_ncb = sum(
            module.net_cash_balance 
            for module in self.modules.values()
        )
        return total_ncb
    
    def calculate_cumulated_ncb(
        self,
        previous_cumulated_ncb: float
    ) -> float:
        """
        Calculate cumulated net cash balance.
        
        This equals the Cash line in the Balance Sheet.
        
        Args:
            previous_cumulated_ncb: Cumulated NCB from previous period
            
        Returns:
            New cumulated NCB
        """
        period_ncb = self.calculate_period_ncb()
        self.cumulated_ncb = previous_cumulated_ncb + period_ncb
        return self.cumulated_ncb
    
    def to_dict(self) -> Dict[str, any]:
        """
        Convert cash budget to dictionary format.
        
        Returns:
            Dictionary with all cash budget details
        """
        result = {
            'cumulated_ncb': self.cumulated_ncb,
            'minimum_cash_required': self.minimum_cash_required,
            'period_ncb': self.calculate_period_ncb()
        }
        
        # Add each module
        for module_type, module in self.modules.items():
            prefix = module_type.value
            result[f'{prefix}_inflows'] = module.inflows
            result[f'{prefix}_outflows'] = module.outflows
            result[f'{prefix}_ncb'] = module.net_cash_balance
        
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert cash budget to DataFrame.
        
        Returns:
            DataFrame with cash budget
        """
        cb_dict = self.to_dict()
        return pd.DataFrame([cb_dict])
    
    def validate(self, tolerance: float = 1e-6) -> Tuple[bool, List[str]]:
        """
        Validate cash budget calculations.
        
        Args:
            tolerance: Maximum acceptable calculation error
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check that each module NCB equals inflows - outflows
        for module_type, module in self.modules.items():
            expected_ncb = module.inflows - module.outflows
            if abs(expected_ncb - module.net_cash_balance) > tolerance:
                errors.append(
                    f"{module_type.value}: NCB calculation error "
                    f"(expected {expected_ncb}, got {module.net_cash_balance})"
                )
        
        return len(errors) == 0, errors