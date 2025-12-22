"""
Cash Flow Calculations

Implements calculation of various cash flow metrics:
- Free Cash Flow (FCF)
- Cash Flow to Equity (CFE)
- Cash Flow to Debt (CFD)
- Capital Cash Flow (CCF)
- Tax Shields (TS)

Based on Vélez-Pareja & Tham (2004) and related research.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class CashFlowType(Enum):
    """Types of cash flows supported."""
    FCF = "Free Cash Flow"
    CFE = "Cash Flow to Equity"
    CFD = "Cash Flow to Debt"
    CCF = "Capital Cash Flow"
    TS = "Tax Shields"


@dataclass
class CashFlowComponents:
    """Components used in cash flow calculations."""
    ebit: float
    depreciation: float
    taxes: float
    change_in_nwc: float
    capex: float
    interest_expense: float
    interest_income: float
    principal_payment: float
    new_debt: float
    dividends: float
    new_equity: float
    
    def __post_init__(self):
        """Validate components."""
        if self.depreciation < 0:
            raise ValueError("Depreciation must be non-negative")
        if self.capex < 0:
            raise ValueError("Capex must be non-negative")


class CashFlowCalculator:
    """
    Calculates various cash flow metrics for firm valuation.
    
    Key Relationships:
    1. FCF = NOPLAT + Depreciation - ΔNW C - Capex
    2. CFE = NI + Depreciation - ΔNW C - Capex - Principal + New Debt
    3. CFD = Interest + Principal - New Debt
    4. CCF = FCF + TS = CFE + CFD
    5. TS = T × Interest Expense (when EBIT > Interest)
    """
    
    def __init__(self, tax_rate: float):
        """
        Initialize cash flow calculator.
        
        Args:
            tax_rate: Corporate tax rate
        """
        if not 0 <= tax_rate < 1:
            raise ValueError(f"Tax rate must be between 0 and 1, got {tax_rate}")
        
        self.tax_rate = tax_rate
    
    def calculate_noplat(self, ebit: float) -> float:
        """
        Calculate Net Operating Profit Less Adjusted Taxes.
        
        NOPLAT = EBIT × (1 - T)
        
        Args:
            ebit: Earnings Before Interest and Taxes
            
        Returns:
            NOPLAT
        """
        return ebit * (1 - self.tax_rate)
    
    def calculate_fcf(
        self,
        ebit: float,
        depreciation: float,
        change_in_nwc: float,
        capex: float
    ) -> float:
        """
        Calculate Free Cash Flow.
        
        FCF = NOPLAT + Depreciation - ΔNWC - Capex
        
        Where:
        - NOPLAT = EBIT × (1 - T)
        - Depreciation is added back (non-cash expense)
        - ΔNWC is the increase in net working capital
        - Capex is capital expenditures
        
        Args:
            ebit: Earnings Before Interest and Taxes
            depreciation: Depreciation expense
            change_in_nwc: Change in net working capital (positive = increase)
            capex: Capital expenditures
            
        Returns:
            Free Cash Flow
        """
        noplat = self.calculate_noplat(ebit)
        fcf = noplat + depreciation - change_in_nwc - capex
        return fcf
    
    def calculate_cfe(
        self,
        net_income: float,
        depreciation: float,
        change_in_nwc: float,
        capex: float,
        principal_payment: float,
        new_debt: float
    ) -> float:
        """
        Calculate Cash Flow to Equity.
        
        CFE = NI + Depreciation - ΔNWC - Capex - Principal + New Debt
        
        This is the actual cash available to equity holders.
        
        Args:
            net_income: Net income after taxes
            depreciation: Depreciation expense
            change_in_nwc: Change in net working capital
            capex: Capital expenditures
            principal_payment: Debt principal repayment
            new_debt: New debt raised
            
        Returns:
            Cash Flow to Equity
        """
        cfe = (
            net_income + 
            depreciation - 
            change_in_nwc - 
            capex - 
            principal_payment + 
            new_debt
        )
        return cfe
    
    def calculate_cfd(
        self,
        interest_expense: float,
        principal_payment: float,
        new_debt: float
    ) -> float:
        """
        Calculate Cash Flow to Debt.
        
        CFD = Interest + Principal - New Debt
        
        This is the cash paid to debt holders (net of new borrowing).
        
        Args:
            interest_expense: Interest paid on debt
            principal_payment: Principal repayment
            new_debt: New debt raised
            
        Returns:
            Cash Flow to Debt
        """
        cfd = interest_expense + principal_payment - new_debt
        return cfd
    
    def calculate_tax_shield(
        self,
        ebit: float,
        interest_expense: float,
        other_income: float = 0.0
    ) -> float:
        """
        Calculate Tax Shield from interest expense.
        
        Critical: TS is only realized if EBIT + Other Income ≥ Interest
        
        TS = T × min(EBIT + Other Income, Interest Expense)
        
        If EBIT + Other Income < Interest Expense:
            TS = T × (EBIT + Other Income) if positive, else 0
        
        Args:
            ebit: Earnings Before Interest and Taxes
            interest_expense: Interest expense
            other_income: Other income (e.g., interest income)
            
        Returns:
            Tax Shield
        """
        # Calculate maximum deductible amount
        operating_income = ebit + other_income
        
        if operating_income >= interest_expense:
            # Full interest is deductible
            deductible = interest_expense
        elif operating_income > 0:
            # Partial interest is deductible
            deductible = operating_income
        else:
            # No deduction possible (operating loss)
            deductible = 0.0
        
        return self.tax_rate * deductible
    
    def calculate_ccf(
        self,
        fcf: float,
        tax_shield: float
    ) -> float:
        """
        Calculate Capital Cash Flow.
        
        CCF = FCF + TS = CFE + CFD
        
        This is the fundamental relationship in cash flow valuation.
        
        Args:
            fcf: Free Cash Flow
            tax_shield: Tax Shield
            
        Returns:
            Capital Cash Flow
        """
        return fcf + tax_shield
    
    def calculate_all_cash_flows(
        self,
        components: CashFlowComponents
    ) -> Dict[str, float]:
        """
        Calculate all cash flow metrics from components.
        
        Args:
            components: CashFlowComponents object with all required data
            
        Returns:
            Dictionary with all cash flow metrics
        """
        # Calculate net income
        net_income = (
            components.ebit - 
            components.interest_expense + 
            components.interest_income
        ) * (1 - self.tax_rate)
        
        # Calculate individual cash flows
        fcf = self.calculate_fcf(
            ebit=components.ebit,
            depreciation=components.depreciation,
            change_in_nwc=components.change_in_nwc,
            capex=components.capex
        )
        
        cfe = self.calculate_cfe(
            net_income=net_income,
            depreciation=components.depreciation,
            change_in_nwc=components.change_in_nwc,
            capex=components.capex,
            principal_payment=components.principal_payment,
            new_debt=components.new_debt
        )
        
        cfd = self.calculate_cfd(
            interest_expense=components.interest_expense,
            principal_payment=components.principal_payment,
            new_debt=components.new_debt
        )
        
        ts = self.calculate_tax_shield(
            ebit=components.ebit,
            interest_expense=components.interest_expense,
            other_income=components.interest_income
        )
        
        ccf = self.calculate_ccf(fcf=fcf, tax_shield=ts)
        
        # Validate fundamental relationship: CCF = CFE + CFD
        ccf_check = cfe + cfd
        if not np.isclose(ccf, ccf_check, rtol=1e-6):
            raise ValueError(
                f"Cash flow consistency check failed: "
                f"CCF ({ccf:.2f}) ≠ CFE + CFD ({ccf_check:.2f})"
            )
        
        return {
            'FCF': fcf,
            'CFE': cfe,
            'CFD': cfd,
            'CCF': ccf,
            'TS': ts,
            'Net_Income': net_income,
            'NOPLAT': self.calculate_noplat(components.ebit)
        }
    
    def calculate_series(
        self,
        components_list: List[CashFlowComponents]
    ) -> pd.DataFrame:
        """
        Calculate cash flows for multiple periods.
        
        Args:
            components_list: List of CashFlowComponents for each period
            
        Returns:
            DataFrame with cash flows for all periods
        """
        results = []
        
        for i, components in enumerate(components_list):
            period_flows = self.calculate_all_cash_flows(components)
            period_flows['Period'] = i
            results.append(period_flows)
        
        df = pd.DataFrame(results)
        df = df.set_index('Period')
        
        return df
    
    def validate_cash_flows(
        self,
        fcf: float,
        cfe: float,
        cfd: float,
        ts: float,
        tolerance: float = 1e-6
    ) -> Tuple[bool, str]:
        """
        Validate that cash flows satisfy fundamental relationships.
        
        Checks:
        1. CCF = FCF + TS
        2. CCF = CFE + CFD
        
        Args:
            fcf: Free Cash Flow
            cfe: Cash Flow to Equity
            cfd: Cash Flow to Debt
            ts: Tax Shield
            tolerance: Acceptable difference
            
        Returns:
            Tuple of (is_valid, message)
        """
        ccf_from_fcf = fcf + ts
        ccf_from_components = cfe + cfd
        
        if not np.isclose(ccf_from_fcf, ccf_from_components, rtol=tolerance):
            message = (
                f"Cash flow validation failed: "
                f"FCF + TS = {ccf_from_fcf:.6f}, "
                f"CFE + CFD = {ccf_from_components:.6f}"
            )
            return False, message
        
        return True, "Cash flows are consistent"


class CashFlowFromStatements:
    """
    Calculate cash flows from financial statements (indirect method).
    
    This is an alternative approach when you have complete financial statements.
    """
    
    def __init__(self, tax_rate: float):
        """Initialize with tax rate."""
        self.tax_rate = tax_rate
        self.calculator = CashFlowCalculator(tax_rate)
    
    def calculate_change_in_nwc(
        self,
        current_assets_t: float,
        current_assets_t_minus_1: float,
        current_liabilities_t: float,
        current_liabilities_t_minus_1: float,
        cash_t: float = 0.0,
        cash_t_minus_1: float = 0.0,
        debt_t: float = 0.0,
        debt_t_minus_1: float = 0.0
    ) -> float:
        """
        Calculate change in Net Working Capital.
        
        NWC = (Current Assets - Cash) - (Current Liabilities - Current Debt)
        ΔNWC = NWC_t - NWC_{t-1}
        
        Args:
            current_assets_t: Current assets at time t
            current_assets_t_minus_1: Current assets at time t-1
            current_liabilities_t: Current liabilities at time t
            current_liabilities_t_minus_1: Current liabilities at time t-1
            cash_t: Cash at time t (excluded from NWC)
            cash_t_minus_1: Cash at time t-1
            debt_t: Short-term debt at time t (excluded from NWC)
            debt_t_minus_1: Short-term debt at time t-1
            
        Returns:
            Change in Net Working Capital
        """
        nwc_t = (current_assets_t - cash_t) - (current_liabilities_t - debt_t)
        nwc_t_minus_1 = (
            (current_assets_t_minus_1 - cash_t_minus_1) - 
            (current_liabilities_t_minus_1 - debt_t_minus_1)
        )
        
        return nwc_t - nwc_t_minus_1
    
    def calculate_capex(
        self,
        net_fixed_assets_t: float,
        net_fixed_assets_t_minus_1: float,
        depreciation: float
    ) -> float:
        """
        Calculate Capital Expenditures.
        
        Capex = NFA_t - NFA_{t-1} + Depreciation
        
        This is derived from: NFA_t = NFA_{t-1} + Capex - Depreciation
        
        Args:
            net_fixed_assets_t: Net fixed assets at time t
            net_fixed_assets_t_minus_1: Net fixed assets at time t-1
            depreciation: Depreciation expense
            
        Returns:
            Capital Expenditures
        """
        return net_fixed_assets_t - net_fixed_assets_t_minus_1 + depreciation
    
    def fcf_from_statements(
        self,
        income_statement: Dict[str, float],
        balance_sheet_t: Dict[str, float],
        balance_sheet_t_minus_1: Dict[str, float]
    ) -> float:
        """
        Calculate FCF from financial statements.
        
        Args:
            income_statement: Dict with EBIT, Depreciation, etc.
            balance_sheet_t: Balance sheet at time t
            balance_sheet_t_minus_1: Balance sheet at time t-1
            
        Returns:
            Free Cash Flow
        """
        # Calculate components
        change_in_nwc = self.calculate_change_in_nwc(
            current_assets_t=balance_sheet_t['current_assets'],
            current_assets_t_minus_1=balance_sheet_t_minus_1['current_assets'],
            current_liabilities_t=balance_sheet_t['current_liabilities'],
            current_liabilities_t_minus_1=balance_sheet_t_minus_1['current_liabilities'],
            cash_t=balance_sheet_t.get('cash', 0),
            cash_t_minus_1=balance_sheet_t_minus_1.get('cash', 0),
            debt_t=balance_sheet_t.get('short_term_debt', 0),
            debt_t_minus_1=balance_sheet_t_minus_1.get('short_term_debt', 0)
        )
        
        capex = self.calculate_capex(
            net_fixed_assets_t=balance_sheet_t['net_fixed_assets'],
            net_fixed_assets_t_minus_1=balance_sheet_t_minus_1['net_fixed_assets'],
            depreciation=income_statement['depreciation']
        )
        
        # Calculate FCF
        return self.calculator.calculate_fcf(
            ebit=income_statement['ebit'],
            depreciation=income_statement['depreciation'],
            change_in_nwc=change_in_nwc,
            capex=capex
        )


def example_usage():
    """Example demonstrating cash flow calculator usage."""
    
    # Initialize calculator
    calculator = CashFlowCalculator(tax_rate=0.35)
    
    # Create cash flow components
    components = CashFlowComponents(
        ebit=50.0,
        depreciation=10.0,
        taxes=17.5,
        change_in_nwc=5.0,
        capex=15.0,
        interest_expense=8.0,
        interest_income=0.5,
        principal_payment=10.0,
        new_debt=5.0,
        dividends=10.0,
        new_equity=0.0
    )
    
    # Calculate all cash flows
    cash_flows = calculator.calculate_all_cash_flows(components)
    
    print("Cash Flow Analysis:")
    print("-" * 40)
    for metric, value in cash_flows.items():
        print(f"{metric:.<30} ${value:>8.2f}")
    
    # Validate cash flows
    is_valid, message = calculator.validate_cash_flows(
        fcf=cash_flows['FCF'],
        cfe=cash_flows['CFE'],
        cfd=cash_flows['CFD'],
        ts=cash_flows['TS']
    )
    print(f"\nValidation: {message}")
    
    # Example: Calculate series
    components_list = [components for _ in range(5)]
    df = calculator.calculate_series(components_list)
    print("\nCash Flow Series:")
    print(df)


if __name__ == "__main__":
    example_usage()