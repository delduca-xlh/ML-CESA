# src/financial_planning/core/cost_of_capital.py
"""
Cost of Capital Calculations

This module implements cost of capital calculations without circularity,
based on the analytical solutions presented in the research papers.
"""

import numpy as np
from typing import Dict, Optional
import pandas as pd


class CostOfCapital:
    """
    Calculate cost of capital metrics without circularity.
    
    Implements the analytical solutions from:
    - Mejía-Peláez & Vélez-Pareja (2011)
    - Vélez-Pareja (2009)
    """
    
    def __init__(self, risk_free_rate: float, market_risk_premium: float):
        """
        Initialize cost of capital calculator.
        
        Args:
            risk_free_rate: Risk-free rate (e.g., 0.05 for 5%)
            market_risk_premium: Market risk premium
        """
        self.rf = risk_free_rate
        self.market_risk_premium = market_risk_premium
    
    def calculate_ku(self, beta_unlevered: float) -> float:
        """
        Calculate unlevered cost of equity (Ku).
        
        Args:
            beta_unlevered: Unlevered beta
            
        Returns:
            Unlevered cost of equity
        """
        return self.rf + beta_unlevered * self.market_risk_premium
    
    def calculate_ke_without_circularity(
        self,
        ku: float,
        kd: float,
        D: float,
        E_next: float,
        CFE_next: float,
        VTS_next: float,
        discount_rate_ts: str = 'ku'
    ) -> float:
        """
        Calculate levered cost of equity without circularity.
        
        Based on equation (11a) from Mejía-Peláez & Vélez-Pareja (2011):
        Ke = [Ku*(E+CFE)+(Ku-Kd)*D-(Ku-ψ)*VTS] / [E+CFE-(Ku-Kd)*D-(Ku-ψ)*VTS]
        
        Args:
            ku: Unlevered cost of equity
            kd: Cost of debt
            D: Market value of debt at t-1
            E_next: Market value of equity at t
            CFE_next: Cash flow to equity at t
            VTS_next: Value of tax shields at t
            discount_rate_ts: Discount rate for tax shields ('ku' or 'kd')
            
        Returns:
            Levered cost of equity (Ke)
        """
        if discount_rate_ts.lower() == 'ku':
            # Equation (11c): ψ = Ku
            numerator = ku * (E_next + CFE_next) + (ku - kd) * D
            denominator = E_next + CFE_next - (ku - kd) * D
            
        elif discount_rate_ts.lower() == 'kd':
            # Equation (11b): ψ = Kd
            numerator = ku * (E_next + CFE_next) + (ku - kd) * (D - VTS_next)
            denominator = E_next + CFE_next - (ku - kd) * (D - VTS_next)
        else:
            raise ValueError("discount_rate_ts must be 'ku' or 'kd'")
        
        if abs(denominator) < 1e-10:
            raise ValueError("Denominator too close to zero in Ke calculation")
            
        return numerator / denominator
    
    def calculate_wacc_without_circularity(
        self,
        ku: float,
        kd: float,
        D: float,
        V_next: float,
        FCF_next: float,
        VTS_next: float,
        TS_next: float,
        discount_rate_ts: str = 'ku'
    ) -> float:
        """
        Calculate WACC without circularity.
        
        Based on equation (10) from Mejía-Peláez & Vélez-Pareja (2011) for ψ=Ku:
        WACC = Ku*(V+FCF)-TS / (V+FCF+TS)
        
        Args:
            ku: Unlevered cost of equity
            kd: Cost of debt
            D: Market value of debt
            V_next: Firm value at t
            FCF_next: Free cash flow at t
            VTS_next: Value of tax shields at t
            TS_next: Tax shield at t
            discount_rate_ts: Discount rate for tax shields ('ku' or 'kd')
            
        Returns:
            Weighted average cost of capital
        """
        if discount_rate_ts.lower() == 'ku':
            # Equation (10): ψ = Ku
            numerator = ku * (V_next + FCF_next) - TS_next
            denominator = V_next + FCF_next + TS_next
            
        elif discount_rate_ts.lower() == 'kd':
            # Equation (9): ψ = Kd
            numerator = ku * (V_next + FCF_next) - (ku - kd) * VTS_next - TS_next
            denominator = V_next + FCF_next + (ku - kd) * VTS_next + TS_next
        else:
            raise ValueError("discount_rate_ts must be 'ku' or 'kd'")
        
        if abs(denominator) < 1e-10:
            raise ValueError("Denominator too close to zero in WACC calculation")
            
        return numerator / denominator
    
    def calculate_cost_of_debt(
        self,
        interest_expense: float,
        debt_balance: float
    ) -> float:
        """
        Calculate cost of debt from interest expense.
        
        Args:
            interest_expense: Interest paid in period
            debt_balance: Beginning debt balance
            
        Returns:
            Cost of debt
        """
        if debt_balance == 0:
            return 0.0
        return interest_expense / debt_balance
    
    def calculate_tax_shield(
        self,
        interest_expense: float,
        tax_rate: float,
        ebit: float
    ) -> float:
        """
        Calculate tax shield considering EBIT limitation.
        
        Based on the formula from the papers:
        TS = T * min(EBIT, Interest)
        
        Args:
            interest_expense: Interest expense
            tax_rate: Corporate tax rate
            ebit: Earnings before interest and taxes
            
        Returns:
            Tax shield value
        """
        # Tax shield cannot exceed EBIT
        taxable_interest = min(max(ebit, 0), interest_expense)
        return tax_rate * taxable_interest
    
    def calculate_vts(
        self,
        tax_shields: pd.Series,
        discount_rates: pd.Series,
        terminal_value: float = 0.0
    ) -> pd.Series:
        """
        Calculate present value of tax shields.
        
        Args:
            tax_shields: Series of tax shields
            discount_rates: Series of discount rates
            terminal_value: Terminal value of tax shields
            
        Returns:
            Series of cumulative PV of tax shields
        """
        n = len(tax_shields)
        vts = pd.Series(index=tax_shields.index, dtype=float)
        
        # Start from the end and work backwards
        vts.iloc[-1] = terminal_value
        
        for i in range(n-1, 0, -1):
            vts.iloc[i-1] = (vts.iloc[i] + tax_shields.iloc[i]) / (1 + discount_rates.iloc[i])
        
        return vts
    
    def calculate_terminal_value_ts(
        self,
        tax_rate: float,
        kd: float,
        leverage_perpetuity: float,
        ku: float,
        growth_rate: float,
        terminal_value_unlevered: float
    ) -> float:
        """
        Calculate terminal value of tax shields.
        
        Based on equation (12) from the paper:
        VTV_TS = (T*Kd*D%*VTV_L) / (Ku - g)
        
        Args:
            tax_rate: Corporate tax rate
            kd: Cost of debt
            leverage_perpetuity: Target leverage ratio (D/V)
            ku: Unlevered cost of equity
            growth_rate: Perpetual growth rate
            terminal_value_unlevered: Unlevered terminal value
            
        Returns:
            Terminal value of tax shields
        """
        if ku - growth_rate <= 0:
            raise ValueError("Ku must be greater than growth rate")
        
        # First calculate levered terminal value (equation 14)
        phi = 1 - (tax_rate * kd * leverage_perpetuity) / (ku - growth_rate)
        terminal_value_levered = terminal_value_unlevered / phi
        
        # Then calculate TS terminal value
        vts_terminal = (tax_rate * kd * leverage_perpetuity * terminal_value_levered) / (ku - growth_rate)
        
        return vts_terminal
    
    def calculate_capm_beta_levered(
        self,
        beta_unlevered: float,
        debt_to_equity: float,
        tax_rate: float
    ) -> float:
        """
        Calculate levered beta using traditional formula.
        
        βL = βU * [1 + (1-T) * D/E]
        
        Args:
            beta_unlevered: Unlevered beta
            debt_to_equity: Debt to equity ratio
            tax_rate: Corporate tax rate
            
        Returns:
            Levered beta
        """
        return beta_unlevered * (1 + (1 - tax_rate) * debt_to_equity)
    
    def calculate_capm_beta_unlevered(
        self,
        beta_levered: float,
        debt_to_equity: float,
        tax_rate: float
    ) -> float:
        """
        Calculate unlevered beta from levered beta.
        
        βU = βL / [1 + (1-T) * D/E]
        
        Args:
            beta_levered: Levered beta
            debt_to_equity: Debt to equity ratio
            tax_rate: Corporate tax rate
            
        Returns:
            Unlevered beta
        """
        return beta_levered / (1 + (1 - tax_rate) * debt_to_equity)