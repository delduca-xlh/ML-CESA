"""
Analytical solution to circularity problem between value and cost of capital.

Based on:
Mejía-Peláez, F. & Vélez-Pareja, I. (2011). "Analytical solution to the 
circularity problem in the discounted cash flow valuation framework."
Innovar, 21(42), 55-68.

Key Innovation:
Solves for equity value, firm value, WACC, and Ke without requiring
iterative calculations, using closed-form analytical solutions.
"""

import numpy as np
from typing import Tuple, Optional, Literal
from dataclasses import dataclass


@dataclass
class CircularityResults:
    """Results from circularity solver calculations."""
    equity_value: float
    firm_value: float
    wacc: float
    levered_cost_of_equity: float
    debt_value: float
    tax_shield_value: float


class CircularitySolver:
    """
    Solves circularity between WACC and firm value analytically.
    
    Core Equations (for ψ = Ku):
    
    1. Equity Value:
       E_t = (E_{t+1} + CFE_t + (Ku - Kd) * D_{t+1}) / (1 + Ku)
    
    2. Firm Value:
       V_t = (V_{t+1} + FCF_t + TS_t) / (1 + Ku)
    
    3. WACC:
       WACC_t = Ku * (V_t + FCF_t - TS_t) / (V_t + FCF_t + TS_t)
    
    4. Levered Cost of Equity:
       Ke_t = Ku * (E_t + CFE_t + (Ku - Kd) * D_{t+1}) / 
              (E_t + CFE_t - (Ku - Kd) * D_{t+1})
    """
    
    def __init__(
        self,
        ku: float,
        kd: float,
        tax_rate: float,
        psi: Literal['Ku', 'Kd'] = 'Ku'
    ):
        """
        Initialize circularity solver.
        
        Args:
            ku: Unlevered cost of equity (cost of equity without debt)
            kd: Cost of debt
            tax_rate: Corporate tax rate
            psi: Discount rate for tax shields ('Ku' or 'Kd')
        """
        self.ku = ku
        self.kd = kd
        self.tax_rate = tax_rate
        self.psi = psi
        
        # Validate inputs
        if not 0 < ku < 1:
            raise ValueError(f"Ku must be between 0 and 1, got {ku}")
        if not 0 < kd < 1:
            raise ValueError(f"Kd must be between 0 and 1, got {kd}")
        if not 0 <= tax_rate < 1:
            raise ValueError(f"Tax rate must be between 0 and 1, got {tax_rate}")
        if psi not in ['Ku', 'Kd']:
            raise ValueError(f"psi must be 'Ku' or 'Kd', got {psi}")
    
    @property
    def psi_rate(self) -> float:
        """Get the numeric value of psi (discount rate for tax shields)."""
        return self.ku if self.psi == 'Ku' else self.kd
    
    def calculate_equity_value(
        self,
        cfe_t: float,
        E_t_plus_1: float,
        D_t_plus_1: float,
        V_TS_t_plus_1: float
    ) -> float:
        """
        Calculate equity value without circularity.
        
        General formula:
        E_t = [E_{t+1} + CFE_t + (Ku - Kd) * D_{t+1} + (Ku - ψ) * V^TS_{t+1}] / (1 + Ku)
        
        For ψ = Ku (simplified):
        E_t = [E_{t+1} + CFE_t + (Ku - Kd) * D_{t+1}] / (1 + Ku)
        
        Args:
            cfe_t: Cash flow to equity at time t
            E_t_plus_1: Equity value at t+1
            D_t_plus_1: Debt value at t+1
            V_TS_t_plus_1: Value of tax shields at t+1
            
        Returns:
            Equity value at time t
        """
        numerator = (
            E_t_plus_1 + 
            cfe_t + 
            (self.ku - self.kd) * D_t_plus_1 + 
            (self.ku - self.psi_rate) * V_TS_t_plus_1
        )
        
        return numerator / (1 + self.ku)
    
    def calculate_firm_value(
        self,
        fcf_t: float,
        V_t_plus_1: float,
        ts_t: float,
        V_TS_t_plus_1: float
    ) -> float:
        """
        Calculate firm value without circularity (CCF approach).
        
        General formula:
        V_t = [V_{t+1} + FCF_t + TS_t + (Ku - ψ) * V^TS_{t+1}] / (1 + Ku)
        
        For ψ = Ku (simplified):
        V_t = [V_{t+1} + FCF_t + TS_t] / (1 + Ku)
        
        This is the Capital Cash Flow (CCF) valuation method.
        
        Args:
            fcf_t: Free cash flow at time t
            V_t_plus_1: Firm value at t+1
            ts_t: Tax shield at time t
            V_TS_t_plus_1: Value of tax shields at t+1
            
        Returns:
            Firm value at time t
        """
        numerator = (
            V_t_plus_1 + 
            fcf_t + 
            ts_t + 
            (self.ku - self.psi_rate) * V_TS_t_plus_1
        )
        
        return numerator / (1 + self.ku)
    
    def calculate_wacc(
        self,
        V_t: float,
        FCF_t: float,
        TS_t: float,
        V_TS_t_plus_1: float
    ) -> float:
        """
        Calculate WACC without circularity.
        
        General formula:
        WACC_t = Ku * (V_t + FCF_t - (Ku - ψ) * V^TS_{t+1} - TS_t) / 
                 (V_t + FCF_t + (Ku - ψ) * V^TS_{t+1} + TS_t)
        
        For ψ = Ku (simplified):
        WACC_t = Ku * (V_t + FCF_t - TS_t) / (V_t + FCF_t + TS_t)
        
        Args:
            V_t: Firm value at time t
            FCF_t: Free cash flow at time t
            TS_t: Tax shield at time t
            V_TS_t_plus_1: Value of tax shields at t+1
            
        Returns:
            WACC at time t
        """
        numerator = self.ku * (
            V_t + FCF_t - (self.ku - self.psi_rate) * V_TS_t_plus_1 - TS_t
        )
        
        denominator = (
            V_t + FCF_t + (self.ku - self.psi_rate) * V_TS_t_plus_1 + TS_t
        )
        
        return numerator / denominator
    
    def calculate_levered_cost_of_equity(
        self,
        E_t: float,
        CFE_t: float,
        D_t_plus_1: float,
        V_TS_t_plus_1: float
    ) -> float:
        """
        Calculate levered cost of equity (Ke) without circularity.
        
        General formula:
        Ke_t = Ku * [E_t + CFE_t + (Ku - Kd) * D_{t+1} + (Ku - ψ) * V^TS_{t+1}] /
               [E_t + CFE_t - (Ku - Kd) * D_{t+1} - (Ku - ψ) * V^TS_{t+1}]
        
        For ψ = Ku (simplified):
        Ke_t = Ku * [E_t + CFE_t + (Ku - Kd) * D_{t+1}] /
               [E_t + CFE_t - (Ku - Kd) * D_{t+1}]
        
        Args:
            E_t: Equity value at time t
            CFE_t: Cash flow to equity at time t
            D_t_plus_1: Debt value at t+1
            V_TS_t_plus_1: Value of tax shields at t+1
            
        Returns:
            Levered cost of equity at time t
        """
        numerator = self.ku * (
            E_t + CFE_t + 
            (self.ku - self.kd) * D_t_plus_1 + 
            (self.ku - self.psi_rate) * V_TS_t_plus_1
        )
        
        denominator = (
            E_t + CFE_t - 
            (self.ku - self.kd) * D_t_plus_1 - 
            (self.ku - self.psi_rate) * V_TS_t_plus_1
        )
        
        return numerator / denominator
    
    def calculate_terminal_value(
        self,
        fcf_n_plus_1: float,
        leverage_ratio: float,
        growth_rate: float = 0.0
    ) -> Tuple[float, float]:
        """
        Calculate terminal value solving circularity.
        
        Formula:
        φ = 1 - [T * Kd * D% / (Ku - g)]
        VTV_L = FCF_{N+1} / [(Ku - g) * φ]
        VTV_TS = [T * Kd * D% * VTV_L] / (Ku - g)
        
        Args:
            fcf_n_plus_1: Free cash flow at N+1 (first year of perpetuity)
            leverage_ratio: Target debt-to-value ratio (D%)
            growth_rate: Perpetual growth rate (g)
            
        Returns:
            Tuple of (levered_terminal_value, tax_shield_terminal_value)
        """
        if growth_rate >= self.ku:
            raise ValueError(
                f"Growth rate ({growth_rate}) must be less than Ku ({self.ku})"
            )
        
        # Calculate phi coefficient
        phi = 1 - (
            self.tax_rate * self.kd * leverage_ratio / 
            (self.ku - growth_rate)
        )
        
        if phi <= 0:
            raise ValueError(
                f"Invalid phi value ({phi}). Check leverage ratio and growth rate."
            )
        
        # Calculate levered terminal value
        levered_tv = fcf_n_plus_1 / ((self.ku - growth_rate) * phi)
        
        # Calculate tax shield terminal value
        ts_tv = (
            self.tax_rate * self.kd * leverage_ratio * levered_tv /
            (self.ku - growth_rate)
        )
        
        return levered_tv, ts_tv
    
    def solve_complete_period(
        self,
        fcf_t: float,
        cfe_t: float,
        ts_t: float,
        D_t: float,
        D_t_plus_1: float,
        next_period_values: Optional[CircularityResults] = None
    ) -> CircularityResults:
        """
        Solve for all values in a period without circularity.
        
        Args:
            fcf_t: Free cash flow at time t
            cfe_t: Cash flow to equity at time t
            ts_t: Tax shield at time t
            D_t: Debt value at time t
            D_t_plus_1: Debt value at t+1
            next_period_values: CircularityResults from t+1 (None for terminal period)
            
        Returns:
            CircularityResults with all calculated values
        """
        if next_period_values is None:
            # Terminal period - values are known/assumed
            E_t = 0.0  # Would be calculated from terminal value
            V_t = D_t
            wacc_t = self.ku
            ke_t = self.ku
            V_TS_t = 0.0
        else:
            # Calculate values using next period
            E_t = self.calculate_equity_value(
                cfe_t=cfe_t,
                E_t_plus_1=next_period_values.equity_value,
                D_t_plus_1=D_t_plus_1,
                V_TS_t_plus_1=next_period_values.tax_shield_value
            )
            
            V_t = self.calculate_firm_value(
                fcf_t=fcf_t,
                V_t_plus_1=next_period_values.firm_value,
                ts_t=ts_t,
                V_TS_t_plus_1=next_period_values.tax_shield_value
            )
            
            wacc_t = self.calculate_wacc(
                V_t=V_t,
                FCF_t=fcf_t,
                TS_t=ts_t,
                V_TS_t_plus_1=next_period_values.tax_shield_value
            )
            
            ke_t = self.calculate_levered_cost_of_equity(
                E_t=E_t,
                CFE_t=cfe_t,
                D_t_plus_1=D_t_plus_1,
                V_TS_t_plus_1=next_period_values.tax_shield_value
            )
            
            # Calculate tax shield value (discounted at psi)
            V_TS_t = (next_period_values.tax_shield_value + ts_t) / (1 + self.psi_rate)
        
        return CircularityResults(
            equity_value=E_t,
            firm_value=V_t,
            wacc=wacc_t,
            levered_cost_of_equity=ke_t,
            debt_value=D_t,
            tax_shield_value=V_TS_t
        )
    
    def validate_modigliani_miller(
        self,
        levered_value: float,
        unlevered_value: float,
        ts_value: float,
        tolerance: float = 1e-6
    ) -> bool:
        """
        Validate Modigliani-Miller proposition: V_L = V_U + V_TS.
        
        Args:
            levered_value: Levered firm value
            unlevered_value: Unlevered firm value
            ts_value: Tax shield value
            tolerance: Acceptable difference
            
        Returns:
            True if M&M holds within tolerance
        """
        difference = abs(levered_value - (unlevered_value + ts_value))
        return difference < tolerance


def example_usage():
    """Example demonstrating circularity solver usage."""
    
    # Initialize solver
    solver = CircularitySolver(
        ku=0.15,      # 15% unlevered cost of equity
        kd=0.13,      # 13% cost of debt
        tax_rate=0.35,  # 35% tax rate
        psi='Ku'      # Discount tax shields at Ku
    )
    
    # Example: Calculate equity value
    equity_value = solver.calculate_equity_value(
        cfe_t=8.18,
        E_t_plus_1=163.44,
        D_t_plus_1=77.00,
        V_TS_t_plus_1=20.64
    )
    print(f"Equity Value: ${equity_value:.2f}")
    
    # Example: Calculate firm value
    firm_value = solver.calculate_firm_value(
        fcf_t=19.26,
        V_t_plus_1=240.44,
        ts_t=4.22,
        V_TS_t_plus_1=20.64
    )
    print(f"Firm Value: ${firm_value:.2f}")
    
    # Example: Calculate WACC
    wacc = solver.calculate_wacc(
        V_t=229.20,
        FCF_t=18.34,
        TS_t=3.56,
        V_TS_t_plus_1=20.64
    )
    print(f"WACC: {wacc:.4%}")
    
    # Example: Calculate terminal value
    levered_tv, ts_tv = solver.calculate_terminal_value(
        fcf_n_plus_1=31.81,
        leverage_ratio=0.25,
        growth_rate=0.03
    )
    print(f"Levered Terminal Value: ${levered_tv:.2f}")
    print(f"Tax Shield Terminal Value: ${ts_tv:.2f}")


if __name__ == "__main__":
    example_usage()