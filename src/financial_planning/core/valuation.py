"""
Valuation Engine

Implements multiple valuation approaches:
1. Adjusted Present Value (APV)
2. WACC method with FCF
3. Capital Cash Flow (CCF)
4. Equity valuation with CFE and Ke

All methods avoid circularity and produce consistent results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from financial_planning.core.circularity_solver import CircularitySolver


@dataclass
class ValuationInputs:
    """Inputs required for valuation."""
    fcf: List[float]
    cfe: List[float]
    ts: List[float]
    debt: List[float]
    ku: float
    kd: float
    tax_rate: float
    discount_rate_ts: Literal['Ku', 'Kd']
    terminal_growth: float = 0.0
    terminal_leverage: float = 0.25


@dataclass
class ValuationResults:
    """Results from valuation."""
    firm_value: float
    equity_value: float
    debt_value: float
    enterprise_value: float
    pv_forecasted_fcf: float
    terminal_value: float
    npv: float
    method: str
    wacc_series: Optional[List[float]] = None
    ke_series: Optional[List[float]] = None
    vts_series: Optional[List[float]] = None  # NEW: Track tax shield values


class ValuationEngine:
    """
    Comprehensive valuation engine supporting multiple methods.
    
    CORRECTED VERSION - Properly tracks tax shield values (VTS).
    """
    
    def __init__(self, inputs: ValuationInputs):
        """
        Initialize valuation engine.
        
        Args:
            inputs: ValuationInputs object with all required data
        """
        self.inputs = inputs
        self.n_periods = len(inputs.fcf)
        
        # Initialize circularity solver
        self.solver = CircularitySolver(
            ku=inputs.ku,
            kd=inputs.kd,
            tax_rate=inputs.tax_rate,
            psi=inputs.discount_rate_ts
        )
        
        # Validate inputs
        self._validate_inputs()
        
        # Calculate VTS series once at initialization
        self.vts_series = self._calculate_vts_series()
    
    def _validate_inputs(self):
        """Validate that inputs are consistent."""
        if len(self.inputs.cfe) != self.n_periods:
            raise ValueError("CFE and FCF must have same length")
        if len(self.inputs.ts) != self.n_periods:
            raise ValueError("TS and FCF must have same length")
        if len(self.inputs.debt) != self.n_periods + 1:
            raise ValueError("Debt must have n_periods + 1 values (including initial)")
    
    def _calculate_vts_series(self) -> List[float]:
        """
        Calculate present value of tax shields for all periods.
        
        This is the KEY FIX - we calculate VTS properly and use it
        throughout all valuation methods.
        
        Formula:
        VTS_t = (VTS_{t+1} + TS_t) / (1 + ψ)
        
        Where ψ is the discount rate for tax shields (Ku or Kd).
        
        Returns:
            List of VTS values for each period (includes terminal)
        """
        # Initialize VTS series (n_periods + 1 to include terminal)
        vts_series = [0.0] * (self.n_periods + 1)
        
        # Calculate terminal value of tax shields
        _, ts_terminal = self.calculate_terminal_value()
        vts_series[-1] = ts_terminal
        
        # Get discount rate for tax shields
        discount_rate = (
            self.inputs.ku if self.inputs.discount_rate_ts == 'Ku' 
            else self.inputs.kd
        )
        
        # Work backwards from terminal value
        for t in range(self.n_periods - 1, -1, -1):
            vts_series[t] = (
                vts_series[t + 1] + self.inputs.ts[t]
            ) / (1 + discount_rate)
        
        return vts_series
    
    def calculate_terminal_value(self) -> Tuple[float, float]:
        """
        Calculate terminal value using circularity solver.
        
        Returns:
            Tuple of (levered_terminal_value, tax_shield_terminal_value)
        """
        # Get FCF for first year of perpetuity
        fcf_terminal = self.inputs.fcf[-1] * (1 + self.inputs.terminal_growth)
        
        # Calculate terminal values
        levered_tv, ts_tv = self.solver.calculate_terminal_value(
            fcf_n_plus_1=fcf_terminal,
            leverage_ratio=self.inputs.terminal_leverage,
            growth_rate=self.inputs.terminal_growth
        )
        
        return levered_tv, ts_tv
    
    def valuation_apv(self) -> ValuationResults:
        """
        Adjusted Present Value (APV) method.
        
        APV = PV(FCF at Ku) + PV(TS at ψ)
        
        This is the simplest method that avoids circularity entirely.
        
        Returns:
            ValuationResults object
        """
        # Discount rate for tax shields
        discount_rate_ts = (
            self.inputs.ku if self.inputs.discount_rate_ts == 'Ku' 
            else self.inputs.kd
        )
        
        # Calculate terminal value
        levered_tv, unlevered_tv = self.calculate_terminal_value()
        
        # Discount FCF at Ku (unlevered cost of equity)
        pv_fcf = 0.0
        for t, fcf in enumerate(self.inputs.fcf, start=1):
            pv_fcf += fcf / (1 + self.inputs.ku) ** t
        
        # Add discounted terminal value
        pv_terminal_fcf = unlevered_tv / (1 + self.inputs.ku) ** self.n_periods
        unlevered_value = pv_fcf + pv_terminal_fcf
        
        # The VTS at time 0 already includes all future tax shields
        # discounted to present value
        pv_ts = self.vts_series[0]
        
        # Calculate firm value
        firm_value = unlevered_value + pv_ts
        
        # Calculate equity value
        initial_debt = self.inputs.debt[0]
        equity_value = firm_value - initial_debt
        
        return ValuationResults(
            firm_value=firm_value,
            equity_value=equity_value,
            debt_value=initial_debt,
            enterprise_value=firm_value,
            pv_forecasted_fcf=pv_fcf,
            terminal_value=levered_tv,
            npv=equity_value,
            method="APV",
            vts_series=self.vts_series
        )
    
    def valuation_ccf(self) -> ValuationResults:
        """
        Capital Cash Flow (CCF) method.
        
        V = Σ [CCF_t / (1 + Ku)^t]
        
        Where CCF = FCF + TS
        
        This method discounts everything at Ku (when ψ = Ku).
        
        Returns:
            ValuationResults object
        """
        if self.inputs.discount_rate_ts != 'Ku':
            raise ValueError("CCF method requires discount_rate_ts = 'Ku'")
        
        # Calculate CCF = FCF + TS
        ccf = [fcf + ts for fcf, ts in zip(self.inputs.fcf, self.inputs.ts)]
        
        # Calculate terminal value
        levered_tv, _ = self.calculate_terminal_value()
        
        # Discount CCF at Ku
        pv_ccf = 0.0
        for t, ccf_t in enumerate(ccf, start=1):
            pv_ccf += ccf_t / (1 + self.inputs.ku) ** t
        
        # Add discounted terminal value
        pv_terminal = levered_tv / (1 + self.inputs.ku) ** self.n_periods
        firm_value = pv_ccf + pv_terminal
        
        # Calculate equity value
        initial_debt = self.inputs.debt[0]
        equity_value = firm_value - initial_debt
        
        return ValuationResults(
            firm_value=firm_value,
            equity_value=equity_value,
            debt_value=initial_debt,
            enterprise_value=firm_value,
            pv_forecasted_fcf=pv_ccf,
            terminal_value=levered_tv,
            npv=equity_value,
            method="CCF",
            vts_series=self.vts_series
        )
    
    def valuation_wacc(self) -> ValuationResults:
        """
        WACC method with FCF (no circularity).
        
        Uses circularity solver to calculate WACC for each period.
        NOW PROPERLY USES VTS VALUES!
        
        Returns:
            ValuationResults object
        """
        # Calculate terminal value first
        levered_tv, _ = self.calculate_terminal_value()
        
        # Work backwards from terminal value
        values = [levered_tv]
        wacc_series = []
        
        for t in range(self.n_periods - 1, -1, -1):
            # Get next period value
            V_t_plus_1 = values[0]
            
            # FIXED: Use actual VTS value instead of 0.0
            V_TS_t_plus_1 = self.vts_series[t + 1]
            
            # Calculate firm value at period t using circularity solver
            V_t = self.solver.calculate_firm_value(
                fcf_t=self.inputs.fcf[t],
                V_t_plus_1=V_t_plus_1,
                ts_t=self.inputs.ts[t],
                V_TS_t_plus_1=V_TS_t_plus_1  # FIXED: Use actual value
            )
            
            # Calculate WACC for period t
            wacc_t = self.solver.calculate_wacc(
                V_t=V_t,
                FCF_t=self.inputs.fcf[t],
                TS_t=self.inputs.ts[t],
                V_TS_t_plus_1=V_TS_t_plus_1  # FIXED: Use actual value
            )
            
            values.insert(0, V_t)
            wacc_series.insert(0, wacc_t)
        
        firm_value = values[0]
        initial_debt = self.inputs.debt[0]
        equity_value = firm_value - initial_debt
        
        # Calculate PV of forecasted FCF
        pv_fcf = firm_value - levered_tv / (1 + wacc_series[-1]) ** self.n_periods
        
        return ValuationResults(
            firm_value=firm_value,
            equity_value=equity_value,
            debt_value=initial_debt,
            enterprise_value=firm_value,
            pv_forecasted_fcf=pv_fcf,
            terminal_value=levered_tv,
            npv=equity_value,
            method="WACC",
            wacc_series=wacc_series,
            vts_series=self.vts_series
        )
    
    def valuation_equity_cfe(self) -> ValuationResults:
        """
        Equity valuation using CFE and Ke (no circularity).
        
        Uses circularity solver to calculate Ke for each period.
        NOW PROPERLY USES VTS VALUES!
        
        Returns:
            ValuationResults object
        """
        # Calculate terminal equity value
        levered_tv, _ = self.calculate_terminal_value()
        initial_debt = self.inputs.debt[0]
        terminal_debt = self.inputs.debt[-1]
        terminal_equity = levered_tv - terminal_debt
        
        # Work backwards from terminal equity value
        equity_values = [terminal_equity]
        ke_series = []
        
        for t in range(self.n_periods - 1, -1, -1):
            # Get next period values
            E_t_plus_1 = equity_values[0]
            D_t_plus_1 = self.inputs.debt[t + 1]
            
            # FIXED: Use actual VTS value instead of 0.0
            V_TS_t_plus_1 = self.vts_series[t + 1]
            
            # Calculate equity value at period t using circularity solver
            E_t = self.solver.calculate_equity_value(
                cfe_t=self.inputs.cfe[t],
                E_t_plus_1=E_t_plus_1,
                D_t_plus_1=D_t_plus_1,
                V_TS_t_plus_1=V_TS_t_plus_1  # FIXED: Use actual value
            )
            
            # Calculate Ke for period t
            ke_t = self.solver.calculate_levered_cost_of_equity(
                E_t=E_t,
                CFE_t=self.inputs.cfe[t],
                D_t_plus_1=D_t_plus_1,
                V_TS_t_plus_1=V_TS_t_plus_1  # FIXED: Use actual value
            )
            
            equity_values.insert(0, E_t)
            ke_series.insert(0, ke_t)
        
        equity_value = equity_values[0]
        firm_value = equity_value + initial_debt
        
        return ValuationResults(
            firm_value=firm_value,
            equity_value=equity_value,
            debt_value=initial_debt,
            enterprise_value=firm_value,
            pv_forecasted_fcf=equity_value,  # Approximation
            terminal_value=levered_tv,
            npv=equity_value,
            method="Equity_CFE",
            ke_series=ke_series,
            vts_series=self.vts_series
        )
    
    def valuation_all_methods(self) -> Dict[str, ValuationResults]:
        """
        Run all valuation methods and compare results.
        
        Returns:
            Dictionary with results from all methods
        """
        results = {
            'APV': self.valuation_apv(),
            'CCF': self.valuation_ccf(),
            'WACC': self.valuation_wacc(),
            'Equity_CFE': self.valuation_equity_cfe()
        }
        
        # Validate that all methods give similar results
        firm_values = [r.firm_value for r in results.values()]
        max_diff = max(firm_values) - min(firm_values)
        max_pct_diff = (max_diff / min(firm_values)) * 100 if min(firm_values) != 0 else 0
        
        if max_pct_diff > 1.0:  # 1% tolerance
            print(f"Warning: Methods show significant differences:")
            print(f"  Max difference: ${max_diff:,.2f} ({max_pct_diff:.2f}%)")
            for method, result in results.items():
                print(f"  {method}: ${result.firm_value:,.2f}")
        
        return results
    
    def create_valuation_summary(
        self,
        results: Dict[str, ValuationResults]
    ) -> pd.DataFrame:
        """
        Create summary DataFrame of valuation results.
        
        Args:
            results: Dictionary of ValuationResults from different methods
            
        Returns:
            DataFrame with comparison of methods
        """
        summary_data = []
        
        for method, result in results.items():
            summary_data.append({
                'Method': method,
                'Firm Value': result.firm_value,
                'Equity Value': result.equity_value,
                'Debt Value': result.debt_value,
                'NPV': result.npv,
                'Terminal Value': result.terminal_value,
                'PV Tax Shields': result.vts_series[0] if result.vts_series else None
            })
        
        df = pd.DataFrame(summary_data)
        return df
    
    def get_detailed_breakdown(self) -> pd.DataFrame:
        """
        Get detailed period-by-period breakdown.
        
        Returns:
            DataFrame with detailed cash flows and values
        """
        # Run WACC method to get period values
        wacc_result = self.valuation_wacc()
        
        breakdown_data = []
        for t in range(self.n_periods):
            breakdown_data.append({
                'Period': t + 1,
                'FCF': self.inputs.fcf[t],
                'Tax Shield': self.inputs.ts[t],
                'CCF': self.inputs.fcf[t] + self.inputs.ts[t],
                'CFE': self.inputs.cfe[t],
                'Debt': self.inputs.debt[t],
                'VTS': self.vts_series[t],
                'WACC': wacc_result.wacc_series[t] if wacc_result.wacc_series else None
            })
        
        return pd.DataFrame(breakdown_data)


def example_usage():
    """Example demonstrating valuation engine usage with proper VTS tracking."""
    
    print("=" * 80)
    print("CORRECTED Valuation Engine Example")
    print("=" * 80)
    
    # Create valuation inputs
    inputs = ValuationInputs(
        fcf=[19.26, 18.34, 23.67, 31.81],
        cfe=[0.0, 8.18, 12.64, 16.88],
        ts=[4.22, 3.56, 3.40, 3.06],
        debt=[91.97, 80.56, 77.00, 72.28, 63.04],
        ku=0.15,
        kd=0.13,
        tax_rate=0.35,
        discount_rate_ts='Ku',
        terminal_growth=0.03,
        terminal_leverage=0.25
    )
    
    # Initialize valuation engine
    engine = ValuationEngine(inputs)
    
    # Show VTS series (the key fix!)
    print("\nTax Shield Values (VTS) by Period:")
    print("-" * 80)
    for t, vts in enumerate(engine.vts_series):
        if t < len(inputs.ts):
            print(f"  Period {t}: VTS = ${vts:,.2f} (TS = ${inputs.ts[t]:,.2f})")
        else:
            print(f"  Terminal: VTS = ${vts:,.2f}")
    
    # Run all methods
    print("\n" + "=" * 80)
    print("Valuation Results (All Methods)")
    print("=" * 80)
    results = engine.valuation_all_methods()
    
    # Create summary
    summary = engine.create_valuation_summary(results)
    print("\n" + summary.to_string(index=False))
    
    # Detailed breakdown
    print("\n" + "=" * 80)
    print("Detailed Period Breakdown")
    print("=" * 80)
    breakdown = engine.get_detailed_breakdown()
    print("\n" + breakdown.to_string(index=False))
    
    # Verify consistency
    print("\n" + "=" * 80)
    print("Consistency Check")
    print("=" * 80)
    firm_values = [r.firm_value for r in results.values()]
    max_diff = max(firm_values) - min(firm_values)
    avg_value = sum(firm_values) / len(firm_values)
    max_pct_diff = (max_diff / avg_value) * 100
    
    print(f"\nAverage Firm Value: ${avg_value:,.2f}")
    print(f"Max Difference: ${max_diff:,.2f} ({max_pct_diff:.4f}%)")
    
    if max_pct_diff < 0.01:
        print("✓ All methods agree (within 0.01%)")
    elif max_pct_diff < 1.0:
        print("⚠ Methods mostly agree (within 1%)")
    else:
        print("✗ Methods show significant differences")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    example_usage()