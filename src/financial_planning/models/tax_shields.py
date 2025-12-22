# src/financial_planning/models/tax_shields.py
"""
Tax Shield Calculations

Calculates tax shields from interest expenses and other tax-deductible items,
considering EBIT limitations and losses carried forward.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TaxShieldResult:
    """Result of tax shield calculation."""
    tax_shield: float
    taxable_income: float
    ebit: float
    interest_expense: float
    other_deductions: float
    tax_rate: float
    losses_carried_forward: float
    losses_used: float


class TaxShieldCalculator:
    """
    Calculate tax shields properly accounting for limitations.
    
    Based on Vélez-Pareja (2008): "Return to Basics: Are You 
    Properly Calculating Tax Shields?"
    
    Key principle: TS = T * min(max(EBIT, 0), Interest + Other Deductions)
    """
    
    def __init__(self, corporate_tax_rate: float):
        """
        Initialize tax shield calculator.
        
        Args:
            corporate_tax_rate: Corporate tax rate (e.g., 0.35 for 35%)
        """
        self.tax_rate = corporate_tax_rate
        self.losses_carried_forward = 0.0
    
    def calculate_tax_shield(
        self,
        ebit: float,
        interest_expense: float,
        other_deductions: float = 0.0,
        consider_losses_carried_forward: bool = True
    ) -> TaxShieldResult:
        """
        Calculate tax shield for a period.
        
        The tax shield is limited by EBIT and considers losses carried forward.
        
        Args:
            ebit: Earnings before interest and taxes
            interest_expense: Interest expense for the period
            other_deductions: Other tax-deductible expenses
            consider_losses_carried_forward: Whether to use loss carryforwards
            
        Returns:
            TaxShieldResult object
        """
        # Calculate taxable income before loss carryforwards
        taxable_income_before_lcf = ebit - interest_expense - other_deductions
        
        # Apply losses carried forward if applicable
        losses_used = 0.0
        if consider_losses_carried_forward and self.losses_carried_forward > 0:
            if taxable_income_before_lcf > 0:
                # Use losses to offset positive income
                losses_used = min(
                    self.losses_carried_forward,
                    taxable_income_before_lcf
                )
                taxable_income = taxable_income_before_lcf - losses_used
                self.losses_carried_forward -= losses_used
            else:
                taxable_income = taxable_income_before_lcf
        else:
            taxable_income = taxable_income_before_lcf
        
        # Calculate tax shield
        # Tax shield exists only when there are deductions and positive EBIT
        if ebit > 0 and interest_expense + other_deductions > 0:
            # Deductible amount is limited by EBIT
            deductible_amount = min(
                ebit,
                interest_expense + other_deductions
            )
            tax_shield = self.tax_rate * deductible_amount
        else:
            tax_shield = 0.0
        
        # If taxable income is negative, add to losses carried forward
        if taxable_income < 0 and consider_losses_carried_forward:
            self.losses_carried_forward += abs(taxable_income)
        
        return TaxShieldResult(
            tax_shield=tax_shield,
            taxable_income=max(taxable_income, 0),  # Can't have negative taxable income
            ebit=ebit,
            interest_expense=interest_expense,
            other_deductions=other_deductions,
            tax_rate=self.tax_rate,
            losses_carried_forward=self.losses_carried_forward,
            losses_used=losses_used
        )
    
    def calculate_tax_expense(
        self,
        ebt: float,
        consider_losses_carried_forward: bool = True
    ) -> Tuple[float, float]:
        """
        Calculate tax expense.
        
        Args:
            ebt: Earnings before taxes
            consider_losses_carried_forward: Whether to use loss carryforwards
            
        Returns:
            Tuple of (tax_expense, effective_tax_rate)
        """
        # Apply losses carried forward if applicable
        if consider_losses_carried_forward and self.losses_carried_forward > 0:
            if ebt > 0:
                losses_used = min(self.losses_carried_forward, ebt)
                taxable_income = ebt - losses_used
                self.losses_carried_forward -= losses_used
            else:
                taxable_income = ebt
        else:
            taxable_income = ebt
        
        # Tax only on positive income
        if taxable_income > 0:
            tax_expense = taxable_income * self.tax_rate
            effective_tax_rate = tax_expense / ebt if ebt != 0 else 0.0
        else:
            tax_expense = 0.0
            effective_tax_rate = 0.0
            
            # Add losses to carryforward
            if taxable_income < 0 and consider_losses_carried_forward:
                self.losses_carried_forward += abs(taxable_income)
        
        return tax_expense, effective_tax_rate
    
    def calculate_noplat(
        self,
        ebit: float,
        consider_losses_carried_forward: bool = True
    ) -> float:
        """
        Calculate NOPLAT (Net Operating Profit Less Adjusted Taxes).
        
        Args:
            ebit: Earnings before interest and taxes
            consider_losses_carried_forward: Whether to use loss carryforwards
            
        Returns:
            NOPLAT
        """
        # Calculate tax on EBIT
        if consider_losses_carried_forward and self.losses_carried_forward > 0:
            if ebit > 0:
                losses_used = min(self.losses_carried_forward, ebit)
                taxable_operating_income = ebit - losses_used
            else:
                taxable_operating_income = ebit
        else:
            taxable_operating_income = ebit
        
        # Tax only on positive income
        if taxable_operating_income > 0:
            tax_on_ebit = taxable_operating_income * self.tax_rate
        else:
            tax_on_ebit = 0.0
        
        return ebit - tax_on_ebit
    
    def reset_losses_carried_forward(self):
        """Reset losses carried forward to zero."""
        self.losses_carried_forward = 0.0
    
    def get_losses_carried_forward(self) -> float:
        """
        Get current losses carried forward balance.
        
        Returns:
            Losses carried forward
        """
        return self.losses_carried_forward
    
    def calculate_tax_shield_series(
        self,
        ebit_series: pd.Series,
        interest_series: pd.Series,
        other_deductions_series: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Calculate tax shields for a series of periods.
        
        Args:
            ebit_series: Series of EBIT values
            interest_series: Series of interest expenses
            other_deductions_series: Optional series of other deductions
            
        Returns:
            DataFrame with tax shield calculations
        """
        if other_deductions_series is None:
            other_deductions_series = pd.Series(
                0.0,
                index=ebit_series.index
            )
        
        results = []
        
        for period, ebit, interest, other_ded in zip(
            ebit_series.index,
            ebit_series,
            interest_series,
            other_deductions_series
        ):
            result = self.calculate_tax_shield(
                ebit=ebit,
                interest_expense=interest,
                other_deductions=other_ded
            )
            
            results.append({
                'period': period,
                'ebit': result.ebit,
                'interest_expense': result.interest_expense,
                'other_deductions': result.other_deductions,
                'tax_shield': result.tax_shield,
                'taxable_income': result.taxable_income,
                'losses_carried_forward': result.losses_carried_forward,
                'losses_used': result.losses_used
            })
        
        return pd.DataFrame(results)