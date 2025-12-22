# src/financial_planning/models/debt_schedule.py
"""
Debt Schedule Management

Manages multiple debt schedules for short-term and long-term debt,
calculating principal and interest payments for each period.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class DebtType(Enum):
    """Type of debt."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


@dataclass
class DebtSchedule:
    """
    Individual debt schedule.
    
    Tracks a single debt issuance and its repayment schedule.
    """
    debt_id: str
    debt_type: DebtType
    issue_period: int
    principal: float
    interest_rate: float
    term_periods: int
    payment_type: str = 'equal_principal'  # or 'amortizing'
    
    def __post_init__(self):
        """Calculate payment schedule after initialization."""
        self.schedule = self._calculate_schedule()
    
    def _calculate_schedule(self) -> pd.DataFrame:
        """
        Calculate the debt repayment schedule.
        
        Returns:
            DataFrame with payment schedule
        """
        if self.payment_type == 'equal_principal':
            return self._equal_principal_schedule()
        elif self.payment_type == 'amortizing':
            return self._amortizing_schedule()
        else:
            raise ValueError(f"Unknown payment type: {self.payment_type}")
    
    def _equal_principal_schedule(self) -> pd.DataFrame:
        """
        Calculate equal principal payment schedule.
        
        Returns:
            DataFrame with schedule
        """
        schedule_data = []
        remaining_principal = self.principal
        annual_principal_payment = self.principal / self.term_periods
        
        for year in range(1, self.term_periods + 1):
            period = self.issue_period + year
            
            # Interest on beginning balance
            interest_payment = remaining_principal * self.interest_rate
            
            # Equal principal payment
            principal_payment = annual_principal_payment
            
            # Total payment
            total_payment = principal_payment + interest_payment
            
            # Update remaining
            remaining_principal -= principal_payment
            
            schedule_data.append({
                'period': period,
                'beginning_balance': remaining_principal + principal_payment,
                'principal_payment': principal_payment,
                'interest_payment': interest_payment,
                'total_payment': total_payment,
                'ending_balance': remaining_principal
            })
        
        return pd.DataFrame(schedule_data)
    
    def _amortizing_schedule(self) -> pd.DataFrame:
        """
        Calculate amortizing (equal payment) schedule.
        
        Returns:
            DataFrame with schedule
        """
        # Calculate equal payment using annuity formula
        if self.interest_rate == 0:
            payment = self.principal / self.term_periods
        else:
            payment = self.principal * (
                self.interest_rate * (1 + self.interest_rate) ** self.term_periods
            ) / (
                (1 + self.interest_rate) ** self.term_periods - 1
            )
        
        schedule_data = []
        remaining_principal = self.principal
        
        for year in range(1, self.term_periods + 1):
            period = self.issue_period + year
            
            # Interest on beginning balance
            interest_payment = remaining_principal * self.interest_rate
            
            # Principal is the remainder
            principal_payment = payment - interest_payment
            
            # Update remaining
            remaining_principal -= principal_payment
            
            schedule_data.append({
                'period': period,
                'beginning_balance': remaining_principal + principal_payment,
                'principal_payment': principal_payment,
                'interest_payment': interest_payment,
                'total_payment': payment,
                'ending_balance': max(remaining_principal, 0)  # Avoid negative due to rounding
            })
        
        return pd.DataFrame(schedule_data)
    
    def get_payment_for_period(self, period: int) -> Dict[str, float]:
        """
        Get payment details for a specific period.
        
        Args:
            period: Period number
            
        Returns:
            Dictionary with payment details
        """
        if period <= self.issue_period:
            return {
                'principal': 0.0,
                'interest': 0.0,
                'total': 0.0
            }
        
        period_data = self.schedule[self.schedule['period'] == period]
        
        if period_data.empty:
            return {
                'principal': 0.0,
                'interest': 0.0,
                'total': 0.0
            }
        
        row = period_data.iloc[0]
        return {
            'principal': row['principal_payment'],
            'interest': row['interest_payment'],
            'total': row['total_payment']
        }
    
    def get_balance_for_period(self, period: int) -> float:
        """
        Get ending balance for a specific period.
        
        Args:
            period: Period number
            
        Returns:
            Ending balance
        """
        if period < self.issue_period:
            return 0.0
        
        if period == self.issue_period:
            return self.principal
        
        period_data = self.schedule[self.schedule['period'] == period]
        
        if period_data.empty:
            return 0.0
        
        return period_data.iloc[0]['ending_balance']


class DebtScheduleManager:
    """
    Manage multiple debt schedules.
    
    This class tracks all debt issuances and calculates aggregate
    payments and balances for each period.
    """
    
    def __init__(self):
        """Initialize debt schedule manager."""
        self.debt_schedules: List[DebtSchedule] = []
        self.next_debt_id = 1
    
    def add_short_term_debt(
        self,
        period: int,
        principal: float,
        interest_rate: float,
        term_periods: int = 1
    ) -> str:
        """
        Add a short-term debt.
        
        Args:
            period: Period when debt is issued
            principal: Principal amount
            interest_rate: Interest rate per period
            term_periods: Number of periods (default 1 for ST)
            
        Returns:
            Debt ID
        """
        debt_id = f"ST_{period}_{self.next_debt_id}"
        self.next_debt_id += 1
        
        debt = DebtSchedule(
            debt_id=debt_id,
            debt_type=DebtType.SHORT_TERM,
            issue_period=period,
            principal=principal,
            interest_rate=interest_rate,
            term_periods=term_periods,
            payment_type='equal_principal'
        )
        
        self.debt_schedules.append(debt)
        return debt_id
    
    def add_long_term_debt(
        self,
        period: int,
        principal: float,
        interest_rate: float,
        term_years: int = 5,
        payment_type: str = 'equal_principal'
    ) -> str:
        """
        Add a long-term debt.
        
        Args:
            period: Period when debt is issued
            principal: Principal amount
            interest_rate: Interest rate per period
            term_years: Number of years for repayment
            payment_type: 'equal_principal' or 'amortizing'
            
        Returns:
            Debt ID
        """
        debt_id = f"LT_{period}_{self.next_debt_id}"
        self.next_debt_id += 1
        
        debt = DebtSchedule(
            debt_id=debt_id,
            debt_type=DebtType.LONG_TERM,
            issue_period=period,
            principal=principal,
            interest_rate=interest_rate,
            term_periods=term_years,
            payment_type=payment_type
        )
        
        self.debt_schedules.append(debt)
        return debt_id
    
    def get_short_term_payment(self, period: int) -> Dict[str, float]:
        """
        Get total short-term debt payment for a period.
        
        Args:
            period: Period number
            
        Returns:
            Dictionary with payment details
        """
        total_principal = 0.0
        total_interest = 0.0
        
        for debt in self.debt_schedules:
            if debt.debt_type == DebtType.SHORT_TERM:
                payment = debt.get_payment_for_period(period)
                total_principal += payment['principal']
                total_interest += payment['interest']
        
        return {
            'principal': total_principal,
            'interest': total_interest,
            'total': total_principal + total_interest
        }
    
    def get_long_term_payment(self, period: int) -> Dict[str, float]:
        """
        Get total long-term debt payment for a period.
        
        Args:
            period: Period number
            
        Returns:
            Dictionary with payment details
        """
        total_principal = 0.0
        total_interest = 0.0
        
        for debt in self.debt_schedules:
            if debt.debt_type == DebtType.LONG_TERM:
                payment = debt.get_payment_for_period(period)
                total_principal += payment['principal']
                total_interest += payment['interest']
        
        return {
            'principal': total_principal,
            'interest': total_interest,
            'total': total_principal + total_interest
        }
    
    def get_total_debt_balance(self, period: int) -> Dict[str, float]:
        """
        Get total debt balances for a period.
        
        Args:
            period: Period number
            
        Returns:
            Dictionary with debt balances
        """
        st_balance = 0.0
        lt_balance = 0.0
        
        for debt in self.debt_schedules:
            balance = debt.get_balance_for_period(period)
            
            if debt.debt_type == DebtType.SHORT_TERM:
                st_balance += balance
            else:
                lt_balance += balance
        
        return {
            'short_term': st_balance,
            'long_term': lt_balance,
            'total': st_balance + lt_balance
        }
    
    def get_all_schedules_dataframe(self) -> pd.DataFrame:
        """
        Get all debt schedules as a single DataFrame.
        
        Returns:
            DataFrame with all schedules
        """
        all_schedules = []
        
        for debt in self.debt_schedules:
            schedule = debt.schedule.copy()
            schedule['debt_id'] = debt.debt_id
            schedule['debt_type'] = debt.debt_type.value
            all_schedules.append(schedule)
        
        if not all_schedules:
            return pd.DataFrame()
        
        return pd.concat(all_schedules, ignore_index=True)
    
    def export_to_excel(self, filename: str):
        """
        Export all debt schedules to Excel.
        
        Args:
            filename: Output filename
        """
        all_schedules = self.get_all_schedules_dataframe()
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # All schedules
            all_schedules.to_excel(
                writer,
                sheet_name='All Debt Schedules',
                index=False
            )
            
            # Individual schedules
            for debt in self.debt_schedules:
                sheet_name = debt.debt_id[:31]  # Excel limit
                debt.schedule.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False
                )