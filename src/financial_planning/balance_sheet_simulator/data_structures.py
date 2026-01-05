"""
data_structures.py

Data classes for the Balance Sheet Simulator.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class QuantileForecast:
    """
    Probabilistic forecast for a single variable.
    
    Like LLM: P(next_word) = distribution over vocabulary
    Our model: P(next_driver) = distribution defined by quantiles
    """
    variable: str
    q05: float
    q10: float
    q25: float
    q50: float  # Median
    q75: float
    q90: float
    q95: float
    
    @property
    def mean(self) -> float:
        return self.q50
    
    @property
    def std(self) -> float:
        return (self.q95 - self.q05) / 3.29
    
    @property
    def ci_90(self) -> Tuple[float, float]:
        return (self.q05, self.q95)
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from the distribution using inverse CDF"""
        quantiles = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]
        values = [
            self.q05 - 2 * (self.q50 - self.q05),
            self.q05, self.q10, self.q25, self.q50,
            self.q75, self.q90, self.q95,
            self.q95 + 2 * (self.q95 - self.q50)
        ]
        u = np.random.uniform(0, 1, n)
        return np.interp(u, quantiles, values)
    
    def contains(self, value: float, level: float = 0.90) -> bool:
        if level == 0.90:
            return self.q05 <= value <= self.q95
        elif level == 0.50:
            return self.q25 <= value <= self.q75
        return False


@dataclass
class CompleteFinancialStatements:
    """Complete Three Financial Statements with 30+ line items."""
    period: str
    
    # Income Statement
    revenue: float = 0
    cogs: float = 0
    gross_profit: float = 0
    opex: float = 0
    sga: float = 0
    rd: float = 0
    depreciation: float = 0
    ebitda: float = 0
    ebit: float = 0
    interest_expense: float = 0
    interest_income: float = 0
    other_income: float = 0
    ebt: float = 0
    income_tax: float = 0
    net_income: float = 0
    
    # Balance Sheet - Assets
    cash: float = 0
    short_term_investments: float = 0
    accounts_receivable: float = 0
    inventory: float = 0
    prepaid_expenses: float = 0
    other_current_assets: float = 0
    total_current_assets: float = 0
    ppe_gross: float = 0
    accumulated_depreciation: float = 0
    ppe_net: float = 0
    goodwill: float = 0
    intangible_assets: float = 0
    long_term_investments: float = 0
    other_noncurrent_assets: float = 0
    total_noncurrent_assets: float = 0
    total_assets: float = 0
    
    # Balance Sheet - Liabilities
    accounts_payable: float = 0
    accrued_expenses: float = 0
    short_term_debt: float = 0
    current_portion_ltd: float = 0
    deferred_revenue: float = 0
    other_current_liabilities: float = 0
    total_current_liabilities: float = 0
    long_term_debt: float = 0
    deferred_tax_liabilities: float = 0
    pension_liabilities: float = 0
    other_noncurrent_liabilities: float = 0
    total_noncurrent_liabilities: float = 0
    total_liabilities: float = 0
    
    # Balance Sheet - Equity
    common_stock: float = 0
    additional_paid_in_capital: float = 0
    retained_earnings: float = 0
    treasury_stock: float = 0
    aoci: float = 0
    total_equity: float = 0
    
    # Cash Flow Statement
    cf_net_income: float = 0
    cf_depreciation: float = 0
    cf_change_receivables: float = 0
    cf_change_inventory: float = 0
    cf_change_payables: float = 0
    cf_change_other: float = 0
    cf_operating: float = 0
    cf_capex: float = 0
    cf_acquisitions: float = 0
    cf_investments: float = 0
    cf_investing: float = 0
    cf_debt_issued: float = 0
    cf_debt_repaid: float = 0
    cf_dividends: float = 0
    cf_buybacks: float = 0
    cf_stock_issued: float = 0
    cf_financing: float = 0
    cf_net_change: float = 0
    cf_beginning_cash: float = 0
    cf_ending_cash: float = 0
    
    # Validation Flags
    identity_balance_sheet: bool = False
    identity_cash_flow: bool = False
    identity_retained_earnings: bool = False
    all_identities_hold: bool = False
