"""
Financial Planning and Valuation System

A comprehensive system for financial modeling and firm valuation
without plugs or circularity, based on academic research.

Main Components:
- Financial Statements (Cash Budget, Income Statement, Balance Sheet)
- Cash Flow Calculations (FCF, CFE, CCF)
- Valuation Methods (APV, WACC, CCF)
- Circularity Solver
- Analysis Tools (Sensitivity, Scenario, Monte Carlo)
"""

__version__ = "1.0.0"
__author__ = "Lihao Xiao"
__email__ = "lx2219.cu@gmail.com"

from financial_planning.models.financial_model import FinancialModel
from financial_planning.core.circularity_solver import CircularitySolver
from financial_planning.core.cash_flow import CashFlowCalculator
from financial_planning.core.valuation import ValuationEngine
from financial_planning.core.cost_of_capital import CostOfCapital

from financial_planning.financial_statements.cash_budget import CashBudget
from financial_planning.financial_statements.income_statement import IncomeStatement
from financial_planning.financial_statements.balance_sheet import BalanceSheet
from financial_planning.financial_statements.statement_builder import StatementBuilder

__all__ = [
    # Main model
    'FinancialModel',
    
    # Core components
    'CircularitySolver',
    'CashFlowCalculator',
    'ValuationEngine',
    'CostOfCapital',
    
    # Financial statements
    'CashBudget',
    'IncomeStatement',
    'BalanceSheet',
    'StatementBuilder',
]