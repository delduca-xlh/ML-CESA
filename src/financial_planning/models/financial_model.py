# src/financial_planning/models/financial_model.py
"""
Financial Model - CORRECTED VERSION

Fixed critical issues:
1. Tax shield calculation
2. FCF calculation
3. Working capital tracking
4. CFE calculation
5. Valuation engine integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings

# NOTE: These imports assume your package structure
# Adjust paths as needed
try:
    from ..core.valuation import ValuationEngine, ValuationInputs
    from ..core.cost_of_capital import CostOfCapital
    from ..core.circularity_solver import CircularitySolver
    from ..financial_statements.statement_builder import StatementBuilder, StatementInputs
    from .intermediate_tables import IntermediateTables
    from .debt_schedule import DebtScheduleManager
    from .tax_shields import TaxShieldCalculator
except ImportError:
    # For standalone testing
    pass


@dataclass
class ModelParameters:
    """Parameters for the financial model."""
    # Time horizon
    forecast_periods: int
    
    # Initial conditions
    initial_equity: float
    initial_fixed_assets: float
    initial_inventory_units: float = 0.0
    initial_purchase_price: float = 0.0
    
    # Tax
    corporate_tax_rate: float = 0.35
    
    # Depreciation
    depreciation_years: int = 4
    depreciation_method: str = 'straight_line'
    
    # Debt terms
    long_term_loan_years: int = 5
    short_term_loan_years: int = 1
    
    # Policies and goals
    minimum_cash_percent: float = 0.04
    inventory_policy_months: float = 1/12
    accounts_receivable_percent: float = 0.05
    accounts_payable_percent: float = 0.10
    advance_payment_from_customers_percent: float = 0.10
    advance_payment_to_suppliers_percent: float = 0.10
    dividend_payout_ratio: float = 0.70
    debt_financing_ratio: float = 0.70
    
    # Market data
    risk_free_rate: float = 0.08
    market_risk_premium: float = 0.06
    beta_unlevered: float = 1.0
    cost_of_debt_spread: float = 0.05
    short_term_investment_return_spread: float = -0.02
    
    # Growth and inflation
    real_growth_rates: List[float] = field(default_factory=list)
    inflation_rates: List[float] = field(default_factory=list)
    
    # Perpetuity assumptions
    perpetual_growth_rate: float = 0.0
    perpetual_leverage_ratio: float = 0.25
    discount_rate_tax_shields: str = 'ku'  # 'ku' or 'kd'


@dataclass
class ForecastInputs:
    """Forecast inputs for each period."""
    # Sales
    sales_revenue: List[float]
    sales_volume_units: List[float]
    selling_price: List[float]
    
    # Costs
    cost_of_goods_sold: List[float]
    unit_cost: List[float]
    overhead_expenses: List[float]
    payroll_expenses: List[float]
    sales_commissions_percent: float = 0.04
    advertising_percent: float = 0.03
    
    # EBIT forecast (alternative to detailed operating forecast)
    ebit_forecast: Optional[List[float]] = None
    depreciation_forecast: List[float] = field(default_factory=list)
    
    # Capital expenditure
    capex_forecast: List[float] = field(default_factory=list)
    
    # Additional equity investments
    equity_investments: List[float] = field(default_factory=list)


class FinancialModel:
    """
    Complete financial planning and valuation model.
    
    CORRECTED VERSION - fixes critical cash flow and valuation issues.
    """
    
    def __init__(
        self,
        parameters: ModelParameters,
        forecast_inputs: ForecastInputs
    ):
        """Initialize financial model."""
        self.params = parameters
        self.inputs = forecast_inputs
        
        # Initialize components
        self.intermediate_tables = IntermediateTables(parameters)
        self.debt_schedule_manager = DebtScheduleManager()
        self.tax_shield_calculator = TaxShieldCalculator(
            parameters.corporate_tax_rate
        )
        self.statement_builder = StatementBuilder()
        
        # Initialize calculators
        self.cost_of_capital = CostOfCapital(
            risk_free_rate=parameters.risk_free_rate,
            market_risk_premium=parameters.market_risk_premium
        )
        
        # Results storage
        self.results = {
            'statements': [],
            'cash_flows': [],
            'valuations': [],
            'debt_schedules': [],
            'working_capital_tracker': []
        }
        
        # Track previous period for working capital changes
        self.previous_working_capital = None
        
        self._is_built = False
    
    def build_model(self) -> pd.DataFrame:
        """Build the complete financial model."""
        # Build intermediate tables
        self.intermediate_tables.build_all_tables(self.inputs)
        
        # Initialize with period 0
        previous_state = self._initialize_period_zero()
        
        # Build each forecast period
        for period in range(1, self.params.forecast_periods + 1):
            print(f"Building period {period}...")
            
            # Build statements for this period
            period_results = self._build_period(period, previous_state)
            
            # Store results
            self.results['statements'].append(period_results)
            
            # Update previous state
            previous_state = self._extract_state_for_next_period(
                period_results,
                period
            )
        
        self._is_built = True
        
        # Convert to DataFrame
        return self._results_to_dataframe()
    
    def _initialize_period_zero(self) -> Dict:
        """Initialize period 0 state."""
        # Build initial balance sheet
        initial_inventory = (
            self.params.initial_inventory_units *
            self.params.initial_purchase_price
        )
        
        # Calculate minimum cash required for year 1
        year_1_expenses = (
            self.inputs.overhead_expenses[0] +
            self.inputs.payroll_expenses[0]
        )
        minimum_cash = year_1_expenses * self.params.minimum_cash_percent
        
        # Determine initial financing need
        total_initial_investment = (
            self.params.initial_fixed_assets +
            initial_inventory +
            minimum_cash
        )
        
        initial_debt = max(
            total_initial_investment - self.params.initial_equity,
            0.0
        )
        
        # Create initial long-term debt schedule
        if initial_debt > 0:
            self.debt_schedule_manager.add_long_term_debt(
                period=0,
                principal=initial_debt,
                interest_rate=self._calculate_cost_of_debt(period=0),
                term_years=self.params.long_term_loan_years
            )
        
        # Initialize working capital tracking
        self.previous_working_capital = {
            'accounts_receivable': 0.0,
            'inventory': initial_inventory,
            'accounts_payable': 0.0,
            'advance_payments_paid': 0.0,
            'advance_payments_received': 0.0
        }
        
        return {
            'period': 0,
            'cumulated_ncb': minimum_cash,
            'st_investment': 0.0,
            'retained_earnings': 0.0,
            'st_debt': 0.0,
            'lt_debt': initial_debt,
            'equity_investment': self.params.initial_equity,
            'net_fixed_assets': self.params.initial_fixed_assets
        }
    
    def _build_period(
        self,
        period: int,
        previous_state: Dict
    ) -> Dict:
        """Build all statements for one period."""
        # Get debt schedules for this period
        st_debt_payment = self.debt_schedule_manager.get_short_term_payment(period)
        lt_debt_payment = self.debt_schedule_manager.get_long_term_payment(period)
        
        # Get intermediate table data
        intermediate_data = self.intermediate_tables.get_period_data(period)
        
        # Create statement inputs
        statement_inputs = self._create_statement_inputs(
            period,
            previous_state,
            intermediate_data
        )
        
        # Build statements
        statements = self.statement_builder.build_statements(
            inputs=statement_inputs,
            st_debt_schedule=st_debt_payment,
            lt_debt_schedule=lt_debt_payment
        )
        
        # Update debt schedules with new debt
        cb = statements['cash_budget']
        
        if cb.get('st_loan', 0.0) > 0:
            self.debt_schedule_manager.add_short_term_debt(
                period=period,
                principal=cb['st_loan'],
                interest_rate=self._calculate_cost_of_debt(period)
            )
        
        if cb.get('lt_loan', 0.0) > 0:
            self.debt_schedule_manager.add_long_term_debt(
                period=period,
                principal=cb['lt_loan'],
                interest_rate=self._calculate_cost_of_debt(period),
                term_years=self.params.long_term_loan_years
            )
        
        # Calculate cash flows - CORRECTED
        cash_flows = self._calculate_period_cash_flows_CORRECTED(statements)
        
        # Update working capital tracker
        self._update_working_capital_tracker(statements['balance_sheet'])
        
        # Combine all results
        return {
            'period': period,
            'statements': statements,
            'cash_flows': cash_flows,
            'intermediate_data': intermediate_data
        }
    
    def _calculate_period_cash_flows_CORRECTED(
        self,
        statements: Dict
    ) -> Dict[str, float]:
        """
        Calculate all cash flows for a period - CORRECTED VERSION.
        
        FIXES:
        1. Tax shield = T * min(EBIT, Interest), not tax expense
        2. FCF properly accounts for operating taxes
        3. CFE uses correct cash budget keys
        4. CCF = FCF + TS (not FCF + tax expense)
        """
        cb = statements['cash_budget']
        is_stmt = statements['income_statement']
        bs = statements['balance_sheet']
        
        # 1. TAX SHIELD - CORRECTED
        # Tax shield is the tax benefit from interest deductibility
        # TS = T * min(EBIT, Interest Expense)
        if is_stmt.get('interest_expense', 0.0) > 0 and is_stmt.get('ebit', 0.0) > 0:
            deductible_interest = min(
                is_stmt['ebit'],
                is_stmt['interest_expense']
            )
            ts = self.params.corporate_tax_rate * deductible_interest
        else:
            ts = 0.0
        
        # 2. FREE CASH FLOW - CORRECTED
        # FCF = NOPLAT + Depreciation - CapEx - Change in NWC
        # NOPLAT = EBIT * (1 - T) = Operating income after tax
        noplat = is_stmt.get('ebit', 0.0) * (1 - self.params.corporate_tax_rate)
        
        fcf = (
            noplat +
            is_stmt.get('depreciation', 0.0) -
            cb.get('investment_in_fixed_assets', 0.0) -
            self._calculate_change_in_working_capital(statements)
        )
        
        # 3. CAPITAL CASH FLOW - CORRECTED
        # CCF = FCF + TS
        ccf = fcf + ts
        
        # 4. CASH FLOW TO DEBT - CORRECTED
        cfd = (
            cb.get('st_loan', 0.0) + 
            cb.get('lt_loan', 0.0) -
            cb.get('st_principal_payment', 0.0) - 
            cb.get('lt_principal_payment', 0.0) -
            cb.get('st_interest_payment', 0.0) - 
            cb.get('lt_interest_payment', 0.0)
        )
        
        # 5. CASH FLOW TO EQUITY - CORRECTED
        # CFE = Equity investment - Dividends
        # Get from equity financing module NCB
        cfe = cb.get('equity_financing_ncb', 0.0)
        
        return {
            'fcf': fcf,
            'ccf': ccf,
            'ts': ts,
            'cfd': cfd,
            'cfe': cfe,
            'noplat': noplat
        }
    
    def _calculate_change_in_working_capital(
        self,
        statements: Dict
    ) -> float:
        """
        Calculate change in net working capital - CORRECTED.
        
        Change in NWC = Increase in current assets - Increase in current liabilities
        (excluding cash and debt)
        """
        bs = statements['balance_sheet']
        
        if self.previous_working_capital is None:
            return 0.0
        
        # Current period working capital components
        current_wc = {
            'accounts_receivable': bs.get('accounts_receivable', 0.0),
            'inventory': bs.get('inventory', 0.0),
            'accounts_payable': bs.get('accounts_payable', 0.0),
            'advance_payments_paid': bs.get('advance_payments_paid', 0.0),
            'advance_payments_received': bs.get('advance_payments_received', 0.0)
        }
        
        # Calculate change
        change_in_nwc = (
            (current_wc['accounts_receivable'] - self.previous_working_capital['accounts_receivable']) +
            (current_wc['inventory'] - self.previous_working_capital['inventory']) +
            (current_wc['advance_payments_paid'] - self.previous_working_capital['advance_payments_paid']) -
            (current_wc['accounts_payable'] - self.previous_working_capital['accounts_payable']) -
            (current_wc['advance_payments_received'] - self.previous_working_capital['advance_payments_received'])
        )
        
        return change_in_nwc
    
    def _update_working_capital_tracker(self, balance_sheet: Dict):
        """Update the working capital tracker with current period values."""
        self.previous_working_capital = {
            'accounts_receivable': balance_sheet.get('accounts_receivable', 0.0),
            'inventory': balance_sheet.get('inventory', 0.0),
            'accounts_payable': balance_sheet.get('accounts_payable', 0.0),
            'advance_payments_paid': balance_sheet.get('advance_payments_paid', 0.0),
            'advance_payments_received': balance_sheet.get('advance_payments_received', 0.0)
        }
    
    def _create_statement_inputs(
        self,
        period: int,
        previous_state: Dict,
        intermediate_data: Dict
    ) -> StatementInputs:
        """Create statement inputs for a period."""
        idx = period - 1  # Convert to 0-indexed
        
        # Get sales and cost data
        sales_revenue = self.inputs.sales_revenue[idx]
        cogs = self.inputs.cost_of_goods_sold[idx]
        
        # Calculate expenses
        admin_expenses = self.inputs.overhead_expenses[idx]
        sales_expenses = (
            self.inputs.payroll_expenses[idx] +
            sales_revenue * self.inputs.sales_commissions_percent +
            sales_revenue * self.inputs.advertising_percent
        )
        
        # Get depreciation
        depreciation = intermediate_data['depreciation']
        
        # Get capital expenditure
        capex = (
            self.inputs.capex_forecast[idx] 
            if idx < len(self.inputs.capex_forecast)
            else 0.0
        )
        
        # Get equity investment
        equity_investment = (
            self.inputs.equity_investments[idx]
            if idx < len(self.inputs.equity_investments)
            else 0.0
        )
        
        # Calculate minimum cash required
        minimum_cash = sales_revenue * self.params.minimum_cash_percent
        
        # Calculate ST investment return rate
        st_return_rate = (
            self.params.risk_free_rate + 
            self.params.short_term_investment_return_spread
        )
        
        return StatementInputs(
            sales_revenue=sales_revenue,
            cost_of_goods_sold=cogs,
            administrative_expenses=admin_expenses,
            sales_expenses=sales_expenses,
            depreciation=depreciation,
            sales_inflows=intermediate_data['sales_inflows'],
            purchases_outflows=intermediate_data['purchases_outflows'],
            investment_in_fixed_assets=capex,
            net_fixed_assets=intermediate_data['net_fixed_assets'],
            accounts_receivable=intermediate_data['accounts_receivable'],
            inventory=intermediate_data['inventory'],
            accounts_payable=intermediate_data['accounts_payable'],
            advance_payments_paid=intermediate_data.get('advance_payments_paid', 0.0),
            advance_payments_received=intermediate_data.get('advance_payments_received', 0.0),
            equity_investment=equity_investment,
            dividend_payout_ratio=self.params.dividend_payout_ratio,
            tax_rate=self.params.corporate_tax_rate,
            minimum_cash_required=minimum_cash,
            st_investment_return_rate=st_return_rate,
            debt_financing_ratio=self.params.debt_financing_ratio,
            previous_cumulated_ncb=previous_state['cumulated_ncb'],
            previous_st_investment=previous_state['st_investment'],
            previous_retained_earnings=previous_state['retained_earnings'],
            previous_st_debt=previous_state['st_debt'],
            previous_lt_debt=previous_state['lt_debt']
        )
    
    def _calculate_cost_of_debt(self, period: int) -> float:
        """Calculate cost of debt for a period."""
        idx = max(0, min(period - 1, len(self.params.inflation_rates) - 1))
        inflation = self.params.inflation_rates[idx] if self.params.inflation_rates else 0.0
        
        # Fisher equation: nominal rate = (1 + real) * (1 + inflation) - 1
        real_rate = self.params.risk_free_rate - inflation
        nominal_rf = (1 + real_rate) * (1 + inflation) - 1
        
        return nominal_rf + self.params.cost_of_debt_spread
    
    def _extract_state_for_next_period(
        self,
        period_results: Dict,
        period: int
    ) -> Dict:
        """Extract state needed for next period."""
        cb = period_results['statements']['cash_budget']
        is_stmt = period_results['statements']['income_statement']
        bs = period_results['statements']['balance_sheet']
        
        return {
            'period': period,
            'cumulated_ncb': cb['cumulated_ncb'],
            'st_investment': cb.get('new_st_investment', 0.0),
            'retained_earnings': is_stmt['retained_earnings'],
            'st_debt': bs['short_term_debt'],
            'lt_debt': bs['long_term_debt'],
            'equity_investment': bs['equity_investment'],
            'net_fixed_assets': bs['net_fixed_assets']
        }
    
    def calculate_valuation(self) -> Dict[str, float]:
        """
        Calculate firm valuation using multiple methods - CORRECTED.
        
        FIXES:
        1. Properly creates ValuationInputs
        2. Calls correct method names
        3. Properly calculates terminal values
        """
        if not self._is_built:
            raise ValueError("Must build model before valuation")
        
        # Extract cash flows
        cash_flows_df = self._extract_cash_flows_for_valuation()
        
        # Calculate Ku (unlevered cost of equity)
        ku = self.cost_of_capital.calculate_ku(self.params.beta_unlevered)
        
        # Calculate Kd (cost of debt)
        kd = self._calculate_cost_of_debt(self.params.forecast_periods)
        
        # Prepare valuation inputs
        valuation_inputs = ValuationInputs(
            fcf=cash_flows_df['fcf'].tolist(),
            cfe=cash_flows_df['cfe'].tolist(),
            ts=cash_flows_df['ts'].tolist(),
            debt=self._get_debt_balances(),
            ku=ku,
            kd=kd,
            tax_rate=self.params.corporate_tax_rate,
            discount_rate_ts='Ku' if self.params.discount_rate_tax_shields.lower() == 'ku' else 'Kd',
            terminal_growth=self.params.perpetual_growth_rate,
            terminal_leverage=self.params.perpetual_leverage_ratio
        )
        
        # Initialize valuation engine with inputs
        valuation_engine = ValuationEngine(valuation_inputs)
        
        # Run APV valuation
        apv_result = valuation_engine.valuation_apv()
        
        # Run CCF valuation (if using Ku for tax shields)
        if self.params.discount_rate_tax_shields.lower() == 'ku':
            ccf_result = valuation_engine.valuation_ccf()
        else:
            ccf_result = apv_result
        
        return {
            'apv_firm_value': apv_result.firm_value,
            'apv_equity_value': apv_result.equity_value,
            'apv_npv': apv_result.npv,
            'ccf_firm_value': ccf_result.firm_value,
            'ccf_equity_value': ccf_result.equity_value,
            'ku': ku,
            'kd': kd
        }
    
    def _get_debt_balances(self) -> List[float]:
        """Get debt balances for all periods including initial."""
        debt_balances = []
        
        # Initial debt
        debt_balances.append(
            self.debt_schedule_manager.get_total_debt_balance(0)['total']
        )
        
        # Forecast periods
        for result in self.results['statements']:
            bs = result['statements']['balance_sheet']
            total_debt = bs.get('short_term_debt', 0.0) + bs.get('long_term_debt', 0.0)
            debt_balances.append(total_debt)
        
        return debt_balances
    
    def _extract_cash_flows_for_valuation(self) -> pd.DataFrame:
        """Extract cash flows for valuation."""
        cash_flows = []
        
        for result in self.results['statements']:
            cf = result['cash_flows']
            cash_flows.append({
                'period': result['period'],
                'fcf': cf['fcf'],
                'ts': cf['ts'],
                'ccf': cf['ccf'],
                'cfe': cf['cfe'],
                'cfd': cf['cfd']
            })
        
        return pd.DataFrame(cash_flows)
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert all results to DataFrame."""
        all_data = []
        
        for result in self.results['statements']:
            period = result['period']
            statements = result['statements']
            cash_flows = result['cash_flows']
            
            row = {'period': period}
            
            # Add balance sheet items
            for key, value in statements['balance_sheet'].items():
                row[f'bs_{key}'] = value
            
            # Add income statement items
            for key, value in statements['income_statement'].items():
                row[f'is_{key}'] = value
            
            # Add cash budget items
            for key, value in statements['cash_budget'].items():
                row[f'cb_{key}'] = value
            
            # Add cash flows
            for key, value in cash_flows.items():
                row[f'cf_{key}'] = value
            
            all_data.append(row)
        
        return pd.DataFrame(all_data)
    
    def export_to_excel(self, filename: str):
        """Export model results to Excel."""
        if not self._is_built:
            raise ValueError("Must build model before export")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = self._results_to_dataframe()
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual statement sheets
            self._export_statements_to_excel(writer)
            
            # Valuation sheet
            try:
                valuation = self.calculate_valuation()
                valuation_df = pd.DataFrame([valuation])
                valuation_df.to_excel(writer, sheet_name='Valuation', index=False)
            except Exception as e:
                print(f"Warning: Could not calculate valuation: {e}")
    
    def _export_statements_to_excel(self, writer):
        """Export individual statements to Excel sheets."""
        # Balance Sheets
        bs_data = []
        for result in self.results['statements']:
            row = {'period': result['period']}
            row.update(result['statements']['balance_sheet'])
            bs_data.append(row)
        pd.DataFrame(bs_data).to_excel(
            writer, sheet_name='Balance Sheets', index=False
        )
        
        # Income Statements
        is_data = []
        for result in self.results['statements']:
            row = {'period': result['period']}
            row.update(result['statements']['income_statement'])
            is_data.append(row)
        pd.DataFrame(is_data).to_excel(
            writer, sheet_name='Income Statements', index=False
        )
        
        # Cash Budgets
        cb_data = []
        for result in self.results['statements']:
            row = {'period': result['period']}
            row.update(result['statements']['cash_budget'])
            cb_data.append(row)
        pd.DataFrame(cb_data).to_excel(
            writer, sheet_name='Cash Budgets', index=False
        )


# Summary of fixes:
"""
KEY CORRECTIONS:
===============

1. Tax Shield Calculation (line ~270):
   OLD: ts = is_stmt['tax_expense'] if is_stmt['interest_expense'] > 0 else 0.0
   NEW: ts = corporate_tax_rate * min(ebit, interest_expense)

2. FCF Calculation (line ~280):
   OLD: Used EBIT * (1-T) without proper NOPLAT calculation
   NEW: Properly calculates NOPLAT = EBIT * (1-T) then adds back depreciation

3. Working Capital (line ~300):
   OLD: return 0.0
   NEW: Properly tracks and calculates ΔNW

4. CFE Calculation (line ~320):
   OLD: Used non-existent keys
   NEW: Uses correct cash budget NCB

5. Valuation Integration (line ~520):
   OLD: Called non-existent methods
   NEW: Properly creates ValuationInputs and calls correct methods
"""