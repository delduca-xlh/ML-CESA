# src/financial_planning/models/intermediate_tables.py
"""
Intermediate Tables

Constructs intermediate tables that feed into the financial statements.
These tables organize forecasts for sales, costs, inventory, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


class IntermediateTables:
    """
    Construct intermediate tables for financial planning.
    
    These tables are built from input data and policies to create
    the detailed forecasts needed for the financial statements.
    """
    
    def __init__(self, parameters):
        """
        Initialize intermediate tables.
        
        Args:
            parameters: ModelParameters object
        """
        self.params = parameters
        self.tables = {}
    
    def build_all_tables(self, forecast_inputs) -> None:
        """
        Build all intermediate tables.
        
        Args:
            forecast_inputs: ForecastInputs object
        """
        self.inputs = forecast_inputs
        
        # Build each table
        self.tables['prices'] = self._build_price_table()
        self.tables['volumes'] = self._build_volume_table()
        self.tables['sales'] = self._build_sales_table()
        self.tables['costs'] = self._build_cost_table()
        self.tables['inventory'] = self._build_inventory_table()
        self.tables['accounts_receivable'] = self._build_ar_table()
        self.tables['accounts_payable'] = self._build_ap_table()
        self.tables['depreciation'] = self._build_depreciation_table()
        self.tables['fixed_assets'] = self._build_fixed_assets_table()
    
    def _build_price_table(self) -> pd.DataFrame:
        """
        Build price forecast table.
        
        Returns:
            DataFrame with price forecasts
        """
        periods = self.params.forecast_periods
        
        data = {
            'period': list(range(1, periods + 1)),
            'selling_price': self.inputs.selling_price[:periods],
            'unit_cost': self.inputs.unit_cost[:periods]
        }
        
        return pd.DataFrame(data)
    
    def _build_volume_table(self) -> pd.DataFrame:
        """
        Build volume forecast table.
        
        Returns:
            DataFrame with volume forecasts
        """
        periods = self.params.forecast_periods
        
        data = {
            'period': list(range(1, periods + 1)),
            'sales_volume': self.inputs.sales_volume_units[:periods]
        }
        
        # Calculate required inventory (based on policy)
        data['ending_inventory_units'] = [
            vol * self.params.inventory_policy_months
            for vol in data['sales_volume']
        ]
        
        # Calculate purchases
        purchases = []
        prev_inventory = self.params.initial_inventory_units
        
        for i, (sales, ending_inv) in enumerate(
            zip(data['sales_volume'], data['ending_inventory_units'])
        ):
            purchase = sales + ending_inv - prev_inventory
            purchases.append(purchase)
            prev_inventory = ending_inv
        
        data['purchases_units'] = purchases
        
        return pd.DataFrame(data)
    
    def _build_sales_table(self) -> pd.DataFrame:
        """
        Build sales collection table.
        
        Returns:
            DataFrame with sales collection timing
        """
        periods = self.params.forecast_periods
        sales_revenue = self.inputs.sales_revenue[:periods]
        
        # Calculate sales collection timing
        current_year_collection = [
            sales * (1 - self.params.accounts_receivable_percent -
                    self.params.advance_payment_from_customers_percent)
            for sales in sales_revenue
        ]
        
        ar_collection = [0.0] + [
            sales_revenue[i] * self.params.accounts_receivable_percent
            for i in range(periods - 1)
        ]
        
        advance_collection = [
            sales_revenue[i+1] * self.params.advance_payment_from_customers_percent
            if i < periods - 1 else 0.0
            for i in range(periods)
        ]
        
        total_inflows = [
            curr + ar + adv
            for curr, ar, adv in zip(
                current_year_collection,
                ar_collection,
                advance_collection
            )
        ]
        
        return pd.DataFrame({
            'period': list(range(1, periods + 1)),
            'sales_revenue': sales_revenue,
            'current_year_collection': current_year_collection,
            'ar_collection': ar_collection,
            'advance_collection': advance_collection,
            'total_sales_inflows': total_inflows
        })
    
    def _build_cost_table(self) -> pd.DataFrame:
        """
        Build cost and expense table.
        
        Returns:
            DataFrame with cost forecasts
        """
        periods = self.params.forecast_periods
        
        return pd.DataFrame({
            'period': list(range(1, periods + 1)),
            'cogs': self.inputs.cost_of_goods_sold[:periods],
            'overhead': self.inputs.overhead_expenses[:periods],
            'payroll': self.inputs.payroll_expenses[:periods]
        })
    
    def _build_inventory_table(self) -> pd.DataFrame:
        """
        Build inventory valuation table.
        
        Returns:
            DataFrame with inventory values
        """
        volume_table = self.tables['volumes']
        price_table = self.tables['prices']
        
        # FIFO inventory valuation
        inventory_values = [
            units * cost
            for units, cost in zip(
                volume_table['ending_inventory_units'],
                price_table['unit_cost']
            )
        ]
        
        return pd.DataFrame({
            'period': volume_table['period'],
            'ending_inventory_units': volume_table['ending_inventory_units'],
            'unit_cost': price_table['unit_cost'],
            'ending_inventory_value': inventory_values
        })
    
    def _build_ar_table(self) -> pd.DataFrame:
        """
        Build accounts receivable table.
        
        Returns:
            DataFrame with AR balances
        """
        sales_table = self.tables['sales']
        
        ar_balances = [
            sales * self.params.accounts_receivable_percent
            for sales in sales_table['sales_revenue']
        ]
        
        return pd.DataFrame({
            'period': sales_table['period'],
            'accounts_receivable': ar_balances
        })
    
    def _build_ap_table(self) -> pd.DataFrame:
        """
        Build accounts payable table.
        
        Returns:
            DataFrame with AP balances
        """
        periods = self.params.forecast_periods
        cogs = self.inputs.cost_of_goods_sold[:periods]
        
        # Calculate payment timing
        current_year_payment = [
            cost * (1 - self.params.accounts_payable_percent -
                   self.params.advance_payment_to_suppliers_percent)
            for cost in cogs
        ]
        
        ap_payment = [0.0] + [
            cogs[i] * self.params.accounts_payable_percent
            for i in range(periods - 1)
        ]
        
        advance_payment = [
            cogs[i+1] * self.params.advance_payment_to_suppliers_percent
            if i < periods - 1 else 0.0
            for i in range(periods)
        ]
        
        total_outflows = [
            curr + ap + adv
            for curr, ap, adv in zip(
                current_year_payment,
                ap_payment,
                advance_payment
            )
        ]
        
        ap_balances = [
            cost * self.params.accounts_payable_percent
            for cost in cogs
        ]
        
        return pd.DataFrame({
            'period': list(range(1, periods + 1)),
            'current_year_payment': current_year_payment,
            'ap_payment': ap_payment,
            'advance_payment': advance_payment,
            'total_purchases_outflows': total_outflows,
            'accounts_payable': ap_balances
        })
    
    def _build_depreciation_table(self) -> pd.DataFrame:
        """
        Build depreciation schedule.
        
        Returns:
            DataFrame with depreciation
        """
        periods = self.params.forecast_periods
        
        # Initial asset depreciation
        annual_depreciation_initial = (
            self.params.initial_fixed_assets /
            self.params.depreciation_years
        )
        
        # Track depreciation for each asset vintage
        depreciation_schedule = []
        
        for period in range(1, periods + 1):
            total_depreciation = 0.0
            
            # Initial assets
            if period <= self.params.depreciation_years:
                total_depreciation += annual_depreciation_initial
            
            # New assets from previous periods
            for past_period in range(1, period):
                capex = (
                    self.inputs.capex_forecast[past_period - 1]
                    if past_period - 1 < len(self.inputs.capex_forecast)
                    else 0.0
                )
                
                years_since_purchase = period - past_period
                if years_since_purchase < self.params.depreciation_years:
                    total_depreciation += capex / self.params.depreciation_years
            
            depreciation_schedule.append({
                'period': period,
                'depreciation': total_depreciation
            })
        
        return pd.DataFrame(depreciation_schedule)
    
    def _build_fixed_assets_table(self) -> pd.DataFrame:
        """
        Build net fixed assets table.
        
        Returns:
            DataFrame with net fixed assets
        """
        periods = self.params.forecast_periods
        depreciation_table = self.tables['depreciation']
        
        # Calculate cumulative depreciation
        cumulative_depreciation = []
        cumulative = 0.0
        
        for depr in depreciation_table['depreciation']:
            cumulative += depr
            cumulative_depreciation.append(cumulative)
        
        # Calculate gross fixed assets
        gross_fixed_assets = [self.params.initial_fixed_assets]
        
        for period in range(1, periods):
            capex = (
                self.inputs.capex_forecast[period - 1]
                if period - 1 < len(self.inputs.capex_forecast)
                else 0.0
            )
            gross_fixed_assets.append(gross_fixed_assets[-1] + capex)
        
        # Add last period
        last_capex = (
            self.inputs.capex_forecast[periods - 1]
            if periods - 1 < len(self.inputs.capex_forecast)
            else 0.0
        )
        gross_fixed_assets.append(gross_fixed_assets[-1] + last_capex)
        
        # Net fixed assets
        net_fixed_assets = [
            gross - cum_depr
            for gross, cum_depr in zip(
                gross_fixed_assets[1:],  # Shift by one
                cumulative_depreciation
            )
        ]
        
        return pd.DataFrame({
            'period': list(range(1, periods + 1)),
            'gross_fixed_assets': gross_fixed_assets[1:],
            'cumulative_depreciation': cumulative_depreciation,
            'net_fixed_assets': net_fixed_assets
        })
    
    def get_period_data(self, period: int) -> Dict:
        """
        Get all intermediate data for a specific period.
        
        Args:
            period: Period number (1-indexed)
            
        Returns:
            Dictionary with all intermediate data
        """
        idx = period - 1
        
        return {
            'sales_inflows': self.tables['sales'].iloc[idx]['total_sales_inflows'],
            'purchases_outflows': self.tables['accounts_payable'].iloc[idx]['total_purchases_outflows'],
            'accounts_receivable': self.tables['accounts_receivable'].iloc[idx]['accounts_receivable'],
            'inventory': self.tables['inventory'].iloc[idx]['ending_inventory_value'],
            'accounts_payable': self.tables['accounts_payable'].iloc[idx]['accounts_payable'],
            'depreciation': self.tables['depreciation'].iloc[idx]['depreciation'],
            'net_fixed_assets': self.tables['fixed_assets'].iloc[idx]['net_fixed_assets']
        }
    
    def to_excel(self, filename: str):
        """
        Export all intermediate tables to Excel.
        
        Args:
            filename: Output filename
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for table_name, table_df in self.tables.items():
                table_df.to_excel(
                    writer,
                    sheet_name=table_name.replace('_', ' ').title(),
                    index=False
                )