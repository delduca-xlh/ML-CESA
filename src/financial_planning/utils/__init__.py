# src/financial_planning/utils/__init__.py
"""
Utility Functions Module

This module contains utility functions for financial calculations,
data fetching, and document processing.
"""

from .fisher_equation import (
    nominal_to_real,
    real_to_nominal,
    implied_inflation,
    compound_growth_rate,
    present_value,
    future_value,
    effective_annual_rate,
    inflation_adjust_series,
    real_discount_factor
)

from .yahoo_finance_fetcher import YahooFinanceDataFetcher
from .fmp_data_fetcher import FMPDataFetcher

try:
    from .pdf_extractor import PDFFinancialExtractor, FinancialRatioCalculator
except ImportError:
    # PDF extractor may have dependency issues
    PDFFinancialExtractor = None
    FinancialRatioCalculator = None

__all__ = [
    # Fisher equation utilities
    'nominal_to_real',
    'real_to_nominal',
    'implied_inflation',
    'compound_growth_rate',
    'present_value',
    'future_value',
    'effective_annual_rate',
    'inflation_adjust_series',
    'real_discount_factor',
    
    # Data fetchers
    'YahooFinanceDataFetcher',
    'FMPDataFetcher',
    
    # PDF processing
    'PDFFinancialExtractor',
    'FinancialRatioCalculator',
]
