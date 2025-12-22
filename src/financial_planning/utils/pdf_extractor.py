"""
Financial Statement Extraction from PDFs and Annual Reports

This module handles extraction of financial statements from various sources
including PDFs, using both traditional parsing and LLM-based methods.

Addresses JP Morgan Interview Question Part 2.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import pdfplumber
import requests
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FinancialStatementData:
    """Extracted financial statement data."""
    income_statement: Dict[str, float]
    balance_sheet: Dict[str, float]
    cash_flow_statement: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class PDFFinancialExtractor:
    """
    Extract financial statements from PDF annual reports.
    
    Supports multiple extraction methods:
    1. Rule-based parsing with pdfplumber
    2. LLM-based extraction
    3. Hybrid approach
    """
    
    def __init__(self, use_llm: bool = False, llm_model: str = "gpt-4o"):
        """
        Initialize extractor.
        
        Args:
            use_llm: Whether to use LLM for extraction
            llm_model: Which LLM model to use
        """
        self.use_llm = use_llm
        self.llm_model = llm_model
    
    def extract_from_pdf(
        self,
        pdf_path: str,
        company_name: Optional[str] = None
    ) -> FinancialStatementData:
        """
        Extract financial statements from PDF.
        
        Args:
            pdf_path: Path to PDF file
            company_name: Company name (for validation)
            
        Returns:
            FinancialStatementData object
        """
        if self.use_llm:
            return self._extract_with_llm(pdf_path, company_name)
        else:
            return self._extract_with_rules(pdf_path, company_name)
    
    def _extract_with_rules(
        self,
        pdf_path: str,
        company_name: Optional[str]
    ) -> FinancialStatementData:
        """Extract using rule-based parsing."""
        income_statement = {}
        balance_sheet = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text from all pages
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"
            
            # Find income statement section
            is_section = self._find_income_statement_section(full_text)
            if is_section:
                income_statement = self._parse_income_statement(is_section)
            
            # Find balance sheet section
            bs_section = self._find_balance_sheet_section(full_text)
            if bs_section:
                balance_sheet = self._parse_balance_sheet(bs_section)
            
            # Extract tables
            tables = []
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)
            
            # Try to match tables to statements
            if not income_statement and tables:
                income_statement = self._parse_income_statement_from_tables(tables)
            
            if not balance_sheet and tables:
                balance_sheet = self._parse_balance_sheet_from_tables(tables)
        
        return FinancialStatementData(
            income_statement=income_statement,
            balance_sheet=balance_sheet,
            metadata={'extraction_method': 'rules', 'company': company_name}
        )
    
    def _extract_with_llm(
        self,
        pdf_path: str,
        company_name: Optional[str]
    ) -> FinancialStatementData:
        """Extract using LLM."""
        # Extract text from PDF
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages[:50]:  # Limit to first 50 pages
                full_text += page.extract_text() + "\n"
        
        # Use LLM to extract structured data
        income_statement = self._llm_extract_income_statement(full_text)
        balance_sheet = self._llm_extract_balance_sheet(full_text)
        
        return FinancialStatementData(
            income_statement=income_statement,
            balance_sheet=balance_sheet,
            metadata={
                'extraction_method': 'llm',
                'model': self.llm_model,
                'company': company_name
            }
        )
    
    def _find_income_statement_section(self, text: str) -> Optional[str]:
        """Find income statement section in text."""
        patterns = [
            r'income statement.*?(?=balance sheet|statement of financial|page \d+)',
            r'statement of operations.*?(?=balance sheet|statement of financial|page \d+)',
            r'consolidated statement of income.*?(?=balance sheet|statement of financial|page \d+)',
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _find_balance_sheet_section(self, text: str) -> Optional[str]:
        """Find balance sheet section in text."""
        patterns = [
            r'balance sheet.*?(?=statement of cash|notes to|page \d+)',
            r'statement of financial position.*?(?=statement of cash|notes to|page \d+)',
            r'consolidated balance sheet.*?(?=statement of cash|notes to|page \d+)',
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _parse_income_statement(self, text: str) -> Dict[str, float]:
        """Parse income statement from text."""
        result = {}
        
        # Common line items and their patterns
        line_items = {
            'revenue': [r'revenue', r'total revenue', r'sales', r'net sales'],
            'cost_of_revenue': [r'cost of revenue', r'cost of sales', r'cost of goods sold'],
            'gross_profit': [r'gross profit', r'gross income'],
            'operating_expenses': [r'operating expenses', r'total operating expenses'],
            'operating_income': [r'operating income', r'income from operations', r'ebit'],
            'interest_expense': [r'interest expense', r'interest paid'],
            'income_before_tax': [r'income before tax', r'pretax income', r'ebt'],
            'tax_expense': [r'income tax', r'tax expense', r'provision for income tax'],
            'net_income': [r'net income', r'net earnings', r'net profit'],
        }
        
        for key, patterns in line_items.items():
            value = self._extract_financial_value(text, patterns)
            if value is not None:
                result[key] = value
        
        return result
    
    def _parse_balance_sheet(self, text: str) -> Dict[str, float]:
        """Parse balance sheet from text."""
        result = {}
        
        line_items = {
            'cash': [r'cash and cash equivalents', r'cash'],
            'accounts_receivable': [r'accounts receivable', r'trade receivables'],
            'inventory': [r'inventory', r'inventories'],
            'current_assets': [r'total current assets'],
            'ppe': [r'property, plant and equipment', r'ppe', r'fixed assets'],
            'total_assets': [r'total assets'],
            'accounts_payable': [r'accounts payable', r'trade payables'],
            'short_term_debt': [r'short-term debt', r'current portion of long-term debt'],
            'current_liabilities': [r'total current liabilities'],
            'long_term_debt': [r'long-term debt', r'long term debt'],
            'total_liabilities': [r'total liabilities'],
            'shareholders_equity': [r'shareholders\' equity', r'stockholders\' equity', r'total equity'],
        }
        
        for key, patterns in line_items.items():
            value = self._extract_financial_value(text, patterns)
            if value is not None:
                result[key] = value
        
        return result
    
    def _extract_financial_value(
        self,
        text: str,
        patterns: List[str]
    ) -> Optional[float]:
        """Extract a financial value from text given patterns."""
        for pattern in patterns:
            # Look for the pattern followed by numbers
            regex = rf'{pattern}[:\s]+[\$]?([\d,]+\.?\d*)'
            match = re.search(regex, text, re.IGNORECASE)
            
            if match:
                value_str = match.group(1).replace(',', '')
                try:
                    return float(value_str)
                except ValueError:
                    continue
        
        return None
    
    def _parse_income_statement_from_tables(
        self,
        tables: List[List[List[str]]]
    ) -> Dict[str, float]:
        """Parse income statement from extracted tables."""
        result = {}
        
        for table in tables:
            # Check if this looks like an income statement
            if self._is_income_statement_table(table):
                result = self._extract_values_from_table(
                    table,
                    ['revenue', 'cost', 'gross', 'operating', 'net income']
                )
                if result:
                    break
        
        return result
    
    def _parse_balance_sheet_from_tables(
        self,
        tables: List[List[List[str]]]
    ) -> Dict[str, float]:
        """Parse balance sheet from extracted tables."""
        result = {}
        
        for table in tables:
            # Check if this looks like a balance sheet
            if self._is_balance_sheet_table(table):
                result = self._extract_values_from_table(
                    table,
                    ['assets', 'liabilities', 'equity', 'cash', 'debt']
                )
                if result:
                    break
        
        return result
    
    def _is_income_statement_table(self, table: List[List[str]]) -> bool:
        """Check if table is likely an income statement."""
        table_text = ' '.join([' '.join(row) for row in table if row]).lower()
        keywords = ['revenue', 'income', 'expense', 'earnings']
        return sum(keyword in table_text for keyword in keywords) >= 2
    
    def _is_balance_sheet_table(self, table: List[List[str]]) -> bool:
        """Check if table is likely a balance sheet."""
        table_text = ' '.join([' '.join(row) for row in table if row]).lower()
        keywords = ['assets', 'liabilities', 'equity', 'balance']
        return sum(keyword in table_text for keyword in keywords) >= 2
    
    def _extract_values_from_table(
        self,
        table: List[List[str]],
        keywords: List[str]
    ) -> Dict[str, float]:
        """Extract numerical values from table based on keywords."""
        result = {}
        
        for row in table:
            if not row or len(row) < 2:
                continue
            
            label = row[0].lower() if row[0] else ""
            
            # Check if this row matches any keyword
            for keyword in keywords:
                if keyword in label:
                    # Extract numerical value
                    for cell in row[1:]:
                        if cell:
                            value = self._parse_number(cell)
                            if value is not None:
                                result[keyword] = value
                                break
        
        return result
    
    def _parse_number(self, text: str) -> Optional[float]:
        """Parse a number from text, handling various formats."""
        # Remove common formatting
        text = text.replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
        text = text.strip()
        
        try:
            return float(text)
        except ValueError:
            return None
    
    def _llm_extract_income_statement(self, text: str) -> Dict[str, float]:
        """Use LLM to extract income statement."""
        # This would call an LLM API (OpenAI, Claude, etc.)
        # Placeholder implementation
        prompt = f"""
        Extract the income statement from the following annual report text.
        Return a JSON object with these fields:
        - revenue
        - cost_of_revenue
        - gross_profit
        - operating_expenses
        - operating_income
        - interest_expense
        - income_before_tax
        - tax_expense
        - net_income
        
        Text:
        {text[:10000]}  # Limit text length
        
        Return ONLY valid JSON, no other text.
        """
        
        # Call LLM (implementation depends on which service you use)
        # For now, return empty dict
        return {}
    
    def _llm_extract_balance_sheet(self, text: str) -> Dict[str, float]:
        """Use LLM to extract balance sheet."""
        prompt = f"""
        Extract the balance sheet from the following annual report text.
        Return a JSON object with these fields:
        - cash
        - accounts_receivable
        - inventory
        - current_assets
        - ppe
        - total_assets
        - accounts_payable
        - short_term_debt
        - current_liabilities
        - long_term_debt
        - total_liabilities
        - shareholders_equity
        
        Text:
        {text[:10000]}
        
        Return ONLY valid JSON, no other text.
        """
        
        # Call LLM
        return {}


class FinancialRatioCalculator:
    """Calculate financial ratios from statements."""
    
    @staticmethod
    def calculate_all_ratios(
        income_statement: Dict[str, float],
        balance_sheet: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate all standard financial ratios."""
        ratios = {}
        
        # Profitability ratios
        if 'revenue' in income_statement and income_statement['revenue'] > 0:
            ratios['gross_margin'] = (
                income_statement.get('gross_profit', 0) / income_statement['revenue']
            )
            ratios['net_margin'] = (
                income_statement.get('net_income', 0) / income_statement['revenue']
            )
            ratios['cost_to_income'] = (
                income_statement.get('cost_of_revenue', 0) / income_statement['revenue']
            )
        
        # Liquidity ratios
        if 'current_liabilities' in balance_sheet and balance_sheet['current_liabilities'] > 0:
            ratios['current_ratio'] = (
                balance_sheet.get('current_assets', 0) / balance_sheet['current_liabilities']
            )
            
            quick_assets = (
                balance_sheet.get('current_assets', 0) -
                balance_sheet.get('inventory', 0)
            )
            ratios['quick_ratio'] = quick_assets / balance_sheet['current_liabilities']
        
        # Leverage ratios
        total_debt = (
            balance_sheet.get('short_term_debt', 0) +
            balance_sheet.get('long_term_debt', 0)
        )
        
        if 'shareholders_equity' in balance_sheet and balance_sheet['shareholders_equity'] > 0:
            ratios['debt_to_equity'] = total_debt / balance_sheet['shareholders_equity']
        
        if 'total_assets' in balance_sheet and balance_sheet['total_assets'] > 0:
            ratios['debt_to_assets'] = total_debt / balance_sheet['total_assets']
        
        total_capital = total_debt + balance_sheet.get('shareholders_equity', 0)
        if total_capital > 0:
            ratios['debt_to_capital'] = total_debt / total_capital
        
        # Calculate EBITDA for debt-to-EBITDA
        ebitda = (
            income_statement.get('operating_income', 0) +
            income_statement.get('depreciation', 0) +
            income_statement.get('amortization', 0)
        )
        if ebitda > 0:
            ratios['debt_to_ebitda'] = total_debt / ebitda
        
        # Coverage ratios
        if 'interest_expense' in income_statement and income_statement['interest_expense'] > 0:
            ratios['interest_coverage'] = (
                income_statement.get('operating_income', 0) /
                income_statement['interest_expense']
            )
        
        return ratios


def download_annual_report(url: str, save_path: str) -> bool:
    """Download annual report PDF from URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading report: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    print("Financial Statement Extractor")
    print("=" * 60)
    
    # Example: Extract from GM annual report
    gm_url = "https://investor.gm.com/static-files/1fff6f59-551f-4fe0-bca9-74bfc9a56aeb"
    
    print(f"\nDownloading GM annual report...")
    if download_annual_report(gm_url, "gm_annual_report.pdf"):
        print("✓ Downloaded successfully")
        
        print("\nExtracting financial statements...")
        extractor = PDFFinancialExtractor(use_llm=False)
        data = extractor.extract_from_pdf("gm_annual_report.pdf", "General Motors")
        
        print("\nIncome Statement:")
        for key, value in data.income_statement.items():
            print(f"  {key}: ${value:,.2f}")
        
        print("\nBalance Sheet:")
        for key, value in data.balance_sheet.items():
            print(f"  {key}: ${value:,.2f}")
        
        print("\nCalculating ratios...")
        calc = FinancialRatioCalculator()
        ratios = calc.calculate_all_ratios(
            data.income_statement,
            data.balance_sheet
        )
        
        print("\nFinancial Ratios:")
        for key, value in ratios.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2%}" if value < 10 else f"  {key}: {value:.2f}")
