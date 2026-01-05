#!/usr/bin/env python3
"""
pdf_extractor.py - Extract Financial Statements from PDF Annual Reports

This module extracts income statements and balance sheets from PDF files
and converts them to a format compatible with the ML forecasting pipeline.

Supports:
- GM, LVMH, Microsoft, Google, JPMorgan, Exxon, Tencent, Alibaba, Volkswagen
- Uses Claude API for intelligent table extraction
- Handles different report formats and languages

Usage:
    from financial_planning.utils.pdf_extractor import PDFExtractor
    
    extractor = PDFExtractor()
    data = extractor.extract_from_pdf("annual_report.pdf")
    
    # Or from URL
    data = extractor.extract_from_url("https://...")

Requirements:
    pip install pymupdf anthropic pandas
"""

import os
import re
import json
import base64
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
import requests
import tempfile
import sys

# PDF libraries - use what's available
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

# API Key
ANTHROPIC_API_KEY = os.environ.get(
    "ANTHROPIC_API_KEY",
    "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"
)

# Model version for reproducibility
MODEL_VERSION = "claude-sonnet-4-20250514"

# Global stderr suppression for PDF color warnings
_original_stderr = sys.stderr
_devnull = open(os.devnull, 'w')

def suppress_pdf_warnings():
    sys.stderr = _devnull

# Auto-suppress on import
suppress_pdf_warnings()

@dataclass
class FinancialData:
    """Standardized financial data structure."""
    # Income Statement
    revenue: float = 0.0
    cost_of_goods_sold: float = 0.0
    gross_profit: float = 0.0
    operating_expenses: float = 0.0
    operating_income: float = 0.0  # EBIT
    interest_expense: float = 0.0
    income_before_tax: float = 0.0
    income_tax: float = 0.0
    net_income: float = 0.0
    
    # Balance Sheet - Assets
    cash: float = 0.0
    accounts_receivable: float = 0.0
    inventory: float = 0.0
    current_assets: float = 0.0
    ppe: float = 0.0  # Property, Plant & Equipment
    total_assets: float = 0.0
    
    # Balance Sheet - Liabilities
    accounts_payable: float = 0.0
    short_term_debt: float = 0.0
    current_liabilities: float = 0.0
    long_term_debt: float = 0.0
    total_liabilities: float = 0.0
    
    # Balance Sheet - Equity
    total_equity: float = 0.0
    retained_earnings: float = 0.0
    
    # Additional
    ebitda: float = 0.0
    depreciation: float = 0.0
    shares_outstanding: float = 0.0
    
    # Metadata
    company: str = ""
    period: str = ""
    currency: str = "USD"
    unit: str = "millions"
    source_pages: List[int] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def calculate_ratios(self) -> dict:
        """Calculate key financial ratios."""
        ratios = {}
        
        # Profitability
        if self.revenue > 0:
            ratios['gross_margin'] = self.gross_profit / self.revenue
            ratios['operating_margin'] = self.operating_income / self.revenue
            ratios['net_margin'] = self.net_income / self.revenue
            ratios['cost_to_income_ratio'] = (self.cost_of_goods_sold + self.operating_expenses) / self.revenue
        
        # Liquidity
        if self.current_liabilities > 0:
            ratios['current_ratio'] = self.current_assets / self.current_liabilities
            quick_assets = self.current_assets - self.inventory
            ratios['quick_ratio'] = quick_assets / self.current_liabilities
        
        # Leverage
        total_debt = self.short_term_debt + self.long_term_debt
        if self.total_equity > 0:
            ratios['debt_to_equity'] = total_debt / self.total_equity
        if self.total_assets > 0:
            ratios['debt_to_assets'] = total_debt / self.total_assets
        total_capital = total_debt + self.total_equity
        if total_capital > 0:
            ratios['debt_to_capital'] = total_debt / total_capital
        if self.ebitda > 0:
            ratios['debt_to_ebitda'] = total_debt / self.ebitda
        
        # Coverage
        if self.interest_expense > 0:
            ratios['interest_coverage'] = self.operating_income / self.interest_expense
        
        return ratios


class PDFExtractor:
    """Extract financial statements from PDF annual reports."""
    
    def __init__(self, api_key: str = None, verbose: bool = True):
        self.api_key = api_key or ANTHROPIC_API_KEY
        self.verbose = verbose
        self.model_version = MODEL_VERSION
        self._last_response = None  # For debugging
        
    def log(self, msg: str):
        """Print log message if verbose."""
        if self.verbose:
            print(msg)
    
    def download_pdf(self, url: str, save_path: str = None) -> str:
        """Download PDF from URL."""
        self.log(f"  Downloading PDF from URL...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        
        if save_path is None:
            # Create temp file
            fd, save_path = tempfile.mkstemp(suffix='.pdf')
            os.close(fd)
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        self.log(f"  ✓ Downloaded to: {save_path}")
        return save_path
    
    def extract_pages_as_images(self, pdf_path: str, pages: List[int] = None, 
                                 dpi: int = 150) -> List[Tuple[int, bytes]]:
        """Extract specific pages as images."""
        images = []
        
        if HAS_FITZ:
            # Use PyMuPDF (fastest)
            doc = fitz.open(pdf_path)
            if pages is None:
                pages = list(range(len(doc)))
            
            for page_num in pages:
                if page_num < len(doc):
                    page = doc[page_num]
                    mat = fitz.Matrix(dpi/72, dpi/72)
                    pix = page.get_pixmap(matrix=mat)
                    img_bytes = pix.tobytes("png")
                    images.append((page_num, img_bytes))
            doc.close()
            
        elif HAS_PDF2IMAGE:
            # Use pdf2image (requires poppler)
            try:
                all_images = convert_from_path(pdf_path, dpi=dpi)
                if pages is None:
                    pages = list(range(len(all_images)))
                
                for page_num in pages:
                    if page_num < len(all_images):
                        img = all_images[page_num]
                        # Convert PIL Image to bytes
                        import io
                        buf = io.BytesIO()
                        img.save(buf, format='PNG')
                        images.append((page_num, buf.getvalue()))
            except Exception as e:
                self.log(f"  pdf2image failed: {e}, falling back to pdfplumber")
                # Fall through to pdfplumber
        
        if not images and HAS_PDFPLUMBER:
            # Use pdfplumber (extract text instead of images)
            self.log("  Using pdfplumber text extraction (no image support)")
            # This will be handled differently - return empty and use text extraction
            pass
        
        return images
    
    def extract_text_from_pages(self, pdf_path: str, pages: List[int] = None) -> Dict[int, str]:
        """Extract text from specific pages."""
        texts = {}
        
        if HAS_PDFPLUMBER:
            with pdfplumber.open(pdf_path) as pdf:
                if pages is None:
                    pages = list(range(len(pdf.pages)))
                
                for page_num in pages:
                    if page_num < len(pdf.pages):
                        page = pdf.pages[page_num]
                        texts[page_num] = page.extract_text() or ""
        elif HAS_FITZ:
            doc = fitz.open(pdf_path)
            if pages is None:
                pages = list(range(len(doc)))
            
            for page_num in pages:
                if page_num < len(doc):
                    page = doc[page_num]
                    texts[page_num] = page.get_text()
            doc.close()
        
        return texts
    
    def extract_tables_from_pages(self, pdf_path: str, pages: List[int] = None) -> Dict[int, List]:
        """Extract tables from specific pages using pdfplumber."""
        tables = {}
        
        if HAS_PDFPLUMBER:
            with pdfplumber.open(pdf_path) as pdf:
                if pages is None:
                    pages = list(range(len(pdf.pages)))
                
                for page_num in pages:
                    if page_num < len(pdf.pages):
                        page = pdf.pages[page_num]
                        page_tables = page.extract_tables()
                        if page_tables:
                            tables[page_num] = page_tables
        
        return tables
    
    def extract_auto(self, pdf_path: str) -> FinancialData:
        """
        Automatically find and extract financial data from a PDF.
        
        Uses LLM to find the correct pages, then extracts data.
        """
        self.log(f"\n{'='*60}")
        self.log(f"PDF AUTO EXTRACTOR")
        self.log(f"{'='*60}")
        self.log(f"File: {pdf_path}")
        
        page_count = self._get_page_count(pdf_path)
        self.log(f"  Total pages: {page_count}")
        
        if page_count == 0:
            self.log("  ⚠ Could not read PDF")
            return FinancialData()
        
        # Step 1: Find financial statement pages
        self.log(f"  Step 1: Finding financial statement pages...")
        
        # For large PDFs (>50 pages), use text-based LLM search first
        if page_count > 50:
            indices = self._find_pages_via_llm_text(pdf_path)
            if indices:
                self.log(f"  Found via LLM text search: {indices}")
                return self.extract_from_pdf(pdf_path, income_pages=indices, balance_pages=indices)
        
        # For smaller PDFs, use vision on first pages
        num_toc_pages = min(10, page_count)
        toc_pages = list(range(num_toc_pages))
        page_images_raw = self.extract_pages_as_images(pdf_path, toc_pages)
        
        if not page_images_raw:
            self.log("  ⚠ Could not extract page images")
            return FinancialData()
        
        # Convert bytes to base64 strings
        page_images = []
        for page_num, img_bytes in page_images_raw:
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            page_images.append(img_b64)
        
        # Ask LLM to find pages - use PDF INDEX (0-based)
        find_prompt = """Look at these pages from an annual report. Find the table of contents (INDEX).

I need the PDF PAGE INDEX (0-based) for:
1. Consolidated Statements of Income (or Income Statement / P&L / Statement of Operations)
2. Consolidated Balance Sheets (or Statement of Financial Position)

For BANKS: Look for "Consolidated statements of income" and "Consolidated balance sheets"
For 10-K filings: Look in "Financial Statements and Supplementary Data" section

To calculate PDF INDEX from the printed page number in TOC:
1. Look at ANY page footer in these images that shows a page number
2. Note which PDF page index you're on (index 0, 1, 2, 3...)
3. Calculate: offset = pdf_index - footer_number
4. Then: target_pdf_index = toc_printed_page + offset

Example: If PDF index 3 shows footer "2", offset = 3 - 2 = 1
Then for TOC page 150: pdf_index = 150 + 1 = 151

IMPORTANT: If the TOC shows "Financial Section" or "Financial Statements" without specific statement pages, find the PAGE NUMBER where the Financial Section starts and add typical offsets:
- Income Statement is usually first (offset +0 to +2)
- Balance Sheet usually follows (offset +2 to +4)

If you can NOT find a table of contents or page numbers:
- Return 0 for both indices
- I will use keyword search instead

RESPOND WITH JSON:
{
    "income_statement_index": <calculated PDF index>,
    "balance_sheet_index": <calculated PDF index>,
    "toc_income_page": <printed page from TOC, or 0 if not found>,
    "toc_balance_page": <printed page from TOC, or 0 if not found>,
    "financial_section_start": <page number where Financial Section starts, if applicable>,
    "offset": <calculated offset, or null if unknown>,
    "confidence": "high" or "low"
}"""

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            content = [{"type": "text", "text": find_prompt}]
            for img_data in page_images:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_data
                    }
                })
            
            response = client.messages.create(
                model=self.model_version,
                max_tokens=500,
                messages=[{"role": "user", "content": content}]
            )
            
            response_text = response.content[0].text
            self.log(f"  LLM response: {response_text[:300]}...")
            
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                income_idx = result.get('income_statement_index', 0)
                balance_idx = result.get('balance_sheet_index', 0)
                
                # Also try to get TOC page numbers as fallback
                toc_income = result.get('toc_income_page', 0)
                toc_balance = result.get('toc_balance_page', 0)
                
                # Validate indices are numbers
                def to_int(val, fallback=0):
                    if isinstance(val, int):
                        return val
                    if isinstance(val, str):
                        # Try to extract number from string
                        nums = re.findall(r'\d+', str(val))
                        if nums:
                            return int(nums[0])
                    return fallback
                
                income_idx = to_int(income_idx)
                balance_idx = to_int(balance_idx)
                toc_income = to_int(toc_income)
                toc_balance = to_int(toc_balance)
                financial_section_start = to_int(result.get('financial_section_start', 0))
                
                self.log(f"  DEBUG: income_idx={income_idx}, balance_idx={balance_idx}")
                self.log(f"  DEBUG: toc_income={toc_income}, toc_balance={toc_balance}")
                if financial_section_start > 0:
                    self.log(f"  DEBUG: financial_section_start={financial_section_start}")
                
                # If direct index failed but we have TOC pages, use simple offset
                # Most PDFs: index = printed_page - 1 (if no cover) or printed_page (if cover)
                if income_idx == 0 and toc_income > 0:
                    # Try printed_page - 1 as default
                    income_idx = toc_income - 1
                    self.log(f"  Using fallback: income_idx = toc_page({toc_income}) - 1 = {income_idx}")
                
                if balance_idx == 0 and toc_balance > 0:
                    balance_idx = toc_balance - 1
                    self.log(f"  Using fallback: balance_idx = toc_page({toc_balance}) - 1 = {balance_idx}")
                
                # If we only have financial_section_start, use that as base
                if income_idx == 0 and balance_idx == 0 and financial_section_start > 0:
                    # In 10-K filings, actual statements are typically 40-50 pages after start
                    # Try searching from financial_section_start
                    self.log(f"  Using financial_section_start ({financial_section_start}) to search...")
                    
                    # Use keyword search starting from financial section
                    indices = self._find_statement_indices_by_keyword(pdf_path)
                    if indices:
                        self.log(f"  Found via keyword search: {indices}")
                        return self.extract_from_pdf(pdf_path, income_pages=indices, balance_pages=indices)
                
                self.log(f"  Found: Income Statement at index {income_idx}, Balance Sheet at index {balance_idx}")
                
                if income_idx > 0 or balance_idx > 0:
                    indices = []
                    if income_idx > 0:
                        indices.append(income_idx)
                    if balance_idx > 0 and balance_idx != income_idx:
                        indices.append(balance_idx)
                    
                    indices = sorted(indices)
                    self.log(f"  Step 2: Extracting data from pages {indices}...")
                    
                    # Try extraction
                    data = self.extract_from_pdf(pdf_path, income_pages=indices, balance_pages=indices)
                    
                    # Check if extraction was successful (got actual data)
                    if data.revenue > 0 or data.total_assets > 0:
                        # Check if we got enough data (not just summary page)
                        zero_count = sum([
                            1 for v in [data.cogs, data.operating_income, data.total_liabilities, 
                                       data.cash, data.accounts_receivable]
                            if v == 0
                        ])
                        
                        # If most values are 0, we might have a summary page - try keyword search
                        if zero_count >= 4 and financial_section_start > 0:
                            self.log(f"  ⚠ Got summary page (many zeros), searching for complete statements...")
                            search_indices = self._find_statement_indices_by_keyword(pdf_path)
                            if search_indices and search_indices != indices:
                                self.log(f"  Found complete statements at: {search_indices}")
                                better_data = self.extract_from_pdf(pdf_path, income_pages=search_indices, balance_pages=search_indices)
                                # Check if the new extraction is better
                                new_zero_count = sum([
                                    1 for v in [better_data.cogs, better_data.operating_income, 
                                               better_data.total_liabilities, better_data.cash, 
                                               better_data.accounts_receivable]
                                    if v == 0
                                ])
                                if new_zero_count < zero_count:
                                    return better_data
                        
                        return data
                    
                    # If failed and we have TOC pages, try different offsets
                    confidence = result.get('confidence', 'low')
                    if confidence == 'low' or (data.revenue == 0 and data.total_assets == 0):
                        self.log(f"  ⚠ First attempt failed, trying different offsets...")
                        
                        # Common offsets to try: 0, +2, +4, +5 (for various PDF structures)
                        for offset_adj in [0, 2, 4, 5, 3, 6]:
                            if toc_income > 0:
                                test_income = toc_income + offset_adj
                                test_balance = toc_balance + offset_adj if toc_balance > 0 else test_income + 1
                                
                                if test_income != income_idx:  # Don't retry same pages
                                    test_indices = sorted(set([test_income, test_balance]))
                                    self.log(f"    Trying offset +{offset_adj}: pages {test_indices}")
                                    
                                    test_data = self.extract_from_pdf(pdf_path, income_pages=test_indices, balance_pages=test_indices)
                                    
                                    if test_data.revenue > 0 or test_data.total_assets > 0:
                                        self.log(f"    ✓ Success with offset +{offset_adj}")
                                        return test_data
                        
                        self.log(f"  ⚠ All offset attempts failed")
                    
                    return data
                    
        except Exception as e:
            self.log(f"  ⚠ LLM error: {e}")
            import traceback
            self.log(traceback.format_exc())
        
        # Fallback: use keyword search if LLM didn't find pages
        self.log("  ⚠ LLM could not find TOC, trying keyword search...")
        indices = self._find_statement_indices_by_keyword(pdf_path)
        
        if indices:
            self.log(f"  Found pages via keyword search: {indices}")
            return self.extract_from_pdf(pdf_path, income_pages=indices, balance_pages=indices)
        
        self.log("  ⚠ Could not find pages automatically")
        return FinancialData()
    
    def _find_pages_via_llm_text(self, pdf_path: str) -> List[int]:
        """
        For large PDFs, extract text from all pages and ask LLM to find the
        financial statement pages.
        """
        self.log("  Using LLM text search for large PDF...")
        
        if not HAS_PDFPLUMBER and not HAS_FITZ:
            return []
        
        # Patterns for income statement
        income_patterns = [
            'CONSOLIDATED STATEMENTS OF INCOME',
            'CONSOLIDATED STATEMENT OF INCOME', 
            'CONSOLIDATED INCOME STATEMENTS',
            'CONSOLIDATED INCOME STATEMENT',
            'CONSOLIDATED STATEMENTS OF OPERATIONS',
            'CONSOLIDATED STATEMENT OF OPERATIONS',
            'CONSOLIDATED STATEMENTS OF EARNINGS',
            'CONSOLIDATED STATEMENT OF EARNINGS',
            'INCOME STATEMENTS',  # Microsoft style
            'INCOME STATEMENT',
            'STATEMENTS OF INCOME',
            'STATEMENT OF INCOME',
            'STATEMENTS OF OPERATIONS',
            'STATEMENT OF OPERATIONS',
            'STATEMENT OF EARNINGS',
            'STATEMENTS OF EARNINGS',
        ]
        
        # Patterns for balance sheet
        balance_patterns = [
            'CONSOLIDATED BALANCE SHEETS',
            'CONSOLIDATED BALANCE SHEET',
            'CONSOLIDATED STATEMENTS OF FINANCIAL POSITION',
            'CONSOLIDATED STATEMENT OF FINANCIAL POSITION',
            'BALANCE SHEETS',  # Microsoft style
            'BALANCE SHEET',
            'STATEMENTS OF FINANCIAL POSITION',
            'STATEMENT OF FINANCIAL POSITION',
        ]
        
        all_patterns = income_patterns + balance_patterns
        
        # Track candidates by type
        income_candidates = []
        balance_candidates = []
        
        # First, do a quick keyword search to find candidate pages
        candidates = []
        
        if HAS_PDFPLUMBER:
            with pdfplumber.open(pdf_path) as pdf:
                for idx in range(len(pdf.pages)):
                    text = (pdf.pages[idx].extract_text() or "")[:1000].upper()
                    lines = text.split('\n')
                    
                    # Only check first 5 lines for the title (must be page header)
                    header_lines = '\n'.join(lines[:5])
                    # Check more text for content validation
                    first_lines = '\n'.join(lines[:10])
                    
                    # Skip audit reports and notes
                    if 'REPORT OF INDEPENDENT' in header_lines or 'NOTES TO CONSOLIDATED' in header_lines:
                        continue
                    
                    # Check income patterns - title must be in header
                    for pattern in income_patterns:
                        if pattern in header_lines:
                            # Numbers: 339,247 or 1,234,567 or plain 123456
                            has_numbers = bool(re.search(r'\d{1,3}(?:,\d{3})+|\d{4,}', text))
                            # Check it looks like actual statement (has Revenue/Sales)
                            if has_numbers and ('REVENUE' in text or 'NET SALES' in text):
                                income_candidates.append((idx, first_lines[:250]))
                                candidates.append((idx, first_lines[:250]))
                            break
                    
                    # Check balance patterns - title must be in header
                    for pattern in balance_patterns:
                        if pattern in header_lines:
                            # Numbers: 339,247 or 1,234,567 or plain 123456
                            has_numbers = bool(re.search(r'\d{1,3}(?:,\d{3})+|\d{4,}', text))
                            # Check it looks like actual statement (has Assets)
                            if has_numbers and ('ASSETS' in text or 'TOTAL ASSETS' in text):
                                balance_candidates.append((idx, first_lines[:250]))
                                candidates.append((idx, first_lines[:250]))
                            break
        
        elif HAS_FITZ:
            doc = fitz.open(pdf_path)
            for idx in range(len(doc)):
                text = doc[idx].get_text()[:1000].upper()
                lines = text.split('\n')
                header_lines = '\n'.join(lines[:5])
                first_lines = '\n'.join(lines[:10])
                
                if 'REPORT OF INDEPENDENT' in header_lines or 'NOTES TO CONSOLIDATED' in header_lines:
                    continue
                
                for pattern in income_patterns:
                    if pattern in header_lines:
                        has_numbers = bool(re.search(r'\d{1,3}(?:,\d{3})+|\d{4,}', text))
                        if has_numbers and ('REVENUE' in text or 'NET SALES' in text):
                            income_candidates.append((idx, first_lines[:250]))
                            candidates.append((idx, first_lines[:250]))
                        break
                
                for pattern in balance_patterns:
                    if pattern in header_lines:
                        has_numbers = bool(re.search(r'\d{1,3}(?:,\d{3})+|\d{4,}', text))
                        if has_numbers and ('ASSETS' in text or 'TOTAL ASSETS' in text):
                            balance_candidates.append((idx, first_lines[:250]))
                            candidates.append((idx, first_lines[:250]))
                        break
            doc.close()
        
        if not candidates:
            self.log("    No candidate pages found via keyword search")
            return []
        
        self.log(f"    Found {len(candidates)} candidate pages: {[c[0] for c in candidates]}")
        
        # Now ask LLM to identify which are the actual statements
        candidate_text = "\n\n".join([
            f"=== PDF Index {idx} ===\n{text}" 
            for idx, text in candidates[:10]  # Limit to first 10 candidates
        ])
        
        prompt = f"""I found these pages in a PDF that might contain financial statements.
Tell me which PDF INDEX numbers contain:
1. The Consolidated Statements of Income (or Income Statement)
2. The Consolidated Balance Sheets

{candidate_text}

RESPOND ONLY WITH JSON:
{{
    "income_statement_index": <PDF index number>,
    "balance_sheet_index": <PDF index number>
}}

Pick the FIRST page of each statement (the one with the title and data)."""

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model=self.model_version,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            self.log(f"    LLM response: {response_text[:200]}...")
            
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                income_idx = result.get('income_statement_index')
                balance_idx = result.get('balance_sheet_index')
                
                # Convert to int if valid
                income_idx = income_idx if isinstance(income_idx, int) and income_idx > 0 else None
                balance_idx = balance_idx if isinstance(balance_idx, int) and balance_idx > 0 else None
                
                # If we only found one type, look for the other nearby
                if balance_idx and not income_idx:
                    # Income is usually 1-3 pages before Balance Sheet
                    self.log(f"    Only found Balance Sheet at {balance_idx}, checking nearby for Income...")
                    for offset in [-1, -2, -3, 1, 2]:
                        check_idx = balance_idx + offset
                        if check_idx > 0:
                            indices.append(check_idx)
                    indices.append(balance_idx)
                    return sorted(set(indices))[:4]  # Return up to 4 pages
                    
                elif income_idx and not balance_idx:
                    # Balance Sheet is usually 1-3 pages after Income Statement
                    self.log(f"    Only found Income Statement at {income_idx}, checking nearby for Balance...")
                    indices.append(income_idx)
                    for offset in [1, 2, 3, -1]:
                        check_idx = income_idx + offset
                        if check_idx > 0:
                            indices.append(check_idx)
                    return sorted(set(indices))[:4]
                
                indices = []
                if income_idx:
                    indices.append(income_idx)
                if balance_idx and balance_idx != income_idx:
                    indices.append(balance_idx)
                    # Balance Sheet often spans 2-3 pages (Assets, then Equity/Liabilities)
                    indices.append(balance_idx + 1)
                    indices.append(balance_idx + 2)
                
                return sorted(set(indices))
                
        except Exception as e:
            self.log(f"    LLM error: {e}")
        
        # Fallback: return first income + first balance candidate
        indices = []
        if income_candidates:
            indices.append(income_candidates[0][0])
        if balance_candidates:
            idx = balance_candidates[0][0]
            if idx not in indices:
                indices.append(idx)
        
        if indices:
            return sorted(indices)
        
        # Last resort: return first two candidates
        if len(candidates) >= 2:
            return [candidates[0][0], candidates[1][0]]
        elif len(candidates) == 1:
            return [candidates[0][0]]
        
        return []
    
    def _find_statement_indices_by_keyword(self, pdf_path: str) -> List[int]:
        """Find financial statement pages by searching for keywords in page headers.
        
        Supports multi-page statements by including consecutive pages.
        """
        indices = []
        income_start = None
        balance_start = None
        
        # Choose PDF library
        if HAS_PDFPLUMBER:
            return self._find_statements_pdfplumber(pdf_path)
        elif HAS_FITZ:
            return self._find_statements_fitz(pdf_path)
        else:
            self.log("    ⚠ No PDF library available for keyword search")
            return []
    
    def _find_statements_pdfplumber(self, pdf_path: str) -> List[int]:
        """Find statements using pdfplumber."""
        indices = []
        income_start = None
        balance_start = None
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            # First pass: find the START of each statement
            for idx in range(total_pages):
                text = (pdf.pages[idx].extract_text() or "")
                text_upper = text.upper()
                
                # Get first few lines to check for page title
                first_lines = '\n'.join(text.split('\n')[:15]).upper()
                
                # Skip if this looks like a TOC page (has page numbers after statement names)
                is_toc_page = bool(re.search(r'(INCOME|BALANCE|FINANCIAL)\s*\.{0,50}\s*\d{1,3}\s*$', first_lines, re.MULTILINE))
                if is_toc_page:
                    continue
                
                # Skip "COMMENTS ON" or "NOTES TO" pages
                if 'COMMENTS ON THE CONSOLIDATED' in first_lines or 'NOTES TO CONSOLIDATED' in first_lines:
                    continue
                
                # Look for Income Statement
                if income_start is None:
                    income_patterns = [
                        r'CONSOLIDATED\s+STATEMENTS?\s+OF\s+INCOME',
                        r'CONSOLIDATED\s+INCOME\s+STATEMENTS?',
                        r'CONSOLIDATED\s+STATEMENTS?\s+OF\s+OPERATIONS',
                        r'STATEMENTS?\s+OF\s+INCOME\s*\n',
                        r'STATEMENTS?\s+OF\s+OPERATIONS\s*\n',
                    ]
                    
                    for pattern in income_patterns:
                        if re.search(pattern, first_lines):
                            # Verify it has tabular financial data (multiple numbers)
                            numbers = re.findall(r'\d{1,3}(?:,\d{3})+|\(\d{1,3}(?:,\d{3})+\)', text)
                            if len(numbers) >= 5:  # Should have several numbers
                                income_start = idx
                                self.log(f"    Found Income Statement at index {idx}")
                                break
                
                # Look for Balance Sheet
                if balance_start is None:
                    balance_patterns = [
                        r'CONSOLIDATED\s+BALANCE\s+SHEETS?',
                        r'CONSOLIDATED\s+STATEMENTS?\s+OF\s+FINANCIAL\s+POSITION',
                        r'BALANCE\s+SHEETS?\s*\n.*?ASSET',
                        r'STATEMENTS?\s+OF\s+FINANCIAL\s+POSITION',
                    ]
                    
                    for pattern in balance_patterns:
                        if re.search(pattern, first_lines):
                            # Verify it has tabular financial data
                            numbers = re.findall(r'\d{1,3}(?:,\d{3})+|\(\d{1,3}(?:,\d{3})+\)', text)
                            if len(numbers) >= 5:
                                balance_start = idx
                                self.log(f"    Found Balance Sheet at index {idx}")
                                break
            
            # Add found pages to indices
            if income_start is not None:
                indices.append(income_start)
                
                # Check if next page is continuation
                if income_start + 1 < total_pages:
                    next_text = (pdf.pages[income_start + 1].extract_text() or "")[:500].upper()
                    
                    is_new_statement = (
                        ('CONSOLIDATED' in next_text and 'BALANCE' in next_text) or
                        ('CONSOLIDATED' in next_text and 'CASH FLOW' in next_text) or
                        ('CONSOLIDATED' in next_text and 'EQUITY' in next_text) or
                        ('NOTES TO' in next_text)
                    )
                    
                    has_numbers = any(c.isdigit() for c in next_text[:200])
                    
                    if has_numbers and not is_new_statement:
                        indices.append(income_start + 1)
                        self.log(f"    Including continuation page at index {income_start + 1}")
            
            if balance_start is not None:
                if balance_start not in indices:
                    indices.append(balance_start)
                
                # Check if next page is continuation
                if balance_start + 1 < total_pages:
                    next_text = (pdf.pages[balance_start + 1].extract_text() or "")[:500].upper()
                    
                    is_new_statement = (
                        ('CONSOLIDATED' in next_text and 'INCOME' in next_text) or
                        ('CONSOLIDATED' in next_text and 'CASH FLOW' in next_text) or
                        ('CONSOLIDATED' in next_text and 'EQUITY' in next_text) or
                        ('NOTES TO' in next_text)
                    )
                    
                    has_numbers = any(c.isdigit() for c in next_text[:200])
                    
                    if has_numbers and not is_new_statement and balance_start + 1 not in indices:
                        indices.append(balance_start + 1)
                        self.log(f"    Including continuation page at index {balance_start + 1}")
        
        return sorted(list(set(indices)))
    
    def _find_statements_fitz(self, pdf_path: str) -> List[int]:
        """Find statements using PyMuPDF (fitz)."""
        indices = []
        income_start = None
        balance_start = None
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # First pass: find the START of each statement
        for idx in range(total_pages):
            text = doc[idx].get_text()
            text_upper = text[:800].upper()
            
            # Skip if this looks like a TOC page
            is_toc_page = bool(re.search(r'(INCOME STATEMENT|BALANCE SHEET)\s*\.{0,50}\s*\d{1,3}\s*\n', text_upper))
            
            if is_toc_page:
                continue
            
            # Look for Income Statement start
            if income_start is None:
                header_text = text_upper[:300]
                if ('CONSOLIDATED' in header_text and 
                    ('INCOME STATEMENT' in header_text or 
                     'STATEMENT OF OPERATIONS' in header_text or
                     'STATEMENTS OF INCOME' in header_text or
                     'STATEMENTS OF OPERATIONS' in header_text)):
                    # Verify it has actual numbers
                    has_financial_data = bool(re.search(r'\d{1,3}(?:,\d{3})+|\d{4,}', text))
                    if has_financial_data:
                        income_start = idx
                        self.log(f"    Found Income Statement starting at index {idx}")
            
            # Look for Balance Sheet start
            if balance_start is None:
                header_text = text_upper[:300]
                if ('CONSOLIDATED' in header_text and 
                    ('BALANCE SHEET' in header_text or 
                     'BALANCE SHEETS' in header_text or
                     'FINANCIAL POSITION' in header_text or
                     'STATEMENT OF FINANCIAL POSITION' in header_text)):
                    # Verify it has actual numbers
                    has_financial_data = bool(re.search(r'\d{1,3}(?:,\d{3})+|\d{4,}', text))
                    if has_financial_data:
                        balance_start = idx
                        self.log(f"    Found Balance Sheet starting at index {idx}")
        
        # Second pass: check for continuation pages
        if income_start is not None:
            indices.append(income_start)
            
            if income_start + 1 < total_pages:
                next_text = doc[income_start + 1].get_text()[:500].upper()
                
                is_new_statement = (
                    ('CONSOLIDATED' in next_text and 'BALANCE' in next_text) or
                    ('CONSOLIDATED' in next_text and 'CASH FLOW' in next_text) or
                    ('CONSOLIDATED' in next_text and 'EQUITY' in next_text) or
                    ('NOTES TO' in next_text)
                )
                
                has_numbers = any(c.isdigit() for c in next_text[:200])
                
                if has_numbers and not is_new_statement:
                    indices.append(income_start + 1)
                    self.log(f"    Including continuation page at index {income_start + 1}")
        
        if balance_start is not None:
            if balance_start not in indices:
                indices.append(balance_start)
            
            if balance_start + 1 < total_pages:
                next_text = doc[balance_start + 1].get_text()[:500].upper()
                
                is_new_statement = (
                    ('CONSOLIDATED' in next_text and 'INCOME' in next_text) or
                    ('CONSOLIDATED' in next_text and 'CASH FLOW' in next_text) or
                    ('CONSOLIDATED' in next_text and 'EQUITY' in next_text) or
                    ('NOTES TO' in next_text)
                )
                
                has_numbers = any(c.isdigit() for c in next_text[:200])
                
                if has_numbers and not is_new_statement and balance_start + 1 not in indices:
                    indices.append(balance_start + 1)
                    self.log(f"    Including continuation page at index {balance_start + 1}")
        
        doc.close()
        return sorted(list(set(indices)))
    
    def _get_page_count(self, pdf_path: str) -> int:
        """Get total page count of PDF."""
        if HAS_PDFPLUMBER:
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages)
        elif HAS_FITZ:
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        return 0
    
    def find_financial_pages(self, pdf_path: str) -> Dict[str, List[int]]:
        """Find pages containing income statement and balance sheet."""
        self.log("  Searching for financial statement pages...")
        
        results = {
            'income_statement': [],
            'balance_sheet': [],
            'cash_flow': []
        }
        
        # More specific keywords - look for the actual statement headers
        # These should appear at the TOP of the page, not just mentioned
        income_keywords = [
            'consolidated statements of income',
            'consolidated statement of income', 
            'consolidated statements of operations',
            'consolidated income statement',
            'statements of income',
            'statement of operations',
        ]
        
        balance_keywords = [
            'consolidated balance sheets',
            'consolidated balance sheet',
            'consolidated statements of financial position',
            'consolidated statement of financial position',
            'balance sheets',
        ]
        
        cash_keywords = [
            'consolidated statements of cash flows',
            'consolidated statement of cash flows',
            'statements of cash flows',
        ]
        
        if HAS_PDFPLUMBER:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num in range(len(pdf.pages)):
                    text = (pdf.pages[page_num].extract_text() or "").lower()
                    
                    # Get first 500 chars to check if it's a statement header page
                    header_text = text[:500]
                    
                    # Check income statement
                    for kw in income_keywords:
                        if kw in header_text:
                            if page_num not in results['income_statement']:
                                results['income_statement'].append(page_num)
                            break
                    
                    # Check balance sheet
                    for kw in balance_keywords:
                        if kw in header_text:
                            if page_num not in results['balance_sheet']:
                                results['balance_sheet'].append(page_num)
                            break
                    
                    # Check cash flow
                    for kw in cash_keywords:
                        if kw in header_text:
                            if page_num not in results['cash_flow']:
                                results['cash_flow'].append(page_num)
                            break
                            
        elif HAS_FITZ:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                text = doc[page_num].get_text().lower()
                header_text = text[:500]
                
                for kw in income_keywords:
                    if kw in header_text:
                        if page_num not in results['income_statement']:
                            results['income_statement'].append(page_num)
                        break
                
                for kw in balance_keywords:
                    if kw in header_text:
                        if page_num not in results['balance_sheet']:
                            results['balance_sheet'].append(page_num)
                        break
                
                for kw in cash_keywords:
                    if kw in header_text:
                        if page_num not in results['cash_flow']:
                            results['cash_flow'].append(page_num)
                        break
            doc.close()
        
        self.log(f"  Found pages:")
        self.log(f"    Income Statement: {results['income_statement']}")
        self.log(f"    Balance Sheet: {results['balance_sheet']}")
        self.log(f"    Cash Flow: {results['cash_flow']}")
        
        return results
    
    def extract_with_llm(self, pdf_path: str, pages: List[int], 
                         statement_type: str = "both") -> dict:
        """Use LLM to extract financial data from PDF pages."""
        
        self.log(f"  Extracting {statement_type} using LLM (model: {self.model_version})...")
        
        # Try to get page images first
        images = self.extract_pages_as_images(pdf_path, pages, dpi=150)
        
        # Build prompt
        prompt = self._build_extraction_prompt(statement_type)
        
        # Build message content
        content = []
        
        if images:
            # Use vision mode with images
            self.log(f"  Using vision mode ({len(images)} page images)")
            for page_num, img_bytes in images:
                img_b64 = base64.standard_b64encode(img_bytes).decode('utf-8')
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64
                    }
                })
        else:
            # Fall back to text mode
            self.log(f"  Using text mode (no image support available)")
            texts = self.extract_text_from_pages(pdf_path, pages)
            tables = self.extract_tables_from_pages(pdf_path, pages)
            
            # Build text content
            text_content = ""
            for page_num in sorted(texts.keys()):
                text_content += f"\n\n=== PAGE {page_num + 1} ===\n"
                text_content += texts[page_num]
                
                # Add tables if available
                if page_num in tables:
                    text_content += f"\n\n--- TABLES ON PAGE {page_num + 1} ---\n"
                    for i, table in enumerate(tables[page_num]):
                        text_content += f"\nTable {i+1}:\n"
                        for row in table:
                            text_content += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
            
            self.log(f"  Extracted {len(texts)} pages of text, {len(tables)} pages with tables")
            
            prompt = prompt.replace("shown in the images", "shown in the text below")
            prompt += f"\n\n{text_content}"
        
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # Call LLM
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            self.log(f"  Calling Claude API...")
            response = client.messages.create(
                model=self.model_version,
                max_tokens=4000,
                messages=[{"role": "user", "content": content}]
            )
            
            response_text = response.content[0].text
            self._last_response = response_text
            
            self.log(f"  ✓ Got response ({len(response_text)} chars)")
            
            # Debug: print raw response
            self.log(f"\n  === RAW LLM RESPONSE ===")
            self.log(response_text)
            self.log(f"  === END RAW RESPONSE ===\n")
            
            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    self.log(f"  ✓ Parsed JSON successfully")
                    # Debug: show what we got
                    self.log(f"  DEBUG: Keys in result: {list(result.keys())}")
                    if 'income_statement' in result:
                        self.log(f"  DEBUG: income_statement keys: {list(result['income_statement'].keys())}")
                        self.log(f"  DEBUG: revenue = {result['income_statement'].get('revenue', 'NOT FOUND')}")
                    else:
                        self.log(f"  DEBUG: No 'income_statement' key found!")
                        self.log(f"  DEBUG: Full result: {json.dumps(result, indent=2)[:1000]}")
                    return result
                except json.JSONDecodeError as e:
                    self.log(f"  ⚠ JSON parse error: {e}")
                    self.log(f"  Raw response: {response_text[:500]}...")
                    return {}
            else:
                self.log(f"  ⚠ Could not find JSON in response")
                self.log(f"  Raw response: {response_text[:500]}...")
                return {}
                
        except Exception as e:
            self.log(f"  ⚠ LLM extraction error: {e}")
            import traceback
            self.log(f"  {traceback.format_exc()}")
            return {}
    
    def _build_extraction_prompt(self, statement_type: str) -> str:
        """Build the extraction prompt for LLM."""
        
        prompt = """You are a senior financial analyst extracting data from annual report pages.
You can read financial statements in ANY language (English, Chinese, German, French, etc.).

TASK: Extract all numerical values from the financial statements shown in the images.

IMPORTANT INSTRUCTIONS:
1. Extract values for the MOST RECENT YEAR (usually the rightmost or first data column)
2. Identify the unit stated in the document (millions, billions, thousands, 百万, Mio., etc.)
3. Convert ALL values to MILLIONS in your response
4. Use POSITIVE numbers for: assets, revenue, income, profits
5. Use POSITIVE numbers for: expenses, liabilities, costs (we handle signs later)
6. If a value is in parentheses like (1,234), it means negative - extract as NEGATIVE
7. Look carefully at column headers to identify the correct year
8. Note the original currency (USD, EUR, CNY, JPY, etc.)

HANDLING DIFFERENT REPORT FORMATS:
- US GAAP: Look for "Revenue", "Net Income", "Total Assets"
- IFRS: Look for "Revenue", "Profit for the year", "Total Assets"  
- Chinese: Look for "营业收入", "净利润", "资产总计"
- German: Look for "Umsatzerlöse", "Jahresüberschuss", "Bilanzsumme"
- French: Look for "Chiffre d'affaires", "Résultat net", "Total actif"

HANDLING SPECIAL INDUSTRIES:
- Banks (JPMorgan): "Net Interest Income" = revenue, "Deposits" in liabilities
- Oil/Gas (Exxon): "Sales and other operating revenue" = revenue
- Luxury (LVMH): May have segment breakdowns, use consolidated total

RESPOND WITH THIS EXACT JSON FORMAT:
{
    "company": "Company Name",
    "period": "Year ending YYYY-MM-DD or FY2024",
    "currency": "USD/EUR/CNY/JPY/etc",
    "unit": "millions",
    "original_unit": "as stated in document",
    
    "income_statement": {
        "revenue": 0,
        "cost_of_goods_sold": 0,
        "gross_profit": 0,
        "operating_expenses": 0,
        "operating_income": 0,
        "interest_expense": 0,
        "interest_income": 0,
        "income_before_tax": 0,
        "income_tax": 0,
        "net_income": 0,
        "depreciation_amortization": 0,
        "ebitda": 0
    },
    
    "balance_sheet": {
        "cash": 0,
        "short_term_investments": 0,
        "accounts_receivable": 0,
        "inventory": 0,
        "other_current_assets": 0,
        "current_assets": 0,
        "ppe_net": 0,
        "intangible_assets": 0,
        "goodwill": 0,
        "other_non_current_assets": 0,
        "total_assets": 0,
        "accounts_payable": 0,
        "short_term_debt": 0,
        "current_portion_long_term_debt": 0,
        "other_current_liabilities": 0,
        "current_liabilities": 0,
        "long_term_debt": 0,
        "other_non_current_liabilities": 0,
        "total_liabilities": 0,
        "common_stock": 0,
        "retained_earnings": 0,
        "total_equity": 0
    },
    
    "shares_outstanding": 0,
    "notes": "Any important observations, data quality issues, or assumptions made"
}

CRITICAL: 
- If you cannot find a specific value, use 0 and explain in notes
- Double-check that Total Assets = Total Liabilities + Total Equity (roughly)
- Double-check that Revenue - Costs ≈ Operating Income (roughly)
- If numbers seem wrong, note the discrepancy

Extract the data now:"""

        return prompt
    
    def extract_from_pdf(self, pdf_path: str, 
                         income_pages: List[int] = None,
                         balance_pages: List[int] = None,
                         auto_find: bool = True,
                         use_llm_finder: bool = False) -> FinancialData:
        """
        Extract financial data from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            income_pages: Specific pages for income statement (0-indexed)
            balance_pages: Specific pages for balance sheet (0-indexed)
            auto_find: If True, automatically find statement pages
            use_llm_finder: If True, use LLM to find pages from TOC
            
        Returns:
            FinancialData object with extracted values
        """
        self.log(f"\n{'='*60}")
        self.log(f"PDF EXTRACTOR")
        self.log(f"{'='*60}")
        self.log(f"File: {pdf_path}")
        
        all_pages = []
        
        # If pages specified, use them directly (no truncation)
        if income_pages is not None or balance_pages is not None:
            if income_pages:
                all_pages.extend(income_pages)
            if balance_pages:
                all_pages.extend(balance_pages)
        # Otherwise, use LLM to find pages
        elif use_llm_finder:
            llm_result = self.find_pages_with_llm(pdf_path)
            all_pages = llm_result.get('pages', [])
        # Fallback to keyword search
        elif auto_find:
            found = self.find_financial_pages(pdf_path)
            if found['income_statement']:
                all_pages.extend(found['income_statement'][:2])
            if found['balance_sheet']:
                all_pages.extend(found['balance_sheet'][:2])
        
        # Remove duplicates and sort
        all_pages = sorted(list(set(all_pages)))
        
        # Limit to 5 pages max for efficiency
        all_pages = all_pages[:5]
        
        if not all_pages:
            self.log("  ⚠ No financial statement pages found!")
            return FinancialData()
        
        self.log(f"  Extracting from pages: {all_pages} (PDF pages: {[p+1 for p in all_pages]})")
        
        # Extract with LLM
        extracted = self.extract_with_llm(pdf_path, all_pages, "both")
        
        # Convert to FinancialData
        data = self._convert_to_financial_data(extracted)
        data.source_pages = all_pages
        
        return data
    
    def extract_from_url(self, url: str, **kwargs) -> FinancialData:
        """Extract financial data from a PDF URL."""
        # Download PDF
        pdf_path = self.download_pdf(url)
        
        try:
            # Extract data
            data = self.extract_from_pdf(pdf_path, **kwargs)
            return data
        finally:
            # Clean up temp file
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
    
    def _convert_to_financial_data(self, extracted: dict) -> FinancialData:
        """Convert extracted dict to FinancialData object."""
        data = FinancialData()
        
        if not extracted:
            return data
        
        # Metadata
        data.company = extracted.get('company', '')
        data.period = extracted.get('period', '')
        data.currency = extracted.get('currency', 'USD')
        data.unit = extracted.get('unit', 'millions')
        
        # Helper to safely get float
        def safe_float(val):
            try:
                return float(val) if val else 0.0
            except (ValueError, TypeError):
                return 0.0
        
        # Income Statement
        inc = extracted.get('income_statement', {})
        data.revenue = safe_float(inc.get('revenue', 0))
        data.cost_of_goods_sold = safe_float(inc.get('cost_of_goods_sold', 0))
        data.gross_profit = safe_float(inc.get('gross_profit', 0))
        data.operating_expenses = safe_float(inc.get('operating_expenses', 0))
        data.operating_income = safe_float(inc.get('operating_income', 0))
        data.interest_expense = safe_float(inc.get('interest_expense', 0))
        data.income_before_tax = safe_float(inc.get('income_before_tax', 0))
        data.income_tax = safe_float(inc.get('income_tax', 0))
        data.net_income = safe_float(inc.get('net_income', 0))
        data.depreciation = safe_float(inc.get('depreciation_amortization', 0))
        data.ebitda = safe_float(inc.get('ebitda', 0))
        
        # Balance Sheet
        bs = extracted.get('balance_sheet', {})
        data.cash = safe_float(bs.get('cash', 0)) + safe_float(bs.get('short_term_investments', 0))
        data.accounts_receivable = safe_float(bs.get('accounts_receivable', 0))
        data.inventory = safe_float(bs.get('inventory', 0))
        data.current_assets = safe_float(bs.get('current_assets', 0))
        data.ppe = safe_float(bs.get('ppe_net', 0))
        data.total_assets = safe_float(bs.get('total_assets', 0))
        data.accounts_payable = safe_float(bs.get('accounts_payable', 0))
        data.short_term_debt = safe_float(bs.get('short_term_debt', 0)) + safe_float(bs.get('current_portion_long_term_debt', 0))
        data.current_liabilities = safe_float(bs.get('current_liabilities', 0))
        data.long_term_debt = safe_float(bs.get('long_term_debt', 0))
        data.total_liabilities = safe_float(bs.get('total_liabilities', 0))
        data.total_equity = safe_float(bs.get('total_equity', 0))
        data.retained_earnings = safe_float(bs.get('retained_earnings', 0))
        
        # Additional
        data.shares_outstanding = safe_float(extracted.get('shares_outstanding', 0))
        
        # Calculate EBITDA if not provided
        if data.ebitda == 0 and data.operating_income > 0:
            data.ebitda = data.operating_income + data.depreciation
        
        # Validation
        self._validate_data(data, extracted.get('notes', ''))
        
        return data
    
    def _validate_data(self, data: FinancialData, notes: str):
        """Validate extracted data for consistency."""
        warnings = []
        
        # Check balance sheet equation
        if data.total_assets > 0 and data.total_liabilities > 0 and data.total_equity > 0:
            balance_check = abs(data.total_assets - (data.total_liabilities + data.total_equity))
            if balance_check > data.total_assets * 0.05:  # 5% tolerance
                warnings.append(f"Balance sheet doesn't balance: Assets={data.total_assets:.0f}, L+E={data.total_liabilities + data.total_equity:.0f}")
        
        # Check gross profit calculation
        if data.revenue > 0 and data.cost_of_goods_sold > 0:
            expected_gp = data.revenue - data.cost_of_goods_sold
            if data.gross_profit > 0 and abs(data.gross_profit - expected_gp) > data.revenue * 0.05:
                warnings.append(f"Gross profit mismatch: Expected={expected_gp:.0f}, Got={data.gross_profit:.0f}")
        
        # Check for suspiciously low values
        if data.total_assets > 0 and data.revenue == 0:
            warnings.append("Revenue is 0 but company has assets - may need manual check")
        
        if warnings:
            self.log(f"  ⚠ Validation warnings:")
            for w in warnings:
                self.log(f"    - {w}")
    
    def to_ml_format(self, data: FinancialData) -> pd.DataFrame:
        """
        Convert FinancialData to format compatible with ML pipeline.
        
        Returns DataFrame with columns matching fmp_data_fetcher output.
        """
        # Determine multiplier based on unit
        if 'billion' in data.unit.lower():
            multiplier = 1e9
        elif 'million' in data.unit.lower():
            multiplier = 1e6
        elif 'thousand' in data.unit.lower():
            multiplier = 1e3
        else:
            multiplier = 1
        
        # Create row matching ML pipeline format
        row = {
            'date': data.period,
            'sales_revenue': data.revenue * multiplier,
            'cost_of_goods_sold': data.cost_of_goods_sold * multiplier,
            'gross_profit': data.gross_profit * multiplier,
            'overhead_expenses': data.operating_expenses * multiplier / 2,  # Split
            'payroll_expenses': data.operating_expenses * multiplier / 2,
            'ebit': data.operating_income * multiplier,
            'interest_expense': data.interest_expense * multiplier,
            'net_income': data.net_income * multiplier,
            'total_assets': data.total_assets * multiplier,
            'total_equity': data.total_equity * multiplier,
            'total_liabilities': data.total_liabilities * multiplier,
            'total_debt': (data.short_term_debt + data.long_term_debt) * multiplier,
            'cash': data.cash * multiplier,
            'accounts_receivable': data.accounts_receivable * multiplier,
            'inventory': data.inventory * multiplier,
            'accounts_payable': data.accounts_payable * multiplier,
            'retained_earnings': data.retained_earnings * multiplier,
            'shares_outstanding': data.shares_outstanding,
            'capex': data.ppe * multiplier * 0.1,  # Estimate 10% of PPE
            'dividends_paid': 0,  # Would need cash flow statement
            'stock_repurchased': 0,
        }
        
        return pd.DataFrame([row])
    
    def print_summary(self, data: FinancialData):
        """Print a summary of extracted data."""
        print(f"\n{'='*60}")
        print(f"EXTRACTED FINANCIAL DATA: {data.company}")
        print(f"{'='*60}")
        print(f"Period: {data.period}")
        print(f"Currency: {data.currency} ({data.unit})")
        print(f"Source pages: {data.source_pages}")
        
        print(f"\n--- Income Statement ---")
        print(f"  Revenue:           {data.revenue:>12,.0f}")
        print(f"  COGS:              {data.cost_of_goods_sold:>12,.0f}")
        print(f"  Gross Profit:      {data.gross_profit:>12,.0f}")
        print(f"  Operating Income:  {data.operating_income:>12,.0f}")
        print(f"  Interest Expense:  {data.interest_expense:>12,.0f}")
        print(f"  Net Income:        {data.net_income:>12,.0f}")
        
        print(f"\n--- Balance Sheet ---")
        print(f"  Total Assets:      {data.total_assets:>12,.0f}")
        print(f"  Total Liabilities: {data.total_liabilities:>12,.0f}")
        print(f"  Total Equity:      {data.total_equity:>12,.0f}")
        print(f"  Total Debt:        {data.short_term_debt + data.long_term_debt:>12,.0f}")
        
        # Calculate and print ratios
        ratios = data.calculate_ratios()
        
        print(f"\n--- Key Ratios ---")
        if 'net_margin' in ratios:
            print(f"  Net Margin:        {ratios['net_margin']:>12.2%}")
        if 'cost_to_income_ratio' in ratios:
            print(f"  Cost-to-Income:    {ratios['cost_to_income_ratio']:>12.2%}")
        if 'quick_ratio' in ratios:
            print(f"  Quick Ratio:       {ratios['quick_ratio']:>12.2f}")
        if 'debt_to_equity' in ratios:
            print(f"  Debt/Equity:       {ratios['debt_to_equity']:>12.2f}")
        if 'debt_to_assets' in ratios:
            print(f"  Debt/Assets:       {ratios['debt_to_assets']:>12.2%}")
        if 'debt_to_capital' in ratios:
            print(f"  Debt/Capital:      {ratios['debt_to_capital']:>12.2%}")
        if 'debt_to_ebitda' in ratios:
            print(f"  Debt/EBITDA:       {ratios['debt_to_ebitda']:>12.2f}x")
        if 'interest_coverage' in ratios:
            print(f"  Interest Coverage: {ratios['interest_coverage']:>12.2f}x")
        
        print(f"{'='*60}\n")


def test_robustness(extractor: PDFExtractor, pdf_path: str, 
                    pages: List[int], n_runs: int = 3) -> dict:
    """Test robustness of LLM extraction across multiple runs."""
    print(f"\n{'='*60}")
    print(f"ROBUSTNESS TEST ({n_runs} runs)")
    print(f"{'='*60}")
    
    results = []
    for i in range(n_runs):
        print(f"\nRun {i+1}/{n_runs}...")
        data = extractor.extract_from_pdf(pdf_path, income_pages=pages, balance_pages=pages)
        results.append(data)
    
    # Compare key metrics
    metrics = ['revenue', 'net_income', 'total_assets', 'total_equity']
    
    print(f"\n--- Consistency Check ---")
    all_consistent = True
    
    for metric in metrics:
        values = [getattr(r, metric) for r in results]
        unique = len(set(values))
        consistent = "✓" if unique == 1 else "✗"
        if unique > 1:
            all_consistent = False
        print(f"  {metric}: {values} {consistent}")
    
    print(f"\n  Overall: {'CONSISTENT' if all_consistent else 'INCONSISTENT'}")
    print(f"  Model: {extractor.model_version}")
    
    return {
        'n_runs': n_runs,
        'consistent': all_consistent,
        'model_version': extractor.model_version,
        'results': [r.to_dict() for r in results]
    }


# Command line interface
if __name__ == "__main__":
    import sys
    
    print(f"PDF Financial Statement Extractor")
    print(f"Model: {MODEL_VERSION}")
    print(f"="*60)
    
    if len(sys.argv) < 2:
        print("""
Usage:
    python pdf_extractor.py <pdf_path_or_url> [page1,page2,...]
    
Examples:
    # Extract from local file (auto-find pages)
    python pdf_extractor.py annual_report.pdf
    
    # Extract from specific pages
    python pdf_extractor.py annual_report.pdf 55,56
    
    # Extract from URL
    python pdf_extractor.py https://example.com/report.pdf
    
    # Test GM annual report
    python pdf_extractor.py GM
""")
        sys.exit(1)
    
    # Predefined URLs for common companies
    # Set pages=None to use auto-detection
    COMPANY_URLS = {
        'GM': {
            'url': 'https://investor.gm.com/static-files/1fff6f59-551f-4fe0-bca9-74bfc9a56aeb',
            'pages': None,  # Auto-detect
            'notes': 'General Motors 2023 - US GAAP, USD millions'
        },
        'LVMH': {
            'url': 'https://lvmh-com.cdn.prismic.io/lvmh-com/Z5kVBpbqstJ999KR_Financialdocuments-December31%2C2024.pdf',
            'pages': None,  # Auto-detect
            'notes': 'LVMH 2024 - IFRS, EUR millions'
        },
        'JPMORGAN': {
            'url': 'https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/annualreport-2024.pdf',
            'pages': None,  # Auto-detect via LLM text search
            'notes': 'JPMorgan Chase 2024 - US GAAP, USD millions (Bank format)'
        },
        'JPM': {
            'url': 'https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/investor-relations/documents/annualreport-2024.pdf',
            'pages': None,  # Auto-detect via LLM text search
            'notes': 'JPMorgan Chase 2024 - US GAAP, USD millions (Bank format)'
        },
        'EXXON': {
            'url': 'https://investor.exxonmobil.com/sec-filings/all-sec-filings/content/0000034088-25-000010/0000034088-25-000010.pdf',
            'pages': None,
            'notes': 'Exxon Mobil 2024 - US GAAP, USD millions (Oil & Gas)'
        },
        'MICROSOFT': {
            'url': 'https://microsoft.gcs-web.com/static-files/1c864583-06f7-40cc-a94d-d11400c83cc8',
            'pages': None,
            'notes': 'Microsoft FY2024 - US GAAP, USD millions'
        },
        'GOOGLE': {
            'url': None,
            'local_path': 'data/goog-10-k-2024.pdf',  # Local file in project data folder
            'pages': None,
            'notes': 'Alphabet/Google 2024 - US GAAP, USD millions'
        },
        'ALPHABET': {
            'url': None,
            'local_path': 'data/goog-10-k-2024.pdf',
            'pages': None,
            'notes': 'Alphabet/Google 2024 - US GAAP, USD millions'
        },
        'VOLKSWAGEN': {
            'url': None,
            'local_path': 'data/Y_2024_e.pdf',  # Local file in project data folder
            'pages': None,
            'notes': 'Volkswagen 2024 - IFRS, EUR millions'
        },
        'VW': {
            'url': None,
            'local_path': 'data/Y_2024_e.pdf',
            'pages': None,
            'notes': 'Volkswagen 2024 - IFRS, EUR millions'
        },
        'TENCENT': {
            'url': 'https://static.www.tencent.com/uploads/2025/04/08/1132b72b565389d1b913aea60a648d73.pdf',
            'pages': None,
            'notes': 'Tencent 2024 - IFRS, RMB millions'
        },
        'ALIBABA': {
            'url': 'https://data.alibabagroup.com/ecms-files/1508664153/99622d51-2f4d-4081-9256-29e5200ecdae/2024%20SEC%20Form%2020-F.pdf',
            'pages': None,
            'notes': 'Alibaba FY2024 (ending Mar 2024) - US GAAP, RMB millions'
        },
        'BABA': {
            'url': 'https://data.alibabagroup.com/ecms-files/1508664153/99622d51-2f4d-4081-9256-29e5200ecdae/2024%20SEC%20Form%2020-F.pdf',
            'pages': None,
            'notes': 'Alibaba FY2024 (ending Mar 2024) - US GAAP, RMB millions'
        },
        'SHELL': {
            'url': 'https://www.shell.com/investors/results-and-reporting/annual-report-archive/_jcr_content/root/main/section_812377294/tabs/tab/text.multi.stream/1742905301176/ce28b952e201476287788cfcf35406e464f9785c/shell-annual-report-2023.pdf',
            'pages': None,
            'notes': 'Shell 2023 - IFRS, USD millions (Oil & Gas, 402 pages)'
        },
        'SHEL': {
            'url': 'https://www.shell.com/investors/results-and-reporting/annual-report-archive/_jcr_content/root/main/section_812377294/tabs/tab/text.multi.stream/1742905301176/ce28b952e201476287788cfcf35406e464f9785c/shell-annual-report-2023.pdf',
            'pages': None,
            'notes': 'Shell 2023 - IFRS, USD millions (Oil & Gas, 402 pages)'
        },
    }
    
    extractor = PDFExtractor(verbose=True)
    
    source = sys.argv[1]
    pages = None
    use_auto = True  # Default to auto mode
    
    if len(sys.argv) > 2:
        pages = [int(p) for p in sys.argv[2].split(',')]
        use_auto = False  # Manual pages specified
    
    # Check if it's a company shortcut
    if source.upper() in COMPANY_URLS:
        company_info = COMPANY_URLS[source.upper()]
        # Check for local file first
        local_path = company_info.get('local_path')
        if local_path:
            import os
            # Try relative to current directory and common locations
            possible_paths = [
                local_path,
                os.path.join(os.getcwd(), local_path),
                os.path.join(os.path.dirname(__file__), '..', '..', '..', local_path),
            ]
            
            pdf_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    pdf_path = p
                    break
            
            if pdf_path:
                print(f"Using local file for {source.upper()}: {pdf_path}")
                print(f"Notes: {company_info['notes']}")
                
                if pages:
                    data = extractor.extract_from_pdf(pdf_path, income_pages=pages, balance_pages=pages)
                elif company_info.get('pages'):
                    data = extractor.extract_from_pdf(pdf_path, income_pages=company_info['pages'], balance_pages=company_info['pages'])
                else:
                    data = extractor.extract_auto(pdf_path)
            else:
                print(f"Local file not found: {local_path}")
                print(f"Please ensure the file exists in your project's data folder.")
                print(f"Notes: {company_info['notes']}")
                sys.exit(1)
        elif company_info['url'] is None:
            print(f"No predefined URL for {source.upper()}. Please provide the PDF URL or path.")
            print(f"Notes: {company_info['notes']}")
            sys.exit(1)
        else:
            url = company_info['url']
            print(f"Using predefined URL for {source.upper()}")
            print(f"Notes: {company_info['notes']}")
            
            # Download PDF
            pdf_path = extractor.download_pdf(url)
            
            try:
                if pages:
                    # Manual pages specified
                    data = extractor.extract_from_pdf(pdf_path, income_pages=pages, balance_pages=pages)
                elif company_info.get('pages'):
                    # Use predefined pages
                    data = extractor.extract_from_pdf(pdf_path, income_pages=company_info['pages'], balance_pages=company_info['pages'])
                else:
                    # Auto-detect pages
                    data = extractor.extract_auto(pdf_path)
            finally:
                import os
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                
    elif source.startswith('http'):
        # URL - download and auto-extract
        pdf_path = extractor.download_pdf(source)
        try:
            if pages:
                data = extractor.extract_from_pdf(pdf_path, income_pages=pages, balance_pages=pages)
            else:
                data = extractor.extract_auto(pdf_path)
        finally:
            import os
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
    else:
        # Local file
        if pages:
            data = extractor.extract_from_pdf(source, income_pages=pages, balance_pages=pages)
        else:
            data = extractor.extract_auto(source)
    
    extractor.print_summary(data)
    
    # Print ML-compatible format
    ml_df = extractor.to_ml_format(data)
    print("\nML Pipeline Compatible Format:")
    print(ml_df.T)