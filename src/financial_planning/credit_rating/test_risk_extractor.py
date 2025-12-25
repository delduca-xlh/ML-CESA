#!/usr/bin/env python
"""
Risk Extractor Usage Examples
==============================

Demonstrates risk_extractor.py with REAL PDF annual reports.

Data Sources:
- Evergrande 2020: data/annual_reports/evergrande/ar2020.pdf
- Evergrande 2021: data/annual_reports/evergrande/car2021.pdf  
- Lehman 2007: data/annual_reports/lehman/lehman.pdf

Usage:
    cd /Users/lihao/Documents/GitHub/ML-CESA
    python src/financial_planning/credit_rating/test_risk_extractor.py
"""

import subprocess
import sys
import os

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from risk_extractor import (
    AnnualReportRiskExtractor,
    AnnualReportComparator,
    RiskLevel,
)


# =============================================================================
# PDF PATHS
# =============================================================================

PDF_PATHS = {
    'evergrande_2020': 'data/annual_reports/evergrande/ar2020.pdf',
    'evergrande_2021': 'data/annual_reports/evergrande/car2021.pdf',
    'lehman_2007': 'data/annual_reports/lehman/lehman.pdf',
    'enron_2000': 'data/annual_reports/enron/EnronAnnualReport2000.pdf',
}


def find_project_root():
    """Find the project root directory"""
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.exists(os.path.join(current, 'data', 'annual_reports')):
            return current
        current = os.path.dirname(current)
    return os.getcwd()


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF"""
    import os as _os
    
    # Method 1: pdftotext - completely suppress stderr
    try:
        with open(_os.devnull, 'w') as devnull:
            result = subprocess.run(
                ['pdftotext', '-layout', pdf_path, '-'],
                stdout=subprocess.PIPE,
                stderr=devnull,
                text=True, 
                timeout=120
            )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except Exception:
        pass
    
    # Method 2: pdfplumber (suppress all warnings)
    try:
        import warnings
        import logging
        for logger_name in ['pdfplumber', 'pdfminer', 'pdfminer.pdfpage', 
                           'pdfminer.converter', 'pdfminer.pdfdocument']:
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
                return "\n".join(pages)
    except Exception:
        pass
    
    # Method 3: PyPDF2
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
    except Exception:
        pass
    
    return ""


# =============================================================================
# EXAMPLE 1: Single Annual Report Analysis
# =============================================================================

def example_single_report():
    """Analyze a single annual report from real PDF"""
    print("=" * 80)
    print("EXAMPLE 1: Single Annual Report Analysis (Real PDF)")
    print("=" * 80)
    
    project_root = find_project_root()
    pdf_path = os.path.join(project_root, PDF_PATHS['evergrande_2020'])
    
    if not os.path.exists(pdf_path):
        print(f"\n❌ PDF not found: {pdf_path}")
        print("   Please ensure the file exists in data/annual_reports/evergrande/")
        return None
    
    print(f"\n📄 Loading: {PDF_PATHS['evergrande_2020']}")
    text = extract_pdf_text(pdf_path)
    print(f"   Extracted: {len(text):,} characters")
    
    # Analyze
    extractor = AnnualReportRiskExtractor()
    report = extractor.extract_risks(text, "Evergrande 2020")
    extractor.print_report(report)
    
    # Programmatic access
    print("\n--- Programmatic Access ---")
    print(f"Company: {report.company_name}")
    print(f"Auditor Opinion: {report.auditor_opinion}")
    print(f"Going Concern: {report.going_concern}")
    print(f"Overall Risk: {report.overall_risk.name}")
    print(f"Number of Warnings: {len(report.warnings)}")
    
    return report


# =============================================================================
# EXAMPLE 2: Year-over-Year Comparison
# =============================================================================

def example_yoy_comparison():
    """Compare two years of annual reports from real PDFs"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Year-over-Year Comparison (Real PDFs)")
    print("=" * 80)
    
    project_root = find_project_root()
    pdf_2020 = os.path.join(project_root, PDF_PATHS['evergrande_2020'])
    pdf_2021 = os.path.join(project_root, PDF_PATHS['evergrande_2021'])
    
    # Check files exist
    if not os.path.exists(pdf_2020) or not os.path.exists(pdf_2021):
        print("\n❌ PDF files not found. Please ensure both files exist:")
        print(f"   - {PDF_PATHS['evergrande_2020']}")
        print(f"   - {PDF_PATHS['evergrande_2021']}")
        return None
    
    # Extract text
    print(f"\n📄 Loading 2020 report...")
    text_2020 = extract_pdf_text(pdf_2020)
    print(f"   Extracted: {len(text_2020):,} characters")
    
    print(f"\n📄 Loading 2021 report...")
    text_2021 = extract_pdf_text(pdf_2021)
    print(f"   Extracted: {len(text_2021):,} characters")
    
    # Compare
    comparator = AnnualReportComparator()
    report = comparator.compare(
        prior_text=text_2020,
        current_text=text_2021,
        company_name="China Evergrande",
        prior_year="2020",
        current_year="2021"
    )
    comparator.print_comparison_report(report)
    
    # Programmatic access
    print("\n--- Programmatic Access ---")
    print(f"Risk Escalation: {report.risk_escalation}")
    print(f"Total Changes: {len(report.changes)}")
    
    new_risks = [c for c in report.changes if c.change_type == 'NEW']
    worsened = [c for c in report.changes if c.change_type == 'WORSENED']
    print(f"New Risks: {len(new_risks)}")
    print(f"Worsened: {len(worsened)}")
    
    return report


# =============================================================================
# EXAMPLE 3: Lehman Brothers Analysis
# =============================================================================

def example_lehman():
    """Analyze Lehman Brothers 2007 10-K from real PDF"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Lehman Brothers 2007 (Real PDF)")
    print("=" * 80)
    
    project_root = find_project_root()
    pdf_path = os.path.join(project_root, PDF_PATHS['lehman_2007'])
    
    if not os.path.exists(pdf_path):
        print(f"\n❌ PDF not found: {pdf_path}")
        print("   Please download Lehman 2007 10-K from SEC EDGAR")
        return None
    
    print(f"\n📄 Loading: {PDF_PATHS['lehman_2007']}")
    text = extract_pdf_text(pdf_path)
    print(f"   Extracted: {len(text):,} characters")
    
    # Analyze
    extractor = AnnualReportRiskExtractor()
    report = extractor.extract_risks(text, "Lehman Brothers 2007")
    extractor.print_report(report)
    
    return report


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           ANNUAL REPORT RISK EXTRACTOR - REAL PDF EXAMPLES                   ║
║                                                                              ║
║  Based on "Financial Shenanigans" by Howard Schilit                          ║
║  Data: Real annual reports from Evergrande and Lehman Brothers               ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    results = []
    
    # Run examples
    r1 = example_single_report()
    if r1:
        results.append(("Evergrande 2020", r1))
    
    example_yoy_comparison()
    
    r3 = example_lehman()
    if r3:
        results.append(("Lehman Brothers 2007", r3))
    
    # Summary
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\n{'Company':<25} {'Opinion':<22} {'Going Concern':<15} {'Warnings':<10} {'Risk':<10}")
        print("-" * 90)
        for name, r in results:
            gc = "YES ⚠️" if r.going_concern else "No"
            print(f"{name:<25} {r.auditor_opinion:<22} {gc:<15} {len(r.warnings):<10} {r.overall_risk.name:<10}")
    
    # Usage summary
    print("\n" + "=" * 80)
    print("USAGE SUMMARY")
    print("=" * 80)
    print("""
    1. SINGLE REPORT ANALYSIS:
       
       from risk_extractor import AnnualReportRiskExtractor
       
       extractor = AnnualReportRiskExtractor()
       report = extractor.extract_risks(text, "Company Name")
       extractor.print_report(report)
       
       # Access data:
       report.auditor_opinion   # "Qualified", "Adverse", "Disclaimer", etc.
       report.going_concern     # True/False
       report.overall_risk      # RiskLevel.CRITICAL, HIGH, MEDIUM, LOW
       report.warnings          # List of RiskWarning objects
    
    2. YEAR-OVER-YEAR COMPARISON:
       
       from risk_extractor import AnnualReportComparator
       
       comparator = AnnualReportComparator()
       report = comparator.compare(
           prior_text=text_2020,
           current_text=text_2021,
           company_name="Company",
           prior_year="2020",
           current_year="2021"
       )
       comparator.print_comparison_report(report)
    
    3. PDF TEXT EXTRACTION:
       
       import subprocess
       result = subprocess.run(['pdftotext', '-layout', 'report.pdf', '-'], 
                              capture_output=True, text=True)
       text = result.stdout
    """)
