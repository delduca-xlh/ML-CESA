"""
Financial Fraud Detection
=========================

Bonus 1d: Detect financial fraud/manipulation

Includes:
1. Beneish M-Score: Earnings manipulation detection
2. Altman Z-Score: Bankruptcy risk prediction
3. Red Flags: Financial warning indicators

Reference: "Financial Shenanigans" by Howard Schilit

Author: Lihao Xiao
"""

import numpy as np
from typing import Dict, List


class AltmanZScore:
    """
    Altman Z-Score Bankruptcy Risk Model
    
    Mathematical Form (Private Company Version):
    Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5
    
    Where:
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Book Value of Equity / Total Liabilities
    X5 = Sales / Total Assets
    
    Interpretation:
    Z' > 2.9  : Safe Zone
    1.23 < Z' < 2.9 : Grey Zone
    Z' < 1.23 : Distress Zone
    """
    
    @staticmethod
    def calculate(financials: Dict) -> Dict:
        """Calculate Altman Z-Score"""
        # Extract data
        current_assets = financials.get('current_assets', 0) or 0
        current_liabilities = financials.get('current_liabilities', 0) or 0
        total_assets = financials.get('total_assets', 1) or 1
        retained_earnings = financials.get('retained_earnings', 0) or 0
        ebit = financials.get('operating_income', 0) or financials.get('ebit', 0) or 0
        total_equity = financials.get('total_equity', 0) or 0
        total_liabilities = financials.get('total_liabilities', 1) or 1
        revenue = financials.get('revenue', 0) or 0
        
        # Calculate components
        working_capital = current_assets - current_liabilities
        
        X1 = working_capital / total_assets
        X2 = retained_earnings / total_assets
        X3 = ebit / total_assets
        X4 = total_equity / total_liabilities
        X5 = revenue / total_assets
        
        # Z-Score (Private company version)
        z_score = (0.717 * X1 + 0.847 * X2 + 3.107 * X3 + 
                   0.420 * X4 + 0.998 * X5)
        
        # Determine zone
        if z_score > 2.9:
            zone = "SAFE"
            risk = "Low bankruptcy risk"
        elif z_score > 1.23:
            zone = "GREY"
            risk = "Moderate bankruptcy risk"
        else:
            zone = "DISTRESS"
            risk = "High bankruptcy risk"
        
        return {
            'z_score': round(z_score, 3),
            'zone': zone,
            'risk_assessment': risk,
            'components': {
                'X1_working_capital_ratio': round(X1, 4),
                'X2_retained_earnings_ratio': round(X2, 4),
                'X3_ebit_ratio': round(X3, 4),
                'X4_equity_to_liabilities': round(X4, 4),
                'X5_asset_turnover': round(X5, 4)
            }
        }


class FraudDetector:
    """
    Financial Fraud/Manipulation Detection
    
    Includes:
    1. Beneish M-Score
    2. Red flag warning checks
    """
    
    @staticmethod
    def calculate_m_score(current: Dict, prior: Dict) -> Dict:
        """
        Calculate Beneish M-Score
        
        M-Score > -1.78 indicates likely earnings manipulation
        
        M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI 
            + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
        """
        def safe_get(d, key, default=1):
            val = d.get(key, default)
            return val if val and val != 0 else default
        
        # Current year
        rev_c = safe_get(current, 'revenue')
        rec_c = safe_get(current, 'accounts_receivable', current.get('current_assets', 0) * 0.2)
        assets_c = safe_get(current, 'total_assets')
        gm_c = (rev_c - safe_get(current, 'cost_of_goods_sold', rev_c * 0.7)) / rev_c
        ppe_c = safe_get(current, 'ppe', assets_c * 0.3)
        dep_c = safe_get(current, 'depreciation', rev_c * 0.05)
        sga_c = safe_get(current, 'sga_expense', rev_c * 0.15)
        ni_c = safe_get(current, 'net_income', rev_c * 0.05)
        cfo_c = safe_get(current, 'operating_cash_flow', ni_c * 0.8)
        liab_c = safe_get(current, 'total_liabilities')
        
        # Prior year
        rev_p = safe_get(prior, 'revenue')
        rec_p = safe_get(prior, 'accounts_receivable', prior.get('current_assets', 0) * 0.2)
        assets_p = safe_get(prior, 'total_assets')
        gm_p = (rev_p - safe_get(prior, 'cost_of_goods_sold', rev_p * 0.7)) / rev_p
        ppe_p = safe_get(prior, 'ppe', assets_p * 0.3)
        dep_p = safe_get(prior, 'depreciation', rev_p * 0.05)
        sga_p = safe_get(prior, 'sga_expense', rev_p * 0.15)
        liab_p = safe_get(prior, 'total_liabilities')
        ca_p = safe_get(prior, 'current_assets')
        
        # Calculate indices
        DSRI = (rec_c / rev_c) / (rec_p / rev_p)  # Days Sales Receivable Index
        GMI = gm_p / gm_c  # Gross Margin Index
        AQI = (1 - (safe_get(current, 'current_assets') + ppe_c) / assets_c) / \
              (1 - (ca_p + ppe_p) / assets_p)  # Asset Quality Index
        SGI = rev_c / rev_p  # Sales Growth Index
        DEPI = (dep_p / (dep_p + ppe_p)) / (dep_c / (dep_c + ppe_c))  # Depreciation Index
        SGAI = (sga_c / rev_c) / (sga_p / rev_p)  # SGA Index
        TATA = (ni_c - cfo_c) / assets_c  # Total Accruals to Total Assets
        LVGI = (liab_c / assets_c) / (liab_p / assets_p)  # Leverage Index
        
        # M-Score
        m_score = (-4.84 + 0.92 * DSRI + 0.528 * GMI + 0.404 * AQI + 
                   0.892 * SGI + 0.115 * DEPI - 0.172 * SGAI + 
                   4.679 * TATA - 0.327 * LVGI)
        
        assessment = "LIKELY MANIPULATOR" if m_score > -1.78 else "UNLIKELY MANIPULATOR"
        
        return {
            'm_score': round(m_score, 3),
            'threshold': -1.78,
            'assessment': assessment,
            'components': {
                'DSRI': round(DSRI, 3),
                'GMI': round(GMI, 3),
                'AQI': round(AQI, 3),
                'SGI': round(SGI, 3),
                'DEPI': round(DEPI, 3),
                'SGAI': round(SGAI, 3),
                'TATA': round(TATA, 3),
                'LVGI': round(LVGI, 3)
            }
        }
    
    @staticmethod
    def check_red_flags(financials: Dict) -> List[str]:
        """
        Check for financial red flag warnings
        
        Based on "Financial Shenanigans" by Howard Schilit
        """
        red_flags = []
        
        # Extract data
        revenue = financials.get('revenue', 0) or 0
        net_income = financials.get('net_income', 0) or 0
        total_assets = financials.get('total_assets', 1) or 1
        total_equity = financials.get('total_equity', 0) or 0
        total_debt = financials.get('total_debt', 0) or 0
        cash = financials.get('cash', 0) or 0
        inventory = financials.get('inventory', 0) or 0
        current_assets = financials.get('current_assets', 0) or 0
        current_liabilities = financials.get('current_liabilities', 1) or 1
        interest_expense = financials.get('interest_expense', 0) or 0
        operating_income = financials.get('operating_income', 0) or 0
        retained_earnings = financials.get('retained_earnings', 0) or 0
        
        # 1. Negative working capital
        if current_assets < current_liabilities:
            red_flags.append("NEGATIVE WORKING CAPITAL: Current liabilities exceed current assets")
        
        # 2. Cash crisis
        if interest_expense > 0 and cash < interest_expense:
            red_flags.append("CASH CRISIS: Cash insufficient to cover interest payments")
        
        # 3. Excessive inventory
        if revenue > 0 and inventory / revenue > 0.5:
            red_flags.append("EXCESSIVE INVENTORY: Inventory > 50% of revenue")
        
        # 4. Extreme leverage
        if total_equity > 0 and total_debt / total_equity > 3:
            red_flags.append("EXTREME LEVERAGE: Debt-to-Equity > 3x")
        elif total_equity <= 0:
            red_flags.append("NEGATIVE EQUITY: Shareholders' equity is negative")
        
        # 5. Operating losses
        if operating_income < 0:
            red_flags.append("OPERATING LOSSES: Core business is unprofitable")
        
        # 6. Accumulated deficit
        if retained_earnings < 0:
            red_flags.append("ACCUMULATED DEFICIT: Company has accumulated losses")
        
        # 7. Insufficient interest coverage
        if interest_expense > 0 and operating_income / interest_expense < 1:
            red_flags.append("INTEREST COVERAGE < 1x: Cannot cover interest from operations")
        
        # 8. Asset quality issues
        if total_assets > 0 and inventory / total_assets > 0.6:
            red_flags.append("ASSET QUALITY: Inventory dominates assets (60%+)")
        
        # 9. Going concern risk
        if (current_assets < current_liabilities and 
            cash < interest_expense and 
            operating_income < 0):
            red_flags.append("GOING CONCERN RISK: Multiple severe indicators present")
        
        return red_flags
    
    @staticmethod
    def generate_report(financials: Dict, company_name: str = "Company") -> str:
        """Generate comprehensive risk report"""
        lines = [
            "=" * 60,
            f"FINANCIAL WARNING REPORT: {company_name}",
            "=" * 60
        ]
        
        red_flags = FraudDetector.check_red_flags(financials)
        
        if red_flags:
            lines.append(f"\n{len(red_flags)} WARNING(S) DETECTED:\n")
            for flag in red_flags:
                lines.append(f"  - {flag}")
        else:
            lines.append("\nNo major red flags detected")
        
        # Overall risk assessment
        severe_count = sum(1 for f in red_flags if 'CRISIS' in f or 'NEGATIVE' in f or 'GOING CONCERN' in f)
        if severe_count >= 3:
            lines.extend([
                "\n" + "=" * 60,
                "OVERALL RISK: CRITICAL",
                "Recommendation: Avoid lending / Immediate review required"
            ])
        elif severe_count >= 1:
            lines.extend([
                "\n" + "=" * 60,
                "OVERALL RISK: HIGH",
                "Recommendation: Enhanced due diligence required"
            ])
        elif len(red_flags) >= 2:
            lines.extend([
                "\n" + "=" * 60,
                "OVERALL RISK: MODERATE",
                "Recommendation: Additional analysis recommended"
            ])
        else:
            lines.extend([
                "\n" + "=" * 60,
                "OVERALL RISK: LOW"
            ])
        
        lines.append("=" * 60)
        return "\n".join(lines)