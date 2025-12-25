#!/usr/bin/env python
"""
Annual Report Risk Warning Extractor
=====================================

Bonus 2: Automatically extract risk warnings from annual reports

Features:
1. Auditor Opinion Analysis (Qualified, Adverse, Disclaimer, Going Concern)
2. Risk Factor Extraction
3. Litigation & Contingency Detection
4. Management Discussion Analysis
5. Key Risk Phrase Detection

Reference: 
- "Financial Shenanigans" by Howard Schilit
- PCAOB Auditing Standards
- ISA 570 Going Concern
- ISA 706 Emphasis of Matter

Author: Credit Rating System
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(Enum):
    """Risk severity levels"""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    INFO = 0


@dataclass
class RiskWarning:
    """A single risk warning extracted from the report"""
    category: str
    description: str
    level: RiskLevel
    source_text: str = ""
    page_hint: str = ""
    

@dataclass
class RiskReport:
    """Complete risk analysis report"""
    company_name: str
    warnings: List[RiskWarning] = field(default_factory=list)
    auditor_opinion: str = "Unknown"
    going_concern: bool = False
    overall_risk: RiskLevel = RiskLevel.LOW
    summary: str = ""


class AnnualReportRiskExtractor:
    """
    Extracts risk warnings from annual report text
    
    Based on key sections:
    1. Independent Auditor's Report
    2. Risk Factors
    3. Management Discussion & Analysis (MD&A)
    4. Notes to Financial Statements
    5. Legal Proceedings
    """
    
    # =========================================================================
    # RISK KEYWORDS AND PATTERNS
    # =========================================================================
    
    # Auditor Opinion Types (most critical)
    AUDITOR_OPINION_PATTERNS = {
        'qualified': [
            r'qualified\s+opinion',
            r'except\s+for\s+the\s+(?:effects?|matters?)',
            r'qualification\s+(?:of|to)\s+(?:the\s+)?opinion',
            r'basis\s+for\s+qualified\s+opinion',
        ],
        'adverse': [
            r'adverse\s+opinion',
            r'do\s+not\s+present\s+fairly',
            r'financial\s+statements\s+are\s+not\s+(?:fairly\s+)?presented',
            r'material\s+and\s+pervasive',
        ],
        'disclaimer': [
            r'disclaimer\s+of\s+opinion',
            r'disclaim\s+(?:an\s+)?opinion',
            r'we\s+do\s+not\s+express\s+(?:an\s+)?opinion',
            r'unable\s+to\s+(?:form|express)\s+(?:an\s+)?opinion',
        ],
        'going_concern': [
            r'going\s+concern',
            r'substantial\s+doubt\s+about\s+(?:the\s+)?(?:company\'?s?|entity\'?s?|its)\s+ability\s+to\s+continue',
            r'material\s+uncertainty\s+(?:related\s+to|regarding)\s+(?:the\s+)?(?:company\'?s?|entity\'?s?)?\s*(?:ability\s+to\s+)?continue',
            r'ability\s+to\s+continue\s+as\s+a\s+going\s+concern',
            r'may\s+not\s+be\s+able\s+to\s+continue',
            r'raise(?:s)?\s+substantial\s+doubt',
        ],
        'emphasis_of_matter': [
            r'emphasis\s+of\s+matter',
            r'we\s+draw\s+attention\s+to',
            r'without\s+(?:further\s+)?(?:modifying|qualifying)\s+our\s+opinion',
        ],
    }
    
    # Financial Distress Indicators
    FINANCIAL_DISTRESS_PATTERNS = [
        (r'recurring\s+(?:operating\s+)?loss(?:es)?', 'Recurring losses', RiskLevel.HIGH),
        (r'accumulated\s+(?:deficit|losses?)', 'Accumulated deficit', RiskLevel.HIGH),
        (r'negative\s+(?:working\s+capital|cash\s+flow|equity|net\s+worth)', 'Negative financial position', RiskLevel.HIGH),
        (r'(?:net\s+)?capital\s+deficiency', 'Capital deficiency', RiskLevel.CRITICAL),
        (r'default(?:ed)?\s+(?:on|under)\s+(?:loan|debt|credit|covenant)', 'Loan default', RiskLevel.CRITICAL),
        (r'breach(?:ed)?\s+(?:of\s+)?(?:loan\s+)?covenant', 'Covenant breach', RiskLevel.CRITICAL),
        (r'(?:fail(?:ed|ure)?|unable)\s+to\s+(?:pay|meet|satisfy)\s+(?:debt|obligation|payment)', 'Payment failure', RiskLevel.CRITICAL),
        (r'debt\s+restructuring', 'Debt restructuring', RiskLevel.HIGH),
        (r'credit\s+(?:rating\s+)?downgrade', 'Credit downgrade', RiskLevel.HIGH),
        (r'liquidity\s+(?:crisis|problem|concern|constraint|shortage)', 'Liquidity issues', RiskLevel.HIGH),
        (r'cash\s+(?:flow\s+)?(?:crisis|shortage|constraint|problem)', 'Cash flow problems', RiskLevel.HIGH),
        (r'insolvency|insolvent', 'Insolvency', RiskLevel.CRITICAL),
        (r'bankruptcy|chapter\s+11|chapter\s+7', 'Bankruptcy', RiskLevel.CRITICAL),
    ]
    
    # Litigation & Legal Risks
    LITIGATION_PATTERNS = [
        (r'(?:material|significant|substantial)\s+(?:legal\s+)?(?:proceeding|litigation|lawsuit|claim)', 'Material litigation', RiskLevel.HIGH),
        (r'class\s+action\s+(?:lawsuit|suit|litigation)', 'Class action lawsuit', RiskLevel.HIGH),
        (r'securities\s+(?:litigation|fraud|investigation)', 'Securities litigation', RiskLevel.HIGH),
        (r'(?:SEC|DOJ|regulatory)\s+(?:investigation|inquiry|enforcement)', 'Regulatory investigation', RiskLevel.HIGH),
        (r'(?:criminal|fraud)\s+(?:investigation|charge|indictment)', 'Criminal investigation', RiskLevel.CRITICAL),
        (r'(?:settlement|judgment|verdict)\s+(?:of|for|exceeding)\s+\$?\d+\s*(?:million|billion)', 'Large settlement/judgment', RiskLevel.HIGH),
        (r'patent\s+(?:infringement|litigation)', 'Patent litigation', RiskLevel.MEDIUM),
        (r'antitrust\s+(?:investigation|litigation|violation)', 'Antitrust issues', RiskLevel.HIGH),
    ]
    
    # Internal Control Weaknesses
    CONTROL_WEAKNESS_PATTERNS = [
        (r'material\s+weakness(?:es)?\s+in\s+internal\s+control', 'Material weakness in internal controls', RiskLevel.HIGH),
        (r'significant\s+deficien(?:cy|cies)\s+in\s+internal\s+control', 'Significant control deficiency', RiskLevel.MEDIUM),
        (r'(?:inadequate|ineffective)\s+internal\s+control', 'Inadequate internal controls', RiskLevel.HIGH),
        (r'restatement\s+of\s+(?:financial\s+)?(?:statements?|results?)', 'Financial restatement', RiskLevel.HIGH),
        (r'(?:accounting\s+)?irregularit(?:y|ies)', 'Accounting irregularities', RiskLevel.HIGH),
        (r'(?:accounting\s+)?(?:error|misstatement)', 'Accounting errors', RiskLevel.MEDIUM),
    ]
    
    # Management & Governance Risks
    MANAGEMENT_PATTERNS = [
        (r'(?:CEO|CFO|director|executive)\s+(?:resign|departure|terminated|left)', 'Executive departure', RiskLevel.MEDIUM),
        (r'auditor\s+(?:resign|change|replacement|dismissal)', 'Auditor change', RiskLevel.MEDIUM),
        (r'related\s+party\s+transaction', 'Related party transactions', RiskLevel.MEDIUM),
        (r'conflict\s+of\s+interest', 'Conflict of interest', RiskLevel.MEDIUM),
        (r'(?:whistleblower|qui\s+tam)\s+(?:complaint|allegation)', 'Whistleblower complaint', RiskLevel.HIGH),
    ]
    
    # Operational Risks
    OPERATIONAL_PATTERNS = [
        (r'(?:loss|termination)\s+of\s+(?:key|major|principal)\s+(?:customer|client|contract)', 'Loss of key customer', RiskLevel.HIGH),
        (r'(?:loss|termination)\s+of\s+(?:key|major|principal)\s+(?:supplier|vendor)', 'Loss of key supplier', RiskLevel.MEDIUM),
        (r'product\s+recall', 'Product recall', RiskLevel.HIGH),
        (r'cybersecurity\s+(?:breach|incident|attack)', 'Cybersecurity incident', RiskLevel.HIGH),
        (r'data\s+breach', 'Data breach', RiskLevel.HIGH),
        (r'environmental\s+(?:liability|contamination|violation)', 'Environmental issues', RiskLevel.MEDIUM),
        (r'(?:plant|facility|factory)\s+(?:closure|shutdown)', 'Facility closure', RiskLevel.MEDIUM),
        (r'(?:significant|material)\s+(?:impairment|write-?down|write-?off)', 'Asset impairment', RiskLevel.MEDIUM),
    ]
    
    # Revenue Recognition Red Flags (from Financial Shenanigans)
    REVENUE_RED_FLAGS = [
        (r'bill\s+and\s+hold', 'Bill and hold revenue', RiskLevel.HIGH),
        (r'channel\s+stuffing', 'Channel stuffing', RiskLevel.HIGH),
        (r'(?:change|revision)\s+(?:in|to)\s+revenue\s+recognition', 'Revenue recognition change', RiskLevel.MEDIUM),
        (r'(?:unusual|significant)\s+(?:increase|growth)\s+in\s+(?:accounts?\s+)?receivable', 'Unusual receivables growth', RiskLevel.MEDIUM),
        (r'(?:extended|longer)\s+(?:payment\s+)?terms', 'Extended payment terms', RiskLevel.LOW),
        (r'side\s+(?:letter|agreement)', 'Side agreements', RiskLevel.HIGH),
    ]
    
    # Leverage & Funding Risks (critical for financial institutions)
    LEVERAGE_FUNDING_PATTERNS = [
        (r'leverage\s+ratio\s+(?:of\s+|was\s+)?(?:\d+[:\s]*1|\d+\s*to\s*1)', 'High leverage ratio disclosed', RiskLevel.HIGH),
        (r'(?:gross\s+)?leverage\s+(?:ratio\s+)?(?:of\s+|was\s+)?(?:2[5-9]|[3-9]\d)[:to\s]+1', 'Extreme leverage (25:1+)', RiskLevel.CRITICAL),
        (r'short[\s-]term\s+(?:funding|borrowing|financing)', 'Short-term funding dependence', RiskLevel.MEDIUM),
        (r'(?:repo|repurchase)\s+(?:agreement|transaction)', 'Repurchase agreement exposure', RiskLevel.MEDIUM),
        (r'overnight\s+(?:funding|borrowing|repo)', 'Overnight funding reliance', RiskLevel.HIGH),
        (r'(?:loss|disruption)\s+(?:of\s+)?(?:confidence|funding|liquidity)', 'Funding/confidence risk', RiskLevel.HIGH),
        (r'(?:secured|unsecured)\s+financing\s+(?:of\s+)?\$?\d+\s*(?:billion|million)', 'Large financing exposure', RiskLevel.MEDIUM),
        (r'funding\s+(?:concentration|dependence|reliance)', 'Funding concentration', RiskLevel.MEDIUM),
    ]
    
    # Valuation & Fair Value Risks
    VALUATION_PATTERNS = [
        (r'level\s+3\s+(?:assets?|instruments?|investments?)', 'Level 3 assets (hard to value)', RiskLevel.HIGH),
        (r'(?:unobservable|significant)\s+(?:inputs?|assumptions?)', 'Unobservable valuation inputs', RiskLevel.MEDIUM),
        (r'(?:fair\s+value|valuation)\s+(?:uncertainty|estimation|judgment)', 'Valuation uncertainty', RiskLevel.MEDIUM),
        (r'(?:mark[\s-]to[\s-]model|model[\s-]based)\s+(?:valuation|pricing)', 'Model-based valuation', RiskLevel.MEDIUM),
        (r'illiquid\s+(?:assets?|investments?|positions?|securities)', 'Illiquid assets', RiskLevel.HIGH),
        (r'(?:hard|difficult)\s+to\s+(?:value|price|sell)', 'Hard to value assets', RiskLevel.MEDIUM),
        (r'mark[\s-]to[\s-]market\s+(?:accounting|valuation|revenue)', 'Mark-to-market accounting', RiskLevel.MEDIUM),
        (r'(?:management|significant)\s+(?:judgment|estimates?|assumptions?)', 'Significant management judgment', RiskLevel.MEDIUM),
    ]
    
    # Concentration & Exposure Risks
    CONCENTRATION_PATTERNS = [
        (r'(?:significant|substantial|material)\s+(?:exposure|concentration)\s+(?:to|in)', 'Significant concentration/exposure', RiskLevel.MEDIUM),
        (r'(?:subprime|mortgage[\s-]?backed|mbs|cdo|clo)', 'Subprime/structured product exposure', RiskLevel.HIGH),
        (r'(?:real\s+estate|property|housing)\s+(?:exposure|concentration|portfolio)', 'Real estate concentration', RiskLevel.MEDIUM),
        (r'counterparty\s+(?:risk|exposure|concentration)', 'Counterparty risk', RiskLevel.MEDIUM),
        (r'(?:single|one)\s+(?:customer|counterparty|borrower)\s+(?:represents?|accounts?\s+for)\s+(?:\d+%|significant)', 'Single counterparty concentration', RiskLevel.HIGH),
        (r'derivative\s+(?:exposure|positions?|transactions?)\s+(?:of\s+)?\$?\d+\s*(?:billion|trillion)', 'Large derivative exposure', RiskLevel.HIGH),
    ]
    
    # Market Stress Indicators  
    MARKET_STRESS_PATTERNS = [
        (r'(?:credit|market|financial)\s+(?:crisis|stress|turmoil|disruption)', 'Market stress conditions', RiskLevel.HIGH),
        (r'(?:significant|substantial|material)\s+(?:decline|deterioration|loss)', 'Significant decline/loss', RiskLevel.MEDIUM),
        (r'(?:adverse|difficult|challenging)\s+(?:market|economic)\s+(?:conditions?|environment)', 'Adverse market conditions', RiskLevel.MEDIUM),
        (r'(?:housing|property|real\s+estate)\s+(?:market\s+)?(?:decline|downturn|crisis)', 'Housing market decline', RiskLevel.HIGH),
    ]
    
    # Enron-style Fraud Indicators (SPEs, Related Party, Complex Structures)
    FRAUD_STRUCTURE_PATTERNS = [
        (r'(?:special\s+purpose|structured)\s+(?:entity|entities|vehicle)', 'Special Purpose Entities (SPE)', RiskLevel.HIGH),
        (r'off[\s-]balance[\s-]sheet\s+(?:arrangement|obligation|financing|entity|debt)', 'Off-balance sheet arrangements', RiskLevel.HIGH),
        (r'(?:CFO|CEO|officer|executive)\s+(?:is\s+a\s+)?(?:managing\s+member|principal|partner)', 'Executive with outside entity interest', RiskLevel.CRITICAL),
        (r'(?:officer|executive|director)\s+(?:received|earn|paid)\s+(?:compensation|fees|million)', 'Executive compensation from related party', RiskLevel.HIGH),
        (r'conflict\s+of\s+interest', 'Conflict of interest disclosed', RiskLevel.HIGH),
        (r'(?:arm\'?s?\s+length|fair\s+to\s+(?:the\s+)?company)', 'Arm\'s length assertion (verify carefully)', RiskLevel.MEDIUM),
        (r'(?:unconsolidated|variable\s+interest)\s+(?:affiliate|entity|subsidiary)', 'Unconsolidated entities', RiskLevel.MEDIUM),
        (r'(?:downgrade|rating)\s+(?:could|would|may)\s+trigger', 'Rating downgrade trigger', RiskLevel.HIGH),
        (r'(?:extensive|numerous|significant)\s+(?:related\s+party\s+)?transactions?\s+(?:with|totaling)\s+(?:\$|over)', 'Extensive related party transactions', RiskLevel.HIGH),
        (r'(?:sold|transferred)\s+(?:assets?|interests?)\s+to\s+(?:related|affiliated)', 'Asset transfers to related parties', RiskLevel.HIGH),
    ]
    
    # Chinese language patterns (for Chinese annual reports)
    CHINESE_PATTERNS = [
        # Simplified Chinese
        (r'持[续續][经經][营營]', 'Going concern (Chinese)', RiskLevel.CRITICAL),
        (r'重大不确定性', 'Material uncertainty (Chinese)', RiskLevel.CRITICAL),
        (r'保留意[见見]', 'Qualified opinion (Chinese)', RiskLevel.HIGH),
        (r'否定意[见見]', 'Adverse opinion (Chinese)', RiskLevel.CRITICAL),
        (r'无法表示意[见見]', 'Disclaimer of opinion (Chinese)', RiskLevel.CRITICAL),
        (r'[亏虧]损', 'Loss (Chinese)', RiskLevel.MEDIUM),
        (r'[负負]债[率比]', 'Debt ratio (Chinese)', RiskLevel.MEDIUM),
        (r'流[动動]性[风風][险險]', 'Liquidity risk (Chinese)', RiskLevel.MEDIUM),
        (r'[违違][约約]', 'Default (Chinese)', RiskLevel.CRITICAL),
        (r'诉讼', 'Litigation (Chinese)', RiskLevel.MEDIUM),
        (r'资不抵债', 'Insolvent (Chinese)', RiskLevel.CRITICAL),
        
        # Traditional Chinese (Hong Kong annual reports)
        (r'無法表示意見', 'Disclaimer of opinion (Traditional Chinese)', RiskLevel.CRITICAL),
        (r'保留意見', 'Qualified opinion (Traditional Chinese)', RiskLevel.HIGH),
        (r'否定意見', 'Adverse opinion (Traditional Chinese)', RiskLevel.CRITICAL),
        (r'持續經營', 'Going concern (Traditional Chinese)', RiskLevel.CRITICAL),
        (r'重大不確定', 'Material uncertainty (Traditional Chinese)', RiskLevel.CRITICAL),
        (r'虧損', 'Loss (Traditional Chinese)', RiskLevel.MEDIUM),
        (r'淨虧損', 'Net loss (Traditional Chinese)', RiskLevel.HIGH),
        (r'負債', 'Liabilities (Traditional Chinese)', RiskLevel.LOW),
        (r'流動負債超過', 'Current liabilities exceed (Traditional Chinese)', RiskLevel.HIGH),
        (r'資不抵債', 'Insolvent (Traditional Chinese)', RiskLevel.CRITICAL),
        (r'違約', 'Default (Traditional Chinese)', RiskLevel.CRITICAL),
        (r'訴訟', 'Litigation (Traditional Chinese)', RiskLevel.MEDIUM),
        (r'清盤', 'Liquidation/Winding up (Traditional Chinese)', RiskLevel.CRITICAL),
        (r'破產', 'Bankruptcy (Traditional Chinese)', RiskLevel.CRITICAL),
        (r'核數師', 'Auditor (Traditional Chinese)', RiskLevel.INFO),
        (r'獨立核數師報告', 'Independent Auditor Report (Traditional Chinese)', RiskLevel.INFO),
        
        # Numbers indicating severe losses (works for both)
        (r'(?:淨?虧損|净?亏损).*?[\d,]+(?:百萬|百万|億|亿)', 'Large loss amount (Chinese)', RiskLevel.HIGH),
        (r'負債總額超過.*?資產總額', 'Liabilities exceed assets (Traditional Chinese)', RiskLevel.CRITICAL),
        (r'负债总额超过.*?资产总额', 'Liabilities exceed assets (Simplified Chinese)', RiskLevel.CRITICAL),
    ]
    
    # Numeric patterns for detecting severe financial distress
    NUMERIC_DISTRESS_PATTERNS = [
        # Large losses (negative numbers in financial statements context)
        (r'\(6[0-9]{2},\d{3}\)', 'Large loss (~600B+ in parentheses)', RiskLevel.CRITICAL),
        (r'\(4[0-9]{2},\d{3}\)', 'Large negative equity (~400B+ in parentheses)', RiskLevel.CRITICAL),
        (r'-\s*6[0-9]{2},\d{3}', 'Large loss (~600B+ negative)', RiskLevel.CRITICAL),
        (r'-\s*4[0-9]{2},\d{3}', 'Large negative amount (~400B+)', RiskLevel.CRITICAL),
        # Negative equity patterns
        (r'\(3[0-9]{2},\d{3}\).*?(?:equity|權益|权益)', 'Negative equity (~300B+)', RiskLevel.CRITICAL),
        (r'\(4[0-9]{2},\d{3}\).*?(?:equity|權益|权益)', 'Negative equity (~400B+)', RiskLevel.CRITICAL),
    ]
    
    # Important Sections to Extract
    SECTION_HEADERS = [
        r'independent\s+auditor\'?s?\s+report',
        r'report\s+of\s+independent\s+(?:registered\s+public\s+)?(?:accounting\s+firm|auditor)',
        r'risk\s+factors?',
        r'(?:management\'?s?\s+)?discussion\s+and\s+analysis',
        r'(?:md&a|mda)',
        r'legal\s+proceedings?',
        r'litigation',
        r'commitments?\s+and\s+contingenc(?:y|ies)',
        r'going\s+concern',
        r'subsequent\s+events?',
        r'critical\s+audit\s+matters?',
        r'key\s+audit\s+matters?',
    ]
    
    def __init__(self):
        """Initialize the extractor"""
        # Compile all patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns"""
        self.compiled_auditor = {
            k: [re.compile(p, re.IGNORECASE) for p in v]
            for k, v in self.AUDITOR_OPINION_PATTERNS.items()
        }
        
        self.compiled_risks = []
        for pattern_list in [
            self.FINANCIAL_DISTRESS_PATTERNS,
            self.LITIGATION_PATTERNS,
            self.CONTROL_WEAKNESS_PATTERNS,
            self.MANAGEMENT_PATTERNS,
            self.OPERATIONAL_PATTERNS,
            self.REVENUE_RED_FLAGS,
            self.LEVERAGE_FUNDING_PATTERNS,
            self.VALUATION_PATTERNS,
            self.CONCENTRATION_PATTERNS,
            self.MARKET_STRESS_PATTERNS,
            self.CHINESE_PATTERNS,
            self.NUMERIC_DISTRESS_PATTERNS,
            self.FRAUD_STRUCTURE_PATTERNS,
        ]:
            for pattern, desc, level in pattern_list:
                self.compiled_risks.append(
                    (re.compile(pattern, re.IGNORECASE), desc, level)
                )
        
        self.compiled_sections = [
            re.compile(p, re.IGNORECASE) for p in self.SECTION_HEADERS
        ]
    
    def extract_risks(self, text: str, company_name: str = "Company") -> RiskReport:
        """
        Main method: Extract all risk warnings from annual report text
        
        Args:
            text: Full text of the annual report
            company_name: Name of the company
            
        Returns:
            RiskReport with all findings
        """
        report = RiskReport(company_name=company_name)
        
        # 1. Analyze auditor's opinion (most critical)
        self._analyze_auditor_opinion(text, report)
        
        # 2. Extract risk warnings by pattern matching
        self._extract_pattern_risks(text, report)
        
        # 3. Extract context around key findings
        self._extract_context(text, report)
        
        # 4. Calculate overall risk level
        self._calculate_overall_risk(report)
        
        # 5. Generate summary
        self._generate_summary(report)
        
        return report
    
    def _analyze_auditor_opinion(self, text: str, report: RiskReport):
        """Analyze the auditor's opinion section"""
        
        # Check for going concern (most critical)
        for pattern in self.compiled_auditor['going_concern']:
            if pattern.search(text):
                report.going_concern = True
                report.warnings.append(RiskWarning(
                    category="Auditor Opinion",
                    description="GOING CONCERN WARNING - Substantial doubt about ability to continue operations",
                    level=RiskLevel.CRITICAL,
                    source_text=self._extract_surrounding_text(text, pattern)
                ))
                break
        
        # Check opinion type
        if any(p.search(text) for p in self.compiled_auditor['adverse']):
            report.auditor_opinion = "Adverse"
            report.warnings.append(RiskWarning(
                category="Auditor Opinion",
                description="ADVERSE OPINION - Financial statements do not present fairly",
                level=RiskLevel.CRITICAL
            ))
        elif any(p.search(text) for p in self.compiled_auditor['disclaimer']):
            report.auditor_opinion = "Disclaimer"
            report.warnings.append(RiskWarning(
                category="Auditor Opinion", 
                description="DISCLAIMER OF OPINION - Auditor unable to form opinion",
                level=RiskLevel.CRITICAL
            ))
        elif any(p.search(text) for p in self.compiled_auditor['qualified']):
            report.auditor_opinion = "Qualified"
            report.warnings.append(RiskWarning(
                category="Auditor Opinion",
                description="QUALIFIED OPINION - Exception noted in audit report",
                level=RiskLevel.HIGH
            ))
        elif any(p.search(text) for p in self.compiled_auditor['emphasis_of_matter']):
            report.auditor_opinion = "Unqualified with Emphasis"
            report.warnings.append(RiskWarning(
                category="Auditor Opinion",
                description="Emphasis of Matter paragraph included",
                level=RiskLevel.MEDIUM
            ))
        else:
            report.auditor_opinion = "Unqualified (Clean)"
    
    def _extract_pattern_risks(self, text: str, report: RiskReport):
        """Extract risks based on pattern matching"""
        seen_descriptions = set()  # Avoid duplicates
        
        for pattern, description, level in self.compiled_risks:
            if pattern.search(text) and description not in seen_descriptions:
                seen_descriptions.add(description)
                report.warnings.append(RiskWarning(
                    category=self._categorize_risk(description),
                    description=description,
                    level=level,
                    source_text=self._extract_surrounding_text(text, pattern)
                ))
    
    def _extract_surrounding_text(self, text: str, pattern, chars: int = 200) -> str:
        """Extract text surrounding a pattern match"""
        match = pattern.search(text)
        if match:
            start = max(0, match.start() - chars)
            end = min(len(text), match.end() + chars)
            return "..." + text[start:end].strip() + "..."
        return ""
    
    def _categorize_risk(self, description: str) -> str:
        """Categorize a risk based on its description"""
        desc_lower = description.lower()
        if any(x in desc_lower for x in ['litigation', 'lawsuit', 'legal', 'investigation', 'sec', 'doj']):
            return "Legal/Regulatory"
        elif any(x in desc_lower for x in ['loss', 'deficit', 'negative', 'default', 'breach', 'liquidity', 'insolvency']):
            return "Financial Distress"
        elif any(x in desc_lower for x in ['control', 'restatement', 'irregularity', 'error']):
            return "Internal Controls"
        elif any(x in desc_lower for x in ['ceo', 'cfo', 'director', 'auditor', 'executive']):
            return "Governance"
        elif any(x in desc_lower for x in ['revenue', 'receivable', 'bill and hold', 'channel']):
            return "Revenue Quality"
        else:
            return "Operational"
    
    def _extract_context(self, text: str, report: RiskReport):
        """Extract additional context from key sections"""
        # This could be expanded to use NLP for better extraction
        pass
    
    def _calculate_overall_risk(self, report: RiskReport):
        """Calculate overall risk level"""
        if not report.warnings:
            report.overall_risk = RiskLevel.LOW
            return
        
        # Count by severity
        critical = sum(1 for w in report.warnings if w.level == RiskLevel.CRITICAL)
        high = sum(1 for w in report.warnings if w.level == RiskLevel.HIGH)
        
        if critical >= 2 or report.going_concern:
            report.overall_risk = RiskLevel.CRITICAL
        elif critical >= 1 or high >= 3:
            report.overall_risk = RiskLevel.HIGH
        elif high >= 1:
            report.overall_risk = RiskLevel.MEDIUM
        else:
            report.overall_risk = RiskLevel.LOW
    
    def _generate_summary(self, report: RiskReport):
        """Generate a summary of findings"""
        lines = []
        
        # Auditor opinion
        lines.append(f"Auditor Opinion: {report.auditor_opinion}")
        
        if report.going_concern:
            lines.append("⚠️  GOING CONCERN WARNING PRESENT")
        
        # Count by category
        categories = {}
        for w in report.warnings:
            categories[w.category] = categories.get(w.category, 0) + 1
        
        if categories:
            lines.append(f"\nWarnings by Category:")
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                lines.append(f"  - {cat}: {count}")
        
        # Count by severity
        severity_counts = {}
        for w in report.warnings:
            severity_counts[w.level.name] = severity_counts.get(w.level.name, 0) + 1
        
        if severity_counts:
            lines.append(f"\nWarnings by Severity:")
            for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
                if sev in severity_counts:
                    lines.append(f"  - {sev}: {severity_counts[sev]}")
        
        report.summary = "\n".join(lines)
    
    def print_report(self, report: RiskReport):
        """Print a formatted risk report"""
        print("=" * 80)
        print(f"RISK WARNING REPORT: {report.company_name}")
        print("=" * 80)
        
        # Overall risk
        risk_emoji = {
            RiskLevel.CRITICAL: "🔴",
            RiskLevel.HIGH: "🟠", 
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.LOW: "🟢",
            RiskLevel.INFO: "ℹ️"
        }
        print(f"\n{risk_emoji[report.overall_risk]} OVERALL RISK LEVEL: {report.overall_risk.name}")
        print(f"\n📋 Auditor Opinion: {report.auditor_opinion}")
        
        if report.going_concern:
            print("\n⚠️  *** GOING CONCERN WARNING PRESENT ***")
        
        # Warnings by category
        if report.warnings:
            print(f"\n{'─' * 60}")
            print(f"WARNINGS DETECTED: {len(report.warnings)}")
            print(f"{'─' * 60}")
            
            # Group by category
            by_category = {}
            for w in report.warnings:
                if w.category not in by_category:
                    by_category[w.category] = []
                by_category[w.category].append(w)
            
            for category, warnings in sorted(by_category.items()):
                print(f"\n📁 {category}:")
                for w in sorted(warnings, key=lambda x: -x.level.value):
                    level_str = f"[{w.level.name}]"
                    print(f"   {risk_emoji[w.level]} {level_str:<10} {w.description}")
                    if w.source_text:
                        # Print truncated source
                        source = w.source_text[:150] + "..." if len(w.source_text) > 150 else w.source_text
                        print(f"      └─ {source}")
        else:
            print("\n✅ No significant risk warnings detected")
        
        print("\n" + "=" * 80)


# =============================================================================
# Helper function for PDF text extraction
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file
    
    Requires: pdfplumber or PyPDF2
    """
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts)
    except ImportError:
        pass
    
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n".join(text_parts)
    except ImportError:
        pass
    
    # Fallback to subprocess pdftotext
    import subprocess
    try:
        result = subprocess.run(
            ['pdftotext', '-layout', pdf_path, '-'],
            capture_output=True,
            text=True
        )
        return result.stdout
    except:
        raise ImportError("No PDF library available. Install pdfplumber or PyPDF2")


# =============================================================================
# Main entry point
# =============================================================================

def analyze_annual_report(pdf_path: str = None, text: str = None, 
                          company_name: str = "Company") -> RiskReport:
    """
    Analyze an annual report for risk warnings
    
    Args:
        pdf_path: Path to PDF file
        text: Raw text (if PDF already extracted)
        company_name: Company name for the report
        
    Returns:
        RiskReport with all findings
    """
    if text is None and pdf_path:
        text = extract_text_from_pdf(pdf_path)
    
    if not text:
        raise ValueError("Either pdf_path or text must be provided")
    
    extractor = AnnualReportRiskExtractor()
    return extractor.extract_risks(text, company_name)


# =============================================================================
# Year-over-Year Comparison Class
# =============================================================================

@dataclass
class YoYChange:
    """A single year-over-year change detected"""
    category: str
    description: str
    level: RiskLevel
    prior_value: str = ""
    current_value: str = ""
    change_type: str = ""  # 'NEW', 'REMOVED', 'WORSENED', 'IMPROVED'


@dataclass 
class YoYComparisonReport:
    """Year-over-year comparison report"""
    company_name: str
    prior_year: str
    current_year: str
    changes: List[YoYChange] = field(default_factory=list)
    risk_escalation: str = ""  # 'SIGNIFICANT', 'MODERATE', 'MINIMAL', 'IMPROVED'
    summary: str = ""


class AnnualReportComparator:
    """
    Compare two annual reports to detect worsening risk signals
    
    Key detection:
    1. Risk Factors section length changes
    2. New risk keywords appearing
    3. Wording escalation (may -> will, possible -> likely)
    4. Auditor opinion changes
    5. New warning types
    """
    
    # Wording escalation patterns
    WORDING_LEVELS = {
        'mild': ['may', 'could', 'might', 'possible', 'potential', 'uncertain', 'risk'],
        'moderate': ['likely', 'expected', 'probable', 'significant', 'material', 'substantial'],
        'severe': ['will', 'imminent', 'immediate', 'critical', 'unable', 'default', 'breach', 
                   'failed', 'failure', 'insolvent', 'bankruptcy'],
    }
    
    # Key risk phrases to track
    TRACKED_PHRASES = [
        # Financial health
        'going concern',
        'liquidity',
        'cash flow',
        'debt',
        'covenant',
        'default',
        'refinancing',
        'borrowing',
        # Operations
        'loss of key customer',
        'material weakness',
        'internal control',
        # Legal
        'litigation',
        'investigation',
        'class action',
        'SEC',
        'regulatory',
    ]
    
    def __init__(self):
        self.extractor = AnnualReportRiskExtractor()
    
    def compare(self, prior_text: str, current_text: str, 
                company_name: str, prior_year: str, current_year: str) -> YoYComparisonReport:
        """
        Compare two annual reports and identify changes
        
        Args:
            prior_text: Text of prior year annual report
            current_text: Text of current year annual report
            company_name: Company name
            prior_year: Prior year label (e.g., "2020")
            current_year: Current year label (e.g., "2021")
            
        Returns:
            YoYComparisonReport with all detected changes
        """
        report = YoYComparisonReport(
            company_name=company_name,
            prior_year=prior_year,
            current_year=current_year
        )
        
        # 1. Extract risks from both years
        prior_risks = self.extractor.extract_risks(prior_text, f"{company_name} {prior_year}")
        current_risks = self.extractor.extract_risks(current_text, f"{company_name} {current_year}")
        
        # 2. Compare auditor opinions
        self._compare_auditor_opinions(prior_risks, current_risks, report)
        
        # 3. Compare going concern status
        self._compare_going_concern(prior_risks, current_risks, report)
        
        # 4. Compare risk section length
        self._compare_risk_section_length(prior_text, current_text, report)
        
        # 5. Detect new warnings
        self._detect_new_warnings(prior_risks, current_risks, report)
        
        # 6. Detect wording escalation
        self._detect_wording_escalation(prior_text, current_text, report)
        
        # 7. Track key phrase changes
        self._track_phrase_frequency(prior_text, current_text, report)
        
        # 8. Calculate overall escalation
        self._calculate_escalation(report)
        
        # 9. Generate summary
        self._generate_comparison_summary(prior_risks, current_risks, report)
        
        return report
    
    def _compare_auditor_opinions(self, prior: RiskReport, current: RiskReport, 
                                   report: YoYComparisonReport):
        """Compare auditor opinions between years"""
        opinion_severity = {
            'Unqualified (Clean)': 0,
            'Unqualified with Emphasis': 1,
            'Qualified': 2,
            'Adverse': 3,
            'Disclaimer': 4,
            'Unknown': -1,
        }
        
        prior_sev = opinion_severity.get(prior.auditor_opinion, -1)
        current_sev = opinion_severity.get(current.auditor_opinion, -1)
        
        if current_sev > prior_sev and prior_sev >= 0:
            report.changes.append(YoYChange(
                category="Auditor Opinion",
                description=f"Auditor opinion WORSENED: {prior.auditor_opinion} → {current.auditor_opinion}",
                level=RiskLevel.CRITICAL if current_sev >= 3 else RiskLevel.HIGH,
                prior_value=prior.auditor_opinion,
                current_value=current.auditor_opinion,
                change_type="WORSENED"
            ))
        elif current_sev < prior_sev and current_sev >= 0:
            report.changes.append(YoYChange(
                category="Auditor Opinion",
                description=f"Auditor opinion improved: {prior.auditor_opinion} → {current.auditor_opinion}",
                level=RiskLevel.INFO,
                prior_value=prior.auditor_opinion,
                current_value=current.auditor_opinion,
                change_type="IMPROVED"
            ))
    
    def _compare_going_concern(self, prior: RiskReport, current: RiskReport,
                                report: YoYComparisonReport):
        """Compare going concern status"""
        if current.going_concern and not prior.going_concern:
            report.changes.append(YoYChange(
                category="Going Concern",
                description="NEW GOING CONCERN WARNING - Not present in prior year!",
                level=RiskLevel.CRITICAL,
                prior_value="No",
                current_value="Yes",
                change_type="NEW"
            ))
        elif not current.going_concern and prior.going_concern:
            report.changes.append(YoYChange(
                category="Going Concern",
                description="Going concern warning removed",
                level=RiskLevel.INFO,
                prior_value="Yes",
                current_value="No",
                change_type="IMPROVED"
            ))
    
    def _compare_risk_section_length(self, prior_text: str, current_text: str,
                                      report: YoYComparisonReport):
        """Compare risk factors section length"""
        def extract_risk_section(text):
            # Try to find Risk Factors section
            pattern = r'(?:risk\s+factors?)(.*?)(?:item\s+\d|management|legal\s+proceedings|$)'
            match = re.search(pattern, text.lower(), re.DOTALL)
            if match:
                return match.group(1)
            return ""
        
        prior_section = extract_risk_section(prior_text)
        current_section = extract_risk_section(current_text)
        
        prior_len = len(prior_section)
        current_len = len(current_section)
        
        if prior_len > 0 and current_len > 0:
            change_pct = (current_len - prior_len) / prior_len * 100
            
            if change_pct > 30:
                report.changes.append(YoYChange(
                    category="Risk Factors Section",
                    description=f"Risk Factors section INCREASED by {change_pct:.0f}%",
                    level=RiskLevel.HIGH if change_pct > 50 else RiskLevel.MEDIUM,
                    prior_value=f"{prior_len:,} chars",
                    current_value=f"{current_len:,} chars",
                    change_type="WORSENED"
                ))
            elif change_pct < -30:
                report.changes.append(YoYChange(
                    category="Risk Factors Section",
                    description=f"Risk Factors section decreased by {abs(change_pct):.0f}%",
                    level=RiskLevel.INFO,
                    prior_value=f"{prior_len:,} chars",
                    current_value=f"{current_len:,} chars",
                    change_type="IMPROVED"
                ))
    
    def _detect_new_warnings(self, prior: RiskReport, current: RiskReport,
                              report: YoYComparisonReport):
        """Detect warnings that are new in current year"""
        prior_descriptions = {w.description for w in prior.warnings}
        
        for warning in current.warnings:
            if warning.description not in prior_descriptions:
                if warning.level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                    report.changes.append(YoYChange(
                        category="New Warning",
                        description=f"NEW: {warning.description}",
                        level=warning.level,
                        prior_value="Not present",
                        current_value="Present",
                        change_type="NEW"
                    ))
    
    def _detect_wording_escalation(self, prior_text: str, current_text: str,
                                    report: YoYComparisonReport):
        """Detect escalation in wording (may -> will, possible -> certain)"""
        prior_lower = prior_text.lower()
        current_lower = current_text.lower()
        
        # Count words at each severity level
        def count_level(text, words):
            return sum(len(re.findall(r'\b' + w + r'\b', text)) for w in words)
        
        prior_mild = count_level(prior_lower, self.WORDING_LEVELS['mild'])
        prior_severe = count_level(prior_lower, self.WORDING_LEVELS['severe'])
        current_mild = count_level(current_lower, self.WORDING_LEVELS['mild'])
        current_severe = count_level(current_lower, self.WORDING_LEVELS['severe'])
        
        # Check for escalation
        if current_severe > prior_severe * 1.5 and current_severe > 10:
            report.changes.append(YoYChange(
                category="Wording Escalation",
                description=f"Severe risk language increased: {prior_severe} → {current_severe} occurrences",
                level=RiskLevel.HIGH,
                prior_value=str(prior_severe),
                current_value=str(current_severe),
                change_type="WORSENED"
            ))
        
        # Check ratio change
        prior_ratio = prior_severe / max(prior_mild, 1)
        current_ratio = current_severe / max(current_mild, 1)
        
        if current_ratio > prior_ratio * 2 and current_ratio > 0.5:
            report.changes.append(YoYChange(
                category="Wording Tone",
                description=f"Language tone shifted from cautious to definitive",
                level=RiskLevel.MEDIUM,
                prior_value=f"Ratio: {prior_ratio:.2f}",
                current_value=f"Ratio: {current_ratio:.2f}",
                change_type="WORSENED"
            ))
    
    def _track_phrase_frequency(self, prior_text: str, current_text: str,
                                 report: YoYComparisonReport):
        """Track frequency changes in key risk phrases"""
        prior_lower = prior_text.lower()
        current_lower = current_text.lower()
        
        for phrase in self.TRACKED_PHRASES:
            prior_count = len(re.findall(phrase, prior_lower))
            current_count = len(re.findall(phrase, current_lower))
            
            # Significant increase
            if current_count > prior_count * 2 and current_count >= 5:
                report.changes.append(YoYChange(
                    category="Key Phrase Frequency",
                    description=f"'{phrase}' mentions increased: {prior_count} → {current_count}",
                    level=RiskLevel.MEDIUM,
                    prior_value=str(prior_count),
                    current_value=str(current_count),
                    change_type="WORSENED"
                ))
            # New appearance
            elif prior_count == 0 and current_count >= 3:
                report.changes.append(YoYChange(
                    category="Key Phrase",
                    description=f"NEW topic appearing: '{phrase}' ({current_count} times)",
                    level=RiskLevel.MEDIUM,
                    prior_value="0",
                    current_value=str(current_count),
                    change_type="NEW"
                ))
    
    def _calculate_escalation(self, report: YoYComparisonReport):
        """Calculate overall risk escalation level"""
        critical_changes = sum(1 for c in report.changes 
                              if c.level == RiskLevel.CRITICAL and c.change_type in ['NEW', 'WORSENED'])
        high_changes = sum(1 for c in report.changes 
                         if c.level == RiskLevel.HIGH and c.change_type in ['NEW', 'WORSENED'])
        improvements = sum(1 for c in report.changes if c.change_type == 'IMPROVED')
        
        if critical_changes >= 2:
            report.risk_escalation = "SIGNIFICANT"
        elif critical_changes >= 1 or high_changes >= 3:
            report.risk_escalation = "MODERATE"
        elif high_changes >= 1:
            report.risk_escalation = "MINIMAL"
        elif improvements > 0:
            report.risk_escalation = "IMPROVED"
        else:
            report.risk_escalation = "UNCHANGED"
    
    def _generate_comparison_summary(self, prior: RiskReport, current: RiskReport,
                                      report: YoYComparisonReport):
        """Generate summary of changes"""
        lines = [
            f"Comparison: {report.prior_year} vs {report.current_year}",
            f"Risk Escalation: {report.risk_escalation}",
            "",
            f"Prior Year Warnings: {len(prior.warnings)}",
            f"Current Year Warnings: {len(current.warnings)}",
            f"Changes Detected: {len(report.changes)}",
        ]
        
        worsened = sum(1 for c in report.changes if c.change_type == 'WORSENED')
        new = sum(1 for c in report.changes if c.change_type == 'NEW')
        improved = sum(1 for c in report.changes if c.change_type == 'IMPROVED')
        
        if worsened or new:
            lines.append(f"  - Worsened/New: {worsened + new}")
        if improved:
            lines.append(f"  - Improved: {improved}")
        
        report.summary = "\n".join(lines)
    
    def print_comparison_report(self, report: YoYComparisonReport):
        """Print formatted comparison report"""
        print("=" * 80)
        print(f"YEAR-OVER-YEAR COMPARISON: {report.company_name}")
        print(f"{report.prior_year} vs {report.current_year}")
        print("=" * 80)
        
        # Escalation status
        escalation_emoji = {
            'SIGNIFICANT': '🔴',
            'MODERATE': '🟠',
            'MINIMAL': '🟡',
            'UNCHANGED': '⚪',
            'IMPROVED': '🟢',
        }
        emoji = escalation_emoji.get(report.risk_escalation, '⚪')
        print(f"\n{emoji} RISK ESCALATION: {report.risk_escalation}")
        
        if report.changes:
            print(f"\n{'─' * 60}")
            print(f"CHANGES DETECTED: {len(report.changes)}")
            print(f"{'─' * 60}")
            
            # Group by change type
            for change_type in ['NEW', 'WORSENED', 'IMPROVED']:
                changes = [c for c in report.changes if c.change_type == change_type]
                if changes:
                    type_emoji = {'NEW': '🆕', 'WORSENED': '📈', 'IMPROVED': '📉'}.get(change_type, '•')
                    print(f"\n{type_emoji} {change_type}:")
                    for c in sorted(changes, key=lambda x: -x.level.value):
                        level_str = f"[{c.level.name}]"
                        print(f"   {level_str:<12} {c.description}")
                        if c.prior_value and c.current_value:
                            print(f"               {c.prior_value} → {c.current_value}")
        else:
            print("\n✅ No significant changes detected")
        
        print("\n" + "=" * 80)


def compare_annual_reports(prior_pdf: str = None, current_pdf: str = None,
                           prior_text: str = None, current_text: str = None,
                           company_name: str = "Company",
                           prior_year: str = "Prior", current_year: str = "Current") -> YoYComparisonReport:
    """
    Compare two annual reports for risk changes
    
    Args:
        prior_pdf: Path to prior year PDF
        current_pdf: Path to current year PDF
        prior_text: Prior year text (if already extracted)
        current_text: Current year text (if already extracted)
        company_name: Company name
        prior_year: Prior year label
        current_year: Current year label
        
    Returns:
        YoYComparisonReport
    """
    if prior_text is None and prior_pdf:
        prior_text = extract_text_from_pdf(prior_pdf)
    if current_text is None and current_pdf:
        current_text = extract_text_from_pdf(current_pdf)
    
    if not prior_text or not current_text:
        raise ValueError("Both prior and current year texts must be provided")
    
    comparator = AnnualReportComparator()
    return comparator.compare(prior_text, current_text, company_name, prior_year, current_year)


if __name__ == "__main__":
    # Demo with sample text containing various risk indicators
    sample_text = """
    INDEPENDENT AUDITOR'S REPORT
    
    To the Board of Directors and Shareholders of Sample Company:
    
    We have audited the accompanying consolidated financial statements of Sample Company.
    
    Basis for Qualified Opinion
    
    As discussed in Note 15 to the financial statements, the Company has suffered recurring 
    losses from operations and has a net capital deficiency that raise substantial doubt 
    about its ability to continue as a going concern. Management's plans in regard to these 
    matters are also described in Note 15. The financial statements do not include any 
    adjustments that might result from the outcome of this uncertainty.
    
    Qualified Opinion
    
    In our opinion, except for the possible effects of the matter described in the Basis 
    for Qualified Opinion paragraph, the financial statements present fairly, in all 
    material respects, the financial position of the Company.
    
    Emphasis of Matter
    
    We draw attention to Note 20 to the financial statements, which describes a material 
    uncertainty related to the outcome of ongoing litigation against the Company.
    
    RISK FACTORS
    
    The Company faces significant risks including:
    - Liquidity constraints and negative cash flow from operations
    - Material weakness in internal control over financial reporting
    - Pending SEC investigation regarding revenue recognition practices
    - Loss of key customer representing 30% of revenue
    - Class action lawsuit alleging securities fraud
    
    The Company defaulted on its credit facility covenants in Q3 2023.
    
    LEGAL PROCEEDINGS
    
    The Company is subject to various legal proceedings including a class action 
    securities lawsuit filed in the Southern District of New York.
    """
    
    print("=" * 80)
    print("ANNUAL REPORT RISK EXTRACTOR - DEMO")
    print("=" * 80)
    
    extractor = AnnualReportRiskExtractor()
    report = extractor.extract_risks(sample_text, "Sample Company (Demo)")
    extractor.print_report(report)