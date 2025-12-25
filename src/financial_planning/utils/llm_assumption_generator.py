#!/usr/bin/env python3
"""
llm_assumption_generator.py - Enhanced Version

Key improvements:
1. LLM sees the ticker name (AAPL, GOOGL, etc.) - can use external knowledge
2. LLM sees ALL historical ratios including dividends, buybacks, retention
3. LLM can predict ALL ratios, leveraging its vast knowledge about the company
4. More detailed prompt with company context

LLM's advantage: It knows Apple's capital allocation history, competitive position,
product cycles, and strategic direction - not just the 8 quarters we show it.
"""

import os
import re
import json
import pandas as pd
import numpy as np
from typing import Dict, Optional


# Default API key
DEFAULT_API_KEY = os.environ.get(
    "ANTHROPIC_API_KEY", 
    "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"
)


def generate_quarterly_table(df: pd.DataFrame, columns: list) -> str:
    """Generate markdown table rows for quarterly data."""
    rows = []
    for i, (idx, row) in enumerate(df.iterrows()):
        q_label = f"Q{i+1}"
        values = []
        for col in columns:
            if col in df.columns and col in row:
                val = row[col]
                if pd.notna(val) and val != 0:
                    values.append(f"{val/1e9:.2f}")
                else:
                    values.append("-")
            elif col == 'gross_profit' and 'sales_revenue' in row and 'cost_of_goods_sold' in row:
                # Calculate gross profit if not available
                gp = row['sales_revenue'] - row['cost_of_goods_sold']
                values.append(f"{gp/1e9:.2f}")
            else:
                values.append("-")
        rows.append(f"| {q_label} | " + " | ".join(values) + " |")
    return "\n".join(rows)


def generate_margin_table(df: pd.DataFrame) -> str:
    """Generate markdown table for margin trends."""
    rows = []
    for i, (idx, row) in enumerate(df.iterrows()):
        q_label = f"Q{i+1}"
        
        revenue = row.get('sales_revenue', 0)
        if revenue > 0:
            cogs = row.get('cost_of_goods_sold', 0)
            gross_margin = (revenue - cogs) / revenue * 100
            
            ebit = row.get('ebit', 0)
            ebit_margin = ebit / revenue * 100 if ebit else 0
            
            ni = row.get('net_income', 0)
            ni_margin = ni / revenue * 100 if ni else 0
            
            rows.append(f"| {q_label} | {gross_margin:.1f}% | {ebit_margin:.1f}% | {ni_margin:.1f}% |")
        else:
            rows.append(f"| {q_label} | N/A | N/A | N/A |")
    return "\n".join(rows)


def generate_cf_table(df: pd.DataFrame) -> str:
    """Generate cash flow table."""
    rows = []
    for i, (idx, row) in enumerate(df.iterrows()):
        q_label = f"Q{i+1}"
        
        ocf = row.get('operating_cash_flow', 0)
        capex = abs(row.get('capex', 0))
        div = abs(row.get('dividends_paid', 0))
        buyback = abs(row.get('stock_repurchased', 0))
        
        rows.append(f"| {q_label} | {ocf/1e9:.2f} | {capex/1e9:.2f} | {div/1e9:.2f} | {buyback/1e9:.2f} |")
    return "\n".join(rows)


def generate_retention_table(df: pd.DataFrame) -> str:
    """Generate quarterly retention ratio table."""
    rows = []
    for i, (idx, row) in enumerate(df.iterrows()):
        q_label = f"Q{i+1}"
        
        ni = row.get('net_income', 0)
        div = abs(row.get('dividends_paid', 0))
        buyback = abs(row.get('stock_repurchased', 0))
        
        if ni > 0:
            payout = (div + buyback) / ni
            retention = 1 - payout
            rows.append(f"| {q_label} | {ni/1e9:.2f} | {div/1e9:.2f} | {buyback/1e9:.2f} | {payout*100:.1f}% | {retention*100:.1f}% |")
        else:
            rows.append(f"| {q_label} | {ni/1e9:.2f} | {div/1e9:.2f} | {buyback/1e9:.2f} | N/A | N/A |")
    return "\n".join(rows)


def calculate_historical_context(historical_data: pd.DataFrame) -> Dict:
    """Calculate all historical ratios to show LLM."""
    
    df = historical_data
    recent = df.tail(8)
    
    revenue = recent['sales_revenue'].mean()
    
    # Margins
    gross_margin = ((recent['sales_revenue'] - recent['cost_of_goods_sold']) / recent['sales_revenue']).mean()
    ni_margin = (recent['net_income'] / recent['sales_revenue']).mean()
    ebit_margin = (recent['ebit'] / recent['sales_revenue']).mean() if 'ebit' in recent.columns else ni_margin * 1.2
    capex_ratio = (recent['capex'].abs() / recent['sales_revenue']).mean()
    
    # Retention ratio calculation
    total_net_income = df['net_income'].tail(8).sum()
    total_dividends = abs(df['dividends_paid'].tail(8).sum()) if 'dividends_paid' in df.columns else 0
    total_buybacks = abs(df['stock_repurchased'].tail(8).sum()) if 'stock_repurchased' in df.columns else 0
    
    if total_net_income > 0:
        payout_ratio = (total_dividends + total_buybacks) / total_net_income
        retention_ratio = 1 - payout_ratio
    else:
        payout_ratio = 0
        retention_ratio = 0.30
    
    # Other ratios
    cash_ratio = (recent['cash'] / recent['sales_revenue']).mean() if 'cash' in recent.columns else 0.10
    
    # Interest rate
    if 'interest_expense' in recent.columns and 'total_debt' in recent.columns:
        avg_interest = recent['interest_expense'].mean()
        avg_debt = recent['total_debt'].mean()
        if avg_debt > 0 and avg_interest > 0:
            interest_rate = (avg_interest * 4) / avg_debt
        else:
            interest_rate = 0.005
    else:
        interest_rate = 0.02
    
    return {
        'revenue_avg': revenue,
        'gross_margin': gross_margin,
        'ni_margin': ni_margin,
        'ebit_margin': ebit_margin,
        'capex_ratio': capex_ratio,
        'retention_ratio': retention_ratio,
        'payout_ratio': payout_ratio,
        'total_net_income_8q': total_net_income,
        'total_dividends_8q': total_dividends,
        'total_buybacks_8q': total_buybacks,
        'cash_ratio': cash_ratio,
        'interest_rate': interest_rate,
    }


def generate_assumptions(
    ticker: str, 
    historical_data: pd.DataFrame,
    api_key: str = None
) -> Dict:
    """
    Generate financial assumptions using LLM with FULL context.
    
    The LLM sees:
    1. Ticker name (AAPL, GOOGL, etc.) - so it can use its knowledge
    2. Full historical data including dividends, buybacks
    3. Calculated ratios for reference
    
    LLM can adjust ANY ratio based on its knowledge of the company.
    """
    api_key = api_key or DEFAULT_API_KEY
    
    recent = historical_data.tail(8)
    
    # Calculate full historical context
    hist = calculate_historical_context(historical_data)
    
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=api_key)
        
        prompt = f"""
You are a senior financial analyst at JP Morgan. You are analyzing **{ticker}** to generate forecasting assumptions for the next 4 quarters.

## COMPANY: {ticker}

You have extensive knowledge about {ticker} from your training data - its business model, competitive position, recent news, product cycles, management strategy, and capital allocation history.

## HISTORICAL DATA (Last 8 Quarters from FMP API)

### Income Statement Trends (Quarterly, in $B)
| Quarter | Revenue | COGS | Gross Profit | EBIT | Net Income |
|---------|---------|------|--------------|------|------------|
{generate_quarterly_table(recent, ['sales_revenue', 'cost_of_goods_sold', 'gross_profit', 'ebit', 'net_income'])}

### Margin Trends (Quarterly, in %)
| Quarter | Gross Margin | EBIT Margin | Net Margin |
|---------|--------------|-------------|------------|
{generate_margin_table(recent)}

### Balance Sheet Trends (Quarterly, in $B)
| Quarter | Total Assets | Total Liabilities | Total Equity | Cash |
|---------|--------------|-------------------|--------------|------|
{generate_quarterly_table(recent, ['total_assets', 'total_liabilities', 'total_equity', 'cash'])}

### Cash Flow & Capital Allocation (Quarterly, in $B)
| Quarter | Operating CF | CapEx | Dividends | Buybacks |
|---------|--------------|-------|-----------|----------|
{generate_cf_table(recent)}

### Summary Statistics (for reference)
- 8Q Total Net Income: ${hist['total_net_income_8q']/1e9:.2f}B
- 8Q Total Dividends: ${hist['total_dividends_8q']/1e9:.2f}B  
- 8Q Total Buybacks: ${hist['total_buybacks_8q']/1e9:.2f}B
- 8Q Payout Ratio: {hist['payout_ratio']:.1%}
- 8Q Retention Ratio: {hist['retention_ratio']:.1%}

## YOUR TASK

Based on:
1. The quarterly trends shown above (look for patterns, seasonality, changes)
2. Your knowledge of {ticker}'s business, strategy, and market position
3. Any recent developments you know about {ticker}

Provide your assumptions for the NEXT 4 quarters.

Think step by step:
- Are margins trending up, down, or stable?
- Is there seasonality in the data?
- What is the company's capital allocation strategy?
- Any upcoming catalysts or risks?

## OUTPUT FORMAT

Provide ONLY valid JSON:

{{
    "gross_margin": 0.XX,
    "avg_net_income_margin": 0.XX,
    "avg_ebit_margin": 0.XX,
    "capex_to_revenue": 0.XX,
    "retention_ratio": X.XX,
    "reasoning": "Detailed reasoning with specific references to trends you observed"
}}

Note: retention_ratio can be NEGATIVE (like {ticker}'s historical {hist['retention_ratio']:.1%}) if the company pays out more than it earns.
"""
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = response.content[0].text
        
        # Parse JSON from response
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            assumptions = json.loads(json_match.group(0))
            assumptions['source'] = 'LLM'
            assumptions['ticker'] = ticker
            
            # Validate - fill missing with historical
            if 'gross_margin' not in assumptions:
                assumptions['gross_margin'] = float(hist['gross_margin'])
            if 'avg_net_income_margin' not in assumptions:
                assumptions['avg_net_income_margin'] = float(hist['ni_margin'])
            if 'avg_ebit_margin' not in assumptions:
                assumptions['avg_ebit_margin'] = float(hist['ebit_margin'])
            if 'capex_to_revenue' not in assumptions:
                assumptions['capex_to_revenue'] = float(hist['capex_ratio'])
            if 'retention_ratio' not in assumptions:
                assumptions['retention_ratio'] = float(hist['retention_ratio'])
            
            return assumptions
        else:
            raise ValueError("Could not parse JSON from LLM response")
            
    except Exception as e:
        print(f"  LLM Error: {e}")
        print(f"  Using historical-based assumptions...")
        
        # Return historical values as fallback
        return {
            "gross_margin": float(hist['gross_margin']),
            "avg_net_income_margin": float(hist['ni_margin']),
            "avg_ebit_margin": float(hist['ebit_margin']),
            "capex_to_revenue": float(hist['capex_ratio']),
            "retention_ratio": float(hist['retention_ratio']),
            "reasoning": f"[MOCK] Based on {ticker}'s historical averages. LLM unavailable.",
            "source": "MOCK",
            "ticker": ticker
        }


def generate_cfo_report(
    ticker: str,
    historical_data: pd.DataFrame,
    forecast_statements: pd.DataFrame,
    assumptions: Dict,
    api_key: str = None
) -> str:
    """Generate CFO recommendation report using LLM."""
    api_key = api_key or DEFAULT_API_KEY
    
    recent = historical_data.tail(4)
    
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=api_key)
        
        prompt = f"""
You are a senior financial advisor at JP Morgan preparing a CFO briefing for **{ticker}**.

## CURRENT POSITION (Latest Quarter)
- Revenue: ${recent['sales_revenue'].iloc[-1]/1e9:.2f}B
- Net Income: ${recent['net_income'].iloc[-1]/1e9:.2f}B
- Total Assets: ${recent['total_assets'].iloc[-1]/1e9:.2f}B
- Total Equity: ${recent['total_equity'].iloc[-1]/1e9:.2f}B

## FORECAST (Next 4 Quarters Average)
- Revenue: ${forecast_statements['is_revenue'].mean()/1e9:.2f}B
- Net Income: ${forecast_statements['is_net_income'].mean()/1e9:.2f}B
- Net Margin: {forecast_statements['is_net_margin'].mean():.1%}
- Total Assets: ${forecast_statements['bs_total_assets'].mean()/1e9:.2f}B

## ASSUMPTIONS USED
- Gross Margin: {assumptions.get('gross_margin', 0):.1%}
- Net Income Margin: {assumptions.get('avg_net_income_margin', 0):.1%}
- EBIT Margin: {assumptions.get('avg_ebit_margin', 0):.1%}
- CapEx Ratio: {assumptions.get('capex_to_revenue', 0):.1%}
- Retention Ratio: {assumptions.get('retention_ratio', 0):.1%}

## Analyst Reasoning
{assumptions.get('reasoning', 'N/A')}

---

Provide a professional CFO briefing with:
1. **Executive Summary** (2-3 sentences)
2. **Key Opportunities** (3-4 bullet points specific to {ticker})
3. **Key Risks** (3-4 bullet points specific to {ticker})
4. **Strategic Recommendations** (3-5 actionable items)
5. **Metrics to Monitor** (4-5 KPIs)

Be specific to {ticker}'s business and competitive position.
"""
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
        
    except Exception as e:
        print(f"  LLM Error: {e}")
        
        return f"""
# CFO Briefing: {ticker}

## Executive Summary
{ticker} maintains stable financial performance with projected revenue of ${forecast_statements['is_revenue'].mean()/1e9:.1f}B and net margin of {assumptions.get('avg_net_income_margin', 0.25):.1%}.

## Key Opportunities
- Strong cash generation capability
- Stable margin profile
- Brand strength and ecosystem lock-in

## Key Risks
- Market saturation in core segments
- Currency fluctuations
- Regulatory pressures

## Strategic Recommendations
1. Maintain current capital allocation strategy
2. Monitor margin trends closely
3. Continue shareholder return programs
4. Invest in growth initiatives

## Metrics to Monitor
- Revenue growth rate
- Net income margin
- Free cash flow conversion
- Return on invested capital

---
*[MOCK REPORT - LLM API unavailable]*
"""


def print_assumptions(assumptions: Dict, title: str = "LLM Assumptions"):
    """Print assumptions in a formatted way."""
    print(f"\n  {title} ({assumptions.get('source', 'Unknown')}):")
    print(f"  " + "-" * 50)
    print(f"  Gross Margin:      {assumptions.get('gross_margin', 0):.2%}")
    print(f"  Net Income Margin: {assumptions.get('avg_net_income_margin', 0):.2%}")
    print(f"  EBIT Margin:       {assumptions.get('avg_ebit_margin', 0):.2%}")
    print(f"  CapEx/Revenue:     {assumptions.get('capex_to_revenue', 0):.2%}")
    print(f"  Retention Ratio:   {assumptions.get('retention_ratio', 0):.2%}")
    print(f"  " + "-" * 50)
    if 'reasoning' in assumptions:
        reasoning = assumptions['reasoning']
        if len(reasoning) > 200:
            reasoning = reasoning[:200] + "..."
        print(f"  Reasoning: {reasoning}")


if __name__ == "__main__":
    print("Enhanced LLM Assumption Generator")
    print("=" * 50)
    print("Key Features:")
    print("  1. LLM sees ticker name (can use external knowledge)")
    print("  2. LLM sees ALL historical data including dividends/buybacks")
    print("  3. LLM can predict ALL ratios including retention_ratio")
    print("  4. More detailed prompt with company context")
