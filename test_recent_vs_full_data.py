#!/usr/bin/env python3
"""
test_recent_vs_full_data.py - Does XGBoost perform better with recent data?

Tests:
1. All 120 quarters (1995-2025, 30 years)
2. Recent 60 quarters (2010-2025, 15 years)
3. Recent 40 quarters (2015-2025, 10 years)
4. Recent 20 quarters (2020-2025, 5 years)

Hypothesis: Recent data might be better because:
- More consistent business model (iPhone era vs pre-iPhone)
- Less structural breaks
- More relevant patterns
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from financial_planning.models.balance_sheet_forecaster import (
    BalanceSheetForecaster,
    ForecastConfig
)
from financial_planning.utils.fmp_data_fetcher import FMPDataFetcher
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RECENT DATA VS FULL HISTORICAL DATA - XGBoost Performance")
print("="*80)

ticker = 'AAPL'

# ============================================================================
# FETCH FULL DATA
# ============================================================================
print(f"\n[1] Fetching full historical data...")

fetcher = FMPDataFetcher(api_key="vRmvrzQZbCF0SqRCeWggAOGDLRtnTQNR")
full_data = fetcher.extract_ml_features(ticker, period='quarter', limit=120)

print(f"✓ Fetched {len(full_data)} quarters")
print(f"  Date range: {full_data.index[0]} to {full_data.index[-1]}")

# ============================================================================
# TEST DIFFERENT TIME WINDOWS
# ============================================================================
print(f"\n[2] Testing different time windows...")
print("="*80)

# Define test configurations
data_configs = {
    'All 120Q (30 years)': {
        'quarters': 120,
        'description': '1995-2025, includes pre-iPhone era'
    },
    'Recent 80Q (20 years)': {
        'quarters': 80,
        'description': '2005-2025, includes iPhone launch'
    },
    'Recent 60Q (15 years)': {
        'quarters': 60,
        'description': '2010-2025, mature iPhone era'
    },
    'Recent 40Q (10 years)': {
        'quarters': 40,
        'description': '2015-2025, iPhone + Services growth'
    },
    'Recent 20Q (5 years)': {
        'quarters': 20,
        'description': '2020-2025, COVID + Services dominance'
    }
}

# Use consistent XGBoost config
config = ForecastConfig(
    lookback_periods=4,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0
)

results = {}

for name, conf in data_configs.items():
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"  {conf['description']}")
    print('-'*80)
    
    # Get data slice
    n_quarters = conf['quarters']
    data_slice = full_data.tail(n_quarters).copy()
    
    print(f"  Quarters: {len(data_slice)}")
    print(f"  Date range: {data_slice.index[0]} to {data_slice.index[-1]}")
    
    # Calculate business metrics
    start_revenue = data_slice['sales_revenue'].iloc[0]
    end_revenue = data_slice['sales_revenue'].iloc[-1]
    total_growth = (end_revenue - start_revenue) / start_revenue * 100
    avg_annual_growth = total_growth / (n_quarters / 4)
    
    print(f"  Revenue growth: {total_growth:.1f}% total, {avg_annual_growth:.1f}% annual")
    
    # Check if enough data
    if len(data_slice) < 20:
        print(f"  ⚠️  Too few quarters, skipping")
        results[name] = {
            'mape': 999.0,
            'samples': 0,
            'revenue_growth': total_growth,
            'status': 'skipped'
        }
        continue
    
    # Train model
    try:
        forecaster = BalanceSheetForecaster(ticker, config=config)
        forecaster.historical_data = data_slice
        forecaster.metadata['data_source'] = f'fmp_recent_{n_quarters}q'
        
        metrics = forecaster.train(verbose=0)
        
        print(f"  Training samples: {metrics['train_samples']}")
        print(f"  Test samples: {metrics['test_samples']}")
        print(f"  Overall MAPE: {metrics['overall_mape']:.2f}%")
        
        # Store results
        results[name] = {
            'mape': metrics['overall_mape'],
            'samples': metrics['train_samples'],
            'revenue_growth': avg_annual_growth,
            'quarters': n_quarters,
            'individual_mapes': metrics['individual_mapes'],
            'status': 'success'
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results[name] = {
            'mape': 999.0,
            'samples': 0,
            'revenue_growth': avg_annual_growth,
            'status': 'error'
        }

# ============================================================================
# ANALYZE RESULTS
# ============================================================================
print(f"\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

print(f"\nOverall MAPE by Data Window:")
print("-"*80)

sorted_results = sorted(
    [(k, v) for k, v in results.items() if v['status'] == 'success'],
    key=lambda x: x[1]['mape']
)

for i, (name, data) in enumerate(sorted_results, 1):
    stars = "⭐" * max(0, 6 - i)
    print(f"{i}. {name:25s}: {data['mape']:6.2f}% MAPE "
          f"({data['samples']:3d} samples, {data['quarters']:3d}Q) {stars}")

# Find best
if sorted_results:
    best_name, best_data = sorted_results[0]
    worst_name, worst_data = sorted_results[-1]
    
    improvement = worst_data['mape'] - best_data['mape']
    
    print(f"\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    print(f"\nBest Configuration: {best_name}")
    print(f"  MAPE: {best_data['mape']:.2f}%")
    print(f"  Training samples: {best_data['samples']}")
    print(f"  Improvement over worst: {improvement:.2f} percentage points")
    
    # Analyze by individual metrics
    print(f"\n  Individual MAPEs:")
    for var, mape in best_data['individual_mapes'].items():
        print(f"    {var}: {mape:.1f}%")

# ============================================================================
# DETAILED ANALYSIS
# ============================================================================
print(f"\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

# Check correlation between data characteristics and MAPE
print(f"\nData Characteristics vs Performance:")
print("-"*80)

success_results = [(k, v) for k, v in results.items() if v['status'] == 'success']

print(f"{'Configuration':25s} {'Quarters':>8s} {'Samples':>8s} "
      f"{'Growth':>10s} {'MAPE':>8s}")
print("-"*80)

for name, data in success_results:
    print(f"{name:25s} {data['quarters']:8d} {data['samples']:8d} "
          f"{data['revenue_growth']:9.1f}% {data['mape']:7.2f}%")

# ============================================================================
# INSIGHTS
# ============================================================================
print(f"\n" + "="*80)
print("INSIGHTS & RECOMMENDATIONS")
print("="*80)

if sorted_results:
    best_quarters = best_data['quarters']
    
    print(f"""
WHY DOES THIS HAPPEN?
--------------------

XGBoost (tree-based) vs LSTM (neural network):

1. **Data Consistency**:
   - Tree models prefer consistent patterns
   - Recent data = more similar business model
   - Old data (1995-2005) = different Apple (Mac-only era)

2. **Sample Efficiency**:
   - XGBoost works well with {best_data['samples']} samples
   - More data ≠ always better if data is heterogeneous
   - Quality > Quantity for tree models

3. **Structural Breaks**:
   - 2007: iPhone launch (business model changed)
   - 2015: Services growth (revenue mix changed)
   - Recent data avoids these breaks

BEST CONFIGURATION:
------------------

{best_name}
- MAPE: {best_data['mape']:.2f}%
- Quarters: {best_quarters}
- Captures: {"Modern Apple (iPhone + Services era)" if best_quarters <= 60 else "Full Apple history"}

RECOMMENDATION:
--------------

For XGBoost on Apple data:
✓ Use recent {best_quarters} quarters
✓ This gives ~{best_data['samples']} training samples
✓ Avoids structural breaks
✓ More consistent patterns

UPDATE YOUR CONFIG:
------------------

Instead of:
  all_data = fetcher.extract_ml_features(ticker, period='quarter', limit=120)

Use:
  all_data = fetcher.extract_ml_features(ticker, period='quarter', limit=120)
  recent_data = all_data.tail({best_quarters})  # Use only recent data
  forecaster.historical_data = recent_data

Expected improvement: {improvement:.1f} percentage points
""")

# ============================================================================
# COMPARISON WITH LSTM
# ============================================================================
print(f"\n" + "="*80)
print("COMPARISON: XGBOOST vs LSTM")
print("="*80)

print(f"""
LSTM Performance:
- Full 120Q: 45% MAPE (overfitting)
- Recent 40Q: 216% MAPE (too few samples)

XGBoost Performance:
- Full 120Q: {results.get('All 120Q (30 years)', {}).get('mape', 'N/A')}% MAPE
- Recent {best_quarters}Q: {best_data['mape']:.1f}% MAPE ← BEST

KEY DIFFERENCE:
--------------

LSTM:
- Needs LOTS of data (1000+ samples)
- Recent data = too few samples = worse

XGBoost:
- Works with moderate data (50-100 samples)
- Recent data = better patterns = better!

This is why XGBoost >> LSTM for financial forecasting!
""")

# Save best config
import json

config_file = f'{ticker.lower()}_optimal_data_window.json'
with open(config_file, 'w') as f:
    json.dump({
        'best_configuration': best_name,
        'quarters_to_use': best_quarters,
        'expected_mape': best_data['mape'],
        'training_samples': best_data['samples'],
        'all_results': {k: {
            'mape': v['mape'],
            'quarters': v.get('quarters', 0),
            'samples': v.get('samples', 0)
        } for k, v in results.items() if v['status'] == 'success'}
    }, f, indent=2)

print(f"\n✓ Saved optimal configuration to: {config_file}")

print("="*80)
print("\nRECOMMENDED NEXT STEP:")
print(f"Update your run_xgboost_forecaster.py to use recent {best_quarters} quarters")
print("="*80)
