#!/usr/bin/env python3
"""
auto_forecast_pipeline.py - FINAL FIXED VERSION

Fixes:
1. Show MORE accounting metrics (not just 3)
2. Better Net Income calculation (reduce from 16% to 4-6% margin)
3. Fix all accounting derivations
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from financial_planning.models.balance_sheet_forecaster import (
    BalanceSheetForecaster,
    ForecastConfig,
    train_xgboost_models,
    predict_with_models,
    calculate_mape
)
from financial_planning.utils.fmp_data_fetcher import FMPDataFetcher
from financial_planning.models.accounting_engine import AccountingEngine
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_args():
    if len(sys.argv) < 2:
        print("Usage: python auto_forecast_pipeline.py TICKER")
        sys.exit(1)
    return sys.argv[1].upper()

def train_xgboost_models_fixed(X_train, y_train, config, target_variables):
    """Train XGBoost with proper regularization."""
    models = {}
    
    for i, var in enumerate(target_variables):
        model = XGBRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            min_child_weight=getattr(config, 'min_child_weight', 3),
            gamma=getattr(config, 'gamma', 0.1),
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train[:, i], verbose=False)
        models[var] = model
    
    return models

# ============================================================================
# STEP 1-4: Same as before
# ============================================================================

def find_optimal_window(ticker, output_dir):
    print("\n" + "="*80)
    print("STEP 1: FINDING OPTIMAL DATA WINDOW")
    print("="*80)
    
    fetcher = FMPDataFetcher(api_key="vRmvrzQZbCF0SqRCeWggAOGDLRtnTQNR")
    full_data = fetcher.extract_ml_features(ticker, period='quarter', limit=120)
    
    print(f"\n✓ Fetched {len(full_data)} quarters for {ticker}")
    
    windows = [20, 40, 60, 80, min(120, len(full_data))]
    results = {}
    
    base_config = ForecastConfig(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        reg_alpha=0.5,
        reg_lambda=2.0
    )
    
    for window in windows:
        if window > len(full_data):
            continue
            
        print(f"\n  Testing {window}Q window...")
        
        data_slice = full_data.tail(window)
        forecaster = BalanceSheetForecaster(ticker, config=base_config)
        forecaster.historical_data = data_slice
        
        X, y = forecaster.prepare_features(forecaster.historical_data)
        
        if len(X) < 20:
            print(f"    ⚠️  Too few samples ({len(X)}), skipping")
            results[f'{window}Q'] = {'mape': 999.0, 'status': 'skipped'}
            continue
        
        try:
            metrics = forecaster.train(verbose=0)
            results[f'{window}Q'] = {
                'mape': metrics['overall_mape'],
                'samples': metrics['train_samples'],
                'quarters': window,
                'status': 'success'
            }
            print(f"    MAPE: {metrics['overall_mape']:.2f}%")
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results[f'{window}Q'] = {'mape': 999.0, 'status': 'error'}
    
    valid_results = {k: v for k, v in results.items() if v['status'] == 'success'}
    
    if not valid_results:
        optimal_window = min(40, len(full_data))
        optimal_mape = None
    else:
        best_key = min(valid_results.keys(), key=lambda k: valid_results[k]['mape'])
        optimal_window = valid_results[best_key]['quarters']
        optimal_mape = valid_results[best_key]['mape']
        print(f"\n✓ Optimal window: {optimal_window}Q (MAPE: {optimal_mape:.2f}%)")
    
    with open(output_dir / '01_data_window_analysis.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'tested_windows': results,
            'optimal_window': optimal_window,
            'optimal_mape': optimal_mape
        }, f, indent=2)
    
    return optimal_window, full_data

def hold_out_test_set(full_data, optimal_window, output_dir):
    print("\n" + "="*80)
    print("STEP 2: HOLDING OUT TEST SET")
    print("="*80)
    
    data = full_data.tail(optimal_window)
    test_size = int(len(data) * 0.2)
    
    development_data = data.iloc[:-test_size].copy()
    test_data = data.iloc[-test_size:].copy()
    
    print(f"\n  Optimal window: {optimal_window}Q")
    print(f"  Development: {len(development_data)}Q (80%)")
    print(f"  Test: {len(test_data)}Q (20%) 🔒")
    
    with open(output_dir / '02_data_split.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'optimal_window': optimal_window,
            'development_quarters': len(development_data),
            'test_quarters': len(test_data)
        }, f, indent=2)
    
    # Save development data for Part 2 to use same ratios
    development_data.to_csv(output_dir / '03_development_data.csv', index=False)
    print(f"  ✓ Saved development data for Part 2")
    
    return development_data, test_data

def tune_hyperparameters(ticker, development_data, output_dir):
    print("\n" + "="*80)
    print("STEP 3: HYPERPARAMETER TUNING (5-FOLD CV)")
    print("="*80)
    
    forecaster = BalanceSheetForecaster(ticker)
    forecaster.historical_data = development_data
    X, y = forecaster.prepare_features(development_data)
    
    print(f"\n✓ Development: {len(development_data)}Q, Samples: {len(X)}")
    
    configurations = {
        'Conservative': { 
        'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.01,
        'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 2.0,
        'reg_lambda': 20.0, 'min_child_weight': 10, 'gamma': 0.5
    },
        'Balanced': {
            'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.03,
            'subsample': 0.7, 'colsample_bytree': 0.7,
            'reg_alpha': 1.0, 'reg_lambda': 10.0,
            'min_child_weight': 5, 'gamma': 0.2
        },
        'Aggressive...': {
            'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.5, 'reg_lambda': 5.0,
            'min_child_weight': 3, 'gamma': 0.1
        }
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = {}
    
    for config_name, params in configurations.items():
        print(f"\n  Testing {config_name}...")
        
        config = ForecastConfig(**params)
        fold_mapes = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            models = train_xgboost_models_fixed(X_train, y_train, config, forecaster.config.target_variables)
            val_preds = predict_with_models(models, X_val, forecaster.config.target_variables)
            
            fold_mape = np.mean(list(calculate_mape(y_val, val_preds, forecaster.config.target_variables).values()))
            fold_mapes.append(fold_mape)
            
            print(f"    Fold {fold_idx}: {fold_mape:.2f}%")
        
        avg_mape = np.mean(fold_mapes)
        cv_results[config_name] = {'mean_mape': avg_mape, 'std_mape': np.std(fold_mapes)}
        
        print(f"    Average: {avg_mape:.2f}% ± {np.std(fold_mapes):.2f}%")
    
    best_config_name = min(cv_results.keys(), key=lambda k: cv_results[k]['mean_mape'])
    best_cv_mape = cv_results[best_config_name]['mean_mape']
    best_params = configurations[best_config_name]
    
    print(f"\n✓ Best: {best_config_name} (CV MAPE: {best_cv_mape:.2f}%)")
    
    with open(output_dir / '03_hyperparameter_tuning.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'cv_results': {k: {'mean_mape': float(v['mean_mape']), 'std_mape': float(v['std_mape'])}
                          for k, v in cv_results.items()},
            'best_configuration': best_config_name,
            'best_cv_mape': float(best_cv_mape),
            'best_parameters': best_params
        }, f, indent=2)
    
    return best_params, best_cv_mape

def train_final_model(ticker, development_data, best_params, output_dir):
    print("\n" + "="*80)
    print("STEP 4: TRAINING FINAL MODEL")
    print("="*80)
    
    config = ForecastConfig(**best_params)
    forecaster = BalanceSheetForecaster(ticker, config=config)
    forecaster.historical_data = development_data
    
    X, y = forecaster.prepare_features(development_data)
    
    print(f"\n  Training on {len(X)} samples...")
    
    forecaster.models = train_xgboost_models_fixed(X, y, config, forecaster.config.target_variables)
    forecaster.metadata['trained'] = True
    
    train_preds = predict_with_models(forecaster.models, X, forecaster.config.target_variables)
    train_mapes = calculate_mape(y, train_preds, forecaster.config.target_variables)
    train_mape = np.mean(list(train_mapes.values()))
    
    print(f"\n✓ Train MAPE: {train_mape:.2f}%")
    
    model_path = output_dir / '04_final_model'
    forecaster.save_model(str(model_path))
    
    return forecaster, train_mape

# ============================================================================
# STEP 5: ENHANCED EVALUATION (FIXED)
# ============================================================================

def evaluate_on_test(forecaster, test_data, output_dir, ticker):
    print("\n" + "="*80)
    print("STEP 5: EVALUATING ON TEST SET")
    print("="*80)
    
    X_test, y_test = forecaster.prepare_features(test_data)
    
    # ML Predictions
    test_preds = predict_with_models(forecaster.models, X_test, forecaster.config.target_variables)
    ml_mapes = calculate_mape(y_test, test_preds, forecaster.config.target_variables)
    overall_ml_mape = np.mean(list(ml_mapes.values()))
    
    print(f"\n  A. ML Predictions:")
    print(f"  {'Variable':<30s} {'Actual (Avg)':<15s} {'Predicted (Avg)':<15s} {'MAPE':<10s}")
    print(f"  " + "-"*70)
    
    for i, var in enumerate(forecaster.config.target_variables):
        actual_avg = np.mean(y_test[:, i])
        pred_avg = np.mean(test_preds[var])
        mape = ml_mapes[var]
        print(f"  {var:<30s} ${actual_avg/1e9:>7.2f}B       ${pred_avg/1e9:>7.2f}B       {mape:>6.2f}%")
    
    print(f"  " + "-"*70)
    print(f"  {'Overall ML MAPE':<30s} {'':15s} {'':15s} {overall_ml_mape:>6.2f}%")
    
    test_df = pd.DataFrame(test_preds)
    test_df.to_csv(output_dir / '05_test_ml_predictions.csv', index=False)
    
    # Accounting Engine
    print(f"\n  B. Accounting Metrics:")
    
    try:
        engine = AccountingEngine(forecaster.historical_data.tail(20))
        
        # SAVE historical ratios for Part 2 comparison
        # Convert to float and handle NaN/Inf for JSON serialization
        def safe_float(val):
            f = float(val)
            if np.isnan(f) or np.isinf(f):
                return 0.0
            return f
        
        historical_ratios = {
            'gross_margin': safe_float(engine.ratios.gross_margin),
            'avg_ebit_margin': safe_float(engine.ratios.avg_ebit_margin),
            'avg_net_income_margin': safe_float(engine.ratios.avg_net_income_margin),
            'capex_to_revenue': safe_float(engine.ratios.capex_to_revenue),
            'retention_ratio': safe_float(engine.ratios.retention_ratio),
            'avg_interest_rate': safe_float(engine.ratios.avg_interest_rate),
            'tax_rate': safe_float(engine.ratios.tax_rate),
            'overhead_to_revenue': safe_float(engine.ratios.overhead_to_revenue),
            'payroll_to_revenue': safe_float(engine.ratios.payroll_to_revenue),
            'cash_to_revenue': safe_float(engine.ratios.cash_to_revenue),
            'ar_days': safe_float(engine.ratios.ar_days),
            'inventory_days': safe_float(engine.ratios.inventory_days),
            'ap_days': safe_float(engine.ratios.ap_days),
            'last_shares_outstanding': safe_float(engine.ratios.last_shares_outstanding),
        }
        with open(output_dir / '03_historical_ratios.json', 'w') as f:
            json.dump(historical_ratios, f, indent=2)
        print(f"  ✓ Saved historical ratios for Part 2")
        
        complete_statements = engine.build_complete_statements(
            predictions=test_preds,
            periods=len(X_test)
        )
        
        complete_statements.to_csv(output_dir / '06_test_complete_statements.csv', index=False)
        
        # Compare MORE accounting metrics
        test_indices = test_data.index[-len(X_test):]
        actual_test = test_data.loc[test_indices]
        
        # SAVE test actuals for Part 2 comparison
        actual_test.to_csv(output_dir / '04_test_actuals.csv', index=False)
        print(f"  ✓ Saved test actuals: {len(actual_test)} periods")
        
        print(f"  {'Metric':<30s} {'Actual (Avg)':<15s} {'Predicted (Avg)':<15s} {'MAPE':<10s}")
        print(f"  " + "-"*70)
        
        accounting_mapes = {}
        
        # EXPANDED list of metrics to compare
        key_metrics = [
            ('net_income', 'is_net_income', 'Net Income'),
            ('ebit', 'is_ebit', 'EBIT'),
            ('ebitda', 'is_ebitda', 'EBITDA'),
            ('gross_profit', 'is_gross_profit', 'Gross Profit'),
            ('total_assets', 'bs_total_assets', 'Total Assets'),
            ('total_equity', 'bs_total_equity', 'Total Equity'),
            ('total_liabilities', 'bs_total_liabilities', 'Total Liabilities'),
            ('operating_cash_flow', 'cf_operating_cash_flow', 'Operating Cash Flow'),
            ('free_cash_flow', 'cf_free_cash_flow', 'Free Cash Flow'),
        ]
        
        for actual_col, pred_col, display_name in key_metrics:
            if actual_col in actual_test.columns and pred_col in complete_statements.columns:
                actual_vals = actual_test[actual_col].values
                pred_vals = complete_statements[pred_col].values
                
                valid_mask = actual_vals != 0
                if valid_mask.sum() > 0:
                    mape = np.mean(np.abs((actual_vals[valid_mask] - pred_vals[valid_mask]) / actual_vals[valid_mask])) * 100
                    accounting_mapes[display_name] = mape
                    
                    actual_avg = np.mean(actual_vals)
                    pred_avg = np.mean(pred_vals)
                    
                    print(f"  {display_name:<30s} ${actual_avg/1e9:>7.2f}B       ${pred_avg/1e9:>7.2f}B       {mape:>6.2f}%")
        
        print(f"  " + "-"*70)
        if accounting_mapes:
            overall_acc_mape = np.mean(list(accounting_mapes.values()))
            print(f"  {'Overall Accounting MAPE':<30s} {'':15s} {'':15s} {overall_acc_mape:>6.2f}%")
        
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        complete_statements = None
    
    # Grade
    if overall_ml_mape < 15:
        grade = "⭐⭐⭐⭐⭐ Excellent!"
    elif overall_ml_mape < 20:
        grade = "⭐⭐⭐⭐ Very Good!"
    else:
        grade = "⭐⭐⭐ Good"
    
    print(f"\n  Grade: {grade}")
    
    with open(output_dir / '07_test_evaluation.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'test_mape': float(overall_ml_mape),
            'grade': grade
        }, f, indent=2)
    
    return overall_ml_mape, grade, complete_statements

# ============================================================================
# STEP 6: Same as before
# ============================================================================

def generate_forecasts(forecaster, ticker, output_dir):
    print("\n" + "="*80)
    print("STEP 6: GENERATING FUTURE FORECASTS")
    print("="*80)
    
    X_hist, y_hist = forecaster.prepare_features(forecaster.historical_data)
    
    periods = 4
    future_predictions = {var: [] for var in forecaster.config.target_variables}
    
    current_features = X_hist[-1:].copy()
    
    for t in range(periods):
        period_preds = {}
        for var in forecaster.config.target_variables:
            pred = forecaster.models[var].predict(current_features)[0]
            period_preds[var] = pred
            future_predictions[var].append(pred)
        
        # Rolling window
        new_features = current_features[0, len(forecaster.config.target_variables):].tolist()
        new_features.extend([period_preds[var] for var in forecaster.config.target_variables])
        current_features = np.array([new_features])
    
    future_predictions = {var: np.array(vals) for var, vals in future_predictions.items()}
    
    future_df = pd.DataFrame(future_predictions)
    future_df.to_csv(output_dir / '08_future_ml_predictions.csv', index=False)
    
    # Build complete statements
    try:
        engine = AccountingEngine(forecaster.historical_data.tail(20))
        future_statements = engine.build_complete_statements(
            predictions=future_predictions,
            periods=periods
        )
        
        future_statements.to_csv(output_dir / '09_future_complete_statements.csv', index=False)
        
        print(f"\n  Future Forecasts:")
        for i in range(periods):
            rev = future_statements.iloc[i]['is_revenue']
            ni = future_statements.iloc[i]['is_net_income']
            margin = ni / rev * 100 if rev > 0 else 0
            print(f"    Q{i+1}: Revenue ${rev/1e9:.2f}B, Net Income ${ni/1e9:.2f}B ({margin:.1f}% margin)")
        
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        future_statements = None
    
    print(f"\n✓ Forecasts saved")
    
    return future_predictions, future_statements

# ============================================================================
# MAIN
# ============================================================================

def main():
    ticker = parse_args()
    
    print("="*80)
    print(f"XGBOOST FORECASTING PIPELINE FOR {ticker}")
    print("="*80)
    
    output_dir = Path('outputs/xgboost_models') / ticker.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_start = datetime.now()
    
    try:
        optimal_window, full_data = find_optimal_window(ticker, output_dir)
        development_data, test_data = hold_out_test_set(full_data, optimal_window, output_dir)
        best_params, best_cv_mape = tune_hyperparameters(ticker, development_data, output_dir)
        forecaster, train_mape = train_final_model(ticker, development_data, best_params, output_dir)
        test_mape, grade, test_statements = evaluate_on_test(forecaster, test_data, output_dir, ticker)
        future_preds, future_statements = generate_forecasts(forecaster, ticker, output_dir)
        
        duration = (datetime.now() - timestamp_start).total_seconds()
        
        summary = f"""
================================================================================
PIPELINE COMPLETE FOR {ticker}
================================================================================

⏱️  Duration: {duration:.1f} seconds

📊 RESULTS:
----------
Optimal Window:   {optimal_window}Q
CV MAPE:          {best_cv_mape:.2f}%
Train MAPE:       {train_mape:.2f}%
Test MAPE:        {test_mape:.2f}%

Grade: {grade}

📁 OUTPUT FILES:
---------------
05_test_ml_predictions.csv      - ML predictions (test)
06_test_complete_statements.csv - Complete financials (test)
08_future_ml_predictions.csv    - ML predictions (future)
09_future_complete_statements.csv - Complete financials (future)

✅ COMPLETE!
================================================================================
"""
        
        print(summary)
        
        with open(output_dir / 'pipeline_summary.txt', 'w') as f:
            f.write(summary)
        
        print(f"\nResults in: {output_dir}/")
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()