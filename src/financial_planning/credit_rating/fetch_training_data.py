"""
Fetch and Save Training Data
=============================

This script fetches financial data for all 450+ companies with known S&P ratings
and saves to CSV for offline training.

Run once to collect data, then train model without API calls.

Usage:
------
python fetch_training_data.py

Output:
-------
- data/credit_rating_training_data.csv
- data/credit_rating_raw_data.json (optional, for debugging)

Author: Lihao Xiao
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from local files (not as package)
from training_data import LARGE_TRAINING_COMPANIES
from trainer import FMPCreditRatingFetcher, RATING_MAP, FEATURE_NAMES


def fetch_and_save_training_data(
    api_key: str = "vRmvrzQZbCF0SqRCeWggAOGDLRtnTQNR",
    output_dir: str = "data",
    period: str = "annual",
    delay: float = 0.2,  # Delay between API calls to avoid rate limiting
    max_companies: int = None
):
    """
    Fetch financial data for all training companies and save to CSV.
    
    Parameters:
    -----------
    api_key : str
        FMP API key
    output_dir : str
        Directory to save output files
    period : str
        'annual' or 'quarter'
    delay : float
        Seconds to wait between API calls
    max_companies : int, optional
        Limit number of companies (for testing)
    """
    print("=" * 70)
    print("Fetching Credit Rating Training Data from FMP API")
    print("=" * 70)
    print(f"Total companies in list: {len(LARGE_TRAINING_COMPANIES)}")
    print(f"Output directory: {output_dir}")
    print(f"Period: {period}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize fetcher
    fetcher = FMPCreditRatingFetcher(api_key)
    
    # Companies to fetch
    companies = LARGE_TRAINING_COMPANIES[:max_companies] if max_companies else LARGE_TRAINING_COMPANIES
    
    # Results storage
    results = []
    raw_data = {}
    failed = []
    
    start_time = time.time()
    
    for i, company in enumerate(companies):
        ticker = company['ticker']
        rating = company['rating']
        name = company.get('name', ticker)
        sector = company.get('sector', 'Unknown')
        
        # Convert rating to numeric
        rating_num = RATING_MAP.get(rating)
        if rating_num is None:
            print(f"  [{i+1}/{len(companies)}] {ticker}: Skipped (unknown rating: {rating})")
            continue
        
        # Progress
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(companies) - i - 1) / rate if rate > 0 else 0
            print(f"\nProgress: {i+1}/{len(companies)} ({(i+1)/len(companies)*100:.1f}%)")
            print(f"Elapsed: {elapsed/60:.1f} min, Remaining: {remaining/60:.1f} min\n")
        
        try:
            # Fetch financial data
            features = fetcher.fetch_company_features(ticker, period)
            
            if features:
                # Store results
                row = {
                    'ticker': ticker,
                    'name': name,
                    'sector': sector,
                    'rating': rating,
                    'rating_num': rating_num,
                    'date': features.get('date', ''),
                }
                
                # Add features
                for f in FEATURE_NAMES:
                    row[f] = features.get(f, np.nan)
                
                results.append(row)
                
                # Store raw data for debugging
                raw_data[ticker] = features.get('raw_data', {})
                
                print(f"  [{i+1}/{len(companies)}] {ticker} ({rating}): OK")
            else:
                failed.append(ticker)
                print(f"  [{i+1}/{len(companies)}] {ticker}: NO DATA")
                
        except Exception as e:
            failed.append(ticker)
            print(f"  [{i+1}/{len(companies)}] {ticker}: ERROR - {str(e)[:50]}")
        
        # Rate limiting
        time.sleep(delay)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successfully fetched: {len(results)}/{len(companies)}")
    print(f"Failed: {len(failed)}")
    
    if len(results) > 0:
        print(f"\nRating distribution:")
        rating_names = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']
        for i, name in enumerate(rating_names):
            count = sum(df['rating_num'] == i)
            if count > 0:
                pct = count / len(df) * 100
                print(f"  {name:>4}: {count:>4} ({pct:>5.1f}%)")
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "credit_rating_training_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved training data to: {csv_path}")
    
    # Save raw data (optional)
    raw_path = os.path.join(output_dir, "credit_rating_raw_data.json")
    with open(raw_path, 'w') as f:
        json.dump(raw_data, f, indent=2, default=str)
    print(f"Saved raw data to: {raw_path}")
    
    # Save failed tickers
    if failed:
        failed_path = os.path.join(output_dir, "credit_rating_failed_tickers.txt")
        with open(failed_path, 'w') as f:
            f.write("\n".join(failed))
        print(f"Saved failed tickers to: {failed_path}")
    
    # Save metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'total_companies': len(companies),
        'successful': len(results),
        'failed': len(failed),
        'period': period,
        'features': FEATURE_NAMES
    }
    meta_path = os.path.join(output_dir, "credit_rating_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {meta_path}")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch credit rating training data')
    parser.add_argument('--api-key', default="vRmvrzQZbCF0SqRCeWggAOGDLRtnTQNR",
                        help='FMP API key')
    parser.add_argument('--output-dir', default='data',
                        help='Output directory')
    parser.add_argument('--max', type=int, default=None,
                        help='Max companies to fetch (for testing)')
    parser.add_argument('--delay', type=float, default=0.2,
                        help='Delay between API calls')
    
    args = parser.parse_args()
    
    fetch_and_save_training_data(
        api_key=args.api_key,
        output_dir=args.output_dir,
        max_companies=args.max,
        delay=args.delay
    )