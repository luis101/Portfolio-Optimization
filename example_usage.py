"""
Example usage of the Robust Portfolio Optimization Suite.
Downloads global equity index data and compares all optimization methods.
"""

import numpy as np
import pandas as pd

from robust_portfolio_optimizer import PortfolioOptimizer
from portfolio_utils import download_fin_data, compare_all_methods, ticker


if __name__ == "__main__":

    print("=" * 80)
    print("ROBUST PORTFOLIO OPTIMIZATION SUITE")
    print("=" * 80)

    # Download financial data
    assets, asset_df = download_fin_data(ticker)
    assets = assets.sort_values(by=['Ticker', 'month_id']).reset_index(drop=True)

    # Prepare returns DataFrame
    returns_df = assets.pivot(index='month_id', columns='Ticker', values='ret').reset_index() # Reshape to (time_steps, n_assets)
    returns_df = returns_df.drop(columns=['month_id'])  
    returns_df = returns_df[returns_df.index > 839] # Only asset data with index larger 839
    returns_df = returns_df.fillna(0.0)
    returns_df = returns_df * 100  # convert to percentage returns

    print(f"\nDataset: {len(assets['month_id'].unique())} periods, "f"{len(assets['Ticker'].unique())} assets")
    # print(f"Assets: {asset_names}")

    # Initialize optimizer
    optimizer = PortfolioOptimizer(returns_df)

    # Compare all methods
    print("\n" + "=" * 80)
    print("COMPARING ALL OPTIMIZATION METHODS")
    print("=" * 80)

    results, comparison = compare_all_methods(optimizer)

    # print("\n" + comparison.to_string(index=False))

    # Detailed results
    print("\n" + "=" * 60)
    print("DETAILED RESULTS SUMMARY")
    print("=" * 60)

    for method, stats in results.items():
        print(f"\n{method}:")
        print(f"  Sharpe Ratio (annual):    {stats['sharpe_ratio']:.4f}")
        print(f"  Expected Return (annual): {stats['expected_return']:.4f}")
        print(f"  Volatility (annual):      {stats['volatility']:.4f}")
        # print("  Weights:")
        # for asset, weight in stats['weights'].items():
        #    if weight > 0.01:  # Only show weights > 1%
        #        print(f"    {asset}: {weight:.3f}")

    # Portfolio weights table
    weights_df = pd.DataFrame()
    for method, stats in results.items():
        weights_df[method] = pd.Series(stats['weights'])

    if not weights_df.empty:
        print("\nPortfolio Weights by Method:")
        print(weights_df.round(3))

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Observations:")
    print("- Classical Markowitz may show concentrated positions")
    print("- Robust methods generally produce more diversified portfolios")
    print("- Resampling and shrinkage methods balance performance and stability")
    print("- Elastic Net creates sparse portfolios (fewer active positions)")
