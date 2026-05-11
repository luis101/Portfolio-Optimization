"""
Utility functions for portfolio data download and method comparison.
"""

import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf

from robust_portfolio_optimizer import PortfolioOptimizer

# ticker = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V',
#          'WMT', 'PG', 'UNH', 'DIS', 'NVDA', 'HD', 'MA', 'PYPL', 'BAC', 'VZ']

ticker = ['^GSPC', '^IXIC', '^DJI', '^GDAXI', '^FTSE', '^FCHI', '^HSI', '^AXJO',
          '^BSESN', '^TWII', '^MXX', '^KS11', '^N225', '^BVSP', '^STI']

# end_date is end of previous month:
end_date = pd.to_datetime("today").replace(day=1) - pd.Timedelta(days=1)

def download_fin_data(ticker, start_date="1985-01-01", end_date=end_date):
    asset_df = pd.DataFrame()
    assets = pd.DataFrame()

    for symbol in ticker:
        print("Ticker: " + symbol)

        asset_data = yf.download(symbol, start=start_date, end=end_date)
        asset_data = asset_data.stack(1)
        asset_data = asset_data.reset_index(level=1)

        asset_data['month_id'] = asset_data.index.strftime('%Y-%m')
        asset_data['numst'] = asset_data.groupby(['month_id'])['Ticker'].transform('count')
        asset_data = asset_data[(asset_data['numst'] >= 17)] # Only keep months with at least 17 trading days

        data_at = asset_data.groupby(['month_id']).last().reset_index()
        asset_df = pd.concat([asset_df, data_at], axis=0)

        asset = yf.Ticker(symbol)
        try:
            data = asset.history(period="max")
        except Exception:
            continue

        if len(data) == 0:
            continue

        data['Ticker'] = symbol
        data['month_id'] = data.index.strftime('%Y-%m')
        data[['Vol', 'Div']] = data.groupby(['month_id'])[['Volume', 'Dividends']].transform('sum')
        data['numst'] = data.groupby(['month_id'])['Ticker'].transform('count')

        sdf = data.groupby(['month_id']).last().reset_index()
        # sdf["ret"] = ((sdf["Close"]+sdf['Div']) - sdf["Close"].shift(1)) / sdf["Close"].shift(1)
        # sdf["ret"] = sdf["Close"].pct_change().fillna(0) + (sdf['Div'] / sdf["Close"].shift(1)).fillna(0)
        sdf["ret"] = sdf["Close"].pct_change().fillna(0)
        sdf = sdf[(sdf['numst'] >= 17)]
        sdf = sdf[['month_id', 'Ticker', 'Close', 'Volume', 'Div', 'ret']]

        assets = pd.concat([assets, sdf], axis=0)

    return assets, asset_df


def compare_all_methods(optimizer):
    """
    Run all optimization methods and compare results.

    Returns
    -------
    results : dict
        Per-method portfolio statistics.
    comparison_df : pd.DataFrame
        Tabular summary of all methods.
    """
    results = {}

    _methods = [
        ("Classical Mean Variance", lambda: optimizer.mean_variance_optimization()),
        ("Minimum Variance",        lambda: optimizer.min_variance()),
        ("Wasserstein",             lambda: optimizer.wasserstein_optimization()),
        ("Worst-case Ellipsoidal",  lambda: optimizer.ellipsoidal_uncertainty_optimization()),
        ("Black-Litterman",         lambda: optimizer.black_litterman()),
        ("Resampled",               lambda: optimizer.resampling_optimization()),
        ("Ledoit-Wolf Shrinkage",   lambda: optimizer.shrinkage_covariance_optimization()),
        ("Factor Model",            lambda: optimizer.factor_model_optimization()),
        ("CVaR Optimization",       lambda: optimizer.cvar_optimization()),
        ("Wasserstein CVaR",        lambda: optimizer.wasserstein_cvar_optimization()),
        ("Elastic Net",             lambda: optimizer.elastic_net_optimization()),
    ]

    for name, fn in _methods:
        try:
            weights = fn()
            results[name] = optimizer.calculate_portfolio_stats(weights)
        except Exception as e:
            print(f"{name} failed: {e}")

    comparison_df = pd.DataFrame.from_dict(results, orient='index')
    comparison_df = comparison_df.reset_index().rename(columns={'index': 'Method'})

    return results, comparison_df
