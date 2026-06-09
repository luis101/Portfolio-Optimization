"""
Rolling out-of-sample backtest of the Robust Portfolio Optimization Suite.
Downloads global equity index data and back-tests all optimization methods over rolling windows.
"""

from backtest import Backtest
from portfolio_utils import download_fin_data, ticker


###### End-to-end example

if __name__ == "__main__":

    print("=" * 80)
    print("ROLLING OUT-OF-SAMPLE BACKTEST")
    print("=" * 80)

    #Download data and reshape to a wide, decimal returns matrix (time x assets)
    assets, _ = download_fin_data(ticker)
    assets = assets.sort_values(by=['Ticker', 'month_id']).reset_index(drop=True)

    returns_df = assets.pivot(index='month_id', columns='Ticker', values='ret')  # Reshape to (time_steps, n_assets), month_id index
    returns_df = returns_df[returns_df.reset_index().index > 839]  # drop early sparse history
    # returns_df = returns_df.fillna(0.0)
    # returns_df = returns_df * 100  # convert to percentage returns

    print(f"\nDataset: {len(returns_df)} periods, {returns_df.shape[1]} assets")

    #Rolling backtest: 60-month look-back, 12-month out-of-sample holding windows.
    #step defaults to oos_window (non-overlapping); pass step=1 for overlapping windows.

    # Initialize backtest with returns data and parameters
    bt_eng = Backtest(returns_df, 
                      lookback_window=60, 
                      oos_window=1, 
                      periods_per_year=12, 
                      verbose=True)

    # Run backtest
    bt = bt_eng.run()

    # Summarize results
    summary = bt.summary()

    # Detailed results
    print("\n" + "=" * 60)
    print("DETAILED RESULTS SUMMARY")
    print("=" * 60)

    n_oos = len(bt.oos_returns().index.unique())
    print(f"\n{len(bt.window_returns)} windows, {n_oos} out-of-sample periods\n")
    print(summary.round(4).to_string())
