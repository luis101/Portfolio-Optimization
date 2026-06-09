
###### Rolling out-of-sample backtest for the Robust Portfolio Optimization Suite


### Needed packages:

import numpy as np
import pandas as pd

from robust_portfolio_optimizer import PortfolioOptimizer


class Backtest:
    """
    Rolling-window out-of-sample backtest of the optimization suite.

    For each window the weights are estimated on a `lookback_window`-period look-back window
    with the PortfolioOptimizer, held over the next `oos_window` periods, and the realized
    (out-of-sample) portfolio returns are recorded. The look-back window advances by `step`
    periods each iteration:
      - step == oos_window (default): out-of-sample blocks are non-overlapping, so every
        held-out period is realized exactly once and the windows join into one continuous
        return series.
      - step < oos_window: out-of-sample blocks overlap (the same period is evaluated under
        several windows' weights); the summary then works on holding-period returns.
      - step > oos_window: leaves gaps between evaluated blocks.

    Parameters:

    returns: DataFrame      [wide returns, rows = time periods (sorted), columns = assets]
    lookback_window: int    [length of the estimation / look-back window, e.g. 60 months]
    oos_window: int         [length of the out-of-sample holding window, e.g. 12 months]
    step: int               [periods to advance each iteration; defaults to oos_window
                             (non-overlapping). Set < oos_window for overlapping windows]
    method_set: callable    [optimizer, return_scale -> list of (name, fn); defaults to
                             Backtest.default_method_set]
    return_scale: numeric   [units divisor forwarded to the Kelly methods]
    risk_free_rate: numeric [per-period risk-free rate passed to the optimizer]
    periods_per_year: int   [12 for monthly, 252 for daily; used for annualization]

    Examples:
    bt = Backtest(returns_df, lookback_window=60, oos_window=12).run()
    summary = bt.summary()
    hpr, h = bt.holding_period_returns()
    bt = Backtest(returns_df, 60, 12, step=1).run()  # overlapping windows
    """

    def __init__(self, returns, lookback_window, oos_window, step=None, method_set=None,
                 return_scale=1.0, risk_free_rate=0.0, periods_per_year=12, verbose=False,
                 weight_bounds=(0.0, 1.0), leverage=1.0, max_position=None, transaction_cost=None):
        self.returns = returns.sort_index()
        self.lookback_window = lookback_window
        self.oos_window = oos_window
        self.step = step if step is not None else oos_window  # default: non-overlapping blocks
        self.method_set = method_set if method_set is not None else Backtest.default_method_set
        self.return_scale = return_scale
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.verbose = verbose
        #Constraint config forwarded to each per-window PortfolioOptimizer (see its __init__)
        self.weight_bounds = weight_bounds
        self.leverage = leverage
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        #Per-window results, populated by run()
        self.window_returns = None   # OOS-start -> DataFrame of out-of-sample returns
        self.window_weights = None   # OOS-start -> {method: weight vector}
        self.window_pred_vol = None  # OOS-start -> {method: predicted per-period volatility}

    @classmethod
    def from_results(cls, window_returns, window_weights=None, window_pred_vol=None,
                     risk_free_rate=0.0, periods_per_year=12):
        """Build a Backtest from pre-computed per-window results (e.g. from a previous run), so
        the summary / holding-period views can be obtained without re-running the loop. Missing
        weights / predicted volatilities default to empty dicts (their stats then come out NaN)."""
        bt = cls.__new__(cls)  # bypass __init__: there is nothing to run, only results to view
        bt.returns = None
        bt.lookback_window = None
        bt.oos_window = None
        bt.step = None
        bt.method_set = cls.default_method_set
        bt.return_scale = 1.0
        bt.risk_free_rate = risk_free_rate
        bt.periods_per_year = periods_per_year
        bt.window_returns = window_returns
        bt.window_weights = window_weights if window_weights is not None else {}
        bt.window_pred_vol = window_pred_vol if window_pred_vol is not None else {}
        return bt

    # ====================================================================
    # Method set
    # ====================================================================

    @staticmethod
    def default_method_set(optimizer, return_scale=1.0):
        """
        Build the list of optimization methods evaluated by the backtest, rebuilt for every
        estimation window so each method sees the look-back PortfolioOptimizer instance.

        optimizer: PortfolioOptimizer  [fitted on the current look-back window]
        return_scale: numeric          [divisor passed to the Kelly methods so that 1 + r is
                                         formed from decimal returns; 100 for percent, 1 for decimals]

        Returns a list of (name, callable) where each callable returns a weight vector or None.
        """
        return [
            ("Equal Weight",             lambda: optimizer.equal_weight_portfolio()),
            ("Classical Mean Variance",  lambda: optimizer.mean_variance_optimization()),
            ("Minimum Variance",         lambda: optimizer.min_variance()),
            ("Wasserstein",              lambda: optimizer.wasserstein_optimization()),
            ("Worst-case Ellipsoidal",   lambda: optimizer.ellipsoidal_uncertainty_optimization()),
            ("Black-Litterman",          lambda: optimizer.black_litterman()),
            ("Resampled",                lambda: optimizer.resampling_optimization(n_samples=250, seed=42)),
            ("Ledoit-Wolf Shrinkage",    lambda: optimizer.shrinkage_covariance_optimization()),
            ("MCD",                      lambda: optimizer.mcd_robust_covariance_optimization()),
            ("Factor Model",             lambda: optimizer.factor_model_optimization()),
            ("CVaR",                     lambda: optimizer.cvar_optimization()),
            ("Wasserstein CVaR",         lambda: optimizer.wasserstein_cvar_optimization()),
            ("Mean-CDaR",                lambda: optimizer.cdar_optimization()),
            ("Elastic Net",              lambda: optimizer.elastic_net_optimization()),
            ("Hierarchical Risk Parity", lambda: optimizer.hierarchical_risk_parity()),
            ("Risk Parity",              lambda: optimizer.risk_parity_optimization()),
            ("Maximum Diversification",  lambda: optimizer.maximum_diversification()),
            ("CVaR Risk Parity (ERC)",   lambda: optimizer.cvar_risk_parity_optimization()),
            ("CDaR Risk Parity (ERC)",   lambda: optimizer.cdar_risk_parity_optimization()),
            ("Kelly (Full)",             lambda: optimizer.kelly_optimization(fraction=1.0, return_scale=return_scale)),
            ("Kelly (Half)",             lambda: optimizer.kelly_optimization(fraction=0.5, return_scale=return_scale)),
            ("Worst-Case Kelly",         lambda: optimizer.worst_case_kelly_optimization(alpha=0.1, return_scale=return_scale)),
        ]

    # ====================================================================
    # Rolling backtest
    # ====================================================================

    def run(self):
        """Run the rolling out-of-sample loop and store the per-window results. Returns self."""
        returns = self.returns
        n_periods = len(returns)
        lookback_window = self.lookback_window
        oos_window = self.oos_window

        self.window_returns = dict()
        self.window_weights = dict()
        self.window_pred_vol = dict()
        prev_weights = dict()  # previous window's target weights, for the turnover penalty

        #Slide a fixed-length look-back window forward by `step` periods each iteration

        starts = range(0, n_periods - lookback_window, self.step)

        for window, start in enumerate(starts):

            if self.verbose:
                print(f"Running window {window + 1}/{len(starts)}: look-back {returns.index[start]} to "
                      f"{returns.index[start + lookback_window - 1]}, OOS {returns.index[start + lookback_window]} to "
                      f"{returns.index[min(start + lookback_window + oos_window - 1, n_periods - 1)]}")
                
            #Estimation window (weights) and the held-out out-of-sample window

            est_window = returns.iloc[start:start + lookback_window]
            oos = returns.iloc[start + lookback_window: start + lookback_window + oos_window]

            #Label results by the first out-of-sample period

            oos_start = returns.index[start + lookback_window]

            #Universe = assets with complete data over the whole look-back window - independent windows

            available = est_window.columns[est_window.notna().all()]
            if len(available) < 2:
                continue  # too few assets to form a portfolio this window
            est_window = est_window[available]
            oos = oos[available]

            #Fit the optimizer on the look-back window only (no look-ahead bias)

            optimizer = PortfolioOptimizer(est_window, risk_free_rate=self.risk_free_rate,
                                           periods_per_year=self.periods_per_year,
                                           weight_bounds=self.weight_bounds,
                                           leverage=self.leverage,
                                           max_position=self.max_position,
                                           transaction_cost=self.transaction_cost)
            methods = self.method_set(optimizer, self.return_scale)

            oos_ret = pd.DataFrame(index=oos.index)
            weights = dict()
            pred_vol = dict()

            #Compute weights per method and hold them across the out-of-sample window

            for name, fn in methods:
                #Turnover penalty: align previous target weights to the current universe (0 if absent last window), None for first appearance
                w_prev = prev_weights.get(name)
                optimizer.prev_weights = (w_prev.reindex(available, fill_value=0.0).values
                                          if w_prev is not None else None)
                try:
                    w = fn()
                except Exception as e:
                    print(f"[window {window} | {oos_start}] {name} failed: {e}")
                    continue

                if w is None:
                    continue

                w = np.asarray(w, dtype=float)

                #Out-of-sample return: determine the assets present in the OOS window
                oos_filled = oos.fillna(0).values
                mask = oos.notna().values

                w_oos  = mask * w  # Set weights to zero for assets missing in the OOS window
                denom = w_oos.sum(axis=1, keepdims=True)
                denom[denom == 0] = 1 
                w_oos_norm = w_oos / denom # Renormalize weights to sum to 1 for each period

                oos_ret[name] = (oos_filled * w_oos_norm).sum(axis=1)
                weights[name] = pd.Series(w, index=available)  # labelled so turnover can reindex
                prev_weights[name] = weights[name]             # carry forward for next window's penalty

                #Predicted per-period volatility from the look-back covariance
                pred_vol[name] = float(np.sqrt(w @ optimizer.cov @ w))

            self.window_returns[oos_start] = oos_ret
            self.window_weights[oos_start] = weights
            self.window_pred_vol[oos_start] = pred_vol

        return self

    def _ensure_run(self):
        """Run the backtest on first access if it has not been run yet."""
        if self.window_returns is None:
            self.run()

    # ====================================================================
    # Result views
    # ====================================================================

    def oos_returns(self):
        """Joined out-of-sample return frame (rows = periods, columns = methods). One continuous
        series for non-overlapping windows; rows repeat across blocks when windows overlap."""
        self._ensure_run()
        return pd.concat(self.window_returns.values(), axis=0).sort_index()

    def holding_period_returns(self):
        """
        Build the time-series of holding-period (oos_window) returns.

        Each window's per-period returns are compounded into a single holding-period return,
        indexed by the window's first out-of-sample period. With step < oos_window this gives an
        overlapping series (e.g. a monthly series of n-month returns, each stamped with the first
        of its n months). Only full-length windows are kept, so every observation spans horizon h.

        Returns (returns, h): a DataFrame [index = first OOS period, columns = methods] and the
        holding horizon h in base periods (the full oos_window length).
        """
        self._ensure_run()
        windows = list(self.window_returns.values())
        if len(windows) == 0:
            return pd.DataFrame(), 0

        hp = max(len(w.index) for w in windows)  # full window length = oos_window

        data = dict()
        for oos_start, ret in self.window_returns.items():
            hp_ret = dict()
            for method in ret.columns:
                r = ret[method].dropna()
                if len(r) == hp:  # keep only full-length windows so horizons match
                    hp_ret[method] = float((1.0 + r).prod() - 1.0)  # compounded holding-period return
            if len(hp_ret) > 0:
                data[oos_start] = hp_ret

        returns = pd.DataFrame.from_dict(data, orient='index').sort_index()
        return returns, hp

    # ====================================================================
    # Summary statistics
    # ====================================================================

    def summary(self):
        """
        Per-method realized out-of-sample summary statistics.

        Non-overlapping windows are joined into one continuous one-period return series and the
        statistics are computed on it exactly (true path-dependent drawdown, etc.). Overlapping
        windows (step < oos_window) cannot be joined without double-counting, so each window's
        per-period returns are compounded into a single holding-period (oos_window) return stamped
        with the window's first period and the statistics are computed on that series (annualized
        by oos_window). Path metrics (drawdown, cumulative return) are undefined on overlapping
        returns and reported as NaN; turnover, effective N and predicted volatility are
        cross-window averages in either case.
        """
        self._ensure_run()
        windows = list(self.window_returns.values())
        if len(windows) == 0:
            return pd.DataFrame()

        #Detect overlapping out-of-sample blocks (a period appearing in more than one window)
        all_index = np.concatenate([w.index.values for w in windows])
        overlapping = len(all_index) != len(pd.unique(all_index))

        #Assemble one returns frame (rows = periods, columns = methods) to compute on: the joined
        #one-period series when non-overlapping, or the holding-period returns when overlapping.
        #hp is the holding horizon in base periods (1 for the one-period series).
        if overlapping:
            method_returns, hp = self.holding_period_returns()
            print(f"Overlapping windows detected -> statistics on the series of {hp}-period holding returns")
        else:
            method_returns = pd.concat(windows, axis=0).sort_index()
            hp = 1

        statistics = dict()

        for method in method_returns.columns:

            returns = method_returns[method].dropna()
            if len(returns) == 0:
                continue

            #Return statistics (hp = holding horizon, 1 for the non-overlapping one-period series)

            stats = self._return_stats(returns, self.risk_free_rate, self.periods_per_year, horizon=hp)

            #Allocation / cost statistics (cross-window averages in both cases)

            stats['avg_turnover'] = self._avg_turnover(self.window_weights, method)
            stats['effective_n'] = self._avg_effective_n(self.window_weights, method)

            pred_vol = self._avg_predicted_vol(self.window_pred_vol, method, self.periods_per_year)
            stats['realized_vs_predicted_vol'] = (stats['ann_volatility'] / pred_vol
                                                  if pred_vol and pred_vol > 0 else np.nan)

            stats['n_windows'] = sum(1 for w in windows if method in w.columns)
            stats['n_periods'] = len(returns)

            statistics[method] = stats

        return pd.DataFrame.from_dict(statistics, orient='index')

    # ====================================================================
    # Stateless statistic helpers
    # ====================================================================

    @staticmethod
    def _max_drawdown(returns):
        """Maximum drawdown of a per-period return series (compounded wealth path)."""
        wealth = (1.0 + returns).cumprod()
        peak = wealth.cummax()
        drawdown = wealth / peak - 1.0
        return float(drawdown.min())

    @staticmethod
    def _return_stats(returns, risk_free_rate, periods_per_year, horizon=1):
        """Annualized return / volatility / Sharpe / Sortino for a return series whose observations
        each span `horizon` base periods (horizon = 1 for a one-period series, oos_window for a
        holding-period series). With horizon == 1 the series is a contiguous one-period wealth path,
        so maximum drawdown and cumulative return are also reported; for horizon > 1 (overlapping
        holding-period returns) those path metrics are undefined and returned as NaN."""
        ppy = periods_per_year / horizon            # number of horizon-length blocks per year
        ann = np.sqrt(ppy)
        rf = (1.0 + risk_free_rate) ** horizon - 1.0  # risk-free compounded over the holding period
        ret_exc = returns - rf
        vol = returns.std(ddof=1)
        downside = returns[returns < rf]
        downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
        path = horizon == 1                          # one-period series -> a valid wealth path
        return {
            'ann_return': (returns.mean() * ppy) * 100,  # annualized return in percentage points
            'ann_volatility': vol * ann * 100,              # annualized volatility in percentage points
            'sharpe': (ret_exc.mean() / vol) * ann if vol > 0 else np.nan,
            'sortino': (ret_exc.mean() / downside_std) * ann if downside_std and downside_std > 0 else np.nan,
            'max_drawdown': Backtest._max_drawdown(returns) * 100 if path else np.nan,
            'cum_return': float((1.0 + returns).prod() - 1.0) * 100 if path else np.nan,
        }

    @staticmethod
    def _avg_turnover(window_weights, method):
        """Average one-step turnover sum_i |w_i^(k) - w_i^(k-1)| across rebalances for a method.
        Each window's weights are reindexed onto the full asset set, filling 0 for assets absent."""
        keys = sorted(w for w in window_weights if method in window_weights[w])
        if len(keys) < 2:
            return np.nan
        wseq = [window_weights[k][method] for k in keys]

        #Full asset set = union of every window's asset index
        universe = wseq[0].index
        for s in wseq[1:]:
            universe = universe.union(s.index)
        wseq = [s.reindex(universe, fill_value=0.0) for s in wseq]

        diffs = [np.sum(np.abs(wseq[i].values - wseq[i - 1].values))
                 for i in range(1, len(wseq))]
        return float(np.mean(diffs))

    @staticmethod
    def _avg_effective_n(window_weights, method):
        """Average effective number of positions 1 / sum_i w_i^2 across windows for a method."""
        vals = [1.0 / np.sum(window_weights[w][method] ** 2)
                for w in window_weights if method in window_weights[w]]
        return float(np.mean(vals)) if len(vals) > 0 else np.nan

    @staticmethod
    def _avg_predicted_vol(window_pred_vol, method, periods_per_year):
        """Average annualized predicted volatility across windows for a method."""
        vals = [window_pred_vol[v][method]
                for v in window_pred_vol if method in window_pred_vol[v]]
        return float(np.mean(vals)) * np.sqrt(periods_per_year) if len(vals) > 0 else np.nan

    def __repr__(self):
        state = "run" if self.window_returns is not None else "not run"
        return (f"Backtest(lookback_window={self.lookback_window}, oos_window={self.oos_window}, "
                f"step={self.step}, periods_per_year={self.periods_per_year}, {state})")


###### Module-level functional interface (thin wrappers around the Backtest class)

def default_method_set(optimizer, return_scale=1.0):
    """Functional alias for Backtest.default_method_set"""
    return Backtest.default_method_set(optimizer, return_scale)

def rolling_backtest(returns, lookback_window, oos_window, step=None, method_set=None,
                     return_scale=1.0, risk_free_rate=0.0, periods_per_year=12):
    """Run a rolling backtest and return the per-window result dicts
    (window_returns, window_weights, window_pred_vol)"""
    bt = Backtest(returns, lookback_window, oos_window, step=step, method_set=method_set,
                  return_scale=return_scale, risk_free_rate=risk_free_rate,
                  periods_per_year=periods_per_year).run()
    return bt.window_returns, bt.window_weights, bt.window_pred_vol

def backtest_summary(window_returns, window_weights=None, window_pred_vol=None,
                     risk_free_rate=0.0, periods_per_year=12):
    """Per-method summary statistics from pre-computed per-window results"""
    return Backtest.from_results(window_returns, window_weights, window_pred_vol,
                                 risk_free_rate=risk_free_rate,
                                 periods_per_year=periods_per_year).summary()

def holding_period_returns(window_returns):
    """Holding-period (oos_window) return series from pre-computed per-window results"""
    return Backtest.from_results(window_returns).holding_period_returns()