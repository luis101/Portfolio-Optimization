

###### Robust Portfolio Optimization Suite ##########################################

# Not included is a rolling backtest framework, which is critical for evaluating out-of-sample performance.
# Fix an estimation window (e.g. 60 months)
# Estimate weights on that window, hold for 1 period
# Slide forward, repeat
# Compute realized statistics on the held-out returns
# Key out-of-sample metrics to compare:
# Realized annualized Sharpe, Realized vs. predicted volatility ratio, Maximum drawdown, Portfolio turnover 
# Effective N (1 / Σwᵢ²), statistical significance of Sharpe differences between methods

# Potential additions:
# 1/N equal weight. 
# Risk Parity / Equal Risk Contribution (ERC). Allocates weights such that each asset contributes equally to total portfolio variance (wᵢ · (Σw)ᵢ = constant).
# Hierarchical Risk Parity (HRP) (López de Prado, 2016). Uses hierarchical clustering on the correlation matrix to build a tree, 
# then allocates risk top-down along the tree. Requires no matrix inversion, so it is robust to near-singular covariances.
# Minimum Covariance Determinant (MCD) robust covariance. Your Ledoit-Wolf shrinkage handles estimation noise but not outliers. 
# MCD fits the covariance on the subset of observations with the smallest determinant, making it resistant to return outliers (crises, flash crashes).
# Transaction-cost-aware objective. Adding a turnover penalty κ · ‖w_t − w_{t-1}‖₁ to the CVXPY objective is a one-line change but makes the comparison realistic.


import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch
from sklearn.covariance import MinCovDet
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """
    A comprehensive portfolio optimization class with multiple robust methods.

    Implements multiple approaches to robust portfolio optimization including:
    1. Classical Markowitz optimization + Minimum variance
    2. Worst-case optimization with ellipsoidal uncertainty sets
    3. Distributional robustness with Wasserstein distance
    4. Black-Litterman (Bayesian approach)
    5. Resampling methods
    6. Robust covariance estimation (Ledoit-Wolf, MCD robust covariance, Factor Models)
    7. Tail & drawdown risk optimization (CVaR, Wasserstein CVaR, Mean-CDaR)
    8. Regularization with L1, L2, or as Elastic Net
    9. Risk parity and hierarchical risk parity
    10. Maximum diversification portfolio
    11. Growth-optimal (Kelly) portfolios: full, fractional, and worst-case robust
    12. Equal weight portfolio (naive diversification)
    """
    
    def __init__(self, returns_data, risk_free_rate=0.0, periods_per_year=12):
        """
        Initialize the optimizer with historical returns data

        Parameters:
        -----------
        returns_data : pd.DataFrame or np.ndarray
            Historical returns (T x N) where T is time periods and N is assets
        risk_free_rate : float
            Risk-free rate for Sharpe ratio calculations (per period)
        periods_per_year : int
            Number of return periods per year, used to annualize reported
            statistics (12 for monthly, 252 for daily, etc.)
        """

        if isinstance(returns_data, pd.DataFrame):
              self.returns = returns_data
              self.asset_names = returns_data.columns.tolist()
              self.assets = returns_data.columns
        else:
              self.returns = pd.DataFrame(returns_data)
              self.asset_names = [f'Asset {i+1}' for i in range(returns_data.shape[1])]
              self.assets = self.returns.columns

        self.n_assets = self.returns.shape[1]
        self.n_periods = self.returns.shape[0]
        self.rf = risk_free_rate
        self.ppy = periods_per_year
        
        # Calculate basic statistics (kept as plain ndarrays for uniform use downstream)
        self.mu = np.asarray(np.mean(self.returns, axis=0))
        self.cov = np.asarray(np.cov(self.returns.T))

    def calculate_portfolio_stats(self, weights):
        """Calculate portfolio statistics (returns/vol/Sharpe annualized via self.ppy)"""
        if weights is None:
            return None

        # Per-period figures
        period_return = self.mu @ weights
        period_volatility = np.sqrt(weights.T @ self.cov @ weights)
        period_sharpe = (period_return - self.rf) / period_volatility if period_volatility > 0 else 0

        # Annualized figures (returns scale with ppy, volatility/Sharpe with sqrt(ppy))
        return {
            'weights': dict(zip(self.assets, weights)),
            'expected_return': period_return * self.ppy,
            'volatility': period_volatility * np.sqrt(self.ppy),
            'sharpe_ratio': period_sharpe * np.sqrt(self.ppy)
        }
        
    def _portfolio_performance(self, weights, mu, cov):
        """Calculate portfolio return and volatility"""

        ret = np.dot(weights, mu)
        vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        return ret, vol
    
    def _neg_sharpe(self, weights, mu, cov):
        """Negative Sharpe ratio for minimization"""

        ret, vol = self._portfolio_performance(weights, mu, cov)
        sharpe = (ret - self.rf) / vol
        return -sharpe
    
    def _solve_mean_variance_sample(self, mu, cov, target_return, risk_aversion=1.0):
        """Helper function for mean-variance optimization"""
        w = cp.Variable(self.n_assets)
        if target_return is not None:
            constraints = [cp.sum(w) == 1, w >= 0, w <= 1, w.T @ mu >= target_return]
            objective = cp.Minimize(cp.quad_form(w, cov))
        else:
            constraints = [cp.sum(w) == 1, w >= 0, w <= 1]
            objective = cp.Minimize(-w.T @ mu + risk_aversion * cp.quad_form(w, cov))
        
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        return w.value if prob.status == 'optimal' else None
    
    def _solve_min_variance_sample(self, cov):
        """Helper function for minimum variance optimization"""
        w = cp.Variable(self.n_assets)
        constraints = [cp.sum(w) == 1, w >= 0, w <= 1]
        objective = cp.Minimize(cp.quad_form(w, cov))
        
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        return w.value if prob.status == 'optimal' else None
    
    # ========================================================================
    # 1. CLASSICAL MARKOWITZ OPTIMIZATION
    # ========================================================================

    def mean_variance_optimization(self, target_return=None, risk_aversion=1.0):
        """
        Classical Markowitz mean-variance optimization
        
        Parameters:
        target_return : float, optional
            Target return for the portfolio
        risk_aversion : float, optional
            Risk aversion coefficient for the optimization
        """
        # constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}] # Sum of weights = 1
        # bounds = tuple((0, 1) for _ in range(self.n_assets)) # No short selling
        # init_weights = np.ones(self.n_assets) / self.n_assets # Equal weights
        
        # Maximize Sharpe ratio
        # result = minimize(self._neg_sharpe, init_weights, args=(self.mu, self.cov), 
        #                  method='SLSQP', bounds=bounds, constraints=constraints)
        
        #weights = result.x
        #ret, vol = self._portfolio_performance(weights, self.mu, self.cov)
        
        # return {'weights': weights, 'return': ret, 'volatility': vol, 
        #        'sharpe': (ret - self.rf) / vol, 'method': 'Markowitz'}
        
        # Using cvxpy for quadratic programming
        w = cp.Variable(self.n_assets)
        
        if target_return is not None:
            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                w <= 1,
                w.T @ self.mu >= target_return
            ]
            objective = cp.Minimize(cp.quad_form(w, self.cov))
        else:
            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                w <= 1
            ]
            objective = cp.Minimize(-w.T @ self.mu + risk_aversion * cp.quad_form(w, self.cov))
        
        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = w.value if prob.status == 'optimal' else None

        # ret, vol = self._portfolio_performance(weights, self.mu, self.cov)
        # return {'weights': weights, 'return': ret, 'volatility': vol, 
        #        'sharpe': (ret - self.rf) / vol, 'method': 'Markowitz'}
        
        return weights
    
    def min_variance(self):
        """Minimum Variance Portfolio"""
        w = cp.Variable(self.n_assets)

        constraints = [cp.sum(w) == 1, w >= 0, w <= 1]
        objective = cp.Minimize(cp.quad_form(w, self.cov))
        
        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = w.value if prob.status == 'optimal' else None
        
        return weights
    
    # ========================================================================
    # 2. WORST-CASE OPTIMIZATION
    # ========================================================================

    def wasserstein_optimization(self, epsilon=0.1, norm_type=2, risk_aversion=1.0):
        """
        Distributionally robust mean-variance over a type-2 Wasserstein ball around
        the empirical distribution (Blanchet, Chen & Zhou 2022, Management Science).

        Exact tractable reformulation ("square-root regularization"): the worst-case
        portfolio standard deviation is the empirical std plus a sqrt(epsilon)-scaled
        norm of the weights, and the worst-case mean shrinks by the same penalty:
            sigma_wc(w) = sqrt(w' Σ w) + sqrt(epsilon) * ||w||
            mu_wc(w)    = w'μ          - sqrt(epsilon) * ||w||
        The robustification is additive in the *standard deviation*, not the variance.

        Parameters
        ----------
        epsilon : float
            Wasserstein ball radius (robustness parameter).
        norm_type : int
            Ground-cost norm. 2 -> Euclidean cost, penalty uses ||w||_2 (dual of l2);
            1 -> l1 cost, penalty uses ||w||_inf (dual of l1).
        risk_aversion : float
            Trade-off weight on the (robust) variance term.
        """

        w = cp.Variable(self.n_assets)

        portfolio_returns = w.T @ self.mu
        # portfolio_var = cp.quad_form(w, self.cov)
        # portfolio_std = cp.sqrt(portfolio_var)

        sqrt_eps = np.sqrt(epsilon)

        # Symmetric sqrt-factor of Σ so the empirical risk is a 2-norm (SOC)
        vals, vecs = np.linalg.eigh(self.cov)
        sigma_sqrt = vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T

        # Dual norm of the transport ground cost drives the robustness penalty
        if norm_type == 2:
            weight_penalty = cp.norm(w, 2)        # dual of Euclidean cost
            # wasserstein_penalty = epsilon * cp.norm(w, 2)
        elif norm_type == 1:
            weight_penalty = cp.norm(w, "inf")    # dual of l1 cost
            # wasserstein_penalty = epsilon * cp.norm(w, 1)
        else:
            raise ValueError("Unsupported norm type")

        # worst_case_variance = portfolio_var  + sqrt_eps * weight_penalty
        worst_case_variance = cp.norm(sigma_sqrt @ w, 2) + sqrt_eps * weight_penalty  # robust std
        robust_return = portfolio_returns - sqrt_eps * weight_penalty   # robust mean

        # Mean - risk_aversion * (robust variance), with robust variance = sigma_wc^2
        objective = cp.Minimize(-(robust_return - self.rf) + risk_aversion * cp.square(worst_case_variance))
        constraints = [cp.sum(w) == 1, w >= 0, w <= 1]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = w.value if prob.status == 'optimal' else None

        #def wasserstein_robust_objective(w, norm_type='2'):
            # """
            # Robust objective: minimize worst-case CVaR over Wasserstein ball
            # Approximation using moment-based approach
            # """
            # Portfolio return and variance
            # portfolio_returns = np.dot(w, self.mu)
            # portfolio_var = np.dot(w, np.dot(self.cov, w))
            # portfolio_std = np.sqrt(portfolio_var)
            # Worst-case adjustment (simplified Wasserstein penalty)
            # Based on: E[R] - epsilon * ||grad E[R]||
            # wasserstein_penalty = epsilon * portfolio_std * np.sqrt(self.n_assets)
            # robust_return = portfolio_returns - wasserstein_penalty

            # Maximize risk-adjusted return
            # return -(robust_return - self.rf) / (portfolio_std + 1e-8)
        
        # constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        # bounds = tuple((0, 1) for _ in range(self.n_assets))
        # init_weights = np.ones(self.n_assets) / self.n_assets
        
        # result = minimize(wasserstein_robust_objective, init_weights, method='SLSQP',
        #                   bounds=bounds, constraints=constraints)
        
        # weights = result.x
        # ret, vol = self._portfolio_performance(weights, self.mu, self.cov)
        # return {'weights': weights, 'return': ret, 'volatility': vol, 'sharpe': (ret - self.rf) / vol,
        #        'epsilon': epsilon, 'method': 'Wasserstein Robust'}

        return weights

    def ellipsoidal_uncertainty_optimization(self, kappa_mu=0.1, kappa_sigma=0.1, risk_aversion=1.0):
        """
        Robust optimization with ellipsoidal uncertainty sets
        for both expected returns and covariance matrix
        """

        # Standard error of mean estimates
        # std_error = np.sqrt(np.diag(self.cov) / self.n_periods)
        
        # def robust_objective(w):
            # """Minimize worst-case Sharpe ratio."""
            # Nominal return
            # nominal_return = np.dot(w, self.mu)
            
            # Worst-case adjustment (robust counterpart)
            # portfolio_vol = np.sqrt(np.dot(w, np.dot(self.cov, w)))
            # uncertainty_penalty = kappa * np.sqrt(np.dot(w**2, std_error**2))
            
            # worst_case_return = nominal_return - uncertainty_penalty
            
            # Negative Sharpe for minimization
            # return -(worst_case_return - self.rf) / (portfolio_vol + 1e-8)
        
        # constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}] # Sum of weights = 1
        # bounds = tuple((0, 1) for _ in range(self.n_assets)) # No short selling
        # init_weights = np.ones(self.n_assets) / self.n_assets # Equal weights
        
        # result = minimize(robust_objective, init_weights, method='SLSQP', 
        # bounds=bounds, constraints=constraints) 
        # weights = result.x

        # Using cvxpy for robust optimization
        w = cp.Variable(self.n_assets)

        # Symmetric matrix square root of the mean estimate's sampling covariance
        # (cp.sqrt would be an element-wise sqrt. We need the matrix square root, which can be done via eigendecomposition.)
        # Built via eigendecomposition with eigenvalues clipped at 0 to ensure positive semidefiniteness.
        vals, vecs = np.linalg.eigh(self.cov / self.n_periods)
        sigma_mu_sqrt = vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T

        # Uncertainty sets
        mu_uncertainty = kappa_mu * cp.norm(sigma_mu_sqrt @ w, 2)  # kappa * sqrt(w' (Sigma/T) w)
        sigma_uncertainty = kappa_sigma * cp.norm(w, 2)

        # Worst-case expected return (min over ellipsoidal uncertainty)
        worst_case_return = w.T @ self.mu - mu_uncertainty

        # Worst-case variance (max over ellipsoidal uncertainty)
        # Simplified approach - in practice this would be more complex
        # cov_robust = (1 + kappa_sigma) * self.cov
        # worst_case_variance = cp.quad_form(w, cov_robust)
        worst_case_variance = cp.quad_form(w, self.cov) + sigma_uncertainty

        constraints = [cp.sum(w) == 1, w >= 0, w <= 1]

        # Objective: maximize worst-case return - risk_aversion * worst-case variance
        # objective = cp.Maximize(worst_case_return - risk_aversion * worst_case_variance)
        objective = cp.Minimize(-worst_case_return + risk_aversion * worst_case_variance)

        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = w.value if prob.status == 'optimal' else None

        # ret, vol = self._portfolio_performance(weights, self.mu, self.cov)
        # return {'weights': weights, 'return': ret, 'volatility': vol, 
        #        'sharpe': (ret - self.rf) / vol, 'method': 'Robust Ellipsoidal Uncertainty Set'}
        
        return weights
    
    # ========================================================================
    # 3. BLACK-LITTERMAN MODEL
    # ========================================================================

    def black_litterman(self, market_caps=None, tau=0.05, risk_aversion=2.5,
                        views=None, view_returns=None, view_confidences=None):
        """
        Black-Litterman model combining market equilibrium with investor views

        Parameters:
        -----------
        market_caps : np.ndarray, optional
            Market capitalizations (used to derive equilibrium weights)
        tau : float
            Uncertainty in prior (typically 0.01-0.05)
        risk_aversion : float
            Market risk aversion coefficient
        views : np.ndarray, optional
            View matrix P (K x N) for K views on N assets
        view_returns : np.ndarray, optional
            View magnitudes Q (length K). The expected return implied by each view.
            Defaults to zeros (no opinion on the level) when omitted.
        view_confidences : np.ndarray, optional
            Confidence in views (K x K diagonal matrix or vector) - higher means more confidence
            Reflecting Omega and omega_scale: Omega = np.eye(k) * omega_scale
        """
        # If no market caps provided, use minimum variance weights
        if market_caps is None:
            # w_mkt = np.ones(self.n_assets) / self.n_assets
            w_mkt = self.min_variance()
        else:
            w_mkt = market_caps / np.sum(market_caps)
        
        # Implied equilibrium returns (reverse optimization)
        pi = risk_aversion * np.dot(self.cov, w_mkt)
        
        # If no views provided, use equilibrium
        if views is None:
            mu_bl = pi
            cov_bl = self.cov
        # Else use views to adjust returns
        else:
            P = views  # View matrix
            # View magnitudes Q: expected return implied by each view.
            # Defaults to zeros (no opinion on the level) when not supplied.
            Q = np.zeros(len(views)) if view_returns is None else np.asarray(view_returns)
            
            # Omega: diagonal matrix of view uncertainties
            if view_confidences is None:
                # Default: proportional to variance of views
                Omega = np.diag(np.diag(P @ self.cov @ P.T)) * tau
                # Alternative with scaling
                # Omega = np.eye(k) * 0.1
            else:
                if isinstance(view_confidences, np.ndarray) and view_confidences.ndim == 1:
                    Omega = np.diag(view_confidences)
                else:
                    Omega = view_confidences
            
            # Black-Litterman formula
 
            # Posterior mean

            # M1 = np.linalg.inv(tau * self.Sigma)
            # M2 = P.T @ np.linalg.inv(Omega) @ P
            # M3 = M1 @ Pi + P.T @ np.linalg.inv(Omega) @ Q
            # mu_bl = np.linalg.inv(M1 + M2) @ M3
            M = np.linalg.inv(np.linalg.inv(tau * self.cov) + P.T @ np.linalg.inv(Omega) @ P)
            mu_bl = M @ (np.linalg.inv(tau * self.cov) @ pi + P.T @ np.linalg.inv(Omega) @ Q)
            
            # Posterior covariance
            cov_bl = self.cov + M
        
        # Optimize with Black-Litterman parameters
        # constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        # bounds = tuple((0, 1) for _ in range(self.n_assets))
        # init_weights = np.ones(self.n_assets) / self.n_assets
        
        # result = minimize(self._neg_sharpe, init_weights, args=(mu_bl, cov_bl, self.rf),
        #                  method='SLSQP', bounds=bounds, constraints=constraints)
        
        # weights = result.x

        weights = self._solve_mean_variance_sample(mu_bl, cov_bl, target_return=None,
                                                   risk_aversion=risk_aversion)

        # ret, vol = self._portfolio_performance(weights, mu_bl, cov_bl)
        # return {'weights': weights, 'return': ret, 'volatility': vol, 
        #        'sharpe': (ret - self.rf) / vol, 
        #        'posterior_returns': mu_bl, 'method': 'Black-Litterman'}
        
        return weights
    
    # ========================================================================
    # 4. RESAMPLING
    # ========================================================================
    
    def resampling_optimization(self, n_samples=1000, target_return=None, risk_aversion=1.0, seed=None):
        """
        Michaud's resampled efficient frontier approach.
        
        Parameters:
        -----------
        n_samples : int
            Number of resampled scenarios
        target_return : float, optional
            Target return for optimization
        risk_free_rate : float
            Risk-free rate
        seed : int, optional
            Random seed for reproducibility (None for non-deterministic sampling)
        """
        # Local generator: reproducible without touching global NumPy RNG state
        rng = np.random.default_rng(seed)

        # Store weights from each resampled optimization
        resampled_weights = []

        for _ in range(n_samples):
            # Resample returns (bootstrap)
            indices = rng.choice(self.n_periods, size=self.n_periods, replace=True)
            sample_returns = self.returns.iloc[indices]
            # sample_returns = self.returns[sample_idx]
            
            # Estimate parameters from resampled data
            mu_sample = np.mean(sample_returns, axis=0)
            cov_sample = np.cov(sample_returns.T)
            
            # Optimize for this sample
            # constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
            # if target_return is not None:
            #     constraints.append({'type': 'eq',
            #                         'fun': lambda w: np.dot(w, mu_sample) - target_return})
            
            # bounds = tuple((0, 1) for _ in range(self.n_assets))
            # init_weights = np.ones(self.n_assets) / self.n_assets
            
            try:
                if target_return is None:
                    # result = minimize(self._neg_sharpe, init_weights, args=(mu_sample, cov_sample, self.rf),
                    #                   method='SLSQP', bounds=bounds, constraints=constraints)
                    w = cp.Variable(self.n_assets)
                    constraints = [cp.sum(w) == 1, w >= 0, w <= 1]
                    objective = cp.Minimize(-w.T @ mu_sample + risk_aversion * cp.quad_form(w, cov_sample))
                    # objective = cp.Minimize(cp.quad_form(w, cov_sample))
        
                    prob = cp.Problem(objective, constraints)
                    prob.solve()
                else:
                    # result = minimize(lambda w: np.dot(w, np.dot(cov_sample, w)), init_weights,
                    #                   method='SLSQP', bounds=bounds, constraints=constraints)
                    w = cp.Variable(self.n_assets)
                    constraints = [cp.sum(w) == 1, w >= 0, w <= 1, w.T @ mu_sample >= target_return]
                    objective = cp.Minimize(cp.quad_form(w, cov_sample))
        
                    prob = cp.Problem(objective, constraints)
                    prob.solve()

                w_sample = w.value if prob.status == 'optimal' else None

                if w_sample is not None:
                    resampled_weights.append(w_sample)      
                
                # if result.success:
                #    resampled_weights.append(result.x)
            except:
                continue
        
        # Average the resampled weights
        weights = np.mean(resampled_weights, axis=0)
        weights = weights / np.sum(weights)  # Normalize to sum to 1
        
        # weight = result.x
        # ret, vol = self._portfolio_performance(weights, self.mu, self.cov)
        # return {'weights': weights, 'return': ret, 'volatility': vol, 'sharpe': (ret - self.rf) / vol,
        #         'n_successful_samples': len(resampled_weights), 'method': 'Resampled Efficiency'}
    
        return weights
    
    # ========================================================================
    # 5. ROBUST COVARIANCE ESTIMATION
    # ========================================================================
    
    # Ledoit-Wolf shrinkage covariance estimation
    
    def shrinkage_covariance_optimization(self, risk_aversion=1.0):
        """
        Optimization using Ledoit-Wolf shrinkage covariance estimator
        """
        # Ledoit-Wolf shrinkage
        cov_lw = self._ledoit_wolf_shrinkage(self.returns)
        
        # constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        # bounds = tuple((0, 1) for _ in range(self.n_assets))
        # init_weights = np.ones(self.n_assets) / self.n_assets
        
        # result = minimize(self._neg_sharpe, init_weights, args=(self.mu, cov_lw, self.rf),
        #                   method='SLSQP', bounds=bounds, constraints=constraints)

        weights = self._solve_mean_variance_sample(self.mu, cov_lw, risk_aversion=risk_aversion, 
                                                   target_return=None)
        
        # weights = result.x
        # ret, vol = self._portfolio_performance(weights, self.mu, cov_lw)
        # return {'weights': weights, 'return': ret, 'volatility': vol, 'sharpe': (ret - self.rf) / vol,
        #         'shrinkage_covariance': cov_lw, 'method': 'Ledoit-Wolf Shrinkage Covariance'}

        return weights
    
    def _ledoit_wolf_shrinkage(self, returns):
        """
        Compute Ledoit-Wolf shrinkage estimator of covariance matrix
        """
        T, N = returns.shape
        
        # Shrinkage target: constant correlation model

        var_diag = np.diag(self.cov)
        # sqrt_var = np.sqrt(var)
        # sample_cor = self.cov / np.outer(sqrt_var, sqrt_var)
        sample_cor = self.cov / np.sqrt(np.outer(var_diag, var_diag))
        # avg_cor = (np.sum(sample_cor) - N) / (N * (N - 1))
        avg_cor = (2/(N*(N-1))) * (np.sum(np.triu(sample_cor, 1))) # Average off-diagonal correlation        
        target = avg_cor * np.sqrt(np.outer(var_diag, var_diag))
        np.fill_diagonal(target, var_diag)
        
        # Optimal shrinkage intensity   
        # shrinkage = max(0, min(0.0, (N + 1) / (T * (N + 1) + 2))) # Simplified calculation

        # ---------- π ----------
        X = returns - np.mean(returns, axis=0) # Demeaned returns
        X = X.values  # Convert to numpy array
        pi_hat = 0.0
        for t in range(T):
            x_t = X[t, :]
            Xt = np.outer(x_t, x_t)
            pi_hat += np.sum((Xt - self.cov) ** 2)
        pi_hat /= T

        # ---------- ρ -----------
        # diagonal terms
        pi_diag = np.zeros((N, N))
        for t in range(T):
            x_t = X[t, :]
            Xt = np.outer(x_t, x_t)
            pi_diag += (Xt - self.cov) ** 2
        pi_diag /= T
        rho_diag = np.sum(np.diag(pi_diag))

        # off-diagonal terms
        rho_off = 0.0
        r_bar = avg_cor.copy()
        X_squared = X ** 2  # T x N
        for t in range(T):
            x_t = X[t, :]  # Shape: (N,)
            outer_t = np.outer(x_t, x_t)  # Shape: (N, N)
            cov_dev = outer_t - self.cov  # Shape: (N, N)
            # var_dev[i] = x_t[i]^2 - self.cov[i]
            var_dev = X_squared[t, :] - self.cov  # Shape: (N,)
            # For each pair (i,j), compute contribution
            # theta_ii_ij = var_dev[i] * cov_dev[i,j]
            # theta_jj_ij = var_dev[j] * cov_dev[i,j]
            # Vectorized computation of theta terms
            # Broadcasting: var_dev[:,None] has shape (N,1), cov_dev has shape (N,N)
            theta_ii = var_dev[:, None] * cov_dev  # Shape: (N, N)
            theta_jj = var_dev[None, :] * cov_dev  # Shape: (N, N)
            # Compute sqrt ratios
            # sqrt_ratio_j_i[i,j] = sqrt(self.cov[j] / self.cov[i])
            sqrt_ratio_j_i = np.sqrt(self.cov[None, :] / self.cov[:, None])
            sqrt_ratio_i_j = np.sqrt(self.cov[:, None] / self.cov[None, :])
            # Sum contributions (exclude diagonal where i == j)
            mask = ~np.eye(N, dtype=bool)
            rho_off += np.sum((theta_ii * sqrt_ratio_j_i + theta_jj * sqrt_ratio_i_j)[mask])

        rho_off /= T
        rho_off *= (r_bar / 2)
        rho_hat = rho_diag + rho_off

        # ---------- γ ----------
        gamma_hat = np.sum((target - self.cov) ** 2)

        # ---------- δ ----------
        kappa_hat = (pi_hat - rho_hat) / gamma_hat
        # delta_hat = np.clip(kappa_hat / T, 0.0, 1.0)
        delta_hat = max(0.0, min(kappa_hat / T, 1.0))

        # Shrunk covariance
        cov_shrunk = delta_hat * target + (1 - delta_hat) * self.cov
        
        return cov_shrunk
    
    # Minimum covariance determinant (MCD) robust covariance estimation

    def mcd_robust_covariance_optimization(self, risk_aversion=1.0, support_fraction=None, 
                                           random_state=None):
        """
        Minimum Covariance Determinant (MCD) robust covariance estimation

        Parameter:
        ----------
        support_fraction : float, optional (default=None)
            Fraction of observations to include in the support of the raw MCD estimate.
            Typically between 0.5 and 1.0, with 0.75 being a common choice for moderate robustness.
        """
        # MCD estimator
        mcd = MinCovDet(support_fraction=support_fraction, random_state=random_state)

        # Fit on returns data
        mcd.fit(self.returns)
        mu_mcd = mcd.location_
        cov_mcd = mcd.covariance_

        # constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        # bounds = tuple((0, 1) for _ in range(self.n_assets))
        # init_weights = np.ones(self.n_assets) / self.n_assets
        
        # result = minimize(self._neg_sharpe, init_weights, args=(self.mu, cov_lw, self.rf),
        #                   method='SLSQP', bounds=bounds, constraints=constraints)

        weights = self._solve_mean_variance_sample(mu_mcd, cov_mcd, risk_aversion=risk_aversion, 
                                                   target_return=None)
        
        # weights = result.x
        # ret, vol = self._portfolio_performance(weights, mu_mcd, cov_mcd)
        # return {'weights': weights, 'return': ret, 'volatility': vol, 'sharpe': (ret - self.rf) / vol,
        #         'shrinkage_covariance': cov_mcd, 'method': 'MCD Robust Covariance'}

        return weights
    
    # Factor model covariance estimation
    
    def factor_model_optimization(self, n_factors=3, factor_returns=None):
        """
        Optimization using factor model for covariance estimation
        
        Parameters:
        -----------
        n_factors : int
            Number of factors to use
        """

        if factor_returns is not None:
            # Use provided factor returns to estimate factor loadings
            # Regress asset returns on factor returns
            factor_returns = factor_returns.values
            # T = self.n_periods
            factor_loadings = np.zeros((self.n_assets, n_factors))
            specific_var = np.zeros(self.n_assets)
            
            for i in range(self.n_assets):
                model = LinearRegression(fit_intercept=True).fit(factor_returns, self.returns.iloc[:, i])
                factor_loadings[i, :] = model.coef_
                residuals = self.returns.iloc[:, i] - model.predict(factor_returns)
                specific_var[i] = np.var(residuals)
            
            # Factor covariance
            F_cov = np.cov(factor_returns.T)
            
            # Reconstruct covariance: B @ F_cov @ B^T + D
            cov_factor = factor_loadings @ F_cov @ factor_loadings.T + np.diag(specific_var)
        else:
            # Simple PCA-based factor model
            # Center the returns
            returns_centered = self.returns - np.mean(self.returns, axis=0)
            
            # Compute covariance
            C = np.cov(returns_centered.T)
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            
            # Sort in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Keep top k factors
            k = min(n_factors, self.n_assets - 1)
            factor_loadings = eigenvectors[:, :k] * np.sqrt(eigenvalues[:k])
        
            # Specific variances (diagonal)
            specific_var = np.diag(C) - np.sum(factor_loadings**2, axis=1)
            specific_var = np.maximum(specific_var, 1e-6)  # Ensure positive
        
            # Reconstruct covariance: B @ B^T + D
            cov_factor = factor_loadings @ factor_loadings.T + np.diag(specific_var)
        
        # Optimize with factor model covariance

        # constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        # bounds = tuple((0, 1) for _ in range(self.n_assets))
        # init_weights = np.ones(self.n_assets) / self.n_assets
        
        # result = minimize(self._neg_sharpe, init_weights, args=(self.mu, cov_factor, risk_free_rate),
        #                   method='SLSQP', bounds=bounds, constraints=constraints)

        weights = self._solve_mean_variance_sample(self.mu, cov_factor, target_return=None)
        
        # weights = result.x
        # ret, vol = self._portfolio_performance(weights, self.mu, cov_factor)
        # return {'weights': weights, 'return': ret, 'volatility': vol, 
        #         'sharpe': (ret - self.rf) / vol, 'n_factors': k, 'method': 'Factor Model Covariance'}
    
        return weights
    
    # ========================================================================
    # 6. TAIL & DRAWDOWN RISK OPTIMIZATION (CVaR / CDaR)
    # ========================================================================

    def cvar_optimization(self, alpha=0.05, risk_aversion=1.0):
        """
        Conditional Value at Risk optimization
        alpha: Confidence level (e.g., 0.05 for 95% CVaR)
        """

        w = cp.Variable(self.n_assets)
        VaR = cp.Variable()
        loss = cp.Variable(self.n_periods)
        
        constraints = [
            cp.sum(w) == 1, w >= 0, w <= 1,
            loss >= -self.returns.values @ w - VaR,
            loss >= 0
        ]
        
        CVaR = VaR + (1/(alpha * self.n_periods)) * cp.sum(loss)
        expected_return = self.mu @ w

        objective = cp.Minimize(-expected_return + risk_aversion * CVaR)
        
        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = w.value if prob.status == 'optimal' else None
        
        return weights

    def wasserstein_cvar_optimization(self, epsilon=0.1, alpha=0.05, risk_aversion=1.0):
        """
        Wasserstein robust CVaR optimization
        Combines distributional robustness with tail risk management

        Wasserstein robust CVaR (Mohajerin Esfahani & Kuhn 2018): 
        For a type-1 ball with Euclidean cost and a loss affine in the returns, the worst-case CVaR is
        the empirical CVaR plus epsilon times the dual-norm Lipschitz constant ||w||_2.
        """
        
        w = cp.Variable(self.n_assets)
        VaR = cp.Variable()
        loss = cp.Variable(self.n_periods)
        
        # CVaR constraints
        constraints = [
            cp.sum(w) == 1, w >= 0, w <= 1,
            loss >= -self.returns.values @ w - VaR,
            loss >= 0
        ]
        
        # Empirical CVaR
        CVaR = VaR + (1/(alpha * self.n_periods)) * cp.sum(loss)
        
        robustness_margin = epsilon * cp.norm(w, 2)

        robust_CVaR = CVaR + robustness_margin
        expected_return = self.mu @ w - epsilon * cp.norm(w, 2)  # Worst-case mean (same margin)
        
        objective = cp.Minimize(-expected_return + risk_aversion * robust_CVaR)
        
        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = w.value if prob.status == 'optimal' else None

        return weights

    def cdar_optimization(self, alpha=0.05, risk_aversion=1.0):
        """
        Mean-Conditional Drawdown-at-Risk (Mean-CDaR) optimization
        (Chekhlov, Uryasev & Zabarankin, 2005).

        CDaR is the drawdown analogue of CVaR: the average of the worst (alpha-tail)
        portfolio drawdowns along the cumulative-return path. Unlike CVaR, a single-
        period tail loss, CDaR penalises sustained peak-to-trough declines, which makes
        it well suited to long-horizon investors who care about drawdown risk.

        Parameters
        ----------
        alpha : float
            Tail probability for the drawdown (e.g. 0.05 -> average of the worst 5%
            drawdowns; alpha -> 1 approaches the average drawdown, alpha -> 0 the maximum).
        risk_aversion : float
            Trade-off weight on CDaR versus expected return.
        """
        T = self.n_periods

        # Uncompounded cumulative return path is affine in w:
        #   C_t = sum_{k<=t} (R_k . w) = (tril(ones) @ R) @ w
        cum_returns = np.tril(np.ones((T, T))) @ self.returns.values  # (T, N)

        w = cp.Variable(self.n_assets)
        u = cp.Variable(T)    # running peak (high-water mark) of the cumulative path
        z = cp.Variable(T)    # drawdown exceedances over the threshold zeta
        zeta = cp.Variable()  # Drawdown-at-Risk threshold (DaR)

        cumulative = cum_returns @ w  # cumulative return at each t (affine in w)

        constraints = [
            cp.sum(w) == 1, w >= 0, w <= 1,
            u >= cumulative,            # peak is at least the current cumulative value
            u[1:] >= u[:-1],            # peak is non-decreasing (running maximum)
            u >= 0,                     # peak measured from initial capital (drawdown from 0)
            z >= u - cumulative - zeta, # exceedance of drawdown D_t = u - cumulative over zeta
            z >= 0
        ]

        # Rockafellar-Uryasev representation of CDaR: minimize over zeta the sum of zeta and the average exceedance
        CDaR = zeta + (1.0 / (alpha * T)) * cp.sum(z)
        expected_return = self.mu @ w

        objective = cp.Minimize(-expected_return + risk_aversion * CDaR)

        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = w.value if prob.status == 'optimal' else None

        return weights

    # ========================================================================
    # 7. ELASTIC NET REGULARIZATION
    # ========================================================================
    
    def elastic_net_optimization(self, risk_aversion=2.0, lambda_l1=0.01, lambda_l2=0.01):
        """
        Portfolio optimization with Elastic Net regularization
        Combines L1 (sparsity) and L2 (diversification) penalties
        
        Parameters:
        -----------
        lambda_l1 : float
            L1 regularization parameter (encourages sparsity)
        lambda_l2 : float
            L2 regularization parameter (penalizes large positions)
        """

        w = cp.Variable(self.n_assets)

        # Portfolio return and risk
        portfolio_return = self.mu @ w
        portfolio_var = cp.quad_form(w, self.cov)
        
        # Sharpe ratio component
        # sharpe_component = -(portfolio_return - self.rf) / np.sqrt(portfolio_var)

        # Mean-variance utility (avoid sqrt for convexity)
        # Using quadratic utility: return - (1/2) * risk_aversion * variance
        utility = portfolio_return - (risk_aversion / 2) * portfolio_var
        
        # Elastic Net penalty: lambda1 * ||w||_1 + lambda2 * ||w||_2^2
        # Effectively L1 encourages sparsity (fewer assets), while L2 encourages diversification (smaller weights)

        # l1_penalty = lambda_l1 * w.sum()
        l1_penalty = lambda_l1 * cp.norm1(w)
        # l2_penalty = lambda_l2 * np.sum(w**2)
        # l2_penalty = lambda_l2 * cp.sum_squares(w)
        l2_penalty = lambda_l2 * cp.norm(w, 2)**2

        utility_penalized = -utility  + l1_penalty + l2_penalty

        objective = cp.Minimize(utility_penalized)
        constraints = [cp.sum(w) == 1, w >= 0, w <= 1]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        # def elastic_net_objective(w):
            # """Objective with Elastic Net penalty."""
            # Portfolio return and risk
            # portfolio_return = np.dot(w, self.mu)
            # portfolio_var = np.dot(w, np.dot(self.cov, w))
            # Sharpe ratio component
            # sharpe_component = -(portfolio_return - self.rf) / np.sqrt(portfolio_var)
            # Elastic Net penalty: lambda1 * ||w||_1 + lambda2 * ||w||_2^2
            # l1_penalty = lambda_l1 * np.sum(np.abs(w))
            ### l2_penalty = lambda_l2 * np.sum(w**2)
            # l2_penalty = lambda_l2 * cp.norm(w, 2)**2
            
            # return sharpe_component + l1_penalty + l2_penalty
        
        # constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        # bounds = tuple((0, 1) for _ in range(self.n_assets))
        # init_weights = np.ones(self.n_assets) / self.n_assets
        
        # result = minimize(elastic_net_objective, init_weights, method='SLSQP',
        #                   bounds=bounds,constraints=constraints)
        
        # weights = result.x
        weights = w.value if prob.status == 'optimal' else None

        # Clean up very small weights (from L1 regularization)
        weights[weights < 1e-4] = 0
        weights = weights / np.sum(weights)  # Re-normalize
        
        # ret, vol = self._portfolio_performance(weights, self.mu, self.cov) 
        # return {'weights': weights, 'return': ret, 'volatility': vol, 'sharpe': (ret - self.rf) / vol,
        #         'n_nonzero': np.sum(weights > 1e-4), 'method': 'Elastic Net Regularization'}
        
        return weights
    
    # ========================================================================
    # 8. RISK BASED ALLOCATION
    # ========================================================================

    # 1. Hierarchical Risk Parity (HRP)
    
    def hierarchical_risk_parity(self, cov=None, method="ward"):
        """
        Performs Hierarchical Risk Parity (HRP) based on Marcos Lopez de Prado.
        1. Tree Clustering (Hierarchical Clustering via correlation distance)
        2. Quasi-Diagonalization (Sorting covariance matrix)
        3. Recursive Bisection (Inverse-variance allocation across clusters)

        Parameters:
        cov : np.ndarray, optional
        Covariance matrix to use (if None, uses self.cov)
        method : str
        'single' for single linkage, 'ward' for Ward's method (more balanced clusters)
        """

        cov = cov if cov is not None else self.cov

        # Step 1: Tree Clustering
        corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        # corr = pd.DataFrame(self.Sigma, index=self.assets, columns=self.assets)
        # corr = self.returns.corr()
        distance = np.sqrt(0.5 * (1.0 - corr))   
        dist_flat = squareform(distance, checks=False)

        if method == "ward":
            try:
                # Try Ward's method first (more balanced clusters)
                linkage = sch.linkage(dist_flat, method='ward')
            except:
                # Fallback: average linkage
                linkage = sch.linkage(dist_flat, method='average')

        else:
            linkage = sch.linkage(dist_flat, method=method)

        # Step 2: Quasi-Diagonalization
        # Get sorted indices from hierarchical clustering
        sorted_indices = sch.leaves_list(linkage)
        assets_sorted = [self.assets[i] for i in sorted_indices]
        cov_sorted = cov[sorted_indices][:, sorted_indices]

        # Step 3: Recursive Bisection
        # weights = pd.Series(1.0, index=range(self.n_assets))  
        weights = np.ones(self.n_assets, dtype=np.float64) # Start with equal weights

        def _cluster_var(cov, indices):
            """Variance of the inverse-variance portfolio within a cluster"""
            sub_cov = cov[np.ix_(indices, indices)]
            ivp     = 1.0 / np.diag(sub_cov)
            ivp    /= ivp.sum()
            return float(ivp @ sub_cov @ ivp)

        def _recursive_bisection(indices=None):   
            """Recursively allocate weights to clusters"""
            n = self.cov.shape[0]
            if indices is None:
                indices = range(n)

            if len(indices) == 1:
                return np.array([1.0])  # Single asset gets all weight
            
            if len(indices) <= 1:
                return
            
            # Split into two clusters
            split = len(indices) // 2
            left_indices = indices[:split]
            right_indices = indices[split:]

            # Compute inverse variance for each cluster
            var_left = _cluster_var(cov, left_indices)
            var_right = _cluster_var(cov, right_indices)

            # No division by zero - if both variances are zero, split equally
            alpha = 1.0 - (var_left / (var_left + var_right)) if (var_left + var_right) > 1e-8 else 0.5
            
            # Allocate weights to clusters
            weights[left_indices] *= alpha
            weights[right_indices] *= (1.0 - alpha)

            # Recurse on each cluster
            _recursive_bisection(left_indices)
            _recursive_bisection(right_indices)

        def _recursive_bisection_alternative(cov, indices=None): 
            """Recursive bisection with explicit queue (iterative approach)"""

            queue = [indices if indices is not None else list(range(cov.shape[0]))]
            
            while len(queue) > 0:
                # Take the next cluster from the queue
                current_cluster = queue.pop(0)
                if len(current_cluster) <= 1:
                    continue
                
                # Split into two clusters
                split = len(current_cluster) // 2
                left_cluster = current_cluster[:split]
                right_cluster = current_cluster[split:]
                
                # Compute variance for each cluster using inverse-variance portfolio
                var_left = _cluster_var(cov, left_cluster)
                var_right = _cluster_var(cov, right_cluster)
                
                # Allocate weights based on relative variances (avoid division by zero)
                alpha = 1.0 - (var_left / (var_left + var_right)) if (var_left + var_right) > 1e-8 else 0.5
                
                # Scale weights of assets in each cluster
                weights[left_cluster] *= alpha
                weights[right_cluster] *= (1.0 - alpha)
                
                # Add sub-clusters back to the queue for further splitting
                queue.append(left_cluster)
                queue.append(right_cluster)
                
            return weights
    
        _recursive_bisection(sorted_indices)

        # Final normalization to ensure weights sum to 1  
        # w        = weights.copy()
        weights /= weights.sum()

        # Alternative iterative approach (uncomment to use)
        # weights = _recursive_bisection_alternative(cov, sorted_indices)

        return weights
    
    # 2. Risk Parity / Equal Risk Contribution (ERC)

    def risk_parity_optimization(self, method='slsqp', cov=None):
        """
        Risk Parity Portfolio with Equal Risk Contribution
        Each asset contributes equally to the overall portfolio risk.

        Parameter:
        ----------
        method : str
            'slsqp' for SLSQP, 'spinu' for Spinu's method
        """

        def _risk_parity_optimization(cov=None):
            """
            Risk Parity via numerical optimization (SLSQP)
            """
            cov = cov if cov is not None else self.cov

            def risk_contributions(weights):
                """Calculate risk contributions for given weights"""
                portfolio_var = weights @ cov @ weights
                if portfolio_var < 1e-10:
                    return np.ones(self.n_assets) * (1.0 / self.n_assets)
                
                marginal_risk = cov @ weights
                risk_contrib = weights * marginal_risk
                return risk_contrib / portfolio_var
            
            def objective(weights):
                """Objective function: Minimize deviation of risk contributions"""
                rc = risk_contributions(weights)
                target_rc = np.ones(self.n_assets) / self.n_assets
                return np.sum((rc - target_rc) ** 2)
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'ineq', 'fun': lambda w: w}  # w >= 0
            ]
            
            # Initial guess: Equal weights
            w0 = np.ones(self.n_assets) / self.n_assets
            bounds = [(0, 1) for _ in range(self.n_assets)]
            
            result = minimize(objective, w0, method='SLSQP', 
                            constraints=constraints, bounds=bounds)
            
            if result.success:
                return result.x
            else:
                print(f"Risk Parity optimization failed: {result.message}")
                return w0
        
        def _risk_parity_spinu(cov=None):
            """
            Risk Parity via Spinu's method (iterative scaling).
            Fast convergence for large portfolios.
            """
            cov = cov if cov is not None else self.cov

            x = cp.Variable(self.n_assets)
            
            # Convex objective function
            objective = cp.Minimize(0.5 * cp.quad_form(x, cov) - cp.sum(cp.log(x)))
            
            # Only constraint needed is x > 0
            constraints = [x >= 1e-5]
            
            prob = cp.Problem(objective, constraints)
            prob.solve()

            if prob.status == 'optimal':
                # Normalize x to get the actual portfolio weights w
                weights = x.value / np.sum(x.value)
                weights[weights < 1e-8] = 0 # Zero out tiny weights
                weights /= np.sum(weights) # Re-normalize

                return weights
            else:
                print(f"CVXPY Risk Parity failed: {prob.status}")
                return None

        if method == 'spinu':
            return _risk_parity_spinu(cov=cov)
        else:
            return _risk_parity_optimization(cov=cov)

    # 3. Maximum Diversification Portfolio (MDP)

    def maximum_diversification(self, cov=None):
        """
        Maximum Diversification Portfolio (Choueifaty & Coignard, 2008).

        Maximises the Diversification Ratio (DR), defined as the ratio of the
        weighted average of individual asset volatilities to the portfolio volatility.
        """

        cov = cov if cov is not None else self.cov    

        # Extract volatilities (standard deviations) from the covariance matrix
        vols = np.sqrt(np.diag(cov))

        w = cp.Variable(self.n_assets)
        
        # Target: Minimize portfolio variance (which is equivalent to maximizing diversification ratio)
        objective = cp.Minimize(cp.quad_form(w, cov))
        
        # Constraints: weighted sum of individual volatilities equals 1, and long-only
        constraints = [
            vols @ w == 1,  # The weighted sum of individual volatilities equals 1
            w >= 0          # Long-only constraint
        ]
        
        prob = cp.Problem(objective, constraints)
        prob.solve()

        w_value = w.value if prob.status == 'optimal' else None

        # def neg_diversification_ratio(w):
            # Weighted average of individual volatilities
            # weighted_vols = w.dot(vols)
            # Portfolio volatility
            # port_vol = np.sqrt(w @ cov @ w) # or np.sqrt(w.dot(cov.dot(w)))
            # return -weighted_vols / (port_vol + 1e-10)

        # constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
        # bounds      = tuple((0.0, 1.0) for _ in range(self.n_assets))
        # w0          = np.ones(self.n_assets) / self.n_assets

        # result = minimize(
            # neg_diversification_ratio, w0,
            # method='SLSQP', bounds=bounds, constraints=constraints,
            # options={'ftol': 1e-12, 'maxiter': 1000}
        # )

        # weights = result.x

        if w_value is not None:
            # Normalize the solution to get actual portfolio weights
            weights = w_value / np.sum(w_value)
            weights[weights < 1e-8] = 0  # zero out tiny weights
            weights /= np.sum(weights)   # re-normalize
            return weights
        else:
            return None

    # 4. Equal Risk Contribution (ERC) with CVaR

    def cvar_risk_parity_optimization(self, alpha=0.05):
        """
        Equal Risk Contribution (ERC) under Conditional Value at Risk.

        Allocates weights so that every asset contributes equally to the portfolio's
        CVaR (expected shortfall) at level alpha, instead of to its variance as in the
        classical ERC / risk-parity portfolio. This is risk parity on tail risk.

        Note: assumes the portfolio CVaR is positive (the usual case for loss-bearing
        return data); if CVaR can turn negative the log-barrier program is unbounded.

        Parameters
        ----------
        alpha : float
            Tail probability for CVaR (e.g. 0.05 for the 95% expected shortfall).
        """

        w = cp.Variable(self.n_assets)
        VaR = cp.Variable()
        loss = cp.Variable(self.n_periods)

        # Rockafellar-Uryasev empirical CVaR of the portfolio loss (-returns @ w)
        constraints = [
            w >= 1e-5,  # strictly positive so the log barrier is well defined
            loss >= -self.returns.values @ w - VaR,
            loss >= 0
        ]
        CVaR = VaR + (1.0 / (alpha * self.n_periods)) * cp.sum(loss)

        # Risk-budgeting objective: equal budgets via the -sum(log w) barrier
        objective = cp.Minimize(CVaR - cp.sum(cp.log(w)))

        prob = cp.Problem(objective, constraints)
        prob.solve()

        w_value = w.value if prob.status == 'optimal' else None

        if w_value is not None:
            # Normalize the risk-budgeting solution to actual portfolio weights
            weights = w_value / np.sum(w_value)
            weights[weights < 1e-8] = 0  # zero out tiny weights
            weights /= np.sum(weights)   # re-normalize
            return weights
        else:
            return None

    # 5. Equal Risk Contribution (ERC) with CDaR

    def cdar_risk_parity_optimization(self, alpha=0.05):
        """
        Equal Risk Contribution (ERC) under Conditional Drawdown-at-Risk.

        Allocates weights so that every asset contributes equally to the portfolio's
        CDaR (expected tail drawdown) at level alpha, instead of to its variance as in the
        classical ERC / risk-parity portfolio. This is risk parity on tail risk.

        Note: assumes the portfolio CDaR is positive (the usual case for loss-bearing
        return data); if it can turn negative the log-barrier program is unbounded.

        Parameters
        ----------
        alpha : float
            Tail probability for the drawdown (e.g. 0.05 -> average of the worst 5% drawdowns)
        """
        T = self.n_periods

        # Uncompounded cumulative return path is affine in w (see cdar_optimization)
        cum_returns = np.tril(np.ones((T, T))) @ self.returns.values  # (T, N)

        w = cp.Variable(self.n_assets)
        u = cp.Variable(T)    # running peak (high-water mark) of the cumulative path
        z = cp.Variable(T)    # drawdown exceedances over the threshold zeta
        zeta = cp.Variable()  # Drawdown-at-Risk threshold (DaR)

        cumulative = cum_returns @ w

        constraints = [
            w >= 1e-5,                  # strictly positive so the log barrier is well defined
            u >= cumulative,            # peak is at least the current cumulative value
            u[1:] >= u[:-1],            # peak is non-decreasing (running maximum)
            u >= 0,                     # peak measured from initial capital (drawdown from 0)
            z >= u - cumulative - zeta, # exceedance of drawdown D_t = u - cumulative over zeta
            z >= 0
        ]

        # Rockafellar-Uryasev representation of CDaR: minimize over zeta the sum of zeta and the average exceedance
        CDaR = zeta + (1.0 / (alpha * T)) * cp.sum(z)

        # Risk-budgeting objective: equal budgets via the -sum(log w) barrier
        objective = cp.Minimize(CDaR - cp.sum(cp.log(w)))

        prob = cp.Problem(objective, constraints)
        prob.solve()

        w_value = w.value if prob.status == 'optimal' else None

        if w_value is not None:
            # Normalize the risk-budgeting solution to actual portfolio weights
            weights = w_value / np.sum(w_value)
            weights[weights < 1e-8] = 0  # zero out tiny weights
            weights /= np.sum(weights)   # re-normalize
            return weights
        else:
            return None
        
    # ========================================================================
    # 9. GROWTH-OPTIMAL (KELLY) PORTFOLIOS
    # ========================================================================

    def kelly_optimization(self, fraction=1.0, return_scale=1.0, baseline=None):
        """
        Kelly / growth-optimal portfolio (log-utility), with optional fractional Kelly.

        Maximises the expected log-growth rate of wealth (the geometric growth rate),
        estimated empirically over the sample. The full Kelly bet maximises long-run growth, 
        but is aggressive and very sensitive to estimation error (it overbets when mu is overestimated).

        Fractional Kelly (fraction < 1) addresses this by blending the Kelly weights with a
        conservative anchor. Since the assumption is a fully-invested and long-only portfolio, 
        we blend toward a low-risk anchor (minimum-variance by default):
            w = fraction * w_kelly + (1 - fraction) * w_anchor

        Note: requires 1 + (R_t * w) / return_scale > 0 for all t (true for diversified
        long-only equity returns). Otherwise the log is undefined and the solve fails.

        Parameters
        ----------
        fraction : float
            Kelly fraction in (0, 1]. 1.0 = full Kelly; 0.5 = half Kelly.
        return_scale : float
            Divisor converting the stored returns to decimals before forming gross
            returns 1 + r. Use 1.0 for decimal returns, 100.0 if returns are in percent.
        baseline : np.ndarray, optional
            Conservative anchor weights for fractional Kelly (defaults to minimum variance).
        """
        R = self.returns.values / return_scale

        w = cp.Variable(self.n_assets)

        growth = cp.sum(cp.log(1 + R @ w)) / self.n_periods
        constraints = [cp.sum(w) == 1, w >= 0, w <= 1]

        prob = cp.Problem(cp.Maximize(growth), constraints)
        prob.solve()

        w_kelly = w.value if prob.status == 'optimal' else None

        if fraction >= 1.0:
            return w_kelly

        # Fractional Kelly: blend toward a conservative anchor
        w_anchor = baseline if baseline is not None else self.min_variance()
        if w_anchor is None:
            return w_kelly
        
        weights = fraction * w_kelly + (1.0 - fraction) * w_anchor
        weights = np.clip(weights, 0, None)
        weights /= np.sum(weights)

        return weights

    def worst_case_kelly_optimization(self, alpha=0.1, return_scale=1.0):
        """
        Worst-Case (distributionally robust) Kelly portfolio (Sun & Boyd, 2018).

        Instead of the average log-growth, maximises the worst-case average log-growth over the 
        fraction alpha of least favourable periods, the lower-tail CVaR of the per-period log-growth. 
        As alpha -> 1 it recovers full Kelly, smaller alpha is more conservative.

        Parameters
        ----------
        alpha : float
            Tail fraction of worst-growth periods to protect (e.g. 0.1 -> worst 10%).
        return_scale : float
            Divisor converting stored returns to decimals (1.0 decimal, 100.0 percent).
        """
        T = self.n_periods
        R = self.returns.values / return_scale

        w = cp.Variable(self.n_assets)
        eta = cp.Variable()

        g = cp.log(1 + R @ w)  # per-period log-growth (concave in w)
        # Lower-tail CVaR of the log-growth = average growth in the worst alpha-fraction
        lower_tail_growth = eta - (1.0 / (alpha * T)) * cp.sum(cp.pos(eta - g))

        constraints = [cp.sum(w) == 1, w >= 0, w <= 1]
        prob = cp.Problem(cp.Maximize(lower_tail_growth), constraints)
        prob.solve()

        weights = w.value if prob.status == 'optimal' else None

        return weights

    # ========================================================================
    # 10. BENCHMARKS
    # ========================================================================

    # 1. 1/N Equal Weight Portfolio

    def equal_weight_portfolio(self):
        """
        Naive 1/N diversification (Equal Weight Portfolio)
        """
        n = self.n_assets
        weights = np.ones(n) / n
        
        return weights
    