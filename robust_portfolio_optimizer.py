
"""
Robust Portfolio Optimization Suite
Implements multiple approaches to robust portfolio optimization including:
1. Classical Markowitz optimization
2. Worst-case optimization with ellipsoidal uncertainty sets
3. Black-Litterman (Bayesian approach)
4. Resampling methods
5. Robust covariance estimation (Ledoit-Wolf and Factor Models)
6. Distributional robustness with Wasserstein distance
7. Regularization with L1, L2, or as Elastic Net
"""

import numpy as np
import pandas as pd
import cvxpy as cp
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """
    A comprehensive portfolio optimization class with multiple robust methods
    """
    
    def __init__(self, returns_data, risk_free_rate=0.0):
        """
        Initialize the optimizer with historical returns data
        
        Parameters:
        -----------
        returns_data : pd.DataFrame or np.ndarray
            Historical returns (T x N) where T is time periods and N is assets
        risk_free_rate : float
            Risk-free rate for Sharpe ratio calculations
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
        
        # Calculate basic statistics
        self.mu = np.mean(self.returns, axis=0)
        self.cov = np.cov(self.returns.T)

    def calculate_portfolio_stats(self, weights):
        """Calculate portfolio statistics"""
        if weights is None:
            return None
            
        portfolio_return = self.mu @ weights
        portfolio_volatility = np.sqrt(weights.T @ self.cov @ weights)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'weights': dict(zip(self.assets, weights)),
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
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
            objective = cp.Minimize(-w.T @ self.mu + risk_aversion * cp.quad_form(w, cov))
        
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

    def wasserstein_optimization(self, epsilon=0.1, kappa=1.0, norm_type=2, risk_aversion=1.0):
        """
        Distributionally robust optimization using Wasserstein distance
        
        Parameters:
        -----------
        epsilon : float
            Wasserstein ball radius (robustness parameter)
        norm_type : int
            Norm type for Wasserstein distance (1 or 2)
        """

        # Direct Wasserstein robust optimization

        w = cp.Variable(self.n_assets)

        # Portfolio return and variance
        portfolio_returns = w.T @ self.mu
        portfolio_var = cp.quad_form(w, self.cov)
        # portfolio_std = cp.sqrt(portfolio_var)

        # Worst-case adjustment (simplified Wasserstein penalty)
        # Based on: E[R] - epsilon * ||grad E[R]||
        if norm_type == 2:
            # wasserstein_penalty = epsilon * portfolio_var * np.sqrt(self.n_assets)
            # wasserstein_penalty = epsilon * np.sqrt(self.n_assets / self.n_periods)
            wasserstein_penalty = epsilon * cp.norm(w, 2)
        elif norm_type == 1:
            wasserstein_penalty = epsilon * cp.norm(w, 1)
        else:
            raise ValueError("Unsupported norm type")
        
        # robust_return = portfolio_returns - wasserstein_penalty * np.sqrt(portfolio_var)
        robust_return = portfolio_returns - wasserstein_penalty

        # Worst-case variance (simplified conservative approximation)
        worst_case_variance = portfolio_var + kappa * epsilon * cp.norm(w, 2)**2

        # Objective: maximize risk-adjusted return
        # objective = cp.Minimize(-(robust_return - self.rf) / portfolio_var)
        # objective = cp.Minimize(-(robust_return - self.rf) + (risk_aversion / 2) * portfolio_var)
        # objective = cp.Minimize(-(robust_return - self.rf) + risk_aversion * cp.quad_form(w, self.cov))¨
        objective = cp.Minimize(-(robust_return - self.rf) + risk_aversion * worst_case_variance)

        constraints = [cp.sum(w) == 1, w >= 0, w <= 1]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = w.value if prob.status == 'optimal' else None

        return weights
    
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
        
        # result = minimize(robust_objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints) 
        # weights = result.x

        # Using cvxpy for robust optimization
        w = cp.Variable(self.n_assets)

        # Uncertainty sets
        mu_uncertainty = kappa_mu * cp.norm(cp.sqrt(self.cov) @ w)
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

    def black_litterman(self, market_caps=None, tau=0.05, risk_aversion=2.5, views=None, view_confidences=None):
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
            Q = np.zeros(len(views))  # View returns (example: all zeros)
            
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

        weights = self._solve_mean_variance_sample(mu_bl, cov_bl, target_return=None)
        
        # ret, vol = self._portfolio_performance(weights, mu_bl, cov_bl)
        # return {'weights': weights, 'return': ret, 'volatility': vol, 
        #        'sharpe': (ret - self.rf) / vol, 'posterior_returns': mu_bl, 'method': 'Black-Litterman'}
        
        return weights
    
    # ========================================================================
    # 4. RESAMPLING
    # ========================================================================
    
    def resampling_optimization(self, n_samples=1000, target_return=None, risk_aversion=1.0): 
                                # seed=42):
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
        seed : int
            Random seed for reproducibility
        """
        # np.random.seed(seed)
        
        # Store weights from each resampled optimization
        resampled_weights = []

        for _ in range(n_samples):
            # Resample returns (bootstrap)
            indices = np.random.choice(self.n_periods, size=self.n_periods, replace=True)
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

        weights = self._solve_mean_variance_sample(self.mu, cov_lw, risk_aversion=risk_aversion, target_return=None)
        
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
                model = LinearRegression().fit(factor_returns, self.returns[:, i], fit_intercept=True)
                factor_loadings[i, :] = model.coef_
                residuals = self.returns[:, i] - model.predict(factor_returns)
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
        # return {'weights': weights, 'return': ret, 'volatility': vol, 'sharpe': (ret - self.rf) / vol, 
        #         'n_factors': k, 'method': 'Factor Model Covariance'}
    
        return weights
    
    # ========================================================================
    # 6. CVAR OPTIMIAZATION
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
        expected_return = self.mu.values @ w
        
        objective = cp.Minimize(-expected_return + risk_aversion * CVaR)
        
        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = w.value if prob.status == 'optimal' else None
        
        return weights

    def wasserstein_cvar_optimization(self, epsilon=0.1, alpha=0.05, risk_aversion=1.0):
        """
        Wasserstein robust CVaR optimization
        Combines distributional robustness with tail risk management
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
        
        # Wasserstein robust CVaR (conservative approximation)
        # Add robustness margin based on epsilon and portfolio norm
        if hasattr(cp, 'norm'):
            robustness_margin = epsilon * cp.norm(w, 2) / alpha
        else:
            # robustness_margin = epsilon * cp.sqrt(cp.sum_squares(w)) / alpha
            robustness_margin = epsilon * cp.sqrt(cp.quad_form(w, self.cov)) * np.sqrt(self.n_assets) / alpha
            
        robust_CVaR = CVaR + robustness_margin
        expected_return = self.mu.values @ w - epsilon * cp.norm(w, 2)  # Worst-case return
        
        objective = cp.Minimize(-expected_return + risk_aversion * robust_CVaR)
        
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
        portfolio_return = self.mu.values @ w
        portfolio_var = cp.quad_form(w, self.cov)
        
        # Sharpe ratio component
        # sharpe_component = -(portfolio_return - self.rf) / np.sqrt(portfolio_var)

        # Mean-variance utility (avoid sqrt for convexity)
        # Using quadratic utility: return - (1/2) * risk_aversion * variance
        utility = portfolio_return - (risk_aversion / 2) * portfolio_var
        
        # Elastic Net penalty: lambda1 * ||w||_1 + lambda2 * ||w||_2^2
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
    

# ======= UTILITY FUNCTIONS FOR COMPARISON AND OUTPUT ========================

# Download or generate returns data

# ticker = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V',
#          'WMT', 'PG', 'UNH', 'DIS', 'NVDA', 'HD', 'MA', 'PYPL', 'BAC', 'VZ']
ticker = ['^GSPC', '^IXIC', '^DJI', '^GDAXI', '^FTSE', '^FCHI', '^HSI', '^AXJO', 
          '^BSESN', '^TWII', '^MXX', '^KS11', '^N225', '^BVSP', '^STI']

def download_fin_data(ticker, start_date = "1985-01-01",
                      end_date = "2025-09-30"):
    
    asset_df = pd.DataFrame()
    assets = pd.DataFrame()

    # Define the stock symbol and loop over symbols

    for symbol in ticker:
        
        # Download historical data
        
        print("Ticker: "+symbol)
        
        asset_data = yf.download(symbol, start=start_date, end=end_date)
        asset_data = asset_data.stack(1)
        asset_data = asset_data.reset_index(level=1)

        asset_data['month_id'] = asset_data.index.strftime('%Y-%m')
        asset_data['numst'] = asset_data.groupby(['month_id'])['Ticker'].transform('count')
        asset_data = asset_data[(asset_data['numst']>=17)]

        data_at = asset_data.groupby(['month_id']).last().reset_index()
        asset_df = pd.concat([asset_df, data_at], axis=0)

        # Load historical data

        asset = yf.Ticker(symbol)
        try:
            data = asset.history(period="max")
        except:
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
        sdf = sdf[(sdf['numst']>=17)]

        sdf = sdf[['month_id', 'Ticker', 'Close', 'Volume', 'Div', 'ret']]

        assets = pd.concat([assets, sdf], axis=0)
        
        #del sdf
        #gc.collect()

    return assets, asset_df

# Compare all methods

def compare_all_methods(optimizer):
    """
    Run all optimization methods and compare results.
    
    Returns:
    --------
    pd.DataFrame : Comparison of all methods
    """
    # results = []
    results = {}
    
    # 1. Markowitz
    try:
        weights_mv = optimizer.mean_variance_optimization()
        stats_mv = optimizer.calculate_portfolio_stats(weights_mv)
        results["Classical Mean Variance"] = stats_mv
        #results.append(res)
    except Exception as e:
        print(f"Mean-Variance failed: {e}")

    # 1. Min Variance
    try:
        weights_minvar = optimizer.min_variance()
        stats_minvar = optimizer.calculate_portfolio_stats(weights_minvar)
        results["Minimum Variance"] = stats_minvar
        #results.append(res)
    except Exception as e:
        print(f"Minimum Variance failed: {e}")

    # 3. Wasserstein
    try:
        weights_wasserstein = optimizer.wasserstein_optimization()
        stats_wasserstein = optimizer.calculate_portfolio_stats(weights_wasserstein)
        results["Wasserstein"] = stats_wasserstein
        # results.append(res)
    except Exception as e:
        print(f"Wasserstein failed: {e}")

    # 4. Worst-case Ellipsoidal
    try:
        weights_worst_case = optimizer.ellipsoidal_uncertainty_optimization()
        stats_worst_case = optimizer.calculate_portfolio_stats(weights_worst_case)
        results["Worst-case Ellipsoidal"] = stats_worst_case
        # results.append(res)
    except Exception as e:
        print(f"Worst-case failed: {e}")
    
    # 5. Black-Litterman
    try:
        weights_black_litterman = optimizer.black_litterman()
        stats_black_litterman = optimizer.calculate_portfolio_stats(weights_black_litterman)
        results["Black-Litterman"] = stats_black_litterman
        # results.append(res)
    except Exception as e:
        print(f"Black-Litterman failed: {e}")
    
    # 6. Resampled
    try:
        weights_resampled = optimizer.resampling_optimization()
        stats_resampled = optimizer.calculate_portfolio_stats(weights_resampled)
        results["Resampled"] = stats_resampled
        # results.append(res)
    except Exception as e:
        print(f"Resampled failed: {e}")
    
    # 7. Ledoit-Wolf Shrinkage
    try:
        weights_ledoit_wolf = optimizer.shrinkage_covariance_optimization()
        stats_ledoit_wolf = optimizer.calculate_portfolio_stats(weights_ledoit_wolf)
        results["Ledoit-Wolf Shrinkage"] = stats_ledoit_wolf
        # results.append(res)
    except Exception as e:
        print(f"Ledoit-Wolf failed: {e}")
    
    # 8. Factor Model
    try:
        weights_factor_model = optimizer.factor_model_optimization()
        stats_factor_model = optimizer.calculate_portfolio_stats(weights_factor_model)
        results["Factor Model"] = stats_factor_model
        # results.append(res)
    except Exception as e:
        print(f"Factor Model failed: {e}")

    # 9. CVaR Optimization
    try:    
        weights_cvar = optimizer.cvar_optimization()
        stats_cvar = optimizer.calculate_portfolio_stats(weights_cvar)
        results["CVaR Optimization"] = stats_cvar
        # results.append(res)
    except Exception as e:
        print(f"CVaR Optimization failed: {e}")
    
    # 10. Wasserstein CVaR
    try:
        weights_wasserstein_cvar = optimizer.wasserstein_cvar_optimization()
        stats_wasserstein_cvar = optimizer.calculate_portfolio_stats(weights_wasserstein_cvar)
        results["Wasserstein CVaR"] = stats_wasserstein_cvar
        # results.append(res)    
    except Exception as e:
        print(f"Wasserstein CVaR failed: {e}")
    
    # 11. Elastic Net
    try:
        weights_elastic_net = optimizer.elastic_net_optimization()
        stats_elastic_net = optimizer.calculate_portfolio_stats(weights_elastic_net)
        results["Elastic Net"] = stats_elastic_net
    except Exception as e:
        print(f"Elastic Net failed: {e}")
    
    # Create comparison DataFrame
    # comparison_data = []
    # for res in results:
        # comparison_data.append({
        # 'Method': res['method'], 'Return': res['return'], 'Volatility': res['volatility'],
        # 'Sharpe': res['sharpe'], 'Max Weight': np.max(res['weights']), 
        # 'Min Weight': np.min(res['weights'][res['weights'] > 1e-4]), 'N Assets': np.sum(res['weights'] > 1e-4)})
    #comparison_df = pd.DataFrame(comparison_data)
    
    comparison_df = pd.DataFrame.from_dict(results, orient='index')
    comparison_df = comparison_df.reset_index().rename(columns={'index': 'Method'})

    return results, comparison_df


# ======= EXAMPLE USAGE ======================================================

if __name__ == "__main__":
    
    # Generate synthetic returns data
    # np.random.seed(42)
    # n_periods = 252  # 1 year of daily returns
    # n_assets = 5
    # Simulate returns with some correlation structure
    # true_mu = np.array([0.0008, 0.0006, 0.0007, 0.0005, 0.0009])
    # true_cov = np.array([
    #     [0.0004, 0.0002, 0.0001, 0.0001, 0.0002],
    #     [0.0002, 0.0003, 0.0001, 0.0001, 0.0001],
    #     [0.0001, 0.0001, 0.0002, 0.0001, 0.0001],
    #     [0.0001, 0.0001, 0.0001, 0.0002, 0.0001],
    #     [0.0002, 0.0001, 0.0001, 0.0001, 0.0005]
    # ])
    # returns = np.random.multivariate_normal(true_mu, true_cov, n_periods)
    # Create DataFrame
    # asset_names = ['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer']
    # returns_df = pd.DataFrame(returns, columns=asset_names)
    
    print("=" * 80)
    print("ROBUST PORTFOLIO OPTIMIZATION SUITE")
    print("=" * 80)

    # Download financial data
    assets, asset_df = download_fin_data(ticker)
    assets = assets.sort_values(by=['Ticker', 'month_id']).reset_index(drop=True)
    # Reshaping data to (time_steps, n_assets)
    returns_df = assets.pivot(index='month_id', columns='Ticker', values='ret').reset_index()

    # Prepare returns DataFrame
    returns_df = returns_df.drop(columns=['month_id'])
    # Only asset data with index larger 839
    returns_df = returns_df[returns_df.index > 839]
    returns_df = returns_df.fillna(0.0)  # Fill missing values with 0.0

    returns_df = returns_df*100  # Convert to percentage returns
        
    print(f"\nDataset: {len(assets['month_id'].unique())} periods, {len(assets['Ticker'].unique())} assets")
    # print(f"Assets: {asset_names}")

    # Initialize optimizer
    optimizer = PortfolioOptimizer(returns_df)
    
    # Compare all methods
    print("\n" + "=" * 80)
    print("COMPARING ALL OPTIMIZATION METHODS")
    print("=" * 80)
    
    results, comparison = compare_all_methods(optimizer)
    
    # print("\n" + comparison.to_string(index=False))

    # Display detailed results
    print("\n" + "=" * 60)
    print("DETAILED RESULTS SUMMARY")
    print("=" * 60)
    
    for method, stats in results.items():
        print(f"\n{method}:")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']*np.sqrt(12):.4f}")
        print(f"  Expected Return (annual): {stats['expected_return']:.4f}")
        print(f"  Volatility (annual): {stats['volatility']:.4f}")
        # print("  Weights:")
        # for asset, weight in stats['weights'].items():
        #    if weight > 0.01:  # Only show weights > 1%
        #        print(f"    {asset}: {weight:.3f}")
    
    # Display weights
    weights_df = pd.DataFrame()
    for method, stats in results.items():
        weights_df[method] = pd.Series(stats['weights'])
    
    # Display only methods that have weights
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
    