
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

        self.returns = returns_data.values
        self.asset_names = returns_data.columns.tolist()
        self.assets = returns_data.columns
        
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
        portfolio_volatility = np.sqrt(weights.T @ self.Sigma @ weights)
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
    
    def _solve_mean_variance_sample(self, mu, sigma, target_return):
        """Helper function for mean-variance optimization"""
        w = cp.Variable(self.n_assets)
        if target_return is not None:
            constraints = [cp.sum(w) == 1, w >= 0, w <= 1, w.T @ mu >= target_return]
            objective = cp.Minimize(cp.quad_form(w, sigma))
        else:
            constraints = [cp.sum(w) == 1, w >= 0, w <= 1]
            objective = cp.Minimize(-w.T @ self.mu + risk_aversion * cp.quad_form(w, sigma))
        
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        return w.value if prob.status == 'optimal' else None
    
    def _solve_min_variance_sample(self, sigma):
        """Helper function for minimum variance optimization"""
        w = cp.Variable(self.n_assets)
        constraints = [cp.sum(w) == 1, w >= 0, w <= 1]
        objective = cp.Minimize(cp.quad_form(w, sigma))
        
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

    def wasserstein_optimization(self, epsilon=0.1, norm_type=2):
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
        portfolio_std = cp.sqrt(portfolio_var)

        # Worst-case adjustment (simplified Wasserstein penalty)
        # Based on: E[R] - epsilon * ||grad E[R]||
        if norm_type == 2:
            wasserstein_penalty = epsilon * portfolio_std * np.sqrt(self.n_assets)
            # wasserstein_penalty = epsilon * cp.norm(w, 2)
        elif norm_type == 1:
            wasserstein_penalty = epsilon * cp.norm(w, 1)
        else:
            raise ValueError("Unsupported norm type")

        robust_return = portfolio_returns - wasserstein_penalty

        # Objective: maximize risk-adjusted return
        objective = cp.Minimize(-(robust_return - self.rf) / portfolio_std)
        # objective = cp.Minimize(-(robust_return - self.rf) + risk_aversion * cp.quad_form(w, self.cov))

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
        mu_uncertainty = kappa_mu * cp.norm(cp.sqrtm(self.cov) @ w)
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
            sample_returns = self.returns[indices]
            # sample_returns = self.returns.iloc[sample_idx]
            
            # Estimate parameters from resampled data
            mu_sample = np.mean(sample_returns, axis=0)
            cov_sample = np.cov(sample_returns.T)
            
            # Optimize for this sample
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
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

        weights = self._solve_mean_variance_sample(self.mu, cov_lw, target_return=None)
        
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
            loss >= -self.returns @ w - VaR,
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
        """
        
        w = cp.Variable(self.n_assets)
        VaR = cp.Variable()
        loss = cp.Variable(self.n_periods)
        
        # CVaR constraints
        constraints = [
            cp.sum(w) == 1, w >= 0, w <= 1,
            loss >= -self.returns @ w - VaR,
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
        expected_return = self.mu @ w - epsilon * cp.norm(w, 2)  # Worst-case return
        
        objective = cp.Minimize(-expected_return + risk_aversion * robust_CVaR)
        
        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = w.value if prob.status == 'optimal' else None
        
        return weights 

    # ========================================================================
    # 7. ELASTIC NET REGULARIZATION
    # ========================================================================
    
    def elastic_net_optimization(self, lambda_l1=0.01, lambda_l2=0.01):
        """
        Portfolio optimization with Elastic Net regularization
        Combines L1 (sparsity) and L2 (diversification) penalties
        
        Parameters:
        -----------
        lambda_l1 : float
            L1 regularization parameter (encourages sparsity)
        lambda_l2 : float
            L2 regularization parameter (penalizes large positions)
        risk_free_rate : float
            Risk-free rate
        """

        w = cp.Variable(self.n_assets)

        # Portfolio return and risk 
        portfolio_return = np.dot(w, self.mu) 
        portfolio_var = np.dot(w, np.dot(self.cov, w)) 
        
        # Sharpe ratio component
        sharpe_component = -(portfolio_return - self.rf) / np.sqrt(portfolio_var)
        
        # Elastic Net penalty: lambda1 * ||w||_1 + lambda2 * ||w||_2^2
        l1_penalty = lambda_l1 * np.sum(np.abs(w))
        #l2_penalty = lambda_l2 * np.sum(w**2)
        l2_penalty = lambda_l2 * cp.norm(w, 2)**2

        sharpe_penalized = sharpe_component + l1_penalty + l2_penalty

        objective = cp.Minimize(sharpe_penalized)
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

optimizer = PortfolioOptimizer(returns_df)

def compare_all_methods():
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

    return comparison_df


# ======= EXAMPLE USAGE ======================================================

if __name__ == "__main__":
    # Generate synthetic returns data
    np.random.seed(42)
    n_periods = 252  # 1 year of daily returns
    n_assets = 5
    
    # Simulate returns with some correlation structure
    true_mu = np.array([0.0008, 0.0006, 0.0007, 0.0005, 0.0009])
    true_cov = np.array([
        [0.0004, 0.0002, 0.0001, 0.0001, 0.0002],
        [0.0002, 0.0003, 0.0001, 0.0001, 0.0001],
        [0.0001, 0.0001, 0.0002, 0.0001, 0.0001],
        [0.0001, 0.0001, 0.0001, 0.0002, 0.0001],
        [0.0002, 0.0001, 0.0001, 0.0001, 0.0005]
    ])
    
    returns = np.random.multivariate_normal(true_mu, true_cov, n_periods)
    
    # Create DataFrame
    asset_names = ['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer']
    returns_df = pd.DataFrame(returns, columns=asset_names)
    
    print("=" * 80)
    print("ROBUST PORTFOLIO OPTIMIZATION SUITE")
    print("=" * 80)
    print(f"\nDataset: {n_periods} periods, {n_assets} assets")
    print(f"Assets: {asset_names}")
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(returns_df)
    
    # Compare all methods
    print("\n" + "=" * 80)
    print("COMPARING ALL OPTIMIZATION METHODS")
    print("=" * 80)
    
    comparison = optimizer.compare_all_methods(risk_free_rate=0.0001)
    
    print("\n" + comparison.to_string(index=False))
    
    # Display weights
    print("\n" + "=" * 80)
    print("PORTFOLIO WEIGHTS BY METHOD")
    print("=" * 80)
    
    weights_df = optimizer.get_weights_dataframe()
    print("\n" + weights_df.to_string())
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Observations:")
    print("- Classical Markowitz may show concentrated positions")
    print("- Robust methods generally produce more diversified portfolios")
    print("- Elastic Net creates sparse portfolios (fewer active positions)")
    print("- Resampling and shrinkage methods balance performance and stability")