
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
    
    # 1. CLASSICAL MARKOWITZ OPTIMIZATION
    
    def mean_variance_optimization(self, target_return=None, risk_aversion=1.0):
        """
        Classical Markowitz mean-variance optimization
        
        Returns:
        # dict : Optimization results including weights and performance
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
    
    # 2. WORST-CASE OPTIMIZATION WITH ELLIPSOIDAL UNCERTAINTY

    def ellipsoidal_uncertainty(self, kappa=2):
        """
        Robust optimization with ellipsoidal uncertainty set for expected returns.
        
        Parameters:
        kappa : float
            Size of uncertainty set (higher = more conservative)
    
        Returns:
        dict : Optimization results
        """
        # Standard error of mean estimates
        std_error = np.sqrt(np.diag(self.cov) / self.n_periods)
        
        def robust_objective(w):
            """Minimize worst-case Sharpe ratio."""
            # Nominal return
            nominal_return = np.dot(w, self.mu)
            
            # Worst-case adjustment (robust counterpart)
            portfolio_vol = np.sqrt(np.dot(w, np.dot(self.cov, w)))
            uncertainty_penalty = kappa * np.sqrt(np.dot(w**2, std_error**2))
            
            worst_case_return = nominal_return - uncertainty_penalty
            
            # Negative Sharpe for minimization
            return -(worst_case_return - self.rf) / (portfolio_vol + 1e-8)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}] # Sum of weights = 1
        bounds = tuple((0, 1) for _ in range(self.n_assets)) # No short selling
        init_weights = np.ones(self.n_assets) / self.n_assets # Equal weights
        
        result = minimize(
            robust_objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        ret, vol = self._portfolio_performance(weights, self.mu, self.cov)
        
        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe': (ret - self.rf) / vol,
            'method': 'Ellipsoidal Uncertainty Set'
        }
    
    # ========================================================================
    # 3. BLACK-LITTERMAN MODEL
    # ========================================================================
    
    def black_litterman(self, market_caps=None, tau=0.05, risk_aversion=2.5,
                       views=None, view_confidences=None, risk_free_rate=0.0):
        """
        Black-Litterman model combining market equilibrium with investor views.
        
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
            Confidence in views (K x K diagonal matrix or vector)
        risk_free_rate : float
            Risk-free rate
            
        Returns:
        --------
        dict : Optimization results
        """
        # If no market caps provided, use equal weights
        if market_caps is None:
            w_mkt = np.ones(self.n_assets) / self.n_assets
        else:
            w_mkt = market_caps / np.sum(market_caps)
        
        # Implied equilibrium returns (reverse optimization)
        pi = risk_aversion * np.dot(self.cov, w_mkt)
        
        # If no views provided, use equilibrium
        if views is None:
            mu_bl = pi
            cov_bl = self.cov
        else:
            P = views  # View matrix
            Q = np.zeros(len(views))  # View returns (example: all zeros)
            
            # Omega: diagonal matrix of view uncertainties
            if view_confidences is None:
                # Default: proportional to variance of views
                Omega = np.diag(np.diag(P @ self.cov @ P.T)) * tau
            else:
                if isinstance(view_confidences, np.ndarray) and view_confidences.ndim == 1:
                    Omega = np.diag(view_confidences)
                else:
                    Omega = view_confidences
            
            # Black-Litterman formula
            tau_cov = tau * self.cov
            
            # Posterior mean
            M = np.linalg.inv(np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(Omega) @ P)
            mu_bl = M @ (np.linalg.inv(tau_cov) @ pi + P.T @ np.linalg.inv(Omega) @ Q)
            
            # Posterior covariance
            cov_bl = self.cov + M
        
        # Optimize with Black-Litterman parameters
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        init_weights = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            self._neg_sharpe,
            init_weights,
            args=(mu_bl, cov_bl, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        ret, vol = self._portfolio_performance(weights, mu_bl, cov_bl)
        
        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe': (ret - risk_free_rate) / vol,
            'posterior_returns': mu_bl,
            'method': 'Black-Litterman'
        }
    
    # ========================================================================
    # 4. RESAMPLED EFFICIENCY (MICHAUD)
    # ========================================================================
    
    def resampled_optimization(self, n_samples=100, target_return=None, 
                              risk_free_rate=0.0, seed=42):
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
            
        Returns:
        --------
        dict : Optimization results
        """
        np.random.seed(seed)
        
        # Store weights from each resampled optimization
        resampled_weights = []
        
        for i in range(n_samples):
            # Resample returns (bootstrap)
            indices = np.random.choice(self.n_periods, size=self.n_periods, replace=True)
            sample_returns = self.returns[indices]
            
            # Estimate parameters from resampled data
            mu_sample = np.mean(sample_returns, axis=0)
            cov_sample = np.cov(sample_returns.T)
            
            # Optimize for this sample
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            
            if target_return is not None:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda w: np.dot(w, mu_sample) - target_return
                })
            
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            init_weights = np.ones(self.n_assets) / self.n_assets
            
            try:
                if target_return is None:
                    result = minimize(
                        self._neg_sharpe,
                        init_weights,
                        args=(mu_sample, cov_sample, risk_free_rate),
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints
                    )
                else:
                    result = minimize(
                        lambda w: np.dot(w, np.dot(cov_sample, w)),
                        init_weights,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints
                    )
                
                if result.success:
                    resampled_weights.append(result.x)
            except:
                continue
        
        # Average the resampled weights
        weights = np.mean(resampled_weights, axis=0)
        weights = weights / np.sum(weights)  # Normalize
        
        ret, vol = self._portfolio_performance(weights, self.mu, self.cov)
        
        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe': (ret - risk_free_rate) / vol,
            'n_successful_samples': len(resampled_weights),
            'method': 'Resampled Efficiency'
        }
    
    # ========================================================================
    # 5. ROBUST COVARIANCE ESTIMATION
    # ========================================================================
    
    def ledoit_wolf_optimization(self, risk_free_rate=0.0):
        """
        Optimization using Ledoit-Wolf shrinkage covariance estimator.
        
        Parameters:
        -----------
        risk_free_rate : float
            Risk-free rate
            
        Returns:
        --------
        dict : Optimization results
        """
        # Ledoit-Wolf shrinkage
        cov_lw = self._ledoit_wolf_shrinkage(self.returns)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        init_weights = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            self._neg_sharpe,
            init_weights,
            args=(self.mu, cov_lw, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        ret, vol = self._portfolio_performance(weights, self.mu, cov_lw)
        
        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe': (ret - risk_free_rate) / vol,
            'shrinkage_covariance': cov_lw,
            'method': 'Ledoit-Wolf Covariance'
        }
    
    def _ledoit_wolf_shrinkage(self, returns):
        """
        Compute Ledoit-Wolf shrinkage estimator of covariance matrix.
        """
        T, N = returns.shape
        
        # Sample covariance
        sample_cov = np.cov(returns.T)
        
        # Shrinkage target: constant correlation model
        var = np.diag(sample_cov)
        sqrt_var = np.sqrt(var)
        sample_cor = sample_cov / np.outer(sqrt_var, sqrt_var)
        avg_cor = (np.sum(sample_cor) - N) / (N * (N - 1))
        
        prior = avg_cor * np.outer(sqrt_var, sqrt_var)
        np.fill_diagonal(prior, var)
        
        # Optimal shrinkage intensity
        # Simplified calculation
        shrinkage = min(1.0, max(0.0, (N + 1) / (T * (N + 1) + 2)))
        
        # Shrunk covariance
        cov_shrunk = shrinkage * prior + (1 - shrinkage) * sample_cov
        
        return cov_shrunk
    
    def factor_model_optimization(self, n_factors=3, risk_free_rate=0.0):
        """
        Optimization using factor model for covariance estimation.
        
        Parameters:
        -----------
        n_factors : int
            Number of factors to use
        risk_free_rate : float
            Risk-free rate
            
        Returns:
        --------
        dict : Optimization results
        """
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
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        init_weights = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            self._neg_sharpe,
            init_weights,
            args=(self.mu, cov_factor, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        ret, vol = self._portfolio_performance(weights, self.mu, cov_factor)
        
        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe': (ret - risk_free_rate) / vol,
            'n_factors': k,
            'method': 'Factor Model Covariance'
        }
    
    # ========================================================================
    # 6. DISTRIBUTIONAL ROBUSTNESS (WASSERSTEIN)
    # ========================================================================
    
    def wasserstein_optimization(self, epsilon=0.1, risk_free_rate=0.0):
        """
        Distributionally robust optimization using Wasserstein distance.
        
        Parameters:
        -----------
        epsilon : float
            Wasserstein ball radius (robustness parameter)
        risk_free_rate : float
            Risk-free rate
            
        Returns:
        --------
        dict : Optimization results
        """
        def wasserstein_robust_objective(w):
            """
            Robust objective: minimize worst-case CVaR over Wasserstein ball.
            Approximation using moment-based approach.
            """
            # Portfolio return and variance
            port_return = np.dot(w, self.mu)
            port_var = np.dot(w, np.dot(self.cov, w))
            port_std = np.sqrt(port_var)
            
            # Worst-case adjustment (simplified Wasserstein penalty)
            # Based on: E[R] - epsilon * ||grad E[R]||
            wasserstein_penalty = epsilon * port_std * np.sqrt(self.n_assets)
            
            robust_return = port_return - wasserstein_penalty
            
            # Maximize risk-adjusted return
            return -(robust_return - risk_free_rate) / (port_std + 1e-8)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        init_weights = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            wasserstein_robust_objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        ret, vol = self._portfolio_performance(weights, self.mu, self.cov)
        
        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe': (ret - risk_free_rate) / vol,
            'epsilon': epsilon,
            'method': 'Wasserstein Robust'
        }
    
    # ========================================================================
    # 7. ELASTIC NET REGULARIZATION
    # ========================================================================
    
    def elastic_net_optimization(self, lambda_l1=0.01, lambda_l2=0.01, 
                                 risk_free_rate=0.0):
        """
        Portfolio optimization with Elastic Net regularization.
        Combines L1 (sparsity) and L2 (diversification) penalties.
        
        Parameters:
        -----------
        lambda_l1 : float
            L1 regularization parameter (encourages sparsity)
        lambda_l2 : float
            L2 regularization parameter (penalizes large positions)
        risk_free_rate : float
            Risk-free rate
            
        Returns:
        --------
        dict : Optimization results
        """
        def elastic_net_objective(w):
            """Objective with Elastic Net penalty."""
            # Portfolio return and risk
            port_return = np.dot(w, self.mu)
            port_var = np.dot(w, np.dot(self.cov, w))
            
            # Sharpe ratio component
            sharpe_component = -(port_return - risk_free_rate) / np.sqrt(port_var + 1e-8)
            
            # Elastic Net penalty: lambda1 * ||w||_1 + lambda2 * ||w||_2^2
            l1_penalty = lambda_l1 * np.sum(np.abs(w))
            l2_penalty = lambda_l2 * np.sum(w**2)
            
            return sharpe_component + l1_penalty + l2_penalty
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        init_weights = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            elastic_net_objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        
        # Clean up very small weights (from L1 regularization)
        weights[weights < 1e-4] = 0
        weights = weights / np.sum(weights)  # Re-normalize
        
        ret, vol = self._portfolio_performance(weights, self.mu, self.cov)
        
        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe': (ret - risk_free_rate) / vol,
            'n_nonzero': np.sum(weights > 1e-4),
            'method': 'Elastic Net Regularization'
        }
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def compare_all_methods(self, risk_free_rate=0.0):
        """
        Run all optimization methods and compare results.
        
        Returns:
        --------
        pd.DataFrame : Comparison of all methods
        """
        results = []
        
        # 1. Markowitz
        try:
            res = self.markowitz_optimization(risk_free_rate=risk_free_rate)
            results.append(res)
        except Exception as e:
            print(f"Markowitz failed: {e}")
        
        # 2. Worst-case
        try:
            res = self.worst_case_optimization(risk_free_rate=risk_free_rate)
            results.append(res)
        except Exception as e:
            print(f"Worst-case failed: {e}")
        
        # 3. Black-Litterman
        try:
            res = self.black_litterman(risk_free_rate=risk_free_rate)
            results.append(res)
        except Exception as e:
            print(f"Black-Litterman failed: {e}")
        
        # 4. Resampled
        try:
            res = self.resampled_optimization(risk_free_rate=risk_free_rate)
            results.append(res)
        except Exception as e:
            print(f"Resampled failed: {e}")
        
        # 5. Ledoit-Wolf
        try:
            res = self.ledoit_wolf_optimization(risk_free_rate=risk_free_rate)
            results.append(res)
        except Exception as e:
            print(f"Ledoit-Wolf failed: {e}")
        
        # 6. Factor Model
        try:
            res = self.factor_model_optimization(risk_free_rate=risk_free_rate)
            results.append(res)
        except Exception as e:
            print(f"Factor Model failed: {e}")
        
        # 7. Wasserstein
        try:
            res = self.wasserstein_optimization(risk_free_rate=risk_free_rate)
            results.append(res)
        except Exception as e:
            print(f"Wasserstein failed: {e}")
        
        # 8. Elastic Net
        try:
            res = self.elastic_net_optimization(risk_free_rate=risk_free_rate)
            results.append(res)
        except Exception as e:
            print(f"Elastic Net failed: {e}")
        
        # Create comparison DataFrame
        comparison_data = []
        for res in results:
            comparison_data.append({
                'Method': res['method'],
                'Return': res['return'],
                'Volatility': res['volatility'],
                'Sharpe': res['sharpe'],
                'Max Weight': np.max(res['weights']),
                'Min Weight': np.min(res['weights'][res['weights'] > 1e-4]),
                'N Assets': np.sum(res['weights'] > 1e-4)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Store all results
        self.all_results = results
        
        return comparison_df
    
    def get_weights_dataframe(self):
        """
        Return a DataFrame with weights from all methods.
        """
        if not hasattr(self, 'all_results'):
            raise ValueError("Run compare_all_methods() first")
        
        weights_dict = {}
        for res in self.all_results:
            weights_dict[res['method']] = res['weights']
        
        return pd.DataFrame(weights_dict, index=self.asset_names)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

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