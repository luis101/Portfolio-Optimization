# Robust Portfolio Optimization Suite

A Python implementation of eleven portfolio optimization methods, ranging from classical mean-variance to modern distributionally robust approaches. All methods share a common `PortfolioOptimizer` interface so their outputs can be compared directly.

## File overview

| File | Purpose |
|---|---|
| `robust_portfolio_optimizer.py` | Core `PortfolioOptimizer` class with all optimization methods |
| `portfolio_utils.py` | Data download helper and `compare_all_methods` utility |
| `example_usage.py` | End-to-end script: download data, run all methods, print results |

## Optimization methods

### 1. Classical Markowitz (`mean_variance_optimization`)
Solves the standard mean-variance problem via convex quadratic programming (CVXPY). Minimizes portfolio variance for a given return target, or maximizes a risk-adjusted utility `E[r] - λ·Var` when no target is specified. Tends to produce concentrated, estimation-sensitive allocations.

### 2. Minimum Variance (`min_variance`)
Special case of Markowitz that ignores expected returns entirely and minimizes portfolio variance subject only to full investment. More stable than full mean-variance because it avoids noisy return estimates.

### 3. Wasserstein Distributionally Robust (`wasserstein_optimization`)
Optimizes over the worst-case distribution within a Wasserstein ball of radius ε around the empirical distribution. The robust penalty `ε‖w‖₂` shrinks the effective expected return, and a conservative variance term `ε‖w‖₂²` inflates the risk estimate, together penalizing concentrated bets against distributional uncertainty.

### 4. Ellipsoidal Uncertainty (`ellipsoidal_uncertainty_optimization`)
Models uncertainty in both the mean vector (via `κ_μ · ‖Σ^½w‖`) and the covariance matrix (via `κ_σ · ‖w‖₂`). The optimization minimizes the worst-case return minus a risk penalty, where the worst-case return subtracts the mean-uncertainty penalty and the worst-case variance adds the covariance-uncertainty penalty.

### 5. Black-Litterman (`black_litterman`)
Bayesian approach that blends market equilibrium returns (implied by a reference portfolio, defaulting to minimum-variance weights) with explicit investor views. The posterior mean and covariance are derived analytically and then fed into a standard mean-variance optimization, producing more stable and economically grounded allocations than pure Markowitz.

### 6. Resampled Efficiency (`resampling_optimization`)
Michaud's method: bootstraps the return series `n_samples` times, solves a mean-variance problem on each resample, and averages the resulting weights. The averaging smooths out the sensitivity to estimation error and yields more diversified, robust portfolios.

### 7. Ledoit-Wolf Shrinkage (`shrinkage_covariance_optimization`)
Replaces the sample covariance with a shrinkage estimator that blends the sample covariance toward a constant-correlation target matrix. The optimal shrinkage intensity δ is computed analytically following Ledoit & Wolf (2004), reducing the impact of extreme sample eigenvalues and improving out-of-sample covariance estimates.

### 8. Factor Model Covariance (`factor_model_optimization`)
Decomposes the covariance matrix into a low-rank systematic component (`B·F·Bᵀ`) plus a diagonal idiosyncratic term (`D`). When no external factors are supplied, PCA is used to extract the top `k` factors. The resulting structured covariance matrix is better conditioned and generalises more reliably to new data.

### 9. CVaR Optimization (`cvar_optimization`)
Minimizes Conditional Value at Risk (expected shortfall) at confidence level α using Rockafellar & Uryasev's linear reformulation. The objective trades off expected return against tail risk, making this approach suitable when loss distributions are non-normal or heavy-tailed.

### 10. Wasserstein Robust CVaR (`wasserstein_cvar_optimization`)
Combines CVaR tail-risk management with distributional robustness. A Wasserstein robustness margin `ε‖w‖₂ / α` is added to the empirical CVaR and the worst-case expected return is adjusted downward, guarding against both tail losses and distributional shift simultaneously.

### 11. Elastic Net Regularization (`elastic_net_optimization`)
Augments the mean-variance objective with an L1 penalty (sparsity — pushes small weights to zero) and an L2 penalty (diversification — discourages very large weights). The combined penalty produces sparse, interpretable portfolios where only the most attractive assets receive non-trivial allocations.

## Data

`download_fin_data` in `portfolio_utils.py` fetches monthly closing prices for a configurable list of tickers from Yahoo Finance and computes simple monthly returns. The default universe is 15 global equity indices (S&P 500, NASDAQ, Dow Jones, DAX, FTSE 100, CAC 40, Hang Seng, ASX 200, SENSEX, TWSE, IPC, KOSPI, Nikkei 225, Bovespa, STI).

## Dependencies

```
numpy
pandas
cvxpy
yfinance
scipy
scikit-learn
matplotlib
```

## Running

```bash
python example_usage.py
```
