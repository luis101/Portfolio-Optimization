# Robust Portfolio Optimization Suite

A Python implementation of more than twenty portfolio construction methods, spanning classical mean-variance, distributionally robust, tail- and drawdown-risk, growth-optimal, and risk-based families. All methods share a common `PortfolioOptimizer` interface so their outputs can be compared directly via `compare_all_methods`.

## File overview

| File | Purpose |
|---|---|
| `robust_portfolio_optimizer.py` | Core `PortfolioOptimizer` class with all optimization methods |
| `portfolio_utils.py` | Data download helper and `compare_all_methods` utility |
| `example_usage.py` | End-to-end script: download data, run all methods, print results |

## Optimization methods

### Classical 

### 1. Classical Markowitz (`mean_variance_optimization`)
Solves the standard mean-variance problem via convex quadratic programming (CVXPY). Minimizes portfolio variance for a given return target, or maximizes a risk-adjusted utility `E[r] - λ·Var` when no target is specified. Tends to produce concentrated, estimation-sensitive allocations.

### 2. Minimum Variance (`min_variance`)
Special case of Markowitz that ignores expected returns entirely and minimizes portfolio variance subject only to full investment. More stable than full mean-variance because it avoids noisy return estimates.

### Robust to parameter / distributional uncertainty

### 3. Wasserstein Distributionally Robust (`wasserstein_optimization`)
Mean-variance optimization over a Wasserstein ball of radius ε around the empirical distribution, using the exact "square-root regularization" reformulation, following Blanchet, Chen, and Zhou (2022). The worst-case standard deviation becomes `√(wᵀΣw) + √ε‖w‖` and the worst-case mean `wᵀμ − √ε‖w‖`, so the robustification is *additive in the standard deviation*. The penalty norm follows the transport ground cost (`‖w‖₂` for Euclidean, `‖w‖∞` for ℓ₁).

### 4. Ellipsoidal Uncertainty (`ellipsoidal_uncertainty_optimization`)
Models uncertainty in both the mean vector (via `κ_μ · ‖(Σ/T)^½ w‖`) and the covariance matrix (via `κ_σ · ‖w‖₂`). Minimizes the worst-case return plus the inflated worst-case variance.
### 5. Black-Litterman (`black_litterman`)
Bayesian approach that blends market equilibrium returns (implied by a reference portfolio, defaulting to minimum-variance weights) with explicit investor views `P` and view magnitudes `Q` (`view_returns`). The posterior mean and covariance are derived analytically and then fed into a mean-variance optimization, producing more stable and economically grounded allocations than pure Markowitz.

### 6. Resampled Efficiency (`resampling_optimization`)
Michaud's method: bootstraps the return series `n_samples` times (reproducible via `seed`), solves a mean-variance problem on each resample, and averages the resulting weights. The averaging smooths out sensitivity to estimation error and yields more diversified, robust portfolios.

### Robust covariance estimation

### 7. Ledoit-Wolf Shrinkage (`shrinkage_covariance_optimization`)
Replaces the sample covariance with a shrinkage estimator that blends it toward a constant-correlation target matrix. The optimal shrinkage intensity δ is computed analytically following Ledoit & Wolf (2004), reducing the impact of extreme sample eigenvalues and improving out-of-sample covariance estimates. Addresses estimation *noise*.

### 8. MCD Robust Covariance (`mcd_robust_covariance_optimization`)
Estimates location and covariance with the Minimum Covariance Determinant estimator, fitted on the subset of observations with the smallest covariance determinant. Complementary to Ledoit-Wolf: it targets *outliers* (crises, flash crashes) rather than noise. The robust mean and covariance are then used in a mean-variance optimization.

### 9. Factor Model Covariance (`factor_model_optimization`)
Decomposes the covariance into a low-rank systematic component (`B·F·Bᵀ`) plus a diagonal idiosyncratic term (`D`). With supplied factor returns, loadings are estimated by regression; otherwise PCA is used to extract the top `k` factors. The resulting structured covariance is better conditioned and generalises more reliably to new data.

### Tail & drawdown risk

### 10. CVaR Optimization (`cvar_optimization`)
Minimizes Conditional Value at Risk (expected shortfall) at confidence level α using Rockafellar & Uryasev's linear reformulation, traded off against expected return. Suitable when loss distributions are non-normal or heavy-tailed.

### 11. Wasserstein Robust CVaR (`wasserstein_cvar_optimization`)
Combines CVaR tail-risk management with distributional robustness. A Wasserstein robustness margin `ε‖w‖₂` is added to the empirical CVaR and the worst-case expected return is adjusted down by the same term (Mohajerin Esfahani & Kuhn 2018). Because CVaR is coherent, the margin carries no `1/α` factor.

### 12. Mean-CDaR (`cdar_optimization`)
Minimizes Conditional Drawdown-at-Risk, the average of the worst α portfolio drawdowns along the cumulative-return path (Chekhlov, Uryasev & Zabarankin 2005), traded off against expected return. Unlike CVaR (a single-period tail loss), CDaR penalizes peak-to-trough declines. Well suited to long-horizon investors who care about path/drawdown risk.

### Regularization

### 13. Elastic Net Regularization (`elastic_net_optimization`)
Augments the mean-variance objective with an L1 penalty (sparsity, pushes small weights to zero) and an L2 penalty (diversification , discourages very large weights). The combined penalty produces sparse, interpretable portfolios where only the most attractive assets receive non-trivial allocations.

### Risk-based allocation

### 14. Hierarchical Risk Parity (`hierarchical_risk_parity`)
Following López de Prado (2016). Three steps: hierarchical clustering on the correlation-distance matrix, quasi-diagonalization (reordering the covariance by the cluster tree), and recursive bisection allocating inverse-variance weights top-down. Requires no matrix inversion, so it is robust to ill-conditioned / near-singular covariances.

### 15. Risk Parity / Equal Risk Contribution (`risk_parity_optimization`)
Allocates so that each asset contributes equally to total portfolio variance. Two solvers: SLSQP minimizing the dispersion of risk contributions, or Spinu's convex log-barrier formulation (`min ½wᵀΣw − Σ log wᵢ`). Diversifies by risk rather than by capital.

### 16. Maximum Diversification (`maximum_diversification`)
Following Choueifaty & Coignard (2008). Maximizes the diversification ratio (weighted-average asset volatility ÷ portfolio volatility) via the convex reformulation `min wᵀΣw s.t. σᵀw = 1, w ≥ 0`, then renormalizes. Tilts toward low-correlation, lower-volatility assets.

### 17. CVaR Risk Parity / ERC-CVaR (`cvar_risk_parity_optimization`)
Risk parity on tail risk: equalizes each asset's contribution to portfolio CVaR rather than variance, via convex log-barrier risk-budgeting `min CVaR(w) − Σ log wᵢ` (Roncalli 2013).

### 18. CDaR Risk Parity / ERC-CDaR (`cdar_risk_parity_optimization`)
Risk parity on drawdown: equalizes contributions to portfolio CDaR, the drawdown analogue of ERC-CVaR, using the same log-barrier risk-budgeting with CDaR (from the drawdown LP) as the risk measure.

### Growth-optimal (Kelly)

### 19. Kelly / Fractional Kelly (`kelly_optimization`)
Maximizes the expected log-growth rate `(1/T)·Σ log(1 + Rₜ·w)`, i.e. the long-run geometric growth of wealth. Full Kelly maximizes growth but is aggressive and very sensitive to estimation error; **fractional Kelly** (`fraction < 1`) blends the Kelly weights toward a conservative anchor (minimum-variance by default), with half-Kelly keeping most of the growth edge at roughly half the log-wealth volatility. Requires decimal returns, pass `return_scale=100` with percentage-point data.

### 20. Worst-Case Kelly (`worst_case_kelly_optimization`)
Distributionally robust Kelly (Sun & Boyd 2018): maximizes the worst-case average log-growth over the fraction α of least-favorable periods (the lower-tail CVaR of per-period log-growth), guarding growth against adverse scenarios. Counterpart to fractional Kelly's fixed shrinkage, it derives *how much* to back off from an explicit ambiguity set.

### Benchmark

#### 21. Equal Weight 1/N (`equal_weight_portfolio`)
Naive diversification, `wᵢ = 1/N`. An estimation-free benchmark that is notoriously hard to beat out-of-sample (DeMiguel, Garlappi & Uppal 2009).

## Data

`download_fin_data` in `portfolio_utils.py` fetches monthly closing prices for a configurable list of tickers from Yahoo Finance and computes simple monthly returns. The default universe is 15 global equity indices (S&P 500, NASDAQ, Dow Jones, DAX, FTSE 100, CAC 40, Hang Seng, ASX 200, SENSEX, TWSE, IPC, KOSPI, Nikkei 225, Bovespa, STI).

**Return units.** Most methods are invariant to a constant rescaling of returns, but the Kelly methods are not (the log of gross returns is nonlinear). `example_usage.py` scales returns to percentage points (`× 100`), so the Kelly methods are invoked with `return_scale=100` to recover decimal gross returns `1 + r`; pass `return_scale=1` if your returns are already decimals.

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
