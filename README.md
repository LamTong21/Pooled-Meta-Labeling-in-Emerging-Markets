# Mitigating Non-Linear Model Degradation in Emerging Markets via Pooled Cross-Sectional Meta-Labeling and Econometric Scaffolding

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Market](https://img.shields.io/badge/Market-VN--Index-red.svg)]()
[![Methodology](https://img.shields.io/badge/Paradigm-De--Prado--Meta--Labeling-brightgreen.svg)]()

This repository provides an institutional quantitative research framework demonstrating how **Pooled Cross-Sectional Data Pooling** and **$\mathcal{L}_1$-Regularized Meta-Labeling** systematically resolve non-linear decision tree collapse and chronological model degradation in emerging equity markets[cite: 1].

This project serves as the final capstone update synthesizing our three-part market microstructure series:
1. [`Idiosyncratic-Noise-to-Systemic-Alpha`](https://github.com/LamTong21/Idiosyncratic-Noise-to-Systemic-Alpha.git): Raw technical signal derivation and noise boundaries.
2. [`Limits-of-Meta-Labeling-in-Equity-Markets`](https://github.com/LamTong21/Limits-of-Meta-Labeling-in-Equity-Markets.git): Single-asset event starvation ($M \approx 130$) and terminal leaf variance explosion in deep learners[cite: 1].
3. **Current Project**: Resolving degrees-of-freedom deficits via a pooled cross-sectional continuous engine ($N=49$ equities, $1,703$ auditable execution events) governed by in-fold econometric scaffolding[cite: 1].

---

## 1. Problem Formulation: The Emerging Market Structural Curse

Deploying non-linear tree ensembles directly on raw heuristics in emerging markets (such as the VN-Index) fails due to two structural anomalies[cite: 1]:
* **Degrees-of-Freedom Deficit & Tree Collapse**: Standard lookbacks over individual assets generate sparse execution events ($M \approx 130$ triggers)[cite: 1]. Deep recursive trees (Random Forests, Gradient Boosted Trees) rapidly deplete statistical power across splits, causing severe tree collapse, terminal leaf variance inflation, and severe out-of-sample degradation[cite: 1].
* **Chronically Depressed Signal-to-Noise Ratio (SNR)**: Baseline primary triggers achieve a raw win rate of only $\sim 41.44\%$[cite: 1]. Factoring in standard institutional round-trip friction of $0.30\%$ (taxes, slippage, and brokerage) collapses standalone performance to an unviable Profit Factor of $0.88$ ($-248.25\%$ net PnL)[cite: 1].

---

## 2. Quantitative Architecture & Leak-Free Design

The framework completely decouples isolated asset-level microstructure calculations from cross-sectional model execution, enforcing mathematical causality across all transformations[cite: 1].

```
MULTI-ASSET CONTINUOUS POOL
(N = 49 Liquid Equities + VN-Index Benchmark, 2018–2026 | 92,569 Daily Bars)
                            │
                            ▼
           ISOLATED ASSET MICROSTRUCTURE & LABELS
  ├─ Mask Trading Suspensions (> 7 Calendar Days)
  ├─ Microstructure: Garman-Klass Volatility, Flow Z-Score, Relative Vol
  └─ Dynamic Triple Barrier: pt = 2.5 * σ, sl = 1.5 * σ, h = 10 bars
                            │
                            ▼
          POOLED CROSS-SECTIONAL MERGE (1,703 Events)
                            │
                            ▼
       PURGED GROUP TIME-SERIES CROSS-VALIDATION (5 Folds)
  [In-Fold Pipeline]
  ├─ Econometric Standardization: z = (x_test - μ_train) / σ_train
  ├─ Statistical Redundancy & Collinear Predictor Pruning
  └─ L1-Penalized Logit Fitting: min { Cross-Entropy + λ||β||₁ }
                            │
                            ▼
          PORTFOLIO EXECUTION & POSITION SIZING
  ├─ Constraints: Max 5 Concurrent Longs | Friction = 0.30% / trade
  └─ Hard Meta-Filter (P ≥ 0.64) vs. Continuous Sizing State Space
```

### Econometric Variable Consensus Across Folds
Feature extraction and variable selection are strictly contained within each fold of a 5-fold Purged Group Time-Series Split[cite: 1]. Five exogenous predictors demonstrated 100% consensus ($5/5$ folds)[cite: 1]:

| Exogenous Feature | Variable Description | Mean Standardized $\beta$ | Selection Consensus | Impact Mechanism[cite: 1] |
| :--- | :--- | :---: | :---: | :--- |
| `feat_gk_vol`[cite: 1] | Garman-Klass Volatility Metric[cite: 1] | **-0.214**[cite: 1] | 5/5 Folds (100%)[cite: 1] | **Primary Risk Detractor**: Continuous intraday dispersion indicates regime breakdown and increases stop-loss triggers[cite: 1]. |
| `feat_market_corr`[cite: 1] | 20-Day Rolling Corr with VN-Index[cite: 1] | **+0.156**[cite: 1] | 5/5 Folds (100%)[cite: 1] | **Primary Alpha Driver**: Alignment with systemic broad market capital flows accelerates upper profit barriers[cite: 1]. |
| `feat_flow_zscore`[cite: 1] | Normalized Volume Flow Z-Score[cite: 1] | **+0.089**[cite: 1] | 5/5 Folds (100%)[cite: 1] | **Conviction Indicator**: Institutional volume accumulation confirms directional persistence[cite: 1]. |
| `feat_relative_vol`[cite: 1] | Asset Vol / Benchmark Vol[cite: 1] | **-0.011**[cite: 1] | 5/5 Folds (100%)[cite: 1] | **Secondary Vol Penalty**: Excess unsystematic volatility degrades hit probability[cite: 1]. |
| `feat_market_breadth`[cite: 1] | Cross-Sectional Advance/Decline[cite: 1] | **+0.010**[cite: 1] | 5/5 Folds (100%)[cite: 1] | **Macro Edge**: System-wide bullish participation provides positive context[cite: 1]. |

---

## 3. Empirical Results & Net Edge

Evaluating out-of-sample classification identified an optimal filter threshold at $P^*(Meta) \ge 0.64$, filtering out 74.5% of false entry triggers and inverting baseline strategy expectancy[cite: 1]:

| Metric Architecture | Standalone Primary Layer[cite: 1] | Meta-Labeling Layer ($P \ge 0.64$)[cite: 1] | Net Empirical Edge ($\Delta$)[cite: 1] |
| :--- | :---: | :---: | :---: |
| **Trade Sample Size ($M$)** | 1,703[cite: 1] | 434[cite: 1] | -1,269 trades (-74.5%)[cite: 1] |
| **Win Rate (%)** | 41.44%[cite: 1] | **47.66%**[cite: 1] | **+6.22% absolute**[cite: 1] |
| **Profit Factor (PF)** | 0.88[cite: 1] | **1.10**[cite: 1] | **+0.22 (Inverts Expectancy)**[cite: 1] |
| **Aggregate Realized PnL** | -248.25%[cite: 1] | **+38.11%**[cite: 1] | **+286.36% net recovery**[cite: 1] |
| **Portfolio Max Drawdown** | N/A | **-6.02%**[cite: 1] | **vs. -40.34% (VN-Index Benchmark)**[cite: 1] |
| **Friction Deducted** | 0.30% per trade[cite: 1] | 0.30% per trade[cite: 1] | Strictly modeled[cite: 1] |

---

## 4. Bridge to Reinforcement Learning: Resolving Cash Drag

While static thresholding ($P \ge 0.64$) caps drawdown at -6.02%, it creates **Opportunity Starvation**: the portfolio holds $> 80\%$ cash during explosive bull phases, inducing heavy cash drag[cite: 1]. 

To eliminate heuristic threshold tuning, the continuous meta-probability $P(Meta) \in [0, 1]$ is formulated as a foundational state-space variable within a Markov Decision Process (MDP)[cite: 1]:

$$\mathcal{S}_t = \left\{ P(Meta)_{i,t}, \frac{\text{Cash}_t}{\text{NAV}_t}, \sigma_{GK,t}, \rho_{i,Mkt,t}, w_{t-1} \right\}$$[cite: 1]

An active Reinforcement Learning agent (e.g., PPO or SAC) maps $\mathcal{S}_t \rightarrow \mathcal{A}_t \in [0, 1]^N$, converting binary filtering into dynamic sizing and continuous capital allocation[cite: 1].

---

## 5. Quickstart & Verification

### Installation
```bash
git clone [https://github.com/LamTong21/Pooled-Cross-Sectional-Meta-Labeling.git](https://github.com/LamTong21/Pooled-Cross-Sectional-Meta-Labeling.git)
cd Pooled-Cross-Sectional-Meta-Labeling
pip install -r requirements.txt
```

### Automated Leakage & Causal Verification Suite
Run the mathematical integrity test suite to verify zero-leakage temporal partitions and causality:
```bash
pytest -v tests/test_leak_free_pipeline.py
```

### Execution Pipeline
```bash
# 1. Ingest, audit candle geometry, and pool cross-sectional events
python -m src.data_engine

# 2. Run in-fold purged cross-validation & L1 meta-label fitting
python -m src.models.meta_classifier

# 3. Simulate constrained portfolio execution and export teardown
python -m src.backtest.execution_engine
```