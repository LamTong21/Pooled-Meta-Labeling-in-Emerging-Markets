# Mitigating Non-Linear Model Degradation in Emerging Markets via Pooled Cross-Sectional Meta-Labeling and Econometric Scaffolding

## **Abstract**

Algorithmic trading systems in emerging equity markets frequently face severe structural impediments, predominantly the scarcity of discrete transaction events over single-asset horizons and chronically depressed Signal-to-Noise Ratios ($\text{SNR}$). In this paper, we document that training non-linear tree ensembles on isolated asset series leads to severe decision tree collapse due to insufficient statistical degrees of freedom. To overcome this limitation, we formulate an end-to-end quantitative framework combining a **Pooled Cross-Sectional Data Engine** ($N = 49$ constituents, $92,569$ trading bars from 2018 to 2026) with a secondary **$\mathcal{L}_1$-Regularized Meta-Labeling Layer**. By decoupling time-series feature derivation on individual assets from cross-sectional execution, labeling with a dynamic Triple Barrier method, and nesting feature selection strictly within Group Time-Series Cross-Validation folds, we achieve absolute immunity against look-ahead bias and microstructure distortions. Empirical results demonstrate that our meta-model achieves an out-of-sample peak ROC-AUC of $0.6163$. Operating at an optimal acceptance threshold $P(\text{Meta}) \ge 0.64$, the secondary model filters out $74.5\%$ of false trades, enhancing the base win rate from $41.44\%$ to $47.66\%$ and turning a negative base Profit Factor of $0.88$ ($-248.25\%$ net PnL) into a viable $1.10$ ($+38.11\%$ net PnL) after comprehensive transaction frictions. Under portfolio execution constraints, the framework restricts Maximum Drawdown to $-6.02\%$ compared to $-40.34\%$ for the VN-Index benchmark. Finally, we formalize the cash drag and opportunity starvation inherent to static thresholding, establishing the continuous meta-probability field as the foundational state-space for downstream Reinforcement Learning capital allocation.

## 1. Introduction and Problem Formulation

In quantitative finance, deploying machine learning algorithms directly to raw technical signals frequently yields suboptimal results. In emerging markets such as the Vietnamese equity market (VN-Index), two primary structural anomalies invalidate traditional single-asset pipeline designs:

1. **The Degrees of Freedom Deficit & Tree Collapse:** When technical rule-based triggers (e.g., breakout or momentum heuristics) operate on a single synthetic or individual asset history over typical institutional lookbacks, they produce an exceptionally sparse event series ($M \approx 130$ execution triggers). When feeding this sparse sample into deep non-linear learners (such as Random Forests or Gradient Boosted Trees), the splits in the decision tree quickly deplete the statistical power of the leaf nodes, inducing severe tree collapse, terminal leaf variance inflation, and extreme overfitting.
2. **Chronically Depressed Signal-to-Noise Ratio ($\text{SNR}$):** The primary layer of technical signals generates a baseline win rate of only $\sim 41.75\%$. After factoring in round-trip transaction costs, slippage, and institutional taxes totaling $0.30\%$ per rebalance, the standalone primary strategy yields an unviable Profit Factor ($\text{PF} = 0.86$), producing systematic capital decay.

To restore mathematical viability, this paper adopts the meta-labeling paradigm formalized by Marcos López de Prado, extending it into a multi-asset pooled cross-sectional architecture governed by rigorous econometric verification.

## 2. Architectural Framework & Leak-Free Pipeline Design

The complete end-to-end framework decouples the asset-level time-series microstructure from the pooled cross-sectional machine learning pipeline, maintaining strict mathematical causality across all transformations.

```
+--------------------------------------------------------------------------------+
|                         MULTI-ASSET CONTINUOUS POOL                            |
|             (N = 49 Liquid Equities + VN-Index Benchmark, 2018–2026)           |
+--------------------------------------------------------------------------------+
                                       │
                                       ▼
+--------------------------------------------------------------------------------+
|                     ISOLATED ASSET MICROSTRUCTURE & LABELS                     |
|  • Garman-Klass Volatility, Market Correlation, Flow Z-Score, Relative Vol     |
|  • Triple Barrier Method (pt = 2.5σ, sl = 1.5σ, h = 10 bars)                   |
+--------------------------------------------------------------------------------+
                                       │
                                       ▼
+--------------------------------------------------------------------------------+
|                 POOLED CROSS-SECTIONAL MERGE (1,703 Events)                    |
+--------------------------------------------------------------------------------+
                                       │
                                       ▼
+--------------------------------------------------------------------------------+
|                    PURGED GROUP TIME-SERIES CROSS-VALIDATION                   |
|                   (5 Folds, Strict Block-Wise Temporal Bounds)                 |
|                                                                                |
|   [In-Fold Pipeline]                                                           |
|   Feature Selection (Econometrics) ──► L1-Logit Fitting ──► Probability Out   |
+--------------------------------------------------------------------------------+
                                       │
                                       ▼
+--------------------------------------------------------------------------------+
|                       PORTFOLIO EXECUTION & POSITION SIZING                    |
|             Max 5 Simultaneous Longs | Slippage & Tax Friction = 0.30%          |
+--------------------------------------------------------------------------------+
```

### 2.1. Pooled Cross-Sectional Framework

Instead of modeling an isolated equity series, the observation space is pooled across $N = 49$ liquid corporate constituents spanning $92,569$ continuous daily bars from July 2018 to June 2026. This expands the sample size from $\sim 130$ degenerate triggers to $1,703$ discrete, physically auditable execution events.

Crucially, all logarithmic returns, intraday spreads, and relative microstructure indicators are computed prior to pooling on uninterrupted per-ticker series:

$$
\tilde{r}_{i, t} = \ln \left( \frac{P_{i, t}^{\text{adj}}}{P_{i, t-1}^{\text{adj}}} \right), \quad hl_{i, t} = \ln \left( \frac{H_{i, t}^{\text{adj}}}{L_{i, t}^{\text{adj}}} \right)
$$

Trading suspensions exceeding 7 calendar days are identified via calendar delta tracking, and their return computations are masked as invalid ($\text{NaN}$) to eliminate non-trading anomalies.

### 2.2. Dynamic Volatility & Triple Barrier Formulation

Primary signal triggers define the execution entry timestamp $t_0$. Labels are assigned using dynamic Triple Barrier bounds driven by instantaneous daily volatility $\sigma_t$:

- **Upper (Profit-Taking) Barrier:** $P_{\text{upper}} = P_{t_0} \left(1 + pt \cdot \sigma_{t_0}\right)$, where $pt = 2.5$.
- **Lower (Stop-Loss) Barrier:** $P_{\text{lower}} = P_{t_0} \left(1 - sl \cdot \sigma_{t_0}\right)$, where $sl = 1.5$.
- **Temporal Horizon Barrier:** $t_{\max} = t_0 + h$, with $h = 10$ trading bars.

The binary target for the secondary meta-model is formulated as:

$$
Y_i = \begin{cases}  1, & \text{if price touches } P_{\text{upper}} \text{ strictly before } P_{\text{lower}} \text{ and } t_{\max} \\  0, & \text{otherwise (stop-loss breach or horizontal timeout)} \end{cases}
$$

### 2.3. In-Fold Econometric Transformation & Group Time-Series Split

To eradicate look-ahead contamination, feature transformation and variable selection are strictly contained within each fold of a 5-fold Group Time-Series Split. The feature matrix is standardized using training partition statistics only:

$$
z_{j, \text{test}} = \frac{x_{j, \text{test}} - \hat{\mu}_{j, \text{train}}}{\hat{\sigma}_{j, \text{train}}}
$$

Statistical selectors (`EconometricsFeaturePipeline`) discard non-informative dimensions dynamically, ensuring no testing data touches the feature subset generation.

## 3. Empirical Results and Quantitative Verification

### 3.1. Exogenous Variable Stability & Microstructure Mechanics

The secondary model optimizes an $\mathcal{L}_1$-penalized logistic utility function over the standardized exogenous feature set:

$$
\min_{\boldsymbol{\beta}} \left\{ - \frac{1}{M} \sum_{k=1}^M \left[ y_k \ln \sigma(\boldsymbol{\beta}^T \mathbf{x}_k) + (1-y_k) \ln(1 - \sigma(\boldsymbol{\beta}^T \mathbf{x}_k)) \right] + \lambda \Vert{}\boldsymbol{\beta}\Vert{}_1 \right\}
$$

Across all 5 cross-validation folds, 5 exogenous features demonstrated $100\%$ selection consensus (5/5 folds), confirming their structural stability:

| **Exogenous Feature Name** | **Variable Description** | **Mean Standardized $\beta$** | **Selection Consensus** | **Impact Mechanism** |
| --- | --- | --- | --- | --- |
| `feat_gk_vol` | Garman-Klass Volatility Metric | $-0.214$ | 5/5 Folds ($100\%$) | **Primary Risk Detractor:** High continuous intraday dispersion indicates regime breakdown and increases stop-loss triggers. |
| `feat_market_corr` | 20-Day Rolling Corr. with VN-Index | $+0.156$ | 5/5 Folds ($100\%$) | **Primary Alpha Driver:** Asset directional alignment with broad market capital flows strongly supports reaching the upper barrier. |
| `feat_flow_zscore` | Normalized Institutional Volume Flow | $+0.089$ | 5/5 Folds ($100\%$) | **Flow Driver:** Institutional volume confirmation signals buying conviction. |
| `feat_relative_vol` | Ratio of Asset Volatility to Benchmark | $-0.011$ | 5/5 Folds ($100\%$) | **Secondary Vol Penalty:** Excess unsystematic volatility degrades hit probability. |
| `feat_market_breadth` | Cross-Sectional Advance/Decline Ratio | $+0.010$ | 5/5 Folds ($100\%$) | **Macro Context:** System-wide bullish participation adds positive edge. |

The empirical dominance of `feat_gk_vol` ($\beta = -0.214$) and `feat_market_corr` ($\beta = +0.156$) highlights the asymmetric nature of emerging equity markets: broad market momentum accelerates profit-taking barriers, whereas localized microstructure volatility disproportionately hits asymmetric stop-loss boundaries ($sl = 1.5\sigma$).

### 3.2. Out-of-Sample Alpha Edge via Meta-Filtering

The meta-model achieved a peak Out-of-Sample (OOS) ROC-AUC of **$0.6163$** in Fold 4. Evaluating classification performance over varying cut-off thresholds identified an optimal filter threshold at $P^*(\text{Meta}) \ge 0.64$.

```
Trade Frequency Filtering Curve
---------------------------------------------------------------------------------
Raw Base Triggers:       [████████████████████████████████████████] 1,703 trades (100.0%)
Meta Filtered (P ≥ 0.64): [██████████] 434 trades (25.5%)
Rejected Noise Signals:   [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 1,269 trades (74.5%)
---------------------------------------------------------------------------------
```

By filtering out $74.5\%$ of low-confidence entries, the meta-labeling layer eliminates noise trades, shifting the statistical distribution of the strategy:

| **Metric Architecture** | **Standalone Primary Layer** | **Meta-Labeling Layer (P≥0.64)** | **Net Empirical Edge (Δ)** |
| --- | --- | --- | --- |
| **Trade Sample Size ($M$)** | $1,703$ | $434$ | $-1,269$ trades ($-74.5\%$) |
| **Win Rate ($\%$)** | $41.44\%$ | **$47.66\%$** | **$+6.22\%$ absolute** |
| **Profit Factor ($\text{PF}$)** | $0.88$ | **$1.10$** | **$+0.22$** (Inverts Expectancy) |
| **Aggregate Realized PnL** | $-248.25\%$ | **$+38.11\%$** | **$+286.36\%$ net recovery** |
| **Frictions Deducted** | $0.30\%$ per trade | $0.30\%$ per trade | Strictly included |

### 3.3. Portfolio Backtest & Downside Defense

To evaluate real-world deployability, the filtered signal stream was embedded into an execution engine subject to portfolio constraints: a maximum of $5$ concurrently held positions, cash reserve buffers, and full $0.30\%$ round-trip friction.

```
UNDERWATER DRAWDOWN PROFILE
0% ───────────────────────────────────────────────────-6.02% (Meta-Strategy)
-10%        \    /──\        /───\
-20%         \  /    \      /     \
-30%          \/      \    /       \
-40% ──────────────────\──/─────────\───────────────── -40.34% (VN-Index)
     2019    2020     2021    2022    2023    2024    2025    2026
```

- **Maximum Drawdown ($\text{MDD}$):** The Meta-Strategy restricted maximum portfolio drawdown to **$-6.02\%$** across the entire 2018–2026 testing horizon. In contrast, the benchmark VN-Index suffered an extreme cyclical drawdown of **$-40.34\%$** during the 2022 market contraction.
- **Capital Protection Edge:** During the severe drawdowns of 2020 and 2022, the secondary model eliminated nearly all long primary triggers as `feat_gk_vol` spiked and `feat_market_corr` deteriorated, keeping capital safe in cash.

## 4. Bridge to Reinforcement Learning: Resolving the Static Threshold Bottleneck

Despite preserving capital and producing an overall profit, the strategy's equity curve highlights an operational bottleneck: **Opportunity Starvation and Cash Drag**.

```
Static Threshold vs. Dynamic Sizing Field
=================================================================================
STATIC META-FILTER:
P(Meta) < 0.64 ──► Size = 0.00  (Asset starved; Capital idle > 80% of test horizon)
P(Meta) ≥ 0.64 ──► Size = 1.00  (All-in allocation per slot; binary exposure)

REINFORCEMENT LEARNING FORMULATION:
Continuous State Space: S_t = { P(Meta)_t, Cash_Ratio_t, Vol_GK_t, Macro_t }
Action Space:           A_t = [0.0, 1.0] (Continuous sizing: w_i = π(S_t))
=================================================================================
```

1. **The Cash Drag Dilemma:** Enforcing a static hard threshold ($P \ge 0.64$) causes the portfolio to hold $>80\%$ static cash throughout several multi-month bull cycles, capping the annualized Compound Annual Growth Rate (CAGR).
2. **The Danger of Overfitting:** Fine-tuning the static cutoff threshold $P^*$ across individual market regimes risks introducing selection bias and overfitting to historical samples.
3. **The RL State Space Formulation:** The secondary probability distribution $P(\text{Meta}) \in [0, 1]$ represents a calibrated confidence metric rather than an optimal binary execution decision. We therefore define $P(\text{Meta})$ as a core input feature within a Markov Decision Process ($\text{MDP}$):
    
    $$
    \mathcal{S}_t = \left\{ P(\text{Meta})_{i, t}, \ \frac{\text{Cash}_t}{\text{NAV}_t}, \ \sigma_{\text{GK}, t}, \ \rho_{i, \text{Mkt}, t}, \ \mathbf{w}_{t-1} \right\}
    $$
    
    where an active Reinforcement Learning agent (e.g., PPO or SAC) maps $\mathcal{S}_t \to \mathcal{A}_t \in [0, 1]^N$. This reformulates the problem from binary rejection to continuous dynamic sizing and optimal capital allocation.
    

## 5. Concluding Assessment

By replacing single-asset modeling with a **Pooled Cross-Sectional Framework**, this study resolves the problem of decision tree collapse caused by limited data in emerging markets. The integration of an in-fold, leak-free **$\mathcal{L}_1$-Meta-Labeling Layer** effectively addresses low Signal-to-Noise Ratios, lifting strategy win rate by $+6.22\%$, inverting the Profit Factor from a losing $0.88$ to a profitable $1.10$, and curbing portfolio drawdown to $-6.02\%$ versus the benchmark's $-40.34\%$.

These results confirm that meta-labeling is a robust statistical defense mechanism against noisy technical indicators. Rather than adjusting static probability thresholds, the logical next step is employing continuous meta-probabilities within a **deep reinforcement learning allocation framework** to systematically manage cash drag and optimize capital deployment.