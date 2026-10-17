# DeFi Liquidation Survival Analysis

A multi-paradigm empirical study of borrower liquidation events on decentralized finance (DeFi) lending protocols. The project combines survival analysis, fuzzy mathematics, multivariate statistics, time-series point-process modeling, and Bayesian game theory into a unified investigation of what predicts liquidation, how liquidations cluster in time, how within-user correlation biases standard inference, and whether borrower-liquidator interaction admits a game-theoretic interpretation.

Built on the FinSurvival 2025 benchmark (Codabench public phase). All quantitative hypotheses are pre-registered before execution; nulls are reported with equal transparency to positive findings.

---

## Table of contents

1. [Headline findings](#headline-findings)
2. [Research questions](#research-questions)
3. [Dataset and feature engineering](#dataset-and-feature-engineering)
4. [Methodology — notebook by notebook](#methodology-notebook-by-notebook)
5. [Repository structure](#repository-structure)
6. [Reproducing the results](#reproducing-the-results)
7. [Pre-registration philosophy](#pre-registration-philosophy)
8. [Known limitations](#known-limitations)
9. [Authors](#authors)
10. [License](#license)

---

## Headline findings

Across 11 pre-registered analyses on 16 DeFi survival tasks:

1. **Independence assumption inflates significance by 5-18× in standard DeFi survival analyses.** Under cluster-robust Cox regression accounting for within-user correlation (nb8), 60% of effects declared significant in naive Cox regression fail to replicate under rigorous inference. In the most dramatic case (Borrow→Liquidated), only 2 of 18 naively-significant effects survive proper inference.
2. **DeFi liquidations exhibit strong self-excitation (Hawkes branching ratio η = 0.51).** Roughly half of all liquidations trigger subsequent liquidations within the exponential decay window — direct empirical evidence of liquidation cascades (nb10).
3. **Market activity Granger-causes daily liquidation rates** at lags of up to 7 days (BH-corrected FDR < 0.05, nb10).
4. **Liquidated and non-liquidated users occupy distinct regions of the 80-feature DeFi behavior space** after leakage correction (Mahalanobis distance = 1.01, Hotelling's T² p ≪ 10⁻³⁰⁰, nb9).
5. **Modest but highly significant borrower type-persistence**: each additional prior liquidation increases current hazard by 3.1% (HR = 1.031, 95% CI [1.025, 1.037], p ≈ 10⁻²⁵, nb11). Consistent with weak Bayesian learning dynamics.
6. **Fuzzy interval type-2 memberships provide a weak uncertainty signal** (significant correlation with model error on 12/16 tasks, nb7) but do not improve C-index discrimination (3 pre-registered formulations tested — marginal features, interactions, uncertainty tiering — all null, nb4/nb6/nb5).

---

## Research questions

**RQ1** — Does the independence assumption in standard DeFi survival benchmarks (FinSurvival 2025, Dyson et al., Aragon et al.) materially bias coefficient inference?

**RQ2** — Do interval type-2 fuzzy regime memberships, integrated with XGBoost+Cox survival models, improve discrimination or uncertainty quantification over gradient-boosted Cox baselines?

**RQ3** — Can DeFi liquidation events be characterized as a self-exciting point process, and does market-wide activity causally precede liquidation bursts?

**RQ4** — Do multivariate feature profiles distinguish liquidated from non-liquidated users in interpretable latent structure?

**RQ5** — Does the empirical pattern of repeat liquidations admit a game-theoretic (Bayesian signaling) interpretation?

---

## Dataset and feature engineering

**FinSurvival 2025 benchmark** (public, from the Codabench challenge). 16 time-to-event tasks spanning 4 index events (Borrow, Deposit, Repay, Withdraw) × 5 outcome events (Deposit, Borrow, Repay, Withdraw, Liquidated), minus 4 trivial self-transitions.

**Feature inventory**: 80 engineered features partitioned into:
- User-level activity history (31 features): userBorrowSum, userLiquidationCount, userActiveDaysWeekly, etc. All features verified as cumulative-as-of (no temporal leakage).
- Market-level liquidity signals (25 features): marketBorrowSum, marketDepositAvgAmountUSD, marketLiquidationCount, etc.
- Temporal and seasonal (24 features): sinDayOfMonth, cosQuarter, userSecondsSincePreviousTransaction, etc.
- Transactional / per-event (5 features): priceInUSD, amount, amountUSD, logAmount, logAmountUSD.

**Chronological split**:
- Training window: 2022-03-12 to 2024-03-31 (~750 days, 24.7 GB)
- Pre-registered holdout: 2024-04-01 to 2024-07-02 (~3 months)
- Challenge test window (not used, Codabench phase closed): 2024-07-03 to 2025-08-21

**Fuzzy inputs (nb3)**: Four orthogonal signals derived from base features — market stress (`mkt_liq_share`), user-vs-market concentration (`user_mkt_ratio`), transaction amount extremeness (`amount_z`), user liquidation pressure (`user_liq_rate`). Fitted with Fuzzy C-Means (K=4, fuzzifier m=2.0) on fit-window data only, generating Interval Type-2 memberships via Gaussian membership functions with FOU = 0.20.

---

## Methodology — notebook by notebook

### nb0_setup.ipynb — Environment and folder scaffolding

Dependencies installation, folder hierarchy creation (`data/{raw,features,baselines,models,fuzzy,game,eval,processed}`), configuration constants (TRAIN_START, TRAIN_END, 16-task definitions, random seeds).

### nb1_data.ipynb — Data extraction and task registration

- Extracts `FinSurvival2025_Dataset.zip` and nested tarballs (`participant_data.tar.gz`, `test_features.tar.gz`).
- Enumerates all 16 index-outcome task directories, confirms chronological non-overlap between train (2022-03-14 to 2024-07-02) and challenge test (2024-08-01 to 2025-07-21).
- Saves `task_registry.json` (16 tasks with paths) and `feature_list.json` (80 features grouped by type).

### nb2b_pinned_baseline.ipynb — XGBoost + Cox baseline on chronological holdout

The pre-registered baseline. For each of 16 tasks:
- Load full train.csv, encode categoricals (userCoinTypeMode, userReserveMode), fill NaN with 0.
- Split by timestamp: fit window < 2024-04-01, holdout ∈ [2024-04-01, 2024-07-02].
- Train XGBoost with `objective="survival:cox"`, max_depth=6, learning_rate=0.05, 300 rounds max with 20-round early stopping on last 10% of fit (chronologically ordered validation).
- Evaluate Harrell's C-index on the holdout.
- Save models and holdout predictions.

Result: **mean holdout C-index 0.8480** across 16 tasks (reference: FinSurvival challenge winner Afinit reports 0.9142 on hidden test).

### nb3_fuzzy.ipynb — IT2 fuzzy regime memberships

- Compute 4 orthogonal input signals from base features for each row.
- Fit Fuzzy C-Means (4 regimes: Safe, Watch, Danger, Critical) on fit-window data only.
- Compute Interval Type-2 memberships with Karnik-Mendel type reduction:
  - Upper and lower memberships via Gaussian with σ(1±FOU)
  - Normalized reduced membership and interval width per row
- Save memberships for all rows (fit + holdout) to parquet, aligned by position with raw CSV.

Fit-window regime distribution: Safe 66.7%, Watch 18.6%, Danger 6.8%, Critical 8.0%.

### nb4_augmented.ipynb — Fuzzy as marginal features (null result)

Pre-registered hypothesis H2: adding fuzzy memberships as 5 additional XGBoost features improves C-index.

Design identical to nb2b with features = 80 base + {mu_Safe, mu_Watch, mu_Danger, mu_Critical, interval_width}. Positional alignment of memberships with raw CSV (assert equal lengths, direct column assignment — no merge).

Result: **mean delta +0.0031**, paired block bootstrap gives 1 WIN, 1 LOSE, 14 NULL. **Pre-reg verdict: NULL.**

### nb6_interactions.ipynb — Fuzzy × base feature interactions (null result)

Pre-registered hypothesis H3: XGBoost cannot easily learn multiplicative interactions between regime memberships and transactional features; adding explicit interactions should help.

12 interactions constructed: {amountUSD_log, userLiquidationCount, marketLiquidationCount, userBorrowCount} × {mu_Critical, mu_Danger, interval_width}. Total feature set: 80 base + 5 fuzzy + 12 interactions = 97.

Pipeline mirrors nb4 exactly except for the 12 extra features. Paired block bootstrap for CIs.

Result: **mean delta vs augmented +0.003**, 5 WIN / 2 LOSE / 9 NULL after safe (non-cross-paired) bootstrap. **Pre-reg verdict: NULL** (threshold +0.005 not met).

Also includes feature-timing audit confirming userLiquidationCount is cumulative-as-of (no temporal leakage).

### nb7_full_calib.ipynb — Full calibration analysis

Pre-registered hypothesis H4: fuzzy memberships provide calibrated uncertainty even if discrimination is null.

- Reload saved baseline and augmented models.
- Score the fit window to get fit-set risks for each model.
- Compute Breslow baseline cumulative hazard (vectorized, numerically stable with max-shift log-sum-exp trick).
- Convert risk scores to survival probabilities at horizons {7, 30, 90} days.
- Compute IPCW Brier scores at each horizon with KM estimator of censoring distribution.
- Integrated Brier Score via trapezoidal rule.
- Prediction interval coverage using normalized interval_width as uncertainty signal.
- Spearman correlation of interval_width with per-row squared Brier error.

Result: IBS NULL (1/16 tasks aug better by >0.005), coverage NULL (naive PI construction gives 27.6% empirical coverage vs 90% nominal), **but width-error correlation POSITIVE and significant on 12/16 tasks** (consistent with chance = 1/16 expected). **Pre-reg verdict: NULL** overall, with secondary finding that interval_width carries weak informative signal.

### nb8_cluster_robust_cox.ipynb — Cluster-robust Cox regression (strong positive)

Pre-registered hypothesis H5: ignoring within-user correlation in standard Cox regression inflates coefficient significance.

Three representative tasks (Borrow_Liquidated, Deposit_Borrow, Withdraw_Repay):
- Select top 20 features by baseline XGBoost gain per task.
- Subsample users (5000 users keeping all their events) for tractability.
- Z-standardize features.
- Fit naive Cox (independence) and cluster-robust Cox (`cluster_col="user", robust=True`) using lifelines.
- Compare SE ratios, flag coefficients with ratio > 20 as possibly numerically unstable.
- Add VIF (Variance Inflation Factor) multicollinearity audit.
- Nelson-Aalen non-parametric cumulative hazard by user activity tier.

Result:

| Task | Mean SE ratio (robust/naive) | Lost significance |
|---|---|---|
| Borrow_Liquidated | 18.05× | 16/18 (89%) |
| Deposit_Borrow | 4.77× | 4/17 (24%) |
| Withdraw_Repay | 59.67× | 13/20 (65%) |
| **Overall** | — | **35/55 (64%)** |

**Pre-reg verdict: STRONG POSITIVE.** Nelson-Aalen plots confirm order-of-magnitude heterogeneity in hazard rates across activity tiers, motivating the cluster-robust framework.

Methodological note (in notebook): The FinSurvival schema (one row per transaction) is "clustered observations," not true recurrent events. Marginal Cox with cluster-robust Huber-White sandwich SEs is the appropriate tool — true Andersen-Gill counting-process construction requires restructuring that this schema does not naturally support.

### nb9_mva.ipynb — Multivariate Analysis (partial positive)

Pre-registered hypotheses H6 (dimensionality reduction), H7 (group separation), H8 (discriminant interpretation).

**Leakage-corrected user aggregation**: for each user, aggregate features only from transactions BEFORE their first liquidation event (if any). This removes circular information where post-liquidation activity patterns are averaged into the group-defining label.

- PCA on 80-feature standardized matrix: 13 PCs needed for 70% variance (H6 **NULL** — stricter than pre-reg bar of 5 PCs > 70%).
- Factor Analysis with varimax rotation (max_iter=5000): 5 interpretable factors (Market Borrow-Repay Volume, User Borrow-Repay Volume, Market Deposit-Withdraw Volume, Transaction Size, Deposit Size vs Borrow Frequency).
- Hotelling's T² test: T² = 800.5, F = 10.0, p ≪ 10⁻³⁰⁰, Mahalanobis distance = 1.01 (H7 **POSITIVE**).
- Linear Discriminant Analysis: top 10 features dominated by user-aggregated borrow/repay sums (no prior-liquidation features in top 10 → H8 NULL per strict rule, but **consistent with Phase 8 finding** that naive methods identify inflated user-aggregated features).

Verdict: **PARTIAL POSITIVE** — group separation confirmed; dimensionality reduction weaker than pre-registered; discriminant pattern consistent with the Phase 8 critique.

### nb10_tsa.ipynb — Time Series point-process analysis (strong positive)

Pre-registered hypotheses H9 (self-excitation), H10 (Granger causality), H11 (non-stationarity).

- Build daily liquidation count series from Borrow_Liquidated events (747 days).
- ACF/PACF plots.
- ADF stationarity test (p < 0.001 → stationary; H11 NULL per direction of pre-reg).
- Hawkes process with exponential kernel: fit via L-BFGS-B maximum likelihood with closed-form intensity recursion and compensator.
  - Background rate μ = 14.55/day
  - Jump size α = 65.26
  - Decay rate β = 126.92/day
  - **Branching ratio η = α/β = 0.514**
  - LR test vs Poisson: statistic = 18,487, p ≈ 0 (H9 **STRONG POSITIVE**)
- Granger causality (lags 1-7): market total events Granger-cause daily liquidations at lags 5, 6, 7 (p < 0.0001 uncorrected; BH-corrected FDR < 0.05, H10 **POSITIVE**).

Verdict: **STRONG POSITIVE** on self-excitation and Granger causality. H11 null (series is stationary) is itself interpretable as "bursts around equilibrium" dynamics characteristic of sub-critical Hawkes processes.

### nb11_game_theory.ipynb — Bayesian signaling game (partial positive)

**Phase 11a — Empirical patterns** (pre/post first-liquidation comparison):
- Log-rank test on Kaplan-Meier survival curves pre vs post first-liquidation: p ≈ 10⁻²⁴⁸, approximate HR 2.26 (**flagged as methodologically fragile** — selection asymmetry in the group construction).
- Spearman(log position size, time-to-liquidation): ρ = +0.034 (direction opposite to naive liquidator-rationality prediction; "paradox" discussed).
- Hawkes branching ratio difference first vs repeat: +0.001 (NULL on strategic recurrence).

**Phase 11b — Formal Bayesian signaling game**:
- Players: Borrower type θ ∈ {safe, risky} with prior q, Liquidator choosing monitoring intensity m ∈ [0,1].
- Payoff: U_L(m, p) = p·α_R·m·π_L + (1-p)·α_S·m·π_L − c·m²/2.
- Perfect Bayesian Equilibrium: m*(p) = π_L/c · (p·α_R + (1-p)·α_S).
- Posterior update after one observed liquidation: q' = q·α_R / (q·α_R + (1-q)·α_S).
- Predicted hazard ratio in equilibrium: HR = m*(q')/m*(q).
- Calibrated to Phase 11a HR = 2.26: q = 0.20, r = α_R/α_S = 7.
- Heterogeneity check across markets (Stable HR=2.44 vs Non-Stable HR=2.11) and activity tiers (low=1.14, mid-low=1.12, mid-high=1.71): cv = 0.25, intermediate — consistent with but not uniquely identifying the game-theoretic mechanism.

**Phase 11c — Proper recurrent-event re-test** of H12:
- Extended cluster-robust Cox with time-dependent `prior_liq_count` covariate.
- Result: HR per additional prior liquidation = **1.031** [1.025, 1.037], p = 6.87 × 10⁻²⁵.
- Phase 11a's dramatic HR 2.26 was substantially inflated by selection asymmetry; true effect is small (3.1% per prior liq) but statistically overwhelming (p = 10⁻²⁵).

Verdict: **PARTIAL POSITIVE** — modest, statistically overwhelming type-persistence effect; model is calibrated and consistent with heterogeneity data; cannot definitively rule out purely mechanical collateral-depletion explanation.

---

## Repository structure
defi_project/
├── README.md
├── requirements.txt
├── .gitignore
├── nb0_setup.ipynb Environment setup
├── nb1_data.ipynb Data extraction and task registry
├── nb2b_pinned_baseline.ipynb XGBoost+Cox baseline (chronological hold)
├── nb3_fuzzy.ipynb IT2 fuzzy regime memberships
├── nb4_augmented.ipynb Baseline + fuzzy marginal features (null)
├── nb6_interactions.ipynb Fuzzy × base feature interactions (null)
├── nb7_full_calib.ipynb Breslow + Brier + prediction intervals
├── nb8_cluster_robust_cox.ipynb Cluster-robust Cox regression (POSITIVE)
├── nb9_mva.ipynb PCA, FA, Hotelling’s T², LDA (PARTIAL)
├── nb10_tsa.ipynb Hawkes, ADF, Granger (POSITIVE)
├── nb11_game_theory.ipynb Bayesian signaling game (PARTIAL)
└── pre_registrations/
├── pre_registrations_phase6.md
└── pre_registrations_phase7.md
(Pre-registrations for phases 8-11 are inline at the top of their notebooks)

Historical note: nb2_baseline.ipynb (initial baseline with random holdout, superseded by nb2b), nb5_calibration.ipynb (preliminary calibration, superseded by nb7), and nb8b_true_ag.ipynb (attempted true Andersen-Gill, withdrawn after discovering the FinSurvival schema does not naturally support counting-process intervals; nb8's cluster-robust Cox is the methodologically correct analysis) are retained locally but not included here.

---

## Reproducing the results

### Prerequisites

- Python 3.10+
- 16 GB RAM minimum (XGBoost training on the larger tasks allocates ~8 GB)
- ~35 GB free disk (24.7 GB train + 6.3 GB test benchmark + working space)

### Setup

```bash
git clone https://github.com/Aadya03/defi-liquidation-survival-analysis.git
cd <repo-name>
pip install -r requirements.txt
```

### Download benchmark data

Obtain `FinSurvival2025_Dataset.zip` from the Codabench challenge page and place it at `data/raw/FinSurvival2025_Dataset.zip`. nb1 extracts and verifies it.

### Run notebooks in numerical order

Approximate runtime on a modern laptop (16 GB RAM, 8-core CPU):

| Notebook | Runtime |
|---|---|
| nb0_setup | <1 min |
| nb1_data | ~5 min (extraction) |
| nb2b_pinned_baseline | ~15 min |
| nb3_fuzzy | ~5 min |
| nb4_augmented | ~20 min |
| nb6_interactions | ~15 min |
| nb7_full_calib | ~30 min |
| nb8_cluster_robust_cox | ~30 min |
| nb9_mva | ~5 min |
| nb10_tsa | ~5 min |
| nb11_game_theory | ~15 min |

Total: ~2.5 hours.

---

## Pre-registration philosophy

Every quantitative hypothesis in this project was pre-registered before execution. Decision rules — including specific numeric thresholds for POSITIVE, NULL, and NEGATIVE verdicts — are locked prior to seeing results and never retroactively modified. Null results are reported with the same transparency as positive findings.

Pre-registrations for Phase 6 (fuzzy interactions) and Phase 7 (calibration) are in `pre_registrations/`. Pre-registrations for Phases 8 through 11 are embedded as markdown cells at the top of their respective notebooks (nb8, nb9, nb10, nb11) so that each notebook is a self-contained record of what was hypothesized and what was found.

Bugs identified during analysis and the corrections applied:
- nb6 initial run: `merge(on="timestamp")` on non-unique keys inflated training rows by 60%. Fix: positional alignment with assertion of equal lengths. Original inflated result (+0.0231) was rejected; corrected result (+0.0031) stands.
- nb6 bootstrap: cross-file row alignment paired labels from one model with risks from another, inflating deltas to +0.36. Fix: safe bootstrap where each model is evaluated on its own labels and risks.
- nb6 and nb10 Granger: statsmodels API change silently failed all tests (NaN p-values). Fix: removed deprecated `verbose` argument.
- nb9 initial run: user aggregation included post-liquidation transactions for liquidated users. Fix: restrict to pre-first-liquidation transactions. Mahalanobis distance dropped from 1.79 to 1.01 after correction; the leakage-corrected result is reported.
- nb11a: pre/post pre/post first-liquidation comparison had selection asymmetry inflating HR to 2.26. Fix: Phase 11c proper recurrent-event framework. True HR per prior liquidation is 1.031, not 2.26.
- nb8 initial labeling: described as "Andersen-Gill" but data structure doesn't support counting-process intervals. Fix: reframed as cluster-robust marginal Cox regression. The methodological finding is unchanged.

---

## Known limitations

1. The FinSurvival benchmark provides transactional features but not position-level collateral ratios, limiting direct testing of some game-theoretic predictions (nb11).
2. The game-theoretic model is calibrated to observed HR rather than independently validated against unseen quantities; a stronger validation would require exogenous variation (e.g., gas price shocks) not available in this schema.
3. The chronological holdout is three months. Longer-horizon out-of-sample validation would strengthen the Hawkes self-excitation finding (nb10).
4. Three Phase 8 tasks (Borrow_Liquidated, Deposit_Borrow, Withdraw_Repay) are representative but not exhaustive. All 16 tasks would strengthen generalizability at substantial computational cost.
5. Fuzzy regime memberships use K=4 clusters chosen a priori. Alternative values (K=3, K=5) were not explored to avoid post-hoc model selection.

---

## Authors

- **Aadya Jha**
- **Nidhi Vernekar**

---

## Acknowledgments

FinSurvival 2025 benchmark credit: original authors of the Codabench challenge.

---

## License

MIT License (see LICENSE file).