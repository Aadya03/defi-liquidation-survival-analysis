# Phase 6 Pre-Registration: Fuzzy-Base Feature Interactions

**Date committed:** 2026-10-15
**Author:** Aadya Jha
**Status:** Locked prior to running nb6_interactions.ipynb

## Hypothesis H3

Fuzzy regime memberships, when combined multiplicatively with base
transactional features (interactions), will improve C-index on the
FinSurvival 2025 chronological holdout over the augmented model
(90 base + 5 fuzzy memberships concatenated).

Rationale: XGBoost already captures marginal effects of both base and
fuzzy features in Phase 4. If regime-dependence is real, it should
manifest as interactions — e.g., the effect of transaction amount on
liquidation hazard differs between Safe and Critical market regimes.

## Interaction features (12, pre-committed)

Four base features chosen a priori from standard DeFi risk literature:
amountUSD_log, userLiquidationCount, marketLiquidationCount,
userBorrowCount.

Multiplied with three uncertainty/stress signals: mu_Critical, mu_Danger,
interval_width.

Total: 4 × 3 = 12 interaction features.

Full feature set: 90 base + 5 fuzzy memberships + 12 interactions = 107.

## Model

XGBoost objective survival:cox. Identical hyperparameters to Phase 4:
learning_rate 0.05, max_depth 6, subsample 0.8, colsample_bytree 0.8,
min_child_weight 10, reg_lambda 1.0, tree_method hist, 500 boosting
rounds with early stopping on 10% random fit-window validation split.

## Evaluation

- Same chronological holdout as Phase 2b/4: 2024-04-01 to 2024-07-02.
- Primary metric: Harrell's C-index on holdout, per task.
- Primary comparison: delta vs augmented model (not vs baseline).
- Paired block bootstrap for CI: N_BOOT=100, BLOCK_SIZE=10,
  MAX_BOOT_N=3000.

## Decision rule (locked)

- POSITIVE: wins > losses AND mean delta across tasks > +0.005
- NULL:     any other combination
- NEGATIVE: losses > wins AND mean delta across tasks < -0.005

The +0.005 threshold reflects that Phase 4 showed +0.003 effects are
not reliable; a smaller effect does not warrant a positive claim.

## No-peek commitment

Decision rule above will not be changed after seeing results.
If POSITIVE: fuzzy interactions become a reported contribution.
If NULL: both discrimination hypotheses (H2, H3) are rejected;
         paper leans on reliability + game theory chapters.
If NEGATIVE: interactions actively hurt; reported as negative result.