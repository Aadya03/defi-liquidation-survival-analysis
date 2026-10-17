# Phase 7 Pre-Registration: Fuzzy Calibration & Uncertainty

**Date committed:** 2026-10-17
**Status:** Locked prior to running nb7_calibration.ipynb

## Hypothesis H4

Fuzzy memberships (particularly interval_width) provide calibrated
uncertainty quantification even when discrimination (C-index) is null.

## Primary metrics

1. **Integrated Brier Score (IBS)** over 7-90 day horizon window
   - Lower is better
2. **Calibration slope** from predicted vs observed survival
   - Closer to 1.0 is better
3. **Prediction interval coverage** at nominal 90%
   - Closer to 0.90 is better
4. **Uncertainty-error correlation**: Spearman(interval_width, Brier residual)
   - Positive and significant means interval_width is informative

## Models compared

- Baseline (nb2b_pinned): 80 features
- Augmented (nb4): 80 + 5 fuzzy

(Interactions model NOT included — ruled null in Phase 6)

## Decision rule (locked)

- POSITIVE: aug IBS < base IBS by at least 0.005 on at least 10/16 tasks
  AND coverage(aug) is closer to 90% than coverage(base) on majority
  AND Spearman(interval_width, residual) > 0 with p < 0.05 on majority

- NEGATIVE: aug IBS > base IBS on at least 10/16 tasks

- NULL: anything else