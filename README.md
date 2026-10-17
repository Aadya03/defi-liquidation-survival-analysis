
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