# Formal-v4.2 Four-Hour Settlement Residual Fix

## Context

Formal-v4.2 forecasts electricity, cooling, heating, and station-side gas consumption. Only electricity, cooling, and heating are rigid terminal demands. Gas remains a fourth forecast target and an auxiliary scheduling prior; realized gas use is derived from CHP and boiler dispatch and contributes to cost and carbon, but it is not a fourth demand-balance constraint.

Gate 1 run `formal_v4_2_20260905_i` correctly preserved that boundary but rejected every candidate because the four-hour settlement recomputed the electricity residual from the 21-column realized dispatch without subtracting settlement-only electric dump `p_dump`. The canonical one-step settlement already includes `p_dump` in its balance residual, so the recomputation turned valid electric surplus into a false physical violation.

## Considered Approaches

1. **Aggregate canonical one-step residuals (selected).** Reuse each `FormalV4RealizedOutcome.balance_residuals` and `conversion_residuals`, and retain the explicit four-hour SOC recurrence check. This keeps one source of truth and includes settlement-only quantities such as `p_dump`.
2. Reconstruct `p_dump` while recomputing from the 21-column dispatch. This is brittle because `p_dump` is intentionally not a learned dispatch column and would duplicate recourse logic.
3. Remove or relax the physical-residual eligibility gate. This would hide a validator defect and weaken the formal protocol, so it is rejected.

## Design

- `settle_formal_v4_four_hour` continues accepting demand with shape `[B,4,3]`; no gas balance is added.
- During its four-step rollout, it stores the canonical one-step realized outcomes.
- The four-hour constraint penalty is computed from:
  - stacked one-step electricity/cooling/heating balance residuals;
  - stacked one-step device-conversion residuals;
  - the explicit SOC recurrence residual across the four steps.
- Shortage remains a separate reporting and decision-loss quantity. It is not treated as a physical residual because the explicit slack variables make the post-recourse balances exact.
- Cost and carbon continue using realized `g_chp + g_gb`.

## Tests

- Add a regression case with deliberate electric surplus so `p_dump > 0` and assert the post-settlement constraint penalty is approximately zero.
- Assert the same case remains finite and records the surplus through the canonical one-step outcome.
- Retain the Gate 1 regression proving four forecast targets are preserved while only the first three enter rigid settlement.
- Run the focused objective/Gate 1 tests and the complete formal-v4.2 suite.

## Gate Execution

After tests and source commit, use a fresh run ID and execute Gate 0, Pilot, and Gate 1 in order. Continue to Gate 2 only when the immutable Gate 1 receipt contains `authorized_gate2=true`. Stop at the first failed gate. Do not run Gate 3, ablations, manuscript generation, or access the 2020 evaluation split outside an authorized Gate 2 execution.

## Success Criteria

- A valid settled trajectory with nonzero `p_dump` has post-recourse physical residual at numerical tolerance.
- Gate 1 candidate eligibility is no longer blocked by the omitted-`p_dump` accounting defect.
- Four-task forecast metrics, three-demand physical settlement, and gas cost/carbon accounting remain unchanged.
