# Interrupted Gate 1 Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Seal the abruptly interrupted Gate 1 `_d` run and resume it without losing completed results.

**Architecture:** A focused sealing function validates the run's immutable lineage and absence of completion/evaluation receipts, then writes one failure receipt. The existing recovery inspector and real Gate 1 runner perform all subsequent row validation and continuation.

**Tech Stack:** Python 3, pathlib, JSON SHA-256 receipts, pytest, existing Gate 1 recovery modules.

## Global Constraints

- Never overwrite an existing run or artifact.
- Never read evaluation-year data while sealing or recovering.
- Reuse only hash-validated checkpoints and receipts.
- Use a new destination run id for continuation.

---

### Task 1: Add interrupted-run sealing utility

**Files:**
- Create: `src/joint_dispatch/complete_formal_gate1_seal.py`
- Create: `scripts/run_rsc_pf_complete_formal_gate1_seal.py`
- Test: `tests/test_complete_formal_gate1_seal.py`

**Interfaces:**
- `seal_interrupted_gate1_run(run_root: str | Path, *, reason: str) -> Path`
- CLI arguments: `--run-root` and optional `--reason`.

- [ ] **Step 1: Write tests for refusal and successful sealing.**
- [ ] **Step 2: Implement immutable checks and failure-receipt writing.**
- [ ] **Step 3: Run the focused seal tests.**

### Task 2: Seal `_d` and validate recovery lineage

**Files:**
- Modify: `reports/rsc_pf_complete_formal/complete_formal_gate1_repaired_20260908_d/gate1/GATE1_FAILURE.json`

- [ ] **Step 1: Run the sealing CLI against `_d` after confirming no Python process.**
- [ ] **Step 2: Run `inspect_recovery_source` and verify all existing rows.**
- [ ] **Step 3: Record the sealed receipt hash and reusable-row count.**

### Task 3: Start continuation in a new immutable run

**Files:**
- Create: `reports/rsc_pf_complete_formal/complete_formal_gate1_repaired_20260908_e/`

- [ ] **Step 1: Launch the existing real Gate 1 runner with `_d` as `--resume-from`.**
- [ ] **Step 2: Confirm the new run reuses completed rows before long training continues.**
- [ ] **Step 3: Leave the process running until Gate 1 evidence and transition are written.**
