# Stage 7.3–7.4 sequential orchestration

## Goal

Provide one unattended entry point that runs the existing Stage 7.3 formal
experiment first and starts Stage 7.4 only after Stage 7.3 has completed
successfully and its root manifest passes validation.

## Scope and non-goals

- The orchestrator does not change either training implementation.
- Stage 7.4 receives only the frozen contract and freeze configuration; it does
  not receive or read Stage 7.3 prediction files.
- Existing stage output directories remain separate and are never overwritten
  implicitly.

## Interface

`frame/scripts/run_stage7_3_then_7_4.py` accepts the shared data, contract and
freeze paths, independent Stage 7.3/7.4 output directories, an orchestration
directory, `--dry-run`, and independent `--resume-7-3`/`--resume-7-4` flags.

In normal mode it invokes the two existing scripts as child processes with the
same Python interpreter, streams their output to the console and an
orchestration log, and writes a summary manifest with command lines, exit
codes, timestamps and validation results.

## Control flow and failure handling

1. Run Stage 7.3.
2. Require a zero exit code and a passed Stage 7.3 root manifest with zero
   failed runs and all expected runs completed.
3. Only then run Stage 7.4.
4. If either process or manifest validation fails, stop and return a non-zero
   exit code. Stage 7.4 is never started after a Stage 7.3 failure.

`--dry-run` invokes both existing dry-run modes and performs no data access or
training. Resume flags are passed only to their corresponding stage.

## Verification

Unit tests cover command construction, the independent output paths, manifest
validation, and the rule that a failed Stage 7.3 prevents Stage 7.4 from being
scheduled. A dry-run with the current Stage 7-R contract must report 20 Stage
7.3 runs followed by 50 Stage 7.4 runs.
