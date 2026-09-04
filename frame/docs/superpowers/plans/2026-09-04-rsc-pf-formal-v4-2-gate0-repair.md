# RSC-PF Formal-v4.2 Gate 0 Evidence Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the formal-v4.2 Gate 0 command's impossible empty-receipt path with a self-contained, content-validated evidence pipeline that can authorize the pilot only after genuine v4.2 source, iTransformer, DiffLP, capacity, and runtime checks pass.

**Architecture:** Add a focused Gate 0 receipt module and a v4.2 capacity producer, then make the existing Gate 0 script orchestrate them under one fresh run root. Producers write immutable receipts bound to the current contract, source manifest, and run id; the validator reloads and verifies them before writing either `CURRENT_GATE.json` or a failure receipt.

**Tech Stack:** Python 3.9, PyTorch 2.8, NumPy, pandas, SciPy HiGHS, CVXPY/CVXPYlayers/diffcp/ECOS, Git, JSON/NPZ/SHA-256 receipts, pytest, PowerShell.

## Global Constraints

- Preserve `reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a` unchanged as a diagnostic failed run.
- Do not reuse or copy a formal-v4.1 receipt as formal-v4.2 evidence.
- Generate the source manifest only from the committed, tracked, clean v4.2 source closure at current `HEAD`.
- Capacity fitting may read only 2015-2018; 2019 may be materialized for later gates but may not influence capacity selection.
- Gate 0 must not read, scan, extract, infer, or materialize 2020 or 2021.
- File existence alone is never sufficient: every prerequisite receipt is parsed and content-validated.
- The official iTransformer receipt must revalidate the frozen THUML commit, imported source hashes, and license hash.
- The Differentiable-LP receipt must bind the frozen environment lock and contain a finite nonzero autograd gradient from a real CVXPYlayers solve.
- Gate 0 must return nonzero and write `GATE0_FAILURE.json` on any producer or validation failure.
- Gate 0 may write `authorized_pilot=true` and `protocol/CURRENT_GATE.json` only after all mandatory checks pass.
- Gate 0 never launches the pilot, Gate 1, or any model-training stage.
- The retry uses a fresh run id; use `formal_v4_2_20260904_b` if it remains unused.
- Run all commands from `D:\Paper\github_work\paper-code\frame` with `D:\Paper\envs\rsc_pf_diffopt_v4\python.exe`.

---

## File Structure Map

- Create `frame/src/joint_dispatch/formal_v4_2_gate0.py`: typed prerequisite receipt schemas, producers, validators, and authorization/failure writers.
- Create `frame/scripts/run_rsc_pf_formal_v4_2_capacity_audit.py`: v4.2 train-only capacity/data producer using the existing audited capacity engine.
- Modify `frame/scripts/run_rsc_pf_formal_v4_2_gate0.py`: self-contained orchestration, stable CLI, real-operation probes, failure exit status, and current-gate authorization.
- Modify `frame/configs/formal_v4_source_closure_v4_2.txt`: include the new runtime files and all newly used transitive producer dependencies.
- Modify `frame/tests/test_joint_dispatch_formal_v4_2_gate0.py`: producer, validation, CLI, failure, and authorization tests.
- Create `frame/tests/test_joint_dispatch_formal_v4_2_capacity.py`: train-year capacity boundary and immutable receipt tests.

### Task 1: Add Typed Formal-v4.2 Gate 0 Receipt Contracts

**Files:**
- Create: `frame/src/joint_dispatch/formal_v4_2_gate0.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_2_gate0.py`

**Interfaces:**
- Consumes: `FormalV42Contract`, `write_once_json`, `sha256_file`, and the frozen run id/contract hash.
- Produces: `Gate0ReceiptError`, `Gate0InputsV42`, `Gate0PrerequisitePaths`, `validate_source_manifest_receipt`, `validate_itransformer_receipt`, `validate_diffopt_receipt`, `validate_capacity_receipt`, and `validate_prerequisites`.

- [ ] **Step 1: Write failing schema and lineage tests**

```python
def _write_receipt(path, schema, run_id, contract_sha256, **extra):
    payload = {
        "schema": schema,
        "run_id": run_id,
        "contract_sha256": contract_sha256,
        "evaluation_year_accessed": False,
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path

def test_gate0_receipts_reject_cross_run_and_cross_contract(tmp_path):
    run_id, contract_hash, source_hash = "formal_v4_2_test", "a" * 64, "b" * 64
    source = _write_receipt(
        tmp_path / "SOURCE_MANIFEST.json", "formal-v4.2-source-manifest-v1",
        run_id, contract_hash, git_commit="deadbeef", entry_count=1,
        entries=[{"path": "frame/a.py", "sha256": "c" * 64, "size_bytes": 1}],
        identity_sha256=source_hash,
    )
    itransformer = _write_receipt(
        tmp_path / "ITRANSFORMER_SOURCE_RECEIPT.json", "formal-v4.2-itransformer-source-v1",
        run_id, contract_hash, source_manifest_sha256=source_hash,
        commit=OFFICIAL_COMMIT, repository=OFFICIAL_REPOSITORY,
        reproduction_level="official_backbone_adaptation", verified=True,
    )
    diffopt = _write_receipt(
        tmp_path / "DIFFERENTIABLE_LP_ENVIRONMENT_RECEIPT.json", "formal-v4.2-diffopt-environment-v1",
        run_id, contract_hash, source_manifest_sha256=source_hash,
        eligible=True, finite=True, gradient_norm=1.0, lock_sha256="d" * 64,
    )
    capacity = _write_receipt(
        tmp_path / "CAPACITY_FREEZE.json", "formal-v4.2-capacity-freeze-v1",
        run_id, contract_hash, source_manifest_sha256=source_hash,
        fit_years=[2015, 2016, 2017, 2018], selection_influenced_capacity=False,
        selected={"multiplier": 1.0}, status="pass",
    )
    paths = Gate0PrerequisitePaths(source, itransformer, diffopt, capacity)
    with pytest.raises(Gate0ReceiptError, match="run_id"):
        validate_prerequisites(paths, run_id="formal_v4_2_other", contract_sha256="a" * 64)
    with pytest.raises(Gate0ReceiptError, match="contract_sha256"):
        validate_prerequisites(paths, run_id="formal_v4_2_test", contract_sha256="b" * 64)

def test_gate0_receipts_reject_file_existence_without_valid_content(tmp_path):
    paths = Gate0PrerequisitePaths(*(tmp_path / name for name in (
        "SOURCE_MANIFEST.json", "ITRANSFORMER_SOURCE_RECEIPT.json",
        "DIFFERENTIABLE_LP_ENVIRONMENT_RECEIPT.json", "CAPACITY_FREEZE.json",
    )))
    for path in paths:
        path.write_text("{}", encoding="utf-8")
    with pytest.raises(Gate0ReceiptError):
        validate_prerequisites(paths, run_id="formal_v4_2_test", contract_sha256="a" * 64)
```

- [ ] **Step 2: Run the tests and confirm the new API is absent**

Run: `& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_2_gate0.py -q`

Expected: FAIL because `formal_v4_2_gate0` and its typed receipt validators do not exist.

- [ ] **Step 3: Implement strict receipt datatypes and shared lineage validation**

```python
class Gate0ReceiptError(ValueError):
    pass

@dataclass(frozen=True)
class Gate0InputsV42:
    contract_path: Path
    output_root: Path
    run_id: str
    data_dir: Path
    diffopt_python: Path
    itransformer_source: Path

@dataclass(frozen=True)
class Gate0PrerequisitePaths:
    source_manifest: Path
    itransformer_receipt: Path
    diffopt_receipt: Path
    capacity_receipt: Path

    def __iter__(self):
        return iter((self.source_manifest, self.itransformer_receipt,
                     self.diffopt_receipt, self.capacity_receipt))

def _require_lineage(payload, *, schema, run_id, contract_sha256):
    if payload.get("schema") != schema:
        raise Gate0ReceiptError(f"schema mismatch: {schema}")
    if payload.get("run_id") != run_id:
        raise Gate0ReceiptError("run_id lineage mismatch")
    if payload.get("contract_sha256") != contract_sha256:
        raise Gate0ReceiptError("contract_sha256 lineage mismatch")
    if payload.get("evaluation_year_accessed") is not False:
        raise Gate0ReceiptError("evaluation-year access is forbidden")
```

Each public validator must reload JSON from disk, check its exact v4.2 schema and lineage, and validate receipt-specific scientific fields. `validate_prerequisites` returns a mapping containing each receipt path and SHA-256 only after all four validators pass.

- [ ] **Step 4: Run focused tests**

Run: `& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_2_gate0.py -q`

Expected: PASS for schema, malformed JSON, cross-run, cross-contract, and evaluation-access denial cases.

- [ ] **Step 5: Commit the receipt boundary**

```powershell
git add src/joint_dispatch/formal_v4_2_gate0.py tests/test_joint_dispatch_formal_v4_2_gate0.py
git commit -m "feat: validate formal v4.2 gate0 receipts"
```

### Task 2: Produce the Source, iTransformer, and DiffLP Receipts

**Files:**
- Modify: `frame/src/joint_dispatch/formal_v4_2_gate0.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_2_gate0.py`

**Interfaces:**
- Consumes: repository root, source closure, current Git `HEAD`, official iTransformer source root, frozen commit constant, and `requirements/formal_v4_diffopt.lock`.
- Produces: `produce_source_manifest(repo_root, closure_path, destination, *, run_id, contract_sha256) -> Mapping[str, Any]`, `produce_itransformer_receipt(source_root, destination, *, run_id, contract_sha256, source_manifest_sha256) -> Mapping[str, Any]`, and `produce_diffopt_receipt(lock_path, destination, *, run_id, contract_sha256, source_manifest_sha256) -> Mapping[str, Any]`.

- [ ] **Step 1: Write failing producer tests**

```python
def test_source_manifest_hashes_only_clean_tracked_closure(repo_fixture, tmp_path):
    receipt = produce_source_manifest(
        repo_fixture.root, repo_fixture.closure, tmp_path / "SOURCE_MANIFEST.json",
        run_id="formal_v4_2_test", contract_sha256="a" * 64,
    )
    assert receipt["git_commit"] == repo_fixture.head
    assert receipt["entry_count"] == 2
    assert receipt["evaluation_year_accessed"] is False

def test_diffopt_receipt_requires_real_nonzero_gradient(tmp_path, monkeypatch):
    monkeypatch.setattr(gate0_module, "run_native_diffopt_probe", lambda: {
        "finite": True, "gradient_norm": 0.0, "solver": "SCS"
    })
    with pytest.raises(Gate0ReceiptError, match="gradient"):
        produce_diffopt_receipt(
            LOCK, tmp_path / "DIFF.json", run_id="formal_v4_2_test",
            contract_sha256="a" * 64, source_manifest_sha256="b" * 64,
        )
```

Also cover a dirty closure entry, wrong iTransformer commit, changed imported file hash, interpreter mismatch, and changed environment-lock hash.

- [ ] **Step 2: Run the new tests and verify producer failures**

Run: `& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_2_gate0.py -q`

Expected: FAIL because the producer functions are absent.

- [ ] **Step 3: Implement the source manifest producer**

Use `build_source_manifest(repo_root, entries, head)` for tracked-clean verification, then wrap its payload in the v4.2 envelope:

```python
payload = {
    "schema": "formal-v4.2-source-manifest-v1",
    "run_id": run_id,
    "contract_sha256": contract_sha256,
    "git_commit": head,
    "entry_count": len(manifest.entries),
    "entries": [entry.to_payload() for entry in manifest.entries],
    "evaluation_year_accessed": False,
}
payload["identity_sha256"] = canonical_sha256(payload)
write_once_json(destination, payload)
```

- [ ] **Step 4: Implement the official iTransformer producer**

Require `git -C <source_root> rev-parse HEAD == OFFICIAL_COMMIT`, hash every imported Python file and the license, call the existing source verifier, then write a v4.2 envelope containing `repository`, `commit`, `backbone_class`, `reproduction_level: official_backbone_adaptation`, all file hashes, and the source-manifest hash.

- [ ] **Step 5: Implement the isolated DiffLP producer**

Require current `sys.executable` to match the interpreter stored in the lock. Rehash the lock and requirements input, import every frozen dependency, construct a one-variable DPP-compliant `CvxpyLayer`, execute a float64 solve and backward pass, and require a finite primal value plus finite nonzero gradient before writing the receipt.

- [ ] **Step 6: Run focused and existing upstream adapter tests**

Run:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest `
  tests/test_joint_dispatch_formal_v4_2_gate0.py `
  tests/test_joint_dispatch_formal_v4_itransformer.py `
  tests/test_joint_dispatch_formal_v4_diffopt.py `
  tests/test_joint_dispatch_formal_v4_diffopt_runner.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the dependency producers**

```powershell
git add src/joint_dispatch/formal_v4_2_gate0.py tests/test_joint_dispatch_formal_v4_2_gate0.py
git commit -m "feat: produce formal v4.2 gate0 dependency receipts"
```

### Task 3: Build a Formal-v4.2 Train-Only Capacity Producer

**Files:**
- Create: `frame/scripts/run_rsc_pf_formal_v4_2_capacity_audit.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_capacity.py`

**Interfaces:**
- Consumes: `FormalV42Contract`, Kitakyushu data directory, v4 benchmark rules and parameter ledger, source-manifest SHA-256, and an existing fresh v4.2 run root.
- Produces: `CapacityEvidenceError`, `capacity_years_for_split(contract, split) -> tuple[int, ...]`, `build_capacity_evidence(contract, run_root, data_dir, source_manifest_sha256) -> Mapping[str, Any]`, `build_capacity_evidence_from_bases(train_base, selection_base, parameters, contract, run_root, source_manifest_sha256, solver=solve_dispatch_lp) -> Mapping[str, Any]`, `gate0/CAPACITY_AUDIT_RESULT.json`, `gate0/CAPACITY_FREEZE.json`, train/selection base archives, and data-access receipts.

- [ ] **Step 1: Write failing train-boundary and receipt tests**

```python
def test_capacity_year_mapping_denies_evaluation():
    contract = load_formal_v4_2_contract(CONFIG)
    assert capacity_years_for_split(contract, "train") == (2015, 2016, 2017, 2018)
    assert capacity_years_for_split(contract, "selection") == (2019,)
    with pytest.raises(CapacityEvidenceError, match="forbidden"):
        capacity_years_for_split(contract, "evaluation")

def test_capacity_builder_uses_train_base_only(tmp_path, monkeypatch, train_base, selection_base, passing_audit):
    observed = []
    def fake_audit(base, parameters, manifest, candidates, **kwargs):
        observed.append(base.split)
        return passing_audit
    monkeypatch.setattr(capacity_module, "run_capacity_audit", fake_audit)
    result = build_capacity_evidence_from_bases(
        train_base, selection_base, PARAMETERS, load_formal_v4_2_contract(CONFIG),
        tmp_path, "b" * 64, solver=lambda inputs: None,
    )
    assert observed == ["train"]
    assert result["fit_years"] == [2015, 2016, 2017, 2018]
    assert result["selection_influenced_capacity"] is False
    assert result["evaluation_year_accessed"] is False

def test_capacity_freeze_requires_both_registered_checks(tmp_path, train_base, selection_base, failing_audit, monkeypatch):
    monkeypatch.setattr(capacity_module, "run_capacity_audit", lambda *a, **k: failing_audit)
    with pytest.raises(CapacityEvidenceError, match="chronological"):
        build_capacity_evidence_from_bases(
            train_base, selection_base, PARAMETERS, load_formal_v4_2_contract(CONFIG),
            tmp_path, "b" * 64, solver=lambda inputs: None,
        )
    assert not (tmp_path / "gate0" / "CAPACITY_FREEZE.json").exists()
```

Define `train_base` and `selection_base` with deterministic hourly arrays using `FormalV4BaseSeries`. Define `passing_audit` and `failing_audit` as complete `CapacityAuditReceipt` objects: the passing object has the same multiplier passing both stage rows; the failing object has `status="fail"`, `selected=None`, and a chronological row with `meets_threshold=False`.

- [ ] **Step 2: Run tests and establish the missing producer**

Run: `& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_2_capacity.py -q`

Expected: FAIL because the v4.2 capacity entry point is absent.

- [ ] **Step 3: Implement controlled train and selection reads**

Reuse the canonical Kitakyushu reader and access callbacks. Hard-code the permitted split mapping from the v4.2 contract:

```python
years = {
    "train": tuple(contract.train_years),
    "selection": (contract.selection_year,),
}
if split not in years:
    raise CapacityEvidenceError(f"forbidden capacity split: {split}")
```

Materialize `base_train.npz` and `base_selection.npz`, but pass only the train base into `select_capacity_origins` and `run_capacity_audit`.

- [ ] **Step 4: Run the frozen two-stage capacity audit**

Use the candidate multipliers and thresholds directly from `contract.capacity`. Require 500 registered train origins, a passing stratified diagnostic row, and a passing full chronological row for the same selected multiplier. Do not write the freeze on failure.

- [ ] **Step 5: Write the v4.2 capacity and access receipts**

The capacity freeze must contain:

```python
{
    "schema": "formal-v4.2-capacity-freeze-v1",
    "run_id": run_root.name,
    "contract_sha256": contract.contract_sha256,
    "source_manifest_sha256": source_manifest_sha256,
    "fit_years": [2015, 2016, 2017, 2018],
    "selection_influenced_capacity": False,
    "evaluation_year_accessed": False,
    "selected": audit.selected,
    "candidate_multipliers": list(audit.candidate_multipliers),
    "thresholds": dict(audit.thresholds),
    "capacity_scenario_hash": audit.capacity_scenario_hash,
}
```

Write immutable data/archive access receipts demonstrating that no 2020/2021 member was opened.

- [ ] **Step 6: Run focused and existing capacity tests**

Run:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest `
  tests/test_joint_dispatch_formal_v4_2_capacity.py `
  tests/test_joint_dispatch_formal_v4_capacity.py `
  tests/test_joint_dispatch_formal_v4_access.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the capacity producer**

```powershell
git add scripts/run_rsc_pf_formal_v4_2_capacity_audit.py tests/test_joint_dispatch_formal_v4_2_capacity.py
git commit -m "feat: certify formal v4.2 train-only capacity"
```

### Task 4: Orchestrate the Self-Contained Gate 0 Command

**Files:**
- Modify: `frame/scripts/run_rsc_pf_formal_v4_2_gate0.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_2_gate0.py`

**Interfaces:**
- Consumes: producer functions from Tasks 2-3, the real probe functions already present in the script, and a fresh run id.
- Produces: `run_gate0_orchestrator(inputs: Gate0InputsV42) -> Mapping[str, Any]`, `GATE0_EVIDENCE.json`, `RESOURCE_PROJECTION.json`, `GATE0_FAILURE.json`, and `CURRENT_GATE.json`.

- [ ] **Step 1: Write failing end-to-end orchestration tests**

```python
def _inputs(tmp_path):
    return Gate0InputsV42(
        contract_path=CONFIG,
        output_root=tmp_path,
        run_id="formal_v4_2_gate0_test",
        data_dir=tmp_path / "data",
        diffopt_python=Path(sys.executable),
        itransformer_source=tmp_path / "iTransformer",
    )

def install_valid_producer_fakes(monkeypatch, inputs):
    contract = load_formal_v4_2_contract(inputs.contract_path)
    source_hash = "b" * 64
    def source(*args, **kwargs):
        path = inputs.output_root / inputs.run_id / "protocol" / "SOURCE_MANIFEST.json"
        return _write_receipt(
            path, "formal-v4.2-source-manifest-v1", inputs.run_id,
            contract.contract_sha256, git_commit="deadbeef", entry_count=1,
            entries=[{"path": "frame/a.py", "sha256": "c" * 64, "size_bytes": 1}],
            identity_sha256=source_hash,
        )
    def itransformer(*args, **kwargs):
        path = inputs.output_root / inputs.run_id / "protocol" / "ITRANSFORMER_SOURCE_RECEIPT.json"
        return _write_receipt(
            path, "formal-v4.2-itransformer-source-v1", inputs.run_id,
            contract.contract_sha256, source_manifest_sha256=source_hash,
            repository=OFFICIAL_REPOSITORY, commit=OFFICIAL_COMMIT,
            reproduction_level="official_backbone_adaptation", verified=True,
        )
    def diffopt(*args, **kwargs):
        path = inputs.output_root / inputs.run_id / "protocol" / "DIFFERENTIABLE_LP_ENVIRONMENT_RECEIPT.json"
        return _write_receipt(
            path, "formal-v4.2-diffopt-environment-v1", inputs.run_id,
            contract.contract_sha256, source_manifest_sha256=source_hash,
            eligible=True, finite=True, gradient_norm=1.0, lock_sha256="d" * 64,
        )
    def capacity(*args, **kwargs):
        path = inputs.output_root / inputs.run_id / "gate0" / "CAPACITY_FREEZE.json"
        return _write_receipt(
            path, "formal-v4.2-capacity-freeze-v1", inputs.run_id,
            contract.contract_sha256, source_manifest_sha256=source_hash,
            fit_years=[2015, 2016, 2017, 2018], selection_influenced_capacity=False,
            selected={"multiplier": 1.0}, status="pass",
        )
    monkeypatch.setattr(gate0_script, "produce_source_manifest", source)
    monkeypatch.setattr(gate0_script, "produce_itransformer_receipt", itransformer)
    monkeypatch.setattr(gate0_script, "produce_diffopt_receipt", diffopt)
    monkeypatch.setattr(gate0_script, "build_capacity_evidence", capacity)

def test_gate0_cli_produces_receipts_without_manual_receipt_paths(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    install_valid_producer_fakes(monkeypatch, inputs)
    receipt = run_gate0_orchestrator(inputs)
    root = tmp_path / inputs.run_id
    assert receipt["authorized_pilot"] is True
    assert (root / "protocol" / "SOURCE_MANIFEST.json").is_file()
    assert (root / "protocol" / "ITRANSFORMER_SOURCE_RECEIPT.json").is_file()
    assert (root / "protocol" / "DIFFERENTIABLE_LP_ENVIRONMENT_RECEIPT.json").is_file()
    assert (root / "gate0" / "CAPACITY_FREEZE.json").is_file()
    current = json.loads((root / "protocol" / "CURRENT_GATE.json").read_text())
    assert current["next_gate"] == "pilot"

def test_gate0_failure_is_non_authorizing_and_does_not_launch_pilot(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    install_valid_producer_fakes(monkeypatch, inputs)
    monkeypatch.setattr(gate0_script, "produce_diffopt_receipt", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("diffopt failed")))
    with pytest.raises(Gate0ExecutionError):
        run_gate0_orchestrator(inputs)
    root = tmp_path / inputs.run_id
    assert (root / "gate0" / "GATE0_FAILURE.json").is_file()
    assert not (root / "protocol" / "CURRENT_GATE.json").exists()
    assert not (root / "pilot").exists()
```

The helper monkeypatches only the expensive producers; it does not bypass `validate_prerequisites`, so the orchestration tests still exercise disk reload, schema checks, lineage checks, and authorization logic.

- [ ] **Step 2: Run orchestration tests and verify failure**

Run: `& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_2_gate0.py -q`

Expected: FAIL because the existing CLI does not orchestrate prerequisites.

- [ ] **Step 3: Add the stable CLI and typed inputs**

```python
parser.add_argument("--contract", type=Path, default=FRAME_ROOT / "configs" / "joint_forecast_dispatch_formal_v4_2.json")
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--run-id", required=True)
parser.add_argument("--data-dir", type=Path, default=Path(r"D:\Paper\Kitakyushu dataset"))
parser.add_argument("--diffopt-python", type=Path, default=Path(r"D:\Paper\envs\rsc_pf_diffopt_v4\python.exe"))
parser.add_argument("--itransformer-source", type=Path, default=FRAME_ROOT / "third_party" / "iTransformer_source")
```

The public CLI must not expose the four receipt paths; it generates them beneath its own run root.

- [ ] **Step 4: Implement ordered prerequisite production and validation**

Create the run root and source manifest first, then iTransformer and capacity receipts, then DiffLP receipt after the train archive exists. Reload and validate all four receipts before the real-operation measurement loop.

- [ ] **Step 5: Replace boolean path checks with validated evidence**

Remove the current `bool(path) and Path(path).is_file()` authorization logic. Populate `checks` from the typed validators, including path, schema, SHA-256, and scientific summary. `required_components` is true only when all receipt and real-operation checks report `passed=true`.

- [ ] **Step 6: Implement success and failure transitions**

On success, write:

```python
evidence_sha256 = write_once_json(gate_root / "GATE0_EVIDENCE.json", evidence)
write_once_json(root / "protocol" / "CURRENT_GATE.json", {
    "schema": "formal-v4.2-current-gate-v1",
    "run_id": run_id,
    "contract_sha256": contract.contract_sha256,
    "gate": "gate0",
    "gate_evidence_sha256": evidence_sha256,
    "authorized_pilot": True,
    "next_gate": "pilot",
})
```

On failure, write `GATE0_FAILURE.json`, omit `CURRENT_GATE.json`, return exit code 2, and leave all partial receipts untouched.

- [ ] **Step 7: Run focused Gate 0 tests**

Run: `& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_2_gate0.py tests/test_joint_dispatch_formal_v4_2_capacity.py -q`

Expected: PASS.

- [ ] **Step 8: Commit the orchestrator repair**

```powershell
git add scripts/run_rsc_pf_formal_v4_2_gate0.py tests/test_joint_dispatch_formal_v4_2_gate0.py
git commit -m "fix: make formal v4.2 gate0 self contained"
```

### Task 5: Close the Runtime Source Manifest and Run Full Regression

**Files:**
- Modify: `frame/configs/formal_v4_source_closure_v4_2.txt`

**Interfaces:**
- Consumes: final Gate 0 runtime import graph and all Task 1-4 commits.
- Produces: a complete, tracked, clean runtime closure at the final repair `HEAD`.

- [ ] **Step 1: Add all new and newly invoked runtime paths to the closure**

At minimum add the new Gate 0 module, new capacity script and its directly imported runtime producers. Include every transitive project file imported for raw Kitakyushu reading, access logging, benchmark construction, capacity certification, official iTransformer verification, and DiffLP verification.

- [ ] **Step 2: Verify existence, tracking, and clean status**

Run:

```powershell
$rows = Get-Content configs/formal_v4_source_closure_v4_2.txt |
  Where-Object { $_.Trim() -and -not $_.Trim().StartsWith('#') }
$missing = @($rows | Where-Object { -not (Test-Path (Join-Path '..' $_)) })
$untracked = @($rows | Where-Object { -not (git -C .. ls-files --error-unmatch -- $_ 2>$null) })
if ($missing.Count -or $untracked.Count) { throw "source closure incomplete" }
```

Expected: no missing or untracked entries.

- [ ] **Step 3: Run formal-v4.2, v4 compatibility, and DiffLP regression suites**

Run:

```powershell
$tests = @(Get-ChildItem tests -Filter 'test_joint_dispatch_formal_v4_2_*.py' | ForEach-Object FullName)
$tests += @(Get-ChildItem tests -Filter 'test_joint_dispatch_formal_v4_*.py' | ForEach-Object FullName)
$tests += @((Resolve-Path tests/test_dispatch_lp.py).Path, (Resolve-Path tests/test_scheme2r.py).Path)
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest @tests `
  --basetemp D:\Paper\pytest_tmp_formal_v42_gate0_repair -p no:cacheprovider -q
```

Expected: PASS.

- [ ] **Step 4: Commit the final closure**

```powershell
git add configs/formal_v4_source_closure_v4_2.txt
git commit -m "chore: close formal v4.2 gate0 sources"
```

- [ ] **Step 5: Recheck the closure after the final commit**

Run:

```powershell
$rows = Get-Content configs/formal_v4_source_closure_v4_2.txt |
  Where-Object { $_.Trim() -and -not $_.Trim().StartsWith('#') }
$dirty = @(git -C .. status --porcelain -- $rows)
if ($dirty.Count) { throw $dirty }
```

Expected: no output.

### Task 6: Run a Fresh Formal-v4.2 Gate 0 and Inspect Authorization

**Files:**
- Generated only: `frame/reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_b/**`

**Interfaces:**
- Consumes: committed clean source closure, frozen v4.2 contract, real Kitakyushu data, isolated DiffLP interpreter, and frozen iTransformer checkout.
- Produces: one immutable Gate 0 result for run id `formal_v4_2_20260904_b`.

- [ ] **Step 1: Confirm the retry run id is unused**

Run:

```powershell
$runRoot = 'reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_b'
if (Test-Path $runRoot) { throw "fresh run id required: $runRoot" }
```

Expected: no output.

- [ ] **Step 2: Execute the self-contained Gate 0 command**

Run:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' `
  scripts/run_rsc_pf_formal_v4_2_gate0.py `
  --contract configs/joint_forecast_dispatch_formal_v4_2.json `
  --output-root reports/joint_forecast_dispatch_formal_v4_2 `
  --run-id formal_v4_2_20260904_b `
  --data-dir 'D:\Paper\Kitakyushu dataset' `
  --diffopt-python 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' `
  --itransformer-source third_party/iTransformer_source
```

Expected: exit 0 only if all real prerequisites and probes pass. Otherwise exit 2 with an immutable failure receipt.

- [ ] **Step 3: Inspect the formal evidence without launching later gates**

Run:

```powershell
$root = 'reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_b'
$evidence = Get-Content "$root/gate0/GATE0_EVIDENCE.json" -Raw | ConvertFrom-Json
if (-not $evidence.authorized_pilot) { Get-Content "$root/gate0/GATE0_FAILURE.json"; throw "Gate 0 denied" }
$current = Get-Content "$root/protocol/CURRENT_GATE.json" -Raw | ConvertFrom-Json
if ($current.next_gate -ne 'pilot') { throw "invalid Gate 0 transition" }
$evidence.checks | ConvertTo-Json -Depth 8
```

Expected: `authorized_pilot=true`, `evaluation_year_accessed=false`, every mandatory check passes, and `next_gate=pilot`.

- [ ] **Step 4: Stop at the repaired Gate 0 boundary**

Do not launch the pilot or Gate 1 in this repair plan. Report the run id, authorization, selected capacity multiplier, source-manifest hash, dependency identities, real probe timings, and any non-fatal warnings.
