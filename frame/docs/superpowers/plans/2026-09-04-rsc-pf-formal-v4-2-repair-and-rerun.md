# RSC-PF Formal-v4.2 Repair and Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 formal-v4.1 的训练、教师、滚动结算、基线、资源评估和门控缺陷，建立一条可审计的 formal-v4.2 实验链，并且只在 Gate 2 独立审计授权后执行锁定的 2020 Gate 3 主实验。

**Architecture:** 保留现有 RSC-PF 的四任务预测、15 维连续调度表示和 21 维物理解码输出，但把实验控制面升级为版本化协议、不可变产物、阶段化训练和统一的逐小时闭环评估。每个 seed 完整执行 Stage P → 同信息教师 → Stage S → 字节一致分支克隆 → Stage J；所有可部署方法在同一因果信息集与同一物理结算器下比较，Perfect-Information-MPC 只作为不可部署参照。

**Tech Stack:** Python 3.9、PyTorch 2.8、NumPy 2.0、SciPy 1.13、pandas、pytest、SciPy HiGHS、官方 THUML iTransformer 源码、独立 CVXPY/CVXPYlayers 环境、JSON/JSONL/CSV/NPZ/PT/SHA-256 收据、PowerShell。

## Global Constraints

- formal-v4.1 的现有代码输出与 `reports/joint_forecast_dispatch_formal_v4_1` 产物只读保留；`formal_v4_1_gate2_20260904_104000` 登记为 `invalid_diagnostic`，不得进入论文结果。
- formal-v4.2 使用唯一 run id 和单一 run root；Gate 0--Gate 3 的每项产物都携带父收据和源文件 SHA-256。
- 训练只允许 2015--2018；模型、超参数、检查点和容量选择只允许 2019；Gate 2 授权前禁止读取、解压、扫描或推断 2020；2021 永久排除。
- 输入历史长度固定为 24 小时，预测与规划时域固定为 4 小时。
- 预测任务顺序固定为 `electricity, cooling, heating, gas`；前三项是刚性物理需求，gas 仅是标准化站侧聚合燃气使用先验，绝不进入终端燃气需求平衡。
- PV/WT 规划输入固定使用上一小时可用出力持久性预测；未来真实 PV/WT 只用于监督标签、已实现结算和 Perfect-Information-MPC。
- 归一化统计只拟合 2015--2018，2019 与 2020 只能加载同一份冻结收据。
- Stage P、Stage S、Stage J 严格串行；Stage J 的 RSC-PF 与 Decoupled-RSC-PF 从同一 Stage S 文件逐字节克隆，唯一设计差异是调度损失能否更新预测器。
- 每个训练阶段在整个阶段内保持一个持久 optimizer；检查点必须包含模型、optimizer、epoch、早停状态、RNG 状态、协议/数据/源代码哈希。
- 滚动评估每个 origin 只执行规划的第一小时，以真实负荷与真实 PV/WT 做一次物理结算，再推进 SOC、上一小时 CHP 和 24 小时历史。
- 物理违规指标必须从数组计算；缺失指标、非有限数、默认零占位、把 shortage 当 balance residual、把 PV curtailment 当 `p_dump` 均为硬失败。
- Gate 1 使用预注册的 2019 季节/活跃度 origin manifest；接近全零的冷/热样本不得放行。
- Gate 2 固定运行 stochastic methods 的 seeds `2026, 2027, 2028` 和 deterministic references 一次；训练 epoch、patience、候选值与验证频率只能来自冻结 contract。
- Gate 3 固定运行 seeds `2026, 2027, 2028, 2029, 2030`；只加载 Gate 2 冻结方案，不允许在 2020 训练、调参、早停或重选 checkpoint。
- 主统计对比是 seed-matched `RSC-PF - Decoupled-RSC-PF`；Gate 3 用连续 168 小时块、2,000 次 bootstrap，五个 seed 才是独立模型重复。
- 价格/碳条件响应不是核心主张，不新增该主张；消融实验、论文改写和 2021 数据不属于本计划。
- 任何 gate 失败都写失败收据并停止；禁止看见结果后原地修改阈值或预算，变更必须生成新 run id。
- 所有命令从 `D:\Paper\github_work\paper-code\frame` 运行。会话开头设置 `$Repo = 'D:\Paper\github_work\paper-code'`、`$Frame = Join-Path $Repo 'frame'`、`$Py = (Get-Command python).Source`；CVXPYlayers 命令固定使用 `D:\Paper\envs\rsc_pf_diffopt_v4\python.exe`。

---

## File Structure Map

### Versioned protocol and control plane

- Create `frame/configs/joint_forecast_dispatch_formal_v4_2.json`: frozen scientific protocol, method matrix, fixed budgets, gate thresholds, data years and output root.
- Create `frame/configs/formal_v4_source_closure_v4_2.txt`: exact source/config/script/test closure used by provenance manifests.
- Modify `frame/configs/joint_dispatch_invalid_runs_v4.json`: register the formal-v4.1 Gate 2 run as diagnostic-only.
- Create `frame/src/joint_dispatch/formal_v4_2_contract.py`: typed contract loading, canonical hashing and gate-transition checks.
- Create `frame/src/joint_dispatch/formal_v4_2_artifacts.py`: write-once artifacts, parent-lineage verification, resume and failure receipts.
- Create `frame/src/joint_dispatch/formal_v4_2_access.py`: fail-closed year access with one-time Gate 3 authorization.

### Data, teachers, training and evaluation

- Create `frame/src/joint_dispatch/formal_v4_2_data.py`: train-only normalization, state-window loading and representative Gate 1 origin manifests.
- Create `frame/src/joint_dispatch/formal_v4_2_teacher.py`: seed/checkpoint/state-bound same-information LP overlays.
- Create `frame/src/joint_dispatch/formal_v4_2_checkpoint.py`: persistent optimizer/RNG checkpoint save, load and byte-identical Stage S cloning.
- Create `frame/src/joint_dispatch/formal_v4_2_training.py`: complete Stage P/S/J loops, scheduled roll-in, gradient receipts and early stopping.
- Create `frame/src/joint_dispatch/formal_v4_2_rollout.py`: common chronological first-step evaluator and causal state advancement.
- Create `frame/src/joint_dispatch/formal_v4_2_metrics.py`: forecast, realized dispatch, physical residual, activity/season and block-bootstrap metrics.
- Create `frame/src/joint_dispatch/formal_v4_2_methods.py`: checkpoint-backed adapters and executors for all registered methods.

### Gate entry points

- Create `frame/scripts/run_rsc_pf_formal_v4_2_gate0.py`: provenance, capacity, dependencies and real operation resource probe.
- Create `frame/scripts/run_rsc_pf_formal_v4_2_pilot.py`: one-seed train-only engineering pilot.
- Create `frame/scripts/run_rsc_pf_formal_v4_2_gate1.py`: representative 2019 calibration and freeze receipt.
- Create `frame/scripts/run_rsc_pf_formal_v4_2_gate2.py`: complete fixed 2019 method/seed selection matrix.
- Create `frame/scripts/audit_rsc_pf_formal_v4_2_gate2.py`: independent fail-closed Gate 2 audit and frozen seed-extension authorization.
- Create `frame/scripts/extend_rsc_pf_formal_v4_2_seeds.py`: after Gate 2 passes, train the frozen 2029/2030 checkpoint extension without opening 2020.
- Create `frame/scripts/run_rsc_pf_formal_v4_2_gate3.py`: authorized locked 2020 evaluation only.
- Create `frame/scripts/summarize_rsc_pf_formal_v4_2.py`: machine-readable and human-readable main-result bundle.

### Tests

- Create `frame/tests/test_joint_dispatch_formal_v4_2_contract.py`
- Create `frame/tests/test_joint_dispatch_formal_v4_2_artifacts.py`
- Create `frame/tests/test_joint_dispatch_formal_v4_2_access.py`
- Create `frame/tests/test_joint_dispatch_formal_v4_2_data.py`
- Create `frame/tests/test_joint_dispatch_formal_v4_2_teacher.py`
- Create `frame/tests/test_joint_dispatch_formal_v4_2_checkpoint.py`
- Create `frame/tests/test_joint_dispatch_formal_v4_2_training.py`
- Create `frame/tests/test_joint_dispatch_formal_v4_2_rollout.py`
- Create `frame/tests/test_joint_dispatch_formal_v4_2_metrics.py`
- Create `frame/tests/test_joint_dispatch_formal_v4_2_methods.py`
- Create `frame/tests/test_joint_dispatch_formal_v4_2_gate0.py`
- Create `frame/tests/test_joint_dispatch_formal_v4_2_pilot.py`
- Create `frame/tests/test_joint_dispatch_formal_v4_2_gate1.py`
- Create `frame/tests/test_joint_dispatch_formal_v4_2_gate2.py`
- Create `frame/tests/test_joint_dispatch_formal_v4_2_seed_extension.py`
- Create `frame/tests/test_joint_dispatch_formal_v4_2_gate3.py`

---

### Task 1: Freeze the Formal-v4.2 Contract and Retire the Invalid v4.1 Result

**Files:**
- Create: `frame/configs/joint_forecast_dispatch_formal_v4_2.json`
- Create: `frame/configs/formal_v4_source_closure_v4_2.txt`
- Modify: `frame/configs/joint_dispatch_invalid_runs_v4.json`
- Create: `frame/src/joint_dispatch/formal_v4_2_contract.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_contract.py`

**Interfaces:**
- Consumes: the approved design spec and existing v4.1 contract.
- Produces: `FormalV42Contract`, `load_formal_v4_2_contract(path: str | Path) -> FormalV42Contract`, `assert_gate_transition(contract, completed_gate, requested_gate) -> None`, and canonical `contract_sha256`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_v42_contract_freezes_information_and_budget_boundaries(tmp_path):
    contract = load_formal_v4_2_contract(CONFIG)
    assert contract.train_years == (2015, 2016, 2017, 2018)
    assert contract.selection_year == 2019
    assert contract.evaluation_year == 2020
    assert contract.excluded_years == (2021,)
    assert contract.lookback == 24 and contract.horizon == 4
    assert contract.gate2_seeds == (2026, 2027, 2028)
    assert contract.gate3_seeds == (2026, 2027, 2028, 2029, 2030)
    assert contract.gas_semantics == "station_side_auxiliary_prior"
    assert contract.training["stage_order"] == ["P", "teacher", "S", "clone", "J"]
    assert contract.latent_control_dim == 15
    assert len(contract.dispatch_order) == 21
    assert contract.allow_future_binary_decisions is False

def test_invalid_v41_gate2_is_not_a_paper_result():
    registry = json.loads(INVALID_REGISTRY.read_text(encoding="utf-8"))
    row = next(x for x in registry["runs"] if x["run_id"] == "formal_v4_1_gate2_20260904_104000")
    assert row["status"] == "invalid_diagnostic"
    assert row["paper_eligible"] is False
```

- [ ] **Step 2: Run the tests and verify the new version is absent**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_contract.py -q`

Expected: FAIL because `formal_v4_2_contract` and the v4.2 config do not exist.

- [ ] **Step 3: Add the frozen contract and typed loader**

The JSON must carry schema `joint-forecast-dispatch-formal-v4.2`, `protocol_status: frozen`, task and dispatch orders from v4.1, 15 latent continuous controls, no future binary decision head, the exact global constraints above, all nine registered methods, fixed stage budgets, Gate 1 origin strata, Gate 2 criteria, the post-Gate-2 frozen seed extension `[2029, 2030]`, Gate 3 seeds, and `paths.output_root: reports/joint_forecast_dispatch_formal_v4_2`.

```python
@dataclass(frozen=True)
class FormalV42Contract:
    payload: Mapping[str, Any]
    contract_sha256: str

    @property
    def train_years(self) -> tuple[int, ...]:
        return tuple(self.payload["train_years"])

    @property
    def selection_year(self) -> int:
        return int(self.payload["selection_year"])

    @property
    def evaluation_year(self) -> int:
        return int(self.payload["evaluation_year"])

    @property
    def excluded_years(self) -> tuple[int, ...]:
        return tuple(self.payload["excluded_years"])

    @property
    def gate2_seeds(self) -> tuple[int, ...]:
        return tuple(self.payload["selection"]["gate2_seeds"])

    @property
    def gate3_seeds(self) -> tuple[int, ...]:
        return tuple(self.payload["selection"]["gate3_seeds"])

    def __getattr__(self, name: str) -> Any:
        if name in self.payload:
            return self.payload[name]
        raise AttributeError(name)

    def require_frozen(self) -> None:
        if self.payload["schema_version"] != "joint-forecast-dispatch-formal-v4.2":
            raise ValueError("formal-v4.2 schema required")
        if self.payload["protocol_status"] != "frozen":
            raise ValueError("formal-v4.2 contract is not frozen")

def load_formal_v4_2_contract(path: str | Path) -> FormalV42Contract:
    raw = Path(path).read_bytes()
    contract = FormalV42Contract(json.loads(raw), hashlib.sha256(raw).hexdigest())
    contract.require_frozen()
    return contract
```

- [ ] **Step 4: Register v4.1 Gate 2 and build the source closure**

Add a registry row containing the exact run id, corrected receipt path, `status: invalid_diagnostic`, `paper_eligible: false`, and reasons `incomplete_method_matrix`, `realized_future_teacher`, `open_loop_overlap_evaluation`, `nonpersistent_optimizer`, and `missing_checkpoint_lineage`. List every formal-v4.2 config, module, script and test named in this plan in the source closure.

- [ ] **Step 5: Run focused and existing protocol tests**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_contract.py tests/test_joint_dispatch_formal_protocol_v4.py tests/test_joint_dispatch_formal_v4_provenance.py -q`

Expected: PASS; the old v4.1 contract remains loadable but its invalid Gate 2 cannot be classified as a paper result.

- [ ] **Step 6: Commit the protocol boundary**

```powershell
git add configs/joint_forecast_dispatch_formal_v4_2.json configs/formal_v4_source_closure_v4_2.txt configs/joint_dispatch_invalid_runs_v4.json src/joint_dispatch/formal_v4_2_contract.py tests/test_joint_dispatch_formal_v4_2_contract.py
git commit -m "feat: freeze formal v4.2 protocol"
```

### Task 2: Add Immutable Lineage, Resume, Failure, and Access Control

**Files:**
- Create: `frame/src/joint_dispatch/formal_v4_2_artifacts.py`
- Create: `frame/src/joint_dispatch/formal_v4_2_access.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_artifacts.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_access.py`

**Interfaces:**
- Consumes: `FormalV42Contract.contract_sha256`.
- Produces: `MethodSeedKey`, `ArtifactStore`, `create_v42_run_root`, `write_once_json`, `verify_lineage`, `write_failure_receipt`, `ArtifactStore.completed_rows`, `FormalV42AccessController.request_years`, and `validate_gate3_envelope`.

- [ ] **Step 1: Write failing immutable-artifact tests**

```python
def test_write_once_rejects_changed_payload(tmp_path):
    path = tmp_path / "receipt.json"
    write_once_json(path, {"status": "pass"})
    with pytest.raises(FileExistsError):
        write_once_json(path, {"status": "fail"})

def test_resume_accepts_only_hash_identical_completed_rows(tmp_path):
    row = MethodSeedKey("RSC-PF", 2026)
    store = ArtifactStore(tmp_path, contract_sha256="a" * 64)
    store.complete(row, {"checkpoint_sha256": "b" * 64})
    assert store.completed_rows() == {row}
    with pytest.raises(LineageError):
        ArtifactStore(tmp_path, contract_sha256="c" * 64).completed_rows()
```

- [ ] **Step 2: Write failing year-access tests**

```python
def test_gate2_cannot_request_2020(tmp_path):
    controller = FormalV42AccessController(run_root=tmp_path, gate="gate2")
    with pytest.raises(EvaluationAccessDenied):
        controller.request_years("evaluation", [2020], caller="gate2")

def test_gate3_requires_signed_authorization_envelope(tmp_path):
    controller = FormalV42AccessController(run_root=tmp_path, gate="gate3")
    with pytest.raises(EvaluationAccessDenied):
        controller.request_years("evaluation", [2020], caller="gate3")
```

- [ ] **Step 3: Run tests to establish both failures**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_artifacts.py tests/test_joint_dispatch_formal_v4_2_access.py -q`

Expected: FAIL because the versioned artifact and access controllers are absent.

- [ ] **Step 4: Implement write-once lineage and resumable row receipts**

```python
@dataclass(frozen=True, order=True)
class MethodSeedKey:
    method_id: str
    seed: int | None

def write_once_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    target = Path(path)
    encoded = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
    if target.exists():
        if target.read_bytes() == encoded:
            return hashlib.sha256(encoded).hexdigest()
        raise FileExistsError(f"immutable artifact differs: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()
```

`ArtifactStore.complete` must require method id, actual seed, contract hash, source hash, data hash, checkpoint hash, runtime and status. `ArtifactStore.failed` writes the same identity fields plus exception class and message; a failed row remains incomplete and blocks gate authorization.

- [ ] **Step 5: Implement fail-closed access and one-time envelope validation**

```python
def validate_gate3_envelope(envelope: Mapping[str, Any], contract_sha256: str) -> None:
    if envelope.get("schema") != "formal-v4.2-gate3-authorization-v1":
        raise EvaluationAccessDenied("invalid Gate 3 authorization schema")
    if envelope.get("contract_sha256") != contract_sha256:
        raise EvaluationAccessDenied("Gate 3 contract hash mismatch")
    if envelope.get("allowed_years") != [2020] or envelope.get("consumed") is not False:
        raise EvaluationAccessDenied("Gate 3 envelope is not an unused 2020 authorization")
```

The controller must reject 2021 at every gate, reject 2020 before Gate 3, record every allowed/blocked access event, and atomically create a separate immutable `GATE3_AUTHORIZATION_CONSUMED.json` receipt when Gate 3 first opens 2020. The original authorization envelope remains immutable.

- [ ] **Step 6: Run focused tests**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_artifacts.py tests/test_joint_dispatch_formal_v4_2_access.py -q`

Expected: PASS, including changed-payload rejection, hash-mismatch resume rejection, 2020 denial and 2021 denial.

- [ ] **Step 7: Commit the control plane**

```powershell
git add src/joint_dispatch/formal_v4_2_artifacts.py src/joint_dispatch/formal_v4_2_access.py tests/test_joint_dispatch_formal_v4_2_artifacts.py tests/test_joint_dispatch_formal_v4_2_access.py
git commit -m "feat: enforce formal v4.2 lineage and access"
```

### Task 3: Build Train-Only Normalization and Representative 2019 Origins

**Files:**
- Create: `frame/src/joint_dispatch/formal_v4_2_data.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_data.py`

**Interfaces:**
- Consumes: `FormalV4WindowSplit`, the v4.2 contract and the access controller.
- Produces: `calculate_field_statistics`, `hash_normalization_inputs`, `NormalizationReceiptV42`, `fit_train_normalization`, `apply_normalization`, `Gate1OriginManifestV42`, and `select_gate1_origins`.

- [ ] **Step 1: Write failing normalization and origin-selection tests**

```python
def test_normalization_ignores_selection_extremes(train_split, selection_split):
    receipt = fit_train_normalization(train_split, years=(2015, 2016, 2017, 2018))
    shifted = replace(selection_split, target=selection_split.target + 1e9)
    repeated = fit_train_normalization(train_split, years=(2015, 2016, 2017, 2018))
    assert repeated.receipt_sha256 == receipt.receipt_sha256
    assert np.isfinite(apply_normalization(shifted, receipt).target).all()

def test_state_windows_keep_complete_24_hour_device_trajectory(train_split):
    assert train_split.device_history.shape[1:] == (24, 21)
    assert train_split.activity_history.shape[1:] == (24, 6)

def test_gate1_manifest_covers_activity_and_seasons(selection_split, contract):
    manifest = select_gate1_origins(selection_split, contract.payload["selection"]["gate1_origin_design"])
    assert set(manifest.strata) >= {"cooling_active", "heating_active", "winter", "summer", "shoulder", "chronological_remaining"}
    assert manifest.cooling_active_fraction >= 0.20
    assert manifest.heating_active_fraction >= 0.20
    assert len(np.unique(manifest.origin_indices)) == len(manifest.origin_indices)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_data.py -q`

Expected: FAIL because train-only receipts and representative-origin manifests are not implemented.

- [ ] **Step 3: Implement hash-bound train normalization**

```python
@dataclass(frozen=True)
class NormalizationReceiptV42:
    train_years: tuple[int, ...]
    field_mean: Mapping[str, np.ndarray]
    field_scale: Mapping[str, np.ndarray]
    train_data_sha256: str
    receipt_sha256: str

    @classmethod
    def from_train_split(cls, split: FormalV4WindowSplit, years: tuple[int, ...]):
        means, scales, zero_scale_masks = calculate_field_statistics(split)
        identity = hash_normalization_inputs(split, years, means, scales, zero_scale_masks)
        return cls(years, means, scales, split.identity_sha256, identity)

def fit_train_normalization(split: FormalV4WindowSplit, years: tuple[int, ...]) -> NormalizationReceiptV42:
    if years != (2015, 2016, 2017, 2018):
        raise ValueError("normalization years must be 2015-2018")
    return NormalizationReceiptV42.from_train_split(split, years)
```

Persist load-history, exogenous-history, device-history, rigid-target and gas-target statistics separately. Replace zero standard deviations with 1.0 and record that replacement mask.

- [ ] **Step 4: Implement deterministic representative origins**

The selector must derive activity thresholds from positive 2019 values, allocate the contract-fixed counts by cooling-active, heating-active, winter, summer, shoulder and chronological remaining strata, deduplicate indices in a fixed priority order, and write timestamps, indices, sampling weights, task activity fractions, source hash and identity hash.

```python
manifest = Gate1OriginManifestV42(
    origin_indices=np.asarray(indices, dtype=np.int64),
    timestamps=selection.timestamps[indices],
    strata=tuple(labels),
    sampling_weights=np.asarray(weights, dtype=np.float64),
    activity_fraction={"cooling": cooling_fraction, "heating": heating_fraction},
    selection_data_sha256=selection.identity_sha256,
)
manifest.validate(min_active_fraction=0.20)
```

- [ ] **Step 5: Run data tests and existing split tests**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_data.py tests/test_joint_dispatch_formal_v4_data.py tests/test_joint_dispatch_formal_v4_access.py -q`

Expected: PASS; test output must show no request for evaluation years.

- [ ] **Step 6: Commit the data boundary**

```powershell
git add src/joint_dispatch/formal_v4_2_data.py tests/test_joint_dispatch_formal_v4_2_data.py
git commit -m "feat: add formal v4.2 data manifests"
```

### Task 4: Persist Optimizers, RNG State, and Exact Stage-S Clones

**Files:**
- Create: `frame/src/joint_dispatch/formal_v4_2_checkpoint.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_checkpoint.py`

**Interfaces:**
- Consumes: a PyTorch model, one optimizer per stage and lineage hashes.
- Produces: `TrainingCheckpointV42`, `save_training_checkpoint`, `load_training_checkpoint`, `clone_stage_s_branches`, and `optimizer_step_value`.

- [ ] **Step 1: Write failing persistence and clone tests**

```python
def test_optimizer_step_increases_across_two_batches(tmp_path):
    model, optimizer = tiny_model_and_adamw()
    train_one_batch(model, optimizer)
    first = optimizer_step_value(optimizer)
    train_one_batch(model, optimizer)
    second = optimizer_step_value(optimizer)
    assert first == 1 and second == 2

def test_stage_s_clones_are_byte_identical(tmp_path):
    source = write_checkpoint(tmp_path / "stage_s.pt")
    joint, decoupled = clone_stage_s_branches(source, tmp_path / "joint.pt", tmp_path / "decoupled.pt")
    assert sha256_file(joint) == sha256_file(decoupled)
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_checkpoint.py -q`

Expected: FAIL because checkpoint and clone APIs do not exist.

- [ ] **Step 3: Implement full checkpoint payloads**

```python
def save_training_checkpoint(path, *, model, optimizer, epoch, early_stopping, lineage):
    payload = {
        "schema": "formal-v4.2-training-checkpoint-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "early_stopping": dict(early_stopping),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "lineage": dict(lineage),
    }
    atomic_torch_save(path, payload)
```

Loading must verify contract, source, data, normalization, teacher and parent-checkpoint hashes before restoring any state. The branch clone operation copies the frozen Stage S file twice and verifies equal SHA-256 before either file can be opened for Stage J.

- [ ] **Step 4: Run focused checkpoint tests**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_checkpoint.py -q`

Expected: PASS for optimizer steps 1→2, exact branch hashes, RNG restoration and hash-mismatch rejection.

- [ ] **Step 5: Commit persistent training state**

```powershell
git add src/joint_dispatch/formal_v4_2_checkpoint.py tests/test_joint_dispatch_formal_v4_2_checkpoint.py
git commit -m "feat: persist formal v4.2 training state"
```

### Task 5: Generate a Seed-Specific Same-Information Teacher

**Files:**
- Create: `frame/src/joint_dispatch/formal_v4_2_teacher.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_teacher.py`

**Interfaces:**
- Consumes: frozen Stage P checkpoint, causal rolling state, predicted rigid demand, gas prior, persistence PV/WT, prices and IES parameters.
- Produces: `TeacherKeyV42`, `TeacherOverlayV42`, `build_same_information_teacher_v42`, `save_teacher_overlay`, and `load_teacher_overlay`.

- [ ] **Step 1: Write failing information-boundary and cache tests**

```python
def test_teacher_uses_predictions_and_persistence_not_realized_future(fake_window, fake_stage_p):
    overlay = build_same_information_teacher_v42(fake_stage_p, [fake_window], seed=2026)
    assert np.array_equal(overlay.rigid_demand, fake_stage_p.predicted_rigid)
    assert np.array_equal(overlay.renewable_plan, np.repeat(fake_window.renewable_history[-1:], 4, axis=0))
    assert not np.array_equal(overlay.rigid_demand, fake_window.realized_target[:, :3])

def test_teacher_cache_rejects_checkpoint_or_state_hash_change(tmp_path, overlay):
    save_teacher_overlay(tmp_path, overlay)
    with pytest.raises(TeacherCacheMismatch):
        load_teacher_overlay(tmp_path, replace(overlay.key, checkpoint_sha256="0" * 64))
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_teacher.py -q`

Expected: FAIL because no v4.2 teacher implementation exists.

- [ ] **Step 3: Implement complete teacher identities**

```python
@dataclass(frozen=True)
class TeacherKeyV42:
    seed: int
    split: str
    timestamp_sha256: str
    state_sha256: str
    capacity_sha256: str
    benchmark_sha256: str
    normalization_sha256: str
    stage_p_checkpoint_sha256: str
    implementation_sha256: str
    source_manifest_sha256: str
```

For each chronological origin, run Stage P on its causal state, take the predicted electricity/cooling/heating values, repeat the last observed PV/WT availability across four planning steps, use the current SOC and previous CHP from that same state, and solve one exact LP. Persist forecast, 21-column dispatch, objective, shortage, timestamps, state hashes and LP status.

- [ ] **Step 4: Enforce cache reuse rules**

`load_teacher_overlay` must compare every `TeacherKeyV42` field and the saved array hash. A mismatch never regenerates inside an earlier authorized directory; it raises and forces a new producing-gate directory.

- [ ] **Step 5: Run teacher and LP tests**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_teacher.py tests/test_dispatch_lp.py tests/test_joint_dispatch_formal_v4_data.py -q`

Expected: PASS; a test that changes realized future demand while holding predicted demand fixed must leave teacher input unchanged.

- [ ] **Step 6: Commit the teacher repair**

```powershell
git add src/joint_dispatch/formal_v4_2_teacher.py tests/test_joint_dispatch_formal_v4_2_teacher.py
git commit -m "feat: build same-information formal v4.2 teacher"
```

### Task 6: Repair First-Step Physical Settlement and State Advancement

**Files:**
- Create: `frame/src/joint_dispatch/formal_v4_2_rollout.py`
- Create: `frame/src/joint_dispatch/formal_v4_2_metrics.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_rollout.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_metrics.py`

**Interfaces:**
- Consumes: `FormalV4ClosedLoopState`, a four-hour plan, realized first-hour demand/PV/WT and benchmark parameters.
- Produces: `canonical_recourse_once(planned, realized_first_hour, parameters)`, `calculate_all_residual_families(settled, state, realized_first_hour, parameters)`, `advance_with_executed_first_hour(state, settled, realized_first_hour)`, `settle_and_advance_v42`, `PhysicalResidualsV42`, `ChronologicalRolloutV42`, `evaluate_chronological_v42`, and `compute_v42_metrics`.

- [ ] **Step 1: Write failing first-step and realized-target tests**

```python
def test_rollout_executes_first_plan_row_and_carries_it_forward(state, plan, realized):
    plan[0, SOC] = 0.61
    plan[3, SOC] = 0.19
    result = settle_and_advance_v42(state, plan, realized)
    assert result.next_state.current_soc == pytest.approx(0.61)
    assert result.executed_plan_index == 0

def test_forecast_metrics_use_realized_targets_not_planning_demand(method, windows, state):
    result = evaluate_chronological_v42(method, windows, initial_state=state)
    expected = np.stack([w.realized_target for w in windows])
    assert np.array_equal(result.forecast_target, expected)
```

- [ ] **Step 2: Write failing physical-accounting tests**

```python
def test_physical_residuals_are_computed_and_slack_is_not_balance_error(outcome):
    metrics = compute_v42_metrics(outcome)
    assert metrics.shortage_energy.sum() > 0
    assert metrics.balance_residual_max <= 1e-7
    assert metrics.capacity_violation_max == calculated_capacity_violation(outcome)
    assert metrics.p_dump == outcome.realized_p_dump
    assert metrics.p_dump != outcome.plan[:, PV_CURT].sum()
```

- [ ] **Step 3: Run the tests and verify old behavior is exposed**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_rollout.py tests/test_joint_dispatch_formal_v4_2_metrics.py -q`

Expected: FAIL because v4.2 first-step rollout and calculated residual families are absent.

- [ ] **Step 4: Implement one-pass settlement**

```python
@dataclass(frozen=True)
class PhysicalResidualsV42:
    balance: np.ndarray
    capacity: np.ndarray
    conversion: np.ndarray
    soc: np.ndarray
    ramp: np.ndarray
    exclusivity: np.ndarray
    renewable_accounting: np.ndarray
    finite: np.ndarray

def settle_and_advance_v42(state, four_hour_plan, realized_first_hour, parameters):
    planned = np.asarray(four_hour_plan, dtype=np.float64)[0].copy()
    settled, shortage, p_dump, q_dump = canonical_recourse_once(planned, realized_first_hour, parameters)
    residuals = calculate_all_residual_families(settled, state, realized_first_hour, parameters)
    next_state = advance_with_executed_first_hour(state, settled, realized_first_hour)
    return SettledStepV42(planned, settled, shortage, p_dump, q_dump, residuals, next_state, 0)
```

The recourse function is called exactly once. State history appends settled device outputs and derived activity flags, not the fourth plan row. It detaches tensors before storing the next state.

- [ ] **Step 5: Implement primary and diagnostic metrics**

Forecast outputs: task×horizon MAE/RMSE/WAPE, rigid macro WAPE, gas WAPE, positive-active-hour values and season-stratified values. Dispatch outputs: cost, carbon, penalized objective, carrier shortage energy/rate, curtailment, `p_dump`, `q_dump`, every residual maximum, latency and optimizer calls. Refuse incompatible-unit macro MAE/RMSE.

- [ ] **Step 6: Run rollout, recourse and metric tests**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_rollout.py tests/test_joint_dispatch_formal_v4_2_metrics.py tests/test_joint_dispatch_formal_v4_recourse.py tests/test_joint_dispatch_formal_v4_state.py -q`

Expected: PASS; balance residual can be near zero while shortage remains positive, and next state always comes from plan row zero.

- [ ] **Step 7: Commit closed-loop settlement**

```powershell
git add src/joint_dispatch/formal_v4_2_rollout.py src/joint_dispatch/formal_v4_2_metrics.py tests/test_joint_dispatch_formal_v4_2_rollout.py tests/test_joint_dispatch_formal_v4_2_metrics.py
git commit -m "feat: add realized first-step formal v4.2 rollout"
```

### Task 7: Implement Sequential Stage P/S/J Training and Scheduled Roll-In

**Files:**
- Create: `frame/src/joint_dispatch/formal_v4_2_training.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_training.py`

**Interfaces:**
- Consumes: `RSCPFModel`, train/selection splits, normalization receipt, teacher overlay, checkpoint APIs and contract training block.
- Produces: `StageBudgetV42`, `run_stage_p`, `run_stage_s`, `run_stage_j_pair`, `run_training_seed_v42`, `run_direct_policy_training`, `refresh_rollin_sample`, `GradientBoundaryReceiptV42`, and `TrainingRunReceiptV42`.

- [ ] **Step 1: Write failing stage-order and optimizer tests**

```python
def test_stage_runner_completes_p_then_teacher_then_s_then_clone_then_j(fake_inputs):
    receipt = run_training_seed_v42(seed=2026, **fake_inputs)
    assert receipt.stage_events == ("P_complete", "teacher_complete", "S_complete", "clone_verified", "J_complete")
    assert receipt.optimizer_final_steps["P"] > 1
    assert receipt.optimizer_final_steps["S"] > 1
    assert receipt.optimizer_final_steps["J_joint"] > 1

def test_stage_j_pair_has_exact_gradient_boundary(fake_stage_s):
    joint, decoupled = run_stage_j_pair(fake_stage_s, one_batch_contract())
    assert joint.decision_forecaster_gradient_norm > 0.0
    assert decoupled.decision_forecaster_gradient_norm == 0.0
    assert joint.scheduler_gradient_norm > 0.0
    assert decoupled.scheduler_gradient_norm > 0.0
```

- [ ] **Step 2: Write failing curriculum and roll-in tests**

```python
def test_curriculum_reaches_registered_terminal_weight():
    budget = StageBudgetV42(max_epochs=30, minimum_epochs=18, decision_start=0.05, decision_final=1.0, ramp_epochs=18)
    assert budget.weights(epoch=0).decision == pytest.approx(0.05)
    assert budget.weights(epoch=18).decision == pytest.approx(1.0)

def test_refreshed_rollin_detaches_history_and_zeros_stale_imitation(sample):
    refreshed = refresh_rollin_sample(sample, recompute_teacher=False)
    assert refreshed.device_history.requires_grad is False
    assert refreshed.imitation_weight == 0.0
```

- [ ] **Step 3: Run tests and verify failures**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_training.py -q`

Expected: FAIL because current training recreates AdamW per call and interleaves stages.

- [ ] **Step 4: Implement persistent complete-stage loops**

```python
def run_stage_p(model, loaders, budget, lineage, seed):
    seed_everything(seed)
    optimizer = torch.optim.AdamW(model.forecaster_parameters(), lr=budget.forecaster_lr, weight_decay=budget.weight_decay)
    for epoch in range(budget.max_epochs):
        train_epoch_forecast(model, loaders.train, optimizer, budget)
        score = validate_forecast_and_guardrails(model, loaders.selection, budget)
        save_if_improved(model, optimizer, epoch, score, lineage)
        if budget.can_stop(epoch) and early_stopper.should_stop(score):
            break
    return load_best_checkpoint()
```

Stage S creates one scheduler optimizer outside the batch loop. Stage J creates separate joint and decoupled optimizers after loading byte-identical clones. Remove every hard-coded `seed=2026`; the receipt seed must equal the RNG seed and directory seed.

- [ ] **Step 5: Implement joint/decoupled loss routing**

```python
forecast_for_dispatch = forecast if mode == "joint" else forecast.detach()
dispatch = model.schedule_from_forecast(forecast_for_dispatch, context, state)
loss = weights.forecast * forecast_loss(forecast, target)
loss += weights.imitation * sample.imitation_weight * imitation_loss(dispatch, teacher_dispatch)
loss += weights.decision * realized_decision_loss(dispatch, realized_first_hour)
loss.backward()
```

Both modes update the scheduler. Both retain forecast supervision. Only `forecast_for_dispatch.detach()` differs. Check gradient receipts before `optimizer.step()` and parameter deltas after it.

- [ ] **Step 6: Implement registered early stopping and scheduled roll-in**

Early stopping cannot run before the contract minimum epoch needed to reach terminal decision weight. At the frozen roll-in fraction, replace the registered portion of teacher-forced histories with first-step settled model histories; recompute a teacher for that exact state or set imitation weight to zero for that refreshed sample.

- [ ] **Step 7: Run training tests and legacy model tests**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_training.py tests/test_joint_dispatch_formal_v4_training.py tests/test_joint_dispatch_formal_v4_models.py -q`

Expected: PASS for persistent step counters, strict stage order, exact clone, gradient boundary, actual seed propagation, curriculum endpoint and roll-in detach.

- [ ] **Step 8: Commit the repaired training engine**

```powershell
git add src/joint_dispatch/formal_v4_2_training.py tests/test_joint_dispatch_formal_v4_2_training.py
git commit -m "feat: implement sequential formal v4.2 training"
```

### Task 8: Make Every Registered Method Executable from Frozen Checkpoints

**Files:**
- Create: `frame/src/joint_dispatch/formal_v4_2_methods.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_methods.py`

**Interfaces:**
- Consumes: method id, checkpoint receipt, normalization receipt, benchmark parameters and rolling state.
- Produces: `FormalV42Method`, `MethodStepV42`, `build_v42_method`, `train_v42_baseline`, and `registered_method_rows`.

- [ ] **Step 1: Write failing full-matrix and checkpoint-loading tests**

```python
def test_registered_rows_cover_full_gate2_matrix(contract):
    rows = registered_method_rows(contract, gate="gate2")
    stochastic = [r for r in rows if r.seed is not None]
    deterministic = [r for r in rows if r.seed is None]
    assert len(stochastic) == 7 * 3
    assert {r.method_id for r in deterministic} == {"Perfect-Information-MPC", "Seasonal-Naive-PTO"}

def test_state_conditioned_pto_loads_exact_stage_p_checkpoint(stage_p_receipt, normalization):
    method = build_v42_method("State-Conditioned-PTO", seed=2026, checkpoint=stage_p_receipt, normalization=normalization)
    assert method.forecaster_sha256 == stage_p_receipt.model_sha256
```

- [ ] **Step 2: Write failing rolling-state and optimizer-count tests**

```python
@pytest.mark.parametrize("method_id", DEPLOYABLE_METHODS)
def test_method_consumes_supplied_rolling_state(method_id, two_distinct_states):
    method = make_method(method_id)
    a = method.plan(window(), two_distinct_states[0])
    b = method.plan(window(), two_distinct_states[1])
    assert a.state_sha256 != b.state_sha256

def test_online_lp_call_counts_are_truthful():
    assert make_method("RSC-PF").plan(window(), state()).optimizer_calls == 0
    assert make_method("Scheme2R-PTO").plan(window(), state()).optimizer_calls == 1
```

- [ ] **Step 3: Run tests and verify failures**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_methods.py -q`

Expected: FAIL because current adapters ignore rolling state, omit training/checkpoint paths, or do not cover the full matrix.

- [ ] **Step 4: Implement internal methods**

```python
@dataclass(frozen=True)
class MethodStepV42:
    forecast: np.ndarray | None
    four_hour_plan: np.ndarray
    state_sha256: str
    optimizer_calls: int
    latency_seconds: float
    metadata: Mapping[str, Any]

class FormalV42Method(Protocol):
    method_id: str
    deployable: bool
    def plan(self, window: WindowV42, rolling_state: FormalV4ClosedLoopState) -> MethodStepV42: ...
```

Implement RSC-PF, Decoupled-RSC-PF, Direct-Policy, Scheme2R-PTO, State-Conditioned-PTO, Seasonal-Naive-PTO and Perfect-Information-MPC. Direct-Policy uses the same causal history/context and its own trained decision head but returns `forecast=None`. PTO methods execute exactly one LP per window. PI-MPC is flagged `deployable=False` and alone may consume realized future information.

- [ ] **Step 5: Implement external adaptations**

Official iTransformer-PTO must instantiate `OfficialITransformerAdapter` from the verified THUML source, train its four-task head on 2015--2018, select on 2019, load normalization from the train receipt and run one LP per window. Differentiable-LP must train its forecast pathway through `DifferentiableIESLayer`, record its CVXPYlayers lock, and report one optimizer layer call per window; label both as adaptations, not exact reproduction of a published full forecasting-dispatch system.

- [ ] **Step 6: Run method and external-source tests in both environments**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_methods.py tests/test_joint_dispatch_formal_v4_itransformer.py -q`

Run: `& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_2_methods.py tests/test_joint_dispatch_formal_v4_diffopt.py -q`

Expected: PASS; every Gate 2 row can be constructed, required checkpoint hashes match, rolling-state hashes change with state, and optimizer counts match method semantics.

- [ ] **Step 7: Commit the complete method layer**

```powershell
git add src/joint_dispatch/formal_v4_2_methods.py tests/test_joint_dispatch_formal_v4_2_methods.py
git commit -m "feat: execute all formal v4.2 methods"
```

### Task 9: Build a Real Gate 0 Resource and Provenance Probe

**Files:**
- Create: `frame/scripts/run_rsc_pf_formal_v4_2_gate0.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_gate0.py`

**Interfaces:**
- Consumes: v4.2 contract, source closure, data/access APIs, all method builders and both Python environments.
- Produces: one run root with `gate0/GATE0_EVIDENCE.json`, `gate0/RESOURCE_PROJECTION.json`, source/data/access manifests and `authorized_pilot`.

- [ ] **Step 1: Write failing real-operation tests**

```python
def test_resource_probe_calls_real_components(monkeypatch, gate0_fixture):
    calls = Counter()
    monkeypatch.setattr(probe, "run_rsc_forward", counted(calls, "rsc_forward"))
    monkeypatch.setattr(probe, "run_rsc_backward", counted(calls, "rsc_backward"))
    monkeypatch.setattr(probe, "run_highs_lp", counted(calls, "highs_lp"))
    monkeypatch.setattr(probe, "run_diff_lp", counted(calls, "diff_lp"))
    receipt = run_gate0_probe(gate0_fixture)
    assert calls == Counter(rsc_forward=1, rsc_backward=1, highs_lp=1, diff_lp=1)
    assert receipt["synthetic_probe"] is False

def test_gate0_fails_when_source_or_dependency_receipt_is_missing(gate0_fixture):
    gate0_fixture.itransformer_receipt.unlink()
    assert run_gate0_probe(gate0_fixture)["authorized_pilot"] is False

def test_gate0_capacity_fit_and_normalization_never_use_selection_or_evaluation(gate0_fixture):
    receipt = run_gate0_probe(gate0_fixture)
    assert receipt["capacity_fit_years"] == [2015, 2016, 2017, 2018]
    assert receipt["normalization_fit_years"] == [2015, 2016, 2017, 2018]
    assert receipt["materialized_years"] == [2015, 2016, 2017, 2018, 2019]
    assert receipt["evaluation_year_accessed"] is False
```

- [ ] **Step 2: Run the test and confirm the NumPy-only probe is rejected**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_gate0.py -q`

Expected: FAIL because the existing resource gate times synthetic NumPy operations.

- [ ] **Step 3: Implement the real probe and receipts**

Time warmup and measured repetitions for: RSC forward, forecast backward, decision backward into forecaster, Direct-Policy forward/backward, official iTransformer forward/backward, SciPy HiGHS same-information LP, and CVXPYlayers forward/backward in its locked environment. Invoke the CVXPYlayers micro-run through `D:\Paper\envs\rsc_pf_diffopt_v4\python.exe`, return one JSON measurement to the main runner, and include the locked environment hash. Record median/p95, peak RSS, disk bytes per checkpoint/window, CPU model, thread count, Python/package versions and determinism flags.

```python
operations = {
    "rsc_forward": lambda: rsc_model(**real_train_batch),
    "rsc_decision_backward": lambda: backward_decision_once(rsc_model, real_train_batch),
    "highs_lp": lambda: solve_same_information_lp(real_train_row),
}
for name, operation in operations.items():
    measurements[name] = benchmark_real_operation(operation, warmup=2, iterations=5)
```

Materialize fresh v4.2 train and selection artifacts under `gate0/data`, but fit both normalization and the capacity multiplier using 2015--2018 only. Project the pilot, Gate 1, Gate 2, the 2029/2030 frozen seed extension and Gate 3 separately from measured operation counts. Authorization requires credible nonzero time/memory, ≥20% disk margin, official source receipt, diffopt lock, capacity receipt, immutable source/data hashes, no evaluation-year access and no mutation of v4.1.

- [ ] **Step 4: Run focused and existing Gate 0 tests**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_gate0.py tests/test_joint_dispatch_formal_v4_gate0_integration.py tests/test_joint_dispatch_formal_v4_resources.py -q`

Expected: PASS; deliberately replacing a real callable with `lambda: np.zeros(1)` must fail the component-identity assertion.

- [ ] **Step 5: Commit the Gate 0 runner**

```powershell
git add scripts/run_rsc_pf_formal_v4_2_gate0.py tests/test_joint_dispatch_formal_v4_2_gate0.py
git commit -m "feat: add real formal v4.2 resource gate"
```

### Task 10: Implement the One-Seed Train-Only Engineering Pilot

**Files:**
- Create: `frame/scripts/run_rsc_pf_formal_v4_2_pilot.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_pilot.py`

**Interfaces:**
- Consumes: valid Gate 0 receipt and only 2015--2018 windows.
- Produces: `authorize_pilot(receipt, shortage_rate_max) -> bool`, non-paper pilot checkpoints, loss curves, gradient/clone/state receipts and `authorized_gate1`.

- [ ] **Step 1: Write failing pilot-authorization tests**

```python
def test_pilot_requires_numerical_and_structural_checks(pilot_receipt):
    pilot_receipt.update({
        "stage_p_loss_decreased": True,
        "stage_s_loss_decreased": True,
        "stage_j_loss_finite": True,
        "persistent_optimizer_steps": True,
        "stage_s_clone_identical": True,
        "joint_forecast_decision_gradient_positive": True,
        "decoupled_forecast_decision_gradient_zero": True,
        "first_step_state_carry": True,
        "shortage_rate": 0.50,
    })
    assert authorize_pilot(pilot_receipt, shortage_rate_max=0.80) is True
    pilot_receipt["stage_s_clone_identical"] = False
    assert authorize_pilot(pilot_receipt, shortage_rate_max=0.80) is False
```

- [ ] **Step 2: Run the test and confirm failure**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_pilot.py -q`

Expected: FAIL because no pilot runner/authorizer exists.

- [ ] **Step 3: Implement the bounded pilot**

Use seed 2026, a contract-fixed train-only subset and small but multi-batch/multi-epoch budgets. The pilot must execute complete P→teacher→S→clone→J ordering, both Stage J modes and at least 24 chronological first-step settlements. Mark every artifact `paper_eligible: false`; do not compute a 2019 ranking.

- [ ] **Step 4: Run pilot tests**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_pilot.py tests/test_joint_dispatch_formal_v4_2_training.py tests/test_joint_dispatch_formal_v4_2_rollout.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the pilot implementation**

```powershell
git add scripts/run_rsc_pf_formal_v4_2_pilot.py tests/test_joint_dispatch_formal_v4_2_pilot.py
git commit -m "feat: add formal v4.2 engineering pilot"
```

### Task 11: Implement Representative Gate 1 Calibration

**Files:**
- Create: `frame/scripts/run_rsc_pf_formal_v4_2_gate1.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_gate1.py`

**Interfaces:**
- Consumes: `authorized_gate1=true`, representative origin manifest and 2019 only.
- Produces: `run_gate1_calibration(input: Gate1InputV42) -> Gate1FreezeV42`, frozen epochs, patience, validation frequency, learning rates, curriculum, candidate selection, exact manifest hashes and `GATE1_FREEZE.json`.

- [ ] **Step 1: Write failing calibration-integrity tests**

```python
def test_gate1_rejects_near_zero_activity_manifest(gate1_input):
    bad_manifest = replace(gate1_input.manifest, activity_fraction={"cooling": 0.00025, "heating": 0.0685})
    with pytest.raises(Gate1AuthorizationError):
        run_gate1_calibration(replace(gate1_input, manifest=bad_manifest))

def test_gate1_freeze_records_every_gate2_budget(gate1_receipt):
    required = {"stage_p_max_epochs", "stage_s_max_epochs", "stage_j_max_epochs", "minimum_epochs", "patience", "validation_interval", "learning_rates", "curriculum", "candidate_values"}
    assert required <= set(gate1_receipt["frozen_training"])
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_gate1.py -q`

Expected: FAIL because the current Gate 1 uses a dynamically selected near-zero block and does not freeze the complete Gate 2 configuration.

- [ ] **Step 3: Implement weighted representative calibration**

Load exactly the precomputed manifest and its sampling weights. Run only registered contract candidates. Use weighted selection metrics plus active-hour guardrails; save every candidate row, not only the winner. The winner is determined by the contract scoring function before results are opened.

- [ ] **Step 4: Make the freeze receipt exhaustive and immutable**

```python
freeze = {
    "schema": "formal-v4.2-gate1-freeze-v1",
    "authorized_gate2": all(checks.values()),
    "origin_manifest_sha256": sha256_file(origin_manifest_path),
    "normalization_sha256": normalization.receipt_sha256,
    "contract_sha256": contract.contract_sha256,
    "frozen_training": selected_training_config,
    "frozen_gate2_methods": gate2_method_rows,
    "checks": checks,
    "access_receipt_sha256": access_sha256,
}
```

- [ ] **Step 5: Run Gate 1 tests**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_gate1.py tests/test_joint_dispatch_formal_v4_2_data.py -q`

Expected: PASS; an activity-poor manifest and any candidate outside the contract both fail closed.

- [ ] **Step 6: Commit Gate 1 implementation**

```powershell
git add scripts/run_rsc_pf_formal_v4_2_gate1.py tests/test_joint_dispatch_formal_v4_2_gate1.py
git commit -m "feat: freeze representative formal v4.2 gate1"
```

### Task 12: Implement the Complete Fixed Gate 2 Matrix

**Files:**
- Create: `frame/scripts/run_rsc_pf_formal_v4_2_gate2.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_gate2.py`

**Interfaces:**
- Consumes: valid immutable Gate 1 freeze, all method executors and 2019 chronology.
- Produces: `build_gate2_parser() -> argparse.ArgumentParser`, `authorize_gate2(rows, contract) -> Gate2DecisionV42`, seed checkpoints, first-step arrays, per-row receipts, full matrix, summary and provisional `GATE2_DECISION.json`.

- [ ] **Step 1: Write failing CLI and coverage tests**

```python
def test_gate2_parser_exposes_no_training_budget_overrides():
    options = {a.dest for a in build_gate2_parser()._actions}
    assert options.isdisjoint({"epochs", "patience", "learning_rate", "candidate_values", "validation_interval"})

def test_gate2_authorization_requires_every_registered_row(contract, complete_rows):
    missing = [r for r in complete_rows if not (r.method_id == "Differentiable-LP" and r.seed == 2028)]
    assert authorize_gate2(missing, contract).authorized_gate3 is False
    assert "Differentiable-LP/2028" in authorize_gate2(missing, contract).missing_rows
```

- [ ] **Step 2: Write failing scientific-criterion tests**

```python
def test_gate2_requires_primary_direction_reliability_and_guardrails(valid_matrix):
    decision = authorize_gate2(valid_matrix, contract())
    assert decision.authorized_gate3 is True
    valid_matrix["RSC-PF", 2026]["shortage_energy"] = 1.051 * valid_matrix["State-Conditioned-PTO", 2026]["shortage_energy"]
    assert authorize_gate2(valid_matrix, contract()).authorized_gate3 is False
```

- [ ] **Step 3: Run tests and verify current runner fails the contract**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_gate2.py -q`

Expected: FAIL because the current runner accepts `--epochs`, runs two methods/six rows, lacks checkpoint identities and performs open-loop overlap evaluation.

- [ ] **Step 4: Implement fixed row enumeration and resume**

Enumerate seven stochastic methods × three Gate 2 seeds plus two deterministic rows. For every stochastic seed, run/restore Stage P and relevant baseline training, build its same-information teacher, run Stage S and Stage J branches, then evaluate chronologically. Before reusing any row, verify contract, Gate 1, data, normalization, source, teacher and checkpoint hashes.

```python
for key in registered_method_rows(contract, gate="gate2"):
    if store.is_complete_and_valid(key, lineage):
        continue
    try:
        result = execute_gate2_row(key, frozen_gate1, selection_2019)
        store.complete(key, result.receipt)
    except Exception as exc:
        store.failed(key, exc, lineage)
```

- [ ] **Step 5: Implement fail-closed provisional authorization**

Require complete rows; unchanged Gate 1 hashes; finite forecasts/dispatches; calculated physical tolerances; every forecast guardrail; RSC mean penalized objective lower than Decoupled; at least two of three favorable seed directions; RSC shortage ≤1.05× true seed-matched State-Conditioned-PTO; exact optimizer-call counts; complete checkpoint/runtime/failure receipts; and no 2020/2021 access.

- [ ] **Step 6: Run Gate 2 integration tests**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_gate2.py tests/test_joint_dispatch_formal_v4_2_methods.py tests/test_joint_dispatch_formal_v4_2_training.py tests/test_joint_dispatch_formal_v4_2_rollout.py -q`

Expected: PASS; deleting a row, changing a hash, inserting a NaN, weakening a physical metric, using realized targets as planning demand or recording the wrong seed each denies authorization.

- [ ] **Step 7: Commit Gate 2 implementation before launching it**

```powershell
git add scripts/run_rsc_pf_formal_v4_2_gate2.py tests/test_joint_dispatch_formal_v4_2_gate2.py
git commit -m "feat: add complete formal v4.2 gate2 matrix"
```

### Task 13: Implement the Independent Gate 2 Audit

**Files:**
- Create: `frame/scripts/audit_rsc_pf_formal_v4_2_gate2.py`
- Extend: `frame/tests/test_joint_dispatch_formal_v4_2_gate2.py`

**Interfaces:**
- Consumes: closed Gate 2 directory and read-only source/data manifests.
- Produces: `audit_gate2(run_root: str | Path) -> Gate2AuditV42`, independently recomputed metrics/criteria and, only on pass, authorization to create the frozen 2029/2030 checkpoints while 2020 remains inaccessible.

- [ ] **Step 1: Add failing tamper and label-identity tests**

```python
def test_independent_audit_recomputes_state_conditioned_pto_identity(gate2_copy):
    gate2_copy.replace_checkpoint("State-Conditioned-PTO", 2026, "Perfect-Information-MPC")
    audit = audit_gate2(gate2_copy.root)
    assert audit.authorized_gate3 is False
    assert "stage_p_checkpoint_identity" in audit.failed_checks

def test_failed_audit_never_authorizes_seed_extension(gate2_copy):
    gate2_copy.delete_row("RSC-PF", 2028)
    audit_gate2(gate2_copy.root)
    assert not (gate2_copy.root / "protocol/SEED_EXTENSION_AUTHORIZATION.json").exists()
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_gate2.py -q`

Expected: FAIL because an independent v4.2 audit does not exist.

- [ ] **Step 3: Implement independent recomputation**

The audit must read raw first-step arrays rather than trusting `GATE2_DECISION.json`; recompute forecast, dispatch, residual, seed-pair and shortage-ratio criteria; verify actual checkpoint file hashes and Stage P/Stage S parentage; verify no source closure change and scan access receipts for 2020/2021. It must not import authorization values from the Gate 2 runner.

- [ ] **Step 4: Authorize only the frozen seed extension after all checks pass**

Write `SEED_EXTENSION_AUTHORIZATION.json` once with allowed seeds `[2029, 2030]`, allowed years `[2015, 2016, 2017, 2018, 2019]`, and the contract/Gate 1/Gate 2/source hashes. This receipt must explicitly keep `allow_evaluation_year=false`. If any check fails, retain the audit and stop before Task 14 execution and Task 22 launch.

- [ ] **Step 5: Run audit tests**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_gate2.py tests/test_joint_dispatch_formal_v4_2_access.py tests/test_joint_dispatch_formal_v4_2_artifacts.py -q`

Expected: PASS for recomputation, tamper detection, mislabel detection and absence of seed-extension authorization on failure.

- [ ] **Step 6: Commit the independent audit**

```powershell
git add scripts/audit_rsc_pf_formal_v4_2_gate2.py tests/test_joint_dispatch_formal_v4_2_gate2.py
git commit -m "feat: independently audit formal v4.2 gate2"
```

### Task 14: Implement the Frozen 2029/2030 Seed Extension and Audit

**Files:**
- Create: `frame/scripts/extend_rsc_pf_formal_v4_2_seeds.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_seed_extension.py`

**Interfaces:**
- Consumes: valid `SEED_EXTENSION_AUTHORIZATION.json`, the unchanged Gate 1 freeze and only 2015--2019 data.
- Produces: `run_seed_extension(input: SeedExtensionInputV42) -> SeedExtensionReceiptV42`, `execute_frozen_training_seed`, `audit_extension_and_mint_gate3`, hash-audited checkpoints for seeds 2029 and 2030, and `mint_gate3_envelope(gate2_audit, seed_extension_audit, contract_sha256) -> dict[str, Any]`.

- [ ] **Step 1: Write failing seed-coverage and no-evaluation-access tests**

```python
def test_seed_extension_creates_only_missing_frozen_seeds(extension_fixture):
    receipt = run_seed_extension(extension_fixture)
    assert receipt["trained_seeds"] == [2029, 2030]
    assert receipt["changed_frozen_configuration"] is False
    assert receipt["evaluation_year_accessed"] is False

def test_gate3_envelope_requires_all_five_checkpoint_seeds(extension_fixture):
    extension_fixture.delete_checkpoint("RSC-PF", 2030)
    with pytest.raises(EvaluationAccessDenied):
        audit_extension_and_mint_gate3(extension_fixture.root)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_seed_extension.py -q`

Expected: FAIL because no post-Gate-2 frozen seed-extension path exists.

- [ ] **Step 3: Implement the fixed two-seed extension**

Run the exact Gate 1 training configuration for seeds 2029 and 2030 on 2015--2018, use the already frozen 2019 checkpoint-scoring rule, and produce all stochastic-method checkpoints needed by Gate 3. Do not reopen candidate selection, alter epoch budgets, compare alternative configurations, or calculate a new Gate 2 decision.

```python
allowed = authorization["allowed_seeds"]
if allowed != [2029, 2030] or authorization["allow_evaluation_year"] is not False:
    raise SeedExtensionError("invalid frozen seed extension authorization")
for seed in allowed:
    execute_frozen_training_seed(seed, gate1_freeze, train_split, selection_split, store)
```

- [ ] **Step 4: Audit checkpoint lineage before exposing 2020**

Verify that all required stochastic method checkpoints now exist for seeds 2026--2030, every 2029/2030 receipt uses the same contract/Gate 1/source/data/normalization hashes, every RSC/Decoupled pair has byte-identical Stage S parents, and access logs contain no 2020/2021 event.

- [ ] **Step 5: Mint the one-time Gate 3 envelope**

```python
def mint_gate3_envelope(gate2_audit, seed_extension_audit, contract_sha256):
    if not gate2_audit["authorized_seed_extension"] or not seed_extension_audit["all_five_seeds_ready"]:
        raise EvaluationAccessDenied("five-seed frozen checkpoint set is incomplete")
    return {
        "schema": "formal-v4.2-gate3-authorization-v1",
        "contract_sha256": contract_sha256,
        "gate2_audit_sha256": canonical_sha256(gate2_audit),
        "seed_extension_audit_sha256": canonical_sha256(seed_extension_audit),
        "allowed_years": [2020],
        "consumed": False,
    }
```

- [ ] **Step 6: Run seed-extension tests**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_seed_extension.py tests/test_joint_dispatch_formal_v4_2_access.py tests/test_joint_dispatch_formal_v4_2_training.py -q`

Expected: PASS; any missing seed, changed frozen field, wrong hash or evaluation access blocks the envelope.

- [ ] **Step 7: Commit the seed extension**

```powershell
git add scripts/extend_rsc_pf_formal_v4_2_seeds.py tests/test_joint_dispatch_formal_v4_2_seed_extension.py
git commit -m "feat: freeze formal v4.2 five-seed checkpoints"
```

### Task 15: Implement the Locked Gate 3 Evaluator

**Files:**
- Create: `frame/scripts/run_rsc_pf_formal_v4_2_gate3.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_gate3.py`

**Interfaces:**
- Consumes: a valid unconsumed Gate 3 authorization envelope, fixed Gate 2 selections and 2020 data.
- Produces: `run_gate3(run_root: str | Path, envelope: Mapping[str, Any]) -> Gate3ReceiptV42`, five-seed frozen evaluation rows, deterministic references once, raw first-step arrays and `GATE3_COMPLETE.json`.

- [ ] **Step 1: Write failing authorization and no-training tests**

```python
def test_gate3_refuses_missing_or_consumed_envelope(tmp_path):
    with pytest.raises(EvaluationAccessDenied):
        run_gate3(tmp_path, envelope=None)
    with pytest.raises(EvaluationAccessDenied):
        run_gate3(tmp_path, envelope={"consumed": True})

def test_gate3_never_calls_training(monkeypatch, authorized_fixture):
    monkeypatch.setattr(training, "run_stage_p", forbidden)
    monkeypatch.setattr(training, "run_stage_s", forbidden)
    monkeypatch.setattr(training, "run_stage_j_pair", forbidden)
    run_gate3(authorized_fixture.root, authorized_fixture.envelope)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_gate3.py -q`

Expected: FAIL because no locked Gate 3 runner exists.

- [ ] **Step 3: Implement one-time authorized 2020 loading**

Validate the envelope and all upstream hashes, atomically consume it, request only 2020 through the access controller, load frozen checkpoints for seeds 2026--2030, and run the shared chronological evaluator. Do not expose arguments for epochs, learning rate, patience, checkpoint choice, normalization, capacity or candidate selection.

- [ ] **Step 4: Implement complete output and statistics**

Save raw forecasts, targets, plans, settled actions, states, physical residual arrays, latency and optimizer calls for every row. Compute seed-matched RSC-minus-Decoupled effects with contiguous 168-hour blocks and 2,000 replicates; compute time-block uncertainty only for deterministic references. Mark PI-MPC non-deployable.

- [ ] **Step 5: Run Gate 3 tests without accessing real 2020**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_gate3.py tests/test_joint_dispatch_formal_v4_2_access.py tests/test_joint_dispatch_formal_v4_2_rollout.py tests/test_joint_dispatch_formal_v4_2_metrics.py -q`

Expected: PASS using synthetic fixtures; source scan confirms tests do not open actual evaluation files.

- [ ] **Step 6: Commit the Gate 3 runner**

```powershell
git add scripts/run_rsc_pf_formal_v4_2_gate3.py tests/test_joint_dispatch_formal_v4_2_gate3.py
git commit -m "feat: add locked formal v4.2 gate3 evaluation"
```

### Task 16: Implement the Independent Main-Experiment Summarizer

**Files:**
- Create: `frame/scripts/summarize_rsc_pf_formal_v4_2.py`
- Extend: `frame/tests/test_joint_dispatch_formal_v4_2_gate3.py`

**Interfaces:**
- Consumes: a complete Gate 3 raw-array manifest or a failed earlier-gate receipt.
- Produces: `summarize_gate3(run_root: str | Path) -> dict[str, Any]`, `summarize_run(run_root: str | Path) -> dict[str, Any]`, sealed main-result tables and an evidence-backed audit, without ablations or manuscript prose.

- [ ] **Step 1: Add failing summary-provenance tests**

```python
def test_summary_is_recomputed_from_raw_arrays(gate3_fixture):
    summary = summarize_gate3(gate3_fixture.root)
    assert summary["primary_contrast"] == "RSC-PF_minus_Decoupled-RSC-PF"
    assert summary["bootstrap"]["block_hours"] == 168
    assert summary["bootstrap"]["replicates"] == 2000
    assert summary["paper_eligible"] is True

def test_incomplete_or_unauthorized_run_is_not_paper_eligible(failed_gate2_fixture):
    summary = summarize_run(failed_gate2_fixture.root)
    assert summary["paper_eligible"] is False
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_gate3.py -q`

Expected: FAIL because the independent v4.2 summary does not exist.

- [ ] **Step 3: Implement raw-array recomputation and sealing**

Recompute all published numbers from raw arrays, verify every parent hash, identify stochastic versus deterministic uncertainty correctly, emit task×horizon forecast tables, active-hour/seasonal diagnostics, realized dispatch/physics/runtime tables and the primary paired contrast. A failed earlier gate produces only a diagnostic audit with `paper_eligible=false`.

- [ ] **Step 4: Run the entire v4.2 and relevant regression suite**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_*.py tests/test_joint_dispatch_formal_v4_*.py tests/test_dispatch_lp.py tests/test_scheme2r.py --basetemp D:\Paper\pytest_tmp_formal_v42_final -p no:cacheprovider -q`

Expected: PASS; no test mutates the completed run root or reads real 2020 without a valid fixture envelope.

- [ ] **Step 5: Commit the summarizer**

```powershell
git add scripts/summarize_rsc_pf_formal_v4_2.py tests/test_joint_dispatch_formal_v4_2_gate3.py
git commit -m "feat: seal formal v4.2 main results"
```

### Task 17: Close the Source Tree and Run Gate 0

**Files:**
- Verify: every path in `frame/configs/formal_v4_source_closure_v4_2.txt`
- Generated only: `frame/reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/gate0/*`

**Interfaces:**
- Consumes: committed implementations and tests from Tasks 1--16.
- Produces: immutable source/data/capacity/dependency/resource receipts and `authorized_pilot` under the single v4.2 run root.

- [ ] **Step 1: Verify that the source closure is complete and every listed path exists**

Run: `& $Py -c "from pathlib import Path; p=Path('configs/formal_v4_source_closure_v4_2.txt'); rows=[x.strip() for x in p.read_text(encoding='utf-8').splitlines() if x.strip() and not x.startswith('#')]; missing=[x for x in rows if not (Path('..')/x).exists()]; assert not missing, missing; print(len(rows))"`

Expected: a positive file count and no missing paths. The source closure includes the contract, all v4.2 modules, all gate/pilot/audit/summary entry points, the seed-extension entry point and every v4.2 test.

- [ ] **Step 2: Run the complete main-environment regression suite**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_*.py tests/test_joint_dispatch_formal_v4_*.py tests/test_dispatch_lp.py tests/test_scheme2r.py --basetemp D:\Paper\pytest_tmp_formal_v42_gate0 -p no:cacheprovider -q`

Expected: PASS. Any failure stops before the run root is created.

- [ ] **Step 3: Run the differentiable-LP environment regression suite**

Run: `& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_2_methods.py tests/test_joint_dispatch_formal_v4_diffopt.py tests/test_joint_dispatch_formal_v4_diffopt_runner.py -q`

Expected: PASS with CVXPY, CVXPYlayers, diffcp and the registered solver importable from the isolated environment.

- [ ] **Step 4: Confirm that no listed source file is dirty after the final implementation commit**

Run: `& $Py -c "import subprocess,pathlib; rows=[x.strip() for x in pathlib.Path('configs/formal_v4_source_closure_v4_2.txt').read_text(encoding='utf-8').splitlines() if x.strip() and not x.startswith('#')]; dirty=subprocess.check_output(['git','status','--porcelain','--']+rows, cwd='..', text=True); assert not dirty, dirty"`

Expected: no output. Unrelated pre-existing user changes outside the closure are allowed and remain untouched.

- [ ] **Step 5: Execute Gate 0 with one unique run id**

Run: `& $Py scripts/run_rsc_pf_formal_v4_2_gate0.py --contract configs/joint_forecast_dispatch_formal_v4_2.json --run-id formal_v4_2_20260904_a`

Expected: process exit 0 and `authorized_pilot=true`. Gate 0 persists new train/selection artifacts, train-only normalization, a train-only capacity receipt, source/dependency receipts, real forward/backward/HiGHS/DiffLP timings, and separate p95 projections for the pilot, Gate 1, Gate 2, two-seed extension and Gate 3. It records no 2020/2021 access.

- [ ] **Step 6: Inspect the immutable Gate 0 receipt**

Run: `& $Py -c "import json,pathlib; p=pathlib.Path('reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/gate0/GATE0_EVIDENCE.json'); d=json.loads(p.read_text()); assert d['authorized_pilot'] is True; assert d['evaluation_year_accessed'] is False; assert d['resource_projection']['synthetic_probe'] is False; print(d['resource_projection'])"`

Expected: assertions pass; the receipt, rather than an informal duration estimate, controls later launch expectations.

- [ ] **Step 7: Stop cleanly if Gate 0 fails**

On failure, keep the full run root and `GATE0_FAILURE.json`, do not run Task 18, and use a new run id after any source/config repair. On success, `protocol/CURRENT_GATE.json` is written once with the Gate 0 hash and `next_gate: pilot`.

### Task 18: Run the Non-Paper Engineering Pilot

**Files:**
- Generated only: `frame/reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/pilot/*`

**Interfaces:**
- Consumes: `authorized_pilot=true` from Task 17 and train years only.
- Produces: `PILOT_DECISION.json` with `paper_eligible=false` and `authorized_gate1`.

- [ ] **Step 1: Execute the one-seed pilot**

Run: `& $Py scripts/run_rsc_pf_formal_v4_2_pilot.py --contract configs/joint_forecast_dispatch_formal_v4_2.json --run-id formal_v4_2_20260904_a`

Expected: bounded multi-batch Stage P/S/J execution for seed 2026, at least 24 first-step settlements and no 2019/2020/2021 access. Duration is bounded by the Gate 0 pilot projection.

- [ ] **Step 2: Verify every pilot hard check**

Run: `& $Py -c "import json,pathlib; p=pathlib.Path('reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/pilot/PILOT_DECISION.json'); d=json.loads(p.read_text()); assert d['paper_eligible'] is False; assert d['authorized_gate1'] is True; assert all(d['checks'].values()); print(d['loss_summary'])"`

Expected: Stage P/S losses decrease, Stage J is finite, optimizer steps exceed one, Stage S clones match, joint decision gradient is positive, decoupled forecast decision gradient is zero, first-step carry is correct and shortage is below the registered pilot ceiling.

- [ ] **Step 3: Stop cleanly if the pilot fails**

Keep the pilot failure receipt, do not open 2019 through Gate 1, and do not treat pilot numbers as model-selection evidence. Any repair requires a new source manifest and new run id beginning again at Gate 0.

### Task 19: Run Gate 1 and Freeze the Selection Protocol

**Files:**
- Generated only: `frame/reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/gate1/*`

**Interfaces:**
- Consumes: valid pilot receipt and the pre-registered representative 2019 origin manifest.
- Produces: immutable `GATE1_FREEZE.json` with `authorized_gate2`.

- [ ] **Step 1: Execute representative Gate 1**

Run: `& $Py scripts/run_rsc_pf_formal_v4_2_gate1.py --contract configs/joint_forecast_dispatch_formal_v4_2.json --run-id formal_v4_2_20260904_a`

Expected: exit 0, candidate rows for only registered candidates, exact origin indices/weights/timestamps and activity/season diagnostics. Duration is the Gate 0 Gate 1 p95 projection.

- [ ] **Step 2: Verify the freeze receipt**

Run: `& $Py -c "import json,pathlib; p=pathlib.Path('reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/gate1/GATE1_FREEZE.json'); d=json.loads(p.read_text()); assert d['authorized_gate2'] is True; assert d['activity_fraction']['cooling'] >= .20; assert d['activity_fraction']['heating'] >= .20; assert d['evaluation_year_accessed'] is False; print(d['frozen_training'])"`

Expected: all assertions pass, and the printed block contains every epoch, minimum epoch, patience, validation interval, learning rate, curriculum and candidate value required by Gate 2.

- [ ] **Step 3: Stop cleanly if Gate 1 fails**

Retain all candidate receipts and the failure decision; do not run Gate 2. Changing the manifest, candidates or scoring rule requires a new protocol/run id, not an edit inside this lineage.

### Task 20: Run the Complete Three-Seed Gate 2 Matrix

**Files:**
- Generated only: `frame/reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/gate2/*`

**Interfaces:**
- Consumes: `authorized_gate2=true`, frozen Gate 1 settings and 2019 chronology.
- Produces: 23 complete method rows, seed checkpoints, raw chronological arrays and provisional `GATE2_DECISION.json`.

- [ ] **Step 1: Launch or resume Gate 2**

Run: `& $Py scripts/run_rsc_pf_formal_v4_2_gate2.py --contract configs/joint_forecast_dispatch_formal_v4_2.json --run-id formal_v4_2_20260904_a`

Expected: seven stochastic methods × seeds 2026/2027/2028 plus Perfect-Information-MPC and Seasonal-Naive-PTO once. The runner resumes only hash-identical completed rows and writes a failure receipt for each unsuccessful row. Runtime may be many hours on CPU and is governed by the Gate 0 p95 projection.

- [ ] **Step 2: Verify matrix completeness without trusting the decision**

Run: `& $Py -c "import json,pathlib; p=pathlib.Path('reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/gate2/GATE2_DECISION.json'); d=json.loads(p.read_text()); assert d['expected_rows']==23; assert d['completed_rows']==23; assert not d['missing_rows']; assert d['evaluation_year_accessed'] is False; print(d['provisional_checks'])"`

Expected: 23/23 rows, no missing/failure rows and no 2020/2021 access. This check does not itself authorize Gate 3.

- [ ] **Step 3: Stop cleanly on an incomplete or failed matrix**

If any row failed, rerunning the identical command may resume incomplete work. A scientific criterion failure after all rows complete is final for this run id; do not change thresholds, budgets or candidates in place.

### Task 21: Run the Independent Gate 2 Audit

**Files:**
- Generated only: `frame/reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/audit/GATE2_INDEPENDENT_AUDIT.json`
- Generated conditionally: `frame/reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/protocol/SEED_EXTENSION_AUTHORIZATION.json`

**Interfaces:**
- Consumes: closed raw Gate 2 arrays and immutable upstream receipts.
- Produces: independently recomputed decision and, only on pass, frozen seed-extension authorization.

- [ ] **Step 1: Execute the independent audit**

Run: `& $Py scripts/audit_rsc_pf_formal_v4_2_gate2.py --contract configs/joint_forecast_dispatch_formal_v4_2.json --run-id formal_v4_2_20260904_a`

Expected: raw metrics, method/checkpoint identities, primary seed directions, shortage ratio, forecast guardrails, physical tolerances, method coverage and access boundaries are recomputed rather than copied.

- [ ] **Step 2: Inspect the audit decision**

Run: `& $Py -c "import json,pathlib; p=pathlib.Path('reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/audit/GATE2_INDEPENDENT_AUDIT.json'); d=json.loads(p.read_text()); print(json.dumps({'authorized_seed_extension':d['authorized_seed_extension'],'failed_checks':d['failed_checks'],'primary':d['primary_contrast'],'shortage_ratio':d['shortage_ratio']},indent=2))"`

Expected on pass: `authorized_seed_extension=true`, no failed checks and an authorization file that still has `allow_evaluation_year=false`. Expected on fail: diagnostic receipt only and no authorization.

- [ ] **Step 3: Stop before five-seed expansion when unauthorized**

If the audit fails, stop with 2020 unopened. The run remains useful engineering evidence but is not a paper result.

### Task 22: Run the Frozen Two-Seed Extension and Mint Gate 3 Authorization

**Files:**
- Generated only: `frame/reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/seed_extension/*`
- Generated conditionally: `frame/reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/protocol/GATE3_AUTHORIZATION.json`

**Interfaces:**
- Consumes: valid seed-extension authorization, frozen configuration and 2015--2019 only.
- Produces: checkpoint-complete seeds 2026--2030 and a one-time unconsumed Gate 3 envelope.

- [ ] **Step 1: Execute the frozen 2029/2030 extension**

Run: `& $Py scripts/extend_rsc_pf_formal_v4_2_seeds.py --contract configs/joint_forecast_dispatch_formal_v4_2.json --run-id formal_v4_2_20260904_a`

Expected: only seeds 2029 and 2030 are newly trained, using the exact Gate 1 freeze; no candidate search, threshold change or 2020 access occurs. Runtime is the Gate 0 seed-extension p95 projection.

- [ ] **Step 2: Verify five-seed checkpoint readiness and envelope state**

Run: `& $Py -c "import json,pathlib; root=pathlib.Path('reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a'); a=json.loads((root/'seed_extension/SEED_EXTENSION_AUDIT.json').read_text()); e=json.loads((root/'protocol/GATE3_AUTHORIZATION.json').read_text()); assert a['all_five_seeds_ready'] is True; assert a['evaluation_year_accessed'] is False; assert e['allowed_years']==[2020] and e['consumed'] is False; print(a['checkpoint_coverage'])"`

Expected: every required stochastic method has valid checkpoints for seeds 2026--2030; the envelope is unconsumed.

- [ ] **Step 3: Stop if extension lineage is incomplete**

Any missing checkpoint, changed hash or access violation suppresses the Gate 3 envelope and leaves 2020 unopened.

### Task 23: Run the One-Time Locked 2020 Gate 3 Evaluation

**Files:**
- Generated conditionally: `frame/reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/gate3/*`

**Interfaces:**
- Consumes: valid unconsumed Gate 3 envelope and frozen five-seed checkpoint set.
- Produces: locked 2020 main evaluation and `GATE3_COMPLETE.json`.

- [ ] **Step 1: Execute Gate 3 once**

Run: `& $Py scripts/run_rsc_pf_formal_v4_2_gate3.py --contract configs/joint_forecast_dispatch_formal_v4_2.json --run-id formal_v4_2_20260904_a`

Expected: the envelope is consumed atomically; 2020 is read once; no training, tuning, early stopping or checkpoint selection occurs; all stochastic methods run for five seeds and deterministic references once. Runtime is the Gate 0 Gate 3 p95 projection.

- [ ] **Step 2: Verify the locked evaluation receipt**

Run: `& $Py -c "import json,pathlib; root=pathlib.Path('reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a'); d=json.loads((root/'gate3/GATE3_COMPLETE.json').read_text()); c=json.loads((root/'protocol/GATE3_AUTHORIZATION_CONSUMED.json').read_text()); assert d['complete'] is True; assert d['training_calls']==0; assert d['retuning_events']==0; assert c['allowed_years']==[2020]; assert d['accessed_years']==[2020]; print(d['row_coverage'])"`

Expected: full five-seed/deterministic coverage, zero training/retuning and no 2021 access.

- [ ] **Step 3: Do not retry a consumed evaluation under the same run id**

A second launch must fail before data access. A runtime interruption is recoverable only through row-level immutable resume that uses the already-consumed envelope's matching execution id; it cannot start a second evaluation or alter any checkpoint.

### Task 24: Generate and Verify the Final Main-Result Bundle

**Files:**
- Generated conditionally: `frame/reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/summary/MAIN_RESULTS.json`
- Generated conditionally: `frame/reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/summary/MAIN_RESULTS.csv`
- Generated conditionally: `frame/reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/summary/FORMAL_V4_2_AUDIT.md`

**Interfaces:**
- Consumes: raw Gate 3 arrays and complete lineage.
- Produces: sealed machine-readable tables and a human-readable audit; no figures, ablations or manuscript edits.

- [ ] **Step 1: Generate the result bundle**

Run: `& $Py scripts/summarize_rsc_pf_formal_v4_2.py --contract configs/joint_forecast_dispatch_formal_v4_2.json --run-id formal_v4_2_20260904_a`

Expected: task×horizon forecast metrics, rigid macro/gas/active-hour/seasonal diagnostics, dispatch/shortage/physics/runtime tables, optimizer calls, the paired primary contrast and 168-hour/2,000-replicate uncertainty are recomputed from raw arrays.

- [ ] **Step 2: Verify paper eligibility and raw-array hashes**

Run: `& $Py -c "import json,pathlib; p=pathlib.Path('reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260904_a/summary/MAIN_RESULTS.json'); d=json.loads(p.read_text()); assert d['paper_eligible'] is True; assert d['primary_contrast']=='RSC-PF_minus_Decoupled-RSC-PF'; assert d['bootstrap']=={'block_hours':168,'replicates':2000}; assert all(d['raw_array_hashes_verified'].values()); print(d['headline_results'])"`

Expected: all assertions pass. If an earlier gate failed, the summarizer instead writes only `FORMAL_V4_2_AUDIT.md` with `paper_eligible=false`.

- [ ] **Step 3: Run the final read-only verification suite**

Run: `& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_*.py tests/test_joint_dispatch_formal_v4_*.py tests/test_dispatch_lp.py tests/test_scheme2r.py --basetemp D:\Paper\pytest_tmp_formal_v42_final -p no:cacheprovider -q`

Expected: PASS without modifying any completed receipt or opening 2020 outside the completed Gate 3 access record.

- [ ] **Step 4: Stop at the agreed boundary**

Report the experiment results and audit status. Do not start ablations, edit either manuscript language version, or access 2021 in this plan.


---

## Mandatory Stop/Go Sequence

1. Complete and commit Tasks 1--16; every downstream executable and test must exist before source closure.
2. Run Task 17. Gate 0 hashes the final source tree and authorizes only the engineering pilot.
3. Run Task 18. The pilot must authorize Gate 1 and remains non-paper evidence.
4. Run Task 19. Gate 1 freezes every Gate 2 choice and authorizes Gate 2.
5. Run Task 20. Gate 2 completes the fixed 2019 three-seed matrix; no 2020 access occurs.
6. Run Task 21. Only the independent Gate 2 audit may authorize a frozen seed extension.
7. Run Task 22. Train and audit seeds 2029/2030 with unchanged settings and no 2020 access; only this audit may mint the Gate 3 envelope.
8. Run Task 23 only when that envelope exists and is unconsumed.
9. Run Task 24 and stop after sealed main results; do not begin ablations or manuscript edits.

## Final Verification Commands

```powershell
$Repo = 'D:\Paper\github_work\paper-code'
$Frame = Join-Path $Repo 'frame'
$Py = (Get-Command python).Source
Set-Location $Frame
& $Py -m pytest tests/test_joint_dispatch_formal_v4_2_*.py tests/test_joint_dispatch_formal_v4_*.py tests/test_dispatch_lp.py tests/test_scheme2r.py --basetemp D:\Paper\pytest_tmp_formal_v42_final -p no:cacheprovider -q
git status --short
```

Expected: all tests pass. `git status --short` may still show the user's pre-existing unrelated changes, but none may have been staged or overwritten by this plan. Generated run artifacts remain outside Git unless the repository's evidence policy explicitly tracks a compact receipt.
