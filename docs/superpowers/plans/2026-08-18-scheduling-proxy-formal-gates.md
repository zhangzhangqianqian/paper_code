# 调度代理模型 V2 正式 Gate 4/5 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 建立“正式训练与验证验收”和“一次性测试评估”严格分离的运行流程，确保测试集不能参与模型选择，并用哈希冻结所有进入测试阶段的证据。

**架构：** 新增纯数据的 Gate 4/5 契约与验收模块，并新增一个具有 `prepare`、`validate`、`test` 三个互斥阶段的正式运行脚本。现有调度代理流水线只增加显式评估数据划分接口；验证通过后写入冻结记录并停止，测试必须由用户另行执行，且完成后拒绝第二次运行。

**技术栈：** Python 3.9、PyTorch、NumPy、SciPy（仅用于离线 LP 教师标签）、pytest、JSON/NPZ、SHA-256、原子文件替换。

## 全局约束

- 正式调度代理固定为 `feasible_scheduling_proxy_v2`，输出形状固定为 `[N,4,21]`。
- 正式数据规模固定为 train/validation/test = `8192/2048/2048`，种子固定为 `2026/2027/2028`。
- 模型选择只能使用验证集；Gate 4 前不得加载测试集计算预测或指标。
- Gate 4 通过后必须停止；Gate 5 只能通过单独的 `test` 命令启动。
- Gate 5 不提供 `--force`、覆盖或自动重试选项。
- v2 推理不得导入或调用 LP/优化器，不得使用回退，不得使用标签派生的 gas prior。
- 不修改 Scheme2R、预测模型、v1 调度代理、已有正式结果、论文或图件。
- 实施阶段只运行小规模测试夹具，不启动全规模正式训练。
- 新计划、设计说明和用户文档默认使用中文；代码标识符和固定专业术语保留英文。

---

## 文件结构与职责

### 新建文件

- `frame/configs/scheduling_proxy_formal_gate_v2.json`：冻结 Gate 4 阈值、延迟测量口径和一次性测试策略。
- `frame/src/scheduling/proxy_formal_gate.py`：解析 Gate 契约、计算验收条件、构建并校验冻结记录、原子写入 JSON。
- `frame/scripts/run_scheduling_proxy_formal.py`：`prepare`、`validate`、`test` 三阶段命令入口。
- `frame/tests/test_scheduling_proxy_formal_gate.py`：阈值边界、非有限值、冻结哈希和防篡改测试。
- `frame/tests/test_scheduling_proxy_formal_runner.py`：三阶段编排、数据隔离、一次性测试和中断标记测试。

### 修改文件

- `frame/scripts/run_scheduling_proxy_pipeline.py`：让 `evaluate_from_dataset` 接受显式 `evaluation_split`，保持现有 CLI 语义。
- `frame/src/scheduling/proxy_evaluation.py`：允许冻结延迟测量的 batch size、warm-up 和 repeats；默认行为不变。
- `frame/tests/test_scheduling_proxy_pipeline.py`：补充显式 validation/test 路由回归测试。
- `frame/tests/test_scheduling_proxy_diagnostics.py`：验证正式验证与测试审计使用正确的数据划分名称。

---

### 任务 1：建立机器可读的 Gate 4/5 契约与纯验收逻辑

**文件：**

- 新建：`frame/configs/scheduling_proxy_formal_gate_v2.json`
- 新建：`frame/src/scheduling/proxy_formal_gate.py`
- 新建：`frame/tests/test_scheduling_proxy_formal_gate.py`

**接口：**

- 输入：正式 Gate JSON、validation 指标、约束诊断和物理预测数组。
- 输出：`FormalGateContract`、`GateConditionResult`、`GateDecision`。

测试文件先定义 `valid_gate_payload()`、`write_json()`、`passing_inputs()` 和 `set_metric()` 四个本地夹具；它们分别返回上述固定 JSON、写入 UTF-8 JSON、构造恰好满足全部阈值的指标/诊断/预测，以及按正式指标路径修改单个数值。

- [ ] **步骤 1：编写 Gate 契约解析失败测试**

```python
def test_gate_contract_rejects_auto_test_and_negative_threshold(tmp_path):
    payload = valid_gate_payload()
    payload["workflow"]["automatic_test_after_validation"] = True
    path = write_json(tmp_path / "gate.json", payload)
    with pytest.raises(ValueError, match="automatic test"):
        load_formal_gate_contract(path)

    payload = valid_gate_payload()
    payload["thresholds"]["cost_regret_relative"] = -0.1
    path = write_json(tmp_path / "gate.json", payload)
    with pytest.raises(ValueError, match="non-negative"):
        load_formal_gate_contract(path)
```

- [ ] **步骤 2：运行测试并确认失败**

```powershell
& $py -m pytest frame/tests/test_scheduling_proxy_formal_gate.py -q
```

预期：因 `proxy_formal_gate.py` 尚不存在而失败。

- [ ] **步骤 3：写入冻结 Gate JSON**

```json
{
  "schema_version": "scheduling-proxy-formal-gate-v2",
  "scheduler_schema_version": "scheduling-proxy-contract-v2",
  "thresholds": {
    "raw_feasible_rate": 1.0,
    "max_invariant_residual": 1e-6,
    "max_mean_slack": 1e-6,
    "max_peak_slack": 1e-6,
    "fallback_rate": 0.0,
    "inference_exact_lp_calls": 0,
    "objective_gap_relative_absolute_mean": 0.05,
    "cost_regret_relative": 0.05,
    "carbon_relative_error": 0.10,
    "proxy_batch_median_ms": 10.0
  },
  "latency_protocol": {
    "batch_size": 2048,
    "warmup_runs": 2,
    "repeats": 5,
    "device": "cpu"
  },
  "workflow": {
    "automatic_test_after_validation": false,
    "allow_force_test": false,
    "allow_test_overwrite": false
  }
}
```

- [ ] **步骤 4：实现不可变数据类和严格解析器**

```python
@dataclass(frozen=True)
class FormalGateContract:
    schema_version: str
    scheduler_schema_version: str
    thresholds: Mapping[str, float]
    latency_protocol: Mapping[str, int | str]
    workflow: Mapping[str, bool]

@dataclass(frozen=True)
class GateConditionResult:
    name: str
    measured: float
    comparator: str
    threshold: float
    passed: bool

@dataclass(frozen=True)
class GateDecision:
    schema_version: str
    status: str
    conditions: tuple[GateConditionResult, ...]
```

解析器必须拒绝缺字段、未知字段、布尔值伪装成数值、负阈值、非有限数值、自动测试和强制覆盖。

- [ ] **步骤 5：编写验收边界测试**

覆盖：等于阈值时通过、任一指标刚高于阈值时失败、可行率低于 1 时失败、NaN/Infinity/缺字段失败。

```python
@pytest.mark.parametrize("field", [
    "objective_gap_relative_absolute_mean",
    "cost_regret_relative",
    "carbon_relative_error",
    "proxy_batch_median_ms",
])
def test_gate_fails_immediately_above_threshold(field):
    metrics, diagnostics, prediction = passing_inputs()
    assert evaluate_gate4_acceptance(
        metrics, diagnostics, prediction, contract()
    ).status == "passed"
    set_metric(metrics, field, contract().thresholds[field] + 1e-12)
    assert evaluate_gate4_acceptance(
        metrics, diagnostics, prediction, contract()
    ).status == "failed"
```

- [ ] **步骤 6：实现 `evaluate_gate4_acceptance`**

必须从 `raw_proxy` 和 `proxy_latency` 读取指标，对全部物理残差取最大值，从 `[N,4,21]` 预测数组直接计算三类 slack 的 mean/max，不使用四舍五入后的数值；任何非有限值或结构错误均失败关闭，并返回固定顺序的全部条件。

- [ ] **步骤 7：运行测试并提交**

```powershell
& $py -m pytest frame/tests/test_scheduling_proxy_formal_gate.py -q
git add frame/configs/scheduling_proxy_formal_gate_v2.json frame/src/scheduling/proxy_formal_gate.py frame/tests/test_scheduling_proxy_formal_gate.py
git commit -m "feat: add formal scheduler gate contract"
```

---

### 任务 2：增加显式 validation/test 评估路由

**文件：**

- 修改：`frame/scripts/run_scheduling_proxy_pipeline.py`
- 修改：`frame/src/scheduling/proxy_evaluation.py`
- 修改：`frame/tests/test_scheduling_proxy_pipeline.py`
- 修改：`frame/tests/test_scheduling_proxy_diagnostics.py`

**接口：**

- 输入：明确命名的 `evaluation_split="validation" | "test"`。
- 输出：对应的 `metrics_<split>.json` 与 `predictions_<split>.npz`。

- [ ] **步骤 1：编写显式路由失败测试**

```python
def test_explicit_validation_evaluation_never_loads_test(tmp_path, monkeypatch):
    output = make_prepared_bundle(tmp_path)
    original = pipeline.load_proxy_split

    def guarded(path, *args, **kwargs):
        if Path(path).name == "test.npz":
            raise AssertionError("test split was accessed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pipeline, "load_proxy_split", guarded)
    pipeline.evaluate_from_dataset(
        contract(), BENCHMARK, output,
        evaluation_split="validation", smoke=False,
    )
    assert (output / "metrics_validation.json").exists()
    assert not (output / "metrics_test.json").exists()
```

- [ ] **步骤 2：运行测试并确认失败**

```powershell
& $py -m pytest frame/tests/test_scheduling_proxy_pipeline.py -q
```

预期：函数尚不接受显式 split，因此失败。

- [ ] **步骤 3：实现显式评估接口**

```python
def evaluate_from_dataset(
    contract: ProxyContract,
    benchmark_path: Path,
    output_dir: Path,
    *,
    evaluation_split: Literal["validation", "test"],
    smoke: bool = False,
) -> dict[str, Any]:
    return evaluate_named_split_and_write_artifacts(
        contract=contract,
        benchmark_path=benchmark_path,
        output_dir=output_dir,
        evaluation_split=evaluation_split,
        smoke=smoke,
    )
```

现有 CLI 内部映射保持：v2 smoke -> validation，普通 evaluate -> test。

- [ ] **步骤 4：冻结延迟协议**

为 `evaluate_model_on_split` 增加 `latency_repeats=5` 与 `latency_warmup=2` 可选参数并传入 `measure_proxy_latency`。正式验证必须用完整的 2048 个验证场景作为一个 batch；Gate 4 读取 batch median，不得使用逐场景延迟替代。

- [ ] **步骤 5：补充 test 路由与非法 split 回归测试**

验证显式 `test` 只写 test 文件，provenance 中 `split == "test"`，非法 split 被拒绝。

- [ ] **步骤 6：运行并提交**

```powershell
& $py -m pytest frame/tests/test_scheduling_proxy_pipeline.py frame/tests/test_scheduling_proxy_diagnostics.py -q
git add frame/scripts/run_scheduling_proxy_pipeline.py frame/src/scheduling/proxy_evaluation.py frame/tests/test_scheduling_proxy_pipeline.py frame/tests/test_scheduling_proxy_diagnostics.py
git commit -m "feat: add explicit scheduler evaluation split"
```

---

### 任务 3：实现冻结记录和全链路哈希校验

**文件：**

- 修改：`frame/src/scheduling/proxy_formal_gate.py`
- 修改：`frame/tests/test_scheduling_proxy_formal_gate.py`

**接口：**

- 输入：正式输出目录与 Gate 4 决策。
- 输出：`formal_freeze.json` 与 `validate_formal_freeze(output_dir: Path, freeze: Mapping[str, Any]) -> None`。

- [ ] **步骤 1：编写冻结文件完整性测试**

冻结记录必须绑定：

```text
benchmark
scheduler contract
formal gate contract
dataset/train.npz
dataset/validation.npz
dataset/test.npz
dataset/manifest.json
normalization_stats.npz
best_model.pt
history.json
metrics_validation.json
predictions_validation.npz
validation_constraint_diagnostics.json
formal_validation_acceptance.json
```

- [ ] **步骤 2：编写逐文件篡改拒绝测试**

```python
@pytest.mark.parametrize("relative_path", FROZEN_ARTIFACTS)
def test_freeze_rejects_any_changed_artifact(tmp_path, relative_path):
    root, freeze = make_frozen_bundle(tmp_path)
    mutate_one_byte(resolve_artifact(root, relative_path))
    with pytest.raises(ValueError, match="SHA-256"):
        validate_formal_freeze(root, freeze)
```

- [ ] **步骤 3：实现原子 JSON 写入**

```python
def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)
```

- [ ] **步骤 4：实现冻结记录构建和校验**

还要保存：`status="frozen"`、全部 Gate 条件、模型/解码器版本、三个 split 的 seed/digest、Python/PyTorch/NumPy 版本、CPU/线程/延迟协议、UTC 时间戳和 `test_evaluation_started=false`。

- [ ] **步骤 5：运行并提交**

```powershell
& $py -m pytest frame/tests/test_scheduling_proxy_formal_gate.py -q
git add frame/src/scheduling/proxy_formal_gate.py frame/tests/test_scheduling_proxy_formal_gate.py
git commit -m "feat: freeze formal scheduler evidence"
```

---

### 任务 4：实现 prepare 和 validate 正式阶段

**文件：**

- 新建：`frame/scripts/run_scheduling_proxy_formal.py`
- 新建：`frame/tests/test_scheduling_proxy_formal_runner.py`

**接口：**

- 命令：`--stage prepare` 与 `--stage validate`。
- 输出：prepare 清单、验证证据、Gate 4 决策和可选冻结记录。

- [ ] **步骤 1：编写 prepare 隔离测试**

使用临时小规模契约，要求非空输出目录被拒绝；成功后存在训练数据、统计量、检查点和 history；不存在 test 指标、test 预测或测试收据；history 明确 `selection_split=validation` 和 `test_split_used_for_selection=false`。

- [ ] **步骤 2：编写 validate 成败测试**

通过夹具注入可控指标：全部条件满足时写 acceptance/freeze；任一条件失败时只写 acceptance 并返回非零；validate 若加载 `test.npz`，测试立即失败。

- [ ] **步骤 3：实现 CLI 参数**

```python
parser.add_argument("--stage", choices=("prepare", "validate", "test"), required=True)
parser.add_argument("--benchmark", required=True)
parser.add_argument("--contract", required=True)
parser.add_argument("--gate-contract", required=True)
parser.add_argument("--output-dir", required=True)
```

不得加入 `--force`、`--overwrite` 或 `--auto-test`。

- [ ] **步骤 4：实现 prepare**

```python
generate_dataset(contract, benchmark_path, output_dir, smoke=False)
train_from_dataset(contract, benchmark_path, output_dir, smoke=False)
```

完成后对全部 prepare 工件计算 SHA-256，并原子写入 `formal_prepare_manifest.json`。

- [ ] **步骤 5：实现 validate**

```text
校验 prepare 清单
-> 显式 validation 评估
-> 生成 validation diagnostics
-> 重新计算指标一致性
-> 计算 GateDecision
-> 写 acceptance
-> 仅在全通过时写 freeze
-> 立即退出，不调用 test
```

- [ ] **步骤 6：运行并提交**

```powershell
& $py -m pytest frame/tests/test_scheduling_proxy_formal_runner.py -q
git add frame/scripts/run_scheduling_proxy_formal.py frame/tests/test_scheduling_proxy_formal_runner.py
git commit -m "feat: add formal scheduler prepare and validation gates"
```

---

### 任务 5：实现人工触发的一次性 Gate 5 测试

**文件：**

- 修改：`frame/scripts/run_scheduling_proxy_formal.py`
- 修改：`frame/src/scheduling/proxy_formal_gate.py`
- 修改：`frame/tests/test_scheduling_proxy_formal_runner.py`
- 修改：`frame/tests/test_scheduling_proxy_formal_gate.py`

**接口：**

- 命令：`--stage test`。
- 输出：一次性的 test 指标、预测、诊断、manifest 和 receipt。

- [ ] **步骤 1：编写未冻结时拒绝测试**

```python
def test_test_stage_refuses_without_passing_freeze(tmp_path):
    result = run_cli("test", output_dir=prepared_but_unfrozen(tmp_path))
    assert result.returncode != 0
    assert not any_test_output_exists(tmp_path)
```

- [ ] **步骤 2：编写重复测试和中断标记测试**

已有 receipt、任一完整 test 输出或 `formal_test_in_progress.json` 时必须拒绝，而且拒绝必须发生在 `test.npz` 被加载之前。

- [ ] **步骤 3：实现测试前完整校验**

```text
检查不存在 test 工件
-> 读取 passing acceptance 与 freeze
-> 验证全部冻结哈希
-> 验证模型/解码器/no-LP/no-fallback 元数据
-> 原子建立 in-progress 标记
-> 首次加载 test.npz
```

- [ ] **步骤 4：实现原子测试输出**

先写同目录临时文件，审计成功后再替换为：

```text
metrics_test.json
predictions_test.npz
test_constraint_diagnostics.json
formal_test_manifest.json
formal_test_receipt.json
```

收据最后写入；只有收据存在才表示 Gate 5 完成。

- [ ] **步骤 5：实现失败行为**

失败时保留 `formal_test_in_progress.json` 和错误信息，不自动删除或重试。

- [ ] **步骤 6：运行并提交**

```powershell
& $py -m pytest frame/tests/test_scheduling_proxy_formal_runner.py frame/tests/test_scheduling_proxy_formal_gate.py -q
git add frame/scripts/run_scheduling_proxy_formal.py frame/src/scheduling/proxy_formal_gate.py frame/tests/test_scheduling_proxy_formal_runner.py frame/tests/test_scheduling_proxy_formal_gate.py
git commit -m "feat: enforce one-time formal scheduler test"
```

---

### 任务 6：完成回归、干运行和最终审计

**文件：**

- 仅在测试暴露缺陷时修改本计划范围内文件。
- 不修改 v1、Scheme2R、正式结果、论文、图件和现有用户未提交文件。

- [ ] **步骤 1：运行新增和相关测试**

```powershell
& $py -m pytest `
  frame/tests/test_scheduling_proxy_formal_gate.py `
  frame/tests/test_scheduling_proxy_formal_runner.py `
  frame/tests/test_scheduling_proxy_pipeline.py `
  frame/tests/test_scheduling_proxy_diagnostics.py -q
```

- [ ] **步骤 2：运行完整测试**

```powershell
& $py -m pytest frame/tests -q
```

预期：无失败；如实记录已有跳过项和非阻断警告。

- [ ] **步骤 3：运行小规模 prepare 和 validate**

使用测试专用小契约和新临时目录，确认 freeze 存在，而 test 指标、预测、manifest 和 receipt 均不存在。

- [ ] **步骤 4：单独运行一次小规模 test**

确认 test receipt 存在且哈希完整，test provenance 为 `split=test`，物理可行率 100%、回退 0、LP 推理 0，并确认第二次运行被拒绝。

- [ ] **步骤 5：检查工作树边界**

```powershell
git diff --check
git status --short
```

确认以下已有用户文件没有被暂存或修改：

```text
frame/tests/test_scheduling_proxy_repairs.py
frame/scripts/compare_scheduling_proxy_repairs.py
docs/superpowers/plans/2026-08-17-scheduling-proxy-feasibility-repair.md
```

- [ ] **步骤 6：只提交本计划范围内的必要修正**

```powershell
git add `
  frame/configs/scheduling_proxy_formal_gate_v2.json `
  frame/src/scheduling/proxy_formal_gate.py `
  frame/scripts/run_scheduling_proxy_formal.py `
  frame/scripts/run_scheduling_proxy_pipeline.py `
  frame/src/scheduling/proxy_evaluation.py `
  frame/tests/test_scheduling_proxy_formal_gate.py `
  frame/tests/test_scheduling_proxy_formal_runner.py `
  frame/tests/test_scheduling_proxy_pipeline.py `
  frame/tests/test_scheduling_proxy_diagnostics.py
git commit -m "test: verify formal scheduler gate workflow"
```

- [ ] **步骤 7：只交付、不执行正式命令**

交付三条独立 PowerShell 命令：

```powershell
# 1. prepare：正式数据生成与训练
# 2. validate：正式验证集验收并冻结，执行后停止
# 3. test：仅在 Gate 4 通过后由用户单独运行一次
```

实施阶段不得运行正式 8192/2048/2048 训练，也不得自动执行第三条命令。

---

## 完成验收清单

- [ ] prepare 不读取测试集产生模型指标或预测。
- [ ] validate 只使用验证集并生成逐条件 Gate 4 报告。
- [ ] Gate 4 失败时不存在冻结文件。
- [ ] Gate 4 通过时冻结记录绑定全部输入、模型和验证证据。
- [ ] validate 完成后程序停止，不自动执行测试。
- [ ] test 只能通过单独命令启动。
- [ ] 任一冻结工件被修改都会阻止测试。
- [ ] Gate 5 成功后存在唯一收据，第二次测试被拒绝。
- [ ] v2 推理保持 100% 物理可行、零回退、零 LP 推理。
- [ ] v1、Scheme2R 和既有结果未被改变。
