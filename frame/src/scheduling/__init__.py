"""双轨预测—调度模块的契约与参数审计工具。"""

from importlib import import_module

from .contracts import (
    DEFAULT_SCHEDULING_CONTRACT_PATH,
    SchedulingContract,
    load_scheduling_contract,
    validate_scheduling_contract,
)
from .parameter_audit import (
    AuditReport,
    ParameterLedger,
    audit_identifiability,
    read_parameter_ledger,
)
from .data import (
    PlanningInformation,
    RealizedInformation,
    SchedulingFrame,
    build_scheduling_frame,
    make_plan_view,
    make_settlement_view,
)
from .renewable_forecasts import FrozenRenewableForecaster, fit_renewable_forecaster
from .renewables import pv_available, wt_available
from .real_replay import (
    EnergyNomination,
    ReplayResult,
    build_energy_nomination,
    settle_first_step_replay,
    settle_real_replay,
)
from .imbalance_settlement import summarize_replay
from .benchmark_parameters import SchedulingParameters, derive_benchmark_parameters

# LP/rolling-horizon modules are intentionally lazy.  The V2 neural inference
# graph can therefore import without SciPy; callers that request the historical
# solver APIs still receive the same public symbols on first access.
_LAZY_EXPORTS = {
    "DispatchInputs": ("dispatch_lp", "DispatchInputs"),
    "DispatchResult": ("dispatch_lp", "DispatchResult"),
    "solve_dispatch_lp": ("dispatch_lp", "solve_dispatch_lp"),
    "RealizedStep": ("recourse", "RealizedStep"),
    "evaluate_planned_first_step": ("recourse", "evaluate_planned_first_step"),
    "settle_first_step": ("recourse", "settle_first_step"),
    "build_oracle_forecast_set": ("oracle", "build_oracle_forecast_set"),
    "run_perfect_information_oracle": ("oracle", "run_perfect_information_oracle"),
    "ActualStream": ("rolling_horizon", "ActualStream"),
    "RollingForecastSet": ("rolling_horizon", "RollingForecastSet"),
    "RollingResult": ("rolling_horizon", "RollingResult"),
    "run_rolling_dispatch": ("rolling_horizon", "run_rolling_dispatch"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value

__all__ = [
    "AuditReport",
    "PlanningInformation",
    "RealizedInformation",
    "SchedulingFrame",
    "FrozenRenewableForecaster",
    "DEFAULT_SCHEDULING_CONTRACT_PATH",
    "ParameterLedger",
    "SchedulingContract",
    "audit_identifiability",
    "build_scheduling_frame",
    "load_scheduling_contract",
    "make_plan_view",
    "make_settlement_view",
    "fit_renewable_forecaster",
    "pv_available",
    "wt_available",
    "EnergyNomination",
    "ReplayResult",
    "build_energy_nomination",
    "settle_first_step_replay",
    "settle_real_replay",
    "summarize_replay",
    "SchedulingParameters",
    "derive_benchmark_parameters",
    "DispatchInputs",
    "DispatchResult",
    "solve_dispatch_lp",
    "RealizedStep",
    "evaluate_planned_first_step",
    "build_oracle_forecast_set",
    "run_perfect_information_oracle",
    "settle_first_step",
    "ActualStream",
    "RollingForecastSet",
    "RollingResult",
    "run_rolling_dispatch",
    "read_parameter_ledger",
    "validate_scheduling_contract",
]
