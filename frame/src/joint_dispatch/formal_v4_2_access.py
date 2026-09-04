"""Fail-closed, auditable dataset access for formal-v4.2."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .formal_v4_2_artifacts import canonical_sha256, sha256_file, write_once_json


SPLIT_YEARS = {
    "train": (2015, 2016, 2017, 2018),
    "selection": (2019,),
    "evaluation": (2020,),
}
GATE3_ENVELOPE_SCHEMA = "formal-v4.2-gate3-authorization-v1"


class EvaluationAccessDenied(PermissionError):
    """Raised before a forbidden year can be opened."""


def validate_gate3_envelope(
    envelope: Mapping[str, Any],
    contract_sha256: str,
) -> None:
    if envelope.get("schema") != GATE3_ENVELOPE_SCHEMA:
        raise EvaluationAccessDenied("invalid Gate 3 authorization schema")
    if envelope.get("contract_sha256") != contract_sha256:
        raise EvaluationAccessDenied("Gate 3 contract hash mismatch")
    if envelope.get("allowed_years") != [2020]:
        raise EvaluationAccessDenied("Gate 3 authorization does not allow exactly 2020")
    if envelope.get("consumed") is not False:
        raise EvaluationAccessDenied("Gate 3 authorization is already consumed")
    for field in ("gate2_audit_sha256", "seed_extension_audit_sha256"):
        value = str(envelope.get(field, ""))
        if len(value) != 64:
            raise EvaluationAccessDenied(f"Gate 3 authorization is missing {field}")


@dataclass(frozen=True)
class AccessEventV42:
    split: str
    years: tuple[int, ...]
    caller: str
    decision: str
    reason: str
    path: str = ""
    path_sha256: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "formal-v4.2-access-event-v1",
            "split": self.split,
            "years": list(self.years),
            "caller": self.caller,
            "decision": self.decision,
            "reason": self.reason,
            "path": self.path,
            "path_sha256": self.path_sha256,
        }


class FormalV42AccessController:
    def __init__(
        self,
        *,
        run_root: str | Path,
        gate: str,
        contract_sha256: str,
        gate3_envelope: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_root = Path(run_root).resolve()
        self.gate = str(gate)
        self.contract_sha256 = str(contract_sha256)
        self.gate3_envelope = dict(gate3_envelope) if gate3_envelope is not None else None
        self._consumed_in_session = False

    @property
    def consumption_path(self) -> Path:
        return self.run_root / "protocol" / "GATE3_AUTHORIZATION_CONSUMED.json"

    def _record(self, event: AccessEventV42) -> None:
        root = self.run_root / "access" / "events"
        existing = sorted(root.glob("*.json")) if root.exists() else []
        index = len(existing) + 1
        write_once_json(root / f"{index:08d}.json", event.to_payload())

    def _deny(self, split: str, years: tuple[int, ...], caller: str, reason: str) -> None:
        self._record(AccessEventV42(split, years, caller, "deny", reason))
        raise EvaluationAccessDenied(reason)

    def _authorize_evaluation(self, caller: str) -> None:
        if self.gate != "gate3":
            raise EvaluationAccessDenied("2020 is locked until Gate 3")
        if self._consumed_in_session:
            return
        if self.consumption_path.exists():
            raise EvaluationAccessDenied("Gate 3 authorization was already consumed")
        if self.gate3_envelope is None:
            raise EvaluationAccessDenied("Gate 3 authorization is required")
        validate_gate3_envelope(self.gate3_envelope, self.contract_sha256)
        write_once_json(
            self.consumption_path,
            {
                "schema": "formal-v4.2-gate3-consumption-v1",
                "contract_sha256": self.contract_sha256,
                "envelope_sha256": canonical_sha256(self.gate3_envelope),
                "caller": caller,
                "allowed_years": [2020],
            },
        )
        self._consumed_in_session = True

    def request_years(
        self,
        split: str,
        years: Iterable[int],
        *,
        caller: str,
    ) -> tuple[int, ...]:
        requested = tuple(sorted(set(int(year) for year in years)))
        if 2021 in requested:
            self._deny(split, requested, caller, "2021 is excluded from formal-v4.2")
        if split not in SPLIT_YEARS:
            self._deny(split, requested, caller, "unknown formal-v4.2 split")
        if not requested or not set(requested).issubset(SPLIT_YEARS[split]):
            self._deny(split, requested, caller, "request violates the frozen split boundary")
        if split == "evaluation":
            try:
                self._authorize_evaluation(caller)
            except EvaluationAccessDenied as exc:
                self._deny(split, requested, caller, str(exc))
        self._record(AccessEventV42(split, requested, caller, "allow", "frozen split allowed"))
        return requested

    def request(
        self,
        path: str | Path,
        *,
        split: str,
        years: Iterable[int],
        caller: str,
    ) -> Path:
        requested = self.request_years(split, years, caller=caller)
        resolved = Path(path).resolve()
        if not resolved.is_file():
            self._deny(split, requested, caller, f"requested data file does not exist: {resolved}")
        self._record(
            AccessEventV42(
                split,
                requested,
                caller,
                "allow",
                "file bytes authorized",
                str(resolved),
                sha256_file(resolved),
            )
        )
        return resolved

    def build_receipt(self) -> dict[str, Any]:
        event_root = self.run_root / "access" / "events"
        events = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(event_root.glob("*.json"))
        ] if event_root.exists() else []
        return {
            "schema": "formal-v4.2-data-access-v1",
            "gate": self.gate,
            "contract_sha256": self.contract_sha256,
            "events": events,
            "evaluation_year_accessed": any(
                row["decision"] == "allow" and 2020 in row["years"] for row in events
            ),
            "excluded_year_accessed": any(
                row["decision"] == "allow" and 2021 in row["years"] for row in events
            ),
        }


__all__ = [
    "AccessEventV42",
    "EvaluationAccessDenied",
    "FormalV42AccessController",
    "GATE3_ENVELOPE_SCHEMA",
    "SPLIT_YEARS",
    "validate_gate3_envelope",
]
