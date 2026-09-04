from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from src.kitakyushu_pipeline import (
    GAS_ZIP_NAME,
    LOAD_ZIP_NAME,
    WEATHER_ZIP_NAME,
    read_kitakyushu_canonical,
)


def _xlsx_bytes(kind: str, year: int) -> bytes:
    timestamp = pd.Timestamp(f"{year}-01-01 00:00:00")
    if kind == "load":
        frame = pd.DataFrame({
            "Date": [timestamp],
            "Electricity load (kW)": [10.0],
            "Cooling load (kW)": [20.0],
            "Heating load (kW)": [30.0],
        })
    elif kind == "gas":
        frame = pd.DataFrame({
            "Date": [timestamp],
            "Boiler (m3)": [1.0],
            "Fuel cell (m3)": [1.0],
            "Gas engine (m3)": [1.0],
            "Absorption chiller 1 (m3)": [1.0],
            "Absorption chiller 2 (m3)": [1.0],
            "Absorption chiller 3 (m3)": [1.0],
        })
    else:
        frame = pd.DataFrame({
            "Date": [timestamp],
            "Outdoor air temperature (°C)": [10.0],
            "Outdoor air humidity (%)": [50.0],
            "Horizontal solar irradiation (W)": [100.0],
            "Wind speed (m/s)": [2.0],
            "Wind direction": [180.0],
        })
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False)
    return output.getvalue()


def _write_archives(root: Path, years: tuple[int, ...]) -> None:
    names = {"load": LOAD_ZIP_NAME, "gas": GAS_ZIP_NAME, "weather": WEATHER_ZIP_NAME}
    for kind, name in names.items():
        with zipfile.ZipFile(root / name, "w") as archive:
            for year in years:
                archive.writestr(f"source__{year}.xlsx", _xlsx_bytes(kind, year))


def test_canonical_reader_emits_container_and_member_hashes(tmp_path: Path) -> None:
    _write_archives(tmp_path, (2015, 2019))
    events: list[dict] = []
    frame, metadata = read_kitakyushu_canonical(tmp_path, years=(2015, 2019), audit_sink=events.append)
    assert len(frame) == 2
    assert len(events) == 6
    assert {item["source_kind"] for item in events} == {"load", "gas", "weather"}
    assert {item["year"] for item in events} == {2015, 2019}
    for event in events:
        assert len(event["container_sha256"]) == 64
        assert len(event["member_sha256"]) == 64
        assert event["member_name"].endswith(".xlsx")
    assert set(metadata["years"]) == {2015, 2019}


def test_access_guard_runs_before_forbidden_member_bytes_are_read(tmp_path: Path) -> None:
    _write_archives(tmp_path, (2020,))
    audit_events: list[dict] = []
    guard_calls: list[tuple[str, int, str]] = []

    def deny(source_kind: str, year: int, archive: Path, member: str) -> None:
        guard_calls.append((source_kind, year, member))
        raise PermissionError("evaluation member denied")

    with pytest.raises(PermissionError, match="evaluation member denied"):
        read_kitakyushu_canonical(tmp_path, years=(2020,), audit_sink=audit_events.append, access_guard=deny)
    assert guard_calls == [("load", 2020, "source__2020.xlsx")]
    assert audit_events == []
