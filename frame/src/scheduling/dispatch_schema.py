"""Pure dispatch schema shared by LP labels and neural-proxy code.

This module intentionally has no numerical-solver imports.  It is safe to
load from the V2 inference graph even when SciPy/``dispatch_lp`` is absent.
"""

from __future__ import annotations

from typing import Tuple


VARIABLES: Tuple[str, ...] = (
    "grid",
    "pv_use",
    "pv_curt",
    "wt_use",
    "wt_curt",
    "g_chp",
    "g_gb",
    "p_chp",
    "q_chp",
    "q_gb",
    "p_ec",
    "q_ec",
    "q_ac_in",
    "q_ac",
    "p_charge",
    "p_discharge",
    "soc",
    "slack_e",
    "slack_c",
    "slack_h",
    "q_dump",
)


__all__ = ["VARIABLES"]
