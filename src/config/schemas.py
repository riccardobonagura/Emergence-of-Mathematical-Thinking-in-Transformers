"""
schemas.py — canonical inter-module config contracts (SSOT, S-03).

Single home for PropConfig so probing_dataset and validate_configs share one
definition instead of two divergent copies. Imports only stdlib/typing, so it is
import-cycle-safe for every src.config / src.probing / src.utils consumer.
"""

from typing import Literal, TypedDict


class _PropConfigOptional(TypedDict, total=False):
    # Absent OR null both mean "no category filter" at runtime (probing_dataset).
    category: str | None


class PropConfig(_PropConfigOptional):
    # The TYPE is permissive: fixtures and defensive runtime paths may omit category.
    # PRODUCTION validation still enforces its presence in
    # validate_configs._validate_probing_schema (anti-contamination guardrail).
    label_field: str
    type: Literal["binary", "multiclass"]
