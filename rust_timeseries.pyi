"""
Type stubs for the rust_timeseries extension package.
Generated manually until we have an automatic exporter.
"""

from typing import Any, Protocol

# ─────────────────── sub-module: statistical_tests ────────────────────

class _EscancianoLobato(Protocol):
    statistic: float
    pvalue: float
    p_tilde: int
    def __init__(
        self,
        data: Any,
        *,
        q: float = 2.4,
        d: int | None = ...,
    ) -> None: ...

class _statistical_tests(Protocol):
    EscancianoLobato: type[_EscancianoLobato]

statistical_tests: _statistical_tests
