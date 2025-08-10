# rust_timeseries.pyi
from typing import Any, Protocol

class _EscancianoLobato(Protocol):
    """
    Escanciano–Lobato robust automatic portmanteau test (heteroskedasticity-robust).

    Parameters
    ----------
    data : ArrayLike[float64]
    q : float, default 2.4
    d : int | None, default None

    Attributes
    ----------
    statistic : float
    pvalue    : float
    p_tilde   : int
    """
    def __init__(self, data: Any, *, q: float = 2.4, d: int | None = ...) -> None: ...
    statistic: float
    pvalue: float
    p_tilde: int

class _statistical_tests(Protocol):
    """Statistical tests for serial dependence."""
    EscancianoLobato: type[_EscancianoLobato]

# Exposed submodule
statistical_tests: _statistical_tests
"""Submodule containing statistical tests."""

