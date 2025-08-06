# rust_timeseries/statistical_tests.pyi
from typing import Any, Optional

__all__ = ["EscancianoLobato"]

class EscancianoLobato:
    """
    Result of the Escanciano–Lobato heteroskedasticity proxy test.

    Constructor parameters:
      - raw_data: array-like of float64 (numpy.ndarray, pandas.Series, or list)
      - q: float > 0 (default 2.4)
      - d: Optional[int] > 0 (default floor(n**0.2))
    """

    def __init__(self, raw_data: Any, q: float = ..., d: Optional[int] = ...) -> None: ...
    @property
    def statistic(self) -> float: ...
    @property
    def pvalue(self) -> float: ...
    @property
    def p_tilde(self) -> int: ...
