# rust_timeseries.pyi
from typing import Any, Optional

__all__ = ["EscancianoLobato"]

class EscancianoLobato:
    """
    Result of the Escanciano–Lobato heteroskedasticity proxy test.
    """
    def __init__(
        self,
        raw_data: Any,
        q: float = 2.4,
        d: Optional[int] = ...,
    ) -> None: ...
    @property
    def statistic(self) -> float: ...
    @property
    def pvalue(self) -> float: ...
    @property
    def p_tilde(self) -> int: ...
