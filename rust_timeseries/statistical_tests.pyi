from __future__ import annotations

from typing import Final, Iterable, Sequence, overload
import numpy as np
import numpy.typing as npt

__all__: Final[list[str]] = ["EscancianoLobato"]

_ArrayLikeF64 = npt.NDArray[np.float64] | Sequence[float] | Iterable[float]

class EscancianoLobato:
    @overload
    def __init__(self, data: _ArrayLikeF64, /) -> None: ...
    @overload
    def __init__(self, data: _ArrayLikeF64, /, *, q: float | int = ..., d: int | None = ...) -> None: ...

    @property
    def statistic(self) -> float: ...

    @property
    def pvalue(self) -> float: ...

    @property
    def p_tilde(self) -> int: ...
