from __future__ import annotations

from typing import Final, Iterable, Sequence, overload
import numpy as np
import numpy.typing as npt

__all__: Final[list[str]] = ["EscancianoLobato"]

_ArrayLikeF64 = npt.NDArray[np.float64] | Sequence[float] | Iterable[float]

class EscancianoLobato:
    """
    Escanciano–Lobato heteroskedasticity proxy test.

    Parameters
    ----------
    raw_data : array-like of float64, shape *(n,)*
        Must not contain any NaN values.
    q : int, optional
        Order of the proxy. `q > 0`. Default = 2.4.
    d : int, optional
        Maximum lag. Default = ⌊n^0.2⌋.

    Raises
    ------
    PyValueError
        If `data` is empty, contains NaN values, or if `q` or `d` are not positive integers.

    Returns
    -------
    EscancianoLobato
        Object with ``statistic``, ``pvalue`` and ``p_tilde`` attributes.

    Notes
    -----
    The statistic is asymptotically χ²(1) under the null.
    """

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
