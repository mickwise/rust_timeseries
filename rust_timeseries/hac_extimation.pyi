"""
HAC covariance estimation for robust standard errors.

Purpose
-------
Expose a thin Python wrapper around the Rust implementation of
heteroskedasticity- and autocorrelation-consistent (HAC) covariance
matrices of *average* per-observation scores.  The main entry point is
:func:`estimate_hac_covariance_matrix`, which delegates to the
``rust_timeseries._rust_timeseries.hac_estimation`` extension module.

This submodule is intended for users who already have per-observation
scores/residuals and want a HAC covariance matrix that can be plugged
 into sandwich-style standard error calculations.

Notes
-----
- Rows of the input correspond to time or observation index
  ``t = 1, …, n``.
- Columns correspond to score components / parameters
  ``i = 1, …, k``.
- The returned covariance is on the *average-score* scale, matching the
  average log-likelihood convention used elsewhere in ``rust_timeseries``.
- The exact HAC kernel and bandwidth selection follow the configuration
  implied by the ``kernel``, ``bandwidth``, ``center``, and
  ``small_sample_correction`` options of
  :func:`estimate_hac_covariance_matrix`.
"""
from __future__ import annotations

from typing import Final, Iterable, Sequence

import numpy as np
import numpy.typing as npt
from . import _rust_timeseries as _rt

__all__: Final[list[str]] = ["estimate_hac_covariance_matrix"]

_ArrayLikeF64_2D = (
    npt.NDArray[np.float64]
    | Sequence[Sequence[float]]
    | Iterable[Sequence[float]]
)


def estimate_hac_covariance_matrix(
    data: _ArrayLikeF64_2D,
    *,
    kernel: str = "bartlett",
    bandwidth: int | None = None,
    center: bool = False,
    small_sample_correction: bool = True,
) -> npt.NDArray[np.float64]:
    """
    HAC covariance matrix of average scores.

    This function wraps the Rust implementation
    ``rust_timeseries._rust_timeseries.hac_estimation.estimate_hac_covariance_matrix``
    and returns a NumPy array on the *average-score* scale. It is
    intended for robust (sandwich) variance estimation when you already
    have per-observation score vectors or similar quantities.

    Parameters
    ----------
    data : array-like of float64, shape *(n, k)*
        - Per-observation scores or related series.
        - Each row ``i`` contains the length-``k`` score vector at
          observation / time index ``i``.
        - Input must be finite and non-empty; NaNs are not allowed.
        - Accepted types include:
            - ``numpy.ndarray`` with ``dtype=float64`` and 2D shape,
            - nested Python sequences (e.g. ``list[list[float]]``).

    kernel : {"iid", "bartlett", "parzen", "quadratic_spectral"}, optional
        HAC kernel used to weight higher lags:
        - ``"iid"``  
          - Treats observations as serially independent.
          - The effective bandwidth is ``L = 0``, so the estimator reduces to the
            outer-product-of-gradients (OPG) covariance ``(1/n) Sᵀ S`` with
            no serial-correlation adjustment.
        - ``"bartlett"``  
          - Triangular Newey–West kernel with compact support. Weights
            decay linearly to zero at lag ``L``.  When ``bandwidth`` is
            ``None``, an Andrews plug-in bandwidth is selected with order ``q = 1``.
        - ``"parzen"``  
          - Smoother compact-support kernel with cubic tapering.  When
            ``bandwidth`` is ``None``, an Andrews plug-in bandwidth with
            order ``q = 2`` is used.
        - ``"quadratic_spectral"``  
          - Infinite-support quadratic spectral kernel, truncated at the
            chosen bandwidth ``L``.  When ``bandwidth`` is ``None``, an
          - Andrews plug-in bandwidth with order ``q = 2`` is used.
        - Default is ``"bartlett"``.

    bandwidth : int or None, optional
        - Maximum lag ``L`` used by the HAC estimator.
        - If an integer is provided, the effective bandwidth is
          ``L_eff = min(bandwidth, n - 1)``.
        - Larger values than `n - 1`` are automatically truncated.
        - If ``None`` (default), a plug-in bandwidth is selected
          internally based on the chosen kernel and the data in
          ``data``.
        - When the plug-in calculation is numerically unstable, 
          a conservative rule-of-thumb of order ``n^{1/4}``
          is used instead.
        - For ``kernel="iid"``, the effective bandwidth is always
          ``L_eff = 0`` regardless of this argument.

    center : bool, optional
        - If ``True``, columns of ``data`` are demeaned before both
          bandwidth selection and HAC aggregation.
        - This is often appropriate when the covariance is meant to be taken around
          the sample mean or MLE, and can improve numerical stability
          for strongly trending scores.
        - If ``False`` (default), the raw series in ``data`` are used as-is. 
        - When the columns of ``data`` are already mean-zero,
          enabling or disabling centering has no effect (up to floating
          point noise).

    small_sample_correction : bool, optional
        - Controls the finite-sample scaling used in the HAC estimator.
        - If ``True`` (default), uses Newey–West scaling
          ``c_k = 1 / (n - k)`` for lags ``k > 0``.
        - This typically inflates the estimated variance relative to the asymptotic
          scaling and is more conservative in small samples.
        - If ``False``, uses the asymptotic scaling ``c_k = 1 / n``
          at all lags, which can underestimate variance when ``n`` is
          small and serial correlation is non-negligible.

    Returns
    -------
    cov : ndarray of float64, shape *(k, k)*
        - Estimated HAC covariance matrix of the *average* score vector.
        - The matrix is symmetric and positive semi-definite up to
          numerical tolerance.
        - The ``[i, j]`` entry corresponds to the covariance between
          the ``i``-th and ``j``-th score components.

    Notes
    -----
    - The estimator has the generic form:

      ``S = Γ₀ + Σ_{k=1}^L w_k (Γ_k + Γ_kᵀ)``,

      where ``Γ₀ = (1/n) S_fullᵀ S_full`` and ``Γ_k`` aggregates
      lag-``k`` cross-products with either ``1/n`` or Newey–West
      ``1/(n - k)`` scaling, depending on ``small_sample_correction``.
    - The returned covariance is on the *average-score* scale, i.e.
      if ``S_full`` stores per-observation scores, ``S`` corresponds to
      the covariance of ``(1/n) Σ_t s_t``.  This aligns with the
      sandwich variance formulas used in the rest of
      ``rust_timeseries``.
    """
    cov = _rt.hac_estimation.estimate_hac_covariance_matrix(
        data,
        kernel=kernel,
        bandwidth=bandwidth,
        center=center,
        small_sample_correction=small_sample_correction,
    )
    return np.asarray(cov, dtype=np.float64)
