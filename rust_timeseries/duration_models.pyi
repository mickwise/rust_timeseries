# pylint: disable=C0302:too-many-lines
"""
Purpose
-------
Provide a complete ACD(p, q) model API for univariate duration data,
including maximum-likelihood fitting, forecasting of conditional mean
durations, and computation of classical or HAC-robust covariance
matrices for model-space parameters. This module is the Python entry
point for ACD models in the :mod:`rust_timeseries` package.

Key behaviors
-------------
- Construct ACD(p, q) models with a chosen innovation family
  (exponential, Weibull, generalized gamma) for a given in-sample
  length ``n``.
- Fit models in an unconstrained parameter space and expose fitted
  parameters, optimization diagnostics, and forecast paths as
  Python-native objects.
- Compute classical (observed-information) or HAC-robust (sandwich)
  covariance matrices for the model-space parameters (ω, α, β),
  derived from the unconstrained optimizer vector θ via a delta method.

Conventions
-----------
- Durations are one-dimensional, strictly positive, finite arrays of
  ``float64`` with shape ``(n,)``.
- Parameters are represented in an unconstrained vector
  ``θ = (θ₀, θ₁, …, θ_{p+q})``:
  - ``ω = softplus(θ₀)`` is the baseline scale parameter.
  - ``(α, β, slack)`` come from a scaled softmax of ``θ[1:]`` so that
    all weights are non-negative and
    ``∑α + ∑β + slack < 1 − margin`` for strict stationarity.
- Time indexing is conceptually 0-based and follows the
  Engle–Russell ACD(p, q) recursion

  ψ_t = ω + Σ_{i=1..q} α_i τ_{t−i} + Σ_{j=1..p} β_j ψ_{t−j},

  with conditional mean durations ``E[τ_t | F_{t−1}] = ψ_t``.

Downstream usage
----------------
Import from :mod:`rust_timeseries.duration_models`:

    >>> from rust_timeseries.duration_models import ACD
    >>> model = ACD(data_length=len(durations), p=1, q=1)
    >>> model.fit(durations, theta0)
    >>> psi_T_plus_H = model.forecast(durations, horizon=5)
    >>> cov = model.covariance_matrix(durations, robust=True)
    >>> cov = np.asarray(cov, dtype=np.float64)
    >>> se = np.sqrt(np.diag(cov))

Typical workflows build an :class:`ACD` instance once for an
in-sample dataset, call :meth:`ACD.fit`, and then reuse the same
instance for forecasting and inference.
"""

from __future__ import annotations

from typing import Final, Iterable, Sequence

import numpy as np
import numpy.typing as npt

from . import _rust_timeseries as _rt
__all__: Final[list[str]] = ["ACD", "ACDOptimOutcome", "ACDFittedParams"]

_ArrayLikeF64 = (
    npt.NDArray[np.float64] | Sequence[float] | Iterable[float]
)

# Aliases to the Rust-backed PyO3 classes
# pylint: disable=c-extension-no-member
_RustACD = _rt.duration_models.ACD
_RustOutcome = _rt.duration_models.ACDOptimOutcome
_RustFittedParams = _rt.duration_models.ACDFittedParams


class ACD:
    """
    Purpose
    -------
    High-level Python wrapper for univariate ACD(p, q) duration models.

    Key behaviors
    -------------
    - Construct ACD models with exponential, Weibull, or generalized-gamma
      innovations using a shared Rust core.
    - Expose `fit`, `forecast`, and `covariance_matrix` as Python methods
      while delegating numerical work to the compiled implementation.
    - Provide access to optimization diagnostics and fitted parameters
      via the :attr:`results` and :attr:`fitted_params` properties.

    Parameters
    ----------
    data_length : int
        In-sample length used to size internal buffers in the Rust model.
    p : int, optional
        ACD order for ψ-lags (β).
    q : int, optional
        ACD order for duration lags (α).
    init : str, optional
        Initialization scheme for ψ, e.g. ``"uncond_mean"``,
        ``"sample_mean"``, ``"fixed"``, or ``"fixed_vector"``.
    init_fixed : float, optional
        Scalar ψ initialization used when ``init="fixed"``.
    init_psi_lags : array-like of float, optional
        ψ-lag vector used when ``init="fixed_vector"``.
    init_durations_lags : array-like of float, optional
        Duration-lag vector used when ``init="fixed_vector"``.
    tol_grad : float, optional
        Gradient-norm tolerance for the optimizer.
    tol_cost : float, optional
        Cost-function tolerance for the optimizer.
    max_iter : int, optional
        Maximum number of optimization iterations.
    line_searcher : str, optional
        Line-search strategy name, e.g. ``"more_thuente"``.
    lbfgs_mem : int, optional
        Memory parameter for L-BFGS-based solvers.
    psi_guards : tuple[float, float], optional
        Minimum and maximum ψ guard rails for numerical stability.

    Attributes
    ----------
    _inner : _rt.duration_models.ACD
        Underlying Rust-backed ACD model instance.

    Notes
    -----
    - The default constructor uses exponential innovations. For Weibull or
      generalized-gamma innovations, use :meth:`ACD.wacd` or
      :meth:`ACD.gacd` classmethods.
    """

    def __init__(
        self,
        data_length: int,
        *,
        p: int | None = None,
        q: int | None = None,
        init: str | None = None,
        init_fixed: float | None = None,
        init_psi_lags: _ArrayLikeF64 | None = None,
        init_durations_lags: _ArrayLikeF64 | None = None,
        tol_grad: float | None = None,
        tol_cost: float | None = None,
        max_iter: int | None = None,
        line_searcher: str | None = None,
        lbfgs_mem: int | None = None,
        psi_guards: tuple[float, float] | None = None,
    ) -> None:
        """
        Purpose
        -------
        Construct an exponential-innovation ACD(p, q) model.

        Key behaviors
        -------------
        - Forwards all configuration options to the underlying Rust model.
        - Allocates internal buffers sized for ``data_length``.

        Parameters
        ----------
        data_length : int
            Number of in-sample durations.
        p : int, optional
            ACD order for ψ-lags; see :class:`ACD`.
        q : int, optional
            ACD order for duration lags; see :class:`ACD`.
        init : str, optional
            Initialization policy; see :class:`ACD`.
        init_fixed : float, optional
            Scalar ψ initialization for ``init="fixed"``.
        init_psi_lags : array-like of float, optional
            ψ-lag initialization for ``init="fixed_vector"``.
        init_durations_lags : array-like of float, optional
            Duration-lag initialization for ``init="fixed_vector"``.
        tol_grad : float, optional
            Gradient tolerance for the optimizer.
        tol_cost : float, optional
            Cost tolerance for the optimizer.
        max_iter : int, optional
            Maximum optimizer iterations.
        line_searcher : str, optional
            Line-search algorithm name.
        lbfgs_mem : int, optional
            L-BFGS memory parameter.
        psi_guards : tuple[float, float], optional
            Minimum and maximum allowable ψ values.

        Raises
        ------
        ValueError
            If any configuration option is invalid (propagated from Rust).

        Notes
        -----
        - This is equivalent to calling the Rust-backed ``ACD.eacd`` constructor.
        """
        self._inner = _RustACD(
            data_length,
            p,
            q,
            init,
            init_fixed,
            init_psi_lags,
            init_durations_lags,
            tol_grad,
            tol_cost,
            max_iter,
            line_searcher,
            lbfgs_mem,
            psi_guards,
        )

    @classmethod
    def wacd(
        cls,
        data_length: int,
        k: float,
        *,
        p: int | None = None,
        q: int | None = None,
        init: str | None = None,
        init_fixed: float | None = None,
        init_psi_lags: _ArrayLikeF64 | None = None,
        init_durations_lags: _ArrayLikeF64 | None = None,
        tol_grad: float | None = None,
        tol_cost: float | None = None,
        max_iter: int | None = None,
        line_searcher: str | None = None,
        lbfgs_mem: int | None = None,
        psi_guards: tuple[float, float] | None = None,
    ) -> ACD:
        """
        Purpose
        -------
        Construct a Weibull-innovation ACD(p, q) model.

        Key behaviors
        -------------
        - Uses a unit-mean Weibull innovation with shape parameter `k`.
        - Delegates all validation to the Rust model constructor.

        Parameters
        ----------
        data_length : int
            Number of in-sample durations.
        k : float
            Weibull shape parameter (must be > 0).
        p, q, init, init_fixed, init_psi_lags, init_durations_lags,
        tol_grad, tol_cost, max_iter, line_searcher, lbfgs_mem, psi_guards
            As in :class:`ACD`.

        Returns
        -------
        ACD
            A Python wrapper around the underlying Weibull-ACD model.

        Raises
        ------
        ValueError
            If the Weibull shape or any other configuration is invalid.

        Notes
        -----
        - This is a convenience wrapper around the Rust ``ACD.wacd`` constructor.
        """
        inner = _RustACD.wacd(
            data_length,
            p,
            q,
            k,
            init,
            init_fixed,
            init_psi_lags,
            init_durations_lags,
            tol_grad,
            tol_cost,
            max_iter,
            line_searcher,
            lbfgs_mem,
            psi_guards,
        )
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    @classmethod
    def gacd(
        cls,
        data_length: int,
        p_shape: float,
        d_shape: float,
        *,
        p: int | None = None,
        q: int | None = None,
        init: str | None = None,
        init_fixed: float | None = None,
        init_psi_lags: _ArrayLikeF64 | None = None,
        init_durations_lags: _ArrayLikeF64 | None = None,
        tol_grad: float | None = None,
        tol_cost: float | None = None,
        max_iter: int | None = None,
        line_searcher: str | None = None,
        lbfgs_mem: int | None = None,
        psi_guards: tuple[float, float] | None = None,
    ) -> ACD:
        """
        Purpose
        -------
        Construct a generalized-gamma-innovation ACD(p, q) model.

        Key behaviors
        -------------
        - Uses a unit-mean generalized gamma innovation with shape parameters
          ``p_shape`` and ``d_shape``.
        - Delegates parameter validation and stationarity checks to Rust.

        Parameters
        ----------
        data_length : int
            Number of in-sample durations.
        p_shape : float
            First generalized-gamma shape parameter.
        d_shape : float
            Second generalized-gamma shape parameter.
        p, q, init, init_fixed, init_psi_lags, init_durations_lags,
        tol_grad, tol_cost, max_iter, line_searcher, lbfgs_mem, psi_guards
            As in :class:`ACD`.

        Returns
        -------
        ACD
            A Python wrapper around the underlying generalized-gamma model.

        Raises
        ------
        ValueError
            If the generalized-gamma parameters or other options are invalid.

        Notes
        -----
        - This is a convenience wrapper around the Rust ``ACD.gacd`` constructor.
        """
        inner = _RustACD.gacd(
            data_length,
            p,
            q,
            p_shape,
            d_shape,
            init,
            init_fixed,
            init_psi_lags,
            init_durations_lags,
            tol_grad,
            tol_cost,
            max_iter,
            line_searcher,
            lbfgs_mem,
            psi_guards,
        )
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    def fit(
        self,
        durations: _ArrayLikeF64,
        theta0: _ArrayLikeF64,
        *,
        unit: str = "seconds",
        t0: int | None = None,
        diurnal_adjusted: bool = False,
    ) -> None:
        """
        Purpose
        -------
        Fit the ACD(p, q) model by quasi-MLE.

        Key behaviors
        -------------
        - Validates the duration series and initial parameter vector.
        - Forwards the data to the Rust core, which runs the optimizer and
          caches the optimization outcome and fitted parameters.

        Parameters
        ----------
        durations : array-like of float
            Observed durations, shape ``(n,)``.
        theta0 : array-like of float
            Initial unconstrained parameter vector of length ``1 + p + q``.
        unit : str, optional
            Time unit for the durations: ``"seconds"``, ``"milliseconds"``,
            or ``"microseconds"``. Default is ``"seconds"``.
        t0 : int, optional
            Burn-in index controlling which observations enter the likelihood.
        diurnal_adjusted : bool, optional
            If ``True``, indicates that durations have already been
            diurnally adjusted. Default is ``False``.

        Returns
        -------
        None
            On success, updates :attr:`results` and :attr:`fitted_params`.

        Raises
        ------
        ValueError
            If the data, initial parameters, or optimizer options are invalid.

        Notes
        -----
        - This method is thin glue around the Rust-backed ``ACD.fit``.
        """
        durations_arr = np.asarray(durations, dtype=np.float64)
        theta0_arr = np.asarray(theta0, dtype=np.float64)
        self._inner.fit(
            durations_arr,
            theta0_arr,
            unit,
            t0,
            diurnal_adjusted,
        )

    def forecast(
        self,
        durations: _ArrayLikeF64,
        horizon: int,
        *,
        unit: str = "seconds",
        t0: int | None = None,
        diurnal_adjusted: bool = False,
    ) -> float:
        """
        Purpose
        -------
        Compute an H-step-ahead forecast of the conditional mean duration.

        Key behaviors
        -------------
        - Uses the fitted ACD parameters and the tail of the duration series
          to roll the ψ-recursion forward.
        - Returns the last point in the forecast horizon, ψ̂_{T+H}.

        Parameters
        ----------
        durations : array-like of float
            Duration series used to provide the final ``q`` lags.
        horizon : int
            Forecast horizon ``H >= 1``.
        unit : str, optional
            Time unit for ``durations``; see :meth:`fit`.
        t0 : int, optional
            Burn-in index for the likelihood. Passed through for consistency.
        diurnal_adjusted : bool, optional
            Indicates whether durations are diurnally adjusted.

        Returns
        -------
        float
            The H-step-ahead conditional mean duration ψ̂_{T+H}.

        Raises
        ------
        ValueError
            If the model has not been fitted or forecasting fails.

        Notes
        -----
        - The full forecast path is accessible via :attr:`forecast_result`.
        """
        durations_arr = np.asarray(durations, dtype=np.float64)
        return float(
            self._inner.forecast(
                durations_arr,
                horizon,
                unit,
                t0,
                diurnal_adjusted,
            )
        )

    def covariance_matrix(
        self,
        durations: _ArrayLikeF64,
        *,
        unit: str = "seconds",
        t0: int | None = None,
        diurnal_adjusted: bool = False,
        robust: bool = False,
        kernel: str = "bartlett",
        bandwidth: int | None = None,
        center: bool = False,
        small_sample_correction: bool = True,
    ) -> list[list[float]]:
        """
        Purpose
        -------
        Compute a classical or HAC-robust covariance matrix for
        (ω, α, β) at the fitted ACD(p, q) model.

        Key behaviors
        -------------
        - Forwards the duration series and HAC options to the Rust core.
        - Returns a row-major (1 + q + p) × (1 + q + p) covariance matrix
          for (ω, α, β) in the order implied by the model:
          ``[ω, α₁, …, α_q, β₁, …, β_p]``.

        Parameters
        ----------
        durations : array-like of float
            Duration series used to compute gradients and per-observation
            scores. Typically the same in-sample data used for fitting.
        unit : str, optional
            Time unit for ``durations``; see :meth:`fit`. Default is
            ``"seconds"``.
        t0 : int, optional
            Burn-in index; passed through for consistency with :meth:`fit`.
        diurnal_adjusted : bool, optional
            Indicates whether ``durations`` have been diurnally adjusted.
        robust : bool, optional
            If ``False``, use the classical observed-information estimator.
            If ``True``, use a HAC-robust “sandwich” estimator based on
            per-observation score covariances.
        kernel : str, optional
            HAC kernel name (ignored if ``robust=False``). One of
            ``"iid"``, ``"bartlett"``, ``"parzen"``, or
            ``"quadratic_spectral"``. Default is ``"bartlett"``.
        bandwidth : int, optional
            HAC kernel bandwidth (ignored if ``robust=False``).
        center : bool, optional
            If ``True``, center per-observation scores in the HAC estimator.
        small_sample_correction : bool, optional
            If ``True``, apply Newey–West small-sample corrections when
            building the HAC score covariance.

        Returns
        -------
        list[list[float]]
            A row-major (1 + q + p) × (1 + q + p) covariance matrix for the
            model-space parameters (ω, α, β). The diagonal entries are the
            parameter variances; standard errors are obtained as
            ``np.sqrt(np.diag(cov))``.

        Raises
        ------
        ValueError
            If the model has not been fitted or if the HAC configuration
            is invalid. Errors are propagated from the Rust implementation.

        Notes
        -----
        - The returned covariance is **parameter-space**, not θ-space.
        - Internally, the Rust core builds a covariance in θ-space and
          applies a multivariate delta method to map it to (ω, α, β).
        """
        durations_arr = np.asarray(durations, dtype=np.float64)
        return self._inner.covariance_matrix(
            durations_arr,
            unit,
            t0,
            diurnal_adjusted,
            robust,
            kernel,
            bandwidth,
            center,
            small_sample_correction,
        )

    @property
    def results(self) -> "ACDOptimOutcome":
        """
        Purpose
        -------
        Expose the optimization outcome from the last call to :meth:`fit`.

        Key behaviors
        -------------
        - Returns a structured wrapper around the final θ̂, log-likelihood,
          convergence status, and evaluation counters.

        Returns
        -------
        ACDOptimOutcome
            Optimization diagnostics for the last fit.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        Notes
        -----
        - Errors are propagated from the underlying Rust getter.
        """
        return ACDOptimOutcome(self._inner.results)

    @property
    def fitted_params(self) -> "ACDFittedParams":
        """
        Purpose
        -------
        Expose model-space parameters from the last successful fit.

        Key behaviors
        -------------
        - Provides read-only access to (ω, α, β, slack, ψ-lags) implied by
          the final θ̂.

        Returns
        -------
        ACDFittedParams
            Fitted model-space parameters.

        Raises
        ------
        ValueError
            If the model has not been fitted.

        Notes
        -----
        - Errors are propagated from the underlying Rust getter.
        """
        return ACDFittedParams(self._inner.fitted_params)

    @property
    def forecast_result(self) -> list[float]:
        """
        Purpose
        -------
        Return the full ψ-forecast path from the last call to :meth:`forecast`.

        Key behaviors
        -------------
        - Delegates to the underlying Rust forecast buffer.
        - Returns an empty list if no forecast has been run yet.

        Returns
        -------
        list[float]
            ψ̂ forecasts for the most recent horizon, or an empty list if
            :meth:`forecast` has not been called.

        Raises
        ------
        ValueError
            Never raised by this accessor.

        Notes
        -----
        - The final element in this list coincides with the scalar returned
          by :meth:`forecast`.
        """
        return list(self._inner.forecast_result)


class ACDOptimOutcome:
    """
    Purpose
    -------
    Structured view of optimization diagnostics for an ACD fit.

    Key behaviors
    -------------
    - Exposes θ̂, log-likelihood, convergence status, and gradient norm.
    - Provides evaluation counters for cost, gradient, and related calls.

    Parameters
    ----------
    inner : _rt.duration_models.ACDOptimOutcome
        Rust-backed optimization outcome instance.

    Attributes
    ----------
    _inner : _rt.duration_models.ACDOptimOutcome
        Wrapped low-level result object.

    Notes
    -----
    - Users typically obtain instances via :attr:`ACD.results` rather than
      constructing this class directly.
    """

    def __init__(self, inner: _RustOutcome) -> None:
        self._inner = inner

    @property
    def theta_hat(self) -> list[float]:
        """
        Purpose
        -------
        Return the final unconstrained parameter vector θ̂.

        Key behaviors
        -------------
        - Delegates to the Rust-backed getter and converts to a Python list.

        Returns
        -------
        list[float]
            Unconstrained parameter vector of length ``1 + p + q``.

        Raises
        ------
        ValueError
            Never raised directly by this accessor.

        Notes
        -----
        - Errors would have been raised earlier during model fitting.
        """
        return list(self._inner.theta_hat)

    @property
    def value(self) -> float:
        """
        Purpose
        -------
        Return the log-likelihood evaluated at θ̂.

        Key behaviors
        -------------
        - Exposes the best log-likelihood value from the optimizer.

        Returns
        -------
        float
            Log-likelihood value ℓ(θ̂).

        Raises
        ------
        ValueError
            Never raised directly.

        Notes
        -----
        - This is not the cost (negative log-likelihood); it is the
          maximized log-likelihood itself.
        """
        return float(self._inner.value)

    @property
    def converged(self) -> bool:
        """
        Purpose
        -------
        Indicate whether the optimizer reported termination.

        Key behaviors
        -------------
        - Returns ``True`` if the termination status is not ``NotTerminated``.

        Returns
        -------
        bool
            Convergence flag reported by the optimizer.

        Raises
        ------
        ValueError
            Never raised directly.

        Notes
        -----
        - For finer-grained status information, consult :attr:`status`.
        """
        return bool(self._inner.converged)

    @property
    def status(self) -> str:
        """
        Purpose
        -------
        Provide a human-readable termination status string.

        Key behaviors
        -------------
        - Mirrors the status string from the underlying optimizer.

        Returns
        -------
        str
            Human-readable description of the termination reason.

        Raises
        ------
        ValueError
            Never raised directly.

        Notes
        -----
        - The exact wording is determined by the Rust implementation.
        """
        return str(self._inner.status)

    @property
    def iterations(self) -> int:
        """
        Purpose
        -------
        Report the number of iterations executed by the optimizer.

        Key behaviors
        -------------
        - Exposes the final iteration count as a Python integer.

        Returns
        -------
        int
            Number of iterations performed.

        Raises
        ------
        ValueError
            Never raised directly.

        Notes
        -----
        - This count may be zero if the optimizer terminated immediately.
        """
        return int(self._inner.iterations)

    @property
    def grad_norm(self) -> float | None:
        """
        Purpose
        -------
        Report the L2 norm of the gradient at θ̂, if available.

        Key behaviors
        -------------
        - Returns ``None`` when the optimizer did not provide a final gradient.

        Returns
        -------
        float or None
            Gradient norm at the solution, or ``None`` if unavailable.

        Raises
        ------
        ValueError
            Never raised directly.

        Notes
        -----
        - Useful as a sanity check for first-order optimality.
        """
        return self._inner.grad_norm

    @property
    def fn_evals(self) -> list[tuple[str, int]]:
        """
        Purpose
        -------
        Expose evaluation counters for the optimization run.

        Key behaviors
        -------------
        - Returns a list of (name, count) pairs for cost, gradient,
          and related evaluations.

        Returns
        -------
        list[tuple[str, int]]
            Evaluation counters reported by the optimizer.

        Raises
        ------
        ValueError
            Never raised directly.

        Notes
        -----
        - The keys and their meaning are defined by the Rust optimizer layer.
        """
        return [
            (str(name), int(count)) for name, count in self._inner.fn_evals
        ]


class ACDFittedParams:
    """
    Purpose
    -------
    Read-only view of model-space ACD parameters after fitting.

    Key behaviors
    -------------
    - Exposes ω, α, β, slack, and ψ-lags implied by the final θ̂.
    - Provides a stable Python interface over the Rust ACDParams struct.

    Parameters
    ----------
    inner : _rt.duration_models.ACDFittedParams
        Rust-backed fitted-parameter container.

    Attributes
    ----------
    _inner : _rt.duration_models.ACDFittedParams
        Wrapped low-level parameter object.

    Notes
    -----
    - Instances are typically obtained via :attr:`ACD.fitted_params`.
    """

    def __init__(self, inner: _RustFittedParams) -> None:
        self._inner = inner

    @property
    def omega(self) -> float:
        """
        Purpose
        -------
        Return the fitted baseline scale parameter ω.

        Key behaviors
        -------------
        - Delegates to the Rust getter and converts to Python float.

        Returns
        -------
        float
            Baseline intensity parameter ω > 0.

        Raises
        ------
        ValueError
            Never raised directly.

        Notes
        -----
        - ω is measured in the same time units as the input durations.
        """
        return float(self._inner.omega)

    @property
    def slack(self) -> float:
        """
        Purpose
        -------
        Return the fitted slack mass completing the simplex.

        Key behaviors
        -------------
        - Reports the amount of mass not assigned to α or β.

        Returns
        -------
        float
            Slack ≥ 0 such that ∑α + ∑β + slack = 1 − margin.

        Raises
        ------
        ValueError
            Never raised directly.

        Notes
        -----
        - A small positive slack ensures strict stationarity.
        """
        return float(self._inner.slack)

    @property
    def alpha(self) -> list[float]:
        """
        Purpose
        -------
        Return the fitted α coefficients for duration lags.

        Key behaviors
        -------------
        - Converts the underlying ndarray into a Python list.

        Returns
        -------
        list[float]
            α coefficients of length q.

        Raises
        ------
        ValueError
            Never raised directly.

        Notes
        -----
        - All entries are non-negative and satisfy the stationarity constraint
          together with β and slack.
        """
        return list(self._inner.alpha)

    @property
    def beta(self) -> list[float]:
        """
        Purpose
        -------
        Return the fitted β coefficients for ψ-lags.

        Key behaviors
        -------------
        - Converts the underlying ndarray into a Python list.

        Returns
        -------
        list[float]
            β coefficients of length p.

        Raises
        ------
        ValueError
            Never raised directly.

        Notes
        -----
        - All entries are non-negative and contribute to the stationarity
          constraint.
        """
        return list(self._inner.beta)

    @property
    def psi_lags(self) -> list[float]:
        """
        Purpose
        -------
        Return the last p in-sample ψ values used as pre-sample lags.

        Key behaviors
        -------------
        - Provides the ψ-lag vector used for out-of-sample forecasting.

        Returns
        -------
        list[float]
            ψ-lag vector of length p.

        Raises
        ------
        ValueError
            Never raised directly.

        Notes
        -----
        - These values are used as initial ψ-lags by the forecast recursion.
        """
        return list(self._inner.psi_lags)
