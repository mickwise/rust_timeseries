# rust_timeseries — Python bindings for duration models and tests

`rust_timeseries` is a **Python-first** package for

- modelling **event-time durations** with ACD-type models,
- calling a Rust-implemented **maximum-likelihood optimizer**,
- extracting fitted parameters and optimizer diagnostics, and
- running the **Escanciano–Lobato** heteroskedasticity proxy test.

All heavy computation runs in Rust (see `lib.rs`, `duration`, and `statistical_tests`),
but the public surface is a small, typed Python API defined by the compiled
`_rust_timeseries` extension and the stub files

- `duration_models.pyi`
- `statistical_tests.pyi`.

This document describes only what is actually exposed through those bindings.

---

## 1. Installation

From source, using `maturin` (development workflow):

    maturin develop --release

Once the wheel is installed into your environment you can import:

    import rust_timeseries as rts

and typically:

    from rust_timeseries import duration_models, statistical_tests

A future PyPI release will allow installation via

    pip install rust_timeseries

without changing the Python API.

---

## 2. Python modules

The Python-visible layout mirrors the bindings defined in `lib.rs`:

    rust_timeseries/
        duration_models.py      # ACD models and optimization results
        statistical_tests.py    # Escanciano–Lobato test
        _rust_timeseries.*      # compiled extension (internal)

Rust-only modules such as `duration`, `inference`, `optimization`, and `utils`
are implementation details and are **not** part of the public Python API.

Users should import only from

- `rust_timeseries.duration_models`
- `rust_timeseries.statistical_tests`.

---

## 3. Duration models (`rust_timeseries.duration_models`)

The `duration_models` module is the main entry point for ACD(p, q)-type duration
processes. The public classes implemented in `lib.rs` and described in
`duration_models.pyi` are:

- `ACD`               – duration model used from Python,
- `ACDOptimOutcome`   – optimization diagnostics,
- `ACDFittedParams`   – fitted model-space parameters.

### 3.1 Constructing an ACD model

`ACD` instances are configured once and reused across fits. The constructor
exposes the same arguments that the PyO3 wrapper forwards into Rust:

    from rust_timeseries.duration_models import ACD

    acd = ACD(
        data_length=len(durations),  # in-sample length
        p=1,                         # AR order of ψ_t (optional)
        q=1,                         # MA order of durations (optional)
        init=None,
        init_fixed=None,
        init_psi_lags=None,
        init_durations_lags=None,
        tol_grad=None,
        tol_cost=None,
        max_iter=None,
        line_searcher=None,
        lbfgs_mem=None,
        psi_guards=None,
    )

Key arguments (see `duration_models.pyi` for full details):

- `data_length: int`  
  Length of the in-sample duration series. This determines internal buffer
  sizes and is checked whenever you call `fit`, `forecast`, or
  `standard_errors`.
- `p: int | None`, `q: int | None`  
  Model orders. At least one of them must be positive.
- `init`, `init_fixed`, `init_psi_lags`, `init_durations_lags`  
  Initialization policy and associated values, forwarded unchanged to
  Rust-side initialization logic.
- `tol_grad`, `tol_cost`, `max_iter`, `line_searcher`, `lbfgs_mem`  
  Optimizer tolerances and configuration.
- `psi_guards: tuple[float, float] | None`  
  Optional lower/upper bounds on the conditional mean process ψ_t, mapped
  into Rust as `PsiGuards`.

The default constructor uses an **exponential** innovation distribution.
Two alternative constructors are available:

- `ACD.wacd(...)` – Weibull-innovation ACD(p, q) with shape parameter `k`.
- `ACD.gacd(...)` – generalized-gamma-innovation ACD(p, q) with shape
  parameters `p_shape` and `d_shape`.

Both `wacd` and `gacd` accept the same configuration arguments as `ACD`
(including `data_length`, `p`, `q`, and optimizer settings), plus their
respective shape parameters.

---

### 3.2 Fitting a model (`ACD.fit`)

Model fitting is performed by calling `fit` on an `ACD` instance. The method
is backed by the Rust function `ACDModel::fit` and stores results on the
Python object; it does **not** return the fitted parameters directly.

Minimal example:

    import numpy as np
    from rust_timeseries.duration_models import ACD

    durations = np.asarray(durations, dtype=float)

    acd = ACD(data_length=len(durations), p=1, q=1)

    # Initial parameter guess in the unconstrained parameterization.
    theta0 = np.asarray(theta0, dtype=float)

    acd.fit(
        durations=durations,
        theta0=theta0,
        unit="seconds",        # or other supported unit strings
        t0=None,               # optional index offset
        diurnal_adjusted=False # whether durations are already adjusted
    )

Signature (from the bindings):

    def fit(
        self,
        durations,
        theta0,
        unit: str | None = "seconds",
        t0: int | None = None,
        diurnal_adjusted: bool | None = False,
    ) -> None: ...

On success, the fitted model state is cached inside the Rust `ACDModel`.
You can then access:

- `acd.results`        – an `ACDOptimOutcome` instance with optimizer diagnostics,
- `acd.fitted_params`  – an `ACDFittedParams` instance with model parameters.

Both are defined as `@property`-style getters in `lib.rs`.

If you call `results` or `fitted_params` before any successful `fit`, the
bindings raise a Python exception originating from the Rust
`ACDError::ModelNotFitted` variant.

---

### 3.3 Forecasting (`ACD.forecast` and `ACD.forecast_result`)

Forecasts are produced via `ACD.forecast`, which delegates to the Rust
`ACDModel::forecast` and stores the forecast path internally:

    horizon = 20

    psi_h = acd.forecast(
        durations=durations,
        horizon=horizon,
        unit="seconds",
        t0=None,
        diurnal_adjusted=False,
    )

Signature:

    def forecast(
        self,
        durations,
        horizon: int,
        unit: str | None = "seconds",
        t0: int | None = None,
        diurnal_adjusted: bool | None = False,
    ) -> float: ...

The return value is the final forecasted ψ at the requested horizon, while the
full path is cached and exposed through

- `acd.forecast_result` – a 1-D list of `float` corresponding to the most
  recent forecast call.

If `forecast` has not yet been called, `forecast_result` returns an empty list.

---

### 3.4 Standard errors (`ACD.standard_errors`)

The `standard_errors` method computes model parameter standard errors for a
given duration series, optionally using **HAC** corrections. It forwards
options directly into the Rust-side HAC implementation via `extract_hac_options`
and `ACDModel::standard_errors`.

Example:

    se = acd.standard_errors(
        durations=durations,
        unit="seconds",
        t0=None,
        diurnal_adjusted=False,
        robust=True,
        kernel="bartlett",
        bandwidth=None,
        center=False,
        small_sample_correction=True,
    )

Signature:

    def standard_errors(
        self,
        durations,
        unit: str | None = "seconds",
        t0: int | None = None,
        diurnal_adjusted: bool | None = False,
        robust: bool | None = False,
        kernel: str | None = "bartlett",
        bandwidth: int | None = None,
        center: bool | None = False,
        small_sample_correction: bool | None = True,
    ) -> list[float]: ...

Return value:

- A Python `list[float]` with standard errors corresponding to the fitted
  parameter vector. When `robust=False`, these are based on the model-based
  information matrix; when `robust=True`, a HAC estimator is used.

---

### 3.5 Optimization diagnostics (`ACDOptimOutcome`)

The `ACD.results` property returns an `ACDOptimOutcome` object, which is a
thin wrapper around the Rust `OptimOutcome` type. The PyO3 bindings defined
in `lib.rs` expose the following attributes:

- `theta_hat: list[float]`  
  Final parameter vector in the unconstrained parameterization.
- `value: float`  
  Objective value at the solution (e.g. maximized log-likelihood or its
  negation, depending on configuration).
- `converged: bool`  
  Whether the optimizer terminated with a successful status.
- `status: str`  
  Human-readable termination reason.
- `iterations: int`  
  Number of iterations taken.
- `grad_norm: float | None`  
  Norm of the gradient at the solution, when available.
- `fn_evals: list[tuple[str, int]]`  
  Evaluation counters keyed by function name.

A typical access pattern:

    outcome = acd.results

    print("Converged:", outcome.converged, "-", outcome.status)
    print("Objective value:", outcome.value)
    print("Iterations:", outcome.iterations)
    print("Gradient norm:", outcome.grad_norm)
    print("Function evals:", outcome.fn_evals)

---

### 3.6 Fitted parameters (`ACDFittedParams`)

The `ACD.fitted_params` property returns an `ACDFittedParams` instance that
wraps the Rust `ACDParams` struct. The bindings in `lib.rs` expose the
following read-only properties:

- `omega: float`  
  Baseline level parameter.
- `slack: float`  
  Positive slack term used to enforce strict stationarity.
- `alpha: list[float]`  
  Vector of ARCH-type coefficients (size `p`).
- `beta: list[float]`  
  Vector of GARCH-type coefficients (size `q`).
- `psi_lags: list[float]`  
  Initialization lags for ψ_t used at the start of the sample.

Example:

    params = acd.fitted_params

    print("omega:", params.omega)
    print("slack:", params.slack)
    print("alpha:", params.alpha)
    print("beta:", params.beta)
    print("psi_lags:", params.psi_lags)

These values are guaranteed (by the Rust-side constructors) to satisfy the
model invariants documented in `ACDParams` (positivity, stationarity, shape
constraints).

---

## 4. Escanciano–Lobato test (`rust_timeseries.statistical_tests`)

The `statistical_tests` module currently exposes a single class,
`EscancianoLobato`, which implements the **Escanciano–Lobato heteroskedasticity
proxy test** from the Rust type `ELOutcome`.

### 4.1 Constructing `EscancianoLobato`

The constructor validates inputs, converts them into a contiguous `float64`
slice, runs the test in Rust, and stores the result:

    import numpy as np
    from rust_timeseries.statistical_tests import EscancianoLobato

    residuals = np.asarray(residuals, dtype=float)

    el = EscancianoLobato(
        residuals,  # 1-D array-like of float64, no NaNs, length >= 1
        q=2.4,      # positive float proxy order (optional)
        d=None,     # positive integer max lag; default floor(n**0.2)
    )

Signature (from `statistical_tests.pyi` and `lib.rs`):

    class EscancianoLobato:
        def __init__(
            self,
            data,
            q: float | None = 2.4,
            d: int | None = None,
        ) -> None: ...

Validation rules enforced in the bindings:

- `data` must be non-empty and contain no NaNs,
- `q` must be positive when provided,
- `d` must be positive when provided.

If any of these fail, a `ValueError` is raised from the PyO3 binding.

### 4.2 Accessing test results

`EscancianoLobato` wraps a Rust `ELOutcome` and exposes three read-only
properties:

- `statistic: float`  
  The test statistic.
- `pvalue: float`  
  Asymptotic p-value under the null.
- `p_tilde: int`  
  Data-driven lag choice that maximizes the penalized statistic.

Example:

    print("EL statistic:", el.statistic)
    print("EL p-value:", el.pvalue)
    print("Selected lag p~:", el.p_tilde)

---

## 5. Design notes

- **Python-first**  
  The core of the codebase lives in Rust, but the bindings are designed so
  that quantitative researchers can stay in Python. Inputs are standard
  array-like objects (`numpy.ndarray`, `pandas.Series`, Python lists), and
  outputs are plain Python scalars and lists.

- **Tight coupling to Rust invariants**  
  The PyO3 layer performs only shape checks, basic validation, and error
  mapping. All model invariants (positivity, stationarity, etc.) are enforced
  in the Rust types (`ACDModel`, `ACDParams`, `ELOutcome`).

- **Explicit error reporting**  
  Whenever a Rust error is raised (for example via the `ACDError` type), it is
  surfaced as a Python exception with a clear string message. Fitted-state-
  dependent getters (`results`, `fitted_params`) fail loudly if you call them
  before `fit`.

- **Minimal, inspectable objects**  
  The Python wrappers (`ACD`, `ACDOptimOutcome`, `ACDFittedParams`,
  `EscancianoLobato`) are intentionally small and read-only. They are easy
  to log, serialize, or convert to dictionaries for downstream analysis.

---

## 6. Status and extension points

The current Python API is intentionally narrow and corresponds exactly to the
bindings implemented in `lib.rs` and documented in `duration_models.pyi` and
`statistical_tests.pyi`. Natural extensions, to be added in Rust and then
surfaced through the same pattern, include:

- additional duration families that reuse the same `ACD` interface,
- further goodness-of-fit and residual tests under `statistical_tests`,
- higher-level helpers for multi-asset or panel duration data.

When extending the library, keep the following principles:

- add functionality first in the Rust core,
- expose it through a PyO3 wrapper (`#[pyclass]` or `#[pymethods]`),
- update the `.pyi` stubs to match,
- and only then document it here.

This keeps the README, the stubs, and the compiled extension **in sync**.
