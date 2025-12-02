# rust_timeseries — Python bindings for duration models and tests

`rust_timeseries` is a **Python-first** library that wraps high-performance Rust code for:

- modelling event-time durations with ACD-type models,
- calling a Rust-implemented maximum-likelihood optimizer,
- extracting fitted parameters and optimizer diagnostics, and
- running the Escanciano–Lobato (2009) portmanteau test, which is robust to conditional heteroskedasticity.

All heavy computation lives in Rust; the public surface is a small, typed Python API exposed via the compiled `_rust_timeseries` extension and the stub files:

- `duration_models.pyi`
- `statistical_tests.pyi`

This README documents only what is actually exposed through those bindings.

---

## 1. Installation

### 1.1 From PyPI (recommended)

Version `1.1.0` is available on PyPI:

```bash
pip install rust_timeseries
```

The project currently ships wheels for **Python 3.11–3.13** on common Linux, macOS, and Windows targets.  
For unsupported platforms/versions, you can build from source (see below).

### 1.2 From source (development / contributing)

You’ll need a recent Rust **stable** toolchain and Python ≥ 3.11.

Clone the repo and create a virtual environment:

```bash
git clone https://github.com/mickwise/rust_timeseries.git
cd rust_timeseries

python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -U pip maturin
maturin develop --release
```

This builds the extension in place and installs it into your current environment. The Python usage is identical to the PyPI installation.

---

## 2. Python modules

The compiled extension `_rust_timeseries.*` is **not** imported directly; instead, use the public modules:

```text
rust_timeseries/
    duration_models.py      # ACD models and related utilities
    statistical_tests.py    # Escanciano–Lobato test
    _rust_timeseries.*      # compiled extension (internal)
```

Use:

```python
import rust_timeseries
from rust_timeseries import duration_models, statistical_tests
# or
from rust_timeseries.duration_models import ACD
from rust_timeseries.statistical_tests import EscancianoLobato
```

Do **not** import from internal Rust modules (`duration`, `optimization`, etc.). Always go through the public Python modules above.

---

## 3. Duration models (`rust_timeseries.duration_models`)

The `duration_models` module exposes ACD-type processes via the `ACD` class.

> **Note:** Only the main surface API is documented here.  
> For full signatures, see the Python type stubs (`duration_models.pyi`) and the docstrings.

### 3.1 Minimal example

```python
import numpy as np
from rust_timeseries.duration_models import ACD

# synthetic durations (strictly positive floats)
durations = 1.0 + np.abs(np.random.randn(200))

# unconstrained parameter guess (length 1 + p + q)
theta0 = np.zeros(3, dtype=np.float64)  # (ω, α₁, β₁)

# configure ACD(1,1); see docstring for all options
model = ACD(data_length=len(durations), p=1, q=1)

# fit model in unconstrained parameter space
model.fit(durations=durations, theta0=theta0)

# short-horizon forecast of conditional duration
psi_hat = model.forecast(durations=durations, horizon=5)
print("ψ(5) =", psi_hat)

# classical vs. HAC-robust covariance matrices for θ
cov_classical = model.covariance_matrix(durations, robust=False)
cov_hac       = model.covariance_matrix(durations, robust=True)

print("classical cov shape:", np.asarray(cov_classical).shape)
print("HAC cov shape      :", np.asarray(cov_hac).shape)
```

Key methods (high level):

- `ACD(data_length: int, p: int, q: int, **options)`  
  Construct an ACD(p, q) model for a given sample length. Extra options (e.g., stationarity margins, backcasting choices) are documented in the class docstring.

- `fit(durations: np.ndarray, theta0: np.ndarray, ...) -> None`  
  Estimate parameters in an unconstrained space (internally mapped to a stationary region).

- `forecast(durations: np.ndarray, horizon: int, ...) -> float`  
  Run an out-of-sample forecast and return the final ψ at the requested horizon.

- `covariance_matrix(durations: np.ndarray, robust: bool = False, ...) -> list[list[float]]`  
  Return model-based (`robust=False`) or HAC-robust (`robust=True`) covariance matrices for the unconstrained parameter vector.

- `standard_errors(...) -> list[float]`  
  Convenience wrapper that returns just the standard errors (diagonal of the chosen covariance matrix).

- `results -> ACDOptimOutcome`  
  Optimizer diagnostics (status, iterations, gradient norm, evaluation counts, etc.).

- `fitted_params -> ACDFittedParams`  
  Fitted model-space parameters (ω, slack, α, β, ψ-lags) that satisfy the stationarity and positivity constraints enforced in Rust.

### 3.2 Constructors and variants

The main constructor:

```python
ACD(
    data_length: int,
    p: int,
    q: int,
    *,
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
```

An exponential innovation distribution is used by default. Two alternative constructors are available:

- `ACD.wacd(...)` – Weibull-innovation ACD(p, q) with shape parameter `k`.
- `ACD.gacd(...)` – generalized-gamma-innovation ACD(p, q) with shape parameters `p_shape` and `d_shape`.

Both accept the same core configuration arguments as `ACD` (`data_length`, `p`, `q`, optimizer settings), plus their respective shape parameters.

### 3.3 Standard errors (`ACD.covariance_matrix`)

`covariance_matrix` computes parameter covariance matrix for a given duration series, optionally with HAC corrections:

```python
se = model.covariance_matrix(
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
```

- When `robust=False`, covariance matrix is based on the model-based information matrix.
- When `robust=True`, a HAC estimator is used (kernel and bandwidth options are forwarded to the Rust implementation).

### 3.4 Optimization diagnostics (`ACDOptimOutcome`)

The `results` property returns an `ACDOptimOutcome` object, wrapping the Rust `OptimOutcome` type.

Attributes:

- `theta_hat: list[float]` – unconstrained parameter vector at the solution.
- `value: float` – objective value at the solution.
- `converged: bool` – whether the optimizer terminated successfully.
- `status: str` – human-readable termination reason.
- `iterations: int` – number of iterations taken.
- `grad_norm: float | None` – norm of the gradient at the solution, when available.
- `fn_evals: list[tuple[str, int]]` – evaluation counters keyed by function name.

Typical usage:

```python
outcome = model.results
print("Converged:", outcome.converged, "-", outcome.status)
print("Objective value:", outcome.value)
print("Iterations:", outcome.iterations)
print("Gradient norm:", outcome.grad_norm)
print("Function evals:", outcome.fn_evals)
```

### 3.5 Fitted parameters (`ACDFittedParams`)

The `fitted_params` property returns an `ACDFittedParams` instance that mirrors the Rust `ACDParams` struct:

- `omega: float` – baseline level parameter.
- `slack: float` – positive slack term used to enforce strict stationarity.
- `alpha: list[float]` – ARCH-type coefficients (size `p`).
- `beta: list[float]` – GARCH-type coefficients (size `q`).
- `psi_lags: list[float]` – initialization lags for ψₜ.

Example:

```python
params = model.fitted_params
print("omega:", params.omega)
print("slack:", params.slack)
print("alpha:", params.alpha)
print("beta:", params.beta)
print("psi_lags:", params.psi_lags)
```

All invariants (positivity, stationarity, shape constraints) are enforced on the Rust side; Python sees only valid parameter configurations.

---

## 4. Escanciano–Lobato test (`rust_timeseries.statistical_tests`)

The `statistical_tests` module currently exposes a single class, `EscancianoLobato`, implementing the Escanciano–Lobato heteroskedasticity proxy test.

### 4.1 Constructing `EscancianoLobato`

```python
import numpy as np
from rust_timeseries.statistical_tests import EscancianoLobato

residuals = np.asarray(residuals, dtype=float)

el = EscancianoLobato(
    residuals,  # 1-D array-like of float64, no NaNs, length >= 1
    q=2.4,      # positive float proxy order (optional)
    d=None,     # positive integer max lag; default floor(n**0.2)
)
```

Signature (from `statistical_tests.pyi`):

```python
class EscancianoLobato:
    def __init__(
        self,
        data,
        q: float | None = 2.4,
        d: int | None = None,
    ) -> None: ...
```

Validation rules enforced in the bindings:

- `data` must be non-empty and contain no NaNs;
- `q` must be positive when provided;
- `d` must be positive when provided.

Invalid inputs raise a `ValueError` from the PyO3 binding.

### 4.2 Accessing test results

`EscancianoLobato` wraps a Rust `ELOutcome` and exposes three read-only properties:

- `statistic: float` – the test statistic;
- `pvalue: float` – asymptotic p-value under the null;
- `p_tilde: int` – data-driven lag choice that maximizes the penalized statistic.

Example:

```python
print("EL statistic:", el.statistic)
print("EL p-value  :", el.pvalue)
print("Selected lag p~:", el.p_tilde)
```

---

Here’s a copy-pasteable markdown chunk you can drop straight into the README to document the HAC API.

---

### Where to put it

Put this **between** your current sections:

* **After** `## 4. Escanciano–Lobato test (`rust_timeseries.statistical_tests`)`
* **Before** `## 5. Design notes`

…and then **renumber**:

* `## 5. Design notes` → `## 6. Design notes`
* `## 6. Status and extension points` → `## 7. Status and extension points`
* `## 7. License` → `## 8. License`

So the new section below becomes **`## 5. HAC covariance estimation (`rust_timeseries.hac_estimation`)`**.

---

## 5. HAC covariance estimation (`rust_timeseries.hac_estimation`)

In addition to the ACD model’s built-in `covariance_matrix(…, robust=True)` method,
`rust_timeseries` exposes a **standalone HAC covariance helper** for users who
already have per-observation score vectors (or similar objects) and just want a
Newey–West–style covariance of their *average* score.

The public entry point is:

```python
from rust_timeseries.hac_estimation import estimate_hac_covariance_matrix
```

### 5.1 Minimal example

```python
import numpy as np
from rust_timeseries.hac_estimation import estimate_hac_covariance_matrix

rng = np.random.default_rng(12345)

# Fake "scores": n observations, k parameters
n, k = 500, 4
scores = rng.normal(size=(n, k)).astype(np.float64)

cov_hac = estimate_hac_covariance_matrix(
    scores,
    kernel="bartlett",            # or "iid", "parzen", "quadratic_spectral"
    bandwidth=None,               # plug-in bandwidth if None
    center=False,                 # optionally demean the columns
    small_sample_correction=True, # Newey–West finite-sample scaling
)

print("HAC cov shape:", cov_hac.shape)   # (k, k)
print("HAC cov diag :", np.diag(cov_hac))
```

**Input conventions**

* `data`: array-like of shape `(n, k)` with `float64` entries.

  * Rows = observation or time index.
  * Columns = score components / parameters.
  * Accepted types: `numpy.ndarray`, nested Python sequences, etc.
* The function validates that the matrix is non-empty and rectangular; invalid
  inputs raise `ValueError`.

**Output**

* Returns a NumPy array of shape `(k, k)` containing the HAC covariance matrix
  of the **average** score vector:

  [
  \hat{\Sigma}*{\bar{s}} \approx \operatorname{Cov}\left(\frac{1}{n}
  \sum*{t=1}^n s_t\right)
  ]

  This is the scale typically used inside sandwich-style variance formulas.

### 5.2 Relationship to `ACD.covariance_matrix`

For ACD models, you usually do **not** need to call
`estimate_hac_covariance_matrix` directly:

* `ACD.covariance_matrix(..., robust=False)`
  returns a model-based covariance using the observed information matrix.

* `ACD.covariance_matrix(..., robust=True, kernel=..., bandwidth=..., ...)`
  uses the **same HAC machinery** under the hood, applied to ACD score vectors,
  and then maps the result from unconstrained parameter space to the model-space
  parameters ((\omega, \alpha, \beta)) via a delta method.

The standalone `hac_estimation` module is intended for **“bring your own
scores”** workflows where you want to plug HAC covariances into your own
estimators or sandwich formulas outside the ACD context.


## 6. Design notes

- **Python-first.** The core lives in Rust, but the bindings are designed so that quantitative researchers can stay in Python. Inputs are standard array-like objects (`numpy.ndarray`, `pandas.Series`, lists); outputs are plain Python scalars and lists.
- **Tight coupling to Rust invariants.** The PyO3 layer performs shape checks, basic validation, and error mapping. All model invariants (positivity, stationarity, etc.) are enforced in the Rust types (`ACDModel`, `ACDParams`, `ELOutcome`).
- **Explicit error reporting.** Rust errors (through the `ACDError` type) are surfaced as Python exceptions with clear messages. Fitted-state-dependent getters (`results`, `fitted_params`) fail loudly if you call them before `fit`.
- **Minimal, inspectable objects.** The Python wrappers (`ACD`, `ACDOptimOutcome`, `ACDFittedParams`, `EscancianoLobato`) are intentionally small and read-only. They are easy to log, serialize, or convert to dictionaries for downstream analysis.

---

## 7. Status and extension points

The current Python API is intentionally narrow and corresponds exactly to the bindings implemented in `lib.rs` and documented in `duration_models.pyi` and `statistical_tests.pyi`.

Natural extensions (planned to be added in Rust and then surfaced via the same binding pattern) include:

- simulation module for the ACD family.
- further goodness-of-fit and residual tests under `statistical_tests`.

When extending the library, keep the following steps:

1. Add functionality in the Rust core.
2. Expose it through a PyO3 wrapper (`#[pyclass]` / `#[pymethods]`).
3. Update the `.pyi` stubs to match.
4. Update this README to document the new public surface.

This keeps the README, stubs, and compiled extension in sync.

---

## 8. License

`rust_timeseries` is released under the MIT license. See `LICENSE` for details.