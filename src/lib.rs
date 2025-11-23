//! rust_timeseries — high-performance time-series utilities with Python bindings.
//!
//! Purpose
//! -------
//! Serve as the crate root for Rust callers and as the PyO3 bridge that exposes
//! core time-series routines to Python via the `_rust_timeseries` extension
//! module. When the `python-bindings` feature is enabled, this module defines
//! the Python-facing classes and submodules used by the `rust_timeseries`
//! package.
//!
//! Key behaviors
//! -------------
//! - Re-export the core Rust modules (`duration` and `statistical_tests`)
//!   as the public crate surface.
//! - Define `#[pyclass]` wrappers and the `#[pymodule]` initializer for the
//!   `_rust_timeseries` Python extension.
//! - Create and register Python submodules (`statistical_tests`,
//!   `duration_models`) under `rust_timeseries` so that dot-notation imports
//!   work as expected.
//!
//! Invariants & assumptions
//! ------------------------
//! - All heavy numerical work is implemented in the inner Rust modules; this
//!   file performs only FFI glue, input validation, and error mapping.
//! - When `python-bindings` is enabled, the Python-visible types mirror the
//!   invariants and signatures of their Rust counterparts (e.g. `ACDModel`,
//!   `ELOutcome`).
//! - On successful conversion from Python objects to Rust types, the
//!   invariants documented in the core modules are assumed to hold.
//!
//! Conventions
//! -----------
//! - Python-exposed classes live under `_rust_timeseries.<submodule>` and are
//!   typically wrapped by thin pure-Python facades in the top-level
//!   `rust_timeseries` package.
//! - Indexing, units, and statistical conventions follow the documentation of
//!   the underlying Rust modules (`duration::core`, `statistical_tests`, etc.).
//! - Errors from core Rust code are propagated as rich error types internally
//!   and converted to `PyErr` values at the PyO3 boundary.
//!
//! Downstream usage
//! ----------------
//! - Native Rust code should usually depend directly on the inner modules and
//!   can ignore the PyO3 items guarded by the `python-bindings` feature.
//! - The Python packaging layer imports the `_rust_timeseries` module defined
//!   here and wraps its classes in user-facing Python APIs.
//! - External users are expected to interact with either the safe Rust APIs or
//!   the pure-Python wrappers; the PyO3 plumbing is considered internal.
//!
//! Testing notes
//! -------------
//! - Core numerical behavior is covered by unit tests in the inner modules and
//!   by Python integration tests that exercise the `_rust_timeseries` module.
//! - Smoke tests for the PyO3 bindings verify that classes can be constructed,
//!   called, and round-tripped correctly from Python.

pub mod duration;
pub mod inference;
pub mod optimization;
pub mod statistical_tests;
pub mod utils;

#[cfg(feature = "python-bindings")]
use ndarray::Array1;

#[cfg(feature = "python-bindings")]
use numpy::PyReadonlyArray1;

#[cfg(feature = "python-bindings")]
use pyo3::{exceptions::PyValueError, prelude::*, types::PyAny};

#[cfg(feature = "python-bindings")]
use crate::{
    duration::{
        core::{innovations::ACDInnovation, params::ACDParams},
        errors::ACDError,
        models::acd::ACDModel,
    },
    optimization::loglik_optimizer::traits::OptimOutcome,
    statistical_tests::escanciano_lobato::ELOutcome,
    utils::{build_acd_model, extract_acd_data, extract_f64_array, extract_hac_options},
};

/// EscancianoLobato — Python-facing wrapper for the EL heteroskedasticity robust test.
///
/// Purpose
/// -------
/// Represent the result of the Escanciano–Lobato heteroskedasticity robust test
/// when called from Python and forward all computation to [`ELOutcome`].
///
/// Key behaviors
/// -------------
/// - Validate and convert Python inputs into a contiguous `f64` slice.
/// - Run the EL test via [`ELOutcome::escanciano_lobato`] and store the outcome
///   internally.
/// - Expose scalar accessors (`statistic`, `pvalue`, `p_tilde`) as
///   Python properties.
///
/// Parameters
/// ----------
/// Constructed from Python via `EscancianoLobato(data, q=2.4, d=None)`:
/// - `data`: `&PyAny`
///   One-dimensional array-like of `f64` values with no NaNs and length ≥ 1.
/// - `q`: `Option<f64>`
///   Positive proxy order; defaults to `2.4` when `None`.
/// - `d`: `Option<usize>`
///   Positive maximum lag; defaults to `⌊n^0.2⌋` where `n = len(data)`.
///
/// Fields
/// ------
/// - `inner`: [`ELOutcome`]
///   Rust-side container holding the full test outcome used by the accessors.
///
/// Invariants
/// ----------
/// - `data.len() > 0` and `data` contains no `NaN`s at construction time.
/// - `q > 0` and `d > 0`.
///
/// Performance
/// -----------
/// - At most one allocation is performed to copy Python data into a Rust
///   buffer when needed; property access is O(1).
///
/// Notes
/// -----
/// - This type is primarily intended to be used from Python; native Rust code
///   should prefer calling [`ELOutcome::escanciano_lobato`] directly.
#[cfg(feature = "python-bindings")]
#[pyclass(module = "rust_timeseries.statistical_tests")]
pub struct EscancianoLobato {
    /// The EL test result struct.
    inner: ELOutcome,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl EscancianoLobato {
    /// Result of the Escanciano–Lobato heteroskedasticity proxy \(EL\) test.
    ///
    /// Returned by [`statistical_tests.escanciano_lobato`].  
    /// The statistic is asymptotically χ²(1) under the null.
    #[new]
    #[pyo3(
        text_signature = "(data, /, q=2.4, d=None)",
        signature = (raw_data, q = 2.4, d = None)
    )]
    #[allow(clippy::self_named_constructors)]
    pub fn escanciano_lobato<'py>(
        py: Python<'py>, raw_data: &Bound<'py, PyAny>, q: Option<f64>, d: Option<usize>,
    ) -> PyResult<EscancianoLobato> {
        let q: f64 = q.map_or(Ok(2.4), |v| {
            if v > 0.0 { Ok(v) } else { Err(PyValueError::new_err("q must be positive")) }
        })?;

        let arr: PyReadonlyArray1<f64> = extract_f64_array(py, raw_data)?;
        let data: &[f64] = arr
            .as_slice()
            .expect("expected a 1-D numpy.ndarray, pandas.Series, or sequence of float64");

        if data.is_empty() {
            return Err(PyValueError::new_err("data must not be empty"));
        }
        if data.iter().any(|&v| v.is_nan()) {
            return Err(PyValueError::new_err("data must not contain NaN values"));
        }

        let default_d: usize = (data.len() as f64).powf(0.2) as usize;
        let d: usize = d.map_or(Ok(default_d), |v| {
            if v > 0 { Ok(v) } else { Err(PyValueError::new_err("d must be positive")) }
        })?;
        let result = ELOutcome::escanciano_lobato(data, q, d)?;
        Ok(EscancianoLobato { inner: result })
    }

    /// The selected lag \(p\) that maximizes the penalized statistic.
    #[getter]
    pub fn p_tilde(&self) -> usize {
        self.inner.p_tilde()
    }

    /// The EL test statistic.
    #[getter]
    pub fn statistic(&self) -> f64 {
        self.inner.stat()
    }

    /// The p-value of the EL test.
    #[getter]
    pub fn pvalue(&self) -> f64 {
        self.inner.p_value()
    }
}

/// ACD — Python-facing wrapper for ACD(p, q) duration models.
///
/// Purpose
/// -------
/// Expose the [`ACDModel`] API to Python callers while preserving the core
/// Rust invariants and error handling.
///
/// Key behaviors
/// -------------
/// - Build an [`ACDModel`] with a chosen innovation family and options from
///   Python-friendly arguments.
/// - Provide `fit`, `forecast`, and `covariance_matrix` methods that convert
///   Python arrays into `ACDData` and delegate to the core implementation.
/// - Cache optimization, fitted-parameter, and forecast results for inspection
///   from Python via property getters.
///
/// Parameters
/// ----------
/// Constructed from Python via factory-style constructors:
/// - `ACD(...)`
///   Exponential-innovation ACD(p, q) model.
/// - `ACD.wacd(...)`
///   Weibull-innovation ACD(p, q) model with shape parameter `k`.
/// - `ACD.gacd(...)`
///   Generalized-gamma-innovation ACD(p, q) model with shape parameters
///   `p_shape` and `d_shape`.
///
/// Common parameters:
/// - `data_length`: `usize`
///   In-sample length used to size internal buffers and shape validation.
/// - `p`, `q`: `Option<usize>`
///   Optional ACD orders; At least one greater than 0.
/// - `init`, `init_fixed`, `init_psi_lags`, `init_durations_lags`
///   Initialization policy and associated values, matching [`Init`] semantics.
/// - `tol_grad`, `tol_cost`, `max_iter`, `line_searcher`, `lbfgs_mem`
///   Optimizer tolerances and configuration used to build [`MLEOptions`].
/// - `psi_guards`: `Option<(f64, f64)>`
///   Optional lower/upper bounds for ψ used to construct [`PsiGuards`].
///
/// Fields
/// ------
/// - `inner`: [`ACDModel`]
///   Fully configured ACD(p, q) model that owns scratch buffers and cached
///   results.
///
/// Invariants
/// ----------
/// - `inner` is always a well-formed [`ACDModel`] created through
///   [`build_acd_model`]; model order and buffer sizes are consistent with
///   `data_length`.
///
/// Performance
/// -----------
/// - All heavy numerical work occurs inside `inner`; this wrapper performs
///   only input conversion, dispatch, and error mapping.
///
/// Notes
/// -----
/// - Native Rust callers should usually work with [`ACDModel`] directly; this
///   type exists solely for the PyO3 binding surface.
#[cfg(feature = "python-bindings")]
#[pyclass(module = "rust_timeseries.duration_models", unsendable)]
pub struct ACD {
    /// Underlying Rust ACDModel.
    pub inner: ACDModel,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl ACD {
    #[new]
    #[pyo3(
        signature = (
            data_length,
            p = None,
            q = None,
            init = None,
            init_fixed = None,
            init_psi_lags = None,
            init_durations_lags = None,
            tol_grad = None,
            tol_cost = None,
            max_iter = None,
            line_searcher = None,
            lbfgs_mem = None,
            psi_guards = None,
        ),
        text_signature = "(data_length, /, p=None, q=None, init=None, init_fixed=None, \
                          init_psi_lags=None, init_durations_lags=None, tol_grad=None, \
                          tol_cost=None, max_iter=None, line_searcher=None, \
                          lbfgs_mem=None, psi_guards=None)"
    )]
    pub fn eacd<'py>(
        py: Python<'py>, data_length: usize, p: Option<usize>, q: Option<usize>,
        init: Option<&str>, init_fixed: Option<f64>, init_psi_lags: Option<&Bound<'py, PyAny>>,
        init_durations_lags: Option<&Bound<'py, PyAny>>, tol_grad: Option<f64>,
        tol_cost: Option<f64>, max_iter: Option<usize>, line_searcher: Option<&str>,
        lbfgs_mem: Option<usize>, psi_guards: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        let innovation = ACDInnovation::exponential();
        let inner = build_acd_model(
            py,
            data_length,
            innovation,
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
        )?;
        Ok(ACD { inner })
    }

    #[staticmethod]
    #[pyo3(
        signature = (
            data_length,
            k,
            p = None,
            q = None,
            init = None,
            init_fixed = None,
            init_psi_lags = None,
            init_durations_lags = None,
            tol_grad = None,
            tol_cost = None,
            max_iter = None,
            line_searcher = None,
            lbfgs_mem = None,
            psi_guards = None,
        ),
        text_signature = "(data_length, k, /, p=None, q=None, init=None, init_fixed=None, \
                          init_psi_lags=None, init_durations_lags=None, tol_grad=None, \
                          tol_cost=None, max_iter=None, line_searcher=None, \
                          lbfgs_mem=None, psi_guards=None)"
    )]
    pub fn wacd<'py>(
        py: Python<'py>, data_length: usize, k: f64, p: Option<usize>, q: Option<usize>,
        init: Option<&str>, init_fixed: Option<f64>, init_psi_lags: Option<&Bound<'py, PyAny>>,
        init_durations_lags: Option<&Bound<'py, PyAny>>, tol_grad: Option<f64>,
        tol_cost: Option<f64>, max_iter: Option<usize>, line_searcher: Option<&str>,
        lbfgs_mem: Option<usize>, psi_guards: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        let innovation = ACDInnovation::weibull(k)?;
        let inner = build_acd_model(
            py,
            data_length,
            innovation,
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
        )?;
        Ok(ACD { inner })
    }

    #[staticmethod]
    #[pyo3(
        signature = (
            data_length,
            p_shape,
            d_shape,
            p = None,
            q = None,
            init = None,
            init_fixed = None,
            init_psi_lags = None,
            init_durations_lags = None,
            tol_grad = None,
            tol_cost = None,
            max_iter = None,
            line_searcher = None,
            lbfgs_mem = None,
            psi_guards = None,
        ),
        text_signature = "(data_length, p_shape, d_shape, /, p=None, q=None, init=None, init_fixed=None, \
                          init_psi_lags=None, init_durations_lags=None, tol_grad=None, \
                          tol_cost=None, max_iter=None, line_searcher=None, \
                          lbfgs_mem=None, psi_guards=None)"
    )]
    pub fn gacd<'py>(
        py: Python<'py>, data_length: usize, p_shape: f64, d_shape: f64, p: Option<usize>,
        q: Option<usize>, init: Option<&str>, init_fixed: Option<f64>,
        init_psi_lags: Option<&Bound<'py, PyAny>>, init_durations_lags: Option<&Bound<'py, PyAny>>,
        tol_grad: Option<f64>, tol_cost: Option<f64>, max_iter: Option<usize>,
        line_searcher: Option<&str>, lbfgs_mem: Option<usize>, psi_guards: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        let innovation = ACDInnovation::generalized_gamma(p_shape, d_shape)?;
        let inner = build_acd_model(
            py,
            data_length,
            innovation,
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
        )?;
        Ok(ACD { inner })
    }

    #[pyo3(
        signature = (
            durations,
            theta0,
            unit = None,
            t0 = None,
            diurnal_adjusted = None,
        ),
        text_signature = "(self, durations, theta0, /, unit='seconds', t0=None, diurnal_adjusted=False)"
    )]
    pub fn fit<'py>(
        &mut self, py: Python<'py>, durations: &Bound<'py, PyAny>, theta0: &Bound<'py, PyAny>,
        unit: Option<&str>, t0: Option<usize>, diurnal_adjusted: Option<bool>,
    ) -> PyResult<()> {
        let acd_data = extract_acd_data(py, durations, unit, t0, diurnal_adjusted)?;
        let theta_arr = extract_f64_array(py, theta0)?;
        let theta_slice = theta_arr.as_slice().map_err(|_| {
            PyValueError::new_err("theta0 must be a 1-D contiguous float64 array or sequence")
        })?;
        let theta_vec = Array1::from(theta_slice.to_vec());
        self.inner.fit(theta_vec, &acd_data).map_err(ACDError::from)?;

        Ok(())
    }

    #[pyo3(
        signature = (
            durations,
            horizon,
            unit = None,
            t0 = None,
            diurnal_adjusted = None,
        ),
        text_signature = "(self, durations, horizon, /, unit='seconds', t0=None, diurnal_adjusted=False)"
    )]
    pub fn forecast<'py>(
        &mut self, py: Python<'py>, durations: &Bound<'py, PyAny>, horizon: usize,
        unit: Option<&str>, t0: Option<usize>, diurnal_adjusted: Option<bool>,
    ) -> PyResult<f64> {
        let acd_data = extract_acd_data(py, durations, unit, t0, diurnal_adjusted)?;
        let psi_hat = self.inner.forecast(horizon, &acd_data)?;
        Ok(psi_hat)
    }

    #[pyo3(
        signature = (
            durations,
            unit = None,
            t0 = None,
            diurnal_adjusted = None,
            robust = None,
            kernel = None,
            bandwidth = None,
            center = None,
            small_sample_correction = None,
        ),
        text_signature = "(self, durations, /, unit='seconds', t0=None, diurnal_adjusted=False, \
                          robust=False, kernel='bartlett', bandwidth=None, center=False, \
                          small_sample_correction=True)"
    )]
    pub fn covariance_matrix<'py>(
        &mut self, py: Python<'py>, durations: &Bound<'py, PyAny>, unit: Option<&str>,
        t0: Option<usize>, diurnal_adjusted: Option<bool>, robust: Option<bool>,
        kernel: Option<&str>, bandwidth: Option<usize>, center: Option<bool>,
        small_sample_correction: Option<bool>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let acd_data = extract_acd_data(py, durations, unit, t0, diurnal_adjusted)?;
        let hac_opts = if robust.unwrap_or(false) {
            Some(extract_hac_options(kernel, bandwidth, center, small_sample_correction)?)
        } else {
            None
        };

        let cov = self.inner.covariance_matrix(&acd_data, hac_opts.as_ref())?;

        // Convert Array2<f64> → Vec<Vec<f64>> (row-major)
        let (nrows, _ncols) = cov.dim();
        let mut out = Vec::with_capacity(nrows);
        for i in 0..nrows {
            out.push(cov.row(i).to_vec());
        }
        Ok(out)
    }

    #[getter]
    pub fn results(&self) -> PyResult<ACDOptimOutcome> {
        match &self.inner.results {
            Some(outcome) => Ok(ACDOptimOutcome { inner: outcome.clone() }),
            None => Err(ACDError::ModelNotFitted.into()),
        }
    }

    #[getter]
    pub fn fitted_params(&self) -> PyResult<ACDFittedParams> {
        match &self.inner.fitted_params {
            Some(params) => Ok(ACDFittedParams { inner: params.clone() }),
            None => Err(ACDError::ModelNotFitted.into()),
        }
    }

    #[getter]
    pub fn forecast_result(&self) -> Vec<f64> {
        match &self.inner.forecast_result {
            Some(fr) => fr.psi_forecast.borrow().to_vec(),
            None => Vec::new(),
        }
    }
}

/// ACDOptimOutcome — optimization outcome for an ACD model exposed to Python.
///
/// Purpose
/// -------
/// Present the key optimizer diagnostics from [`OptimOutcome`] to Python code
/// in a lightweight, read-only wrapper.
///
/// Key behaviors
/// -------------
/// - Hold the final parameter vector `theta_hat` and scalar diagnostics such as
///   objective value, convergence flag, status string, iteration count, and
///   gradient norm.
/// - Provide accessors that clone or copy the underlying values into
///   Python-owned containers.
///
/// Parameters
/// ----------
/// Instances are constructed internally by the `ACD.results` getter and are
/// not created directly by user code.
///
/// Fields
/// ------
/// - `inner`: [`OptimOutcome`]
///   Full optimizer result from the log-likelihood maximization.
///
/// Invariants
/// ----------
/// - `inner` always corresponds to the most recent call to
///   [`ACDModel::fit`] on the owning model.
///
/// Performance
/// -----------
/// - Accessors are O(n) only in the length of `theta_hat` and `fn_evals` when
///   cloning into Python; other fields are scalar copies.
///
/// Notes
/// -----
/// - This type is part of the Python FFI surface; Rust code should prefer
///   using [`OptimOutcome`] directly.
#[cfg(feature = "python-bindings")]
#[pyclass(module = "rust_timeseries.duration_models")]
pub struct ACDOptimOutcome {
    /// Underlying Rust OptimOutcome.
    pub inner: OptimOutcome,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl ACDOptimOutcome {
    #[getter]
    pub fn theta_hat(&self) -> Vec<f64> {
        self.inner.theta_hat.to_vec()
    }

    #[getter]
    pub fn value(&self) -> f64 {
        self.inner.value
    }

    #[getter]
    pub fn converged(&self) -> bool {
        self.inner.term_status
    }

    #[getter]
    pub fn status(&self) -> String {
        self.inner.status.clone()
    }

    #[getter]
    pub fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    pub fn grad_norm(&self) -> Option<f64> {
        self.inner.grad_norm
    }

    #[getter]
    pub fn fn_evals(&self) -> Vec<(String, u64)> {
        self.inner.fn_evals.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }
}

/// ACDFittedParams — fitted model-space parameters for an ACD model.
///
/// Purpose
/// -------
/// Provide Python access to the model-space parameters obtained at the fitted
/// optimum of an [`ACDModel`].
///
/// Key behaviors
/// -------------
/// - Expose `omega`, `alpha`, `beta`, `slack`, and `psi_lags` as
///   copy-on-access properties for Python callers.
/// - Mirror the structure of [`ACDParams`] without exposing internal
///   validators to Python.
///
/// Parameters
/// ----------
/// Instances are constructed internally by the `ACD.fitted_params` getter and
/// are not created directly by user code.
///
/// Fields
/// ------
/// - `inner`: [`ACDParams`]
///   Validated model-space parameters corresponding to the last fitted model.
///
/// Invariants
/// ----------
/// - `inner` satisfies all invariants documented on [`ACDParams`], including
///   positivity, stationarity, and shape constraints.
///
/// Performance
/// -----------
/// - Getter methods allocate only when converting `ndarray` vectors into
///   heap-allocated `Vec<f64>` for Python consumption.
///
/// Notes
/// -----
/// - Rust callers should use [`ACDParams`] directly; this wrapper exists
///   solely for the PyO3 binding.
#[cfg(feature = "python-bindings")]
#[pyclass(module = "rust_timeseries.duration_models")]
pub struct ACDFittedParams {
    pub inner: ACDParams,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl ACDFittedParams {
    #[getter]
    pub fn omega(&self) -> f64 {
        self.inner.omega
    }

    #[getter]
    pub fn slack(&self) -> f64 {
        self.inner.slack
    }

    #[getter]
    pub fn alpha(&self) -> Vec<f64> {
        self.inner.alpha.to_vec()
    }

    #[getter]
    pub fn beta(&self) -> Vec<f64> {
        self.inner.beta.to_vec()
    }

    #[getter]
    pub fn psi_lags(&self) -> Vec<f64> {
        self.inner.psi_lags.to_vec()
    }
}

/// _rust_timeseries — PyO3 module initializer for the Python extension.
///
/// Purpose
/// -------
/// Define the `_rust_timeseries` Python module and register its submodules used
/// by the public `rust_timeseries` package.
///
/// Key behaviors
/// -------------
/// - Create `statistical_tests` and `duration_models` submodules.
/// - Attach those submodules to the parent `_rust_timeseries` module.
/// - Register the submodules in `sys.modules` so they are importable via
///   dotted paths from Python.
///
/// Parameters
/// ----------
/// - `_py`: [`Python`]
///   GIL token provided by PyO3 during module initialization.
/// - `m`: `&Bound<PyModule>`
///   Module object representing `_rust_timeseries`.
///
/// Returns
/// -------
/// `PyResult<()>`
///   `Ok(())` on success, or a Python exception if registration fails.
///
/// Errors
/// ------
/// - `PyErr`
///   If creating submodules or manipulating `sys.modules` fails.
///
/// Panics
/// ------
/// - Never panics under normal operation; all failures are mapped into
///   `PyErr`.
///
/// Notes
/// -----
/// - This function is invoked automatically by Python when importing the
///   compiled extension; it is not called directly by user code.
#[cfg(feature = "python-bindings")]
#[pymodule]
fn _rust_timeseries<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    let statistical_tests_mod = PyModule::new(_py, "statistical_tests")?;
    let duration_models_mod = PyModule::new(_py, "duration_models")?;
    statistical_tests(_py, m, &statistical_tests_mod)?;
    duration_models(_py, m, &duration_models_mod)?;

    // Manually add submodules into sys.modules to allow for dot notation.
    _py.import("sys")?
        .getattr("modules")?
        .set_item("rust_timeseries.statistical_tests", statistical_tests_mod)?;

    _py.import("sys")?
        .getattr("modules")?
        .set_item("rust_timeseries.duration_models", duration_models_mod)?;
    Ok(())
}

#[cfg(feature = "python-bindings")]
fn statistical_tests<'py>(
    _py: Python, rust_timeseries: &Bound<'py, PyModule>, m: &Bound<'py, PyModule>,
) -> PyResult<()> {
    m.add_class::<EscancianoLobato>()?;
    rust_timeseries.add_submodule(m)?;
    Ok(())
}

#[cfg(feature = "python-bindings")]
fn duration_models<'py>(
    _py: Python, rust_timeseries: &Bound<'py, PyModule>, m: &Bound<'py, PyModule>,
) -> PyResult<()> {
    m.add_class::<ACD>()?;
    m.add_class::<ACDOptimOutcome>()?;
    m.add_class::<ACDFittedParams>()?;
    rust_timeseries.add_submodule(m)?;
    Ok(())
}
