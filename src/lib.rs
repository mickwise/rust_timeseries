//! # rust_timeseries
//!
//! High-performance time-series utilities exposed to Python via **PyO3**.
//!
//! ## Highlights
//! * Zero-copy ingestion of `numpy.ndarray` / `pandas.Series` buffers  
//! * Strict input validation with clear Python exceptions  
//! * Memory-safe Rust core for heavy numerical work  
//!
//! ### Python quick-start
//! ```python
//! import rust_timeseries as rts
//! test = rts.statistical_tests.EscancianoLobato(data, q=5, d=10)
//! test.statistic, test.pvalue
//! ```
//! See each function’s docstring for details.
//!
//! ---
//! ## Rust quick-start
//! ```rust
//! use rust_timeseries::utils::extract_f64_array;
//! # pyo3::prepare_freethreaded_python();  // in real code
//! Python::with_gil(|py| {
//!     let obj = ... ;                      // some PyAny
//!     let arr = extract_f64_array(py, &obj).unwrap();
//!     let slice = arr.as_slice().unwrap();
//!     // numeric code here
//! });
//! ```
////////////////////////////////////////////////////////////////////////////////

pub mod statistical_tests;
pub mod utils;
use crate::{statistical_tests::escanciano_lobato::ELResult, utils::extract_f64_array};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Result of the Escanciano–Lobato heteroscedasticity proxy \(EL\) test.
///
/// Returned by [`statistical_tests.escanciano_lobato`].  
/// The statistic is asymptotically χ²(1) under the null.
#[pyclass(module = "rust_timeseries.statistical_tests")]
pub struct EscancianoLobato {
    /// The EL test result struct.
    inner: ELResult,
}

#[pymethods]
impl EscancianoLobato {
    /// Escanciano–Lobato heteroskedasticity proxy test.
    ///
    /// Parameters
    /// ----------
    /// raw_data : array-like of float64, shape *(n,)*
    ///     Must not contain any NaN values.
    /// q : int, optional
    ///     Order of the proxy.  `q > 0`.  Default = 2.4.
    /// d : int, optional
    ///     Maximum lag.  Default = ⌊n^0.2⌋.
    ///
    /// Raises
    /// ------
    /// PyValueError
    ///    If `data` is empty, contains NaN values, or if `q` or `d` are not positive integers.
    ///
    /// Returns
    /// -------
    /// EscancianoLobato
    ///     Object with ``statistic``, ``pvalue`` and ``p_tilde`` attributes.
    ///
    /// Notes
    /// -----
    /// The statistic is asymptotically χ²(1) under the null.
    #[new]
    #[pyo3(
        text_signature = "(data, /, q=2.4, d=None)",
        signature = (raw_data, q = 2.4, d = None)
    )]
    pub fn escanciano_lobato<'py>(
        py: Python<'py>,
        raw_data: &Bound<'py, PyAny>,
        q: Option<f64>,
        d: Option<usize>,
    ) -> PyResult<EscancianoLobato> {
        let q: f64 = q.map_or(Ok(2.4), |v| {
            if v > 0.0 {
                Ok(v)
            } else {
                Err(PyValueError::new_err("q must be positive"))
            }
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
            if v > 0 {
                Ok(v)
            } else {
                Err(PyValueError::new_err("d must be positive"))
            }
        })?;
        let result = ELResult::escanciano_lobato(data, q, d)?;
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

#[pymodule]
fn rust_timeseries<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    statistical_tests(_py, m)?;
    Ok(())
}

fn statistical_tests<'py>(
    _py: Python<'py>,
    rust_timeseries: &Bound<'py, PyModule>,
) -> PyResult<()> {
    let statistical_tests =
        PyModule::new(rust_timeseries.py(), "rust_timeseries.statistical_tests")?;
    statistical_tests.add_class::<EscancianoLobato>()?;
    rust_timeseries.add_submodule(&statistical_tests)?;

    let sys_modules: Bound<'py, PyDict> = rust_timeseries
        .py()
        .import("sys")?
        .getattr("modules")?
        .downcast_into::<PyDict>()?;

    sys_modules.set_item("rust_timeseries.statistical_tests", &statistical_tests)?;
    Ok(())
}
