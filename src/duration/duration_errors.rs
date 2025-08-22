//! Errors for ACD duration models (data validation, meta/options checks,
//! recursion invariants, and optimizer failures).
//!
//! This module defines a single error type, [`ACDError`], used across the
//! Python-facing API and the internal Rust core. 
//!
//! ## Conventions
//! - **Indices are 0-based** (match Rust/NumPy).
//! - Durations must be **strictly positive and finite**.
//! - `t0` is the number of initial observations used *only* to warm the
//!   ψ recursion; they are excluded from the likelihood.
//! - Optimizer/backend errors are normalized to
//!   [`ACDError::OptimizationFailed`] with a human-readable status string.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Crate-wide result alias for ACD operations.
pub type ACDResult<T> = Result<T, ACDError>;

/// Crate-wide result alias for parameter errors.
pub type ParamResult<T> = Result<T, ParamError>;

/// Unified error type for ACD modeling.
///
/// Variants cover input/data validation, meta/options checks,
/// recursion/structural invariants, and optimizer/diagnostic failures.
/// The error implements `Display`, `Error`, and converts to a Python
/// `ValueError` for PyO3 boundaries.
#[derive(Debug, Clone, PartialEq)]
pub enum ACDError {
    // ---- Input/data validation ----
    /// Series is empty.
    EmptySeries,

    /// A data point is NaN/±inf.
    NonFiniteData { index: usize, value: f64 },

    /// A data point is ≤ 0 (durations must be strictly positive).
    NonPositiveData { index: usize, value: f64 },

    /// Requested burn-in exceeds series length.
    T0OutOfRange { t0: usize, len: usize },

    // ---- Model innovations and shape ----

    /// Innovations with Weibull distribution must have finite and strictly positive parameters.
    InvalidWeibullParam {param: f64, reason: &'static str },

    ///Innovations for Weibull distribution must fulfill the unit mean condition.
    InvalidUnitMeanWeibull { mean: f64 },

    /// Innovations with generalized Gamma distribution must have finite and strictly positive parameters.
    InvalidGenGammaParam { param: f64, reason: &'static str },

    /// Innovations for generalized Gamma distribution must fulfill the unit mean condition.
    InvalidUnitMeanGenGamma { mean: f64 },

    /// At least one model shape parameter must be finite and > 0.
    InvalidModelShape {param: f64},

    // ---- Meta / options validation ----
    /// epsilon_floor must be finite and > 0.
    InvalidEpsilonFloor { value: f64 },

    /// Psi guards must be finite with 0 < min < max.
    InvalidPsiGuards { min: f64, max: f64, reason: &'static str },

    /// Init::Fixed(v) must be finite and > 0.
    InvalidInitFixed { value: f64 },

    // ---- Model/recursion invariants ----
    /// Recursion produced a non-finite ψ_t (after guards/clamps).
    NonFinitePsi { t: usize, value: f64 },

    // ---- Estimation / optimizer ----
    /// Optimizer failed; include a human-readable status/reason.
    OptimizationFailed { status: String },

    // ---- Diagnostics / residuals (optional features) ----
    /// Requested normalized residuals but innovation CDF/quantile not available.
    CdfNotAvailable,
}

impl std::error::Error for ACDError {}

impl std::fmt::Display for ACDError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ACDError::EmptySeries => {
                write!(f, "Input series is empty.")
            },
            ACDError::NonFiniteData { index, value } => {
                write!(f, "Data point at index {index} is non-finite: {value}")
            }
            ACDError::NonPositiveData { index, value } => {
                write!(f, "Data point at index {index} is non-positive: {value}")
            }
            ACDError::T0OutOfRange { t0, len } => {
                write!(f, "Burn-in t0 ({t0}) exceeds series length ({len}).")
            }
            ACDError::InvalidWeibullParam { param, reason } => {
                write!(f, "Weibull parameter must be finite and > 0; got: {param}. {reason}")
            }
            ACDError::InvalidGenGammaParam { param, reason } => {
                write!(f, "Generalized Gamma parameter must be finite and > 0; got: {param}. {reason}")
            }
            ACDError::InvalidUnitMeanWeibull { mean } => {
                write!(f, "Weibull parameters must fulfill unit mean condition: {mean}.")
            }
            ACDError::InvalidUnitMeanGenGamma {mean} => {
                write!(f, "Generalized Gamma parameters must fulfill unit mean condition: {mean}.")
            }
            ACDError::InvalidModelShape { param } => {
                write!(f, "At least one model shape parameter must be finite and > 0; got: {param}")
            }
            ACDError::InvalidEpsilonFloor { value } => {
                write!(f, "epsilon_floor must be finite and > 0; got: {value}")
            }
            ACDError::InvalidPsiGuards { min, max, reason } => {
                write!(f, "Psi guards must be finite with 0 < min ({min}) < max ({max}); {reason}")
            }
            ACDError::InvalidInitFixed { value } => {
                write!(f, "Init::Fixed must be finite and > 0; got: {value}")
            }
            ACDError::NonFinitePsi { t, value } => {
                write!(f, "Recursion produced non-finite psi_t at index {t}: {value}")
            }
            ACDError::OptimizationFailed { status } => {
                write!(f, "Optimizer failed with status: {status}")
            }
            ACDError::CdfNotAvailable => {
                write!(f, "CDF/quantile not available for normalized residuals.")
            }
        }
    }
}

impl std::convert::From<ACDError> for PyErr {
    fn from(err: ACDError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParamError {
    /// Model not stationary (sum of alpha and beta >= 1).
    StationarityViolated(ACDError),

    /// Theta length mismatch for ACDParams.
    ThetaLengthMismatch { expected: usize, actual: usize },
}

impl std::error::Error for ParamError {}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::StationarityViolated(err) => {
                write!(f, "Model parameters violate stationarity: {}", err)
            },
            ParamError::ThetaLengthMismatch { expected, actual } => {
                write!(f, "Theta length mismatch: expected {}, got {}", expected, actual)
            }
        }
    }
}
