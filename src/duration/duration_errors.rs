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
    InvalidWeibullParam { param: f64, reason: &'static str },

    ///Innovations for Weibull distribution must fulfill the unit mean condition.
    InvalidUnitMeanWeibull { mean: f64 },

    /// Innovations with generalized Gamma distribution must have finite and strictly positive parameters.
    InvalidGenGammaParam { param: f64, reason: &'static str },

    /// Innovations for generalized Gamma distribution must fulfill the unit mean condition.
    InvalidUnitMeanGenGamma { mean: f64 },

    /// At least one model shape parameter must be finite and > 0.
    InvalidModelShape { param: f64 },

    // ---- Meta / options validation ----
    /// epsilon_floor must be finite and > 0.
    InvalidEpsilonFloor { value: f64 },

    /// Psi guards must be finite with 0 < min < max.
    InvalidPsiGuards {
        min: f64,
        max: f64,
        reason: &'static str,
    },

    /// Init::Fixed(v) must be finite and > 0.
    InvalidInitFixed { value: f64 },

    /// Init::FixedVector must have duration lags length equal to q.
    InvalidDurationLength { expected: usize, actual: usize },

    /// Init::FixedVector's duration lags must be positive and finite.
    InvalidDurationLags { index: usize, value: f64 },

    /// Init::FixedVector must have psi length equal to p.
    InvalidPsiLength { expected: usize, actual: usize },

    /// Init::FixedVector's psi values must be finite and > 0.
    InvalidPsiLags { index: usize, value: f64 },

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
            }
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
                write!(
                    f,
                    "Weibull parameter must be finite and > 0; got: {param}. {reason}"
                )
            }
            ACDError::InvalidGenGammaParam { param, reason } => {
                write!(
                    f,
                    "Generalized Gamma parameter must be finite and > 0; got: {param}. {reason}"
                )
            }
            ACDError::InvalidUnitMeanWeibull { mean } => {
                write!(
                    f,
                    "Weibull parameters must fulfill unit mean condition: {mean}."
                )
            }
            ACDError::InvalidUnitMeanGenGamma { mean } => {
                write!(
                    f,
                    "Generalized Gamma parameters must fulfill unit mean condition: {mean}."
                )
            }
            ACDError::InvalidModelShape { param } => {
                write!(
                    f,
                    "At least one model shape parameter must be finite and > 0; got: {param}"
                )
            }
            ACDError::InvalidEpsilonFloor { value } => {
                write!(f, "epsilon_floor must be finite and > 0; got: {value}")
            }
            ACDError::InvalidPsiGuards { min, max, reason } => {
                write!(
                    f,
                    "Psi guards must be finite with 0 < min ({min}) < max ({max}); {reason}"
                )
            }
            ACDError::InvalidInitFixed { value } => {
                write!(f, "Init::Fixed must be finite and > 0; got: {value}")
            }
            ACDError::InvalidDurationLength { expected, actual } => {
                write!(
                    f,
                    "Init::FixedVector must have duration length equal to q: expected {expected}, got {actual}"
                )
            }
            ACDError::InvalidDurationLags { index, value } => {
                write!(
                    f,
                    "Init::FixedVector's duration lags must be positive and finite; index {index} has value {value}"
                )
            }
            ACDError::InvalidPsiLength { expected, actual } => {
                write!(
                    f,
                    "Init::FixedVector must have psi length equal to p: expected {expected}, got {actual}"
                )
            }
            ACDError::InvalidPsiLags { index, value } => {
                write!(
                    f,
                    "Init::FixedVector's psi lag values must be finite and > 0; index {index} has value {value}"
                )
            }
            ACDError::NonFinitePsi { t, value } => {
                write!(
                    f,
                    "Recursion produced non-finite psi_t at index {t}: {value}"
                )
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
    StationarityViolated { coeff_sum: f64 },

    /// Theta length mismatch for ACDParams.
    ThetaLengthMismatch { expected: usize, actual: usize },

    /// Omega must be finite and > 0.
    InvalidOmega { value: f64 },

    /// Alpha length mismatch for ACDParams.
    AlphaLengthMismatch { expected: usize, actual: usize },

    /// Alpha coordinates need to be non-negative.
    InvalidAlpha { index: usize, value: f64 },

    /// Beta length mismatch for ACDParams.
    BetaLengthMismatch { expected: usize, actual: usize },

    /// Beta coordinates need to be non-negative.
    InvalidBeta { index: usize, value: f64 },

    /// Slack value must be non-negative.
    InvalidSlack { value: f64 },
}

impl std::error::Error for ParamError {}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::StationarityViolated { coeff_sum } => {
                write!(
                    f,
                    "Model not stationary: sum of alpha and beta is {} (>= 1 is not allowed)",
                    coeff_sum
                )
            }
            ParamError::ThetaLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "Theta length mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            ParamError::InvalidOmega { value } => {
                write!(f, "Omega must be finite and > 0, got {}", value)
            }
            ParamError::AlphaLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "Alpha length mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            ParamError::InvalidAlpha { index, value } => {
                write!(
                    f,
                    "Alpha coordinate at index {} must be non-negative and finite, got {}",
                    index, value
                )
            }
            ParamError::BetaLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "Beta length mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            ParamError::InvalidBeta { index, value } => {
                write!(
                    f,
                    "Beta coordinate at index {} must be non-negative and finite, got {}",
                    index, value
                )
            }
            ParamError::InvalidSlack { value } => {
                write!(
                    f,
                    "Slack value must be non-negative and finite, got {}",
                    value
                )
            }
        }
    }
}

impl std::convert::From<ParamError> for PyErr {
    fn from(err: ParamError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}
