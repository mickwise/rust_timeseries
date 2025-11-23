//! ACD error types — validation, recursion invariants, and optimizer failures.
//!
//! Purpose
//! -------
//! Define the core error types used throughout the ACD duration modeling
//! stack: [`ACDError`] for high-level model/estimation failures and
//! [`ParamError`] for parameter construction/validation issues. Both enums
//! implement `Display`/`Error` and convert cleanly into `PyErr` for the
//! Python-facing API.
//!
//! Key behaviors
//! -------------
//! - Provide a single crate-wide [`ACDResult`] alias for operations that may
//!   fail due to invalid data, options, recursion violations, or optimizer
//!   errors.
//! - Differentiate between parameter-level issues (`ParamError`) and
//!   model/runtime issues (`ACDError`), with explicit conversion paths to
//!   propagate errors across layers.
//! - Normalize backend distribution and optimizer failures into a small,
//!   well-documented set of domain-specific error variants.
//!
//! Invariants & assumptions
//! ------------------------
//! - Duration data are expected to be strictly positive and finite; violations
//!   are reported via [`ACDError::NonFiniteData`] or
//!   [`ACDError::NonPositiveData`].
//! - Model shapes `(p, q)` and parameter vectors θ are validated before use;
//!   mismatches and stationarity violations are surfaced as [`ParamError`]
//!   and, when lifted, as [`ACDError`].
//! - The burn-in index `t0` is treated as a pure likelihood windowing
//!   parameter: it controls which observations enter the log-likelihood but
//!   does not alter the ψ recursion itself.
//! - Errors originating from external crates (e.g. `statrs`, the optimizer)
//!   are mapped to a subset of [`ACDError`] variants; when no precise mapping
//!   exists, they are collapsed into [`ACDError::UnknownError`].
//!
//! Conventions
//! -----------
//! - Indices are 0-based and refer to Rust/NumPy-style array positions.
//! - Error messages are designed to be human-readable and are propagated as
//!   `ValueError` at PyO3 boundaries.
//! - `ACDResult<T>` and `ParamResult<T>` are the preferred return types for
//!   fallible operations.
//! - Where possible, the same logical condition has a single canonical error
//!   variant (e.g., all invalid ψ values for log-likelihood use
//!   [`ACDError::InvalidPsiLogLik`]).
//!
//! Downstream usage
//! ----------------
//! - The ACD core (`data`, `params`, `psi`, `workspace`, `models::acd`) uses
//!   [`ACDResult`] for any operation that may fail in a domain-specific way.
//! - Parameter builders and validators return [`ParamResult`], and callers may
//!   lift these into [`ACDError`] via `From<ParamError>` when needed.
//! - Python bindings rely on the `From<ACDError> for PyErr` and
//!   `From<ParamError> for PyErr` impls to surface errors as `ValueError`
//!   with the `Display` message.
//!
//! Testing notes
//! -------------
//! - Unit tests verify:
//!   - that specific invalid inputs produce the expected concrete variants,
//!   - that `Display` messages are stable enough for users/logs,
//!   - that `From<ParamError>` and `From<OptError>` map known variants and
//!     collapse unknown ones into [`ACDError::UnknownError`] as documented.
//! - Integration tests in higher-level modules exercise end-to-end
//!   flows (e.g., fitting or simulation) and assert that domain violations
//!   are reported via the appropriate `ACDError` variants.
use crate::optimization::errors::OptError;
use statrs::distribution::{ExpError, GammaError, WeibullError};

#[cfg(feature = "python-bindings")]
use pyo3::{exceptions::PyValueError, PyErr};

/// Crate-wide result alias for ACD operations that may produce [`ACDError`].
pub type ACDResult<T> = Result<T, ACDError>;

/// Result alias for parameter-construction/validation paths that may produce
/// [`ParamError`].
pub type ParamResult<T> = Result<T, ParamError>;

/// ACDError — unified error type for ACD modeling.
///
/// Purpose
/// -------
/// Represent all high-level failures that can occur when working with ACD
/// duration models: input validation, meta/options checks, recursion/structural
/// invariants, simulation issues, and optimizer/backend errors. This is the
/// primary error type used in the crate-wide [`ACDResult`] alias.
///
/// Variants
/// --------
/// - `EmptySeries`
///   Input duration series is empty.
/// - `NonFiniteData { index, value }`
///   Duration at `index` is NaN or ±∞.
/// - `NonPositiveData { index, value }`
///   Duration at `index` is ≤ 0 (violates positivity constraint).
/// - `T0OutOfRange { t0, len }`
///   Requested burn-in index `t0` exceeds the series length `len`.
/// - `InvalidWeibullParam { param, reason }`
///   Weibull parameter is invalid (e.g., non-finite, non-positive).
/// - `InvalidUnitMeanWeibull { mean }`
///   Weibull parameters do not satisfy the unit-mean constraint.
/// - `InvalidGenGammaParam { param, reason }`
///   Generalized Gamma parameter is invalid (e.g., non-finite, non-positive).
/// - `InvalidUnitMeanGenGamma { mean }`
///   Generalized Gamma parameters do not satisfy the unit-mean constraint.
/// - `InvalidModelShape { param, reason }`
///   Model shape `(p, q)` is invalid (e.g., both zero or inconsistent
///   with the sample size).
/// - `InvalidLogLikInput { value }`
///   Log-likelihood input duration is not strictly positive and finite.
/// - `InvalidPsiLogLik { value }`
///   ψ value used in the log-likelihood is not strictly positive and finite.
/// - `InvalidEpsilonFloor { value }`
///   `epsilon_floor` option is not finite and > 0.
/// - `InvalidPsiGuards { min, max, reason }`
///   ψ guard bounds are invalid (e.g., min ≥ max or non-finite).
/// - `InvalidInitFixed { value }`
///   Scalar initialization value is not strictly positive and finite.
/// - `InvalidDurationLength { expected, actual }`
///   Duration lag vector length mismatch in fixed initialization.
/// - `InvalidDurationLags { index, value }`
///   Duration lag entry is non-finite or non-positive.
/// - `InvalidPsiLength { expected, actual }`
///   ψ lag vector length mismatch in fixed initialization.
/// - `InvalidPsiLags { index, value }`
///   ψ lag entry is non-finite or non-positive.
/// - `NonFinitePsi { t, value }`
///   ψ recursion produced a non-finite value at time index `t`.
/// - `OptimizationFailed { status }`
///   Optimizer backend reported a failure with human-readable `status`.
/// - `ModelNotFitted`
///   Operation requiring fitted parameters was invoked before fitting.
/// - `InvalidExpParam`
///   Exponential distribution parameter is invalid (e.g., rate ≤ 0).
/// - `ScaleInvalid`
///   Weibull scale parameter is invalid (≤ 0).
/// - `ShapeInvalid`
///   Weibull shape parameter is invalid (≤ 0).
/// - `AlphaLengthMismatch { expected, actual }`
///   α vector length mismatch at the model level.
/// - `BetaLengthMismatch { expected, actual }`
///   β vector length mismatch at the model level.
/// - `HessianDimMismatch { expected, found }`
///   Hessian matrix dimensions do not match the parameter dimension.
/// - `InvalidHessian { row, col, value }`
///   Hessian entry at `(row, col)` is non-finite.
/// - `ZeroSimulationHorizon`
///   Simulation horizon is zero; at least one step is required.
/// - `InsufficientPsiLength { required, provided }`
///   ψ initialization length is shorter than the required lag order `p`.
/// - `UnknownError`
///   Fallback for errors that cannot be mapped more precisely, including
///   some external library failures and unmapped `ParamError`/`OptError`
///   variants.
///
/// Invariants
/// ----------
/// - Each logical failure mode in the ACD core is mapped to a single, stable
///   `ACDError` variant where possible.
/// - Conversions from `statrs` and optimizer errors never panic; when a
///   precise mapping is unavailable, they produce [`ACDError::UnknownError`].
/// - Conversions from [`ParamError`] map known variants explicitly and
///   collapse all others into [`ACDError::UnknownError`].
///
/// Notes
/// -----
/// - At the Rust↔Python boundary, `ACDError` is converted into a `ValueError`
///   with its `Display` message through `From<ACDError> for PyErr`.
#[derive(Debug, Clone, PartialEq)]
pub enum ACDError {
    // ---- Input/data validation ----
    EmptySeries,
    NonFiniteData {
        index: usize,
        value: f64,
    },
    NonPositiveData {
        index: usize,
        value: f64,
    },
    T0OutOfRange {
        t0: usize,
        len: usize,
    },

    // ---- Model innovations and shape ----
    InvalidWeibullParam {
        param: f64,
        reason: &'static str,
    },
    InvalidUnitMeanWeibull {
        mean: f64,
    },
    InvalidGenGammaParam {
        param: f64,
        reason: &'static str,
    },
    InvalidUnitMeanGenGamma {
        mean: f64,
    },
    InvalidModelShape {
        param: usize,
        reason: &'static str,
    },
    InvalidLogLikInput {
        value: f64,
    },
    InvalidPsiLogLik {
        value: f64,
    },

    // ---- Meta / options validation ----
    InvalidEpsilonFloor {
        value: f64,
    },
    InvalidPsiGuards {
        min: f64,
        max: f64,
        reason: &'static str,
    },
    InvalidInitFixed {
        value: f64,
    },
    InvalidDurationLength {
        expected: usize,
        actual: usize,
    },
    InvalidDurationLags {
        index: usize,
        value: f64,
    },
    InvalidPsiLength {
        expected: usize,
        actual: usize,
    },
    InvalidPsiLags {
        index: usize,
        value: f64,
    },

    // ---- Model/recursion invariants ----
    NonFinitePsi {
        t: usize,
        value: f64,
    },

    // ---- Estimation / optimizer ----
    OptimizationFailed {
        status: String,
    },
    ModelNotFitted,

    // ---- statsrs distribution errors ----
    InvalidExpParam,
    ScaleInvalid,
    ShapeInvalid,

    // ---- ParamError ----
    AlphaLengthMismatch {
        expected: usize,
        actual: usize,
    },
    BetaLengthMismatch {
        expected: usize,
        actual: usize,
    },

    // ---- OptError ----
    HessianDimMismatch {
        expected: usize,
        found: (usize, usize),
    },
    InvalidHessian {
        row: usize,
        col: usize,
        value: f64,
    },

    // ---- Simulation errors ----
    ZeroSimulationHorizon,
    InsufficientPsiLength {
        required: usize,
        provided: usize,
    },

    /// ---- Fallback ----
    UnknownError,
}

impl std::error::Error for ACDError {}

impl std::fmt::Display for ACDError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // ---- Input/data validation ----
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
            // ---- Model innovations and shape ----
            ACDError::InvalidWeibullParam { param, reason } => {
                write!(f, "Weibull parameter must be finite and > 0; got: {param}. {reason}")
            }
            ACDError::InvalidGenGammaParam { param, reason } => {
                write!(
                    f,
                    "Generalized Gamma parameter must be finite and > 0; got: {param}. {reason}"
                )
            }
            ACDError::InvalidUnitMeanWeibull { mean } => {
                write!(f, "Weibull parameters must fulfill unit mean condition: {mean}.")
            }
            ACDError::InvalidUnitMeanGenGamma { mean } => {
                write!(f, "Generalized Gamma parameters must fulfill unit mean condition: {mean}.")
            }
            ACDError::InvalidModelShape { param, reason } => {
                write!(
                    f,
                    "At least one model shape parameter must be positive; got: {param}. {reason}"
                )
            }
            ACDError::InvalidLogLikInput { value } => {
                write!(f, "Log-likelihood input must be strictly positive and finite; got: {value}")
            }
            ACDError::InvalidPsiLogLik { value } => {
                write!(
                    f,
                    "Psi value for log-likelihood must be strictly positive and finite; got: {value}"
                )
            }
            // ---- Meta / options validation ----
            ACDError::InvalidEpsilonFloor { value } => {
                write!(f, "epsilon_floor must be finite and > 0; got: {value}")
            }
            ACDError::InvalidPsiGuards { min, max, reason } => {
                write!(f, "Psi guards must be finite with 0 < min ({min}) < max ({max}); {reason}")
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
            // ---- Model/recursion invariants ----
            ACDError::NonFinitePsi { t, value } => {
                write!(f, "Recursion produced non-finite psi_t at index {t}: {value}")
            }
            // ---- Estimation / optimizer ----
            ACDError::OptimizationFailed { status } => {
                write!(f, "Optimizer failed with status: {status}")
            }
            ACDError::ModelNotFitted => {
                write!(f, "Model hasn't been fitted yet.")
            }
            // ---- statsrs distribution errors ----
            ACDError::InvalidExpParam => {
                write!(f, "Exponential distribution requires rate > 0.")
            }
            ACDError::ScaleInvalid => {
                write!(f, "Weibull distribution scale parameter must be > 0.")
            }
            ACDError::ShapeInvalid => {
                write!(f, "Weibull distribution shape parameter must be > 0.")
            }
            ACDError::UnknownError => {
                write!(f, "An unknown error occurred.")
            }

            // ---- ParamError ----
            ACDError::AlphaLengthMismatch { expected, actual } => {
                write!(f, "Alpha length mismatch: expected {expected}, got {actual}")
            }
            ACDError::BetaLengthMismatch { expected, actual } => {
                write!(f, "Beta length mismatch: expected {expected}, got {actual}")
            }

            // ---- OptError ----
            ACDError::HessianDimMismatch { expected, found } => {
                write!(
                    f,
                    "Hessian dimension mismatch: expected {}x{}, found {}x{}",
                    expected, expected, found.0, found.1
                )
            }
            ACDError::InvalidHessian { row, col, value } => {
                write!(f, "Invalid Hessian value at ({}, {}): {}", row, col, value)
            }

            // ---- Simulation errors ----
            ACDError::ZeroSimulationHorizon => {
                write!(f, "Simulation horizon must be > 0.")
            }
            ACDError::InsufficientPsiLength { required, provided } => {
                write!(
                    f,
                    "Psi length for simulation must be at least p: required {required}, got {provided}"
                )
            }
        }
    }
}

/// Convert an [`ACDError`] into a Python `ValueError` with the error message.
///
/// This is used at the Rust↔Python boundary to surface domain errors cleanly.
#[cfg(feature = "python-bindings")]
impl From<ACDError> for PyErr {
    fn from(err: ACDError) -> PyErr {
        PyValueError::new_err(format!("ACDError: {err:?}"))
    }
}

impl From<ExpError> for ACDError {
    fn from(_: ExpError) -> ACDError {
        ACDError::InvalidExpParam
    }
}

impl From<WeibullError> for ACDError {
    fn from(err: WeibullError) -> ACDError {
        match err {
            WeibullError::ScaleInvalid => ACDError::ScaleInvalid,
            WeibullError::ShapeInvalid => ACDError::ShapeInvalid,
            _ => ACDError::UnknownError,
        }
    }
}

impl From<GammaError> for ACDError {
    fn from(err: GammaError) -> ACDError {
        match err {
            GammaError::ShapeInvalid => ACDError::InvalidGenGammaParam {
                param: 0.0,
                reason: "Shape parameter must be > 0.",
            },
            _ => ACDError::UnknownError,
        }
    }
}

impl From<ParamError> for ACDError {
    fn from(err: ParamError) -> ACDError {
        match err {
            ParamError::AlphaLengthMismatch { expected, actual } => {
                ACDError::AlphaLengthMismatch { expected, actual }
            }
            ParamError::BetaLengthMismatch { expected, actual } => {
                ACDError::BetaLengthMismatch { expected, actual }
            }
            _ => ACDError::UnknownError,
        }
    }
}

impl From<OptError> for ACDError {
    fn from(err: OptError) -> Self {
        match err {
            OptError::HessianDimMismatch { expected, found } => {
                ACDError::HessianDimMismatch { expected, found }
            }
            OptError::InvalidHessian { row, col, value } => {
                ACDError::InvalidHessian { row, col, value }
            }
            _ => ACDError::UnknownError,
        }
    }
}

/// ParamError — parameter construction and validation failures.
///
/// Purpose
/// -------
/// Capture errors that arise when constructing or validating ACD parameters
/// (ω, α, β, slack, θ, ψ lags) before they are used in recursion or
/// optimization. This type is used in [`ParamResult`] and can be lifted into
/// [`ACDError`] where a unified error channel is required.
///
/// Variants
/// --------
/// - `StationarityViolated { coeff_sum }`
///   Sum of α and β coefficients is ≥ 1, violating the stationarity
///   condition.
/// - `ThetaLengthMismatch { expected, actual }`
///   Unconstrained θ vector length does not match `(1 + p + q)`.
/// - `InvalidOmega { value }`
///   ω (baseline intensity) is not strictly positive and finite.
/// - `AlphaLengthMismatch { expected, actual }`
///   α vector length mismatch at the parameter-construction layer.
/// - `InvalidAlpha { index, value }`
///   α coordinate at `index` is negative or non-finite.
/// - `BetaLengthMismatch { expected, actual }`
///   β vector length mismatch at the parameter-construction layer.
/// - `InvalidBeta { index, value }`
///   β coordinate at `index` is negative or non-finite.
/// - `InvalidSlack { value }`
///   Slack parameter is negative or non-finite.
/// - `InvalidThetaInput { index, value }`
///   Unconstrained θ input at `index` is non-finite.
/// - `InvalidPsiLength { expected, actual }`
///   ψ lag vector length mismatch (e.g., for `ACDParams` initialization).
/// - `InvalidPsiLags { index, value }`
///   ψ lag at `index` is not strictly positive and finite.
/// - `UnknownError`
///   Fallback for errors that cannot be mapped more precisely, including
///   unmapped [`ACDError`] variants in the `From<ACDError>` conversion.
///
/// Invariants
/// ----------
/// - All parameter validation routines in the ACD core should return a
///   specific `ParamError` variant for each failure mode they check.
/// - Conversions from [`ACDError`] into `ParamError` are intentionally
///   narrow: only `InvalidPsiLength` and `InvalidPsiLags` are mapped, all
///   other variants collapse into [`ParamError::UnknownError`].
///
/// Notes
/// -----
/// - When building user-facing APIs, it is usually more ergonomic to expose
///   [`ACDError`] and use `From<ParamError> for ACDError` to lift parameter
///   errors into the main error channel.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamError {
    StationarityViolated {
        coeff_sum: f64,
    },
    ThetaLengthMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidOmega {
        value: f64,
    },
    AlphaLengthMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidAlpha {
        index: usize,
        value: f64,
    },
    BetaLengthMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidBeta {
        index: usize,
        value: f64,
    },
    InvalidSlack {
        value: f64,
    },
    InvalidThetaInput {
        index: usize,
        value: f64,
    },

    // ---- ACDError ----
    InvalidPsiLength {
        expected: usize,
        actual: usize,
    },
    InvalidPsiLags {
        index: usize,
        value: f64,
    },

    /// ---- Fallback ----
    UnknownError,
}

impl std::error::Error for ParamError {}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::StationarityViolated { coeff_sum } => {
                write!(
                    f,
                    "Model not stationary: sum of alpha and beta is {coeff_sum} (>= 1 is not allowed)",
                )
            }
            ParamError::ThetaLengthMismatch { expected, actual } => {
                write!(f, "Theta length mismatch: expected {expected}, got {actual}")
            }
            ParamError::InvalidOmega { value } => {
                write!(f, "Omega must be finite and > 0, got {value}")
            }
            ParamError::AlphaLengthMismatch { expected, actual } => {
                write!(f, "Alpha length mismatch: expected {expected}, got {actual}")
            }
            ParamError::InvalidAlpha { index, value } => {
                write!(
                    f,
                    "Alpha coordinate at index {index} must be non-negative and finite, got {value}"
                )
            }
            ParamError::BetaLengthMismatch { expected, actual } => {
                write!(f, "Beta length mismatch: expected {expected}, got {actual}")
            }
            ParamError::InvalidBeta { index, value } => {
                write!(
                    f,
                    "Beta coordinate at index {index} must be non-negative and finite, got {value}",
                )
            }
            ParamError::InvalidSlack { value } => {
                write!(f, "Slack value must be non-negative and finite, got {value}")
            }
            ParamError::InvalidThetaInput { index, value } => {
                write!(f, "Theta input at index {index} must be finite, got {value}")
            }
            ParamError::InvalidPsiLength { expected, actual } => {
                write!(f, "Psi length mismatch: expected {expected}, got {actual}")
            }
            ParamError::InvalidPsiLags { index, value } => {
                write!(f, "Psi lag at index {index} must be finite and > 0, got {value}")
            }
            ParamError::UnknownError => {
                write!(f, "An unknown error occurred in parameter validation.")
            }
        }
    }
}

/// Convert a [`ParamError`] into a Python `ValueError` with the error message.
#[cfg(feature = "python-bindings")]
impl std::convert::From<ParamError> for PyErr {
    fn from(err: ParamError) -> PyErr {
        PyValueError::new_err(format!("PARAMError: {err:?}"))
    }
}

impl From<ACDError> for ParamError {
    fn from(err: ACDError) -> ParamError {
        match err {
            ACDError::InvalidPsiLength { expected, actual } => {
                ParamError::InvalidPsiLength { expected, actual }
            }
            ACDError::InvalidPsiLags { index, value } => {
                ParamError::InvalidPsiLags { index, value }
            }
            _ => Self::UnknownError,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::errors::OptError;
    use statrs::distribution::{GammaError, WeibullError};

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Basic `Display` formatting for key `ACDError` variants.
    // - Mappings from `ParamError`, `OptError`, and `statrs` errors into
    //   `ACDError`.
    //
    // They intentionally DO NOT cover:
    // - PyO3 / `PyErr` behavior at the Rust↔Python boundary.
    // - End-to-end flows that trigger these errors from higher-level
    //   modules (covered by integration / Python tests).
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // `Display` for `ACDError::NonPositiveData` includes both the index and
    // offending value in a human-readable message.
    //
    // Given
    // -----
    // - An `ACDError::NonPositiveData { index, value }` instance.
    //
    // Expect
    // ------
    // - The formatted string mentions the index and the value.
    fn display_non_positive_data_includes_index_and_value() {
        // Arrange
        let err = ACDError::NonPositiveData { index: 7, value: -1.5 };

        // Act
        let msg = err.to_string();

        // Assert
        assert!(msg.contains("index 7"), "message missing index: {msg}");
        assert!(msg.contains("-1.5"), "message missing value; got: {msg}");
    }

    #[test]
    // Purpose
    // -------
    // `From<ParamError>` for `ACDError` maps `AlphaLengthMismatch` into the
    // corresponding `ACDError::AlphaLengthMismatch` variant.
    //
    // Given
    // -----
    // - A `ParamError::AlphaLengthMismatch { expected, actual }` instance.
    //
    // Expect
    // ------
    // - `ACDError::from(param_err)` yields `ACDError::AlphaLengthMismatch`
    //   with the same `expected` and `actual` values.
    fn from_paramerror_maps_alpha_length_mismatch() {
        // Arrange
        let param_err = ParamError::AlphaLengthMismatch { expected: 3, actual: 1 };

        // Act
        let acd_err = ACDError::from(param_err);

        // Assert
        match acd_err {
            ACDError::AlphaLengthMismatch { expected, actual } => {
                assert_eq!(expected, 3);
                assert_eq!(actual, 1);
            }
            other => panic!("unexpected ACDError variant: {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // `From<OptError>` for `ACDError` maps `HessianDimMismatch` into the
    // corresponding `ACDError::HessianDimMismatch` variant.
    //
    // Given
    // -----
    // - An `OptError::HessianDimMismatch { expected, found }` instance.
    //
    // Expect
    // ------
    // - `ACDError::from(opt_err)` yields `ACDError::HessianDimMismatch`
    //   with the same `expected` and `found` fields.
    fn from_opterror_maps_hessian_dim_mismatch() {
        // Arrange
        let opt_err = OptError::HessianDimMismatch { expected: 4, found: (3, 5) };

        // Act
        let acd_err = ACDError::from(opt_err);

        // Assert
        match acd_err {
            ACDError::HessianDimMismatch { expected, found } => {
                assert_eq!(expected, 4);
                assert_eq!(found, (3, 5));
            }
            other => panic!("unexpected ACDError variant: {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // `From<WeibullError>` for `ACDError` maps `ScaleInvalid` into
    // `ACDError::ScaleInvalid`.
    //
    // Given
    // -----
    // - A `WeibullError::ScaleInvalid` instance from `statrs`.
    //
    // Expect
    // ------
    // - `ACDError::from(err)` yields `ACDError::ScaleInvalid`.
    fn from_weibull_error_maps_scale_invalid() {
        // Arrange
        let weibull_err = WeibullError::ScaleInvalid;

        // Act
        let acd_err = ACDError::from(weibull_err);

        // Assert
        assert!(
            matches!(acd_err, ACDError::ScaleInvalid),
            "unexpected ACDError variant: {acd_err:?}"
        );
    }

    #[test]
    // Purpose
    // -------
    // `From<GammaError>` for `ACDError` maps `ShapeInvalid` into
    // `ACDError::InvalidGenGammaParam` with the documented reason string.
    //
    // Given
    // -----
    // - A `GammaError::ShapeInvalid` instance from `statrs`.
    //
    // Expect
    // ------
    // - `ACDError::from(err)` yields `ACDError::InvalidGenGammaParam` whose
    //   `reason` matches the hard-coded message in the conversion.
    fn from_gamma_error_maps_shape_invalid() {
        // Arrange
        let gamma_err = GammaError::ShapeInvalid;

        // Act
        let acd_err = ACDError::from(gamma_err);

        // Assert
        match acd_err {
            ACDError::InvalidGenGammaParam { reason, .. } => {
                assert_eq!(reason, "Shape parameter must be > 0.");
            }
            other => panic!("unexpected ACDError variant: {other:?}"),
        }
    }
}
