//! optimization::errors — unified optimizer error type and conversions.
//!
//! Purpose
//! -------
//! Provide a crate-wide error and result type for optimization-related
//! operations, normalizing errors from Argmin, finite-difference checks,
//! log-likelihood evaluation, and ACD/parameter validation into a single
//! enum [`OptError`].
//!
//! Key behaviors
//! -------------
//! - Define [`OptError`] as the canonical error type for the optimization
//!   layer and expose [`OptResult`] as a convenience alias.
//! - Map upstream errors from `argmin`, duration log-likelihood evaluation
//!   (`ACDError`), and parameter validation (`ParamError`) into a stable
//!   set of variants.
//! - Provide human-readable `Display` messages for all variants to make
//!   diagnostics and logging easier to interpret.
//!
//! Invariants & assumptions
//! ------------------------
//! - All optimizer-facing APIs that can fail return [`OptResult<T>`] so
//!   callers see a consistent error surface regardless of the underlying
//!   source.
//! - `UnknownError` is reserved as a conservative fallback when an upstream
//!   error cannot be mapped to a more specific variant.
//! - Conversions from `ACDError` and `ParamError` only cover variants that
//!   are meaningful in the optimization layer; all others are treated as
//!   `UnknownError`.
//!
//! Conventions
//! -----------
//! - The `Display` implementation is intended for user-facing messages
//!   (e.g., Python bindings, logs). Tests should not depend on its exact
//!   wording unless they intentionally pin it down.
//! - `From<Error>` is used to collapse `argmin::core::Error` into
//!   [`OptError`], preserving recognizable Argmin variants where possible.
//! - Gradient- and Hessian-related variants express dimensions in terms of
//!   parameter length and matrix shape to aid debugging.
//!
//! Downstream usage
//! ----------------
//! - Optimization helpers and MLE routines return [`OptResult<T>`] instead
//!   of embedding Argmin or model-specific error types directly.
//! - Higher-level code (e.g., duration models, Python wrappers) can match
//!   on [`OptError`] to distinguish between configuration/validation
//!   issues, numerical problems, and backend failures.
//! - Logging layers can rely on `Display` to emit concise diagnostic
//!   messages without needing to inspect all variants explicitly.
//!
//! Testing notes
//! -------------
//! - Unit tests in this module verify that:
//!   - each variant’s `Display` string includes key fields (indices,
//!     dimensions, values), and
//!   - conversions from `ACDError`, `ParamError`, and `argmin::core::Error`
//!     select the expected [`OptError`] variant.
//! - Integration tests in the optimizer layer ensure that error paths
//!   surfaced by MLE routines and finite-difference utilities round-trip
//!   through these conversions correctly.
use argmin::core::{ArgminError, Error};

use crate::duration::errors::{ACDError, ParamError};

#[cfg(feature = "python-bindings")]
use pyo3::{PyErr, exceptions::PyValueError};

/// Crate-wide result alias for optimizer operations.
pub type OptResult<T> = Result<T, OptError>;

/// `OptError` — unified error enum for the optimization layer.
///
/// Purpose
/// -------
/// Represent all errors that can occur during configuration, validation,
/// and execution of optimization routines, including Argmin backend errors,
/// gradient/Hessian validation failures, and duration model/parameter
/// validation issues.
///
/// Variants
/// --------
/// Gradient-related:
/// - `GradientNotImplemented`
///   Analytic gradient is unavailable; callers should fall back to finite
///   differences.
/// - `GradientDimMismatch { expected, found }`
///   Gradient length does not match the parameter dimension.
/// - `InvalidGradient { index, value, reason }`
///   Gradient entry is non-finite or otherwise invalid, with a short reason.
///
/// MLE options and configuration:
/// - `InvalidTolGrad { tol, reason }`
///   Gradient tolerance is non-positive or non-finite.
/// - `InvalidTolCost { tol, reason }`
///   Cost change tolerance is non-positive or non-finite.
/// - `InvalidTolF { tol, reason }`
///   Function tolerance is non-positive or non-finite.
/// - `InvalidMaxIter { max_iter, reason }`
///   Maximum iterations is zero or otherwise invalid.
/// - `NoTolerancesProvided`
///   All stopping tolerances were `None`, so the optimizer would not know
///   when to stop.
/// - `InvalidLineSearch { name, reason }`
///   Line searcher name could not be parsed or is unsupported.
/// - `InvalidLBFGSMem { mem, reason }`
///   L-BFGS memory parameter is less than 1 or otherwise invalid.
///
/// Cost function:
/// - `NonFiniteCost { value }`
///   Log-likelihood or cost evaluation returned NaN or ±∞.
///
/// Optimizer outcome:
/// - `InvalidThetaHat { index, value, reason }`
///   Final parameter estimate contains an invalid entry.
/// - `MissingThetaHat`
///   Optimizer terminated without producing a parameter estimate.
///
/// Argmin backend wrappers:
/// - `InvalidParameter { text }`
///   Wraps `ArgminError::InvalidParameter`.
/// - `NotImplemented { text }`
///   Wraps `ArgminError::NotImplemented`.
/// - `NotInitialized { text }`
///   Wraps `ArgminError::NotInitialized`.
/// - `ConditionViolated { text }`
///   Wraps `ArgminError::ConditionViolated`.
/// - `CheckPointNotFound { text }`
///   Wraps `ArgminError::CheckpointNotFound`.
/// - `PotentialBug { text }`
///   Wraps `ArgminError::PotentialBug`.
/// - `ImpossibleError { text }`
///   Wraps `ArgminError::ImpossibleError`.
/// - `BackendError { text }`
///   Any other Argmin error that does not map cleanly to a dedicated
///   variant; `text` contains the formatted backend error.
///
/// Finite-difference / Hessian validation:
/// - `HessianDimMismatch { expected, found }`
///   Hessian shape does not match the parameter dimension.
/// - `InvalidHessian { row, col, value }`
///   Hessian entry is non-finite.
///
/// ACD log-likelihood and distribution parameters:
/// - `InvalidLogLikInput { value }`
///   Log-likelihood input (e.g., duration) is invalid.
/// - `InvalidPsiLogLik { value }`
///   Conditional mean `ψ` passed into log-likelihood is invalid.
/// - `InvalidExpParam`
///   Exponential parameter violates unit-mean or positivity constraints.
/// - `ScaleInvalid`
///   Scale parameter is non-positive or non-finite.
/// - `ShapeInvalid`
///   Shape parameter is non-positive or non-finite.
///
/// ACD parameter validation:
/// - `StationarityViolated { coeff_sum }`
///   Sum of ACD coefficients (α + β) is ≥ 1, violating stationarity.
/// - `ThetaLengthMismatch { expected, actual }`
///   Unconstrained θ length mismatch for ACD parameters.
/// - `InvalidOmega { value }`
///   Baseline `ω` is non-positive or non-finite.
/// - `AlphaLengthMismatch { expected, actual }`
///   α vector length does not match the ACD order `q`.
/// - `InvalidAlpha { index, value }`
///   α entry is negative or non-finite.
/// - `BetaLengthMismatch { expected, actual }`
///   β vector length does not match the ACD order `p`.
/// - `InvalidBeta { index, value }`
///   β entry is negative or non-finite.
/// - `InvalidSlack { value }`
///   Slack mass is negative or non-finite.
/// - `InvalidThetaInput { index, value }`
///   Unconstrained θ entry is non-finite.
///
/// Fallback:
/// - `UnknownError`
///   Catch-all for upstream errors that cannot be mapped more precisely.
///
/// Invariants
/// ----------
/// - Variants that carry indices, dimensions, or parameter values are meant
///   to provide enough context for debugging without requiring access to the
///   original data structures.
/// - Mapping from `ACDError` and `ParamError` to [`OptError`] preserves
///   semantic meaning where possible; the remainder falls back to
///   `UnknownError`.
///
/// Notes
/// -----
/// - This enum is `Clone` and `PartialEq` to make it easy to propagate and
///   assert against in tests; payloads are kept small (scalars and short
///   strings) to avoid unnecessary allocations.
#[derive(Debug, Clone, PartialEq)]
pub enum OptError {
    // ---- Gradient ----
    GradientNotImplemented,
    GradientDimMismatch { expected: usize, found: usize },
    InvalidGradient { index: usize, value: f64, reason: &'static str },

    // ---- MLEOptions ----
    InvalidTolGrad { tol: f64, reason: &'static str },
    InvalidTolCost { tol: f64, reason: &'static str },
    InvalidTolF { tol: f64, reason: &'static str },
    InvalidMaxIter { max_iter: usize, reason: &'static str },
    NoTolerancesProvided,
    InvalidLineSearch { name: String, reason: &'static str },
    InvalidLBFGSMem { mem: usize, reason: &'static str },

    // ---- Cost function ----
    NonFiniteCost { value: f64 },

    // ---- Optimizer outcome ----
    InvalidThetaHat { index: usize, value: f64, reason: &'static str },
    MissingThetaHat,

    // ---- Argmin ---
    InvalidParameter { text: String },
    NotImplemented { text: String },
    NotInitialized { text: String },
    ConditionViolated { text: String },
    CheckPointNotFound { text: String },
    PotentialBug { text: String },
    ImpossibleError { text: String },
    BackendError { text: String },

    // ---- Finite Diffs ----
    HessianDimMismatch { expected: usize, found: (usize, usize) },
    InvalidHessian { row: usize, col: usize, value: f64 },

    // ---- ACD Errors ----
    InvalidLogLikInput { value: f64 },
    InvalidPsiLogLik { value: f64 },
    InvalidExpParam,
    ScaleInvalid,
    ShapeInvalid,

    // ---- Param Errors ----
    StationarityViolated { coeff_sum: f64 },
    ThetaLengthMismatch { expected: usize, actual: usize },
    InvalidOmega { value: f64 },
    AlphaLengthMismatch { expected: usize, actual: usize },
    InvalidAlpha { index: usize, value: f64 },
    BetaLengthMismatch { expected: usize, actual: usize },
    InvalidBeta { index: usize, value: f64 },
    InvalidSlack { value: f64 },
    InvalidThetaInput { index: usize, value: f64 },

    // ---- Fallback ----
    UnknownError { message: String },
}

impl std::error::Error for OptError {}

impl std::fmt::Display for OptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // ---- Gradient ----
            OptError::GradientNotImplemented => {
                write!(f, "Gradient optimization not implemented")
            }
            OptError::GradientDimMismatch { expected, found } => {
                write!(f, "Gradient dimension mismatch: expected {expected}, found {found}")
            }
            OptError::InvalidGradient { index, value, reason } => {
                write!(f, "Invalid gradient at index {index}: {value}: {reason}")
            }

            // ---- MLEOptions ----
            OptError::InvalidTolGrad { tol, reason } => {
                write!(f, "Invalid gradient tolerance {tol}: {reason}")
            }
            OptError::InvalidTolCost { tol, reason } => {
                write!(f, "Invalid cost function change tolerance {tol}: {reason}")
            }
            OptError::InvalidTolF { tol, reason } => {
                write!(f, "Invalid function tolerance {tol}: {reason}")
            }
            OptError::InvalidMaxIter { max_iter, reason } => {
                write!(f, "Invalid maximum iterations {max_iter}: {reason}")
            }
            OptError::NoTolerancesProvided => {
                write!(f, "No tolerances provided")
            }
            OptError::InvalidLineSearch { name, reason } => {
                write!(f, "Invalid line searcher '{name}': {reason}")
            }
            OptError::InvalidLBFGSMem { mem, reason } => {
                write!(f, "Invalid L-BFGS memory {mem}: {reason}")
            }

            // ---- Cost function ----
            OptError::NonFiniteCost { value } => {
                write!(f, "Non-finite cost value: {value}")
            }

            // ---- Optimizer outcome ----
            OptError::InvalidThetaHat { index, value, reason } => {
                write!(f, "Invalid estimated parameter at index {index}: {value}: {reason}")
            }
            OptError::MissingThetaHat => {
                write!(f, "Missing estimated parameters (theta hat)")
            }

            // ---- Argmin ----
            OptError::InvalidParameter { text } => {
                write!(f, "Invalid parameter: {text}")
            }
            OptError::NotImplemented { text } => {
                write!(f, "Not implemented: {text}")
            }
            OptError::NotInitialized { text } => {
                write!(f, "Not initialized: {text}")
            }
            OptError::ConditionViolated { text } => {
                write!(f, "Condition violated: {text}")
            }
            OptError::CheckPointNotFound { text } => {
                write!(f, "Checkpoint not found: {text}")
            }
            OptError::PotentialBug { text } => {
                write!(f, "Potential bug: {text}")
            }
            OptError::ImpossibleError { text } => {
                write!(f, "Impossible error: {text}")
            }
            OptError::BackendError { text } => {
                write!(f, "Backend error: {text}")
            }

            // ---- Finite Diffs ----
            OptError::HessianDimMismatch { expected, found } => {
                write!(
                    f,
                    "Hessian dimension mismatch: expected ({expected}, {expected}), found {found:?}"
                )
            }
            OptError::InvalidHessian { row, col, value } => {
                write!(f, "Invalid Hessian at ({row}, {col}): {value}, must be finite")
            }

            // ---- ACD Errors ----
            OptError::InvalidLogLikInput { value } => {
                write!(f, "Invalid input to log-likelihood function: {value}")
            }
            OptError::InvalidPsiLogLik { value } => {
                write!(f, "Invalid psi value in log-likelihood function: {value}")
            }
            OptError::InvalidExpParam => {
                write!(f, "Invalid exponential parameter")
            }
            OptError::ScaleInvalid => {
                write!(f, "Scale parameter is invalid (<= 0 or non-finite)")
            }
            OptError::ShapeInvalid => {
                write!(f, "Shape parameter is invalid (<= 0 or non-finite)")
            }

            // ---- Param Errors ----
            OptError::StationarityViolated { coeff_sum } => {
                write!(
                    f,
                    "Model not stationary: sum of alpha and beta is {coeff_sum}, which is >= 1"
                )
            }
            OptError::ThetaLengthMismatch { expected, actual } => {
                write!(f, "Theta length mismatch: expected {expected}, actual {actual}")
            }
            OptError::InvalidOmega { value } => {
                write!(f, "Invalid omega parameter: {value}, must be finite and > 0")
            }
            OptError::AlphaLengthMismatch { expected, actual } => {
                write!(f, "Alpha length mismatch: expected {expected}, actual {actual}")
            }
            OptError::InvalidAlpha { index, value } => {
                write!(f, "Invalid alpha at index {index}: {value}, must be non-negative")
            }
            OptError::BetaLengthMismatch { expected, actual } => {
                write!(f, "Beta length mismatch: expected {expected}, actual {actual}")
            }
            OptError::InvalidBeta { index, value } => {
                write!(f, "Invalid beta at index {index}: {value}, must be non-negative")
            }
            OptError::InvalidSlack { value } => {
                write!(f, "Invalid slack value: {value}, must be non-negative")
            }
            OptError::InvalidThetaInput { index, value } => {
                write!(f, "Invalid theta input at index {index}: {value}, must be finite")
            }

            // ---- Fallback ----
            OptError::UnknownError { message } => {
                write!(f, "Unknown OptError error: {message}")
            }
        }
    }
}

#[cfg(feature = "python-bindings")]
impl From<OptError> for PyErr {
    fn from(err: OptError) -> PyErr {
        PyValueError::new_err(format!("OPTError: {err:?}"))
    }
}

impl From<Error> for OptError {
    fn from(original_err: Error) -> Self {
        match original_err.downcast() {
            Ok(argmin_error) => match argmin_error {
                ArgminError::InvalidParameter { text } => OptError::InvalidParameter { text },
                ArgminError::NotImplemented { text } => OptError::NotImplemented { text },
                ArgminError::NotInitialized { text } => OptError::NotInitialized { text },
                ArgminError::ConditionViolated { text } => OptError::ConditionViolated { text },
                ArgminError::CheckpointNotFound { text } => OptError::CheckPointNotFound { text },
                ArgminError::PotentialBug { text } => OptError::PotentialBug { text },
                ArgminError::ImpossibleError { text } => OptError::ImpossibleError { text },
                _ => OptError::UnknownError { message: format!("{argmin_error}") },
            },
            Err(err) => OptError::BackendError { text: err.to_string() },
        }
    }
}

impl From<ACDError> for OptError {
    fn from(err: ACDError) -> Self {
        match err {
            ACDError::InvalidLogLikInput { value } => OptError::InvalidLogLikInput { value },
            ACDError::InvalidPsiLogLik { value } => OptError::InvalidPsiLogLik { value },
            ACDError::InvalidExpParam => OptError::InvalidExpParam,
            ACDError::ScaleInvalid => OptError::ScaleInvalid,
            ACDError::ShapeInvalid => OptError::ShapeInvalid,
            _ => OptError::UnknownError { message: format!("{err}") },
        }
    }
}

impl From<ParamError> for OptError {
    fn from(err: ParamError) -> Self {
        match err {
            ParamError::StationarityViolated { coeff_sum } => {
                OptError::StationarityViolated { coeff_sum }
            }
            ParamError::ThetaLengthMismatch { expected, actual } => {
                OptError::ThetaLengthMismatch { expected, actual }
            }
            ParamError::InvalidOmega { value } => OptError::InvalidOmega { value },
            ParamError::AlphaLengthMismatch { expected, actual } => {
                OptError::AlphaLengthMismatch { expected, actual }
            }
            ParamError::InvalidAlpha { index, value } => OptError::InvalidAlpha { index, value },
            ParamError::BetaLengthMismatch { expected, actual } => {
                OptError::BetaLengthMismatch { expected, actual }
            }
            ParamError::InvalidBeta { index, value } => OptError::InvalidBeta { index, value },
            ParamError::InvalidSlack { value } => OptError::InvalidSlack { value },
            ParamError::InvalidThetaInput { index, value } => {
                OptError::InvalidThetaInput { index, value }
            }
            _ => OptError::UnknownError { message: format!("{err}") },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duration::errors::{ACDError, ParamError};
    use argmin::core::{ArgminError, Error};

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Conversions from `argmin::core::Error` into `OptError` for the
    //   Argmin variants that this module explicitly maps.
    // - Conversions from `ACDError` and `ParamError` into the corresponding
    //   `OptError` variants used by the optimization layer.
    // - Basic `Display` formatting invariants for selected variants to verify
    //   that key fields (indices, dimensions, values) appear in messages.
    //
    // They intentionally DO NOT cover:
    // - End-to-end MLE or optimizer flows; those are better exercised via
    //   integration tests in the optimizer / duration modules.
    // - Exhaustive checking of every `OptError` variant’s exact `Display`
    //   wording, which is considered an implementation detail and may evolve.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Verify that known `ArgminError` variants are mapped to the corresponding
    // `OptError` variants with the original text preserved.
    //
    // Given
    // -----
    // - `ArgminError::InvalidParameter` and `ArgminError::NotImplemented`
    //   wrapped into `argmin::core::Error`.
    //
    // Expect
    // ------
    // - `OptError::from(Error)` yields `OptError::InvalidParameter` and
    //   `OptError::NotImplemented` with identical `text` payloads.
    fn from_argmin_error_maps_known_variants_to_opt_error() {
        // Arrange
        let text_invalid = "invalid parameter".to_string();
        let text_not_impl = "not implemented".to_string();

        let err_invalid: Error =
            ArgminError::InvalidParameter { text: text_invalid.clone() }.into();
        let err_not_impl: Error =
            ArgminError::NotImplemented { text: text_not_impl.clone() }.into();

        // Act
        let opt_invalid: OptError = err_invalid.into();
        let opt_not_impl: OptError = err_not_impl.into();

        // Assert
        assert_eq!(opt_invalid, OptError::InvalidParameter { text: text_invalid });
        assert_eq!(opt_not_impl, OptError::NotImplemented { text: text_not_impl });
    }

    #[test]
    // Purpose
    // -------
    // Ensure that non-Argmin backend errors wrapped in `Error` are surfaced
    // as `OptError::BackendError`.
    //
    // Given
    // -----
    // - A generic `std::io::Error` converted into `argmin::core::Error`.
    //
    // Expect
    // ------
    // - `OptError::from(Error)` yields `OptError::BackendError`.
    // - The backend error message is propagated into the `text` field.
    fn from_error_non_argmin_yields_backend_error() {
        // Arrange
        let io_err = std::io::Error::other("boom");
        let backend: Error = io_err.into();

        // Act
        let opt_err: OptError = backend.into();

        // Assert
        match opt_err {
            OptError::BackendError { text } => {
                assert!(
                    text.contains("boom"),
                    "expected backend text to contain original message, got: {text}"
                );
            }
            other => panic!("expected BackendError, got: {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Validate that `ACDError` variants used by the optimizer layer are
    // mapped to the corresponding `OptError` variants with their payloads
    // preserved.
    //
    // Given
    // -----
    // - Several `ACDError` variants (`InvalidLogLikInput`, `InvalidPsiLogLik`,
    //   `InvalidExpParam`, `ScaleInvalid`, `ShapeInvalid`).
    //
    // Expect
    // ------
    // - `OptError::from(ACDError)` yields the matching `OptError` variant
    //   for each input.
    fn from_acd_error_maps_to_opt_error() {
        // Arrange
        let acd_invalid_input = ACDError::InvalidLogLikInput { value: -1.0 };
        let acd_invalid_psi = ACDError::InvalidPsiLogLik { value: f64::NAN };
        let acd_invalid_exp = ACDError::InvalidExpParam;
        let acd_scale_invalid = ACDError::ScaleInvalid;
        let acd_shape_invalid = ACDError::ShapeInvalid;

        // Act
        let opt_invalid_input: OptError = acd_invalid_input.into();
        let opt_invalid_psi: OptError = acd_invalid_psi.into();
        let opt_invalid_exp: OptError = acd_invalid_exp.into();
        let opt_scale_invalid: OptError = acd_scale_invalid.into();
        let opt_shape_invalid: OptError = acd_shape_invalid.into();

        // Assert
        assert_eq!(opt_invalid_input, OptError::InvalidLogLikInput { value: -1.0 });
        assert!(matches!(
            opt_invalid_psi,
            OptError::InvalidPsiLogLik { value } if value.is_nan()
        ));
        assert_eq!(opt_invalid_exp, OptError::InvalidExpParam);
        assert_eq!(opt_scale_invalid, OptError::ScaleInvalid);
        assert_eq!(opt_shape_invalid, OptError::ShapeInvalid);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `ParamError` variants relevant to the optimizer surface are
    // mapped into the corresponding `OptError` variants.
    //
    // Given
    // -----
    // - `ParamError` variants representing stationarity, length mismatches,
    //   and invalid scalar/vector parameters.
    //
    // Expect
    // ------
    // - `OptError::from(ParamError)` yields the matching `OptError` variants,
    //   preserving numeric payloads.
    fn from_param_error_maps_to_opt_error() {
        // Arrange
        let stationarity = ParamError::StationarityViolated { coeff_sum: 1.1 };
        let theta_len = ParamError::ThetaLengthMismatch { expected: 3, actual: 2 };
        let omega = ParamError::InvalidOmega { value: -0.5 };
        let alpha_len = ParamError::AlphaLengthMismatch { expected: 2, actual: 1 };
        let alpha = ParamError::InvalidAlpha { index: 0, value: -0.1 };
        let beta_len = ParamError::BetaLengthMismatch { expected: 2, actual: 3 };
        let beta = ParamError::InvalidBeta { index: 1, value: -0.2 };
        let slack = ParamError::InvalidSlack { value: -1.0 };
        let theta_input = ParamError::InvalidThetaInput { index: 4, value: f64::INFINITY };

        // Act
        let opt_stationarity: OptError = stationarity.into();
        let opt_theta_len: OptError = theta_len.into();
        let opt_omega: OptError = omega.into();
        let opt_alpha_len: OptError = alpha_len.into();
        let opt_alpha: OptError = alpha.into();
        let opt_beta_len: OptError = beta_len.into();
        let opt_beta: OptError = beta.into();
        let opt_slack: OptError = slack.into();
        let opt_theta_input: OptError = theta_input.into();

        // Assert
        assert_eq!(opt_stationarity, OptError::StationarityViolated { coeff_sum: 1.1 });
        assert_eq!(opt_theta_len, OptError::ThetaLengthMismatch { expected: 3, actual: 2 });
        assert_eq!(opt_omega, OptError::InvalidOmega { value: -0.5 });
        assert_eq!(opt_alpha_len, OptError::AlphaLengthMismatch { expected: 2, actual: 1 });
        assert_eq!(opt_alpha, OptError::InvalidAlpha { index: 0, value: -0.1 });
        assert_eq!(opt_beta_len, OptError::BetaLengthMismatch { expected: 2, actual: 3 });
        assert_eq!(opt_beta, OptError::InvalidBeta { index: 1, value: -0.2 });
        assert_eq!(opt_slack, OptError::InvalidSlack { value: -1.0 });
        assert_eq!(opt_theta_input, OptError::InvalidThetaInput { index: 4, value: f64::INFINITY });
    }

    #[test]
    // Purpose
    // -------
    // Ensure that the `Display` implementation for `HessianDimMismatch`
    // includes both the expected parameter dimension and the actual Hessian
    // shape to aid debugging.
    //
    // Given
    // -----
    // - An `OptError::HessianDimMismatch` with `expected = 3` and
    //   `found = (2, 4)`.
    //
    // Expect
    // ------
    // - `format!("{}", err)` contains the string representation of both the
    //   expected square dimension `(3, 3)` and the found shape `(2, 4)`.
    fn display_hessian_dim_mismatch_includes_dimensions() {
        // Arrange
        let err = OptError::HessianDimMismatch { expected: 3, found: (2, 4) };

        // Act
        let msg = format!("{err}");

        // Assert
        assert!(
            msg.contains("expected (3, 3)"),
            "expected message to mention expected (3, 3), got: {msg}"
        );
        assert!(msg.contains("(2, 4)"), "expected message to mention found (2, 4), got: {msg}");
    }

    #[test]
    // Purpose
    // -------
    // Check that the `Display` implementation for `InvalidThetaHat` includes
    // the offending index, value, and reason string.
    //
    // Given
    // -----
    // - An `OptError::InvalidThetaHat` with a specific index, value, and
    //   reason.
    //
    // Expect
    // ------
    // - `format!("{}", err)` contains the index, value, and reason in the
    //   rendered string.
    fn display_invalid_theta_hat_includes_index_value_and_reason() {
        // Arrange
        let err = OptError::InvalidThetaHat {
            index: 5,
            value: 42.0,
            reason: "non-finite in transformed space",
        };

        // Act
        let msg = format!("{err}");

        // Assert
        assert!(msg.contains("index 5"), "expected message to include index, got: {msg}");
        assert!(msg.contains("42"), "expected message to include value 42.0, got: {msg}");
        assert!(
            msg.contains("non-finite in transformed space"),
            "expected message to include reason string, got: {msg}"
        );
    }
}
