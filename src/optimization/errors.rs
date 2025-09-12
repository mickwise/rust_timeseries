use argmin::core::{ArgminError, Error};

use crate::duration::errors::{ACDError, ParamError};

/// Crate-wide result alias for optimizer operations.
pub type OptResult<T> = Result<T, OptError>;

#[derive(Debug, Clone, PartialEq)]
pub enum OptError {
    // ---- Gradient ----
    /// Implies that FD should be used
    GradientNotImplemented,

    /// Gradient dimensions do not match parameter dimensions.
    GradientDimMismatch {
        expected: usize,
        found: usize,
    },

    /// Gradient elements need to be finite
    InvalidGradient {
        index: usize,
        value: f64,
        reason: &'static str,
    },

    // ---- MLEOptions ----
    /// Gradient tolerance needs to be positive and finite.
    InvalidTolGrad {
        tol: f64,
        reason: &'static str,
    },
    /// Cost change tolerance needs to be positive and finite.
    InvalidTolCost {
        tol: f64,
        reason: &'static str,
    },
    /// Function tolerance needs to be positive and finite.
    InvalidTolF {
        tol: f64,
        reason: &'static str,
    },
    /// Maximum iterations needs to be positive.
    InvalidMaxIter {
        max_iter: usize,
        reason: &'static str,
    },
    /// At least one tolerance must be provided.
    NoTolerancesProvided,

    /// Invalid line searcher name.
    InvalidLineSearch {
        name: String,
        reason: &'static str,
    },

    /// lbfgs_mem needs to be at least 1.
    InvalidLBFGSMem {
        mem: usize,
        reason: &'static str,
    },

    // ---- Cost function ----
    /// Cost function returned a non-finite value.
    NonFiniteCost {
        value: f64,
    },

    // ---- Optimizer outcome ----
    /// Estimated parameters must be finite.
    InvalidThetaHat {
        index: usize,
        value: f64,
        reason: &'static str,
    },

    /// Theta hat is missing
    MissingThetaHat,

    // ---- Argmin ---
    /// Wrapper for argmin::InvalidParameter
    InvalidParameter {
        text: String,
    },
    /// Wrapper for argmin::NotImplemented
    NotImplemented {
        text: String,
    },
    /// Wrapper for argmin::NotInitialized
    NotInitialized {
        text: String,
    },
    /// Wrapper for argmin::ConditionViolated
    ConditionViolated {
        text: String,
    },
    /// Wrapper for argmin::CheckPointNotFound
    CheckPointNotFound {
        text: String,
    },
    /// Wrapper for argmin::PotentialBug
    PotentialBug {
        text: String,
    },
    /// Wrapper for argmin::ImpossibleError
    ImpossibleError {
        text: String,
    },
    /// Wrapper for other argmin::Error types
    BackendError {
        text: String,
    },

    // ---- Finite Diffs ----
    /// Hessian matrix dimensions do not match parameter dimensions.
    HessianDimMismatch {
        expected: usize,
        found: (usize, usize),
    },

    /// Hessian values need to be finite.
    InvalidHessian {
        row: usize,
        col: usize,
        value: f64,
    },

    // ---- ACD Errors ----
    /// Invalid input to log-likelihood function
    InvalidLogLikInput {
        value: f64,
    },
    /// Invalid psi value in log-likelihood function
    InvalidPsiLogLik {
        value: f64,
    },
    /// Invalid Weibull parameter
    InvalidExpParam,
    /// Scale parameter is invalid (<= 0 or non-finite)
    ScaleInvalid,
    /// Shape parameter is invalid (<= 0 or non-finite)
    ShapeInvalid,

    // ---- Param Errors ----
    /// Model not stationary (sum of alpha and beta >= 1).
    StationarityViolated {
        coeff_sum: f64,
    },

    /// Theta length mismatch for ACDParams.
    ThetaLengthMismatch {
        expected: usize,
        actual: usize,
    },

    /// Omega must be finite and > 0.
    InvalidOmega {
        value: f64,
    },

    /// Alpha length mismatch for ACDParams.
    AlphaLengthMismatch {
        expected: usize,
        actual: usize,
    },

    /// Alpha coordinates need to be non-negative.
    InvalidAlpha {
        index: usize,
        value: f64,
    },

    /// Beta length mismatch for ACDParams.
    BetaLengthMismatch {
        expected: usize,
        actual: usize,
    },

    /// Beta coordinates need to be non-negative.
    InvalidBeta {
        index: usize,
        value: f64,
    },

    /// Slack value must be non-negative.
    InvalidSlack {
        value: f64,
    },

    /// Unconstrained optimization input must have finite values.
    InvalidThetaInput {
        index: usize,
        value: f64,
    },

    // ---- Fallback ----
    UnknownError,
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
            OptError::UnknownError => {
                write!(f, "Unknown error")
            }
        }
    }
}

impl From<Error> for OptError {
    fn from(original_err: Error) -> Self {
        match original_err.downcast() {
            Ok(opt_err) => match opt_err {
                ArgminError::InvalidParameter { text } => OptError::InvalidParameter { text },
                ArgminError::NotImplemented { text } => OptError::NotImplemented { text },
                ArgminError::NotInitialized { text } => OptError::NotInitialized { text },
                ArgminError::ConditionViolated { text } => OptError::ConditionViolated { text },
                ArgminError::CheckpointNotFound { text } => OptError::CheckPointNotFound { text },
                ArgminError::PotentialBug { text } => OptError::PotentialBug { text },
                ArgminError::ImpossibleError { text } => OptError::ImpossibleError { text },
                _ => OptError::UnknownError,
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
            _ => OptError::UnknownError,
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
            _ => OptError::UnknownError,
        }
    }
}
