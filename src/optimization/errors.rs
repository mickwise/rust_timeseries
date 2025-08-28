use argmin::core::{ArgminError, Error};

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
    InvalidParameter {
        text: String,
    },
    NotImplemented {
        text: String,
    },
    NotInitialized {
        text: String,
    },
    ConditionViolated {
        text: String,
    },
    CheckPointNotFound {
        text: String,
    },
    PotentialBug {
        text: String,
    },
    ImpossibleError {
        text: String,
    },
    BackendError {
        text: String,
    },
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
