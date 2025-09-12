//! Unified error handling for inference routines.
//!
//! This module defines `InferenceError`, the central error type used by
//! HAC bandwidth selection, robust variance estimation, and related
//! inference utilities. It groups together domain-specific failures
//! (e.g., nonstationarity, numerical underflow) with catch-all and
//! fallback variants. An alias `InferenceResult<T>` standardizes the
//! return type across inference code.

/// Unified error type for inference routines.
///
/// Covers plug-in bandwidth calculation failures, unsupported settings,
/// numerical degeneracies, and generic passthrough errors. Designed to
/// integrate seamlessly with `anyhow::Error` via `From`, and to provide
/// readable diagnostics through `Display`.
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceError {
    // ---- Bandwidth selection ----
    /// Stationarity is violated in an AR(1) process.
    StationarityViolated {
        phi: f64,
    },

    /// Denominator is too close to zero in bandwidth calculation.
    DenominatorTooSmall {
        denominator: f64,
    },

    /// Order not supported for bandwidth calculation.
    OrderNotSupported {
        ord: usize,
    },

    // ---- Anyhow catchall ----
    Anyhow(String),

    // ---- Fallback ----
    UnknownError,
}

pub type InferenceResult<T> = Result<T, InferenceError>;

impl From<anyhow::Error> for InferenceError {
    fn from(err: anyhow::Error) -> Self {
        InferenceError::Anyhow(err.to_string())
    }
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // ---- Bandwidth selection ----
            InferenceError::StationarityViolated { phi } => {
                write!(f, "Inference Error: Stationarity violated (phi = {})", phi)
            }
            InferenceError::DenominatorTooSmall { denominator } => write!(
                f,
                "Inference Error: Denominator too small ({}) in bandwidth calculation",
                denominator
            ),
            InferenceError::OrderNotSupported { ord } => {
                write!(f, "Inference Error: Order {} not supported for bandwidth calculation", ord)
            }

            // ---- Anyhow catchall ----
            InferenceError::Anyhow(msg) => write!(f, "Inference Error: {}", msg),

            // ---- Fallback ----
            InferenceError::UnknownError => write!(f, "Inference Error: Unknown error occurred"),
        }
    }
}
