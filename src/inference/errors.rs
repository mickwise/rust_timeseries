//! inference::errors — unified error handling for inference utilities.
//!
//! Purpose
//! -------
//! Provide a central error type and result alias for inference routines such
//! as HAC bandwidth selection, robust variance estimation, and related
//! plug-in procedures. This keeps inference-level failures distinct from
//! low-level numerical or I/O errors while still integrating cleanly with
//! `anyhow::Error`.
//!
//! Key behaviors
//! -------------
//! - Define [`InferenceError`] as the shared error enum for inference
//!   routines (bandwidth selection, plug-in estimators, etc.).
//! - Provide a crate-wide [`InferenceResult<T>`] alias for functions that
//!   may fail due to inference-specific conditions.
//! - Map generic [`anyhow::Error`] values into [`InferenceError::Anyhow`]
//!   for ergonomic interop with higher-level callers.
//!
//! Invariants & assumptions
//! ------------------------
//! - `StationarityViolated` assumes an AR(1) setting where stationarity
//!   requires `|phi| < 1` (or `|phi| <= 1 - ε` in guarded implementations).
//! - `DenominatorTooSmall` indicates that the bandwidth formula’s
//!   denominator approached zero closely enough that proceeding would be
//!   numerically unstable; callers are expected to treat this as a hard
//!   failure rather than silently regularizing.
//! - `OrderNotSupported` is used when a requested kernel or plug-in order
//!   is outside the set of orders implemented by the bandwidth routine.
//!
//! Conventions
//! -----------
//! - All inference routines that can fail on domain or numerical grounds
//!   return [`InferenceResult<T>`] rather than panicking.
//! - AR(1) coefficients `phi` are dimensionless and are expected to be
//!   estimated under standard time-series conventions (e.g., zero-mean or
//!   demeaned input).
//! - Error messages are prefixed with `Inference Error:` and are intended
//!   to be human-readable in logs and Python exceptions.
//!
//! Downstream usage
//! ----------------
//! - HAC bandwidth selection helpers and robust covariance estimators
//!   propagate inference failures via [`InferenceError`] so that callers
//!   can distinguish inference-level issues from generic runtime errors.
//! - Higher-level APIs may convert other error types into
//!   [`anyhow::Error`] and then into [`InferenceError::Anyhow`] to keep a
//!   single error channel at the surface.
//! - Library users are expected to pattern-match on [`InferenceError`]
//!   when they care about specific failure modes (e.g., stationarity
//!   violations versus unsupported orders).
//!
//! Testing notes
//! -------------
//! - Unit tests for this module focus on:
//!   - `Display` formatting for each concrete [`InferenceError`] variant.
//!   - Correct mapping from `anyhow::Error` into
//!     [`InferenceError::Anyhow`].
//! - Integration tests in inference routines (e.g., HAC bandwidth
//!   selection) assert that domain violations in estimators are surfaced
//!   as the appropriate [`InferenceError`] variants (rather than panics
//!   or opaque `anyhow::Error` values).

/// InferenceError — shared failure modes for inference utilities.
///
/// Purpose
/// -------
/// Represent the domain-specific failures that can occur in inference
/// routines such as HAC bandwidth selection and robust variance estimation.
/// Keeps stationarity and numerical degeneracy issues explicit while still
/// allowing a generic catch-all path for upstream errors.
///
/// Variants
/// --------
/// - `StationarityViolated { phi }`  
///   AR(1) stationarity condition is violated (e.g., `|phi| >= 1`), so
///   plug-in bandwidth is not well defined.
/// - `DenominatorTooSmall { denominator }`  
///   The bandwidth formula’s denominator is too close to zero to proceed
///   safely; using it would lead to numerical instability or overflow.
/// - `OrderNotSupported { ord }`  
///   The requested kernel/order is not implemented by the current
///   bandwidth routine (e.g., only a small finite set of `ord` values is
///   supported).
/// - `Anyhow(String)`  
///   Catch-all wrapper for errors originating as `anyhow::Error`. The
///   string payload is the human-readable message from the original error.
/// - `UnknownError`  
///   Fallback for inference failures that cannot be classified more
///   precisely. Intended primarily as a guard for future extensions.
///
/// Invariants
/// ----------
/// - Inference routines return `InferenceResult<T>` and should prefer
///   concrete variants (`StationarityViolated`, `DenominatorTooSmall`,
///   `OrderNotSupported`) over [`InferenceError::UnknownError`] where a
///   precise condition is known.
/// - `StationarityViolated::phi` is the estimated AR(1) coefficient at
///   which the stationarity check failed; callers may use it for
///   diagnostics and logging.
/// - `DenominatorTooSmall::denominator` captures the actual value that
///   triggered the guard, in the same units as the underlying formula.
///
/// Notes
/// -----
/// - Downstream code is expected to pattern-match on [`InferenceError`]
///   when it needs to distinguish between stationarity issues, numerical
///   degeneracies, and generic upstream failures.
/// - Storing `Anyhow(String)` (rather than `anyhow::Error`) keeps the enum
///   `Clone + PartialEq`, trading richer backtraces for easier testing and
///   comparison.
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceError {
    // ---- Bandwidth selection ----
    StationarityViolated { phi: f64 },
    DenominatorTooSmall { denominator: f64 },
    OrderNotSupported { ord: usize },

    // ---- Anyhow catchall ----
    Anyhow(String),

    // ---- Fallback ----
    UnknownError,
}

/// InferenceResult — standard result alias for inference routines.
///
/// This is the preferred return type for HAC bandwidth selection and
/// related plug-in estimators that may fail with [`InferenceError`].
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

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Display formatting for each concrete `InferenceError` variant.
    // - Conversion from `anyhow::Error` into `InferenceError::Anyhow`.
    // - Basic equality semantics of `InferenceError` variants.
    //
    // They intentionally DO NOT cover:
    // - End-to-end HAC bandwidth selection or covariance estimation flows.
    // - Integration behavior of inference routines that produce these errors.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Verify that `InferenceError::StationarityViolated` renders a stable,
    // human-readable message that includes the phi value.
    //
    // Given
    // -----
    // - An `InferenceError::StationarityViolated` with a specific phi.
    //
    // Expect
    // ------
    // - `Display` produces the documented prefix and embeds the phi value.
    fn inferenceerror_display_stationarityviolated_includes_phi_and_prefix() {
        // Arrange
        let err = InferenceError::StationarityViolated { phi: 1.05 };

        // Act
        let msg = err.to_string();

        // Assert
        assert_eq!(msg, "Inference Error: Stationarity violated (phi = 1.05)");
    }

    #[test]
    // Purpose
    // -------
    // Verify that `InferenceError::DenominatorTooSmall` renders a stable,
    // human-readable message including the problematic denominator value.
    //
    // Given
    // -----
    // - An `InferenceError::DenominatorTooSmall` with a small denominator.
    //
    // Expect
    // ------
    // - `Display` uses the documented prefix and embeds the denominator
    //   using standard `f64` formatting.
    fn inferenceerror_display_denominator_toosmall_includes_value_and_prefix() {
        // Arrange
        let denominator = 1e-12;
        let err = InferenceError::DenominatorTooSmall { denominator };

        // Act
        let msg = err.to_string();

        // Assert
        let expected = format!(
            "Inference Error: Denominator too small ({}) in bandwidth calculation",
            denominator
        );
        assert_eq!(msg, expected);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `InferenceError::OrderNotSupported` renders a stable,
    // human-readable message including the unsupported order.
    //
    // Given
    // -----
    // - An `InferenceError::OrderNotSupported` with a specific order.
    //
    // Expect
    // ------
    // - `Display` uses the documented prefix and embeds the order value.
    fn inferenceerror_display_ordernotsupported_includes_order_and_prefix() {
        // Arrange
        let err = InferenceError::OrderNotSupported { ord: 3 };

        // Act
        let msg = err.to_string();

        // Assert
        assert_eq!(msg, "Inference Error: Order 3 not supported for bandwidth calculation");
    }

    #[test]
    // Purpose
    // -------
    // Verify that `InferenceError::Anyhow` renders the wrapped message with
    // the standard `Inference Error:` prefix.
    //
    // Given
    // -----
    // - An `InferenceError::Anyhow` constructed with a simple message.
    //
    // Expect
    // ------
    // - `Display` includes the prefix and the wrapped message verbatim.
    fn inferenceerror_display_anyhow_wrapped_message_has_prefix() {
        // Arrange
        let err = InferenceError::Anyhow("bandwidth failed".to_string());

        // Act
        let msg = err.to_string();

        // Assert
        assert_eq!(msg, "Inference Error: bandwidth failed");
    }

    #[test]
    // Purpose
    // -------
    // Verify that `InferenceError::UnknownError` renders the documented
    // fallback message.
    //
    // Given
    // -----
    // - An `InferenceError::UnknownError` instance.
    //
    // Expect
    // ------
    // - `Display` returns the standard "Unknown error" message.
    fn inferenceerror_display_unknownerror_uses_standard_message() {
        // Arrange
        let err = InferenceError::UnknownError;

        // Act
        let msg = err.to_string();

        // Assert
        assert_eq!(msg, "Inference Error: Unknown error occurred");
    }

    #[test]
    // Purpose
    // -------
    // Verify that `From<anyhow::Error>` maps into `InferenceError::Anyhow`
    // and that the wrapped message is preserved and formatted correctly.
    //
    // Given
    // -----
    // - An `anyhow::Error` created from a simple string message.
    //
    // Expect
    // ------
    // - Converting into `InferenceError` yields an `Anyhow` variant with the
    //   same message.
    // - `Display` uses the `Inference Error:` prefix plus the original text.
    fn inferenceerror_from_anyhow_preserves_message_and_variant() {
        // Arrange
        let source = anyhow!("plug-in bandwidth computation failed");

        // Act
        let err: InferenceError = source.into();

        // Assert
        assert_eq!(err, InferenceError::Anyhow("plug-in bandwidth computation failed".to_string()));
        assert_eq!(err.to_string(), "Inference Error: plug-in bandwidth computation failed");
    }
}
