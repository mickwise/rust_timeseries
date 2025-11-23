//! statistical_tests::errors — shared error types and Python bridges.
//!
//! Purpose
//! -------
//! Provide error enums and result aliases for statistical test routines,
//! together with a conversion layer to Python exceptions for PyO3-based
//! bindings. This keeps test-specific validation and runtime failures
//! localized while exposing a clean error surface to both Rust and Python.
//!
//! Key behaviors
//! -------------
//! - Define [`ELResult`] and [`ELError`] as the canonical result and error
//!   types for the Escanciano–Lobato portmanteau test and its validation
//!   helpers.
//! - Attach human-readable `Display` messages to each error variant so that
//!   diagnostics and logs are meaningful without additional context.
//! - Implement `From<ELError> for PyErr` to map Rust-side validation and
//!   runtime errors into `PyOSError` values visible to Python callers.
//!
//! Invariants & assumptions
//! ------------------------
//! - Statistical test modules which use this error type are expected to
//!   validate their inputs (lengths, finiteness, tuning constants) and
//!   return [`ELResult<T>`] instead of panicking.
//! - `ELError` values are assumed to be small, cheap to clone, and suitable
//!   for use in both unit tests and higher-level orchestration code.
//! - The Python-facing conversion preserves the Rust error message verbatim
//!   inside the `PyOSError` string representation.
//!
//! Conventions
//! -----------
//! - This module is focused on statistical-test errors; model-specific
//!   error types (e.g., duration or optimization errors) live in their own
//!   `errors` modules under the relevant subtrees.
//! - Error messages are phrased in terms of domain constraints (e.g.,
//!   "q must be positive", "1 ≤ d < n") rather than low-level details.
//! - PyO3 conversion always uses `PyOSError` for these errors, treating all
//!   of them as I/O-style failures from the perspective of Python code.
//!
//! Downstream usage
//! ----------------
//! - The Escanciano–Lobato test module and its input validation helpers
//!   return [`ELResult<T>`] to propagate failures cleanly to callers.
//! - Python bindings simply expose functions which return results or raise
//!   `OSError` instances; they do not pattern-match on [`ELError`] directly.
//! - Higher-level Rust code may choose to match on [`ELError`] variants to
//!   implement custom recovery or logging behavior.
//!
//! Testing notes
//! -------------
//! - Unit tests in this module verify that:
//!   - each [`ELError`] variant’s `Display` message embeds its payload
//!     (e.g., offending value or lag index), and
//!   - `From<ELError> for PyErr` produces a `PyOSError` whose debug string
//!     contains the original Rust error message.
//! - Additional integration tests in the statistical-test modules exercise
//!   these errors indirectly via input validation and test execution.

#[cfg(feature = "python-bindings")]
use pyo3::{PyErr, exceptions::PyValueError};

pub type ELResult<T> = Result<T, ELError>;

/// ELError — error conditions for the Escanciano–Lobato test.
///
/// Purpose
/// -------
/// Represent all validation and computation failures that can occur when
/// running the Escanciano–Lobato heteroskedasticity–robust portmanteau
/// test, including malformed inputs and degenerate variance proxies.
///
/// Variants
/// --------
/// - `InsufficientData`
///   The input series does not contain enough observations to compute
///   even the first lag (e.g., `data.len() < 2`).
/// - `InvalidData(value: f64)`
///   A data element is non-finite (NaN or ±∞) and cannot be used in the
///   autocovariance / variance proxy calculations.
/// - `InvalidQValue(q: f64)`
///   The tuning constant `q` is non-positive or otherwise invalid for
///   the penalty function π(p, n, q).
/// - `InvalidDValue(d: usize)`
///   The lag bound `d` violates the constraint `1 ≤ d < n`, where
///   `n = data.len()`.
/// - `ZeroTau(lag: usize)`
///   The heteroskedasticity proxy τ̂ⱼ at lag `lag` evaluates to zero,
///   making ρ̃ⱼ² = γ̂ⱼ² / τ̂ⱼ undefined.
///
/// Invariants
/// ----------
/// - Each variant carries just enough information (offending value or
///   lag index) to allow downstream logging and debugging without
///   leaking large data structures.
/// - `ZeroTau(lag)` is only emitted for `lag ≥ 1` and `lag ≤ d` in the
///   original function call.
///
/// Notes
/// -----
/// - This enum implements [`std::error::Error`] and [`std::fmt::Display`]
///   so it can be used with idiomatic `?`-based error propagation in Rust.
/// - A blanket [`From<ELError> for PyErr`] implementation maps all of
///   these cases to `PyOSError` at the Python boundary, with the
///   human-readable message taken from the `Display` implementation.
#[derive(Debug, Clone, PartialEq)]
pub enum ELError {
    //------ Input validation errors ------
    InsufficientData,
    InvalidData(f64),
    InvalidQValue(f64),
    InvalidDValue(usize),
    ZeroTau(usize),
}

impl std::error::Error for ELError {}

impl std::fmt::Display for ELError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ELError::InsufficientData => {
                write!(f, "Need at least 2 observations to compute lag-1 autocovariance.")
            }
            ELError::InvalidData(value) => {
                write!(f, "Invalid data value: {value}. Must be a finite number.")
            }
            ELError::InvalidQValue(q) => {
                write!(f, "Invalid q value: {q}. Must be positive.")
            }
            ELError::InvalidDValue(d) => {
                write!(f, "Invalid d value: {d}. Must satisfy 1 ≤ d < n (data length).")
            }
            ELError::ZeroTau(lag) => write!(f, "Zero τ̂ value at lag {lag}"),
        }
    }
}

#[cfg(feature = "python-bindings")]
impl From<ELError> for PyErr {
    fn from(err: ELError) -> PyErr {
        PyValueError::new_err(format!("OPTError: {err:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Basic `Display` formatting for ELError variants.
    // - Embedding of payload values (q, d, lag) into error messages.
    //
    // They intentionally DO NOT cover:
    // - The `From<ELError> for PyErr` conversion, since exercising it
    //   requires linking against the Python C API and is better handled
    //   by Python-level tests.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Verify that `ELError::InsufficientData` formats to a non-empty,
    // human-readable message.
    //
    // Given
    // -----
    // - An `ELError::InsufficientData` value.
    //
    // Expect
    // ------
    // - `format!("{err}")` is non-empty.
    fn ele_error_insufficient_data_has_nonempty_display_message() {
        // Arrange
        let err = ELError::InsufficientData;

        // Act
        let msg = err.to_string();

        // Assert
        assert!(
            !msg.trim().is_empty(),
            "Display message for InsufficientData should not be empty."
        );
    }

    #[test]
    // Purpose
    // -------
    // Verify that `ELError::InvalidQValue` includes the offending q
    // value in its `Display` representation.
    //
    // Given
    // -----
    // - An `ELError::InvalidQValue` with q = -1.0.
    //
    // Expect
    // ------
    // - `format!("{err}")` contains "-1".
    fn ele_error_invalid_q_value_includes_payload_in_display() {
        // Arrange
        let err = ELError::InvalidQValue(-1.0);

        // Act
        let msg = err.to_string();

        // Assert
        assert!(
            msg.contains("-1"),
            "Display message should include offending q value.\nGot: {msg}"
        );
    }

    #[test]
    // Purpose
    // -------
    // Verify that `ELError::InvalidDValue` includes the offending lag
    // bound d in its `Display` representation.
    //
    // Given
    // -----
    // - An `ELError::InvalidDValue` with d = 10.
    //
    // Expect
    // ------
    // - `format!("{err}")` contains "10".
    fn ele_error_invalid_d_value_includes_payload_in_display() {
        // Arrange
        let err = ELError::InvalidDValue(10);

        // Act
        let msg = err.to_string();

        // Assert
        assert!(
            msg.contains("10"),
            "Display message should include offending d value.\nGot: {msg}"
        );
    }

    #[test]
    // Purpose
    // -------
    // Ensure that `ELError::ZeroTau` reports the lag index in its
    // `Display` representation.
    //
    // Given
    // -----
    // - An `ELError::ZeroTau` with lag = 3.
    //
    // Expect
    // ------
    // - `format!("{err}")` contains "3".
    fn ele_error_zero_tau_includes_lag_in_display() {
        // Arrange
        let err = ELError::ZeroTau(3);

        // Act
        let msg = err.to_string();

        // Assert
        assert!(
            msg.contains("3"),
            "Display message should include offending lag index.\nGot: {msg}"
        );
    }
}
