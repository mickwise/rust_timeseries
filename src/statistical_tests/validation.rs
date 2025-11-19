//! statistical_tests::validation — shared input guards for test statistics.
//!
//! Purpose
//! -------
//! Centralize basic input validation for statistical test routines in this
//! crate. This avoids duplicating checks on series length, data finiteness,
//! and tuning parameters (e.g., q, d) across multiple modules.
//!
//! Key behaviors
//! -------------
//! - Enforce simple preconditions on time-series inputs before expensive
//!   computations are performed.
//! - Map invalid inputs into structured `ELError` values for consistent
//!   error handling in Rust and Python bindings.
//!
//! Invariants & assumptions
//! ------------------------
//! - Input series must have length at least 2 to support lag-1 quantities.
//! - All data values must be finite (`!NaN`, not ±∞).
//! - The tuning constant `q` must be strictly positive.
//! - The lag bound `d` must satisfy `1 ≤ d < n`, where `n = data.len()`.
//!
//! Conventions
//! -----------
//! - This module is purely about *validation*; it performs no I/O and does
//!   not allocate beyond what is required for error construction.
//! - Errors are reported via the crate-local `ELError` enum, which is also
//!   convertible to `PyErr` in Python-facing layers.
//! - Callers are responsible for any further model-specific checks
//!   (stationarity, identifiability, etc.).
//!
//! Downstream usage
//! ----------------
//! - Call [`validate_input`] at the top of test routines (e.g. Escanciano–
//!   Lobato) before computing autocovariances or test statistics.
//! - Treat a successful return (`Ok(())`) as a guarantee that basic shape
//!   and parameter constraints are satisfied.
//! - Handle `ELError` variants in Rust or rely on the `From<ELError>`
//!   implementation to surface them as `OSError` in Python.
//!
//! Testing notes
//! -------------
//! - Unit tests in this module cover all error branches of [`validate_input`]
//!   and a simple success path.

use crate::statistical_tests::errors::{ELError, ELResult};

/// Validate basic input constraints for time-series test routines.
///
/// Parameters
/// ----------
/// - `data`: `&[f64]`
///   Input series of real-valued observations (typically residuals).
///   Must have length at least 2, and all values must be finite (no `NaN`
///   or ±∞).
/// - `q`: `f64`
///   Positive tuning constant used in test-specific penalty functions.
///   Must satisfy `q > 0.0`.
/// - `d`: `usize`
///   Upper bound on the candidate lag order. Must satisfy
///   `1 ≤ d < data.len()`.
///
/// Returns
/// -------
/// `ELResult<()>`
///   - `Ok(())` if all basic constraints are satisfied.
///   - `Err(ELError)` if any constraint is violated, with a variant that
///     encodes which condition failed and, where relevant, the offending
///     value.
///
/// Errors
/// ------
/// - `ELError::InsufficientData`
///   Returned when `data.len() < 2`, so lag-based quantities cannot be
///   computed meaningfully.
/// - `ELError::InvalidData(value)`
///   Returned when any element of `data` is not finite (i.e., `NaN` or
///   ±∞), with `value` set to the offending entry.
/// - `ELError::InvalidQValue(q)`
///   Returned when `q <= 0.0`.
/// - `ELError::InvalidDValue(d)`
///   Returned when `d == 0` or `d >= data.len()`, violating
///   `1 ≤ d < n`.
///
/// Panics
/// ------
/// - Never panics. All failures are reported via `ELError`.
///
/// Notes
/// -----
/// - This helper is intentionally minimal and side-effect free; it performs
///   no allocations beyond those required by the `ELError` variants.
/// - Callers may layer additional domain-specific checks (e.g., stationarity
///   constraints) on top of this generic validation.
/// - Keeping this logic centralized makes it easier to maintain consistent
///   error semantics between Rust and Python.
///
/// Examples
/// --------
/// ```rust
/// # use rust_timeseries::statistical_tests::validation::validate_input;
/// # use rust_timeseries::statistical_tests::errors::ELError;
/// let data = vec![0.1_f64, -0.2, 0.3];
/// let q = 3.0;
/// let d = 2;
///
/// // Valid inputs succeed:
/// assert!(validate_input(&data, q, d).is_ok());
///
/// // Invalid q produces an InvalidQValue error:
/// match validate_input(&data, 0.0, d) {
///     Err(ELError::InvalidQValue(_)) => (),
///     other => panic!("expected InvalidQValue error, got {other:?}"),
/// }
/// ```
pub fn validate_input(data: &[f64], q: f64, d: usize) -> ELResult<()> {
    if data.len() < 2 {
        return Err(ELError::InsufficientData);
    }

    for &value in data {
        if !value.is_finite() {
            return Err(ELError::InvalidData(value));
        }
    }

    if q <= 0.0 {
        return Err(ELError::InvalidQValue(q));
    }

    if d == 0 || d >= data.len() {
        return Err(ELError::InvalidDValue(d));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistical_tests::errors::ELError;

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Successful validation of well-formed inputs.
    // - Each error branch in `validate_input`:
    //   * insufficient data length,
    //   * non-finite data value,
    //   * non-positive q,
    //   * invalid d (0 or ≥ n).
    //
    // They intentionally DO NOT cover:
    // - Any interaction with Python / PyO3 (conversion to `PyErr`), which
    //   is exercised by Python-level tests.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Verify that `validate_input` succeeds on a simple, valid input
    // triple (finite data, q > 0, 1 ≤ d < n).
    //
    // Given
    // -----
    // - A finite series of length 3.
    // - q = 3.0 > 0.
    // - d = 2, which satisfies 1 ≤ d < n.
    //
    // Expect
    // ------
    // - `validate_input` returns `Ok(())`.
    fn validate_input_valid_arguments_succeeds() {
        // Arrange
        let data = vec![0.1_f64, -0.2, 0.3];
        let q = 3.0;
        let d = 2;

        // Act
        let result = validate_input(&data, q, d);

        // Assert
        assert!(result.is_ok(), "Expected Ok(()) for valid inputs, got {result:?}");
    }

    #[test]
    // Purpose
    // -------
    // Ensure that a series with fewer than 2 observations is rejected
    // with `ELError::InsufficientData`.
    //
    // Given
    // -----
    // - A single-element series.
    // - q = 3.0 and d = 1.
    //
    // Expect
    // ------
    // - `validate_input` returns `Err(ELError::InsufficientData)`.
    fn validate_input_too_short_series_returns_insufficient_data() {
        // Arrange
        let data = vec![0.1_f64];
        let q = 3.0;
        let d = 1;

        // Act
        let result = validate_input(&data, q, d);

        // Assert
        match result {
            Err(ELError::InsufficientData) => (),
            other => panic!("expected InsufficientData error, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that any non-finite value (e.g., NaN) in the data triggers
    // `ELError::InvalidData` with the offending payload.
    //
    // Given
    // -----
    // - A series containing a `NaN`.
    // - q = 3.0 and d = 1.
    //
    // Expect
    // ------
    // - `validate_input` returns `Err(ELError::InvalidData(value))`.
    fn validate_input_non_finite_value_returns_invalid_data() {
        // Arrange
        let data = vec![0.1_f64, f64::NAN, 0.3];
        let q = 3.0;
        let d = 1;

        // Act
        let result = validate_input(&data, q, d);

        // Assert
        match result {
            Err(ELError::InvalidData(v)) => {
                assert!(
                    !v.is_finite(),
                    "InvalidData payload should itself be non-finite. Got: {v}"
                );
            }
            other => panic!("expected InvalidData error, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Ensure that non-positive q values are rejected with
    // `ELError::InvalidQValue`.
    //
    // Given
    // -----
    // - A finite series of length 3.
    // - q ≤ 0 (e.g., 0.0).
    // - d = 1.
    //
    // Expect
    // ------
    // - `validate_input` returns `Err(ELError::InvalidQValue(q))`.
    fn validate_input_non_positive_q_returns_invalid_q_value() {
        // Arrange
        let data = vec![0.1_f64, -0.2, 0.3];
        let q = 0.0;
        let d = 1;

        // Act
        let result = validate_input(&data, q, d);

        // Assert
        match result {
            Err(ELError::InvalidQValue(v)) => {
                assert!(v <= 0.0, "InvalidQValue payload should be non-positive. Got: {v}");
            }
            other => panic!("expected InvalidQValue error, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that a lag bound d = 0 is rejected with
    // `ELError::InvalidDValue(0)`.
    //
    // Given
    // -----
    // - A finite series of length 3.
    // - q = 3.0.
    // - d = 0.
    //
    // Expect
    // ------
    // - `validate_input` returns `Err(ELError::InvalidDValue(0))`.
    fn validate_input_zero_d_returns_invalid_d_value() {
        // Arrange
        let data = vec![0.1_f64, -0.2, 0.3];
        let q = 3.0;
        let d = 0;

        // Act
        let result = validate_input(&data, q, d);

        // Assert
        match result {
            Err(ELError::InvalidDValue(v)) => {
                assert_eq!(v, 0, "InvalidDValue payload should be the offending d. Got: {v}");
            }
            other => panic!("expected InvalidDValue(0) error, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that a lag bound d equal to the series length (d = n) is
    // rejected with `ELError::InvalidDValue(d)`, since we require
    // `d < n`.
    //
    // Given
    // -----
    // - A finite series of length 3.
    // - q = 3.0.
    // - d = n = 3.
    //
    // Expect
    // ------
    // - `validate_input` returns `Err(ELError::InvalidDValue(3))`.
    fn validate_input_d_equal_to_len_returns_invalid_d_value() {
        // Arrange
        let data = vec![0.1_f64, -0.2, 0.3];
        let q = 3.0;
        let d = data.len();

        // Act
        let result = validate_input(&data, q, d);

        // Assert
        match result {
            Err(ELError::InvalidDValue(v)) => {
                assert_eq!(v, d, "InvalidDValue payload should be the offending d. Got: {v}");
            }
            other => panic!("expected InvalidDValue(d) error, got {other:?}"),
        }
    }
}
