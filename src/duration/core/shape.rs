//! ACD shape — model order (p, q) for ACD models.
//!
//! Purpose
//! -------
//! Represent and validate the Engle–Russell / Wikipedia ACD(p, q) model
//! orders used throughout the duration stack. Centralizes how `p` and `q`
//! are interpreted and enforces basic sanity checks against the available
//! sample size so downstream code can assume a well-posed recursion.
//!
//! Key behaviors
//! -------------
//! - Store the (p, q) orders for ACD models under the Engle–Russell
//!   convention (`q` duration lags, `p` ψ lags).
//! - Validate that at least one of `p` or `q` is positive and strictly less
//!   than the in-sample length `n`.
//! - Fail fast with structured [`ACDError::InvalidModelShape`] when a
//!   requested order is under-identified for the given sample.
//!
//! Invariants & assumptions
//! ------------------------
//! - ACD(0, 0) is not allowed; at least one of `p` or `q` must be > 0.
//! - Both `p` and `q` must satisfy `p < n` and `q < n`, where `n` is the
//!   in-sample length used for fitting / recursion.
//! - Callers interpret `q` as the number of duration lags (α₁…α_q) and `p`
//!   as the number of ψ lags (β₁…β_p) in the Engle–Russell convention.
//!
//! Conventions
//! -----------
//! - We follow the Engle–Russell / Wikipedia indexing: `q` is the τ-lag
//!   order and `p` is the ψ-lag order.
//! - Shape validation is performed at construction time via
//!   [`ACDShape::new`]; this module reports invalid input via
//!   [`ACDError::InvalidModelShape`] and never panics on bad arguments.
//! - Downstream code may construct [`ACDShape`] via struct literals for
//!   internal use, but public APIs should prefer [`ACDShape::new`] so the
//!   documented invariants hold.
//!
//! Downstream usage
//! ----------------
//! - Use [`ACDShape::new`] when building ACD models or allocating scratch
//!   buffers to ensure the chosen orders are compatible with the sample
//!   length.
//! - Treat [`ACDShape`] as the canonical source of `(p, q)` when sizing
//!   ψ / duration lag buffers and configuring recursions and forecasts.
//! - Higher-level components (e.g., [`ACDModel`]) may assume that shapes
//!   passed to them have already been validated via this module.
//!
//! Testing notes
//! -------------
//! - Unit tests exercise [`ACDShape::new`] on representative valid
//!   and invalid `(p, q, n)` combinations, including boundary cases such as
//!   `p = 0`, `q = 0`, `p = n - 1`, and `p = n`.
//! - Error paths assert that [`ACDError::InvalidModelShape`] is
//!   returned with the expected offending parameter value and reason.
use crate::duration::errors::{ACDError, ACDResult};

/// ACDShape — model order (p, q) for an ACD model.
///
/// Purpose
/// -------
/// Represent the Engle–Russell / Wikipedia ACD(p, q) orders and provide a
/// single place to enforce basic validity constraints on `p` and `q`
/// relative to the available sample size.
///
/// Key behaviors
/// -------------
/// - Stores the number of duration lags `q` (α terms) and ψ lags `p`
///   (β terms) for the ACD recursion.
/// - Encodes the convention that at least one of `p` or `q` must be
///   strictly positive.
/// - Exposes a constructor [`ACDShape::new`] that validates model order
///   against an in-sample length `n` and reports invalid shapes via
///   [`ACDError::InvalidModelShape`].
///
/// Parameters
/// ----------
/// Constructed via [`ACDShape::new(p, q, n)`]:
/// - `p`: `usize`
///   Number of ψ lags (β terms) in the ACD model.
/// - `q`: `usize`
///   Number of duration lags (α terms) in the ACD model.
/// - `n`: `usize`
///   Number of in-sample observations the model will be fit to. Must satisfy
///   `n > max(p, q)`.
///
/// Fields
/// ------
/// - `p`: `usize`
///   ψ-lag order (β₁…β_p) of the ACD model.
/// - `q`: `usize`
///   Duration-lag order (α₁…α_q) of the ACD model.
///
/// Invariants
/// ----------
/// - When constructed via [`ACDShape::new`]:
///   - At least one of `p` or `q` is strictly positive (ACD(0, 0) is invalid).
///   - Both `p` and `q` are strictly less than the in-sample length `n`.
/// - Callers are expected to interpret `p` and `q` under the Engle–Russell
///   convention; this type does not support alternative orderings.
///
/// Performance
/// -----------
/// - [`ACDShape`] is a small, `Copy` value type with no heap allocations.
/// - Validation via [`ACDShape::new`] is O(1).
///
/// Notes
/// -----
/// - The fields are public for ergonomic access. Callers that construct
///   [`ACDShape`] via struct literals bypass validation; public APIs should
///   prefer [`ACDShape::new`] so the documented invariants are enforced.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ACDShape {
    pub p: usize,
    pub q: usize,
}

impl ACDShape {
    /// Construct and validate an `ACDShape = ACD(p, q)` against a sample size `n`.
    ///
    /// Parameters
    /// ----------
    /// - `p`: `usize`
    ///   Number of ψ lags (β terms) in the ACD model. Must satisfy `p < n`.
    /// - `q`: `usize`
    ///   Number of duration lags (α terms) in the ACD model. Must satisfy `q < n`.
    /// - `n`: `usize`
    ///   Number of in-sample observations the model will be fit to. Must be
    ///   strictly greater than `max(p, q)` and non-zero.
    ///
    /// Returns
    /// -------
    /// `ACDResult<ACDShape>`
    ///   - `Ok(ACDShape)` when:
    ///       - at least one of `p` or `q` is > 0, and
    ///       - both `p` and `q` are strictly less than `n`.
    ///   - `Err(ACDError::InvalidModelShape)` when these conditions are
    ///     violated.
    ///
    /// Errors
    /// ------
    /// - `ACDError::InvalidModelShape`
    ///   Returned when:
    ///   - `p == 0 && q == 0` (no dynamics — ACD(0, 0) is not supported), or
    ///   - `p >= n` (ψ-lag order is too large for the available sample), or
    ///   - `q >= n` (duration-lag order is too large for the available sample).
    ///
    /// Panics
    /// ------
    /// - Never panics.
    ///
    /// Notes
    /// -----
    /// - This constructor only enforces order vs. sample-size consistency.
    ///   Stationarity and parameter restrictions (e.g., ∑α + ∑β < 1) are
    ///   handled elsewhere.
    /// - Failing early here allows downstream components (e.g., ACD models,
    ///   optimizers, and forecast routines) to assume `n > max(p, q)` without
    ///   performing their own checks.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// use rust_timeseries::duration::core::shape::ACDShape;
    /// use rust_timeseries::duration::errors::ACDResult;
    ///
    /// fn build_shape() -> ACDResult<ACDShape> {
    ///     // A simple ACD(1, 1) specification for a sample of length 100.
    ///     let shape = ACDShape::new(1, 1, 100)?;
    ///     assert_eq!(shape.p, 1);
    ///     assert_eq!(shape.q, 1);
    ///     Ok(shape)
    /// }
    ///
    /// // Invalid: ACD(0, 0) is rejected.
    /// assert!(ACDShape::new(0, 0, 10).is_err());
    ///
    /// // Invalid: `q` cannot be as large as or larger than `n`.
    /// assert!(ACDShape::new(1, 10, 10).is_err());
    /// ```
    pub fn new(p: usize, q: usize, n: usize) -> ACDResult<Self> {
        if p == 0 && q == 0 {
            return Err(ACDError::InvalidModelShape {
                param: p,
                reason: "Both p and q cannot be zero.",
            });
        }
        if p >= n {
            return Err(ACDError::InvalidModelShape {
                param: p,
                reason: "p must be less than the number of observations.",
            });
        }
        if q >= n {
            return Err(ACDError::InvalidModelShape {
                param: q,
                reason: "q must be less than the number of observations.",
            });
        }
        Ok(ACDShape { p, q })
    }
}

#[cfg(test)]
mod tests {
    use super::{ACDError, ACDShape};

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Construction of `ACDShape` via `ACDShape::new` under valid and invalid
    //   (p, q, n) combinations.
    // - Boundary behavior when `p` or `q` approach or exceed the sample size `n`.
    //
    // They intentionally DO NOT cover:
    // - Integration with `ACDModel`, workspace sizing, or recursion logic.
    // - PyO3 conversion or `Display` formatting for `ACDError`.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Verify that `ACDShape::new` accepts valid (p, q, n) where at least one
    // of p, q is > 0 and both are strictly less than n, including boundary
    // cases with p = n - 1, q = n - 1.
    //
    // Given
    // -----
    // - A sample size n = 10.
    // - Orders:
    //   - (p, q) = (1, 1),
    //   - (p, q) = (0, 1),
    //   - (p, q) = (1, 0),
    //   - (p, q) = (n - 1, n - 1).
    //
    // Expect
    // ------
    // - `ACDShape::new` returns `Ok(ACDShape { p, q })` for each combination.
    fn acdshape_new_accepts_valid_orders_less_than_n() {
        // Arrange
        let n = 10usize;

        // Act
        let shape_11 = ACDShape::new(1, 1, n).expect("ACD(1, 1) should be valid");
        let shape_01 = ACDShape::new(0, 1, n).expect("ACD(0, 1) should be valid");
        let shape_10 = ACDShape::new(1, 0, n).expect("ACD(1, 0) should be valid");
        let shape_boundary = ACDShape::new(n - 1, n - 1, n).expect("ACD(n-1, n-1) should be valid");

        // Assert
        assert_eq!(shape_11, ACDShape { p: 1, q: 1 });
        assert_eq!(shape_01, ACDShape { p: 0, q: 1 });
        assert_eq!(shape_10, ACDShape { p: 1, q: 0 });
        assert_eq!(shape_boundary, ACDShape { p: n - 1, q: n - 1 });
    }

    #[test]
    // Purpose
    // -------
    // Verify that `ACDShape::new` rejects the degenerate ACD(0, 0) shape even
    // when there are enough observations.
    //
    // Given
    // -----
    // - (p, q, n) = (0, 0, 10).
    //
    // Expect
    // ------
    // - `ACDShape::new` returns `Err(ACDError::InvalidModelShape)` with:
    //   - `param == 0`,
    //   - `reason == "Both p and q cannot be zero."`.
    fn acdshape_new_rejects_zero_zero_shape_with_invalid_model_shape_error() {
        // Arrange
        let n = 10usize;

        // Act
        let result = ACDShape::new(0, 0, n);

        // Assert
        match result {
            Err(ACDError::InvalidModelShape { param, reason }) => {
                assert_eq!(param, 0);
                assert_eq!(reason, "Both p and q cannot be zero.");
            }
            Ok(shape) => {
                panic!("Expected Err(InvalidModelShape), got Ok({shape:?}) for ACD(0, 0)");
            }
            Err(other) => {
                panic!("Expected InvalidModelShape for ACD(0, 0), got {other:?}");
            }
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `ACDShape::new` rejects shapes where p >= n, i.e. the ψ-lag
    // order is not strictly less than the sample size.
    //
    // Given
    // -----
    // - n = 5.
    // - Two cases with q < n:
    //   - (p, q) = (n, 1),
    //   - (p, q) = (n + 1, 1).
    //
    // Expect
    // ------
    // - `ACDShape::new` returns `Err(ACDError::InvalidModelShape)` in both
    //   cases with:
    //   - `param == p`,
    //   - `reason == "p must be less than the number of observations."`.
    fn acdshape_new_rejects_p_ge_n_with_invalid_model_shape_error() {
        // Arrange
        let n = 5usize;

        // Case 1: p == n
        let result_equal = ACDShape::new(n, 1, n);
        match result_equal {
            Err(ACDError::InvalidModelShape { param, reason }) => {
                assert_eq!(param, n);
                assert_eq!(reason, "p must be less than the number of observations.");
            }
            Ok(shape) => {
                panic!("Expected Err(InvalidModelShape), got Ok({shape:?}) for p == n");
            }
            Err(other) => {
                panic!("Expected InvalidModelShape for p == n, got {other:?}");
            }
        }

        // Case 2: p > n
        let p_gt = n + 1;
        let result_greater = ACDShape::new(p_gt, 1, n);
        match result_greater {
            Err(ACDError::InvalidModelShape { param, reason }) => {
                assert_eq!(param, p_gt);
                assert_eq!(reason, "p must be less than the number of observations.");
            }
            Ok(shape) => {
                panic!("Expected Err(InvalidModelShape), got Ok({shape:?}) for p > n");
            }
            Err(other) => {
                panic!("Expected InvalidModelShape for p > n, got {other:?}");
            }
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `ACDShape::new` rejects shapes where q >= n, i.e. the
    // duration-lag order is not strictly less than the sample size.
    //
    // Given
    // -----
    // - n = 5.
    // - Two cases with p < n:
    //   - (p, q) = (1, n),
    //   - (p, q) = (1, n + 1).
    //
    // Expect
    // ------
    // - `ACDShape::new` returns `Err(ACDError::InvalidModelShape)` in both
    //   cases with:
    //   - `param == q`,
    //   - `reason == "q must be less than the number of observations."`.
    fn acdshape_new_rejects_q_ge_n_with_invalid_model_shape_error() {
        // Arrange
        let n = 5usize;

        // Case 1: q == n
        let result_equal = ACDShape::new(1, n, n);
        match result_equal {
            Err(ACDError::InvalidModelShape { param, reason }) => {
                assert_eq!(param, n);
                assert_eq!(reason, "q must be less than the number of observations.");
            }
            Ok(shape) => {
                panic!("Expected Err(InvalidModelShape), got Ok({shape:?}) for q == n");
            }
            Err(other) => {
                panic!("Expected InvalidModelShape for q == n, got {other:?}");
            }
        }

        // Case 2: q > n
        let q_gt = n + 1;
        let result_greater = ACDShape::new(1, q_gt, n);
        match result_greater {
            Err(ACDError::InvalidModelShape { param, reason }) => {
                assert_eq!(param, q_gt);
                assert_eq!(reason, "q must be less than the number of observations.");
            }
            Ok(shape) => {
                panic!("Expected Err(InvalidModelShape), got Ok({shape:?}) for q > n");
            }
            Err(other) => {
                panic!("Expected InvalidModelShape for q > n, got {other:?}");
            }
        }
    }
}
