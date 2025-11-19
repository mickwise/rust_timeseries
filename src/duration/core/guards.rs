//! ψ-guards for ACD models — enforce lower/upper bounds on the ψ recursion.
//!
//! Purpose
//! -------
//! Provide a small, validated container for ψ-guards used in ACD models to keep
//! the conditional mean duration process `ψ_t` within a safe numeric range during
//! recursion and likelihood evaluation.
//!
//! Key behaviors
//! -------------
//! - Construct [`PsiGuards`] values that enforce strict positivity and finiteness
//!   of the ψ lower/upper bounds.
//! - Reject invalid guard configurations via typed errors (`ACDError`) instead of
//!   panicking at call sites.
//! - Expose simple `min`/`max` fields that downstream recursion code can use to
//!   clamp ψ values.
//!
//! Invariants & assumptions
//! ------------------------
//! - `min < max` must hold for all constructed guards.
//! - Both `min` and `max` must be finite floating-point values.
//! - `min > 0.0` to keep log terms in the likelihood well-defined.
//! - Callers are responsible for choosing bounds that are reasonable for the
//!   model’s scale and data; this module only enforces basic numeric sanity.
//!
//! Conventions
//! -----------
//! - ψ is the conditional mean duration process used by ACD models.
//! - Guards are represented as a pair `(min, max)` in model units (e.g.,
//!   seconds, milliseconds) and are stored as `f64`.
//! - Invalid configurations return `ACDError::InvalidPsiGuards` rather than
//!   panicking.
//!
//! Downstream usage
//! ----------------
//! - Construct [`PsiGuards`] at the beginning of estimation or forecasting and
//!   pass them into ψ-recursion functions to clamp each computed ψ_t.
//! - Use `guards.min` and `guards.max` wherever ψ values are updated to avoid
//!   underflow/overflow and log(0) issues.
//! - Treat this module as part of the public surface for configuring numeric
//!   safety in ACD recursions.
//!
//! Testing notes
//! -------------
//! - Unit tests validate that `PsiGuards::new`:
//!   - accepts valid `(min, max)` pairs with `0.0 < min < max` and finite bounds,
//!   - rejects non-finite bounds, non-positive `min`, and `min >= max` with the
//!     correct `ACDError::InvalidPsiGuards` payload.
//! - Behavior of guards in full ψ-recursions is covered by tests in the
//!   forecasting / recursion modules rather than here.
use crate::duration::errors::{ACDError, ACDResult};

/// PsiGuards — lower/upper bounds for the ACD ψ recursion.
///
/// Purpose
/// -------
/// Represent a pair of validated lower/upper bounds for the conditional mean
/// duration process `ψ_t` in ACD models, ensuring ψ values stay within a safe
/// numeric range during recursion and likelihood evaluation.
///
/// Key behaviors
/// -------------
/// - Stores strict lower/upper bounds (`min`, `max`) for ψ_t, enforcing
///   positivity and finiteness at construction time.
/// - Provides a lightweight container that downstream code can use to clamp
///   ψ values on each recursion step.
/// - Helps prevent numerical issues such as log(0), underflow toward 0, or
///   overflow to extremely large values during optimization.
///
/// Parameters
/// ----------
/// Constructed via [`PsiGuards::new`] with:
/// - `value`: `(f64, f64)`
///   Tuple `(min, max)` specifying the desired lower and upper bounds on ψ.
///   Must satisfy `0.0 < min < max`, and both entries must be finite.
///
/// Fields
/// ------
/// - `min`: `f64`
///   Lower bound for ψ (strictly greater than 0.0). Used to keep log terms
///   well-defined and avoid degeneracy toward zero.
/// - `max`: `f64`
///   Upper bound for ψ (strictly greater than `min`). Used to cap ψ at a
///   numerically safe level and avoid overflow.
///
/// Invariants
/// ----------
/// - `min > 0.0`.
/// - `min < max`.
/// - `min` and `max` are both finite (`is_finite() == true`).
///
/// Performance
/// -----------
/// - Construction is O(1) with a small constant amount of validation logic.
/// - Copying is cheap; the type is `Copy` and can be passed by value.
///
/// Notes
/// -----
/// - This type does not enforce any particular scale or interpretation of ψ;
///   it only checks basic numeric sanity. Scale is determined by the model and
///   upstream preprocessing.
/// - Intended to be used across estimation and forecasting code paths as a
///   simple, centralized contract for ψ bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsiGuards {
    /// Lower bound for ψ (strictly > 0).
    pub min: f64,
    /// Upper bound for ψ (must be > `min`).
    pub max: f64,
}

impl PsiGuards {
    /// Construct validated ψ bounds from a `(min, max)` tuple.
    ///
    /// Parameters
    /// ----------
    /// - `value`: `(f64, f64)`
    ///   Tuple `(min, max)` specifying the desired lower and upper bounds on the
    ///   conditional mean duration process `ψ_t`. The caller is expected to pass
    ///   values in model units (e.g., seconds, milliseconds).
    ///
    /// Returns
    /// -------
    /// ACDResult<PsiGuards>
    ///   - `Ok(PsiGuards)` when `0.0 < min < max` and both bounds are finite.
    ///   - `Err(ACDError::InvalidPsiGuards { .. })` when the provided tuple
    ///     violates any of the guard invariants.
    ///
    /// Errors
    /// ------
    /// - `ACDError::InvalidPsiGuards`
    ///   Returned when:
    ///   - `min >= max`,
    ///   - either bound is not finite (`!is_finite()`),
    ///   - or `min <= 0.0`, which would make log-likelihood terms ill-defined.
    ///
    /// Panics
    /// ------
    /// - Never panics; all invalid inputs are reported via
    ///   `ACDError::InvalidPsiGuards`.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; no additional safety requirements are imposed on
    ///   the caller beyond providing a sensible `(min, max)` pair.
    ///
    /// Notes
    /// -----
    /// - This constructor centralizes ψ-guard validation so that downstream
    ///   recursion code can assume `0.0 < min < max` and finiteness without
    ///   rechecking.
    /// - Callers should choose guard ranges that are wide enough not to bias the
    ///   model but tight enough to prevent numerical pathologies.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::guards::PsiGuards;
    /// # use rust_timeseries::duration::errors::ACDError;
    /// // Valid guards
    /// let guards = PsiGuards::new((1e-6, 1e6)).unwrap();
    /// assert!(guards.min > 0.0);
    /// assert!(guards.max > guards.min);
    ///
    /// // Invalid guards (min >= max)
    /// let err = PsiGuards::new((1.0, 1.0)).unwrap_err();
    /// if let ACDError::InvalidPsiGuards { .. } = err {
    ///     // expected
    /// } else {
    ///     panic!("expected InvalidPsiGuards error");
    /// }
    /// ```
    pub fn new(value: (f64, f64)) -> ACDResult<Self> {
        if value.0 >= value.1 {
            return Err(ACDError::InvalidPsiGuards {
                min: value.0,
                max: value.1,
                reason: "Psi guards must have min < max.",
            });
        }

        if !value.0.is_finite() || !value.1.is_finite() {
            return Err(ACDError::InvalidPsiGuards {
                min: value.0,
                max: value.1,
                reason: "Psi guards must be finite.",
            });
        }

        if value.0 <= 0.0 {
            return Err(ACDError::InvalidPsiGuards {
                min: value.0,
                max: value.1,
                reason: "Psi guards must be strictly positive.",
            });
        }

        Ok(PsiGuards { min: value.0, max: value.1 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duration::errors::ACDError;

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Validation behavior of `PsiGuards::new` for valid and invalid (min, max)
    //   tuples.
    // - Edge cases around positivity, ordering (min < max), and finiteness of
    //   the bounds.
    //
    // They intentionally DO NOT cover:
    // - How guards are applied inside ψ-recursions (that is tested in
    //   forecasting / recursion modules).
    // - Any model-scale-specific choices of guard ranges; only basic numeric
    //   sanity is tested here.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Verify that `PsiGuards::new` accepts a valid (min, max) pair and returns
    // a guard with matching fields.
    //
    // Given
    // -----
    // - `min = 1e-6`, `max = 1e6`, both finite and strictly positive.
    //
    // Expect
    // ------
    // - `PsiGuards::new((min, max))` returns `Ok(PsiGuards)`.
    // - The returned `PsiGuards` has `min` and `max` equal to the inputs.
    fn psiguards_new_accepts_valid_bounds() {
        // Arrange
        let min = 1e-6;
        let max = 1e6;

        // Act
        let result =
            PsiGuards::new((min, max)).expect("valid (min, max) should construct PsiGuards");

        // Assert
        assert_eq!(result.min, min);
        assert_eq!(result.max, max);
    }

    #[test]
    // Purpose
    // -------
    // Ensure that `PsiGuards::new` rejects configurations where `min >= max`.
    //
    // Given
    // -----
    // - A tuple `(min, max)` with `min = max = 1.0`.
    //
    // Expect
    // ------
    // - `PsiGuards::new((min, max))` returns `Err(ACDError::InvalidPsiGuards)`.
    // - The error carries the same `min` and `max` values that were provided.
    fn psiguards_new_rejects_min_greater_or_equal_max() {
        // Arrange
        let min = 1.0;
        let max = 1.0;

        // Act
        let err = PsiGuards::new((min, max)).unwrap_err();

        // Assert
        match err {
            ACDError::InvalidPsiGuards { min: minimum, max: maximum, .. } => {
                assert_eq!(minimum, min);
                assert_eq!(maximum, max);
            }
            other => panic!("expected InvalidPsiGuards, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Ensure that `PsiGuards::new` rejects non-finite bounds (e.g., +∞).
    //
    // Given
    // -----
    // - A tuple `(min, max)` with `min = 1.0`, `max = f64::INFINITY`.
    //
    // Expect
    // ------
    // - `PsiGuards::new((min, max))` returns `Err(ACDError::InvalidPsiGuards)`.
    fn psiguards_new_rejects_non_finite_bounds() {
        // Arrange
        let min = 1.0;
        let max = f64::INFINITY;

        // Act
        let err = PsiGuards::new((min, max)).unwrap_err();

        // Assert
        match err {
            ACDError::InvalidPsiGuards { .. } => {
                // exact fields / reason are not critical here; we just care
                // that the constructor rejects non-finite bounds.
            }
            other => panic!("expected InvalidPsiGuards for non-finite bounds, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Ensure that `PsiGuards::new` rejects non-positive `min` (including 0.0).
    //
    // Given
    // -----
    // - A tuple `(min, max)` with `min = 0.0`, `max = 10.0`.
    //
    // Expect
    // ------
    // - `PsiGuards::new((min, max))` returns `Err(ACDError::InvalidPsiGuards)`.
    fn psiguards_new_rejects_non_positive_min() {
        // Arrange
        let min = 0.0;
        let max = 10.0;

        // Act
        let err = PsiGuards::new((min, max)).unwrap_err();

        // Assert
        match err {
            ACDError::InvalidPsiGuards { min: minimum, max: maximum, .. } => {
                assert_eq!(minimum, min);
                assert_eq!(maximum, max);
            }
            other => panic!("expected InvalidPsiGuards for non-positive min, got {other:?}"),
        }
    }
}
