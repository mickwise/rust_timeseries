//! loglik_optimizer::validation — shared consistency checks for optimizer state.
//!
//! Purpose
//! -------
//! Centralize common validation logic used throughout the log-likelihood
//! optimizer. These helpers enforce basic shape and finiteness invariants
//! on tolerances, gradients, parameter estimates, objective values, and
//! Hessians so that the rest of the optimization layer can assume
//! well-formed inputs.
//!
//! Key behaviors
//! -------------
//! - Validate numerical tolerances for gradient norms and cost deltas
//!   via [`verify_tol_grad`] and [`verify_tol_cost`].
//! - Check gradient and Hessian dimensions and finiteness using
//!   [`validate_grad`] and [`validate_hessian`].
//! - Enforce that candidate parameter estimates and objective values are
//!   present and finite with [`validate_theta_hat`] and [`validate_value`].
//!
//! Invariants & assumptions
//! ------------------------
//! - All tolerances, when present, must be finite and strictly positive.
//! - Gradient vectors and Hessian matrices must match the expected
//!   dimension and contain only finite entries.
//! - A valid `theta_hat` must be provided before constructing a final
//!   optimization outcome; missing or non-finite entries are rejected.
//! - Objective values (cost or log-likelihood) must be finite; NaN and
//!   ±∞ are treated as fatal errors at this layer.
//!
//! Conventions
//! -----------
//! - Dimension arguments (`dim`) refer to both the expected length of
//!   parameter and gradient vectors and the row/column size of Hessians.
//! - All validation functions return `OptResult<()>` or `OptResult<Theta>`
//!   and never panic; callers are expected to propagate or handle
//!   [`OptError`] variants explicitly.
//! - These helpers are side-effect free: no logging, I/O, or mutation
//!   beyond inspecting their inputs.
//!
//! Downstream usage
//! ----------------
//! - [`Tolerances::new`] delegates tolerance checks to
//!   [`verify_tol_grad`] and [`verify_tol_cost`].
//! - The optimizer’s result construction path uses [`validate_theta_hat`]
//!   and [`validate_value`] to guard final outputs before exposing them
//!   via [`OptimOutcome`].
//! - Gradient- and Hessian-based solvers call [`validate_grad`] and
//!   [`validate_hessian`] when consuming user-provided derivatives or
//!   second-order information.
//!
//! Testing notes
//! -------------
//! - Unit tests for this module exercise all validation branches,
//!   including success paths and each `OptError` variant produced by the
//!   helpers.
//! - Integration tests for the full optimizer rely on these helpers
//!   implicitly when constructing tolerances, consuming user gradients,
//!   and normalizing solver outputs.
use crate::optimization::{
    errors::{OptError, OptResult},
    loglik_optimizer::types::{Grad, Hessian, Theta},
};

/// Validate an optional gradient-norm tolerance.
///
/// Parameters
/// ----------
/// - `tol`: `Option<f64>`
///   Gradient-norm tolerance. When `Some(τ)`, `τ` must be finite and
///   strictly positive. `None` disables gradient-based termination.
///
/// Returns
/// -------
/// `OptResult<()>`
///   - `Ok(())` if `tol` is `None` or a finite, strictly positive value.
///   - `Err(e)` if the tolerance violates its constraints.
///
/// Errors
/// ------
/// - `OptError::InvalidTolGrad`
///   Returned when `tol` is `Some` but non-finite (NaN or ±∞) or not
///   strictly positive (≤ 0.0).
///
/// Panics
/// ------
/// - Never panics.
///
/// Safety
/// ------
/// - No `unsafe` code is used.
///
/// Notes
/// -----
/// - This helper does not modify any state; it is intended to be called
///   from constructors such as [`Tolerances::new`] to centralize
///   tolerance validation.
///
/// Examples
/// --------
/// ```rust
/// # use rust_timeseries::optimization::loglik_optimizer::validation::verify_tol_grad;
/// # use rust_timeseries::optimization::errors::OptError;
/// assert!(verify_tol_grad(Some(1e-6)).is_ok());
///
/// let err = verify_tol_grad(Some(0.0)).unwrap_err();
/// match err {
///     OptError::InvalidTolGrad { .. } => {}
///     other => panic!("Expected InvalidTolGrad, got {other:?}"),
/// }
/// ```
pub fn verify_tol_grad(tol: Option<f64>) -> OptResult<()> {
    if let Some(tol) = tol {
        if !tol.is_finite() {
            return Err(OptError::InvalidTolGrad { tol, reason: "Tolerance must be finite." });
        }
        if tol <= 0.0 {
            return Err(OptError::InvalidTolGrad { tol, reason: "Tolerance must be positive." });
        }
    }
    Ok(())
}

/// Validate an optional cost-change tolerance used for convergence.
///
/// Parameters
/// ----------
/// - `tol`: `Option<f64>`
///   Cost-delta tolerance. When `Some(τ)`, `τ` must be finite and
///   strictly positive. `None` disables cost-change-based termination.
///
/// Returns
/// -------
/// `OptResult<()>`
///   - `Ok(())` if `tol` is `None` or a finite, strictly positive value.
///   - `Err(e)` if the tolerance violates its constraints.
///
/// Errors
/// ------
/// - `OptError::InvalidTolCost`
///   Returned when `tol` is `Some` but non-finite (NaN or ±∞) or not
///   strictly positive (≤ 0.0).
///
/// Panics
/// ------
/// - Never panics.
///
/// Safety
/// ------
/// - No `unsafe` code is used.
///
/// Notes
/// -----
/// - This helper is typically called from [`Tolerances::new`] to enforce
///   consistent tolerance semantics across the optimizer.
///
/// Examples
/// --------
/// ```rust
/// # use rust_timeseries::optimization::loglik_optimizer::validation::verify_tol_cost;
/// # use rust_timeseries::optimization::errors::OptError;
/// assert!(verify_tol_cost(Some(1e-8)).is_ok());
///
/// let err = verify_tol_cost(Some(-1.0)).unwrap_err();
/// match err {
///     OptError::InvalidTolCost { .. } => {}
///     other => panic!("Expected InvalidTolCost, got {other:?}"),
/// }
/// ```
pub fn verify_tol_cost(tol: Option<f64>) -> OptResult<()> {
    if let Some(tol) = tol {
        if !tol.is_finite() {
            return Err(OptError::InvalidTolCost { tol, reason: "Tolerance must be finite." });
        }
        if tol <= 0.0 {
            return Err(OptError::InvalidTolCost { tol, reason: "Tolerance must be positive." });
        }
    }
    Ok(())
}

/// Validate the shape and finiteness of a gradient vector.
///
/// Parameters
/// ----------
/// - `grad`: `&Grad`
///   Gradient vector to check. Must have length equal to `dim` and all
///   entries finite (no NaN or ±∞).
/// - `dim`: `usize`
///   Expected dimensionality of the parameter space; the gradient length
///   must match this value.
///
/// Returns
/// -------
/// `OptResult<()>`
///   - `Ok(())` if the gradient length equals `dim` and all elements are
///     finite.
///   - `Err(e)` if the gradient fails any of the checks.
///
/// Errors
/// ------
/// - `OptError::GradientDimMismatch`
///   Returned when `grad.len() != dim`.
/// - `OptError::InvalidGradient`
///   Returned when any element is non-finite, with the index, value, and
///   a short reason describing the failure.
///
/// Panics
/// ------
/// - Never panics.
///
/// Safety
/// ------
/// - No `unsafe` code is used.
///
/// Notes
/// -----
/// - Only the first offending element is reported for non-finite
///   gradients; callers that need full diagnostics must inspect the
///   vector themselves.
/// - This function does not allocate or modify the gradient; it only
///   reads entries.
///
/// Examples
/// --------
/// ```rust
/// # use ndarray::Array1;
/// # use rust_timeseries::optimization::loglik_optimizer::validation::validate_grad;
/// # use rust_timeseries::optimization::errors::OptError;
/// let grad = Array1::from(vec![0.0_f64, 1.0, 2.0]);
/// validate_grad(&grad, 3).unwrap();
///
/// let bad_grad = Array1::from(vec![0.0_f64, f64::NAN]);
/// let err = validate_grad(&bad_grad, 2).unwrap_err();
/// match err {
///     OptError::InvalidGradient { index, .. } => assert_eq!(index, 1),
///     other => panic!("Expected InvalidGradient, got {other:?}"),
/// }
/// ```
pub fn validate_grad(grad: &Grad, dim: usize) -> OptResult<()> {
    if grad.len() != dim {
        return Err(OptError::GradientDimMismatch { expected: dim, found: grad.len() });
    }
    for (index, &value) in grad.iter().enumerate() {
        if !value.is_finite() {
            return Err(OptError::InvalidGradient {
                index,
                value,
                reason: "Gradient elements must be finite.",
            });
        }
    }
    Ok(())
}

/// Validate and unwrap a candidate parameter estimate `theta_hat`.
///
/// Parameters
/// ----------
/// - `theta_hat`: `Option<Theta>`
///   Optional parameter vector returned by the optimizer. Must be `Some`
///   and contain only finite entries to be considered valid.
///
/// Returns
/// -------
/// `OptResult<Theta>`
///   - `Ok(theta)` when `theta_hat` is `Some` and all elements of
///     `theta` are finite. Ownership of the vector is transferred to the
///     caller.
///   - `Err(e)` when the estimate is missing or contains invalid values.
///
/// Errors
/// ------
/// - `OptError::MissingThetaHat`
///   Returned when `theta_hat` is `None`.
/// - `OptError::InvalidThetaHat`
///   Returned when any element of the provided vector is non-finite,
///   including its index, value, and a short reason.
///
/// Panics
/// ------
/// - Never panics.
///
/// Safety
/// ------
/// - No `unsafe` code is used.
///
/// Notes
/// -----
/// - This helper consumes the `Theta` value on success, making it
///   convenient to call inside result constructors (e.g.,
///   [`OptimOutcome::new`]) without additional cloning.
/// - Only the first non-finite entry is reported for invalid vectors.
///
/// Examples
/// --------
/// ```rust
/// # use ndarray::Array1;
/// # use rust_timeseries::optimization::loglik_optimizer::{Theta};
/// # use rust_timeseries::optimization::loglik_optimizer::validation::validate_theta_hat;
/// # use rust_timeseries::optimization::errors::OptError;
/// let theta: Theta = Array1::from(vec![0.0_f64, 1.0]);
/// let validated = validate_theta_hat(Some(theta.clone())).unwrap();
/// assert_eq!(validated, theta);
///
/// let bad_theta: Theta = Array1::from(vec![0.0_f64, f64::INFINITY]);
/// let err = validate_theta_hat(Some(bad_theta)).unwrap_err();
/// match err {
///     OptError::InvalidThetaHat { index, .. } => assert_eq!(index, 1),
///     other => panic!("Expected InvalidThetaHat, got {other:?}"),
/// }
/// ```
pub fn validate_theta_hat(theta_hat: Option<Theta>) -> OptResult<Theta> {
    match theta_hat {
        Some(t) => {
            for (index, &value) in t.iter().enumerate() {
                if !value.is_finite() {
                    return Err(OptError::InvalidThetaHat {
                        index,
                        value,
                        reason: "Parameter estimates must be finite.",
                    });
                }
            }
            Ok(t)
        }
        None => Err(OptError::MissingThetaHat),
    }
}

/// Validate that a scalar objective value is finite.
///
/// Negative values are fine as long as they are finite. This helper is
/// used for both costs and log-likelihood values; non-finite inputs are
/// normalized to `OptError::NonFiniteCost`.
///
/// Parameters
/// ----------
/// - `value`: `f64`
///   Scalar objective to check (e.g., cost or log-likelihood). May be
///   positive or negative, but must be finite.
///
/// Returns
/// -------
/// `OptResult<()>`
///   - `Ok(())` if `value` is finite.
///   - `Err(e)` if `value` is NaN or infinite.
///
/// Errors
/// ------
/// - `OptError::NonFiniteCost`
///   Returned when `value` is NaN or ±∞, carrying the offending value.
///
/// Panics
/// ------
/// - Never panics.
///
/// Safety
/// ------
/// - No `unsafe` code is used.
///
/// Notes
/// -----
/// - This helper does not interpret the sign of the objective; it only
///   enforces finiteness. It is suitable for both cost and
///   log-likelihood scalars as long as they share the same error
///   handling path.
///
/// Examples
/// --------
/// ```rust
/// # use rust_timeseries::optimization::loglik_optimizer::validation::validate_value;
/// # use rust_timeseries::optimization::errors::OptError;
/// validate_value(-10.0).unwrap();
///
/// let err = validate_value(f64::NAN).unwrap_err();
/// match err {
///     OptError::NonFiniteCost { .. } => {}
///     other => panic!("Expected NonFiniteCost, got {other:?}"),
/// }
/// ```
pub fn validate_value(value: f64) -> OptResult<()> {
    if !value.is_finite() {
        return Err(OptError::NonFiniteCost { value });
    }
    Ok(())
}

/// Validate the shape and finiteness of a Hessian matrix.
///
/// Parameters
/// ----------
/// - `hessian`: `&Hessian`
///   Dense Hessian matrix to validate. Must be square with both dimensions
///   equal to `dim` and contain only finite entries.
/// - `dim`: `usize`
///   Expected dimension of the parameter space; the Hessian must be
///   `dim × dim`.
///
/// Returns
/// -------
/// `OptResult<()>`
///   - `Ok(())` if the Hessian is `dim × dim` and all entries are finite.
///   - `Err(e)` if either the dimensions or entries are invalid.
///
/// Errors
/// ------
/// - `OptError::HessianDimMismatch`
///   Returned when the number of rows or columns differs from `dim`.
/// - `OptError::InvalidHessian`
///   Returned when any entry is non-finite, including row/column indices
///   and the offending value.
///
/// Panics
/// ------
/// - Never panics.
///
/// Safety
/// ------
/// - No `unsafe` code is used.
///
/// Notes
/// -----
/// - Only the first non-finite entry is reported for invalid Hessians.
/// - This helper does not inspect symmetry or positive definiteness; it
///   only enforces basic shape and finiteness constraints. Higher-level
///   checks (e.g., for curvature) are done elsewhere.
///
/// Examples
/// --------
/// ```rust
/// # use ndarray::Array2;
/// # use rust_timeseries::optimization::loglik_optimizer::types::Hessian;
/// # use rust_timeseries::optimization::loglik_optimizer::validation::validate_hessian;
/// # use rust_timeseries::optimization::errors::OptError;
/// let h: Hessian = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// validate_hessian(&h, 2).unwrap();
///
/// let bad: Hessian = Array2::from_shape_vec((2, 2), vec![1.0, f64::NAN, 0.0, 1.0]).unwrap();
/// let err = validate_hessian(&bad, 2).unwrap_err();
/// match err {
///     OptError::InvalidHessian { row, col, .. } => {
///         assert_eq!((row, col), (0, 1));
///     }
///     other => panic!("Expected InvalidHessian, got {other:?}"),
/// }
/// ```
pub fn validate_hessian(hessian: &Hessian, dim: usize) -> OptResult<()> {
    if hessian.nrows() != dim || hessian.ncols() != dim {
        return Err(OptError::HessianDimMismatch {
            expected: dim,
            found: (hessian.nrows(), hessian.ncols()),
        });
    }
    for ((i, j), &value) in hessian.indexed_iter() {
        if !value.is_finite() {
            return Err(OptError::InvalidHessian { row: i, col: j, value });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::errors::OptError;
    use ndarray::{Array1, Array2};

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Validation invariants for tolerances (`verify_tol_grad`,
    //   `verify_tol_cost`).
    // - Shape and finiteness checks for gradients, parameter estimates,
    //   scalar objective values, and Hessians.
    //
    // They intentionally DO NOT cover:
    // - End-to-end optimizer behavior or Argmin integration (covered by
    //   higher-level optimizer tests).
    // - Any ACD- or duration-specific error variants on `OptError`.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Confirm that `verify_tol_grad` accepts `None` to disable gradient-based
    // termination.
    //
    // Given
    // -----
    // - `tol = None`.
    //
    // Expect
    // ------
    // - `verify_tol_grad` returns `Ok(())`.
    fn verify_tol_grad_accepts_none() {
        // Arrange
        let tol = None;

        // Act
        let result = verify_tol_grad(tol);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    // Purpose
    // -------
    // Confirm that `verify_tol_grad` accepts a finite, strictly positive
    // tolerance.
    //
    // Given
    // -----
    // - `tol = Some(1e-6)`.
    //
    // Expect
    // ------
    // - `verify_tol_grad` returns `Ok(())`.
    fn verify_tol_grad_accepts_positive_finite_value() {
        // Arrange
        let tol = Some(1e-6);

        // Act
        let result = verify_tol_grad(tol);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    // Purpose
    // -------
    // Verify that `verify_tol_grad` rejects non-finite tolerances.
    //
    // Given
    // -----
    // - `tol = Some(f64::NAN)`.
    //
    // Expect
    // ------
    // - `verify_tol_grad` returns `Err(OptError::InvalidTolGrad { .. })`.
    fn verify_tol_grad_errors_on_non_finite_value() {
        // Arrange
        let tol = Some(f64::NAN);

        // Act
        let result = verify_tol_grad(tol);

        // Assert
        let err = result.expect_err("Non-finite tol_grad should be rejected");
        match err {
            OptError::InvalidTolGrad { .. } => {}
            other => panic!("Expected InvalidTolGrad, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `verify_tol_grad` rejects non-positive tolerances.
    //
    // Given
    // -----
    // - `tol = Some(0.0)`.
    //
    // Expect
    // ------
    // - `verify_tol_grad` returns `Err(OptError::InvalidTolGrad { .. })`.
    fn verify_tol_grad_errors_on_non_positive_value() {
        // Arrange
        let tol = Some(0.0);

        // Act
        let result = verify_tol_grad(tol);

        // Assert
        let err = result.expect_err("Non-positive tol_grad should be rejected");
        match err {
            OptError::InvalidTolGrad { .. } => {}
            other => panic!("Expected InvalidTolGrad, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Confirm that `verify_tol_cost` accepts `None` to disable
    // cost-change-based termination.
    //
    // Given
    // -----
    // - `tol = None`.
    //
    // Expect
    // ------
    // - `verify_tol_cost` returns `Ok(())`.
    fn verify_tol_cost_accepts_none() {
        // Arrange
        let tol = None;

        // Act
        let result = verify_tol_cost(tol);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    // Purpose
    // -------
    // Confirm that `verify_tol_cost` accepts a finite, strictly positive
    // tolerance.
    //
    // Given
    // -----
    // - `tol = Some(1e-8)`.
    //
    // Expect
    // ------
    // - `verify_tol_cost` returns `Ok(())`.
    fn verify_tol_cost_accepts_positive_finite_value() {
        // Arrange
        let tol = Some(1e-8);

        // Act
        let result = verify_tol_cost(tol);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    // Purpose
    // -------
    // Verify that `verify_tol_cost` rejects non-finite tolerances.
    //
    // Given
    // -----
    // - `tol = Some(f64::INFINITY)`.
    //
    // Expect
    // ------
    // - `verify_tol_cost` returns `Err(OptError::InvalidTolCost { .. })`.
    fn verify_tol_cost_errors_on_non_finite_value() {
        // Arrange
        let tol = Some(f64::INFINITY);

        // Act
        let result = verify_tol_cost(tol);

        // Assert
        let err = result.expect_err("Non-finite tol_cost should be rejected");
        match err {
            OptError::InvalidTolCost { .. } => {}
            other => panic!("Expected InvalidTolCost, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `verify_tol_cost` rejects non-positive tolerances.
    //
    // Given
    // -----
    // - `tol = Some(-1.0)`.
    //
    // Expect
    // ------
    // - `verify_tol_cost` returns `Err(OptError::InvalidTolCost { .. })`.
    fn verify_tol_cost_errors_on_non_positive_value() {
        // Arrange
        let tol = Some(-1.0);

        // Act
        let result = verify_tol_cost(tol);

        // Assert
        let err = result.expect_err("Non-positive tol_cost should be rejected");
        match err {
            OptError::InvalidTolCost { .. } => {}
            other => panic!("Expected InvalidTolCost, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `validate_grad` rejects gradients whose length does not
    // match the expected dimension.
    //
    // Given
    // -----
    // - `grad.len() = 2`.
    // - `dim = 3`.
    //
    // Expect
    // ------
    // - `validate_grad` returns `Err(OptError::GradientDimMismatch { .. })`.
    fn validate_grad_errors_on_dimension_mismatch() {
        // Arrange
        let grad: Grad = Array1::from(vec![0.0_f64, 1.0]);
        let dim = 3;

        // Act
        let result = validate_grad(&grad, dim);

        // Assert
        let err = result.expect_err("Dimension mismatch should be rejected");
        match err {
            OptError::GradientDimMismatch { expected, found } => {
                assert_eq!(expected, dim);
                assert_eq!(found, grad.len());
            }
            other => panic!("Expected GradientDimMismatch, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `validate_grad` rejects gradients with non-finite entries.
    //
    // Given
    // -----
    // - `grad = [0.0, NaN]`.
    // - `dim = 2`.
    //
    // Expect
    // ------
    // - `validate_grad` returns `Err(OptError::InvalidGradient { .. })`
    //   with `index = 1`.
    fn validate_grad_errors_on_non_finite_entry() {
        // Arrange
        let grad: Grad = Array1::from(vec![0.0_f64, f64::NAN]);
        let dim = 2;

        // Act
        let result = validate_grad(&grad, dim);

        // Assert
        let err = result.expect_err("Non-finite gradient should be rejected");
        match err {
            OptError::InvalidGradient { index, .. } => {
                assert_eq!(index, 1);
            }
            other => panic!("Expected InvalidGradient, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Confirm that `validate_grad` accepts a gradient of the correct length
    // with all finite entries.
    //
    // Given
    // -----
    // - `grad = [0.0, 1.0, 2.0]`.
    // - `dim = 3`.
    //
    // Expect
    // ------
    // - `validate_grad` returns `Ok(())`.
    fn validate_grad_accepts_valid_gradient() {
        // Arrange
        let grad: Grad = Array1::from(vec![0.0_f64, 1.0, 2.0]);
        let dim = 3;

        // Act
        let result = validate_grad(&grad, dim);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    // Purpose
    // -------
    // Verify that `validate_theta_hat` rejects a missing parameter vector.
    //
    // Given
    // -----
    // - `theta_hat = None`.
    //
    // Expect
    // ------
    // - `validate_theta_hat` returns `Err(OptError::MissingThetaHat)`.
    fn validate_theta_hat_errors_when_missing() {
        // Arrange
        let theta_hat: Option<Theta> = None;

        // Act
        let result = validate_theta_hat(theta_hat);

        // Assert
        let err = result.expect_err("Missing theta_hat should be rejected");
        match err {
            OptError::MissingThetaHat => {}
            other => panic!("Expected MissingThetaHat, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `validate_theta_hat` rejects parameter vectors with
    // non-finite entries.
    //
    // Given
    // -----
    // - `theta_hat = Some([0.0, +∞])`.
    //
    // Expect
    // ------
    // - `validate_theta_hat` returns `Err(OptError::InvalidThetaHat { .. })`
    //   with `index = 1`.
    fn validate_theta_hat_errors_on_non_finite_entry() {
        // Arrange
        let theta: Theta = Array1::from(vec![0.0_f64, f64::INFINITY]);
        let theta_hat = Some(theta);

        // Act
        let result = validate_theta_hat(theta_hat);

        // Assert
        let err = result.expect_err("Non-finite theta_hat should be rejected");
        match err {
            OptError::InvalidThetaHat { index, .. } => {
                assert_eq!(index, 1);
            }
            other => panic!("Expected InvalidThetaHat, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Confirm that `validate_theta_hat` accepts a finite parameter vector
    // and returns it unchanged.
    //
    // Given
    // -----
    // - `theta_hat = Some([0.0, 1.0])`.
    //
    // Expect
    // ------
    // - `validate_theta_hat` returns `Ok(theta)` matching the input.
    fn validate_theta_hat_accepts_valid_theta() {
        // Arrange
        let theta: Theta = Array1::from(vec![0.0_f64, 1.0]);
        let theta_hat = Some(theta.clone());

        // Act
        let validated = validate_theta_hat(theta_hat).expect("Valid theta_hat should be accepted");

        // Assert
        assert_eq!(validated, theta);
    }

    #[test]
    // Purpose
    // -------
    // Confirm that `validate_value` accepts a finite scalar objective.
    //
    // Given
    // -----
    // - `value = -10.0`.
    //
    // Expect
    // ------
    // - `validate_value` returns `Ok(())`.
    fn validate_value_accepts_finite_scalar() {
        // Arrange
        let value = -10.0_f64;

        // Act
        let result = validate_value(value);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    // Purpose
    // -------
    // Verify that `validate_value` rejects non-finite objective values.
    //
    // Given
    // -----
    // - `value = NaN`.
    //
    // Expect
    // ------
    // - `validate_value` returns `Err(OptError::NonFiniteCost { .. })`.
    fn validate_value_errors_on_non_finite_scalar() {
        // Arrange
        let value = f64::NAN;

        // Act
        let result = validate_value(value);

        // Assert
        let err = result.expect_err("Non-finite value should be rejected");
        match err {
            OptError::NonFiniteCost { .. } => {}
            other => panic!("Expected NonFiniteCost, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `validate_hessian` rejects matrices whose dimensions do
    // not match the expected `dim × dim`.
    //
    // Given
    // -----
    // - `hessian` with shape (2, 3).
    // - `dim = 2`.
    //
    // Expect
    // ------
    // - `validate_hessian` returns `Err(OptError::HessianDimMismatch { .. })`.
    fn validate_hessian_errors_on_dimension_mismatch() {
        // Arrange
        let hessian: Hessian =
            Array2::from_shape_vec((2, 3), vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0])
                .expect("shape (2, 3) should be valid");
        let dim = 2;

        // Act
        let result = validate_hessian(&hessian, dim);

        // Assert
        let err = result.expect_err("Dimension mismatch should be rejected");
        match err {
            OptError::HessianDimMismatch { expected, found } => {
                assert_eq!(expected, dim);
                assert_eq!(found, (2, 3));
            }
            other => panic!("Expected HessianDimMismatch, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `validate_hessian` rejects matrices with non-finite
    // entries.
    //
    // Given
    // -----
    // - `hessian` with shape (2, 2) and a NaN entry at (0, 1).
    // - `dim = 2`.
    //
    // Expect
    // ------
    // - `validate_hessian` returns `Err(OptError::InvalidHessian { .. })`
    //   with `(row, col) = (0, 1)`.
    fn validate_hessian_errors_on_non_finite_entry() {
        // Arrange
        let hessian: Hessian = Array2::from_shape_vec((2, 2), vec![1.0_f64, f64::NAN, 0.0, 1.0])
            .expect("shape (2, 2) should be valid");
        let dim = 2;

        // Act
        let result = validate_hessian(&hessian, dim);

        // Assert
        let err = result.expect_err("Non-finite Hessian entry should be rejected");
        match err {
            OptError::InvalidHessian { row, col, .. } => {
                assert_eq!((row, col), (0, 1));
            }
            other => panic!("Expected InvalidHessian, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Confirm that `validate_hessian` accepts a square, finite Hessian of
    // the expected dimension.
    //
    // Given
    // -----
    // - `hessian` with shape (2, 2) and all finite entries.
    // - `dim = 2`.
    //
    // Expect
    // ------
    // - `validate_hessian` returns `Ok(())`.
    fn validate_hessian_accepts_valid_matrix() {
        // Arrange
        let hessian: Hessian = Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0, 0.0, 1.0])
            .expect("shape (2, 2) should be valid");
        let dim = 2;

        // Act
        let result = validate_hessian(&hessian, dim);

        // Assert
        assert!(result.is_ok());
    }
}
