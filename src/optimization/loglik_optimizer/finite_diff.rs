//! loglik_optimizer::finite_diff — finite-difference gradient and Hessian helpers.
//!
//! Purpose
//! -------
//! Provide finite-difference gradient and Hessian approximations around a
//! parameter vector, together with validation and symmetry cleanup, so that
//! the rest of the optimizer can request derivatives without depending
//! directly on the `finitediff` API.
//!
//! Key behaviors
//! -------------
//! - Compute forward-difference gradients with error capture and
//!   post-hoc validation via [`run_fd_diff`].
//! - Construct central-difference Hessians, falling back to forward
//!   differences when validation fails, via [`compute_hessian`].
//! - Enforce symmetry of Hessian matrices in-place using
//!   [`symmetrize_hess`] to prepare them for curvature checks and
//!   factorizations.
//!
//! Invariants & assumptions
//! ------------------------
//! - Parameter vectors, gradients, and Hessians are all represented as
//!   `ndarray` containers over `f64` (`Theta`, `Grad`, `Hessian`).
//! - Any error raised by the user-supplied objective during finite
//!   differencing is routed into the shared `closure_err` cell and
//!   treated as a hard failure for the gradient computation.
//! - Gradients and Hessians returned from this module are guaranteed to
//!   satisfy [`validate_grad`] and [`validate_hessian`] on the chosen
//!   finite-difference path.
//!
//! Conventions
//! -----------
//! - Finite differences are taken with respect to the unconstrained
//!   parameter vector `Theta`; any reparameterization is handled by
//!   higher layers.
//! - Central-difference Hessians are preferred; forward-difference is
//!   used only as a fallback when the central approximation fails
//!   validation.
//! - Domain errors are surfaced as [`OptError`] via `OptResult<T>`;
//!   Argmin’s [`Error`] is confined to the thin boundary where
//!   finite-difference closures are invoked.
//!
//! Downstream usage
//! ----------------
//! - Optimizer adapters call [`run_fd_diff`] when a [`LogLikelihood`]
//!   implementation does not provide an analytic gradient and the solver
//!   needs a finite-difference approximation.
//! - Second-order solvers call [`compute_hessian`] to obtain a validated,
//!   symmetrized Hessian suitable for curvature checks and matrix
//!   factorizations.
//! - This module is internal to the optimizer layer and is not intended
//!   to be invoked directly from Python bindings.
//!
//! Testing notes
//! -------------
//! - Unit tests cover both successful and failing paths for gradient and
//!   Hessian validation, including the central→forward Hessian fallback
//!   behavior.
//! - Integration tests for the full optimizer exercise these helpers
//!   implicitly when derivatives are requested via finite differences.
use crate::optimization::{
    errors::OptResult,
    loglik_optimizer::{
        Grad, Theta,
        types::Hessian,
        validation::{validate_grad, validate_hessian},
    },
};
use argmin::core::Error;
use finitediff::FiniteDiff;
use std::cell::RefCell;

/// run_fd_diff — forward-difference gradient with error capture and validation.
///
/// Purpose
/// -------
/// Compute a forward-difference approximation to the gradient of a scalar
/// objective at `theta`, while capturing any error raised inside the
/// evaluation closure and enforcing basic shape/finiteness invariants on
/// the resulting gradient.
///
/// Parameters
/// ----------
/// - `theta`: `&Theta`
///   Point in parameter space at which the gradient should be
///   approximated. The length of `theta` defines the expected gradient
///   dimension.
/// - `func`: `&G`
///   Objective function mapping `theta` to a scalar value. This is the
///   closure passed to `forward_diff`; it is assumed to route any
///   evaluation errors into `closure_err` and return `NaN` in that case.
/// - `closure_err`: `&RefCell<Option<Error>>`
///   Shared cell used to capture an [`argmin::core::Error`] raised inside
///   `func` while the finite-difference routine is running. This helper
///   clears the cell on entry and inspects it after the FD call.
///
/// Returns
/// -------
/// `OptResult<Grad>`
///   - `Ok(grad)` when finite differencing succeeds, no error was
///     captured in `closure_err`, and the resulting gradient passes
///     [`validate_grad`].
///   - `Err(e)` when either `func` signaled an error via `closure_err`
///     or the gradient fails validation.
///
/// Errors
/// ------
/// - `OptError` (via `impl From<Error> for OptError)`
///   Returned when `closure_err` contains an Argmin error captured from
///   inside `func`.
/// - `OptError::GradientDimMismatch`
///   Returned by [`validate_grad`] when the finite-difference gradient
///   length does not match `theta.len()`.
/// - `OptError::InvalidGradient`
///   Returned by [`validate_grad`] when any gradient element is NaN or
///   infinite.
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
/// - This helper assumes that the caller has wrapped the original
///   objective in a closure that writes any runtime error into
///   `closure_err` and returns `NaN`. If no error is written, the FD
///   path is assumed to have evaluated successfully.
/// - Only the first gradient element failing validation is reported by
///   [`validate_grad`].
///
/// Examples
/// --------
/// ```rust
/// # use std::cell::RefCell;
/// # use argmin::core::Error;
/// # use ndarray::Array1;
/// # use rust_timeseries::optimization::loglik_optimizer::{
/// #     Theta,
/// # };
/// # use rust_timeseries::optimization::loglik_optimizer::finite_diff::run_fd_diff;
/// let theta: Theta = Array1::from(vec![0.0_f64, 1.0]);
/// let closure_err: RefCell<Option<Error>> = RefCell::new(None);
///
/// // Simple quadratic objective with no internal error path.
/// let f = |x: &Theta| x.dot(x);
///
/// let grad = run_fd_diff(&theta, &f, &closure_err).unwrap();
/// assert_eq!(grad.len(), theta.len());
/// ```
pub fn run_fd_diff<G: Fn(&Theta) -> f64>(
    theta: &Theta, func: &G, closure_err: &RefCell<Option<Error>>,
) -> OptResult<Grad> {
    closure_err.replace(None);
    let fd_grad = theta.forward_diff(func);
    let dim = theta.len();
    if let Some(err) = closure_err.take() {
        return Err(err.into());
    }
    validate_grad(&fd_grad, dim)?;
    Ok(fd_grad)
}

/// compute_hessian — finite-difference Hessian with validation and symmetry.
///
/// Purpose
/// -------
/// Approximate the Hessian of a vector-valued gradient function at
/// `theta` using finite differences, preferring a central-difference
/// scheme and falling back to a forward-difference scheme when
/// validation fails. The resulting matrix is symmetrized in-place
/// before being returned.
///
/// Parameters
/// ----------
/// - `f`: `&F`
///   Gradient function mapping `theta` to a gradient vector `Grad`. This
///   is the function passed to the finite-difference Hessian routines,
///   which differentiate each gradient component numerically.
/// - `theta`: `&Theta`
///   Point in parameter space at which the Hessian is to be
///   approximated. The length of `theta` defines the expected Hessian
///   dimension `dim × dim`.
///
/// Returns
/// -------
/// `OptResult<Hessian>`
///   - `Ok(h)` containing a `dim × dim` Hessian matrix with all finite
///     entries, symmetrized via [`symmetrize_hess`].
///   - `Err(e)` when both central- and forward-difference paths fail
///     validation.
///
/// Errors
/// ------
/// - `OptError::HessianDimMismatch`
///   Returned when the forward-difference Hessian dimensions do not
///   match `theta.len()`.
/// - `OptError::InvalidHessian`
///   Returned when the forward-difference Hessian contains any NaN or
///   infinite entries.
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
/// - Central-difference Hessians are attempted first; any validation
///   failure (shape or finiteness) on the central approximation causes
///   an automatic fallback to a forward-difference Hessian.
/// - The central-difference validation error is intentionally discarded
///   to avoid coupling callers to the two-stage strategy; only the
///   forward-difference validation result is surfaced.
/// - Symmetrization is performed after validation to preserve the
///   original diagnostic information for `InvalidHessian` errors.
///
/// Examples
/// --------
/// ```rust
/// # use ndarray::Array1;
/// # use rust_timeseries::optimization::loglik_optimizer::{
/// #     Theta,
/// # };
/// # use rust_timeseries::optimization::loglik_optimizer::finite_diff::compute_hessian;
/// // Gradient of a simple quadratic: g(θ) = 2θ.
/// let grad_fn = |theta: &Theta| theta.mapv(|x| 2.0 * x);
///
/// let theta: Theta = Array1::from(vec![1.0_f64, 2.0]);
/// let hess = compute_hessian(&grad_fn, &theta).unwrap();
/// assert_eq!(hess.shape(), &[2, 2]);
/// ```
pub fn compute_hessian<F: Fn(&Theta) -> Grad>(f: &F, theta: &Theta) -> OptResult<Hessian> {
    let dim = theta.len();
    let mut cent_hess = theta.central_hessian(f);
    match validate_hessian(&cent_hess, dim) {
        Ok(_) => {
            symmetrize_hess(&mut cent_hess);
            Ok(cent_hess)
        }
        Err(_) => {
            let mut forward_hess = theta.forward_hessian(f);
            validate_hessian(&forward_hess, dim)?;
            symmetrize_hess(&mut forward_hess);
            Ok(forward_hess)
        }
    }
}

// ---- Helper methods ----

/// symmetrize_hess — enforce symmetry of a Hessian matrix in-place.
///
/// Purpose
/// -------
/// Replace each off-diagonal pair `(i, j)` / `(j, i)` with their average
/// so that the resulting matrix is numerically symmetric while leaving
/// the diagonal entries unchanged.
///
/// Parameters
/// ----------
/// - `hess`: `&mut Hessian`
///   Dense Hessian matrix to be symmetrized. Must be square; this
///   function does not check dimensions and assumes the caller passes a
///   valid `dim × dim` matrix.
///
/// Returns
/// -------
/// `()`
///   This function mutates `hess` in-place and does not allocate or
///   return a new matrix.
///
/// Errors
/// ------
/// - Never returns an error.
///
/// Panics
/// ------
/// - Never panics, assuming `hess` has consistent row/column indexing
///   (i.e., is a valid `ndarray::Array2`).
///
/// Safety
/// ------
/// - No `unsafe` code is used.
///
/// Notes
/// -----
/// - Symmetrization is performed over the strict lower triangle; the
///   diagonal is left untouched.
/// - This helper is called only after a Hessian has passed
///   [`validate_hessian`], so it does not perform its own finiteness or
///   shape checks.
///
/// Examples
/// --------
/// ```rust
/// # use ndarray::Array2;
/// # use rust_timeseries::optimization::loglik_optimizer::types::Hessian;
/// # use rust_timeseries::optimization::loglik_optimizer::finite_diff::symmetrize_hess;
/// let mut h: Hessian = Array2::from_shape_vec(
///     (2, 2),
///     vec![1.0_f64, 2.0,
///          0.0,     3.0],
/// ).unwrap();
///
/// symmetrize_hess(&mut h);
/// assert_eq!(h[[0, 1]], h[[1, 0]]);
/// ```
fn symmetrize_hess(hess: &mut Hessian) {
    for i in 0..hess.nrows() {
        for j in 0..i {
            let avg = 0.5 * (hess[[i, j]] + hess[[j, i]]);
            hess[[i, j]] = avg;
            hess[[j, i]] = avg;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::errors::OptError;
    use argmin::core::ArgminError;
    use ndarray::{Array1, Array2};

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Forward-difference gradient computation with and without closure errors.
    // - Validation failures for non-finite gradients.
    // - Finite-difference Hessian construction, symmetry, and validation.
    // - In-place symmetrization behavior for Hessian matrices.
    //
    // They intentionally DO NOT cover:
    // - End-to-end optimizer behavior (handled in higher-level integration tests).
    // - Specific LogLikelihood model implementations or Python bindings.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Verify that `run_fd_diff` returns a valid gradient for a simple quadratic
    // objective with no internal error path.
    //
    // Given
    // -----
    // - A parameter vector `theta` in ℝ².
    // - An objective `f(theta) = thetaᵀ theta` with no error side channel.
    //
    // Expect
    // ------
    // - `run_fd_diff` returns `Ok(grad)` with `grad.len() == theta.len()`.
    // - All gradient entries are finite.
    fn run_fd_diff_quadratic_returns_valid_gradient() {
        // Arrange
        let theta: Theta = Array1::from(vec![0.0_f64, 1.0]);
        let closure_err: RefCell<Option<Error>> = RefCell::new(None);
        let f = |x: &Theta| x.dot(x);

        // Act
        let result = run_fd_diff(&theta, &f, &closure_err);

        // Assert
        let grad = result.expect("Gradient for quadratic should be computed successfully");
        assert_eq!(grad.len(), theta.len());
        assert!(grad.iter().all(|v| v.is_finite()));
    }

    #[test]
    // Purpose
    // -------
    // Ensure that `run_fd_diff` propagates an error captured in `closure_err`
    // as an `OptError` via the `From<Error>` implementation.
    //
    // Given
    // -----
    // - A parameter vector `theta` in ℝ¹.
    // - An objective closure that writes an `ArgminError` into `closure_err`
    //   and returns `NaN`.
    //
    // Expect
    // ------
    // - `run_fd_diff` returns `Err(e)` rather than a gradient.
    // - The error is mapped into an appropriate `OptError` variant.
    fn run_fd_diff_closure_error_is_propagated() {
        // Arrange
        let theta: Theta = Array1::from(vec![1.0_f64]);
        let closure_err: RefCell<Option<Error>> = RefCell::new(None);

        let f = |_: &Theta| {
            let argmin_err = ArgminError::NotImplemented { text: "fd test".to_string() };
            // Store the error in the shared cell and return NaN.
            closure_err.replace(Some(argmin_err.into()));
            f64::NAN
        };

        // Act
        let result = run_fd_diff(&theta, &f, &closure_err);

        // Assert
        let err = result.expect_err("Error in closure should cause run_fd_diff to fail");
        // We don't assert the exact variant here, only that it is an OptError.
        // If you want to be stricter, pattern-match on OptError::NotImplemented.
        match err {
            OptError::NotImplemented { .. } | OptError::BackendError { .. } => {}
            other => panic!("Unexpected OptError variant from closure error: {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Confirm that `run_fd_diff` returns an error when the finite-difference
    // gradient contains non-finite entries.
    //
    // Given
    // -----
    // - A parameter vector `theta` in ℝ².
    // - An objective that always returns `NaN`, causing the FD gradient to be
    //   filled with `NaN`.
    //
    // Expect
    // ------
    // - `run_fd_diff` returns `Err(OptError::InvalidGradient { .. })`.
    fn run_fd_diff_non_finite_gradient_yields_invalidgradient_error() {
        // Arrange
        let theta: Theta = Array1::from(vec![0.0_f64, 1.0]);
        let closure_err: RefCell<Option<Error>> = RefCell::new(None);
        let f = |_x: &Theta| f64::NAN;

        // Act
        let result = run_fd_diff(&theta, &f, &closure_err);

        // Assert
        let err = result.expect_err("Non-finite gradient should cause an error");
        match err {
            OptError::InvalidGradient { .. } => {}
            other => panic!("Expected InvalidGradient, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `compute_hessian` produces a finite, symmetric Hessian for a
    // simple quadratic model where the gradient is linear.
    //
    // Given
    // -----
    // - A parameter vector `theta` in ℝ².
    // - A gradient function `g(theta) = 2 * theta` corresponding to
    //   `f(theta) = ||theta||²`.
    //
    // Expect
    // ------
    // - `compute_hessian` returns `Ok(hess)` with shape (2, 2).
    // - `hess` is symmetric and has finite entries.
    fn compute_hessian_quadratic_returns_symmetric_matrix() {
        // Arrange
        let theta: Theta = Array1::from(vec![1.0_f64, 2.0]);
        let grad_fn = |theta: &Theta| theta.mapv(|x| 2.0 * x);

        // Act
        let hess = compute_hessian(&grad_fn, &theta)
            .expect("Hessian for quadratic gradient should be computed successfully");

        // Assert
        assert_eq!(hess.shape(), &[2, 2]);
        // Symmetry check
        assert!((hess[[0, 1]] - hess[[1, 0]]).abs() < 1e-10);
        assert!(hess.iter().all(|v| v.is_finite()));
    }

    #[test]
    // Purpose
    // -------
    // Ensure that `compute_hessian` surfaces a validation error when both the
    // central- and forward-difference Hessians contain non-finite entries.
    //
    // Given
    // -----
    // - A parameter vector `theta` in ℝ¹.
    // - A gradient function that returns `NaN` in its single component.
    //
    // Expect
    // ------
    // - `compute_hessian` returns `Err(OptError::InvalidHessian { .. })`.
    fn compute_hessian_non_finite_entries_yield_invalidhessian_error() {
        // Arrange
        let theta: Theta = Array1::from(vec![0.0_f64]);
        let grad_fn = |_theta: &Theta| Array1::from(vec![f64::NAN]);

        // Act
        let result = compute_hessian(&grad_fn, &theta);

        // Assert
        let err = result.expect_err("Non-finite Hessian entries should cause an error");
        match err {
            OptError::InvalidHessian { .. } => {}
            other => panic!("Expected InvalidHessian, got {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `symmetrize_hess` makes a matrix numerically symmetric by
    // averaging each off-diagonal pair.
    //
    // Given
    // -----
    // - A 2x2 matrix with unequal off-diagonal entries.
    //
    // Expect
    // ------
    // - After calling `symmetrize_hess`, the off-diagonal entries are equal to
    //   their average and the diagonal remains unchanged.
    fn symmetrize_hess_makes_matrix_symmetric() {
        // Arrange
        let mut h: Hessian = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 0.0, 3.0]).unwrap();

        let before_diag = (h[[0, 0]], h[[1, 1]]);
        let expected_avg = 0.5 * (h[[0, 1]] + h[[1, 0]]);

        // Act
        super::symmetrize_hess(&mut h);

        // Assert
        assert_eq!(h[[0, 0]], before_diag.0);
        assert_eq!(h[[1, 1]], before_diag.1);
        assert!((h[[0, 1]] - expected_avg).abs() < 1e-12);
        assert!((h[[1, 0]] - expected_avg).abs() < 1e-12);
        assert_eq!(h[[0, 1]], h[[1, 0]]);
    }
}
