//! Finite-difference utilities: gradient & Hessian with validation and symmetry cleanup.
//!
//! What this module provides
//! - Forward-difference gradient with error capture.
//! - Central/forward finite-difference Hessian with validation.
//! - In-place symmetrization: H ← (H + Hᵀ)/2.
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

/// Compute a forward-difference gradient of `func` at `theta`, with error capture.
///
/// The FD closure can’t return `Result`, so any error raised by `func` is
/// stored into `closure_err` and the closure returns `NaN`. This helper:
/// - clears `closure_err`,
/// - performs `forward_diff`,
/// - if an error was captured, returns it as `Err`,
/// - validates the resulting gradient,
/// - if validation succeeds, returns the gradient as `Ok(grad)`.
///
/// # Errors
/// Returns any error captured during evaluation of `func` inside the FD routine
/// or by validation of the resulting gradient.
pub fn run_fd_diff<G: Fn(&Theta) -> f64>(
    theta: &Theta, func: &G, closure_err: &RefCell<Option<Error>>,
) -> Result<Grad, Error> {
    closure_err.replace(None);
    let fd_grad = theta.forward_diff(func);
    let dim = theta.len();
    if let Some(err) = closure_err.take() {
        return Err(err);
    }
    validate_grad(&fd_grad, dim)?;
    Ok(fd_grad)
}

/// Finite-difference Hessian at `theta` with validation and symmetry cleanup.
///
/// Strategy:
/// - Try a **central** FD Hessian first.
/// - If central FD fails validation, fall back to **forward** FD.
/// - Enforce symmetry by averaging `H` with `Hᵀ`.
///
/// Returns the symmetrized Hessian (`ndarray::Array2<f64>`).
///
/// # Errors
/// Propagates any error from `validate_hessian` on the forward-difference path.
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

/// In-place symmetrization of an `ndarray` Hessian.
///
/// For every off-diagonal pair `(i, j)` and `(j, i)`, replaces both with
/// the average `0.5 * (H[i, j] + H[j, i])`. The diagonal is left unchanged.
///
/// This removes asymmetries introduced by numerical differentiation and
/// prepares the matrix for PD/PSD repairs and factorizations.
fn symmetrize_hess(hess: &mut Hessian) -> () {
    for i in 0..hess.nrows() {
        for j in 0..i {
            let avg = 0.5 * (hess[[i, j]] + hess[[j, i]]);
            hess[[i, j]] = avg;
            hess[[j, i]] = avg;
        }
    }
}
