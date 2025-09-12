//! Validation helpers for log-likelihood optimization.
//!
//! This module centralizes common consistency checks used across the
//! optimizer interface:
//!
//! - **Tolerance checks**: [`verify_tol_grad`], [`verify_tol_cost`] ensure
//!   numeric tolerances are finite and strictly positive when provided.
//! - **Gradient validation**: [`validate_grad`] enforces correct dimension
//!   and finite entries.
//! - **Parameter estimates**: [`validate_theta_hat`] ensures a candidate
//!   `theta_hat` exists and contains only finite values.
//! - **Objective values**: [`validate_value`] checks log-likelihood outputs
//!   for finiteness.
//!
//! These helpers standardize error reporting by returning domain-specific
//! [`OptError`] variants, making higher-level code more uniform and easier
//! to debug.
use crate::optimization::{
    errors::{OptError, OptResult},
    loglik_optimizer::{Grad, Theta, types::Hessian},
};

/// Validate the optional gradient‐norm tolerance.
///
/// - Accepts `None` (no stopping rule on gradient).
/// - If `Some`, the value must be **finite** and **strictly positive**.
///
/// # Errors
/// Returns [`OptError::InvalidTolGrad`] if the value is non-finite or ≤ 0.0.
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

/// Validate the optional cost‐change tolerance (for convergence).
///
/// - Accepts `None` (no stopping rule on cost change).
/// - If `Some`, the value must be **finite** and **strictly positive**.
///
/// # Errors
/// Returns [`OptError::InvalidTolCost`] if the value is non-finite or ≤ 0.0.
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

/// Validate a gradient vector against dimension and finiteness.
///
/// Checks:
/// - `grad.len() == dim`
/// - every element is finite (`NaN` or `±∞` are rejected)
///
/// # Errors
/// - [`OptError::GradientDimMismatch`] if length does not match `dim`.
/// - [`OptError::InvalidGradient`] with the index/value/reason of the first
///   offending element.
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

/// Validate and unwrap an estimated parameter vector (`theta_hat`).
///
/// Accepts only a present vector with all **finite** entries.
///
/// # Returns
/// The owned `Theta` if valid.
///
/// # Errors
/// - [`OptError::MissingThetaHat`] if no vector was provided.
/// - [`OptError::InvalidThetaHat`] if any element is non-finite.
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

/// Validate that a scalar log-likelihood value is finite.
///
/// Negative values are fine as long as they are finite.
///
/// # Errors
/// Returns [`OptError::NonFiniteCost`] if the value is `NaN` or infinite.
pub fn validate_value(value: f64) -> OptResult<()> {
    if !value.is_finite() {
        return Err(OptError::NonFiniteCost { value });
    }
    Ok(())
}

/// Validate the shape and entries of a Hessian matrix.
///
/// # Checks
/// 1. Matrix dimensions must equal `dim × dim`.
/// 2. All entries must be finite (no NaN or ±∞).
///
/// # Arguments
/// - `hessian`: Hessian matrix to validate.
/// - `dim`: expected dimension (both rows and columns).
///
/// # Returns
/// - `Ok(())` if the Hessian passes all checks.
///
/// # Errors
/// - [`OptError::HessianDimMismatch`] if dimensions do not match `dim`.
/// - [`OptError::InvalidHessian`] if any entry is non-finite, with offending
///   row/col indices and value.
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
