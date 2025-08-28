//! Numerical stability utilities.
//!
//! Provides safe implementations of common nonlinear transforms
//! that are prone to overflow/underflow in naïve form.
//! The functions here follow guarded strategies similar to those
//! in major ML libraries (e.g. PyTorch, TensorFlow), using explicit
//! cutoffs (`x > 20.0`) to keep `f64` arithmetic in a well-conditioned regime.
//!
//! # Provided items
//! - [`STATIONARITY_MARGIN`]: a small ε buffer (default 1e-6).  
//!   Used to enforce strict inequalities in stability constraints
//!   (e.g. ∑α + ∑β < 1 in ACD models).
//! - [`safe_softplus(x)`]: stable version of `ln(1 + exp(x))`,
//!   mapping ℝ → (0, ∞) without overflow.
//! - [`safe_softplus_inv(x)`]: inverse of softplus, mapping
//!   (0, ∞) → ℝ without catastrophic cancellation.
//!
//! # Rationale
//! These transforms are building blocks in optimization and
//! probabilistic modeling whenever parameters must be kept
//! strictly positive or constrained away from unstable boundaries.

/// Safety margin for strict stationarity in ACD models.
///
/// In an ACD(p, q), the stability condition requires
///   sum(alpha) + sum(beta) < 1.
/// This margin enforces the inequality *strictly* by reserving a small
/// buffer (default = 1e-6). Practically, the recursion always runs inside
/// the stable region, avoiding borderline cases that can cause blow-ups
/// in likelihood evaluation.
pub const STATIONARITY_MARGIN: f64 = 1e-6;

/// Numerically stable softplus: `softplus(x) = ln(1 + exp(x))`.
///
/// Computes softplus without overflow for large positive `x` and
/// with good precision for large negative `x`. This implementation
/// uses a simple piecewise guard:
///
/// - For sufficiently large `x`, `softplus(x) ≈ x + ln1p(exp(-x)) ≈ x`.
/// - Otherwise, it falls back to `ln1p(exp(x))`.
///
/// The cutoff used here (`x > 20.0`) is a practical threshold that
/// keeps the calculation in a well-conditioned regime for `f64`
/// (similar to the strategy used in common ML libraries like PyTorch).
///
/// # Parameters
/// - `x`: real input
///
/// # Returns
/// - `softplus(x)` as `f64`.
pub fn safe_softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// Stable inverse of softplus on `(0, ∞)`: solves for `t` in
/// `softplus(t) = x`, returning `t = ln(exp(x) - 1)`.
///
/// Direct evaluation of `ln(exp(x) - 1)` can overflow or lose precision.
/// This implementation mirrors the guarded strategy of `safe_softplus`:
///
/// - For sufficiently large `x`, `exp(-x)` is tiny and
///   `ln(exp(x) - 1) ≈ x + ln(1 - exp(-x)) ≈ x`.
/// - Otherwise, it uses `ln(expm1(x))`.
///
/// The cutoff (`x > 20.0`) is chosen for numerical robustness with `f64`.
///
/// # Parameters
/// - `x`: a positive real (the softplus output), must be finite and `> 0`.
///
/// # Returns
/// - `t` such that `softplus(t) = x`.
pub fn safe_softplus_inv(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp_m1().ln() }
}
