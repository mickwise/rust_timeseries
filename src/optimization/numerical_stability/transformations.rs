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
use ndarray::{ArrayView1, ArrayViewMut1, Zip, s};

/// Safety margin for strict stationarity in ACD models.
///
/// In an ACD(p, q), the stability condition requires
///   sum(alpha) + sum(beta) < 1.
/// This margin enforces the inequality *strictly* by reserving a small
/// buffer (default = 1e-6). Practically, the recursion always runs inside
/// the stable region, avoiding borderline cases that can cause blow-ups
/// in likelihood evaluation.
pub const STATIONARITY_MARGIN: f64 = 1e-6;

/// Lower clamp used in logit/softmax transforms.
///
/// Prevents log/exp underflow when converting extremely small probabilities or
/// weights during the optimizer-space ↔ model-space mapping.
pub const LOGIT_EPS: f64 = 1e-15;

/// Eigenvalue threshold for treating tiny/negative curvatures as zero.
///
/// Any eigenvalue `λ <= EIGEN_EPS` is considered numerically nonpositive and is
/// excluded from the variance calculation (equivalently, we use the Moore–Penrose
/// pseudoinverse along those directions). This makes the covariance positive
/// semidefinite and conservative under weak identification.
pub const EIGEN_EPS: f64 = 1e-10;

/// Small clamp to prevent log(0) or division by zero.
pub const GENERAL_TOL: f64 = 1e-10;

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

/// Fill α and β from the logits slice using a three-pass, max-shift softmax.
///
/// Expects `theta.len() == q + p`, corresponding to `[α logits, β logits]`.
/// Performs:
/// 1) `m = max(theta)`
/// 2) `den = Σ exp(theta[i] − m)`
/// 3) Writes:
///    - `α[i] = exp(theta[i] − m) / den * (1 − STATIONARITY_MARGIN)` for `i in 0..q`
///    - `β[j] = exp(theta[q + j] − m) / den * (1 − STATIONARITY_MARGIN)` for `j in 0..p`
///
/// Implementation is **allocation-free** and uses `ndarray::Zip` to write
/// directly into the α/β buffers.
pub fn safe_softmax<'a>(
    alpha: ArrayViewMut1<'a, f64>, beta: ArrayViewMut1<'a, f64>, theta: &ArrayView1<f64>, p: usize,
    q: usize,
) -> () {
    let max_x = theta.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let sum_exp_x: f64 = theta.iter().map(|&v| (v - max_x).exp()).sum();
    let scale = 1.0 - STATIONARITY_MARGIN;

    Zip::from(alpha)
        .and(&theta.slice(ndarray::s![0..q]))
        .for_each(|a, &v| *a = ((v - max_x).exp() / sum_exp_x) * scale);

    Zip::from(beta)
        .and(&theta.slice(ndarray::s![q..q + p]))
        .for_each(|b, &v| *b = ((v - max_x).exp() / sum_exp_x) * scale);
}

/// Compute the Jacobian–vector product of the softmax transform for α/β logits.
///
/// Expects `theta.len() == q + p`, corresponding to `[α logits, β logits]`.
/// Given current softmax weights `(α, β)` and their logits in `theta`, this
/// routine overwrites `theta` in place with the partial derivatives
/// ∂(α,β)/∂logits multiplied by the logits themselves.
///
/// Concretely:
/// - Let `num = α·θ[0..q] + β·θ[q..q+p]`.
/// - Let `c = num / (1 − STATIONARITY_MARGIN)`.
/// - For each `i in 0..q`, write:
///   `θ[i] = α[i] * (θ[i] − c)`.
/// - For each `j in 0..p`, write:
///   `θ[q+j] = β[j] * (θ[q+j] − c)`.
///
/// This corresponds to the row-wise structure of the softmax Jacobian:
/// `∂π_i/∂logit_k = π_i (δ_{ik} − π_k)`, scaled by `(1 − STATIONARITY_MARGIN)`.
///
/// # Side effects
/// - Mutates the provided `theta` slice in place with the derivative values.
/// - No heap allocations.
pub fn safe_softmax_deriv(
    alpha: &ArrayViewMut1<f64>, beta: &ArrayViewMut1<f64>, theta: &mut ArrayViewMut1<f64>,
    p: usize, q: usize,
) -> () {
    let alpha_slice = &theta.slice(s![0..q]);
    let beta_slice = &theta.slice(s![q..q + p]);
    let numerator = alpha.dot(alpha_slice) + beta.dot(beta_slice);
    let c = numerator / (1.0 - STATIONARITY_MARGIN);
    for i in 0..q {
        theta[i] = alpha[i] * (theta[i] - c);
    }
    for j in 0..p {
        theta[q + j] = beta[j] * (theta[q + j] - c);
    }
}

/// Numerically stable logistic function: derivative of softplus.
///
/// Computes `σ(x) = 1 / (1 + exp(-x))` with guarded branches to avoid
/// overflow/underflow:
///
/// - For `x >> 0`, `σ(x) → 1`. Direct evaluation of `exp(-x)` underflows,
///   so we return `1.0` immediately when `x > 20.0`.
/// - For `x ≥ 0`, evaluates `1 / (1 + exp(-x))` safely.
/// - For `x < 0`, uses the equivalent form `exp(x) / (1 + exp(x))`
///   to avoid overflow in `exp(-x)`.
///
/// # Parameters
/// - `x`: real-valued input, must be finite.
///
/// # Returns
/// - `σ(x)` in `(0, 1)`, the logistic value.
pub fn safe_logistic(x: f64) -> f64 {
    if x > 20.0 {
        return 1.0;
    } else if x >= 0.0 {
        let exp_neg_x = (-x).exp();
        1.0 / (1.0 + exp_neg_x)
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}
