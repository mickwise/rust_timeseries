//! transformations — numerically stable parameter transforms.
//!
//! Purpose
//! -------
//! Provide a small set of numerically robust nonlinear transforms and
//! constants used by the duration and optimization layers. These helpers
//! keep `f64` arithmetic in well-conditioned regimes and support the
//! positivity / stationarity constraints required by ACD-style models.
//!
//! Key behaviors
//! -------------
//! - Offer stable implementations of softplus, its inverse, and a guarded
//!   logistic used for unconstrained ↔ constrained parameter mappings.
//! - Map unconstrained logits into ACD coefficient vectors `(α, β)` with an
//!   explicit stationarity slack term via a max-shift softmax.
//! - Provide a Jacobian–vector product for the softmax transform used when
//!   propagating gradients between optimizer space and model space.
//! - Centralize small numerical tolerances (stationarity margin, eigenvalue
//!   thresholds, general epsilons) so downstream modules share consistent
//!   numerical guards.
//!
//! Invariants & assumptions
//! ------------------------
//! - Callers are responsible for enforcing domain constraints (positivity,
//!   finiteness, length checks).
//! - `safe_softmax` and `safe_softmax_deriv` assume that:
//!   - `alpha.len() + beta.len() == theta.len()`,
//!   - `theta` contains logits in unconstrained optimizer space, and
//!   - `alpha` and `beta` have already been sized consistently with the
//!     ACD shape `(p, q)`.
//! - The ACD stationarity constraint is enforced as
//!   `sum(α) + sum(β) + slack ≈ 1 − STATIONARITY_MARGIN` using helpers in
//!   the duration core; this module only supplies the numeric mapping.
//!
//! Conventions
//! -----------
//! - All transforms avoid heap allocation where practical and operate
//!   directly on `ndarray` views (`ArrayView1`, `ArrayViewMut1`).
//! - `STATIONARITY_MARGIN` is treated as a global ε that keeps the recursion
//!   away from the unit-root boundary; it is not tuned per model.
//! - `EIGEN_EPS` is intended for eigenvalue-based regularization in covariance
//!   / Hessian routines (e.g., pseudoinverses).
//! - These helpers do not log, perform I/O, or handle error reporting;
//!   callers surface validation failures as domain-specific error types.
//!
//! Downstream usage
//! ----------------
//! - Duration workspaces use `safe_softplus` to map unconstrained scalars
//!   into strictly positive parameters (e.g., `ω`).
//! - ACD parameter mappings use `safe_softmax` to convert logits into `(α, β)`
//!   and a slack term consistent with `STATIONARITY_MARGIN`.
//! - Gradient propagation between optimizer space and model space uses
//!   `safe_softmax_deriv` to apply the softmax Jacobian to a vector without
//!   constructing the full Jacobian.
//! - Inference and optimizer components use `LOGIT_EPS`, `EIGEN_EPS`, and
//!   `GENERAL_TOL` as shared small constants for clamping and regularization.
//!
//! Testing notes
//! -------------
//! - Unit tests for this module:
//!   - Compare these stable transforms to naïve implementations in regimes
//!     where the naïve formulae are safe and assert approximate agreement.
//!   - Exercise extreme inputs (large positive/negative arguments) to confirm
//!     monotonicity and absence of NaNs / infinities.
//!   - Verify that `safe_softmax` produces non-negative `(α, β)` whose sum,
//!     together with slack, is close to `1 − STATIONARITY_MARGIN` for small
//!     test systems.
//!   - Check that `safe_softmax_deriv` matches a finite-difference Jacobian–
//!     vector product in low dimensions.
//! - Higher-level invariants (full ACD stationarity, parameter validation,
//!   optimizer behavior) are tested in the duration core and optimization
//!   modules, not here.

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, Zip, s};

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

/// `safe_softplus` — numerically stable `ln(1 + exp(x))`.
///
/// Purpose
/// -------
/// Compute the softplus transform without overflow for large positive `x`
/// and with good precision for large negative `x`, using a simple guarded
/// branch.
///
/// Parameters
/// ----------
/// - `x`: `f64`
///   Real-valued input. Callers are expected to pass finite values.
///
/// Returns
/// -------
/// `f64`
///   The value `softplus(x) = ln(1 + exp(x))`, finite for all finite `x`.
///
/// Errors
/// ------
/// - Does not return errors; callers are responsible for checking for NaNs
///   if they pass non-finite inputs.
///
/// Panics
/// ------
/// - Does not panic under normal usage.
///
/// Safety
/// ------
/// - No `unsafe` code is used.
///
/// Notes
/// -----
/// - For sufficiently large `x`, the function returns approximately `x`,
///   as the correction term `ln(1 + exp(-x))` is negligible at `f64`
///   precision.
/// - For other values, it evaluates `exp(x).ln_1p()`, which is well-behaved
///   for moderate and negative arguments.
///
/// Examples
/// --------
/// ```rust
/// use rust_timeseries::optimization::numerical_stability::transformations::safe_softplus;
///
/// let y = safe_softplus(0.0_f64);
/// assert!((y - (1.0_f64).ln_1p()).abs() < 1e-12);
/// ```
pub fn safe_softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// `safe_softplus_inv` — inverse of softplus on `(0, ∞)`.
///
/// Purpose
/// -------
/// Given `x = softplus(t)`, recover `t` via a guarded version of
/// `ln(exp(x) - 1)` that avoids overflow and precision loss.
///
/// Parameters
/// ----------
/// - `x`: `f64`
///   Positive softplus output, expected to satisfy `x > 0` and be finite.
///
/// Returns
/// -------
/// `f64`
///   A real value `t` such that `softplus(t) ≈ x`.
///
/// Errors
/// ------
/// - Does not return errors; callers should avoid passing non-positive
///   or non-finite `x`.
///
/// Panics
/// ------
/// - Does not panic under normal usage.
///
/// Safety
/// ------
/// - No `unsafe` code is used.
///
/// Notes
/// -----
/// - For sufficiently large `x`, the function returns approximately `x`,
///   since `ln(exp(x) - 1)` and `x` coincide numerically.
/// - For moderate `x`, it uses `x.expm1().ln()`, which is more precise
///   than `ln(exp(x) - 1)` for small positive arguments.
///
/// Examples
/// --------
/// ```rust
/// use rust_timeseries::optimization::numerical_stability::transformations::{
///     safe_softplus, safe_softplus_inv,
/// };
///
/// let t = 0.7_f64;
/// let x = safe_softplus(t);
/// let t_back = safe_softplus_inv(x);
/// assert!((t - t_back).abs() < 1e-8);
/// ```
pub fn safe_softplus_inv(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp_m1().ln() }
}

/// `safe_softmax` — logits → `(α, β)` mapping with stationarity slack.
///
/// Purpose
/// -------
/// Map an unconstrained logits vector into two coefficient blocks `(α, β)`
/// whose total mass, together with a slack term, is compatible with the
/// ACD stationarity margin.
///
/// Parameters
/// ----------
/// - `alpha`: `ArrayViewMut1<'a, f64>`
///   Mutable view into the α buffer of length `q`. On return, contains
///   non-negative weights derived from the first `q` logits.
/// - `beta`: `ArrayViewMut1<'a, f64>`
///   Mutable view into the β buffer of length `p`. On return, contains
///   non-negative weights derived from the remaining `p` logits.
/// - `theta`: `&ArrayView1<f64>`
///   Read-only view of logits of length `q + p`. Callers are responsible
///   for ensuring `theta.len() == alpha.len() + beta.len()`.
///
/// Returns
/// -------
/// `f64`
///   The slack mass implied by the transform, such that in downstream
///   validation `sum(α) + sum(β) + slack` is checked against
///   `1 − STATIONARITY_MARGIN`.
///
/// Errors
/// ------
/// - Does not return errors. Length mismatches and invalid values are
///   handled by callers (e.g., in `WorkSpace::update` and validation
///   helpers).
///
/// Panics
/// ------
/// - Does not panic under normal usage.
///
/// Safety
/// ------
/// - No `unsafe` code is used. Callers must provide correctly sized and
///   initialized views.
///
/// Notes
/// -----
/// - Uses a max-shift softmax over `theta` to avoid overflow in the
///   exponentials, then scales by `1 − STATIONARITY_MARGIN`.
/// - The slack value corresponds to the residual mass needed to satisfy
///   the stationarity margin when combined with `α` and `β`.
///
/// Examples
/// --------
/// ```rust
/// use ndarray::{Array1, array};
/// use rust_timeseries::optimization::numerical_stability::transformations::{
///     safe_softmax, STATIONARITY_MARGIN,
/// };
///
/// let logits = Array1::from(vec![0.0, 1.0, -1.0]);
/// let mut alpha = array![0.0_f64]; // q = 1
/// let mut beta  = array![0.0_f64, 0.0_f64]; // p = 2
///
/// let slack = {
///     let theta_view = logits.view();
///     let alpha_view = alpha.view_mut();
///     let beta_view  = beta.view_mut();
///     safe_softmax(alpha_view, beta_view, &theta_view)
/// };
///
/// let total = alpha.sum() + beta.sum() + slack;
/// // downstream validation checks that `total` is close to `1.0 - STATIONARITY_MARGIN`
/// assert!(total <= 1.0);
/// ```
pub fn safe_softmax<'a>(
    alpha: ArrayViewMut1<'a, f64>, beta: ArrayViewMut1<'a, f64>, theta: &ArrayView1<f64>,
) -> f64 {
    let q = alpha.len();
    let p = beta.len();
    let max_x = theta.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let sum_exp_x: f64 = (-max_x).exp() + theta.iter().map(|&v| (v - max_x).exp()).sum::<f64>();
    let scale = 1.0 - STATIONARITY_MARGIN;

    Zip::from(alpha)
        .and(&theta.slice(ndarray::s![0..q]))
        .for_each(|a, &v| *a = ((v - max_x).exp() / sum_exp_x) * scale);

    Zip::from(beta)
        .and(&theta.slice(ndarray::s![q..q + p]))
        .for_each(|b, &v| *b = ((v - max_x).exp() / sum_exp_x) * scale);
    scale * (-max_x).exp() / sum_exp_x
}

/// `safe_softmax_deriv` — Jacobian–vector product for the `(α, β)` softmax.
///
/// Purpose
/// -------
/// Given current softmax weights `(α, β)` and their logits in `theta`,
/// overwrite `theta` with the Jacobian–vector product `J v`, where `J` is
/// the Jacobian of the stationarity-scaled softmax and `v` is taken to be
/// the current logits.
///
/// Parameters
/// ----------
/// - `alpha`: `&ArrayViewMut1<f64>`
///   Read-only view into the α weights of length `q`. Assumed to be the
///   result of a prior `safe_softmax` call using the same logits layout.
/// - `beta`: `&ArrayViewMut1<f64>`
///   Read-only view into the β weights of length `p`.
/// - `theta`: `&mut ArrayViewMut1<f64>`
///   In/out view of logits of length `q + p`. On entry, contains the logits
///   used to compute `(α, β)`; on return, contains the Jacobian–vector
///   product.
///
/// Returns
/// -------
/// `()`
///   The result is written in place into `theta`.
///
/// Errors
/// ------
/// - Does not return errors. Callers are responsible for ensuring that
///   `alpha`, `beta`, and `theta` are consistent and derived from the same
///   softmax mapping.
///
/// Panics
/// ------
/// - Does not panic under normal usage.
///
/// Safety
/// ------
/// - No `unsafe` code is used. This function assumes that `(α, β)`
///   correspond to the current contents of `theta`.
///
/// Notes
/// -----
/// - The implementation follows the standard softmax Jacobian structure for
///   probabilities `π_i`, scaled to account for `STATIONARITY_MARGIN`.
/// - This is intended as a low-level building block for gradient propagation
///   in parameter mappings, not as part of the public API surface.
///
/// Examples
/// --------
/// ```rust
/// use ndarray::Array1;
/// use rust_timeseries::optimization::numerical_stability::transformations::{
///     safe_softmax, safe_softmax_deriv,
/// };
///
/// // Simple example: q = 1, p = 1.
/// let logits = Array1::from(vec![0.0, 1.0]);
/// let mut alpha = Array1::zeros(1);
/// let mut beta  = Array1::zeros(1);
/// let mut theta = logits.clone();
///
/// {
///     let theta_view = theta.view();
///     let alpha_view = alpha.view_mut();
///     let beta_view  = beta.view_mut();
///     let _slack = safe_softmax(alpha_view, beta_view, &theta_view);
/// }
///
/// {
///     let alpha_view = alpha.view_mut();
///     let beta_view  = beta.view_mut();
///     let mut theta_view = theta.view_mut();
///     safe_softmax_deriv(&alpha_view, &beta_view, &mut theta_view);
/// }
///
/// // `theta` now holds Jv for this small system.
/// ```
pub fn safe_softmax_deriv(
    alpha: &ArrayViewMut1<f64>, beta: &ArrayViewMut1<f64>, theta: &mut ArrayViewMut1<f64>,
) {
    let q = alpha.len();
    let p = beta.len();
    let alpha_slice = &theta.slice(s![0..q]);
    let beta_slice = &theta.slice(s![q..q + p]);
    let numerator = alpha.dot(alpha_slice) + beta.dot(beta_slice);
    let scale = 1.0 - STATIONARITY_MARGIN;
    let c = numerator / scale;
    for i in 0..q {
        theta[i] = alpha[i] / scale * (theta[i] - c);
    }
    for j in 0..p {
        theta[q + j] = beta[j] / scale * (theta[q + j] - c);
    }
}

/// `safe_logistic` — numerically stable logistic `1 / (1 + exp(-x))`.
///
/// Purpose
/// -------
/// Compute the logistic function with guarded branches to avoid overflow
/// and underflow for large magnitudes of `x`.
///
/// Parameters
/// ----------
/// - `x`: `f64`
///   Real-valued input, expected to be finite.
///
/// Returns
/// -------
/// `f64`
///   The value `σ(x) ∈ (0, 1)`.
///
/// Errors
/// ------
/// - Does not return errors; callers should avoid passing non-finite inputs.
///
/// Panics
/// ------
/// - Does not panic under normal usage.
///
/// Safety
/// ------
/// - No `unsafe` code is used.
///
/// Notes
/// -----
/// - For sufficiently large `x > 0`, returns `1.0` directly.
/// - For `x ≥ 0`, computes `1 / (1 + exp(-x))`.
/// - For `x < 0`, uses `exp(x) / (1 + exp(x))` to avoid overflow in
///   `exp(-x)`.
///
/// Examples
/// --------
/// ```rust
/// use rust_timeseries::optimization::numerical_stability::transformations::safe_logistic;
///
/// let y = safe_logistic(0.0_f64);
/// assert!((y - 0.5).abs() < 1e-12);
/// ```
pub fn safe_logistic(x: f64) -> f64 {
    if x > 20.0 {
        1.0
    } else if x >= 0.0 {
        let exp_neg_x = (-x).exp();
        1.0 / (1.0 + exp_neg_x)
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// delta_method — map θ-space covariance to (ω, α, β) covariance.
///
/// Purpose
/// -------
/// Apply the multivariate delta method to transform a covariance matrix in
/// unconstrained θ-space into a covariance matrix for the model-space
/// parameters `(ω, α, β)` used in the ACD mapping:
/// - `ω = softplus(θ₀)`,
/// - `(α, β, slack)` = `(1 - STATIONARITY_MARGIN) · softmax(θ₁: )` (with
///   implicit slack).
///
/// Parameters
/// ----------
/// - `theta_space_cov`: `&Array2<f64>`
///   Symmetric `n×n` covariance matrix for the unconstrained parameters
///   θ, where `n = 1 + q + p`.
/// - `theta_hat`: `&Array1<f64>`
///   Unconstrained parameter vector at which the Jacobian is evaluated
///   (typically the MLE θ̂). Must have length `1 + q + p`.
/// - `alpha`: `&Array1<f64>`
///   Fitted `α` weights of length `q` corresponding to `theta_hat` via
///   `safe_softmax`.
/// - `beta`: `&Array1<f64>`
///   Fitted `β` weights of length `p` corresponding to `theta_hat`.
///
/// Returns
/// -------
/// `Array2<f64>`
///   Full `(1 + q + p)×(1 + q + p)` covariance matrix for `(ω, α, β)`
///   obtained via the delta method:
///   `Σ_param = J Σ_θ Jᵀ`, where `J` is the Jacobian of the mapping
///   θ ↦ (ω, α, β) at `theta_hat`.
///
/// Panics
/// ------
/// - Should never panic under normal usage.
///
/// Notes
/// -----
/// - This implementation uses only the existing `safe_logistic` and
///   `safe_softmax_deriv` primitives. It never differentiates the internal
///   `max θ_i` stabilization used in `safe_softmax`; the Jacobian is
///   defined for the *mathematical* softplus/softmax map.
/// - Complexity is O(n³) in the number of parameters `n = 1 + q + p`,
///   which is negligible for typical ACD(p, q) orders.
pub fn delta_method(
    theta_space_cov: &Array2<f64>, theta_hat: &Array1<f64>, alpha: &Array1<f64>, beta: &Array1<f64>,
) -> Array2<f64> {
    let n = theta_hat.len();
    let mut jt = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let mut e_i = Array1::<f64>::zeros(n);
        e_i[i] = 1.0;
        let g_i = apply_jt(theta_hat, alpha, beta, &e_i);
        for k in 0..n {
            jt[[k, i]] = g_i[k];
        }
    }

    // Σ_param = J Σ_θ Jᵀ = (Jᵀ)ᵀ Σ_θ (Jᵀ) = jtᵀ * theta_space_cov * jt
    let tmp = theta_space_cov.dot(&jt);
    let jt_t = jt.t().to_owned();
    jt_t.dot(&tmp)
}

// ----------- Helper methods ------------

/// apply_jt — apply Jᵀ to a parameter-space direction.
///
/// Purpose
/// -------
/// Given a direction `u_param` in parameter space `(ω, α, β)`, compute
/// the corresponding direction in θ-space via the transpose Jacobian
/// `Jᵀ`. This reuses the same primitives used in gradient maximization:
/// - `safe_logistic` for ω,
/// - `safe_softmax_deriv` for `(α, β)`
///
/// Parameters
/// ----------
/// - `theta_hat`: `&Array1<f64>`
///   Unconstrained parameters at which the Jacobian is evaluated.
/// - `alpha`: `&Array1<f64>`
///   Fitted α weights at `theta_hat`.
/// - `beta`: `&Array1<f64>`
///   Fitted β weights at `theta_hat`.
/// - `u_param`: `&Array1<f64>`
///   Direction in parameter space (length `1 + q + p`).
///
/// Returns
/// -------
/// `Array1<f64>`
///   The θ-space direction `Jᵀ u_param`.
fn apply_jt(
    theta_hat: &Array1<f64>, alpha: &Array1<f64>, beta: &Array1<f64>, u_param: &Array1<f64>,
) -> Array1<f64> {
    let n = u_param.len();
    let mut out = Array1::<f64>::zeros(n);
    out[0] = u_param[0] * safe_logistic(theta_hat[0]);
    let mut tmp = u_param.slice(s![1..]).to_owned(); // length q + p
    let mut alpha_tmp = alpha.clone();
    let mut beta_tmp = beta.clone();
    {
        let alpha_view = alpha_tmp.view_mut();
        let beta_view = beta_tmp.view_mut();
        let mut tmp_view = tmp.view_mut();
        safe_softmax_deriv(&alpha_view, &beta_view, &mut tmp_view);
    }

    out.slice_mut(s![1..]).assign(&tmp);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{Array1, Array2, array, s};

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Numerical behavior of stable transforms (softplus, inverse, logistic).
    // - Softmax mapping from logits to (α, β, slack) under STATIONARITY_MARGIN.
    // - The Jacobian–vector product implemented by `safe_softmax_deriv` in
    //   low dimensions via a finite-difference check.
    // - The transpose-Jacobian application in `apply_jt` via a scalar
    //   finite-difference delta-method check.
    // - The full delta-method covariance transformation in `delta_method`
    //   versus an explicit finite-difference Jacobian construction.
    //
    // They intentionally DO NOT cover:
    // - Parameter/domain validation (shape, positivity, finiteness), which is
    //   enforced in duration-core validation and `WorkSpace::update`.
    // - Full ACD stationarity invariants, which are exercised in the duration
    //   modules and higher-level integration tests.
    // -------------------------------------------------------------------------

    fn assert_vec_close(lhs: &Array1<f64>, rhs: &Array1<f64>, tol: f64) {
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "vector lengths differ: lhs = {}, rhs = {}",
            lhs.len(),
            rhs.len()
        );
        for (i, (x, y)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (x - y).abs();
            assert!(
                diff <= tol,
                "index {i}: values not within tol: left = {x}, right = {y}, diff = {diff}, tol = {tol}"
            );
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `safe_softplus` matches the naïve formula on a moderate grid
    // and remains finite/monotone on more extreme inputs.
    //
    // Given
    // -----
    // - A selection of x values in a safe range [-10, 10].
    // - Two extreme values, e.g. +/- 50.
    //
    // Expect
    // ------
    // - For moderate x, `safe_softplus(x)` ≈ `ln(1 + exp(x))`.
    // - For extreme x, the result is finite and preserves ordering.
    fn safe_softplus_moderate_and_extreme_inputs_behave_as_expected() {
        // moderate grid comparison to naïve implementation
        let xs = [-10.0_f64, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0];
        for &x in &xs {
            let naive = (1.0_f64 + x.exp()).ln();
            let stable = safe_softplus(x);
            assert_relative_eq!(stable, naive, max_relative = 1e-10);
        }

        // extreme values: just check finiteness and monotonicity
        let x_lo = -50.0_f64;
        let x_hi = 50.0_f64;
        let y_lo = safe_softplus(x_lo);
        let y_mid = safe_softplus(0.0);
        let y_hi = safe_softplus(x_hi);

        assert!(y_lo.is_finite());
        assert!(y_mid.is_finite());
        assert!(y_hi.is_finite());
        assert!(y_lo < y_mid && y_mid < y_hi);
    }

    #[test]
    // Purpose
    // -------
    // Check that `safe_softplus_inv` approximately inverts `safe_softplus`
    // over a reasonable range of inputs.
    //
    // Given
    // -----
    // - A grid of real t values including negative, zero, and positive values.
    //
    // Expect
    // ------
    // - `safe_softplus_inv(safe_softplus(t)) ≈ t` within a small tolerance.
    fn safe_softplus_inv_composes_back_to_identity_on_reasonable_range() {
        let ts = [-5.0_f64, -1.0, -0.1, 0.1, 1.0, 5.0, 10.0];
        for &t in &ts {
            let x = safe_softplus(t);
            // by construction x > 0 and finite
            let t_back = safe_softplus_inv(x);
            // allow slightly looser tolerance due to nonlinear conditioning
            assert_relative_eq!(t_back, t, max_relative = 1e-8);
        }
    }

    #[test]
    // Purpose
    // -------
    // Validate basic properties of `safe_logistic`: value at 0, symmetry, and
    // saturation in the tails.
    //
    // Given
    // -----
    // - Inputs x = 0, +/- 5, +/- 50.
    //
    // Expect
    // ------
    // - σ(0) ≈ 0.5.
    // - σ(-x) ≈ 1 - σ(x).
    // - σ(50) ≈ 1 and σ(-50) ≈ 0.
    fn safe_logistic_symmetry_and_tail_behavior_are_correct() {
        let mid = safe_logistic(0.0_f64);
        assert_relative_eq!(mid, 0.5, max_relative = 1e-12);

        let xs = [0.5_f64, 1.0, 2.0, 5.0];
        for &x in &xs {
            let pos = safe_logistic(x);
            let neg = safe_logistic(-x);
            assert_relative_eq!(neg, 1.0 - pos, max_relative = 1e-12);
        }

        let big_pos = safe_logistic(50.0);
        let big_neg = safe_logistic(-50.0);
        assert!(big_pos > 0.9999);
        assert!(big_neg < 0.0001);
    }

    #[test]
    // Purpose
    // -------
    // Ensure that `safe_softmax` produces non-negative α/β weights whose sum
    // with slack equals `1 - STATIONARITY_MARGIN` for a small (q, p) system.
    //
    // Given
    // -----
    // - q = 2, p = 1 with arbitrary logits.
    //
    // Expect
    // ------
    // - All α[i], β[j], and slack are ≥ 0.
    // - sum(α) + sum(β) + slack ≈ 1 - STATIONARITY_MARGIN.
    fn safe_softmax_small_system_respects_stationarity_mass() {
        let logits = array![0.3_f64, -1.2_f64, 0.7_f64]; // q = 2, p = 1
        let mut alpha = Array1::<f64>::zeros(2);
        let mut beta = Array1::<f64>::zeros(1);

        let slack = {
            let theta_view = logits.view();
            let alpha_view = alpha.view_mut();
            let beta_view = beta.view_mut();
            safe_softmax(alpha_view, beta_view, &theta_view)
        };

        // non-negativity
        assert!(alpha.iter().all(|v| *v >= 0.0));
        assert!(beta.iter().all(|v| *v >= 0.0));
        assert!(slack >= 0.0);

        let total = alpha.sum() + beta.sum() + slack;
        assert_relative_eq!(total, 1.0 - STATIONARITY_MARGIN, max_relative = 1e-12,);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `safe_softmax_deriv` matches a finite-difference Jacobian–
    // vector product for a tiny (q, p) system.
    //
    // Given
    // -----
    // - q = 1, p = 1, and base logits z ∈ ℝ².
    // - Direction v taken to be z itself, following the implementation
    //   (“v is taken to be the current logits”).
    //
    // Expect
    // ------
    // - The contents of θ after `safe_softmax_deriv` agree with a centered
    //   finite-difference approximation of J(z) · v for the mapping
    //   z ↦ (α(z), β(z)).
    fn safe_softmax_deriv_matches_finite_difference_directional_derivative() {
        // q = 1, p = 1
        let base_logits = array![0.4_f64, -0.8_f64];
        let eps = 1e-6_f64;

        // helper to run softmax and return concatenated [α, β]
        fn eval_softmax(logits: &Array1<f64>) -> Array1<f64> {
            let mut alpha = Array1::<f64>::zeros(1);
            let mut beta = Array1::<f64>::zeros(1);
            let theta_view = logits.view();
            let alpha_view = alpha.view_mut();
            let beta_view = beta.view_mut();
            let _slack = super::safe_softmax(alpha_view, beta_view, &theta_view);

            let mut out = Array1::<f64>::zeros(2);
            out.slice_mut(s![0..1]).assign(&alpha);
            out.slice_mut(s![1..2]).assign(&beta);
            out
        }

        // base evaluation
        let g0 = eval_softmax(&base_logits);

        // direction v = z
        let dir = base_logits.clone();

        // centered finite-difference J(z)·v ≈ (g(z + eps v) - g(z - eps v)) / (2 eps)
        let dir_scaled = dir.mapv(|v_i| eps * v_i);
        let logits_plus = &base_logits + &dir_scaled;
        let logits_minus = &base_logits - &dir_scaled;

        let g_plus = eval_softmax(&logits_plus);
        let g_minus = eval_softmax(&logits_minus);

        let jv_fd = (&g_plus - &g_minus).mapv(|x| x / (2.0 * eps));

        // analytic Jv from `safe_softmax_deriv`
        let (mut alpha0, mut beta0) = {
            let mut alpha = Array1::<f64>::zeros(1);
            let mut beta = Array1::<f64>::zeros(1);
            let theta_view = base_logits.view();
            let alpha_view = alpha.view_mut();
            let beta_view = beta.view_mut();
            let _ = super::safe_softmax(alpha_view, beta_view, &theta_view);
            (alpha, beta)
        };

        let mut theta_for_deriv = base_logits.clone();
        {
            let alpha_view = alpha0.view_mut();
            let beta_view = beta0.view_mut();
            let mut theta_view = theta_for_deriv.view_mut();
            super::safe_softmax_deriv(&alpha_view, &beta_view, &mut theta_view);
        }

        let jv_analytic = theta_for_deriv;

        // compare finite-difference and analytic results
        assert_vec_close(&jv_analytic, &jv_fd, 1e-4);
        // sanity: both directions should be small compared to base g0
        assert!(jv_analytic.iter().all(|v| v.is_finite()));
        assert!(g0.iter().all(|v| v.is_finite()));
    }

    #[test]
    // Purpose
    // -------
    // Verify that `apply_jt` matches a scalar delta-method finite-difference
    // gradient for the mapping θ ↦ (ω, α, β) on a small (q = 1, p = 1) system.
    //
    // Given
    // -----
    // - θ̂ ∈ ℝ³ with components (θ₀, θ₁, θ₂).
    // - A parameter-space direction u_param ∈ ℝ³.
    //
    // Expect
    // ------
    // - `apply_jt(θ̂, α(θ̂), β(θ̂), u_param)` ≈ ∇_θ [u_param · f(θ)] where
    //   f(θ) = (ω(θ₀), α(θ₁, θ₂), β(θ₁, θ₂)).
    fn apply_jt_matches_finite_difference_scalar_delta_method() {
        let q = 1usize;
        let p = 1usize;
        let n = 1 + q + p;

        // θ̂: some non-degenerate point in θ-space
        let theta_hat = array![0.3_f64, -0.5_f64, 0.8_f64];

        // compute α, β at θ̂[1:]
        let mut alpha = Array1::<f64>::zeros(q);
        let mut beta = Array1::<f64>::zeros(p);
        {
            let theta_ab = theta_hat.slice(s![1..]);
            let alpha_view = alpha.view_mut();
            let beta_view = beta.view_mut();
            let _slack = safe_softmax(alpha_view, beta_view, &theta_ab);
        }

        // direction in parameter space (ω, α, β)
        let u_param = array![1.0_f64, -0.3_f64, 0.5_f64];

        // analytic Jᵀ u via apply_jt
        let jtu_analytic = apply_jt(&theta_hat, &alpha, &beta, &u_param);

        // scalar function g(θ) = u_param · f(θ)
        let eval_scalar = |theta: &Array1<f64>| -> f64 {
            // recompute α, β at this θ
            let mut alpha_loc = Array1::<f64>::zeros(q);
            let mut beta_loc = Array1::<f64>::zeros(p);
            {
                let theta_ab = theta.slice(s![1..]);
                let alpha_view = alpha_loc.view_mut();
                let beta_view = beta_loc.view_mut();
                let _slack = safe_softmax(alpha_view, beta_view, &theta_ab);
            }
            let omega = safe_softplus(theta[0]);

            let mut params = Array1::<f64>::zeros(n);
            params[0] = omega;
            params[1] = alpha_loc[0];
            params[2] = beta_loc[0];

            u_param.dot(&params)
        };

        // centered finite-difference gradient ∇_θ g(θ̂)
        let eps = 1e-6_f64;
        let mut jtu_fd = Array1::<f64>::zeros(n);
        for j in 0..n {
            let mut theta_plus = theta_hat.clone();
            let mut theta_minus = theta_hat.clone();
            theta_plus[j] += eps;
            theta_minus[j] -= eps;

            let g_plus = eval_scalar(&theta_plus);
            let g_minus = eval_scalar(&theta_minus);

            jtu_fd[j] = (g_plus - g_minus) / (2.0 * eps);
        }

        assert_vec_close(&jtu_analytic, &jtu_fd, 1e-4);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `delta_method` produces a covariance matrix consistent with
    // an explicit finite-difference Jacobian for θ ↦ (ω, α, β) when the
    // θ-space covariance is the identity.
    //
    // Given
    // -----
    // - θ̂ ∈ ℝ³ with components (θ₀, θ₁, θ₂).
    // - Σ_θ = I₃.
    //
    // Expect
    // ------
    // - `delta_method(Σ_θ, θ̂, α(θ̂), β(θ̂))` ≈ J Jᵀ where
    //   J is the Jacobian of f(θ) = (ω, α, β) at θ̂ constructed by finite
    //   differences.
    fn delta_method_matches_finite_difference_jacobian_covariance() {
        let q = 1usize;
        let p = 1usize;
        let n = 1 + q + p;

        // θ̂: same as previous test for consistency
        let theta_hat = array![0.3_f64, -0.5_f64, 0.8_f64];

        // α, β at θ̂[1:]
        let mut alpha = Array1::<f64>::zeros(q);
        let mut beta = Array1::<f64>::zeros(p);
        {
            let theta_ab = theta_hat.slice(s![1..]);
            let alpha_view = alpha.view_mut();
            let beta_view = beta.view_mut();
            let _slack = safe_softmax(alpha_view, beta_view, &theta_ab);
        }

        // Σ_θ = I_n
        let mut theta_cov = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            theta_cov[[i, i]] = 1.0;
        }

        // covariance via delta_method
        let cov_delta = delta_method(&theta_cov, &theta_hat, &alpha, &beta);

        // explicit finite-difference Jacobian J of f(θ) = (ω, α, β)
        let eval_params = |theta: &Array1<f64>| -> Array1<f64> {
            let mut alpha_loc = Array1::<f64>::zeros(q);
            let mut beta_loc = Array1::<f64>::zeros(p);
            {
                let theta_ab = theta.slice(s![1..]);
                let alpha_view = alpha_loc.view_mut();
                let beta_view = beta_loc.view_mut();
                let _slack = safe_softmax(alpha_view, beta_view, &theta_ab);
            }
            let omega = safe_softplus(theta[0]);

            let mut out = Array1::<f64>::zeros(n);
            out[0] = omega;
            out[1] = alpha_loc[0];
            out[2] = beta_loc[0];
            out
        };

        let eps = 1e-6_f64;
        let mut j_fd = Array2::<f64>::zeros((n, n));
        for j in 0..n {
            let mut theta_plus = theta_hat.clone();
            let mut theta_minus = theta_hat.clone();
            theta_plus[j] += eps;
            theta_minus[j] -= eps;

            let f_plus = eval_params(&theta_plus);
            let f_minus = eval_params(&theta_minus);
            let col = (&f_plus - &f_minus).mapv(|v| v / (2.0 * eps));

            for i in 0..n {
                j_fd[[i, j]] = col[i];
            }
        }

        // Σ_param_fd = J Σ_θ Jᵀ = J Jᵀ (since Σ_θ = I)
        let cov_fd = j_fd.dot(&j_fd.t());

        // compare matrices entrywise
        assert_eq!(cov_delta.nrows(), n);
        assert_eq!(cov_delta.ncols(), n);
        assert_eq!(cov_fd.nrows(), n);
        assert_eq!(cov_fd.ncols(), n);

        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(
                    cov_delta[[i, j]],
                    cov_fd[[i, j]],
                    epsilon = 1e-8,
                    max_relative = 1e-4
                );
            }
        }
    }
}
