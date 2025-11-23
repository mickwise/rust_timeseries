//! numerical_stability — numerically robust transformations and covariance tools.
//!
//! Purpose
//! -------
//! Collect numerically stable scalar and vector transforms plus helper
//! routines for mapping covariances between unconstrained optimizer space
//! and model-space parameters in ACD-style duration models. This module
//! centralizes small numerical tolerances and transform logic so the rest
//! of the optimization and duration layers can assume well-conditioned
//! `f64` arithmetic.
//!
//! Key behaviors
//! -------------
//! - Provide stable scalar transforms (`safe_softplus`, its inverse, and
//!   `safe_logistic`) for mapping unconstrained reals into strictly
//!   positive or (0, 1) parameters without overflow/underflow.
//! - Implement a stationarity-aware softmax mapping from logits to
//!   `(α, β)` plus slack (`safe_softmax`) together with a Jacobian–vector
//!   product helper (`safe_softmax_deriv`) for gradient propagation.
//! - Expose a delta-method covariance transformer (`delta_method`) that
//!   converts θ-space covariance into covariance for `(ω, α, β)` at a
//!   given estimate `θ̂`.
//! - Centralize small numeric tolerances (`STATIONARITY_MARGIN`,
//!   `LOGIT_EPS`, `EIGEN_EPS`, `GENERAL_TOL`) so downstream modules share
//!   consistent guards and clamping behavior.
//!
//! Invariants & assumptions
//! ------------------------
//! - All public transforms assume finite `f64` inputs; domain and shape
//!   validation (e.g., positivity, length checks) is enforced in the
//!   duration and optimizer layers, not here.
//! - Softmax helpers assume `alpha.len() + beta.len() == theta.len()` and
//!   that `alpha`/`beta` are already sized consistently with the chosen
//!   ACD orders `(p, q)`.
//! - `STATIONARITY_MARGIN` is treated as a fixed global slack enforcing
//!   `sum(α) + sum(β) + slack ≈ 1 − STATIONARITY_MARGIN`; higher-level
//!   modules are responsible for checking the full stationarity condition.
//! - `delta_method` assumes a symmetric θ-space covariance matrix and
//!   α/β weights computed from the same `theta_hat` via the softmax
//!   mapping; no attempt is made to repair inconsistent inputs.
//!
//! Conventions
//! -----------
//! - All routines operate on `ndarray` types (`Array1`, `Array2` and
//!   their views) and favor in-place updates to minimize heap allocation.
//! - ACD parameter layout in θ-space follows
//!   `θ = (θ_ω, θ_α[0..q), θ_β[0..p))` with total length `1 + q + p`;
//!   `(α, β)` are derived from `theta[1..]` via a max-shift softmax.
//! - This module never logs, performs I/O, or touch global
//!   state; it is pure numerical helpers suitable for use inside tight
//!   inner loops.
//! - Panics and `unsafe` are avoided under normal usage; invalid inputs
//!   should be caught by upstream validation and surfaced as
//!   domain-specific error types.
//!
//! Downstream usage
//! ----------------
//! - Duration workspaces use these transforms to map optimizer-space
//!   parameters into model-space `(ω, α, β)` and to push gradients between
//!   the two via Jacobian–vector products.
//! - Optimizer and inference code reuse `STATIONARITY_MARGIN`,
//!   `EIGEN_EPS`, and `GENERAL_TOL` as shared tolerances for stationarity
//!   margins, eigenvalue regularization, and generic clamping.
//! - Covariance-estimation routines call `delta_method` to turn θ-space
//!   covariance (e.g., from an inverse Hessian) into interpretable
//!   parameter covariances for reporting and confidence-interval
//!   construction.
//! - Higher-level front-ends are expected to depend only on the
//!   re-exported surface (constants and transforms) or the prelude, not on
//!   internal implementation details of [`transformations`].
//!
//! Testing notes
//! -------------
//! - Unit tests in [`transformations`] cover:
//!   - agreement of stable transforms with naïve formulas on safe grids,
//!   - tail behavior and symmetry properties of the logistic/softplus
//!     helpers,
//!   - softmax mass conservation under `STATIONARITY_MARGIN` and
//!     non-negativity of `(α, β, slack)`,
//!   - correctness of `safe_softmax_deriv` via finite-difference
//!     Jacobian–vector checks in low dimensions,
//!   - `delta_method` versus an explicit finite-difference Jacobian
//!     construction when θ-space covariance is the identity.
//! - Integration tests in duration and optimization modules exercise
//!   higher-level invariants (full ACD stationarity, parameter validation,
//!   optimizer robustness) rather than re-testing these low-level numeric
//!   primitives.

pub mod transformations;

// ---- Re-exports (primary public surface) ----------------------------------

pub use self::transformations::{
    EIGEN_EPS, GENERAL_TOL, LOGIT_EPS, STATIONARITY_MARGIN, delta_method, safe_logistic,
    safe_softmax, safe_softmax_deriv, safe_softplus, safe_softplus_inv,
};

// ---- Optional convenience prelude for downstream crates -------------------
//
// Downstream crates can write
//
//     use rust_timeseries::optimization::numerical_stability::prelude::*;
//
// to import the main numerical-stability surface in a single line.

pub mod prelude {
    pub use super::transformations::{
        EIGEN_EPS, GENERAL_TOL, LOGIT_EPS, STATIONARITY_MARGIN, delta_method, safe_logistic,
        safe_softmax, safe_softplus, safe_softplus_inv,
    };
}
