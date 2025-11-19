//! ACD(p, q) parameters and scratch workspace — model-space container and zero-copy buffers.
//!
//! Purpose
//! -------
//! Provide a validated model-space parameterization for ACD(p, q) models
//! (`ACDParams`) and a reusable, zero-copy workspace (`ACDScratch`) used by
//! likelihood, gradient, and forecasting routines. This module also implements
//! a numerically stable mapping between model-space parameters and an
//! optimizer-space vector θ.
//!
//! Key behaviors
//! -------------
//! - Represent constrained, model-space ACD parameters `(ω, α, β, slack, ψ-lags)`
//!   via [`ACDParams`], including constructors from both raw values and an
//!   optimizer-space vector.
//! - Provide a reusable scratch workspace [`ACDScratch`] holding α, β, ψ,
//!   duration-lag, and derivative buffers sized to a given sample and model
//!   shape, so hot paths run allocation-free.
//! - Implement the mapping between model space and optimizer space using
//!   softplus / softplus⁻¹ and a scaled simplex representation with slack to
//!   enforce strict stationarity.
//!
//! Invariants & assumptions
//! ------------------------
//! - Model-space parameters satisfy:
//!   - `ω > 0`,
//!   - `α.len() == q`, `β.len() == p`, with elementwise non-negativity,
//!   - `∑α + ∑β + slack = 1 − STATIONARITY_MARGIN` with `slack ≥ 0`,
//!   - `psi_lags.len() == p` and finite (used for forecasting).
//! - The stationarity margin `STATIONARITY_MARGIN` enforces strict
//!   stationarity, so `∑α + ∑β < 1`.
//! - Scratch buffers created by [`ACDScratch::new`] are sized consistently
//!   with the sample length `n` and orders `(p, q)` and are reused without
//!   reallocation.
//!
//! Conventions
//! -----------
//! - Model space uses `(ω, α, β, slack, ψ-lags)` under a simplex constraint
//!   for `(α, β, slack)` with total mass `1 − STATIONARITY_MARGIN`.
//! - Optimizer space uses a vector
//!   `θ = [θ₀ | α(1..q) | β(1..p)]`, with `θ₀ = softplus⁻¹(ω)` and α/β
//!   represented as log-odds relative to the slack probability.
//! - All numerics are `f64`; small probabilities are clamped to `LOGIT_EPS`
//!   before taking logs to improve numerical stability.
//! - Functions in this module return `ParamResult<T>` for fallible
//!   constructions and never panic under validated usage.
//!
//! Downstream usage
//! ----------------
//! - Use [`ACDParams::new`] when you have model-space parameters and want to
//!   validate them against ACD constraints or when initializing a model.
//! - Use [`ACDParams::from_theta`] and [`ACDParams::to_theta`] to map back and
//!   forth between model space and optimizer-space vectors when interfacing
//!   with the optimization layer.
//! - Use [`ACDScratch`] to allocate and reuse workspace buffers for
//!   likelihood, gradient, and forecasting routines that operate in tight
//!   loops.
//!
//! Testing notes
//! -------------
//! - Unit tests for this module:
//!   - verify that `ACDParams::new` and `ACDParams::from_theta` enforce
//!     stationarity and shape/positivity invariants,
//!   - check that `ACDParams::to_theta` and `ACDParams::from_theta` are
//!     approximately inverse mappings (up to numerical tolerance),
//!   - confirm that `ACDScratch::new` allocates buffers with the expected
//!     shapes and that they can be reused without reallocation.
//! - Integration tests elsewhere cover end-to-end likelihood evaluation,
//!   optimization, and forecasting behavior using these types.
use crate::{
    duration::{
        core::{
            shape::ACDShape,
            validation::{
                validate_alpha, validate_beta, validate_omega, validate_psi_lags,
                validate_stationarity_and_slack, validate_theta,
            },
        },
        errors::ParamResult,
    },
    optimization::numerical_stability::transformations::{
        LOGIT_EPS, STATIONARITY_MARGIN, safe_softmax, safe_softplus, safe_softplus_inv,
    },
};
use ndarray::{Array1, Array2, ArrayView1, Zip, s};
use std::cell::RefCell;

/// ACDScratch — zero-copy workspace for ACD estimation and forecasting.
///
/// Purpose
/// -------
/// Represent a reusable scratch workspace for ACD(p, q) likelihood, gradient,
/// and forecasting routines, holding all intermediate buffers so that hot
/// loops avoid heap allocations.
///
/// Key behaviors
/// -------------
/// - Allocates fixed-size buffers for α, β, ψ, duration lags, and derivatives
///   based on a given sample length `n` and model orders `(p, q)`.
/// - Exposes buffers via `RefCell<Array*>` to allow interior mutability without
///   cloning in hot paths.
/// - Keeps all buffers zero-initialized on construction and reuses them across
///   calls.
///
/// Parameters
/// ----------
/// Constructed via:
/// - `ACDScratch::new(n: usize, p: usize, q: usize)`
///   Provide sample length and model orders to size all buffers.
///
/// Fields
/// ------
/// - `alpha_buf`: `RefCell<Array1<f64>>`
///   Scratch buffer of length `q` for α-related computations.
/// - `beta_buf`: `RefCell<Array1<f64>>`
///   Scratch buffer of length `p` for β-related computations.
/// - `psi_buf`: `RefCell<Array1<f64>>`
///   Scratch buffer of length `n + p`, holding ψ values including `p`
///   pre-sample lags.
/// - `dur_buf`: `RefCell<Array1<f64>>`
///   Scratch buffer of length `q` for duration lags τ used in recursion.
/// - `deriv_buf`: `RefCell<Array2<f64>>`
///   Scratch buffer of shape `(n + p, 1 + q + p)` for per-time, per-parameter
///   derivatives or related quantities.
///
/// Invariants
/// ----------
/// - Buffer shapes are determined by `(n, p, q)` at construction and remain
///   fixed for the lifetime of the `ACDScratch`.
/// - All buffers are allocated and zero-initialized once; reuse must not rely
///   on any particular content beyond what the caller writes.
/// - Borrowing rules are enforced by `RefCell`; violating them produces a
///   runtime panic.
///
/// Performance
/// -----------
/// - Avoids repeated allocations inside tight loops by centralizing buffer
///   allocation at construction.
/// - Suitable for workflows that perform many likelihood or gradient
///   evaluations for the same `(n, p, q)`.
///
/// Notes
/// -----
/// - Intended primarily as an internal helper type. Public APIs should
///   typically construct a single `ACDScratch` and pass references into
///   lower-level routines.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDScratch {
    /// Scratch buffer for α.
    pub alpha_buf: RefCell<Array1<f64>>,
    /// Scratch buffer for β.
    pub beta_buf: RefCell<Array1<f64>>,
    /// Scratch buffer for ψ.
    pub psi_buf: RefCell<Array1<f64>>,
    /// Scratch buffer for initial durations.
    pub dur_buf: RefCell<Array1<f64>>,
    /// Scratch buffer for derivatives.
    pub deriv_buf: RefCell<Array2<f64>>,
}

impl ACDScratch {
    /// Construct a new [`ACDScratch`] workspace for a given sample and model order.
    ///
    /// Parameters
    /// ----------
    /// - `n`: `usize`
    ///   Length of the in-sample series used for likelihood/gradient evaluation.
    /// - `p`: `usize`
    ///   ACD order for β (ψ-lags). Determines the length of `beta_buf` and the
    ///   parameter dimension in `deriv_buf`.
    /// - `q`: `usize`
    ///   ACD order for α (duration lags). Determines the length of `alpha_buf`
    ///   and part of the parameter dimension in `deriv_buf`.
    ///
    /// Returns
    /// -------
    /// `ACDScratch`
    ///   A fully allocated workspace with buffers sized as:
    ///   - `alpha_buf`: length `q`
    ///   - `beta_buf`: length `p`
    ///   - `psi_buf`: length `n + p`
    ///   - `dur_buf`: length `q`
    ///   - `deriv_buf`: shape `(n + p, 1 + q + p)`
    ///   All buffers are zero-initialized.
    ///
    /// Errors
    /// ------
    /// - Never returns an error; this constructor always succeeds for finite `n`,
    ///   `p`, and `q`.
    ///
    /// Panics
    /// ------
    /// - May panic only if memory allocation fails at the OS/runtime level for
    ///   extremely large sizes.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; callers must simply supply reasonable sizes.
    ///
    /// Notes
    /// -----
    /// - Reuse a single `ACDScratch` across repeated evaluations to avoid
    ///   repeated allocations in hot loops.
    /// - Shapes must be kept consistent with the data and model orders used by
    ///   downstream routines.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::params::ACDScratch;
    /// let scratch = ACDScratch::new(1000, 2, 1);
    /// # assert_eq!(scratch.alpha_buf.borrow().len(), 1);
    /// # assert_eq!(scratch.beta_buf.borrow().len(), 2);
    /// ```
    pub fn new(n: usize, p: usize, q: usize) -> ACDScratch {
        let alpha_buf = RefCell::new(Array1::zeros(q));
        let beta_buf = RefCell::new(Array1::zeros(p));
        let psi_buf = RefCell::new(Array1::zeros(n + p));
        let dur_buf = RefCell::new(Array1::zeros(q));
        let deriv_buf = RefCell::new(Array2::zeros((n + p, 1 + p + q)));
        ACDScratch { alpha_buf, beta_buf, psi_buf, dur_buf, deriv_buf }
    }
}

/// ACDParams — constrained model-space parameters for an ACD(p, q) model.
///
/// Purpose
/// -------
/// Represent validated model-space parameters for an ACD(p, q) duration model,
/// including `ω`, α, β, slack mass, and the last `p` ψ-lags needed for
/// recursion and forecasting.
///
/// Key behaviors
/// -------------
/// - Enforces positivity, shape, and strict-stationarity constraints at
///   construction time via shared validators.
/// - Provides constructors from both raw model-space values (`new`) and an
///   optimizer-space vector θ (`from_theta`).
/// - Implements helper methods to map back to optimizer space (`to_theta`) and
///   to compute the unconditional mean of durations (`uncond_mean`).
///
/// Parameters
/// ----------
/// Constructed via:
/// - `ACDParams::new(omega, alpha, beta, slack, psi_lags, p, q)`
///   Validate raw model-space inputs against ACD invariants.
/// - `ACDParams::from_theta(theta, shape, psi_lags)`
///   Reconstruct parameters from optimizer-space θ and cached ψ-lags.
///
/// Fields
/// ------
/// - `omega`: `f64`
///   Baseline scale parameter `ω > 0` in the ACD recursion.
/// - `alpha`: `Array1<f64>`
///   Non-negative α coefficients (length `q`) multiplying duration lags τ.
/// - `beta`: `Array1<f64>`
///   Non-negative β coefficients (length `p`) multiplying ψ-lags.
/// - `slack`: `f64`
///   Non-negative slack mass satisfying
///   `∑α + ∑β + slack = 1 − STATIONARITY_MARGIN`, ensuring strict stationarity.
/// - `psi_lags`: `Array1<f64>`
///   Last `p` conditional-mean values ψ used as pre-sample lags in recursion
///   and forecasting.
///
/// Invariants
/// ----------
/// - `omega` is finite and strictly positive.
/// - `alpha.len() == q`, `beta.len() == p`, `psi_lags.len() == p`.
/// - Entries of `alpha`, `beta`, and `psi_lags` are finite; α and β are
///   non-negative; ψ-lags satisfy the positivity rules enforced by
///   `validate_psi_lags`.
/// - `∑alpha + ∑beta + slack = 1 − STATIONARITY_MARGIN` with `slack ≥ 0`,
///   implying `∑alpha + ∑beta < 1`.
///
/// Performance
/// -----------
/// - Uses contiguous `ndarray::Array1` buffers suitable for vectorized
///   operations in likelihood and gradient code.
/// - `Clone` and `PartialEq` are inexpensive for typical `(p, q)` sizes.
///
/// Notes
/// -----
/// - This is the canonical model-space container consumed by ACD likelihood
///   and forecasting routines. Public APIs should prefer accepting `ACDParams`
///   rather than decomposing into separate scalars and vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDParams {
    /// ω > 0
    pub omega: f64,
    /// αᵢ ≥ 0
    pub alpha: Array1<f64>,
    /// βⱼ ≥ 0
    pub beta: Array1<f64>,
    /// slack ≥ 0
    pub slack: f64,
    /// Last p psi lags
    pub psi_lags: Array1<f64>,
}

impl ACDParams {
    /// Construct validated model-space ACD(p, q) parameters from raw inputs.
    ///
    /// Parameters
    /// ----------
    /// - `omega`: `f64`
    ///   Baseline scale parameter `ω`. Must be finite and strictly positive.
    /// - `alpha`: `Array1<f64>`
    ///   α coefficients for duration lags, expected length `q`. Entries must be
    ///   finite and non-negative.
    /// - `beta`: `Array1<f64>`
    ///   β coefficients for ψ-lags, expected length `p`. Entries must be finite
    ///   and non-negative.
    /// - `slack`: `f64`
    ///   Slack mass completing the simplex, must be finite and non-negative and
    ///   satisfy `∑alpha + ∑beta + slack = 1 − STATIONARITY_MARGIN`.
    /// - `psi_lags`: `Array1<f64>`
    ///   Last `p` ψ values used as pre-sample lags. Must have length `p` and
    ///   finite entries, with positivity enforced by `validate_psi_lags`.
    /// - `p`: `usize`
    ///   ACD order for β; used to check `beta` length and `psi_lags` length.
    /// - `q`: `usize`
    ///   ACD order for α; used to check `alpha` length.
    ///
    /// Returns
    /// -------
    /// `ParamResult<ACDParams>`
    ///   - `Ok(ACDParams)` when all invariants are satisfied.
    ///   - `Err(..)` when any validation fails (shape, positivity, stationarity,
    ///     or ψ-lag constraints).
    ///
    /// Errors
    /// ------
    /// - Propagates errors from:
    ///   - `validate_omega`,
    ///   - `validate_alpha`,
    ///   - `validate_beta`,
    ///   - `validate_stationarity_and_slack`,
    ///   - `validate_psi_lags`.
    ///
    /// Panics
    /// ------
    /// - Never panics; invalid inputs are reported via `ParamResult::Err`.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; callers must supply numerically reasonable arguments.
    ///
    /// Notes
    /// -----
    /// - Use this constructor when you already have model-space parameters
    ///   (e.g., from a configuration or prior) and want to enforce ACD
    ///   invariants before passing them into likelihood or forecasting code.
    pub fn new(
        omega: f64, alpha: Array1<f64>, beta: Array1<f64>, slack: f64, psi_lags: Array1<f64>,
        p: usize, q: usize,
    ) -> ParamResult<Self> {
        validate_omega(omega)?;
        validate_alpha(alpha.view(), q)?;
        validate_beta(beta.view(), p)?;
        validate_stationarity_and_slack(alpha.view(), beta.view(), slack)?;
        validate_psi_lags(&psi_lags, p)?;
        Ok(ACDParams { omega, alpha, beta, slack, psi_lags })
    }

    /// Reconstruct validated model-space parameters from an optimizer-space vector θ.
    ///
    /// Parameters
    /// ----------
    /// - `theta`: `ArrayView1<f64>`
    ///   Optimizer-space vector with layout
    ///   `θ = [θ₀ | α(1..q) | β(1..p)]`, where `θ₀ = softplus⁻¹(ω)` and the α/β
    ///   blocks contain logits relative to the slack component under a scaled
    ///   simplex of total mass `1 − STATIONARITY_MARGIN`.
    /// - `shape`: `&ACDShape`
    ///   Model order descriptor providing `p` and `q`, used to validate `theta`
    ///   and size α/β.
    /// - `psi_lags`: `Array1<f64>`
    ///   Last `p` ψ values used as pre-sample lags. Must have length `p` and
    ///   satisfy `validate_psi_lags`.
    ///
    /// Returns
    /// -------
    /// `ParamResult<ACDParams>`
    ///   - `Ok(ACDParams)` when the derived parameters satisfy all model-space
    ///     invariants.
    ///   - `Err(..)` when shape validation, stationarity, or ψ-lag checks fail.
    ///
    /// Errors
    /// ------
    /// - Propagates errors from:
    ///   - `validate_theta` when `theta` has wrong length or invalid entries,
    ///   - `validate_psi_lags` when ψ-lags are malformed,
    ///   - `validate_omega` for the recovered `ω`,
    ///   - `validate_alpha` and `validate_beta` for reconstructed α/β,
    ///   - `validate_stationarity_and_slack` for the implied slack.
    ///
    /// Panics
    /// ------
    /// - Never panics; invalid inputs are surfaced via `ParamResult::Err`.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; callers must ensure `theta`, `shape`, and `psi_lags`
    ///   correspond to the same `(p, q)` model.
    ///
    /// Notes
    /// -----
    /// - Internally:
    ///   - validates `theta` against `shape`,
    ///   - recovers `ω` with `safe_softplus(θ₀)`,
    ///   - applies `safe_softmax` to the α/β logits to obtain probabilities on
    ///     a scaled simplex,
    ///   - computes `slack = 1 − STATIONARITY_MARGIN − ∑α − ∑β`,
    ///   - validates stationarity and ψ-lags via shared validators.
    /// - Intended primarily for post-optimization reconstruction of model-space
    ///   parameters from an optimizer’s final θ.
    pub fn from_theta(
        theta: ArrayView1<f64>, shape: &ACDShape, psi_lags: Array1<f64>,
    ) -> ParamResult<Self> {
        let p = shape.p;
        let q = shape.q;
        validate_theta(theta, p, q)?;
        validate_psi_lags(&psi_lags, p)?;
        let mut alpha = Array1::zeros(q);
        let mut beta = Array1::zeros(p);
        let omega = safe_softplus(theta[0]);
        let log_odds = &theta.slice(s![1..]);
        let slack = safe_softmax(alpha.view_mut(), beta.view_mut(), &log_odds);
        validate_omega(omega)?;
        validate_alpha(alpha.view(), q)?;
        validate_beta(beta.view(), p)?;
        validate_stationarity_and_slack(alpha.view(), beta.view(), slack)?;
        Ok(ACDParams { omega, alpha, beta, slack, psi_lags })
    }

    /// Map model-space ACD parameters to an optimizer-space vector θ.
    ///
    /// Parameters
    /// ----------
    /// - `self`: `&ACDParams`
    ///   Model-space parameters assumed to satisfy all invariants, including
    ///   strict stationarity and simplex constraints.
    ///
    /// Returns
    /// -------
    /// `Array1<f64>`
    ///   Optimizer-space vector with layout:
    ///   - `θ[0] = softplus⁻¹(ω)`
    ///   - `θ[1 .. 1+q)` and `θ[1+q .. 1+q+p)` store log-odds of α and β
    ///     relative to slack, where `p = self.beta.len()` and
    ///     `q = self.alpha.len()`.
    ///
    /// Errors
    /// ------
    /// - Never returns an error; assumes `self` already satisfies all
    ///   model-space invariants.
    ///
    /// Panics
    /// ------
    /// - Never panics under valid invariants.
    ///   Supplying inconsistent α/β lengths elsewhere in the codebase would
    ///   constitute a programming error.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; relies on invariants enforced when building
    ///   `ACDParams`.
    ///
    /// Notes
    /// -----
    /// - Internally:
    ///   - computes `θ₀ = safe_softplus_inv(ω)`,
    ///   - writes normalized α/β probabilities into θ using
    ///     [`ACDParams::populate_simple_theta`],
    ///   - clamps small probabilities to `LOGIT_EPS`,
    ///   - computes `log_slack = ln(slack / (1 − STATIONARITY_MARGIN))`,
    ///   - converts probabilities to log-odds relative to slack via
    ///     [`ACDParams::apply_inv_softmax`].
    /// - The resulting θ is suitable for the optimization layer, which operates
    ///   purely in optimizer-space coordinates.
    pub fn to_theta(&self) -> Array1<f64> {
        let q = self.alpha.len();
        let p = self.beta.len();
        let theta0 = safe_softplus_inv(self.omega);
        let denom_inv = 1.0 / (1.0 - STATIONARITY_MARGIN);
        let mut theta = ndarray::Array1::<f64>::zeros(1 + p + q);
        self.populate_simple_theta(&mut theta, denom_inv);

        // Clamp small values to avoid numerical issues in log
        theta.iter_mut().for_each(|x| {
            if *x < LOGIT_EPS {
                *x = LOGIT_EPS;
            }
        });

        let log_slack = (self.slack * denom_inv).ln();
        theta[0] = theta0;
        self.apply_inv_softmax(&mut theta, log_slack);
        theta
    }

    /// Compute the unconditional mean duration implied by the ACD parameters.
    ///
    /// Parameters
    /// ----------
    /// - `self`: `&ACDParams`
    ///   Model-space parameters assumed to satisfy strict stationarity
    ///   (`∑α + ∑β < 1`).
    ///
    /// Returns
    /// -------
    /// `f64`
    ///   Unconditional mean duration:
    ///   `E[τ] = ω / (1 − ∑α − ∑β)`.
    ///
    /// Errors
    /// ------
    /// - Never returns an error.
    ///
    /// Panics
    /// ------
    /// - Never panics under valid invariants.
    ///   If `∑α + ∑β` were to approach or exceed 1 due to invariant violations,
    ///   the denominator would become ill-conditioned, but such cases should
    ///   be prevented by construction.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; relies on invariants enforced at construction.
    ///
    /// Notes
    /// -----
    /// - Useful as a baseline forecast and as a sanity check for fitted models.
    /// - Stationarity is enforced by the simplex constraint with
    ///   `STATIONARITY_MARGIN`, so the denominator is strictly positive.
    pub fn uncond_mean(&self) -> f64 {
        let sum_alpha = self.alpha.sum();
        let sum_beta = self.beta.sum();
        self.omega / (1.0 - sum_alpha - sum_beta)
    }

    /// Populate α and β probability slots in θ from model-space parameters.
    ///
    /// Parameters
    /// ----------
    /// - `theta`: `&mut Array1<f64>`
    ///   Output buffer with length at least `1 + q + p`, where
    ///   `q = self.alpha.len()` and `p = self.beta.len()`. Slot `θ[0]` is left
    ///   unchanged; α and β probabilities are written into the remaining slots.
    /// - `denom_inv`: `f64`
    ///   Reciprocal of the total simplex mass, typically
    ///   `1.0 / (1.0 - STATIONARITY_MARGIN)`, used to normalize α and β.
    ///
    /// Returns
    /// -------
    /// `()`
    ///   Writes normalized probabilities in place:
    ///   - `θ[1 .. 1+q)  = α_i * denom_inv`
    ///   - `θ[1+q .. 1+q+p) = β_j * denom_inv`
    ///   leaving `θ[0]` untouched.
    ///
    /// Errors
    /// ------
    /// - Never returns an error.
    ///
    /// Panics
    /// ------
    /// - May panic if `theta` is shorter than `1 + q + p` due to slice bounds;
    ///   this is considered a programming error.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; callers must ensure `theta` has sufficient length.
    ///
    /// Notes
    /// -----
    /// - This is an internal helper used by [`ACDParams::to_theta`] to write
    ///   normalized probabilities before conversion to log-odds.
    /// - Values written are probabilities, not logits; they are converted to
    ///   log-odds in a separate step.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use ndarray::Array1;
    /// # use rust_timeseries::duration::core::params::ACDParams;
    /// # use rust_timeseries::optimization::numerical_stability::transformations::STATIONARITY_MARGIN;
    /// # fn demo(params: &ACDParams) {
    /// let q = params.alpha.len();
    /// let p = params.beta.len();
    /// let mut theta = Array1::zeros(1 + q + p);
    /// let denom_inv = 1.0 / (1.0 - STATIONARITY_MARGIN);
    /// params.populate_simple_theta(&mut theta, denom_inv);
    /// # }
    /// ```
    fn populate_simple_theta(&self, theta: &mut Array1<f64>, denom_inv: f64) {
        let q = self.alpha.len();
        let p = self.beta.len();
        let mut alpha_slice = theta.slice_mut(s![1..1 + q]);
        Zip::from(&mut alpha_slice).and(self.alpha.view()).for_each(|t, &a| *t = a * denom_inv);

        let mut beta_slice = theta.slice_mut(s![1 + q..1 + q + p]);
        Zip::from(&mut beta_slice).and(self.beta.view()).for_each(|t, &b| *t = b * denom_inv);
    }

    /// Convert α and β probabilities in θ to log-odds relative to slack.
    ///
    /// Parameters
    /// ----------
    /// - `theta`: `&mut Array1<f64>`
    ///   Buffer whose α and β slots currently store probabilities
    ///   `π_i = component_i / (1 − STATIONARITY_MARGIN)` for
    ///   `q = self.alpha.len()` and `p = self.beta.len()`. Slot `θ[0]` is not
    ///   modified.
    /// - `log_slack`: `f64`
    ///   Natural logarithm of the slack probability `π_slack`, where
    ///   `π_slack = slack / (1 − STATIONARITY_MARGIN) = 1 − ∑π_α − ∑π_β`.
    ///
    /// Returns
    /// -------
    /// `()`
    ///   Updates α and β slots in place so that each becomes
    ///   `ln(π_i) − log_slack`, leaving `θ[0]` unchanged.
    ///
    /// Errors
    /// ------
    /// - Never returns an error.
    ///
    /// Panics
    /// ------
    /// - May panic if `theta` is shorter than `1 + q + p` due to slice bounds.
    /// - If any probability slot is zero or negative, `ln` will yield `-inf` or
    ///   `NaN`; callers should clamp probabilities (e.g., with `LOGIT_EPS`)
    ///   before calling this method.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; callers must ensure probability slots are strictly
    ///   positive and that `theta` is sized consistently.
    ///
    /// Notes
    /// -----
    /// - Intended to be called from [`ACDParams::to_theta`] after probabilities
    ///   have been written and clamped.
    /// - This is the final step in constructing optimizer-space α/β coordinates
    ///   used by the optimization layer.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use ndarray::Array1;
    /// # use rust_timeseries::duration::core::params::ACDParams;
    /// # use rust_timeseries::optimization::numerical_stability::transformations::{STATIONARITY_MARGIN, LOGIT_EPS};
    /// # fn demo(params: &ACDParams) {
    /// let q = params.alpha.len();
    /// let p = params.beta.len();
    /// let mut theta = Array1::zeros(1 + q + p);
    /// let denom_inv = 1.0 / (1.0 - STATIONARITY_MARGIN);
    /// params.populate_simple_theta(&mut theta, denom_inv);
    /// theta.iter_mut().for_each(|t| if *t < LOGIT_EPS { *t = LOGIT_EPS; });
    /// let log_slack = (params.slack * denom_inv).ln();
    /// params.apply_inv_softmax(&mut theta, log_slack);
    /// # }
    /// ```
    fn apply_inv_softmax(&self, theta: &mut Array1<f64>, log_slack: f64) {
        let q = self.alpha.len();
        let p = self.beta.len();
        let mut alpha_slice = theta.slice_mut(s![1..1 + q]);
        alpha_slice.iter_mut().for_each(|t| *t = (*t).ln() - log_slack);

        let mut beta_slice = theta.slice_mut(s![1 + q..1 + q + p]);
        beta_slice.iter_mut().for_each(|t| *t = (*t).ln() - log_slack);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duration::{
        core::shape::ACDShape,
        errors::{ParamError, ParamResult},
    };
    use crate::optimization::numerical_stability::transformations::STATIONARITY_MARGIN;
    use ndarray::array;

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - That `ACDScratch::new` allocates buffers with shapes consistent with
    //   (n, p, q) and keeps them reusable.
    // - That `ACDParams::new` enforces basic shape/stationarity invariants and
    //   surfaces `ParamError` variants on invalid input.
    // - That `ACDParams::to_theta` and `ACDParams::from_theta` form an
    //   approximate round-trip between model space and optimizer space.
    // - That `ACDParams::uncond_mean` matches the closed-form formula.
    //
    // They intentionally DO NOT cover:
    // - End-to-end likelihood or gradient behavior (integration tests).
    // - PyO3 conversion behavior for error types (tested at the Python boundary).
    // - Numerical edge cases near machine precision for extreme θ values.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Ensure `ACDScratch::new` allocates all buffers with the expected shapes.
    //
    // Given
    // -----
    // - A sample length n, and model orders p, q.
    //
    // Expect
    // ------
    // - alpha_buf has length q,
    // - beta_buf has length p,
    // - psi_buf has length n + p,
    // - dur_buf has length q,
    // - deriv_buf has shape (n + p, 1 + q + p).
    fn acdscratch_new_allocates_expected_shapes() {
        // Arrange
        let n = 100usize;
        let p = 2usize;
        let q = 3usize;

        // Act
        let scratch = ACDScratch::new(n, p, q);

        // Assert
        assert_eq!(scratch.alpha_buf.borrow().len(), q);
        assert_eq!(scratch.beta_buf.borrow().len(), p);
        assert_eq!(scratch.psi_buf.borrow().len(), n + p);
        assert_eq!(scratch.dur_buf.borrow().len(), q);
        let deriv = scratch.deriv_buf.borrow();
        assert_eq!(deriv.nrows(), n + p);
        assert_eq!(deriv.ncols(), 1 + q + p);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `ACDParams::new` accepts well-formed parameters and returns Ok.
    //
    // Given
    // -----
    // - p = 2, q = 1, and alpha/beta/slack chosen to sum to 1 - STATIONARITY_MARGIN,
    //   with positive psi_lags of length p.
    //
    // Expect
    // ------
    // - `ACDParams::new` returns `Ok(ACDParams)` and all fields match the inputs.
    fn acdparams_new_accepts_valid_parameters() {
        // Arrange
        let p = 2usize;
        let q = 1usize;
        let omega = 0.8_f64;
        let alpha = array![0.1_f64];
        let beta = array![0.2_f64, 0.25_f64];
        let coeff_sum = alpha.sum() + beta.sum();
        let slack = (1.0 - STATIONARITY_MARGIN) - coeff_sum;
        let psi_lags = array![1.0_f64, 1.1_f64];

        // Sanity check: slack is positive.
        assert!(slack > 0.0);

        // Act
        let params =
            ACDParams::new(omega, alpha.clone(), beta.clone(), slack, psi_lags.clone(), p, q)
                .expect("valid parameters should construct successfully");

        // Assert
        assert_eq!(params.omega, omega);
        assert_eq!(params.alpha, alpha);
        assert_eq!(params.beta, beta);
        assert_eq!(params.slack, slack);
        assert_eq!(params.psi_lags, psi_lags);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `ACDParams::new` rejects an alpha length mismatch with ParamError.
    //
    // Given
    // -----
    // - p = 1, q = 2, and an alpha vector of length 1 instead of 2.
    //
    // Expect
    // ------
    // - `ACDParams::new` returns `Err(ParamError::AlphaLengthMismatch { .. })`.
    fn acdparams_new_rejects_alpha_length_mismatch() {
        // Arrange
        let p = 1usize;
        let q = 2usize;
        let omega = 1.0_f64;
        let alpha = array![0.1_f64]; // wrong length
        let beta = array![0.2_f64];
        let coeff_sum = alpha.sum() + beta.sum();
        let slack = (1.0 - STATIONARITY_MARGIN) - coeff_sum;
        let psi_lags = array![1.0_f64];

        // Act
        let result: ParamResult<ACDParams> =
            ACDParams::new(omega, alpha, beta, slack, psi_lags, p, q);

        // Assert
        match result {
            Err(ParamError::AlphaLengthMismatch { expected, actual }) => {
                assert_eq!(expected, q);
                assert_eq!(actual, 1);
            }
            other => panic!("expected AlphaLengthMismatch error, got {:?}", other),
        }
    }

    #[test]
    // Purpose
    // -------
    // Check that `ACDParams::to_theta` and `ACDParams::from_theta` approximately
    // invert each other for a simple ACD(1, 1) configuration.
    //
    // Given
    // -----
    // - An `ACDParams` instance constructed with valid ACD(1, 1) parameters.
    //
    // Expect
    // ------
    // - Mapping to θ with `to_theta` and back with `from_theta` yields parameters
    //   equal to the original up to small numerical tolerance.
    fn acdparams_theta_roundtrip_preserves_parameters() {
        // Arrange
        let p = 1usize;
        let q = 1usize;
        let omega = 0.7_f64;
        let alpha = array![0.15_f64];
        let beta = array![0.25_f64];
        let coeff_sum = alpha.sum() + beta.sum();
        let slack = (1.0 - STATIONARITY_MARGIN) - coeff_sum;
        let psi_lags = array![1.2_f64];

        let params =
            ACDParams::new(omega, alpha.clone(), beta.clone(), slack, psi_lags.clone(), p, q)
                .expect("valid parameters should construct");

        let shape = ACDShape { p, q };

        // Act
        let theta = params.to_theta();
        let reconstructed = ACDParams::from_theta(theta.view(), &shape, psi_lags.clone())
            .expect("round-trip from_theta should succeed for theta produced by to_theta");

        // Assert
        let tol = 1e-10_f64;
        assert!((reconstructed.omega - params.omega).abs() < tol);
        assert_eq!(reconstructed.alpha.len(), params.alpha.len());
        assert_eq!(reconstructed.beta.len(), params.beta.len());
        assert_eq!(reconstructed.psi_lags, params.psi_lags);

        for (a, b) in reconstructed.alpha.iter().zip(params.alpha.iter()) {
            assert!((a - b).abs() < tol, "alpha mismatch: {a} vs {b}");
        }
        for (a, b) in reconstructed.beta.iter().zip(params.beta.iter()) {
            assert!((a - b).abs() < tol, "beta mismatch: {a} vs {b}");
        }
        assert!((reconstructed.slack - params.slack).abs() < tol);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `uncond_mean` matches the analytical formula ω / (1 - sum α - sum β).
    //
    // Given
    // -----
    // - An `ACDParams` instance with known ω, α, and β.
    //
    // Expect
    // ------
    // - `uncond_mean()` equals the manually computed value within a small tolerance.
    fn acdparams_uncond_mean_matches_closed_form() {
        // Arrange
        let p = 2usize;
        let q = 1usize;
        let omega = 0.9_f64;
        let alpha = array![0.1_f64];
        let beta = array![0.2_f64, 0.15_f64];
        let coeff_sum = alpha.sum() + beta.sum();
        let slack = (1.0 - STATIONARITY_MARGIN) - coeff_sum;
        let psi_lags = array![1.0_f64, 1.1_f64];

        let params = ACDParams::new(omega, alpha.clone(), beta.clone(), slack, psi_lags, p, q)
            .expect("valid parameters should construct");

        let expected = omega / (1.0 - alpha.sum() - beta.sum());

        // Act
        let mean = params.uncond_mean();

        // Assert
        let tol = 1e-12_f64;
        assert!((mean - expected).abs() < tol, "uncond_mean mismatch: {mean} vs {expected}");
    }

    #[test]
    // Purpose
    // -------
    // Ensure that `ACDParams::to_theta` produces a vector of the expected length.
    //
    // Given
    // -----
    // - An `ACDParams` instance with p beta coefficients and q alpha coefficients.
    //
    // Expect
    // ------
    // - The returned θ has length 1 + q + p.
    fn acdparams_to_theta_has_expected_length() {
        // Arrange
        let p = 3usize;
        let q = 2usize;
        let omega = 0.6_f64;
        let alpha = array![0.05_f64, 0.1_f64];
        let beta = array![0.15_f64, 0.1_f64, 0.05_f64];
        let coeff_sum = alpha.sum() + beta.sum();
        let slack = (1.0 - STATIONARITY_MARGIN) - coeff_sum;
        let psi_lags = array![1.0_f64, 1.1_f64, 1.2_f64];

        let params = ACDParams::new(omega, alpha, beta, slack, psi_lags, p, q)
            .expect("valid parameters should construct");

        // Act
        let theta = params.to_theta();

        // Assert
        assert_eq!(theta.len(), 1 + q + p);
    }
}
