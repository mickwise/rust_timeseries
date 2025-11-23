//! ψ–recursions for ACD(p, q) — conditional means, likelihood, and derivatives.
//!
//! Purpose
//! -------
//! Implement the in-sample ACD(p, q) conditional-mean recursion `ψ_t` and its
//! analytic derivatives with respect to θ = (ω, α₁..α_q, β₁..β_p), reusing
//! preallocated workspace buffers. This module is the core engine
//! for likelihood evaluation and gradient-based optimization.
//!
//! Key behaviors
//! -------------
//! - Seed pre-sample duration and ψ lags according to [`Init`] policies
//!   (`UncondMean`, `SampleMean`, `Fixed`, `FixedVector`).
//! - Run the ψ–recursion in-place, writing `ψ_t` into `ACDScratch::psi_buf`
//!   without heap allocations.
//! - Evaluate the ACD log-likelihood via the chosen innovation law and
//!   `x_t = ψ_t ε_t` change of variables.
//! - Compute ∂ψ_t/∂θ allocation-free into `ACDScratch::deriv_buf` for use in
//!   analytic gradients.
//! - Clamp ψ values into a guarded range [`PsiGuards`] to keep recursions
//!   numerically stable.
//!
//! Invariants & assumptions
//! ------------------------
//! - Parameters `(ω, α, β, slack)` are validated upstream:
//!   - `ω > 0`,
//!   - `α_i ≥ 0`, `β_j ≥ 0`,
//!   - `∑α + ∑β < 1 − STATIONARITY_MARGIN`.
//! - Duration data in [`ACDData`] are strictly positive and non-empty.
//! - The workspace [`WorkSpace`] and model [`ACDModel`] are shape-consistent:
//!   lengths of α/β, lag buffers, and derivative buffers match `(p, q)` and
//!   sample length `n`.
//! - Error handling (invalid parameters, PDF failures) is reported via
//!   [`ACDResult`] and never by panicking, except where explicitly documented
//!   (e.g., invariants enforced by constructors).
//!
//! Conventions
//! -----------
//! - Time indexing is 0-based for code, but ψ–recursions conceptually follow
//!   the Engle–Russell convention:
//!     - `ψ_t = ω + Σ_{i=1..q} α_i τ_{t−i} + Σ_{j=1..p} β_j ψ_{t−j}`.
//! - Lag buffers store the newest element at the end; windows used in the
//!   recursions are reversed tails (newest → oldest) to align with
//!   `[·_{t−1}, …, ·_{t−k}]`.
//! - Buffers live in `ACDScratch` inside [`ACDModel`]; all inner loops operate
//!   on `ndarray` views and `RefCell` borrows only.
//! - The first `p` rows of `deriv_buf` correspond to ψ–lags
//!   `(ψ_{-1}, …, ψ_{-p})`; rows `p..p+n` correspond to
//!   `(ψ_0, …, ψ_{n-1})`.
//!
//! Downstream usage
//! ----------------
//! - Call [`compute_psi`] to (re)compute ψ for given parameters and data
//!   before likelihood or gradient evaluation.
//! - Use [`likelihood_driver`] as the main entry point for evaluating
//!   `ℓ(θ)` for an `ACDModel` instance.
//! - Use [`compute_derivative`] to fill `deriv_buf` with ∂ψ_t/∂θ, then combine
//!   with innovation-specific derivatives for analytic ∂ℓ/∂θ.
//! - This module is internal to the duration/ACD stack; Python bindings and
//!   high-level APIs should call through the model layer rather than directly
//!   accessing the helpers here.
//!
//! Testing notes
//! -------------
//! - Unit tests in this module cover:
//!   - ψ–recursion shape and lag wiring on small ACD(p, q) examples,
//!   - guard behavior for out-of-range ψ values,
//!   - initialization behavior for all [`Init`] variants,
//!   - derivative recursion on toy inputs (shape and obvious special cases).
//! - Integration tests / Python tests should:
//!   - check likelihood values against a reference implementation
//!     on small grids of parameters,
//!   - validate `compute_derivative` via finite-difference checks of
//!     [`likelihood_driver`] on small n, p, q,
//!   - exercise extreme ψ–guard settings to confirm numerical robustness.
use crate::{
    duration::{
        core::{data::ACDData, guards::PsiGuards, init::Init, workspace::WorkSpace},
        errors::ACDResult,
        models::acd::ACDModel,
    },
    optimization::numerical_stability::transformations::STATIONARITY_MARGIN,
};
use ndarray::{Axis, s};

/// Compute the conditional-mean series ψ in place for an ACD(p, q) model.
///
/// Parameters
/// ----------
/// - `params`: `&WorkSpace`
///   Validated model-space parameters and derived quantities for the ACD model.
///   Must expose `omega`, `alpha`, `beta`, and `uncond_mean()`.
/// - `duration_data`: `&ACDData`
///   Observed duration series. Must be non-empty and strictly positive, with
///   an optional burn-in index `t0` enforced upstream.
/// - `model_spec`: `&ACDModel`
///   Model descriptor providing shape `(p, q)`, options (including [`Init`]
///   and [`PsiGuards`]), innovation family, and a scratch workspace with
///   `psi_buf` and `dur_buf`.
///
/// Returns
/// -------
/// ()
///   Writes ψ values into `model_spec.scratch_bufs.psi_buf[p .. p + n]`, where
///   `n = duration_data.data.len()`. No value is returned.
///
/// Errors
/// ------
/// - Never returns an error directly. All validation is assumed to have
///   occurred upstream (in `ACDData::new` and parameter constructors).
///
/// Panics
/// ------
/// - May panic if `duration_data.data` is empty, via the internal call to
///   `.mean().expect("non-empty by ACDData::new")`.
/// - May panic if internal buffer shapes are inconsistent with `(p, q, n)`
///   (programming error).
///
/// Notes
/// -----
/// - Initializes lag buffers according to `model_spec.options.init` via
///   [`extract_init`] before running the recursion.
/// - Each ψ_t is clamped to `model_spec.options.psi_guards` via [`guard_psi`]
///   to prevent extreme values from destabilizing the recursion.
/// - This routine performs no heap allocations and is intended for
///   high-frequency use in optimization loops.
///
/// Examples
/// --------
/// ```rust
/// # // Pseudocode only; types and constructors are crate-specific.
/// # use rust_timeseries::duration::core::psi_recursion::compute_psi;
/// # let (params, data, model) = unimplemented!();
/// compute_psi(&params, &data, &model);
/// # // ψ values are now stored in model.scratch_bufs.psi_buf[p..p + n].
/// ```
pub fn compute_psi(params: &WorkSpace, duration_data: &ACDData, model_spec: &ACDModel) {
    let p = model_spec.shape.p;

    extract_init(
        model_spec,
        &model_spec.options.init,
        params.uncond_mean(),
        duration_data.data.mean().expect("non-empty by ACDData::new"),
        p,
    );
    recursion_loop(model_spec, duration_data, params, duration_data.data.len());
}

/// Evaluate the total log-likelihood ℓ(θ) for an ACD(p, q) model.
///
/// Parameters
/// ----------
/// - `model_spec`: `&ACDModel`
///   Model descriptor providing shape `(p, q)`, innovation family, options,
///   and scratch buffers. Owns a ψ buffer used by [`compute_psi`].
/// - `workspace`: `&WorkSpace`
///   Validated parameter workspace `(ω, α, β, slack, …)` for the model.
/// - `duration_data`: `&ACDData`
///   Observed duration series and optional burn-in index `t0`. Durations must
///   be strictly positive.
///
/// Returns
/// -------
/// `ACDResult<f64>`
///   - `Ok(log_likelihood)` with the scalar log-likelihood:
///       - `ℓ(θ) = Σ_t log f_ε(τ_t / ψ_t) − log ψ_t`
///   - over all in-sample observations after optional burn-in.
///   - `Err(..)` if the innovation PDF/log-PDF fails or a downstream
///     validation error is encountered.
///
/// Errors
/// ------
/// - Propagates any error from:
///   - [`compute_psi`] (via the innovation layer if it ever returns errors),
///   - `innovation.log_pdf_duration(x_t, ψ_t)` for each observation.
///
/// Panics
/// ------
/// - May panic if internal buffer shapes are inconsistent with `(p, q, n)`
///   (programming error).
///
/// Notes
/// -----
/// - Calls [`compute_psi`] on every invocation to ensure ψ is consistent with
///   the provided `workspace`.
/// - Respects `duration_data.t0` by skipping the first `t0` durations and the
///   corresponding ψ values in the likelihood sum.
/// - Never allocates in the hot path; it uses borrowed views into
///   `psi_buf` and `duration_data.data`.
///
/// Examples
/// --------
/// ```rust
/// # // Pseudocode only; types and constructors are crate-specific.
/// # use rust_timeseries::duration::core::psi_recursion::likelihood_driver;
/// # let (model, workspace, data) = unimplemented!();
/// let ll = likelihood_driver(&model, &workspace, &data)?;
/// # assert!(ll.is_finite());
/// # Ok::<(), _>(())
/// ```
pub fn likelihood_driver(
    model_spec: &ACDModel, workspace: &WorkSpace, duration_data: &ACDData,
) -> ACDResult<f64> {
    let mut start_idx = model_spec.shape.p;
    let duration_series = match duration_data.t0 {
        Some(t0) => {
            start_idx += t0;
            duration_data.data.slice(s![t0..])
        }
        None => duration_data.data.view(),
    };
    compute_psi(workspace, duration_data, model_spec);
    let binding = model_spec.scratch_bufs.psi_buf.borrow();
    let psi_view = binding.slice(s![start_idx..]);
    duration_series
        .iter()
        .zip(psi_view.iter())
        .try_fold(0.0, |acc, (x, psi)| Ok(acc + model_spec.innovation.log_pdf_duration(*x, *psi)?))
}

/// Compute ∂ψ_t/∂θ for all in-sample times and write into `deriv_buf`.
///
/// Parameters
/// ----------
/// - `params`: `&WorkSpace`
///   Validated parameter workspace providing `(ω, α, β, slack, …)` and any
///   derived quantities needed for the derivative of initialization schemes.
/// - `duration_data`: `&ACDData`
///   Observed duration series used by the derivative recursion.
/// - `model_spec`: `&ACDModel`
///   Model descriptor holding shape `(p, q)`, initialization options, and the
///   derivative buffer `scratch_bufs.deriv_buf` of shape `(n + p, 1 + q + p)`.
///
/// Returns
/// -------
/// ()
///   Fills `model_spec.scratch_bufs.deriv_buf` with rows representing
///   ∂ψ_t/∂θ for `t = -p, …, n-1` (lags first, then in-sample times).
///
/// Errors
/// ------
/// - Never returns an error; invalid parameter configurations should have
///   been rejected upstream.
///
/// Panics
/// ------
/// - May panic if `deriv_buf` shape is inconsistent with `(n + p, 1 + q + p)`
///   (programming error).
///
/// Notes
/// -----
/// - Uses [`extract_init_derivative`] to seed the first `p` rows corresponding
///   to ψ–lags `(ψ_{-1}, …, ψ_{-p})`.
/// - Applies [`recursion_loop_derivative`] to propagate sensitivities forward
///   through the recursion, incorporating β feedback terms.
/// - Intended to be combined with innovation-specific derivatives to obtain
///   analytic ∂ℓ/∂θ.
///
/// Examples
/// --------
/// ```rust
/// # // Pseudocode only; types and constructors are crate-specific.
/// # use rust_timeseries::duration::core::psi_recursion::compute_derivative;
/// # let (params, data, model) = unimplemented!();
/// compute_derivative(&params, &data, &model);
/// # // model.scratch_bufs.deriv_buf now holds ∂ψ_t/∂θ rows.
/// ```
pub fn compute_derivative(params: &WorkSpace, duration_data: &ACDData, model_spec: &ACDModel) {
    let p = model_spec.shape.p;
    extract_init_derivative(model_spec, &model_spec.options.init, params, p);
    recursion_loop_derivative(model_spec, duration_data, params, duration_data.data.len());
}

/// Clamp a ψ value into the `[min, max]` range defined by `PsiGuards`.
///
/// Parameters
/// ----------
/// - `value`: `f64`
///   Proposed ψ value from the recursion.
/// - `guards`: `&PsiGuards`
///   Guard structure with fields `min` and `max` defining the allowed ψ range.
///
/// Returns
/// -------
/// `f64`
///   - `guards.min` if `value < guards.min`,
///   - `guards.max` if `value > guards.max`,
///   - `value` otherwise (unchanged).
///
/// Errors
/// ------
/// - Never returns an error.
///
/// Panics
/// ------
/// - Never panics. Assumes `guards.min <= guards.max` is enforced upstream.
///
/// Notes
/// -----
/// - Used both during in-sample fitting (`compute_psi`) and forecasting to
///   prevent extreme ψ values from destabilizing subsequent recursions.
/// - Guard ranges should be chosen wide enough not to bias inference, but
///   tight enough to prevent underflow/overflow in the innovation log-PDF.
///
/// Examples
/// --------
/// ```rust
/// # use rust_timeseries::duration::core::guards::PsiGuards;
/// # use rust_timeseries::duration::core::psi_recursion::guard_psi;
/// let guards = PsiGuards { min: 1e-6, max: 1e6 };
/// assert_eq!(guard_psi(1e-9, &guards), 1e-6);
/// assert_eq!(guard_psi(1e3, &guards), 1e3);
/// ```
pub fn guard_psi(value: f64, guards: &PsiGuards) -> f64 {
    let min_guard = guards.min;
    let max_guard = guards.max;
    if value < min_guard {
        min_guard
    } else if value > max_guard {
        max_guard
    } else {
        value
    }
}

// ---- Helper Methods ----

/// Seed the pre-sample ψ and duration lag buffers according to [`Init`].
///
/// Parameters
/// ----------
/// - `model_spec`: `&ACDModel`
///   Model descriptor providing access to the scratch buffers:
///   `scratch_bufs.psi_buf` and `scratch_bufs.dur_buf`. The first `p` entries
///   of `psi_buf` and the first `q` entries of `dur_buf` are treated as
///   pre-sample lags.
/// - `init_specs`: `&Init`
///   Initialization policy specifying how pre-sample ψ and duration lags
///   should be seeded (`UncondMean`, `SampleMean`, `Fixed`, `FixedVector`).
/// - `uncond_mean`: `f64`
///   Unconditional mean μ implied by the current parameters `(ω, α, β)`,
///   typically computed via [`WorkSpace::uncond_mean`].
/// - `sample_mean`: `f64`
///   Sample mean of the observed durations, precomputed upstream as
///   `duration_data.data.mean().unwrap()`.
/// - `p`: `usize`
///   Number of ψ lags. The duration lag count `q` is inferred from
///   `model_spec.shape.q`.
///
/// Returns
/// -------
/// ()
///   Writes pre-sample values into the ψ and duration lag buffers in-place.
///   No value is returned.
///
/// Side effects
/// ------------
/// - Overwrites:
///   - `model_spec.scratch_bufs.psi_buf[..p]` with initial ψ-lags
///     `(ψ_{-1}, …, ψ_{-p})`.
///   - `model_spec.scratch_bufs.dur_buf[..q]` with initial duration lags
///     `(τ_{-q}, …, τ_{-1})`.
///
/// Errors
/// ------
/// - Never returns an error; invalid `Init` variants should have been rejected
///   at construction time (e.g., by `Init::fixed` / `Init::fixed_vector`).
///
/// Panics
/// ------
/// - May panic if:
///   - `psi_buf` has length `< p`, or
///   - `dur_buf` has length `< q`, or
///   - `Init::FixedVector` contains vectors whose lengths are inconsistent
///     with `(p, q)`.
///
/// Notes
/// -----
/// - `Init::UncondMean`
///   Fills both ψ and duration lags with `uncond_mean`.
/// - `Init::SampleMean`
///   Fills both lag buffers with `sample_mean`.
/// - `Init::Fixed(value)`
///   Fills both lag buffers with the given positive scalar.
/// - `Init::FixedVector { psi_lags, duration_lags }`
///   Copies the provided vectors directly into the ψ and duration lag slices
///   without further validation.
///
/// Examples
/// --------
/// ```rust
/// # // Pseudocode only; types and constructors are crate-specific.
/// # use rust_timeseries::duration::core::psi_recursion::extract_init;
/// # use rust_timeseries::duration::core::init::Init;
/// # let (model, ws, data) = unimplemented!();
/// let p = model.shape.p;
/// let uncond_mean = ws.uncond_mean();
/// let sample_mean = data.data.mean().unwrap();
/// extract_init(&model, &Init::UncondMean, uncond_mean, sample_mean, p);
/// # // model.scratch_bufs.psi_buf[..p] and dur_buf[..q] are now seeded.
/// ```
fn extract_init(
    model_spec: &ACDModel, init_specs: &Init, uncond_mean: f64, sample_mean: f64, p: usize,
) {
    let mut binding = model_spec.scratch_bufs.psi_buf.borrow_mut();
    let mut init_psi_buf = binding.slice_mut(s![..p]);
    let mut init_dur_buf = model_spec.scratch_bufs.dur_buf.borrow_mut();
    match init_specs {
        Init::UncondMean => {
            init_psi_buf.fill(uncond_mean);
            init_dur_buf.fill(uncond_mean);
        }
        Init::SampleMean => {
            init_psi_buf.fill(sample_mean);
            init_dur_buf.fill(sample_mean);
        }
        Init::Fixed(val) => {
            init_psi_buf.fill(*val);
            init_dur_buf.fill(*val);
        }
        Init::FixedVector { psi_lags, duration_lags } => {
            init_psi_buf.assign(&psi_lags.view());
            init_dur_buf.assign(&duration_lags.view());
        }
    }
}

/// Seed the ψ–derivative buffer for pre-sample lags.
///
/// Parameters
/// ----------
/// - `model_spec`: `&ACDModel`
///   Model descriptor providing `scratch_bufs.deriv_buf`, the derivative
///   buffer of shape `(n + p, 1 + q + p)`. The first `p` rows correspond to
///   derivatives of pre-sample ψ-lags `(ψ_{-1}, …, ψ_{-p})`.
/// - `init_specs`: `&Init`
///   Initialization policy governing how pre-sample ψ-lags depend on the
///   parameter vector θ. Only `Init::UncondMean` induces non-zero derivatives
///   here; `SampleMean` and `Fixed*` policies are treated as θ-independent.
/// - `params`: `&WorkSpace`
///   Parameter workspace providing `omega`, `alpha`, `beta`, and `slack`. Used
///   to compute the derivative of the unconditional mean μ with respect to
///   θ when `Init::UncondMean` is selected.
/// - `p`: `usize`
///   Number of ψ-lags. Determines how many rows at the top of `deriv_buf`
///   correspond to pre-sample lags.
///
/// Returns
/// -------
/// ()
///   Writes the first `p` rows of `deriv_buf` in-place. No value is returned.
///
/// Side effects
/// ------------
/// - Overwrites `model_spec.scratch_bufs.deriv_buf[..p, ..]` with the
///   derivatives of `(ψ_{-1}, …, ψ_{-p})` w.r.t. θ = (ω, α₁..α_q, β₁..β_p).
///
/// Errors
/// ------
/// - Never returns an error. Assumes that `params` and buffer shapes are
///   consistent with the model specification.
///
/// Panics
/// ------
/// - May panic if:
///   - `deriv_buf` has fewer than `p` rows, or
///   - its column count is less than `1 + q + p`.
///
/// Notes
/// -----
/// - For `Init::UncondMean`, initial ψ-lags are set to the unconditional mean
///   μ = ω / (1 − ∑α − ∑β), and derivatives are:
///   - ∂μ/∂ω   = 1 / (1 − ∑α − ∑β) ≈ `1 / (params.slack + STATIONARITY_MARGIN)`
///   - ∂μ/∂αᵢ  = ∂μ/∂βⱼ = ω / (1 − ∑α − ∑β)²
/// - These are written into the ω, α, β columns, and replicated across the
///   first `p` rows.
/// - For `Init::SampleMean` and all `Init::Fixed*` variants, initial ψ-lags
///   are treated as constants w.r.t. θ, so the corresponding derivative rows
///   are zeroed.
///
/// Examples
/// --------
/// ```rust
/// # // Pseudocode only.
/// # use rust_timeseries::duration::core::psi_recursion::extract_init_derivative;
/// # use rust_timeseries::duration::core::init::Init;
/// # let (model, workspace) = unimplemented!();
/// let p = model.shape.p;
/// extract_init_derivative(&model, &Init::UncondMean, &workspace, p);
/// # // deriv_buf[..p, ..] now holds ∂ψ_lag/∂θ rows.
/// ```
fn extract_init_derivative(model_spec: &ACDModel, init_specs: &Init, params: &WorkSpace, p: usize) {
    let mut binding = model_spec.scratch_bufs.deriv_buf.borrow_mut();
    let mut init_psi_deriv_buf = binding.slice_mut(s![..p, ..]);
    match init_specs {
        Init::UncondMean => {
            let denominator = params.slack + STATIONARITY_MARGIN;
            init_psi_deriv_buf.column_mut(0).fill(1.0 / denominator);
            init_psi_deriv_buf
                .slice_mut(s![.., 1..])
                .fill(params.omega / (denominator * denominator));
        }
        _ => {
            init_psi_deriv_buf.fill(0.0);
        }
    }
}

/// Core allocation-free ACD(p, q) ψ–recursion over the sample.
///
/// Parameters
/// ----------
/// - `model_spec`: `&ACDModel`
///   Model descriptor providing:
///   - `shape.p`, `shape.q` for the ACD orders,
///   - `options.psi_guards` for clamping,
///   - scratch buffers `psi_buf` and `dur_buf`.
/// - The first `p` entries of `psi_buf` and the first `q` entries of `dur_buf`
///   must already be initialized by [`extract_init`].
/// - `duration_data`: `&ACDData`
///   Observed duration series of length `n` used to build the τ-lag vectors.
///   Must be strictly positive and non-empty (validated by `ACDData::new`).
/// - `params`: `&WorkSpace`
///   Parameter workspace providing `omega`, `alpha` (len = q), and `beta`
///   (len = p) used in the recursion.
/// - `n`: `usize`
///   Number of in-sample observations over which to run the recursion. Must
///   satisfy `n <= duration_data.data.len()`.
///
/// Returns
/// -------
/// ()
///   Writes ψ values into `psi_buf[p .. p + n]` in-place; no value is returned.
///
/// Side effects
/// ------------
/// - For each `t` in `0..n`, computes:
///     - `ψ_t = ω + αᵀ[τ_{t-1}, …, τ_{t-q}] + βᵀ[ψ_{t-1}, …, ψ_{t-p}]`
/// - and stores it at `psi_buf[p + t]`, after applying [`guard_psi`].
///
/// Errors
/// ------
/// - Never returns an error; all input validation (shapes, positivity) is
///   assumed to have been done upstream.
///
/// Panics
/// ------
/// - May panic if:
///   - `psi_buf` does not have length at least `p + n`,
///   - `dur_buf` does not have length at least `q`,
///   - or `duration_data.data` has length `< n`.
///
/// Notes
/// -----
/// - Duration lags are constructed with a single split per time step:
///   - `k_init = max(0, q − t)` lags are taken from the pre-sample duration
///     buffer `dur_buf`,
///   - `k_data = q − k_init` lags are taken from `duration_data.data`.
/// - Within each segment, lags are read in **reverse** (newest → oldest) so
///   they align with the coefficient ordering `[τ_{t-1}, …, τ_{t-q}]`.
/// - ψ-lags are read as `psi_buf[t .. t + p]`, corresponding to
///   `[ψ_{t-1}, …, ψ_{t-p}]`.
/// - Each raw ψ_t is clamped via [`guard_psi`] using `model_spec.options.psi_guards`.
///
/// Examples
/// --------
/// ```rust
/// # // Pseudocode only.
/// # use rust_timeseries::duration::core::psi_recursion::recursion_loop;
/// # let (model, ws, data) = unimplemented!();
/// let n = data.data.len();
/// recursion_loop(&model, &data, &ws, n);
/// # // ψ values are now in model.scratch_bufs.psi_buf[p..p + n].
/// ```
fn recursion_loop(model_spec: &ACDModel, duration_data: &ACDData, params: &WorkSpace, n: usize) {
    let p = model_spec.shape.p;
    let q = model_spec.shape.q;
    let psi_guards = &model_spec.options.psi_guards;
    let omega = params.omega;
    let alpha = &params.alpha;
    let beta = &params.beta;

    let mut psi_buf = model_spec.scratch_bufs.psi_buf.borrow_mut();
    let dur_init = model_spec.scratch_bufs.dur_buf.borrow();

    for t in 0..n {
        // how many init vs observed lags to use
        let k_init = q.saturating_sub(t);
        let k_data = q - k_init;

        let psi_lags = psi_buf.slice(s![t..t + p]);

        let init_tail_rev = dur_init.slice(s![q - k_init .. q; -1]);
        let data_tail_rev = duration_data.data.slice(s![t - k_data .. t; -1]);

        let sum_alpha = alpha.slice(s![0..k_init]).dot(&init_tail_rev)
            + alpha.slice(s![k_init..q]).dot(&data_tail_rev);

        let mut new_psi = omega + sum_alpha + beta.dot(&psi_lags);
        new_psi = guard_psi(new_psi, psi_guards);
        psi_buf[p + t] = new_psi;
    }
}

/// Core allocation-free ACD(p, q) ψ–derivative recursion over the sample.
///
/// Parameters
/// ----------
/// - `model_spec`: `&ACDModel`
///   Model descriptor providing:
///   - `shape.p`, `shape.q` for the ACD orders,
///   - scratch buffers:
///     - `dur_buf` for pre-sample duration lags,
///     - `psi_buf` holding ψ-lags and in-sample ψ values,
///     - `deriv_buf` of shape `(n + p, 1 + q + p)` used to store
///       ∂ψ_t/∂θ rows.
/// - The first `p` rows of `deriv_buf` must already have been seeded by
///   [`extract_init_derivative`], and ψ-lags must be present in `psi_buf`.
/// - `duration_data`: `&ACDData`
///   Observed durations of length `n` used to build τ-lag vectors.
/// - `params`: `&WorkSpace`
///   Parameter workspace providing the β coefficients used in the recursive
///   sensitivity term Σ_j β_j ∂ψ_{t-j}/∂θ.
/// - `n`: `usize`
///   Number of in-sample observations over which to run the derivative
///   recursion. Must satisfy `n <= duration_data.data.len()`.
///
/// Returns
/// -------
/// ()
///   Writes ∂ψ_t/∂θ into rows `p .. p + n` of `deriv_buf` in-place.
///   No value is returned.
///
/// Side effects
/// ------------
/// - For each `t` in `0..n`, overwrites row `p + t` of `deriv_buf` with:
///   - a static part:
///     - `e_ω` in the ω column,
///     - `[τ_{t-1}, …, τ_{t-q}]` in the α columns,
///     - `[ψ_{t-1}, …, ψ_{t-p}]` in the β columns,
///   - plus a recursive part:
///     - Σ_{j=1}^p β_j · ∂ψ_{t-j}/∂θ taken from rows `p + t - j`.
///
/// Errors
/// ------
/// - Never returns an error; assumes upstream validation of shapes and data.
///
/// Panics
/// ------
/// - May panic if:
///   - `deriv_buf` has fewer than `p + n` rows,
///   - its column count is less than `1 + q + p`,
///   - `psi_buf` is too short for the window `t .. t + p`,
///   - or `duration_data.data` has length `< n`.
///
/// Notes
/// -----
/// - Duration lags are constructed exactly as in [`recursion_loop`], via:
///   - `k_init = max(0, q − t)` lags from `dur_buf`,
///   - `k_data = q − k_init` lags from `duration_data.data`,
/// - both read in reverse to align with `[τ_{t-1}, …, τ_{t-q}]`.
/// - ψ-lags are read from `psi_buf[t .. t + p]` for the β block.
/// - Each derivative row is **zeroed** before being filled to avoid stale
///   values from previous runs.
/// - When `model_spec.options.init` is `Init::UncondMean`, each row also
///   includes the analytic contribution of pre-sample duration lags to the
///   derivative via the unconditional mean μ. This appears as:
///   * an additive term `Σ_{i ∈ pre} α_i · ∂μ/∂ω` in the ω column, and
///   * a constant shift `Σ_{i ∈ pre} α_i · ∂μ/∂α_j` / `∂μ/∂β_j` applied
///     uniformly across the α and β columns for that row,
/// - where the sum runs over α-weights on pre-sample lags.
/// - This routine does not itself apply ψ-guards; it differentiates the
///   unconstrained recursion and then applies β-feedback. Guarding behavior
///   is accounted for in how ψ is computed upstream.
///
/// Examples
/// --------
/// ```rust
/// # // Pseudocode only.
/// # use rust_timeseries::duration::core::psi_recursion::recursion_loop_derivative;
/// # let (model, ws, data) = unimplemented!();
/// let n = data.data.len();
/// recursion_loop_derivative(&model, &data, &ws, n);
/// # // deriv_buf[p..p + n, ..] now holds ∂ψ_t/∂θ for in-sample t.
/// ```
fn recursion_loop_derivative(
    model_spec: &ACDModel, duration_data: &ACDData, params: &WorkSpace, n: usize,
) {
    let p = model_spec.shape.p;
    let q = model_spec.shape.q;
    let omega = params.omega;
    let dur_init = model_spec.scratch_bufs.dur_buf.borrow();
    let psi_buf = model_spec.scratch_bufs.psi_buf.borrow();
    let beta = &params.beta;
    let mut binding = model_spec.scratch_bufs.deriv_buf.borrow_mut();
    for t in 0..n {
        let deriv_buf = binding.view_mut();
        let k_init = q.saturating_sub(t);
        let k_data = q - k_init;

        let (deriv_lags_tail, mut deriv_lags_head) = deriv_buf.split_at(Axis(0), p + t);

        let init_tail_rev = dur_init.slice(s![q - k_init .. q; -1]);
        let data_tail_rev = duration_data.data.slice(s![t - k_data .. t; -1]);
        let alpha_sum = calculate_alpha_sum_for_uncond_mean(model_spec, params, k_init);
        let uncond_mean_denom = params.slack + STATIONARITY_MARGIN;
        let parameter_duration_deriv =
            (alpha_sum * omega) / (uncond_mean_denom * uncond_mean_denom);

        let mut curr_row = deriv_lags_head.row_mut(0);
        curr_row.fill(0.0);
        curr_row[0] = 1.0 + alpha_sum / uncond_mean_denom;
        curr_row.slice_mut(s![1..k_init + 1]).assign(&init_tail_rev);
        curr_row.slice_mut(s![k_init + 1..q + 1]).assign(&data_tail_rev);
        curr_row.slice_mut(s![q + 1..]).assign(&psi_buf.slice(s![t..t + p]));
        curr_row.slice_mut(s![1..]).iter_mut().for_each(|x| *x += parameter_duration_deriv);
        for j in 1..=p {
            let curr_beta = beta[j - 1];
            curr_row.scaled_add(curr_beta, &deriv_lags_tail.row(p + t - j));
        }
    }
}

/// Compute the sum of α-weights on pre-sample duration lags under `Init::UncondMean`.
///
/// Parameters
/// ----------
/// - `model_spec`: `&ACDModel`
///   ACD model descriptor providing the initialization policy via
///   `model_spec.options.init` and the ACD order `shape.q`. Only the
///   initialization policy is inspected by this helper.
/// - `params`: `&WorkSpace`
///   Parameter workspace providing the α vector. The first `k_init` entries
///   of `params.alpha` are assumed to correspond to the pre-sample duration
///   lags used at the current time step.
/// - `k_init`: `usize`
///   Number of duration lags taken from the pre-sample buffer for the current
///   recursion time `t` (i.e., the length of the prefix of α that multiplies
///   pre-sample durations). Must satisfy `k_init <= params.alpha.len()`.
///
/// Returns
/// -------
/// `f64`
///   The sum of α-weights over pre-sample duration lags,
///   `Σ_{i=1}^{k_init} α_i`, when `Init::UncondMean` is active. Returns `0.0`
///   for all other initialization policies, so callers can treat the result
///   as a no-op correction in those cases.
///
/// Errors
/// ------
/// - Never returns an error. All consistency checks on model shape and
///   parameter dimensions are assumed to hold upstream.
///
/// Panics
/// ------
/// - Never panics as long as `k_init <= params.alpha.len()`.
///
/// Safety
/// ------
/// - Never `unsafe`. No additional caller guarantees are required beyond
///   shape consistency enforced elsewhere.
///
/// Notes
/// -----
/// - This helper is used to factor out the dependence of pre-sample duration
///   lags on the unconditional mean μ when `Init::UncondMean` is selected.
///   The returned sum is combined with analytic derivatives of μ to construct
///   the static corrections to the ω, α, and β columns in the ψ–derivative
///   recursion.
/// - The function performs no heap allocations and operates only on views
///   into the existing α buffer.
///
/// Examples
/// --------
/// ```rust
/// # use crate::duration::core::psi_recursion::calculate_alpha_sum_for_uncond_mean;
/// # // assuming `model` and `workspace` have been constructed
/// let k_init = 2usize;
/// let alpha_sum = calculate_alpha_sum_for_uncond_mean(&model, &workspace, k_init);
/// # assert!(alpha_sum >= 0.0);
/// ```
fn calculate_alpha_sum_for_uncond_mean(
    model_spec: &ACDModel, params: &WorkSpace, k_init: usize,
) -> f64 {
    match model_spec.options.init {
        Init::UncondMean => {
            let alpha = &params.alpha;
            alpha.slice(s![..k_init]).sum()
        }
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duration::core::{
        data::{ACDData, ACDMeta},
        innovations::ACDInnovation,
        options::ACDOptions,
        shape::ACDShape,
        units::ACDUnit,
    };
    use crate::optimization::loglik_optimizer::MLEOptions;
    use ndarray::{Array1, array};

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - ψ guard behavior (`guard_psi` lower/upper clamping).
    // - Initialization behavior for [`Init::UncondMean`] and [`Init::Fixed`]
    //   via [`extract_init`].
    // - ψ–recursion wiring via [`compute_psi`] on a small ACD(1, 1) example
    //   where ψ_t should stay at the unconditional mean.
    // - Initialization of derivative rows via [`extract_init_derivative`] for
    //   both `Init::UncondMean` and non–`UncondMean` variants.
    //
    // They intentionally DO NOT cover:
    // - Likelihood evaluation and innovation-specific behavior
    //   (covered by higher-level tests on [`ACDModel`] and
    //   [`ACDInnovation`]).
    // - Full gradient correctness for `compute_derivative` (that is exercised
    //   via model-level gradient/optimizer tests).
    // -------------------------------------------------------------------------

    // Helper: minimal ACDMeta stub reused across tests.
    fn make_meta_stub() -> ACDMeta {
        ACDMeta::new(ACDUnit::Seconds, None, false)
    }

    // Helper: build a small ACDModel with given (p, q), n and init/guards.
    fn make_model(p: usize, q: usize, n: usize, init: Init, psi_guards: PsiGuards) -> ACDModel {
        let shape = ACDShape { p, q };
        let innovation = ACDInnovation::exponential();
        let mle_opts = MLEOptions::default();
        let opts = ACDOptions::new(init, mle_opts, psi_guards);
        ACDModel::new(shape, innovation, opts, n)
    }

    // Helper: build a workspace for given shape over caller-owned α/β buffers.
    fn make_workspace<'a>(
        shape: &ACDShape, alpha_buf: &'a mut Array1<f64>, beta_buf: &'a mut Array1<f64>,
    ) -> WorkSpace<'a> {
        WorkSpace::new(alpha_buf.view_mut(), beta_buf.view_mut(), shape).unwrap()
    }

    #[test]
    // Purpose
    // -------
    // Verify that `guard_psi` clamps values below `min` and above `max`, and
    // returns in-range values unchanged.
    //
    // Given
    // -----
    // - `PsiGuards { min: 1e-3, max: 1e3 }`.
    // - Values: `1e-6` (below min), `1.0` (inside), `1e6` (above max).
    //
    // Expect
    // ------
    // - `guard_psi(1e-6) == 1e-3`.
    // - `guard_psi(1.0) == 1.0`.
    // - `guard_psi(1e6) == 1e3`.
    fn guard_psi_clamps_to_bounds() {
        // Arrange
        let guards = PsiGuards { min: 1e-3, max: 1e3 };

        // Act + Assert
        assert_eq!(guard_psi(1e-6, &guards), guards.min);
        assert_eq!(guard_psi(1.0, &guards), 1.0);
        assert_eq!(guard_psi(1e6, &guards), guards.max);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `extract_init` with `Init::UncondMean` fills both ψ and
    // duration lag buffers with the unconditional mean.
    //
    // Given
    // -----
    // - ACD(2, 2) model with `Init::UncondMean`.
    // - `uncond_mean = 1.5`, arbitrary `sample_mean`.
    //
    // Expect
    // ------
    // - `psi_buf[..p]` and `dur_buf[..q]` are all equal to `uncond_mean`.
    fn extract_init_uncond_mean_seeds_lags_with_uncond_mean() {
        // Arrange
        let p = 2usize;
        let q = 2usize;
        let n = 3usize;
        let psi_guards = PsiGuards { min: 1e-6, max: 1e6 };
        let model = make_model(p, q, n, Init::UncondMean, psi_guards);

        let shape = ACDShape { p, q };
        let mut alpha_buf = array![0.2, 0.1];
        let mut beta_buf = array![0.1, 0.05];
        let mut ws = make_workspace(&shape, &mut alpha_buf, &mut beta_buf);
        ws.omega = 1.5;
        ws.slack = 1.0 - STATIONARITY_MARGIN - ws.alpha.sum() - ws.beta.sum(); // keep invariants consistent

        let uncond_mean = ws.uncond_mean();
        let data = array![1.0, 2.0, 3.0];
        let acd_data = ACDData::new(data, None, make_meta_stub()).unwrap();
        let sample_mean = acd_data.data.mean().unwrap();

        // Act
        extract_init(&model, &Init::UncondMean, uncond_mean, sample_mean, p);

        // Assert
        let psi_buf = model.scratch_bufs.psi_buf.borrow();
        let dur_buf = model.scratch_bufs.dur_buf.borrow();

        for v in psi_buf.slice(s![..p]).iter() {
            assert!((v - uncond_mean).abs() < 1e-12);
        }
        for v in dur_buf.slice(s![..q]).iter() {
            assert!((v - uncond_mean).abs() < 1e-12);
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `extract_init` with `Init::Fixed(val)` fills both ψ and
    // duration lags with the provided scalar.
    //
    // Given
    // -----
    // - ACD(2, 2) model with `Init::Fixed(7.0)`.
    //
    // Expect
    // ------
    // - `psi_buf[..p]` and `dur_buf[..q]` are all equal to 7.0.
    fn extract_init_fixed_seeds_lags_with_fixed_value() {
        // Arrange
        let p = 2usize;
        let q = 2usize;
        let n = 3usize;
        let psi_guards = PsiGuards { min: 1e-6, max: 1e6 };
        let model = make_model(p, q, n, Init::Fixed(7.0), psi_guards);

        let uncond_mean = 1.23;
        let sample_mean = 4.56;

        // Act
        extract_init(&model, &Init::Fixed(7.0), uncond_mean, sample_mean, p);

        // Assert
        let psi_buf = model.scratch_bufs.psi_buf.borrow();
        let dur_buf = model.scratch_bufs.dur_buf.borrow();

        for v in psi_buf.slice(s![..p]).iter() {
            assert!((*v - 7.0).abs() < 1e-12);
        }
        for v in dur_buf.slice(s![..q]).iter() {
            assert!((*v - 7.0).abs() < 1e-12);
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `compute_psi` with `Init::UncondMean` and constant durations
    // keeps ψ_t identically equal to the unconditional mean μ for ACD(1, 1).
    //
    // Given
    // -----
    // - ACD(1, 1) with parameters (ω, α, β) chosen to satisfy stationarity.
    // - `Init::UncondMean` and ψ-guards wide enough not to bind.
    // - Duration series `τ_t ≡ μ`, where μ is the workspace unconditional mean.
    //
    // Expect
    // ------
    // - Pre-sample ψ-lag `ψ_{-1}` equals μ.
    // - In-sample ψ values `ψ_0, …, ψ_{n-1}` are all equal to μ.
    fn compute_psi_uncond_mean_constant_durations_keeps_psi_at_uncond_mean() {
        // Arrange
        let p = 1usize;
        let q = 1usize;
        let n = 4usize;
        let shape = ACDShape { p, q };

        let mut alpha_buf = array![0.2];
        let mut beta_buf = array![0.3];
        let mut ws = make_workspace(&shape, &mut alpha_buf, &mut beta_buf);
        ws.omega = 1.5;
        ws.slack = 1.0 - STATIONARITY_MARGIN - ws.alpha.sum() - ws.beta.sum(); // keep invariants consistent

        let mu = ws.uncond_mean();

        let data = array![mu, mu, mu, mu];
        let acd_data = ACDData::new(data, None, make_meta_stub()).unwrap();

        let psi_guards = PsiGuards { min: 1e-9, max: 1e9 };
        let model = make_model(p, q, n, Init::UncondMean, psi_guards);

        // Act
        compute_psi(&ws, &acd_data, &model);

        // Assert
        let psi_buf = model.scratch_bufs.psi_buf.borrow();
        // First p entries are ψ-lags; next n entries are in-sample ψ values.
        let psi_lags = psi_buf.slice(s![..p]);
        let psi_in_sample = psi_buf.slice(s![p..p + n]);

        for v in psi_lags.iter() {
            assert!((v - mu).abs() < 1e-10);
        }
        for v in psi_in_sample.iter() {
            assert!((v - mu).abs() < 1e-10);
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `extract_init_derivative` for `Init::UncondMean` seeds the
    // first `p` rows of `deriv_buf` with the analytic derivatives of μ
    // w.r.t. (ω, α, β).
    //
    // Given
    // -----
    // - ACD(1, 1) with parameters stored in a `WorkSpace`.
    // - `Init::UncondMean` and arbitrary ψ-guards.
    //
    // Expect
    // ------
    // - For the first row of `deriv_buf`:
    //   * derivative w.r.t. ω equals `1 / (slack + STATIONARITY_MARGIN)`,
    //   * derivatives w.r.t. α and β both equal `ω / (slack + STATIONARITY_MARGIN)^2`.
    fn extract_init_derivative_uncond_mean_sets_expected_rows() {
        // Arrange
        let p = 1usize;
        let q = 1usize;
        let n = 3usize;
        let shape = ACDShape { p, q };

        let mut alpha_buf = array![0.25];
        let mut beta_buf = array![0.25];
        let mut ws = make_workspace(&shape, &mut alpha_buf, &mut beta_buf);
        ws.omega = 2.0;
        ws.slack = 1.0 - STATIONARITY_MARGIN - ws.alpha.sum() - ws.beta.sum(); // match optimizer mapping

        let psi_guards = PsiGuards { min: 1e-6, max: 1e6 };
        let model = make_model(p, q, n, Init::UncondMean, psi_guards);
        let init = Init::UncondMean;

        // Act
        extract_init_derivative(&model, &init, &ws, p);

        // Assert
        let deriv_buf = model.scratch_bufs.deriv_buf.borrow();
        let row0 = deriv_buf.row(0);

        let denominator = ws.slack + STATIONARITY_MARGIN;
        let expected_domega = 1.0 / denominator;
        let expected_dweights = ws.omega / (denominator * denominator);

        // Columns: [ω, α₁, β₁]
        assert!((row0[0] - expected_domega).abs() < 1e-12);
        assert!((row0[1] - expected_dweights).abs() < 1e-12);
        assert!((row0[2] - expected_dweights).abs() < 1e-12);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `extract_init_derivative` zeroes the first `p` rows of
    // `deriv_buf` for initialization policies other than `Init::UncondMean`.
    //
    // Given
    // -----
    // - ACD(1, 1) model with `Init::SampleMean`.
    // - `deriv_buf[..p, ..]` pre-filled with non-zero values.
    //
    // Expect
    // ------
    // - After calling `extract_init_derivative`, the first `p` rows are all
    //   zeros.
    fn extract_init_derivative_non_uncond_mean_zeroes_rows() {
        // Arrange
        let p = 1usize;
        let q = 1usize;
        let n = 2usize;
        let shape = ACDShape { p, q };

        let mut alpha_buf = array![0.1];
        let mut beta_buf = array![0.2];
        let mut ws = make_workspace(&shape, &mut alpha_buf, &mut beta_buf);
        ws.omega = 1.0;
        ws.slack = 1.0 - STATIONARITY_MARGIN - ws.alpha.sum() - ws.beta.sum(); // keep consistent

        let psi_guards = PsiGuards { min: 1e-6, max: 1e6 };
        let model = make_model(p, q, n, Init::SampleMean, psi_guards);
        let init = Init::SampleMean;

        // Pre-fill deriv_buf with non-zero values to ensure they are overwritten.
        {
            let mut deriv_buf = model.scratch_bufs.deriv_buf.borrow_mut();
            deriv_buf.fill(42.0);
        }

        // Act
        extract_init_derivative(&model, &init, &ws, p);

        // Assert
        let deriv_buf = model.scratch_bufs.deriv_buf.borrow();
        let row0 = deriv_buf.row(0);
        for v in row0.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that `likelihood_driver` matches a manual sum of
    // `innovation.log_pdf_duration` over the in-sample window, including
    // respect for an in-sample burn-in index `t0`.
    //
    // Given
    // -----
    // - ACD(1, 1) model with valid parameters.
    // - Duration series of length 4.
    // - `t0 = Some(1)` so the first observation is skipped in the likelihood.
    //
    // Expect
    // ------
    // - `likelihood_driver` returns the same scalar as an explicit loop over
    //   `t = t0 .. n-1` summing `innovation.log_pdf_duration(τ_t, ψ_t)`, where
    //   `ψ_t` are taken from `psi_buf[p + t]` after calling `compute_psi`.
    fn likelihood_driver_matches_manual_sum_with_t0() {
        // Arrange
        let p = 1usize;
        let q = 1usize;
        let n = 4usize;
        let shape = ACDShape { p, q };

        let mut alpha_buf = array![0.2];
        let mut beta_buf = array![0.3];
        let mut ws = make_workspace(&shape, &mut alpha_buf, &mut beta_buf);
        ws.omega = 1.5;
        ws.slack = 1.0 - STATIONARITY_MARGIN - ws.alpha.sum() - ws.beta.sum();

        let data = array![1.0, 2.0, 3.0, 4.0];
        let t0 = Some(1usize);
        let acd_data = ACDData::new(data, t0, make_meta_stub()).unwrap();

        let psi_guards = PsiGuards { min: 1e-9, max: 1e9 };
        let model = make_model(p, q, n, Init::UncondMean, psi_guards);

        // Precompute ψ once for the manual likelihood.
        compute_psi(&ws, &acd_data, &model);

        // Manual sum in its own scope so the `psi_buf` borrow is dropped
        // before calling `likelihood_driver`.
        let expected = {
            let psi_buf = model.scratch_bufs.psi_buf.borrow();
            let mut acc = 0.0;
            let t0_idx = acd_data.t0.unwrap();

            for t in t0_idx..n {
                let x_t = acd_data.data[t];
                let psi_t = psi_buf[p + t];
                acc += model
                    .innovation
                    .log_pdf_duration(x_t, psi_t)
                    .expect("innovation log-pdf should succeed for valid inputs");
            }

            acc
        };

        // Act
        let actual = likelihood_driver(&model, &ws, &acd_data).unwrap();

        // Assert
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `compute_derivative` wires the duration lag vector into the
    // α block as documented when there is no β feedback (β = 0).
    //
    // Given
    // -----
    // - ACD(1, 1) with `Init::SampleMean` (θ-independent lags).
    // - `β = 0`, so the recursive feedback term Σ β_j ∂ψ_{t-j}/∂θ vanishes.
    // - Durations τ = [1.0, 2.0, 3.0].
    //
    // Expect
    // ------
    // - For in-sample rows (t >= 1), the ω column equals 1.
    // - For t = 1, the α column equals τ_0.
    // - For t = 2, the α column equals τ_1.
    fn compute_derivative_with_zero_beta_uses_correct_duration_lags_in_alpha_block() {
        // Arrange
        let p = 1usize;
        let q = 1usize;
        let n = 3usize;
        let shape = ACDShape { p, q };

        let mut alpha_buf = array![0.5];
        let mut beta_buf = array![0.0]; // no recursive feedback
        let mut ws = make_workspace(&shape, &mut alpha_buf, &mut beta_buf);
        ws.omega = 1.0;
        ws.slack = 1.0 - STATIONARITY_MARGIN - ws.alpha.sum() - ws.beta.sum();

        let data = array![1.0, 2.0, 3.0];
        let acd_data = ACDData::new(data, None, make_meta_stub()).unwrap();

        let psi_guards = PsiGuards { min: 1e-9, max: 1e9 };
        let model = make_model(p, q, n, Init::SampleMean, psi_guards);

        // Seed ψ and lag buffers.
        compute_psi(&ws, &acd_data, &model);

        // Act
        compute_derivative(&ws, &acd_data, &model);

        // Assert
        let deriv_buf = model.scratch_bufs.deriv_buf.borrow();

        // Rows p..p+n-1 correspond to t = 0..n-1.
        // We examine t = 1 and t = 2 (rows p+1 and p+2).
        let row_t1 = deriv_buf.row(p + 1);
        let row_t2 = deriv_buf.row(p + 2);

        // Columns: [ω, α₁, β₁]
        // ω column should be 1.0.
        assert!((row_t1[0] - 1.0).abs() < 1e-12);
        assert!((row_t2[0] - 1.0).abs() < 1e-12);

        // α column should equal the immediately preceding duration.
        assert!((row_t1[1] - acd_data.data[0]).abs() < 1e-12); // τ_0
        assert!((row_t2[1] - acd_data.data[1]).abs() < 1e-12); // τ_1
    }

    #[test]
    // Purpose
    // -------
    // Verify that with `Init::UncondMean` and constant durations equal to the
    // unconditional mean μ, the derivative recursion satisfies
    //   D_t = c + β D_{t-1}
    // with a time-invariant static part c for all t ≥ 1.
    //
    // Given
    // -----
    // - ACD(1, 1) with parameters satisfying stationarity.
    // - `Init::UncondMean`.
    // - Durations τ_t ≡ μ (the unconditional mean from the workspace).
    //
    // Expect
    // ------
    // - If we form residuals R_t = D_t − β D_{t-1} for t ≥ 1, then R_t is the
    //   same for all t (component-wise), i.e. the time-invariant static part
    //   of the recursion is wired correctly.
    fn compute_derivative_uncond_mean_constant_durations_keeps_derivatives_constant() {
        // Arrange
        let p = 1usize;
        let q = 1usize;
        let n = 4usize;
        let shape = ACDShape { p, q };

        let mut alpha_buf = array![0.2];
        let mut beta_buf = array![0.3];
        let mut ws = make_workspace(&shape, &mut alpha_buf, &mut beta_buf);
        ws.omega = 1.5;
        ws.slack = 1.0 - STATIONARITY_MARGIN - ws.alpha.sum() - ws.beta.sum();

        let beta_val = ws.beta[0];

        let mu = ws.uncond_mean();

        let data = array![mu, mu, mu, mu];
        let acd_data = ACDData::new(data, None, make_meta_stub()).unwrap();

        let psi_guards = PsiGuards { min: 1e-9, max: 1e9 };
        let model = make_model(p, q, n, Init::UncondMean, psi_guards);

        // Seed ψ so derivative recursion sees the correct ψ-lags and ψ_t.
        compute_psi(&ws, &acd_data, &model);

        // Act
        compute_derivative(&ws, &acd_data, &model);

        // Assert
        let deriv_buf = model.scratch_bufs.deriv_buf.borrow();
        let n_cols = deriv_buf.ncols();

        // Indices:
        // row 0   -> D_{-1}
        // row 1   -> D_0
        // row 2   -> D_1
        // row 3   -> D_2
        // row 4   -> D_3
        //
        // Define R_t = D_t − β D_{t-1}.
        // We'll take R_1 as the reference and check R_t == R_1 for t = 2, 3.
        let mut ref_residual = vec![0.0; n_cols];
        for j in 0..n_cols {
            let d1 = deriv_buf[(1 + 1, j)]; // D_1 -> row 2
            let d0 = deriv_buf[(1, j)]; // D_0 -> row 1
            ref_residual[j] = d1 - beta_val * d0;
        }

        for t in 2..n {
            // t = 2,3
            for j in 0..n_cols {
                let d_t = deriv_buf[(t + 1, j)]; // D_t   -> row t+1
                let d_prev = deriv_buf[(t, j)]; // D_{t-1} -> row t
                let residual_t = d_t - beta_val * d_prev;
                assert!(
                    (residual_t - ref_residual[j]).abs() < 1e-10,
                    "residual mismatch at t={}, col={}: got {}, expected {}",
                    t,
                    j,
                    residual_t,
                    ref_residual[j]
                );
            }
        }
    }
}
