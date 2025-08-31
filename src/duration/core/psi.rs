//! ψ–recursions for ACD(p, q): training and derivatives.
//!
//! Implements the in-sample conditional-mean recursion and the allocation-free
//! derivative (sensitivity) recursion used for analytic gradients.
//!
//! ## Model convention (Engle–Russell/Wikipedia)
//! `ψ_t = ω + Σ_{i=1..q} α_i τ_{t−i} + Σ_{j=1..p} β_j ψ_{t−j}`
//!
//! ## What this module does
//! - Seeds pre-sample lag buffers from [`Init`] (`UncondMean`, `SampleMean`,
//!   `Fixed`, or `FixedVector`).
//! - Runs the ψ–recursion over the sample **in place**, writing into the model’s
//!   preallocated buffers (no heap allocations).
//! - Fills the ∂ψ_t/∂θ matrix allocation-free for use in analytic gradients.
//! - Clamps each `ψ_t` to `PsiGuards { min, max }` for numerical safety.
//!
//! ## Ordering assumptions
//! - Duration and ψ lag buffers store the **newest element at the end**.
//!   Lag windows used inside the recursions are taken as **reversed tails**
//!   (newest → oldest) to align with `[·_{t−1}, …, ·_{t−k}]`.
//!
//! ## Invariants (enforced upstream)
//! - `ω > 0`; `α, β ≥ 0` elementwise; `sum(α)+sum(β) < 1 − margin`.
//!
//! ## Zero-copy design
//! Inner loops operate on `ndarray` views only; buffers live in `ACDScratch`.
use crate::{
    duration::{
        core::{data::ACDData, guards::PsiGuards, init::Init, workspace::WorkSpace},
        errors::ACDResult,
        models::acd::ACDModel,
    },
    optimization::numerical_stability::transformations::STATIONARITY_MARGIN,
};
use ndarray::{Axis, s};

/// Compute the conditional-mean series ψ **in place** for an ACD(p, q) model.
///
/// This function seeds the pre-sample lag buffers according to [`Init`] and
/// then runs the ψ–recursion, writing outputs directly into
/// `model_spec.psi_buf[p .. p + n]`, where `n = duration_data.data.len()`.
///
/// # Definition
/// At each time `t` (0-based):
///
/// ψ_t = ω + αᵀ · [τ_{t-1}, …, τ_{t-q}] + βᵀ · [ψ_{t-1}, …, ψ_{t-p}]
///
/// with `ω > 0`, `α_i ≥ 0`, `β_j ≥ 0`. Parameter validity (positivity and
/// stationarity) is enforced upstream.
///
/// # Behavior
/// - Seeds pre-sample ψ and duration lags from `model_spec.options.init`.
/// - Runs the recursion across all `n` observations in `duration_data`.
/// - Clamps each `ψ_t` to `model_spec.options.psi_guards`.
///
/// # Side effects
/// - Writes into `model_spec.psi_buf[p .. p + n]`. No heap allocations.
///
/// # Inputs
/// - `params`: validated workspace with `(ω, α (len=q), β (len=p))`.
/// - `duration_data`: strictly positive durations (validated by `ACDData::new`).
/// - `model_spec`: shape `(p, q)`, innovation, options, and the internal buffers.
pub fn compute_psi(params: &WorkSpace, duration_data: &ACDData, model_spec: &ACDModel) -> () {
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

/// Evaluate the total log-likelihood `ℓ(θ)` for an ACD(p, q) model.
///
/// This driver:
/// 1) Calls [`compute_psi`] to (re)compute and store `ψ` in `model_spec.psi_buf`.
/// 2) Accumulates per-observation terms using the selected innovation law and
///    the ACD change of variables:
///       `log f_X(x_t | ψ_t) = log f_ε(x_t/ψ_t) - log ψ_t`.
///
/// If `ACDData::t0` is set, the first `t0` observations are excluded from the
/// likelihood sum; the matching `ψ` values are skipped so `(x_t, ψ_t)` remain
/// aligned.
///
/// # Inputs
/// - `model_spec`: provides shape `(p, q)`, innovation, options, and ψ buffer.
/// - `workspace`: validated parameters `(ω, α, β)` (via zero-copy views).
/// - `duration_data`: observed durations and optional `t0`.
///
/// # Returns
/// - The scalar log-likelihood `ℓ(θ)`.
///
/// # Errors
/// - Propagates data/parameter validation and innovation PDF errors.
///
/// # Notes
/// - Uses a borrow of the internal `psi_buf` (no allocations).
/// - If any ψ is clamped by guards, the likelihood is evaluated at the clamped value.
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

/// Top-level driver for ACD(p, q) ψ–derivative recursion.
///
/// Computes ∂ψ_t/∂θ for all `t = 0..n-1` (with θ = (ω, α₁..α_q, β₁..β_p)),
/// filling the `model_spec.deriv_buf` matrix of shape `(n+p, 1+q+p)`.
///
/// Procedure:
/// 1. `extract_init_derivative` seeds the first `p` rows with the
///    derivatives of the initial ψ–lags (ψ_{-1}, …, ψ_{-p}) according
///    to the chosen initialization scheme (`Init::UncondMean`, `SampleMean`,
///    or `Fixed*`).
/// 2. `recursion_loop_derivative` propagates these through the sample
///    durations, updating rows `p..p+n-1` by applying the ACD recursion
///    with both static contributions (ω, α·lags, β·lags) and recursive
///    sensitivity (β feedback).
///
/// # Side effects
/// - Overwrites the entire `model_spec.deriv_buf` with derivative rows.
/// - Reads from `duration_data.data`, `model_spec.scratch_bufs.dur_buf`
///   and `psi_buf`. No heap allocations.
pub fn compute_derivative(
    params: &WorkSpace, duration_data: &ACDData, model_spec: &ACDModel,
) -> () {
    let p = model_spec.shape.p;
    extract_init_derivative(model_spec, &model_spec.options.init, params, p);
    recursion_loop_derivative(model_spec, duration_data, params, duration_data.data.len());
}

/// Clamp a ψ value into `[guards.min, guards.max]`.
///
/// Used both in training and forecasting to prevent numerical underflow/overflow
/// from propagating through the recursion. Returns:
/// - `guards.min` if `value < guards.min`
/// - `guards.max` if `value > guards.max`
/// - `value` otherwise
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

/// Seed the pre-sample lag buffers for ψ and durations according to [`Init`].
///
/// This writes into:
/// - `model_spec.psi_buf[..p]`  — ψ lag seed (ψ_{-1}, …, ψ_{-p})
/// - `model_spec.dur_buf[..q]`  — duration lag seed (τ_{-q}, …, τ_{-1})
///
/// Policies:
/// - `UncondMean`   → fill both with μ (the unconditional mean).
/// - `SampleMean`   → fill both with the sample mean of the duration series.
/// - `Fixed(value)` → fill both with the given positive scalar.
/// - `FixedVector { psi_lags, duration_lags }` → copy the provided vectors
///    (must match lengths `p` and `q`).
///
/// # Side effects
/// - In-place writes; no heap allocations.
///
/// # Inputs
/// - `model_spec`: provides the internal ψ and duration buffers.
/// - `init_specs`: initialization policy.
/// - `uncond_mean`: μ implied by current `(ω, α, β)`.
/// - `sample_mean`: mean of the observed durations (precomputed upstream).
/// - `p`: number of ψ lags; `q` is inferred from `model_spec`.
fn extract_init(
    model_spec: &ACDModel, init_specs: &Init, uncond_mean: f64, sample_mean: f64, p: usize,
) -> () {
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

/// Seed ACD(p, q) ψ–derivative buffer with initial lag rows.
///
/// Fills the first `p` rows of `model_spec.deriv_buf` with the derivatives
/// of the initial ψ–lags (ψ_{-1}, …, ψ_{-p}) w.r.t. θ = (ω, α₁..α_q, β₁..β_p).
///
/// Behavior by init spec:
/// - `Init::UncondMean`: each initial ψ–lag equals μ = ω / (1 − Σα − Σβ).
///   The derivative rows are filled with:
///     - ∂μ/∂ω   = 1 / (1 − Σα − Σβ)
///     - ∂μ/∂αᵢ  = ∂μ/∂βⱼ = ω / (1 − Σα − Σβ)²
///   All `p` rows are identical copies of this vector.
/// - `Init::SampleMean` or any `Init::Fixed*`: initial ψ–lags do not depend
///   on θ, so the corresponding derivative rows are zeroed.
///
/// # Side effects
/// - Writes into `model_spec.deriv_buf[..p, ..]`. No heap allocations.
/// - Overwrites any existing contents of these rows.
fn extract_init_derivative(
    model_spec: &ACDModel, init_specs: &Init, params: &WorkSpace, p: usize,
) -> () {
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

/// Core ACD(p, q) ψ–recursion (allocation-free).
///
/// Updates `model_spec.psi_buf[p + t]` for `t = 0..n-1` using:
///
///   ψ_t = ω + αᵀ · [τ_{t-1}, …, τ_{t-q}] + βᵀ · [ψ_{t-1}, …, ψ_{t-p}]
///
/// Duration lags are built per step with **one conditional split**:
/// - `k_init = max(0, q − t)` — lags still coming from the init buffer
/// - `k_data = q − k_init`    — lags coming from observed data
///
/// Convention:
/// - `dur_buf[0] = τ_{-q}` (oldest) … `dur_buf[q-1] = τ_{-1}` (newest).
/// - For the dot with α = `[α₁,…,α_q]`, both segments are read **reversed**
///   (newest → oldest) so that they align to `[τ_{t-1}, …, τ_{t-q}]`.
///
/// # Side effects
/// - Writes `ψ_t` into `model_spec.psi_buf[p + t]`. No heap allocations.
///
/// # Guards
/// - After accumulation, ψ_t is clamped into `model_spec.options.psi_guards`.
fn recursion_loop(
    model_spec: &ACDModel, duration_data: &ACDData, params: &WorkSpace, n: usize,
) -> () {
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

/// Core ACD(p, q) ψ–derivative recursion (allocation-free).
///
/// Updates `model_spec.deriv_buf[p + t, ..]` for `t = 0..n-1` with the
/// gradient row ∂ψ_t/∂θ, where θ = (ω, α₁..α_q, β₁..β_p).
///
/// Recursion:
///   ∂ψ_t/∂θ = e_ω
///           + [τ_{t-1}, …, τ_{t-q}] placed in α–columns
///           + [ψ_{t-1}, …, ψ_{t-p}] placed in β–columns
///           + Σ_{j=1}^p β_j · ∂ψ_{t-j}/∂θ
///
/// Duration lags are built per step with **one conditional split**:
/// - `k_init = max(0, q − t)` — lags still coming from the init buffer
/// - `k_data = q − k_init`    — lags coming from observed data
///
/// Convention:
/// - `dur_buf[0] = τ_{-q}` (oldest) … `dur_buf[q-1] = τ_{-1}` (newest).
/// - For the α block, both init and data tails are read **reversed**
///   (newest → oldest) to align with `[τ_{t-1}, …, τ_{t-q}]`.
/// - For the β block, the ψ–lags window is `psi_buf[t .. t+p]` which
///   corresponds to `[ψ_{t-1}, …, ψ_{t-p}]`.
///
/// # Side effects
/// - Writes the derivative vector into `model_spec.deriv_buf[p + t, ..]`.
/// - Reads ψ–lags from `psi_buf` and previous derivative rows from
///   `deriv_buf[..p+t, ..]`. No heap allocations.
///
/// # Guards
/// - Each row is zeroed before being filled to prevent stale values.
fn recursion_loop_derivative(
    model_spec: &ACDModel, duration_data: &ACDData, params: &WorkSpace, n: usize,
) -> () {
    let p = model_spec.shape.p;
    let q = model_spec.shape.q;
    let dur_init = model_spec.scratch_bufs.dur_buf.borrow();
    let psi_buf = model_spec.scratch_bufs.psi_buf.borrow_mut();
    let beta = &params.beta;
    for t in 0..n {
        let mut binding = model_spec.scratch_bufs.deriv_buf.borrow_mut();
        let deriv_buf = binding.view_mut();
        let k_init = q.saturating_sub(t);
        let k_data = q - k_init;

        let (deriv_lags_tail, mut deriv_lags_head) = deriv_buf.split_at(Axis(0), p + t);

        let init_tail_rev = dur_init.slice(s![q - k_init .. q; -1]);
        let data_tail_rev = duration_data.data.slice(s![t - k_data .. t; -1]);

        let mut curr_row = deriv_lags_head.row_mut(0);
        curr_row.fill(0.0);
        curr_row[0] = 1.0;
        curr_row.slice_mut(s![1..k_init + 1]).assign(&init_tail_rev);
        curr_row.slice_mut(s![k_init + 1..q + 1]).assign(&data_tail_rev);
        curr_row.slice_mut(s![q + 1..]).assign(&psi_buf.slice(s![t..t + p]));
        for j in 1..=p {
            let curr_beta = beta[j - 1];
            curr_row.scaled_add(curr_beta, &deriv_lags_tail.row(p + t - j));
        }
    }
}
