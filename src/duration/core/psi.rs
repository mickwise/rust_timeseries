//! ψ–recursion for ACD(p, q).
//!
//! Computes the conditional mean duration series `ψ_t` for an
//! Autoregressive Conditional Duration (ACD) model under the
//! **Engle–Russell / Wikipedia convention**:
//!
//! ψ_t = ω + Σ_{i=1..q} α_i · τ_{t−i} + Σ_{j=1..p} β_j · ψ_{t−j},
//!
//! where `τ_t` are observed durations, **α multiplies duration lags** (length `q`),
//! and **β multiplies ψ lags** (length `p`). Parameter validity (positivity and
//! stationarity) is enforced upstream by `params.rs`.
//!
//! What this module does
//! ---------------------
//! - Seeds pre-sample lag buffers from [`Init`] (`UncondMean`, `SampleMean`,
//!   `Fixed`, or `FixedVector`).
//! - Runs the full ψ–recursion over the input series **in place**, writing results
//!   into the model’s internal buffers (no allocations, no trimming).
//! - Clamps each `ψ_t` to `PsiGuards { min, max }` for numerical safety.
//!
//! Conventions & edge cases
//! ------------------------
//! - `alpha.len() == q` (duration lags), `beta.len() == p` (ψ lags).
//! - Duration buffer convention: index `0` = oldest pre-sample lag,
//!   index `q−1` = most recent pre-sample lag.
use crate::duration::{
    core::{data::ACDData, init::Init, workspace::WorkSpace},
    errors::ACDResult,
    models::acd::ACDModel,
};
use ndarray::s;

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
    let binding = model_spec.psi_buf.borrow();
    let psi_view = binding.slice(s![start_idx..]);
    duration_series
        .iter()
        .zip(psi_view.iter())
        .try_fold(0.0, |acc, (x, psi)| Ok(acc + model_spec.innovation.log_pdf_duration(*x, *psi)?))
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
    let mut binding = model_spec.psi_buf.borrow_mut();
    let mut init_psi_buf = binding.slice_mut(s![..p]);
    let mut init_dur_buf = model_spec.dur_buf.borrow_mut();
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
fn recursion_loop(model_spec: &ACDModel, duration_data: &ACDData, params: &WorkSpace, n: usize) {
    let p = model_spec.shape.p;
    let q = model_spec.shape.q;
    let psi_guards = model_spec.options.psi_guards;
    let omega = params.omega;
    let alpha = &params.alpha; // len = q
    let beta = &params.beta; // len = p

    let mut psi_buf = model_spec.psi_buf.borrow_mut();
    let dur_init = model_spec.dur_buf.borrow();

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
        if new_psi < psi_guards.min {
            new_psi = psi_guards.min;
        } else if new_psi > psi_guards.max {
            new_psi = psi_guards.max;
        }
        psi_buf[p + t] = new_psi;
    }
}
