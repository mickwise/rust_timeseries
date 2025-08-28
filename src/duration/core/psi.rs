//! ψ–recursion for ACD(p, q).
//!
//! Computes the conditional mean duration series `ψ_t` for an
//! Autoregressive Conditional Duration model under the **Engle–Russell / Wikipedia
//! convention**:
//!
//! ψ_t = ω + Σ_{i=1..q} α_i · τ_{t-i} + Σ_{j=1..p} β_j · ψ_{t-j},
//!
//! where `τ_t` are observed durations, **α multiplies duration lags** (length `q`),
//! and **β multiplies ψ lags** (length `p`). Parameter validity (positivity,
//! stationarity) is enforced upstream by `params.rs`.
//!
//! What this module does
//! ---------------------
//! - Seeds the pre-sample lag buffers from `Init` (UncondMean / SampleMean / Fixed / FixedVector).
//! - Runs the full recursion over the input series (no internal trimming).
//! - Clamps each `ψ_t` to `PsiGuards { min, max }` for numerical safety.
//!
//! Conventions & edge cases
//! ------------------------
//! - `alpha.len() == q` (duration lags), `beta.len() == p` (ψ lags).
//! - `p = 0` and/or `q = 0` are allowed; the corresponding dot-products are zero.
use crate::duration::{
    core::{data::ACDData, init::Init, options::ACDOptions, params::ACDParams},
    errors::ACDResult,
};
use ndarray::{Array1, s};

/// Compute the full conditional-mean series `ψ` for an ACD(p, q) model.
///
/// # Definition
/// At each time `t`:
///
/// ψ_t = ω + αᵀ · [τ_{t-1}, …, τ_{t-q}] + βᵀ · [ψ_{t-1}, …, ψ_{t-p}]
///
/// with `ω > 0`, `α_i ≥ 0`, `β_j ≥ 0`. Parameters are assumed valid as enforced
/// by `ACDParams::new`.
///
/// # Behavior
/// - Seeds pre-sample ψ and duration lags from `model_spec.init`.
/// - Appends the observed durations after the duration-lag buffer so the first
///   step uses `[τ_{t-1}, …, τ_{t-q}]`.
/// - Runs the recursion across the entire input length `n = duration_data.data.len()`.
/// - After each update, clamps `ψ_t` to `model_spec.psi_guards`.
///
/// # Inputs
/// - `params`: validated ACD parameters (ω, α with `len = q`, β with `len = p`).
/// - `duration_data`: strictly positive durations (validated in `data.rs`);
/// - `model_spec`: shape `(p, q)`, initialization policy, ψ guards, etc.
///
/// # Returns
/// - `Array1<f64>` of length `n + p`, i.e., one `ψ_t` per observation.
pub fn compute_psi(
    params: &ACDParams, duration_data: &ACDData, model_spec: &ACDOptions,
) -> Array1<f64> {
    let p = model_spec.shape.p;
    let q = model_spec.shape.q;
    let mut total_psi = Array1::<f64>::zeros(duration_data.data.len() + p);
    let mut total_duration = Array1::<f64>::zeros(duration_data.data.len() + q);

    let (initial_psi, initial_duration) = extract_init(
        &model_spec.init,
        params.uncond_mean(),
        duration_data.data.mean().expect("non-empty by ACDData::new"),
        p,
        q,
    );

    total_psi.slice_mut(s![..p]).assign(&initial_psi);
    total_duration.slice_mut(s![..q]).assign(&initial_duration);
    total_duration.slice_mut(s![q..]).assign(&duration_data.data);
    recursion_loop(&mut total_psi, &total_duration, params, model_spec, duration_data.data.len());

    total_psi
}

/// Evaluate the total log-likelihood `ℓ(θ)` for an ACD(p, q) model.
///
/// This driver:
/// 1) computes the conditional-mean series `ψ` via [`compute_psi`], and
/// 2) accumulates per-observation log-likelihood terms using the selected
///    innovation law and the ACD change of variables:
///    `log f_X(x_t | ψ_t) = log f_ε(x_t/ψ_t) - log ψ_t`.
///
/// If `ACDData::t0` is set, the first `t0` observations are excluded from the
/// likelihood sum (e.g., to drop a burn-in); the corresponding `ψ` values are
/// dropped as well so that `(x_t, ψ_t)` remain aligned.
///
/// # Inputs
/// - `params`: validated ACD parameters (ω, α, β) matching `(p, q)`.
/// - `duration_data`: observed, non-negative durations (length `n`), optionally
///   with `t0` indicating a burn-in offset for the likelihood.
/// - `model_spec`: shape `(p, q)`, initialization policy, and ψ-guard bounds.
///   The `innovation` field supplies the unit-mean innovation distribution.
///
/// # Returns
/// The scalar log-likelihood `ℓ(θ)`.
///
/// # Errors
/// - Propagates parameter/data validation errors.
/// - Returns an error if any `ψ_t`/`x_t` pair is invalid for the selected
///   innovation (domain or constructor errors from `statrs`).
///
/// # Notes
/// - `try_fold` is used to accumulate without intermediate allocations.
/// - If ψ guards clamp any `ψ_t`, the likelihood is evaluated at the clamped
///   value (a deliberate numerical safeguard).
pub fn likelihood_driver(
    params: &ACDParams, duration_data: &ACDData, model_spec: &ACDOptions,
) -> ACDResult<f64> {
    let mut start_idx = model_spec.shape.p;
    let duration_series = match duration_data.t0 {
        Some(t0) => {
            start_idx += t0;
            duration_data.data.slice(s![t0..])
        }
        None => duration_data.data.view(),
    };
    let psi_series = compute_psi(params, duration_data, model_spec);
    let psi_view = psi_series.slice(s![start_idx..]);
    duration_series
        .iter()
        .zip(psi_view.iter())
        .try_fold(0.0, |acc, (x, psi)| Ok(acc + model_spec.innovation.log_pdf_duration(*x, *psi)?))
}

// ---- Helper Methods ----

/// Build the pre-sample lag buffers for ψ and durations from the `Init` policy.
///
/// # Returns
/// - `initial_psi`: length **p** (ψ lags: ψ_{-1}..ψ_{-p})
/// - `initial_duration`: length **q** (duration lags: τ_{-1}..τ_{-q})
///
/// # Policies
/// - `UncondMean`: fill both buffers with the unconditional mean μ.
/// - `SampleMean`: fill both buffers with the sample mean of `duration_data`.
/// - `Fixed(value)`: fill both buffers with the given scalar `value > 0`.
/// - `FixedVector { psi_lags, duration_lags }`: use provided vectors (must be
///   exact lengths `p` and `q`).
fn extract_init(
    init_specs: &Init, uncond_mean: f64, sample_mean: f64, p: usize, q: usize,
) -> (Array1<f64>, Array1<f64>) {
    match init_specs {
        Init::UncondMean => (Array1::from_elem(p, uncond_mean), Array1::from_elem(q, uncond_mean)),
        Init::SampleMean => (Array1::from_elem(p, sample_mean), Array1::from_elem(q, sample_mean)),
        Init::Fixed(val) => (Array1::from_elem(p, *val), Array1::from_elem(q, *val)),
        Init::FixedVector { psi_lags, duration_lags } => (psi_lags.clone(), duration_lags.clone()),
    }
}

/// Core ACD(p, q) recursion over the full series (internal helper).
///
/// Assumptions:
/// - `total_psi` has capacity for the seed region followed by `n` outputs.
/// - `total_duration` has the duration-lag seed followed by observed durations
///   (so windows `[t-1..t-q]` are valid).
///
/// For each `t = 0..n-1`:
/// 1) Read duration lags (length `q`) and ψ lags (length `p`).
/// 2) Compute `ψ_t = ω + α·duration_lags + β·psi_lags`.
/// 3) Clamp `ψ_t` to configured guard bounds.
/// 4) Write `ψ_t` to the output region in `total_psi`.
fn recursion_loop(
    total_psi: &mut Array1<f64>, total_duration: &Array1<f64>, params: &ACDParams,
    model_spec: &ACDOptions, n: usize,
) {
    let p = model_spec.shape.p;
    let q = model_spec.shape.q;
    let psi_guards = model_spec.psi_guards;
    let omega = params.omega;
    let alpha = &params.alpha;
    let beta = &params.beta;

    for i in 0..n {
        let psi_lags = total_psi.slice(s![i..i + p]);
        let duration_lags = total_duration.slice(s![i..i + q]);

        let mut new_psi = omega + alpha.dot(&duration_lags) + beta.dot(&psi_lags);
        if new_psi < psi_guards.min {
            new_psi = psi_guards.min;
        } else if new_psi > psi_guards.max {
            new_psi = psi_guards.max;
        }

        total_psi[p + i] = new_psi;
    }
}
