//! ψ-recursion for ACD(p, q).
//!
//! This module computes the conditional expected duration series `ψ_t` for an
//! Autoregressive Conditional Duration model ACD(p, q) under the Engle–Russell
//! convention:
//!
//!     ψ_t = ω + Σ_{i=1..q} α_i · τ_{t-i} + Σ_{j=1..p} β_j · ψ_{t-j},
//!
//! where τ_t are observed durations, α’s multiply **duration lags** (length q),
//! and β’s multiply **ψ lags** (length p). Parameters are validated upstream
//! (`params.rs`) to guarantee positivity and stationarity (Σα + Σβ < 1 - margin).
//!
//! This module:
//! - seeds the pre-sample lag buffers from `Init` (UncondMean / SampleMean / Fixed / FixedVector),
//! - runs the full recursion over the input series (no internal trimming),
//! - clamps each ψ_t to configured guard bounds for numerical safety.

use crate::duration::{
    acd::{ACDData, ACDOptions, Init},
    params::ACDParams,
};
use ndarray::{Array1, s};

/// Compute the full conditional mean duration series ψ for an ACD(p, q) model.
///
/// # Definition
/// Using Engle–Russell convention, at each time `t`:
///
///     ψ_t = ω + αᵀ · [τ_{t-1}, …, τ_{t-q}] + βᵀ · [ψ_{t-1}, …, ψ_{t-p}]
///
/// with ω > 0, α_i ≥ 0, β_j ≥ 0. Parameters are assumed valid (shape, positivity,
/// stationarity) as enforced by `params.rs`.
///
/// # Behavior
/// - Seeds the pre-sample lag buffers from `model_spec.init`.
/// - Preloads observed durations after the duration-lag buffer so the first
///   step uses `[τ_{t-1}, …, τ_{t-q}]`.
/// - Runs the recursion across the entire input length `n = data.data.len()`.
/// - After each update, clamps ψ_t to `model_spec.psi_guards` to prevent
///   `log(0)`, overflow, or NaNs in downstream code.
///
/// # Inputs
/// - `params`: validated ACD parameters (ω, α (len=q), β (len=p)).
/// - `duration_data`: observed strictly positive durations (validated in `acd.rs`);
///   may include an optional `t0`, which is *not* applied here.
/// - `model_spec`: options including `(p, q)`, initialization policy, and ψ guards.
///
/// # Returns
/// A full-length vector ψ of size `n`, aligned with `duration_data.data`.
pub fn compute_psi(
    params: &ACDParams,
    duration_data: &ACDData,
    model_spec: &ACDOptions,
) -> Array1<f64> {
    let p = model_spec.shape.p;
    let q = model_spec.shape.q;
    let mut total_psi = Array1::<f64>::zeros(duration_data.data.len() + p);
    let mut total_duration = Array1::<f64>::zeros(duration_data.data.len() + q);

    let (initial_psi, initial_duration) = extract_init(
        &model_spec.init,
        params.uncond_mean(),
        duration_data.data.mean().unwrap(),
        p,
        q,
    );

    total_psi.slice_mut(s![..p]).assign(&initial_psi);
    total_duration.slice_mut(s![..q]).assign(&initial_duration);
    total_duration
        .slice_mut(s![q..])
        .assign(&duration_data.data);
    recursion_loop(
        &mut total_psi,
        &total_duration,
        params,
        model_spec,
        duration_data.data.len(),
    );

    total_psi
}

/// ---- Helper Methods ----

/// Build the pre-sample lag buffers for ψ and durations from the `Init` policy.
///
/// # Output
/// Returns a pair:
/// - `initial_psi`: length **p** (ψ lags: ψ_{-1}..ψ_{-p})
/// - `initial_duration`: length **q** (duration lags: τ_{-1}..τ_{-q})
///
/// # Policies
/// - `UncondMean`: fills both buffers with the unconditional mean μ
///   (consistent with stationarity).
/// - `SampleMean`: fills both with the sample mean of `duration_data`.
/// - `Fixed(value)`: fills both with the given scalar.
/// - `FixedVector { psi_lags, duration_lags }`: uses the provided vectors; their
///   lengths must equal `p` and `q` respectively.
fn extract_init(
    init_specs: &Init,
    uncond_mean: f64,
    sample_mean: f64,
    p: usize,
    q: usize,
) -> (Array1<f64>, Array1<f64>) {
    match init_specs {
        Init::UncondMean => {
            return (
                Array1::from_elem(p, uncond_mean),
                Array1::from_elem(q, uncond_mean),
            );
        }
        Init::SampleMean => {
            return (
                Array1::from_elem(p, sample_mean),
                Array1::from_elem(q, sample_mean),
            );
        }
        Init::Fixed(val) => return (Array1::from_elem(p, *val), Array1::from_elem(q, *val)),
        Init::FixedVector {
            psi_lags,
            duration_lags,
        } => return (psi_lags.clone(), duration_lags.clone()),
    };
}

/// Core ACD(p, q) recursion over the full series (internal helper).
///
/// Assumes:
/// - `total_psi` has capacity for the seed region followed by `n` outputs.
/// - `total_duration` contains the duration-lag seed region followed by the
///   observed durations in order (so windowing `[t-1..t-q]` is valid).
///
/// For each t = 0..n-1:
/// 1) Reads duration lags (length q) and ψ lags (length p).
/// 2) Computes `ψ_t = ω + α·duration_lags + β·psi_lags`.
/// 3) Clamps ψ_t to configured guard bounds.
/// 4) Writes ψ_t to the output region in `total_psi`.
fn recursion_loop(
    total_psi: &mut Array1<f64>,
    total_duration: &Array1<f64>,
    params: &ACDParams,
    model_spec: &ACDOptions,
    n: usize,
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
