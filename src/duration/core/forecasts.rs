//! Out-of-sample ψ-forecasting for ACD(p, q).
//!
//! Rolls the ACD ψ-recursion forward beyond the sample, writing ψ̂ forecasts
//! into a preallocated buffer with no heap allocations.
//!
//! ## Model convention
//! `ψ_t = ω + Σ_{i=1..q} α_i τ_{t−i} + Σ_{j=1..p} β_j ψ_{t−j}`
//!
//! ## Forecast logic
//! - For early steps, duration lags use the **last q observed durations**;
//!   once those are exhausted, they use your **own duration forecasts**, with
//!   `τ̂ = ψ̂` under unit-mean innovations.
//! - ψ-lags use the **last p in-sample ψ’s** for the earliest positions and
//!   the **already-computed ψ̂** values thereafter.
//! - Each ψ̂ is clamped with `PsiGuards` to maintain numerical stability.
//!
//! ## Ordering assumptions
//! - Both the provided **duration_lags** (length q) and the cached **ψ-lags**
//!   in `ACDParams` (length p) store the **newest element at the end**.
//!
//! ## Zero-copy design
//! The forecasting loop uses only `ndarray` views; all storage is provided by
//! the caller via [`ACDForecastResult`].
use ndarray::{Array1, ArrayView1, s};
use std::{cell::RefCell, cmp::min};

use crate::duration::{
    core::{guards::PsiGuards, params::ACDParams, psi::guard_psi},
    errors::ACDResult,
};

/// Container for ψ-forecast outputs.
///
/// Holds a mutable, preallocated buffer into which forecasting routines write
/// ψ̂ values sequentially. This separates **storage** (owned by the model) from
/// the **recursion** (which uses only views), keeping inner loops allocation-free.
///
/// Conventions:
/// - `psi_forecast[i]` stores the i-step-ahead ψ-forecast, i.e., `ψ̂_{T+i+1}`
///   when `i` is zero-based.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDForecastResult {
    pub psi_forecast: RefCell<Array1<f64>>,
}

impl ACDForecastResult {
    /// Create a forecast buffer with `len` zeros.
    ///
    /// he buffer is owned and reused by the model to avoid repeated allocations.
    pub fn new(horizon: usize) -> Self {
        Self { psi_forecast: RefCell::new(Array1::zeros(horizon)) }
    }
}

/// Roll the ACD(p, q) recursion forward for `horizon` steps (allocation-free),
/// writing ψ-forecasts into `forecast_result` and returning the last value.
///
/// ## Inputs
/// - `params`: fitted **model-space** parameters with
///   `omega`, `alpha (len=q)`, `beta (len=p)`, and cached **last p ψ-lags**
///   in `psi_lags` (newest at the **end**).
/// - `duration_lags`: the **last q observed durations**, ordered **oldest → newest**
///   (i.e., newest at the **end**).
/// - `horizon`: number of steps to forecast (H ≥ 1).
/// - `forecast_result`: mutable holder containing a preallocated `psi_forecast`
///   buffer. This function **writes** `horizon` ψ-forecasts into indices
///   `0..horizon`.
/// - `guards`: `PsiGuards {min, max}`; each ψ̂ is clamped after computation.
///
/// ## Behavior
/// For each step `i = 0..horizon-1`, build the lag vectors exactly as in the
/// training recursion:
/// - **α (duration) term**: split the q-vector into an **in-sample tail**
///   (last `k_init = max(0, q − i)` durations from `duration_lags`) and a
///   **forecast tail** (last `k_data = q − k_init` duration forecasts, where
///   `τ̂ = ψ̂`), both read **newest → oldest**.
/// - **β (ψ) term**: split the p-vector into an **in-sample tail**
///   (last `psi_init = min(i, p)` ψ-lags from `params.psi_lags`) and a
///   **forecast tail** (last `p − psi_init` ψ̂’s already written), both read
///   **newest → oldest**.
/// - Compute `ψ̂ = ω + α·τ_lags + β·ψ_lags`, clamp with `guards`, and store at
///   `psi_forecast[i]`.
///
/// Under unit-mean innovations, the duration forecast equals the ψ-forecast:
/// `τ̂ = ψ̂`.
///
/// ## Preconditions
/// - `duration_lags.len() == q` (newest at the **end**).
/// - `params.psi_lags.len() == p` (newest at the **end**).
/// - `forecast_result.psi_forecast` has capacity for at least `horizon` values.
/// - Model is strictly stationary (`sum(α) + sum(β) < 1 − margin`).
///
/// ## Returns
/// - `Ok(ψ̂_{T+horizon})` — the last ψ-forecast — after writing all intermediate
///   steps into `forecast_result`.
pub fn forecast_recursion(
    params: &ACDParams, duration_lags: ArrayView1<f64>, horizon: usize,
    forecast_result: &ACDForecastResult, guards: &PsiGuards,
) -> ACDResult<f64> {
    let alpha = &params.alpha;
    let beta = &params.beta;
    let q = alpha.len();
    let p = beta.len();
    let psi_lags = &params.psi_lags;
    let mut forecast_psi = forecast_result.psi_forecast.borrow_mut();
    for i in 0..horizon {
        let k_init = q.saturating_sub(i);
        let k_data = q - k_init;
        let psi_init_idx = min(i, p);
        let psi_idx = p - psi_init_idx;
        let init_tail_rev = duration_lags.slice(s![q - k_init..q; -1]);
        let data_tail_rev = forecast_psi.slice(s![i - k_data..i; -1]);
        let init_psi_lags_rev = psi_lags.slice(s![p - psi_init_idx..p; -1]);
        let psi_lags_rev = forecast_psi.slice(s![i - psi_idx..i; -1]);
        let sum_alpha = alpha.slice(s![..k_init]).dot(&init_tail_rev)
            + alpha.slice(s![k_init..]).dot(&data_tail_rev);
        let sum_beta = beta.slice(s![..psi_init_idx]).dot(&init_psi_lags_rev)
            + beta.slice(s![psi_init_idx..]).dot(&psi_lags_rev);
        let curr_psi = params.omega + sum_alpha + sum_beta;
        forecast_psi[i] = guard_psi(curr_psi, guards);
    }
    Ok(forecast_psi[horizon - 1])
}
