//! ACD validation helpers — reusable checks for parameters, lags, and stationarity.
//!
//! Purpose
//! -------
//! Centralize small, reusable validation routines used across the ACD duration
//! stack. These helpers enforce basic sanity checks for distribution parameters,
//! pre-sample lags, model-space parameters (ω, α, β, slack), and unconstrained
//! optimizer inputs, so higher-level constructors and models can fail fast with
//! structured errors.
//!
//! Key behaviors
//! -------------
//! - Validate innovation-law parameters for Weibull and generalized Gamma
//!   distributions (positivity, finiteness).
//! - Validate pre-sample duration/ψ lags and ACD parameter vectors (ω, α, β,
//!   slack) against shape, positivity, and stationarity constraints.
//! - Validate unconstrained optimizer inputs θ before mapping into model space.
//!
//! Invariants & assumptions
//! ------------------------
//! - Duration and ψ lags must be finite and strictly positive when supplied
//!   as pre-sample buffers.
//! - α and β coordinates must be finite and non-negative; ω must be finite
//!   and strictly positive.
//! - Stationarity is enforced via a margin [`STATIONARITY_MARGIN`] such that
//!   `sum(α) + sum(β) + slack ≈ 1 - STATIONARITY_MARGIN` within a small
//!   tolerance. The “slack” coordinate is treated as non-negative and finite.
//! - The Engle–Russell convention is assumed throughout:
//!   - `q` = number of α terms (lags on durations τ),
//!   - `p` = number of β terms (lags on conditional means ψ).
//!
//! Conventions
//! -----------
//! - Indices are 0-based and follow the usual Rust/ndarray conventions.
//! - Validation functions return [`ACDResult`] or [`ParamResult`] and never
//!   panic on invalid *inputs*; panics are reserved for programming errors
//!   elsewhere (e.g., shape mismatches in other modules).
//! - This module contains no I/O and no logging; it only inspects numeric
//!   values and array lengths.
//!
//! Downstream usage
//! ----------------
//! - Call these helpers from constructors (`ACDData`, `ACDParams`, `ACDModel`,
//!   etc.) to enforce documented invariants at the boundaries of the API.
//! - Use the lag validation functions when building `Init::FixedVector`
//!   pre-sample buffers.
//! - Use the stationarity and θ validation functions in the optimizer
//!   mapping and parameter constructors to ensure consistency with the
//!   stationarity margin and expected θ layout.
//!
//! Testing notes
//! -------------
//! - Unit tests exercise each helper on representative valid and invalid
//!   inputs, including boundary cases (zeros, infinities, NaNs, length off-by-1,
//!   and sums just inside/outside the stationarity margin).
//! - Integration tests and Python-level tests rely on the higher-level
//!   constructors that *call* these helpers rather than re-testing the raw
//!   validation logic.
use crate::{
    duration::errors::{ACDError, ACDResult, ParamError, ParamResult},
    optimization::numerical_stability::transformations::STATIONARITY_MARGIN,
};
use ndarray::{Array1, ArrayView1};

/// Validate a single Weibull parameter (scale or shape).
///
/// Parameters
/// ----------
/// - `param`: `f64`
///   Candidate Weibull parameter (scale or shape). Must be finite and strictly
///   positive.
///
/// Returns
/// -------
/// `ACDResult<f64>`
///   - `Ok(param)` if `param` is finite and strictly > 0.
///   - `Err(ACDError::InvalidWeibullParam)` with the offending value and a
///     descriptive reason otherwise.
///
/// Errors
/// ------
/// - `ACDError::InvalidWeibullParam`
///   - Returned if `param` is NaN, ±∞, or ≤ 0. The `reason` field explains
///     whether finiteness or positivity failed.
///
/// Panics
/// ------
/// - Never panics.
///
/// Notes
/// -----
/// - This helper does not distinguish between scale and shape; it is intended
///   for any Weibull parameter that must be finite and strictly positive.
/// - Used in innovation constructors before instantiating the underlying
///   `statrs` distribution.
///
/// Examples
/// --------
/// ```rust
/// # use rust_timeseries::duration::core::validation::validate_weibull_param;
/// use rust_timeseries::duration::errors::ACDError;
///
/// assert!(validate_weibull_param(1.0).is_ok());
/// assert!(matches!(
///     validate_weibull_param(0.0),
///     Err(ACDError::InvalidWeibullParam { .. })
/// ));
/// ```
pub fn validate_weibull_param(param: f64) -> ACDResult<f64> {
    if !param.is_finite() {
        return Err(ACDError::InvalidWeibullParam {
            param,
            reason: "Weibull parameters must be finite.",
        });
    }
    if param <= 0.0 {
        return Err(ACDError::InvalidWeibullParam {
            param,
            reason: "Weibull parameters must be strictly positive.",
        });
    }
    Ok(param)
}

/// Validate a single generalized Gamma parameter (scale or shape).
///
/// Parameters
/// ----------
/// - `param`: `f64`
///   Candidate generalized-Gamma parameter (scale or shape). Must be finite
///   and strictly positive.
///
/// Returns
/// -------
/// `ACDResult<f64>`
///   - `Ok(param)` if `param` is finite and strictly > 0.
///   - `Err(ACDError::InvalidGenGammaParam)` with the offending value and a
///     descriptive reason otherwise.
///
/// Errors
/// ------
/// - `ACDError::InvalidGenGammaParam`
///   - Returned if `param` is NaN, ±∞, or ≤ 0. The `reason` field explains
///     whether finiteness or positivity failed.
///
/// Panics
/// ------
/// - Never panics.
///
/// Notes
/// -----
/// - As with [`validate_weibull_param`], this treats all generalized-Gamma
///   parameters uniformly.
///
/// Examples
/// --------
/// ```rust
/// # use rust_timeseries::duration::core::validation::validate_gamma_param;
/// use rust_timeseries::duration::errors::ACDError;
///
/// assert!(validate_gamma_param(2.5).is_ok());
/// assert!(matches!(
///     validate_gamma_param(f64::NAN),
///     Err(ACDError::InvalidGenGammaParam { .. })
/// ));
/// ```
pub fn validate_gamma_param(param: f64) -> ACDResult<f64> {
    if !param.is_finite() {
        return Err(ACDError::InvalidGenGammaParam {
            param,
            reason: "Generalized Gamma parameters must be finite.",
        });
    }
    if param <= 0.0 {
        return Err(ACDError::InvalidGenGammaParam {
            param,
            reason: "Generalized Gamma parameters must be strictly positive.",
        });
    }
    Ok(param)
}

/// Validate pre-sample **duration** lags (length q).
///
/// Parameters
/// ----------
/// - `duration_lags`: `&Array1<f64>`
///   Pre-sample duration lags `(τ_{-q}, …, τ_{-1})`. Every element must be
///   finite and strictly positive.
/// - `q`: `usize`
///   Expected number of duration lags (α order).
///
/// Returns
/// -------
/// `ACDResult<()>`
///   - `Ok(())` if `duration_lags.len() == q` and all entries are finite and
///     strictly > 0.
///   - `Err(ACDError)` describing the first violation encountered.
///
/// Errors
/// ------
/// - `ACDError::InvalidDurationLength`
///   - Returned if `duration_lags.len() != q`.
/// - `ACDError::InvalidDurationLags`
///   - Returned if any element is NaN, ±∞, or ≤ 0, with the offending index
///     and value.
///
/// Panics
/// ------
/// - Never panics.
///
/// Notes
/// -----
/// - Intended primarily for validating `Init::FixedVector { duration_lags, .. }`
///   before wiring the lags into the recursion.
/// - This helper does not mutate or normalize the lags; it only checks them.
///
/// Examples
/// --------
/// ```rust
/// # use rust_timeseries::duration::core::validation::validate_duration_lags;
/// # use rust_timeseries::duration::errors::ACDError;
/// use ndarray::array;
///
/// let lags = array![1.0, 2.0, 3.0];
/// assert!(validate_duration_lags(&lags, 3).is_ok());
///
/// let bad_lags = array![1.0, 0.0, 3.0];
/// assert!(matches!(
///     validate_duration_lags(&bad_lags, 3),
///     Err(ACDError::InvalidDurationLags { .. })
/// ));
/// ```
pub fn validate_duration_lags(duration_lags: &Array1<f64>, q: usize) -> ACDResult<()> {
    if duration_lags.len() != q {
        return Err(ACDError::InvalidDurationLength { expected: q, actual: duration_lags.len() });
    }
    for (index, &value) in duration_lags.iter().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(ACDError::InvalidDurationLags { index, value });
        }
    }
    Ok(())
}

/// Validate pre-sample **ψ** lags (length p).
///
/// Parameters
/// ----------
/// - `psi_lags`: `&Array1<f64>`
///   Pre-sample ψ lags `(ψ_{-1}, …, ψ_{-p})`. Every element must be finite
///   and strictly positive.
/// - `p`: `usize`
///   Expected number of ψ lags (β order).
///
/// Returns
/// -------
/// `ACDResult<()>`
///   - `Ok(())` if `psi_lags.len() == p` and all entries are finite and
///     strictly > 0.
///   - `Err(ACDError)` describing the first violation encountered.
///
/// Errors
/// ------
/// - `ACDError::InvalidPsiLength`
///   - Returned if `psi_lags.len() != p`.
/// - `ACDError::InvalidPsiLags`
///   - Returned if any element is NaN, ±∞, or ≤ 0, with the offending index
///     and value.
///
/// Panics
/// ------
/// - Never panics.
///
/// Notes
/// -----
/// - Intended for validating `Init::FixedVector { psi_lags, .. }` and any
///   other pre-sample ψ buffers.
/// - This helper does not modify ψ; it only validates shape and positivity.
///
/// Examples
/// --------
/// ```rust
/// # use rust_timeseries::duration::core::validation::validate_psi_lags;
/// # use rust_timeseries::duration::errors::ACDError;
/// use ndarray::array;
///
/// let lags = array![0.5, 0.6];
/// assert!(validate_psi_lags(&lags, 2).is_ok());
///
/// let bad_lags = array![0.5, f64::INFINITY];
/// assert!(matches!(
///     validate_psi_lags(&bad_lags, 2),
///     Err(ACDError::InvalidPsiLags { .. })
/// ));
/// ```
pub fn validate_psi_lags(psi_lags: &Array1<f64>, p: usize) -> ACDResult<()> {
    if psi_lags.len() != p {
        return Err(ACDError::InvalidPsiLength { expected: p, actual: psi_lags.len() });
    }
    for (index, &value) in psi_lags.iter().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(ACDError::InvalidPsiLags { index, value });
        }
    }
    Ok(())
}

/// Validate the baseline parameter ω.
///
/// Parameters
/// ----------
/// - `omega`: `f64`
///   Baseline duration mean parameter ω. Must be finite and strictly > 0.
///
/// Returns
/// -------
/// `ParamResult<()>`
///   - `Ok(())` if `omega` is finite and strictly > 0.
///   - `Err(ParamError::InvalidOmega)` otherwise.
///
/// Errors
/// ------
/// - `ParamError::InvalidOmega`
///   - Returned if `omega` is NaN, ±∞, or ≤ 0.
///
/// Panics
/// ------
/// - Never panics.
///
/// Notes
/// -----
/// - This helper does not check stationarity; it only enforces the basic
///   domain of ω.
///
/// Examples
/// --------
/// ```rust
/// # use rust_timeseries::duration::core::validation::validate_omega;
/// use rust_timeseries::duration::errors::ParamError;
///
/// assert!(validate_omega(1.0).is_ok());
/// assert!(matches!(validate_omega(0.0), Err(ParamError::InvalidOmega { .. })));
/// ```
pub fn validate_omega(omega: f64) -> ParamResult<()> {
    if omega <= 0.0 || !omega.is_finite() {
        return Err(ParamError::InvalidOmega { value: omega });
    }
    Ok(())
}

/// Validate the **α** vector (lags on durations, length q).
///
/// Parameters
/// ----------
/// - `alpha`: `ArrayView1<'_, f64>`
///   α coefficients of length `q`, corresponding to lags on durations τ.
///   Must be finite and non-negative.
/// - `q`: `usize`
///   Expected length / number of duration lags.
///
/// Returns
/// -------
/// `ParamResult<()>`
///   - `Ok(())` if `alpha.len() == q` and all entries are finite and ≥ 0.
///   - `Err(ParamError)` describing the first violation.
///
/// Errors
/// ------
/// - `ParamError::AlphaLengthMismatch`
///   - Returned if `alpha.len() != q`.
/// - `ParamError::InvalidAlpha`
///   - Returned if any element is NaN, ±∞, or < 0, with its index and value.
///
/// Panics
/// ------
/// - Never panics.
///
/// Notes
/// -----
/// - This function does not enforce stationarity or a specific sum of α; it
///   only checks shape and coordinate-wise validity.
///
/// Examples
/// --------
/// ```rust
/// # use rust_timeseries::duration::core::validation::validate_alpha;
/// # use rust_timeseries::duration::errors::ParamError;
/// use ndarray::array;
///
/// let alpha = array![0.1, 0.2];
/// assert!(validate_alpha(alpha.view(), 2).is_ok());
///
/// let bad_alpha = array![0.1, -0.5];
/// assert!(matches!(
///     validate_alpha(bad_alpha.view(), 2),
///     Err(ParamError::InvalidAlpha { .. })
/// ));
/// ```
pub fn validate_alpha(alpha: ArrayView1<f64>, q: usize) -> ParamResult<()> {
    if alpha.len() != q {
        return Err(ParamError::AlphaLengthMismatch { expected: q, actual: alpha.len() });
    }
    if let Some((index, &value)) =
        alpha.iter().enumerate().find(|(_, v)| **v < 0.0 || !(**v).is_finite())
    {
        return Err(ParamError::InvalidAlpha { index, value });
    }
    Ok(())
}

/// Validate the **β** vector (lags on ψ, length p).
///
/// Parameters
/// ----------
/// - `beta`: `ArrayView1<'_, f64>`
///   β coefficients of length `p`, corresponding to lags on ψ. Must be finite
///   and non-negative.
/// - `p`: `usize`
///   Expected length / number of ψ lags.
///
/// Returns
/// -------
/// `ParamResult<()>`
///   - `Ok(())` if `beta.len() == p` and all entries are finite and ≥ 0.
///   - `Err(ParamError)` describing the first violation.
///
/// Errors
/// ------
/// - `ParamError::BetaLengthMismatch`
///   - Returned if `beta.len() != p`.
/// - `ParamError::InvalidBeta`
///   - Returned if any element is NaN, ±∞, or < 0, with its index and value.
///
/// Panics
/// ------
/// - Never panics.
///
/// Notes
/// -----
/// - As with [`validate_alpha`], this only checks basic coordinate-wise
///   validity; stationarity is enforced separately.
///
/// Examples
/// --------
/// ```rust
/// # use rust_timeseries::duration::core::validation::validate_beta;
/// # use rust_timeseries::duration::errors::ParamError;
/// use ndarray::array;
///
/// let beta = array![0.3];
/// assert!(validate_beta(beta.view(), 1).is_ok());
///
/// let bad_beta = array![f64::NAN];
/// assert!(matches!(
///     validate_beta(bad_beta.view(), 1),
///     Err(ParamError::InvalidBeta { .. })
/// ));
/// ```
pub fn validate_beta(beta: ArrayView1<f64>, p: usize) -> ParamResult<()> {
    if beta.len() != p {
        return Err(ParamError::BetaLengthMismatch { expected: p, actual: beta.len() });
    }
    if let Some((index, &value)) =
        beta.iter().enumerate().find(|(_, v)| **v < 0.0 || !(**v).is_finite())
    {
        return Err(ParamError::InvalidBeta { index, value });
    }
    Ok(())
}

/// Validate α and β vector lengths against (q, p).
///
/// Parameters
/// ----------
/// - `alpha`: `ArrayView1<'_, f64>`
///   α coefficients; only the length is checked here.
/// - `beta`: `ArrayView1<'_, f64>`
///   β coefficients; only the length is checked here.
/// - `q`: `usize`
///   Expected α length.
/// - `p`: `usize`
///   Expected β length.
///
/// Returns
/// -------
/// `ParamResult<()>`
///   - `Ok(())` if `alpha.len() == q` and `beta.len() == p`.
///   - `Err(ParamError)` with a length-mismatch variant otherwise.
///
/// Errors
/// ------
/// - `ParamError::AlphaLengthMismatch` if `alpha.len() != q`.
/// - `ParamError::BetaLengthMismatch` if `beta.len() != p`.
///
/// Panics
/// ------
/// - Never panics.
///
/// Notes
/// -----
/// - This helper is useful when lengths must be checked before more detailed
///   coordinate-wise validation (or in cases where coordinates are known to
///   be valid already).
///
/// Examples
/// --------
/// ```rust
/// # use rust_timeseries::duration::core::validation::validate_alpha_beta_lengths;
/// # use rust_timeseries::duration::errors::ParamError;
/// use ndarray::array;
///
/// let alpha = array![0.1, 0.2];
/// let beta = array![0.3];
/// assert!(validate_alpha_beta_lengths(alpha.view(), beta.view(), 2, 1).is_ok());
///
/// assert!(matches!(
///     validate_alpha_beta_lengths(alpha.view(), beta.view(), 3, 1),
///     Err(ParamError::AlphaLengthMismatch { .. })
/// ));
/// ```
pub fn validate_alpha_beta_lengths(
    alpha: ArrayView1<f64>, beta: ArrayView1<f64>, q: usize, p: usize,
) -> ParamResult<()> {
    if alpha.len() != q {
        return Err(ParamError::AlphaLengthMismatch { expected: q, actual: alpha.len() });
    }
    if beta.len() != p {
        return Err(ParamError::BetaLengthMismatch { expected: p, actual: beta.len() });
    }
    Ok(())
}

/// Validate strict stationarity and the slack component.
///
/// Parameters
/// ----------
/// - `alpha`: `ArrayView1<'_, f64>`
///   α coefficients used in the ACD recursion.
/// - `beta`: `ArrayView1<'_, f64>`
///   β coefficients used in the ACD recursion.
/// - `slack`: `f64`
///   Non-negative slack variable used in the optimizer mapping so that
///   `sum(α) + sum(β) + slack` approximates `1 - STATIONARITY_MARGIN`.
///
/// Returns
/// -------
/// `ParamResult<()>`
///   - `Ok(())` if:
///     - `slack` is finite and ≥ 0, and
///     - `sum(α) + sum(β) + slack` lies within a small tolerance of
///       `1 - STATIONARITY_MARGIN`.
///   - `Err(ParamError)` otherwise.
///
/// Errors
/// ------
/// - `ParamError::InvalidSlack`
///   - Returned if `slack` is NaN, ±∞, or < 0.
/// - `ParamError::StationarityViolated`
///   - Returned if `sum(α) + sum(β) + slack` deviates from
///     `1 - STATIONARITY_MARGIN` by more than the internal tolerance. The
///     `coeff_sum` field reports `sum(α) + sum(β)`.
///
/// Panics
/// ------
/// - Never panics.
///
/// Notes
/// -----
/// - This helper assumes that α and β have already been checked for basic
///   validity (non-negativity, finiteness).
/// - The tolerance used for the equality check is currently `1e-10`.
/// - In the unconstrained→constrained optimizer mapping, this corresponds to
///   enforcing a strict stationarity margin away from the ∑α + ∑β = 1
///   boundary.
///
/// Examples
/// --------
/// ```rust
/// # use rust_timeseries::duration::core::validation::validate_stationarity_and_slack;
/// # use rust_timeseries::duration::errors::ParamError;
/// use ndarray::array;
/// use rust_timeseries::optimization::numerical_stability::transformations::STATIONARITY_MARGIN;
///
/// let alpha = array![0.2];
/// let beta = array![0.3];
/// let slack = 1.0 - STATIONARITY_MARGIN - alpha.sum() - beta.sum();
/// assert!(validate_stationarity_and_slack(alpha.view(), beta.view(), slack).is_ok());
///
/// let bad_slack = -0.1;
/// assert!(matches!(
///     validate_stationarity_and_slack(alpha.view(), beta.view(), bad_slack),
///     Err(ParamError::InvalidSlack { .. })
/// ));
/// ```
pub fn validate_stationarity_and_slack(
    alpha: ArrayView1<f64>, beta: ArrayView1<f64>, slack: f64,
) -> ParamResult<()> {
    if !(slack >= 0.0 && slack.is_finite()) {
        return Err(ParamError::InvalidSlack { value: slack });
    }
    let sum_alpha = alpha.sum();
    let sum_beta = beta.sum();
    let total_sum = sum_alpha + sum_beta + slack;

    const SUM_TOL: f64 = 1e-10;
    let target = 1.0 - STATIONARITY_MARGIN;
    if (total_sum - target).abs() > SUM_TOL {
        return Err(ParamError::StationarityViolated { coeff_sum: sum_alpha + sum_beta });
    }
    Ok(())
}

/// Validate inputs for log-likelihood evaluation.
///
/// Parameters
/// ----------
/// - `x`: `f64`
///   Observed duration value used in the likelihood. Must be finite and
///   positive.
/// - `psi`: `f64`
///   Conditional mean ψ_t at the same index. Must be finite and strictly > 0.
///
/// Returns
/// -------
/// `ACDResult<()>`
///   - `Ok(())` if `x` is finite and > 0, and `psi` is finite and strictly > 0.
///   - `Err(ACDError)` otherwise.
///
/// Errors
/// ------
/// - `ACDError::InvalidLogLikInput`
///   - Returned if `x` is NaN, ±∞, or ≤ 0.
/// - `ACDError::InvalidPsiLogLik`
///   - Returned if `psi` is NaN, ±∞, or ≤ 0.
///
/// Panics
/// ------
/// - Never panics.
///
/// Notes
/// -----
/// - This helper is meant to be called inside the likelihood hot path to
///   guard against numerical issues. It performs only scalar checks.
///
/// Examples
/// --------
/// ```rust
/// # use rust_timeseries::duration::core::validation::validate_loglik_params;
/// # use rust_timeseries::duration::errors::ACDError;
///
/// assert!(validate_loglik_params(1.0, 0.5).is_ok());
///
/// assert!(matches!(
///     validate_loglik_params(-1.0, 0.5),
///     Err(ACDError::InvalidLogLikInput { .. })
/// ));
/// assert!(matches!(
///     validate_loglik_params(1.0, 0.0),
///     Err(ACDError::InvalidPsiLogLik { .. })
/// ));
/// ```
pub fn validate_loglik_params(x: f64, psi: f64) -> ACDResult<()> {
    if !x.is_finite() || x <= 0.0 {
        return Err(ACDError::InvalidLogLikInput { value: x });
    }
    if !psi.is_finite() || psi <= 0.0 {
        return Err(ACDError::InvalidPsiLogLik { value: psi });
    }
    Ok(())
}

/// Validate unconstrained optimizer parameters θ.
///
/// Parameters
/// ----------
/// - `theta`: `ArrayView1<'_, f64>`
///   Unconstrained parameter vector θ, expected to have length `1 + p + q`
///   (ω plus logits or transformed α/β coordinates). All entries must be
///   finite.
/// - `p`: `usize`
///   Number of β coordinates expected in θ.
/// - `q`: `usize`
///   Number of α coordinates expected in θ.
///
/// Returns
/// -------
/// `ParamResult<()>`
///   - `Ok(())` if `theta.len() == 1 + p + q` and all entries are finite.
///   - `Err(ParamError)` otherwise.
///
/// Errors
/// ------
/// - `ParamError::ThetaLengthMismatch`
///   - Returned if `theta.len() != 1 + p + q`.
/// - `ParamError::InvalidThetaInput`
///   - Returned if any entry of θ is NaN or ±∞, with its index and value.
///
/// Panics
/// ------
/// - Never panics.
///
/// Notes
/// -----
/// - This helper does **not** check the semantic meaning of θ’s coordinates;
///   it only enforces length and finiteness.
/// - Use it at the top of your unconstrained→constrained mapping to fail fast
///   on clearly invalid optimization inputs.
///
/// Examples
/// --------
/// ```rust
/// # use rust_timeseries::duration::core::validation::validate_theta;
/// # use rust_timeseries::duration::errors::ParamError;
/// use ndarray::array;
///
/// let theta = array![0.0, 1.0, 2.0]; // 1 + p + q = 3 with p = 1, q = 1
/// assert!(validate_theta(theta.view(), 1, 1).is_ok());
///
/// let bad_theta = array![0.0, f64::NAN, 2.0];
/// assert!(matches!(
///     validate_theta(bad_theta.view(), 1, 1),
///     Err(ParamError::InvalidThetaInput { .. })
/// ));
/// ```
pub fn validate_theta<'a>(
    theta: ndarray::ArrayView1<'a, f64>, p: usize, q: usize,
) -> ParamResult<()> {
    let expected_len = 1 + p + q;
    if theta.len() != expected_len {
        return Err(ParamError::ThetaLengthMismatch {
            expected: expected_len,
            actual: theta.len(),
        });
    }
    for (index, &value) in theta.iter().enumerate() {
        if !value.is_finite() {
            return Err(ParamError::InvalidThetaInput { index, value });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duration::errors::{ACDError, ParamError};
    use crate::optimization::numerical_stability::transformations::STATIONARITY_MARGIN;
    use ndarray::array;

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Validation of scalar distribution parameters (Weibull / generalized Gamma).
    // - Validation of pre-sample duration / ψ lags (length, finiteness, positivity).
    // - Validation of ω, α, β, and θ against their documented domain constraints.
    // - Validation of strict stationarity and slack consistency with STATIONARITY_MARGIN.
    // - Validation of log-likelihood inputs (x, ψ) used in ACD likelihood code.
    //
    // They intentionally DO NOT cover:
    // - High-level ACD model behavior (recursions, likelihood values, forecasts).
    // - Python wrapper semantics or error translation at the PyO3 boundary.
    // - Optimizer convergence properties or Hessian/covariance behavior.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // `validate_weibull_param` accepts finite, strictly positive parameters.
    //
    // Given
    // -----
    // - `param = 1.0` (finite and > 0).
    //
    // Expect
    // ------
    // - `Ok(1.0)` is returned.
    fn validate_weibull_param_with_positive_finite_returns_ok() {
        // Arrange
        let param = 1.0_f64;

        // Act
        let result = validate_weibull_param(param);

        // Assert
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), param);
    }

    #[test]
    // Purpose
    // -------
    // `validate_weibull_param` rejects NaN, ±∞, and non-positive parameters.
    //
    // Given
    // -----
    // - A set of invalid parameters (0.0, -1.0, NaN, ±∞).
    //
    // Expect
    // ------
    // - `Err(ACDError::InvalidWeibullParam { .. })` for each input.
    fn validate_weibull_param_with_invalid_values_returns_error() {
        // Arrange
        let invalid_params = [0.0_f64, -1.0_f64, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

        // Act & Assert
        for &param in &invalid_params {
            let result = validate_weibull_param(param);
            match result {
                Err(ACDError::InvalidWeibullParam { param: p, reason }) => {
                    assert!(!reason.is_empty(), "reason should be a non-empty diagnostic message");
                    if param.is_nan() {
                        assert!(p.is_nan());
                    } else {
                        assert_eq!(p, param);
                    }
                }
                other => panic!("expected InvalidWeibullParam for {param:?}, got: {other:?}"),
            }
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_gamma_param` accepts finite, strictly positive parameters.
    //
    // Given
    // -----
    // - `param = 2.5` (finite and > 0).
    //
    // Expect
    // ------
    // - `Ok(2.5)` is returned.
    fn validate_gamma_param_with_positive_finite_returns_ok() {
        // Arrange
        let param = 2.5_f64;

        // Act
        let result = validate_gamma_param(param);

        // Assert
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), param);
    }

    #[test]
    // Purpose
    // -------
    // `validate_gamma_param` rejects NaN, ±∞, and non-positive parameters.
    //
    // Given
    // -----
    // - A set of invalid parameters (0.0, -1.0, NaN, ±∞).
    //
    // Expect
    // ------
    // - `Err(ACDError::InvalidGenGammaParam { .. })` for each input.
    fn validate_gamma_param_with_invalid_values_returns_error() {
        // Arrange
        let invalid_params = [0.0_f64, -1.0_f64, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

        // Act & Assert
        for &param in &invalid_params {
            let result = validate_gamma_param(param);
            match result {
                Err(ACDError::InvalidGenGammaParam { param: p, reason }) => {
                    assert!(!reason.is_empty(), "reason should be a non-empty diagnostic message");
                    if param.is_nan() {
                        assert!(p.is_nan());
                    } else {
                        assert_eq!(p, param);
                    }
                }
                other => panic!("expected InvalidGenGammaParam for {param:?}, got: {other:?}"),
            }
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_duration_lags` accepts correct length and strictly positive values.
    //
    // Given
    // -----
    // - `duration_lags = [1.0, 2.0, 3.0]`.
    // - `q = 3`.
    //
    // Expect
    // ------
    // - `Ok(())` is returned.
    fn validate_duration_lags_with_correct_length_and_positive_values_returns_ok() {
        // Arrange
        let duration_lags = array![1.0_f64, 2.0_f64, 3.0_f64];
        let q = 3_usize;

        // Act
        let result = validate_duration_lags(&duration_lags, q);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    // Purpose
    // -------
    // `validate_duration_lags` rejects length mismatches with InvalidDurationLength.
    //
    // Given
    // -----
    // - `duration_lags = [1.0, 2.0]`.
    // - `q = 3`.
    //
    // Expect
    // ------
    // - `Err(ACDError::InvalidDurationLength { expected: 3, actual: 2 })`.
    fn validate_duration_lags_with_length_mismatch_returns_invalid_duration_length() {
        // Arrange
        let duration_lags = array![1.0_f64, 2.0_f64];
        let q = 3_usize;

        // Act
        let result = validate_duration_lags(&duration_lags, q);

        // Assert
        match result {
            Err(ACDError::InvalidDurationLength { expected, actual }) => {
                assert_eq!(expected, 3);
                assert_eq!(actual, 2);
            }
            other => panic!("expected InvalidDurationLength error, got: {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_duration_lags` rejects non-finite or non-positive entries.
    //
    // Given
    // -----
    // - `duration_lags = [1.0, 0.0, 3.0]`.
    // - `q = 3`.
    //
    // Expect
    // ------
    // - `Err(ACDError::InvalidDurationLags { index: 1, value: 0.0 })`.
    fn validate_duration_lags_with_non_positive_value_returns_invalid_duration_lags() {
        // Arrange
        let duration_lags = array![1.0_f64, 0.0_f64, 3.0_f64];
        let q = 3_usize;

        // Act
        let result = validate_duration_lags(&duration_lags, q);

        // Assert
        match result {
            Err(ACDError::InvalidDurationLags { index, value }) => {
                assert_eq!(index, 1);
                assert_eq!(value, 0.0);
            }
            other => panic!("expected InvalidDurationLags error at index 1, got: {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_psi_lags` accepts correct length and strictly positive values.
    //
    // Given
    // -----
    // - `psi_lags = [0.5, 0.6]`.
    // - `p = 2`.
    //
    // Expect
    // ------
    // - `Ok(())` is returned.
    fn validate_psi_lags_with_correct_length_and_positive_values_returns_ok() {
        // Arrange
        let psi_lags = array![0.5_f64, 0.6_f64];
        let p = 2_usize;

        // Act
        let result = validate_psi_lags(&psi_lags, p);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    // Purpose
    // -------
    // `validate_psi_lags` rejects length mismatches with InvalidPsiLength.
    //
    // Given
    // -----
    // - `psi_lags = [0.5]`.
    // - `p = 2`.
    //
    // Expect
    // ------
    // - `Err(ACDError::InvalidPsiLength { expected: 2, actual: 1 })`.
    fn validate_psi_lags_with_length_mismatch_returns_invalid_psi_length() {
        // Arrange
        let psi_lags = array![0.5_f64];
        let p = 2_usize;

        // Act
        let result = validate_psi_lags(&psi_lags, p);

        // Assert
        match result {
            Err(ACDError::InvalidPsiLength { expected, actual }) => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            }
            other => panic!("expected InvalidPsiLength error, got: {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_psi_lags` rejects non-finite or non-positive entries.
    //
    // Given
    // -----
    // - `psi_lags = [0.5, f64::INFINITY]`.
    // - `p = 2`.
    //
    // Expect
    // ------
    // - `Err(ACDError::InvalidPsiLags { index: 1, .. })`.
    fn validate_psi_lags_with_non_finite_or_non_positive_value_returns_invalid_psi_lags() {
        // Arrange
        let psi_lags = array![0.5_f64, f64::INFINITY];
        let p = 2_usize;

        // Act
        let result = validate_psi_lags(&psi_lags, p);

        // Assert
        match result {
            Err(ACDError::InvalidPsiLags { index, value }) => {
                assert_eq!(index, 1);
                assert!(value.is_infinite());
            }
            other => panic!("expected InvalidPsiLags error, got: {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_omega` accepts finite, strictly positive ω.
    //
    // Given
    // -----
    // - `omega = 1.0`.
    //
    // Expect
    // ------
    // - `Ok(())` is returned.
    fn validate_omega_with_positive_finite_returns_ok() {
        // Arrange
        let omega = 1.0_f64;

        // Act
        let result = validate_omega(omega);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    // Purpose
    // -------
    // `validate_omega` rejects non-positive or non-finite ω.
    //
    // Given
    // -----
    // - `omega = 0.0` and `omega = NaN` as representative bad values.
    //
    // Expect
    // ------
    // - `Err(ParamError::InvalidOmega { .. })` for each.
    fn validate_omega_with_non_positive_or_non_finite_returns_invalid_omega() {
        // Arrange
        let invalid_omegas = [0.0_f64, -1.0_f64, f64::NAN, f64::INFINITY];

        // Act & Assert
        for &omega in &invalid_omegas {
            let result = validate_omega(omega);
            match result {
                Err(ParamError::InvalidOmega { value }) => {
                    if omega.is_nan() {
                        assert!(value.is_nan());
                    } else {
                        assert_eq!(value, omega);
                    }
                }
                other => panic!("expected InvalidOmega for {omega:?}, got: {other:?}"),
            }
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_alpha` accepts correct length and finite, non-negative entries.
    //
    // Given
    // -----
    // - `alpha = [0.1, 0.2]`, `q = 2`.
    //
    // Expect
    // ------
    // - `Ok(())` is returned.
    fn validate_alpha_with_correct_length_and_non_negative_returns_ok() {
        // Arrange
        let alpha = array![0.1_f64, 0.2_f64];
        let q = 2_usize;

        // Act
        let result = validate_alpha(alpha.view(), q);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    // Purpose
    // -------
    // `validate_alpha` rejects length mismatches with AlphaLengthMismatch.
    //
    // Given
    // -----
    // - `alpha = [0.1]`, `q = 2`.
    //
    // Expect
    // ------
    // - `Err(ParamError::AlphaLengthMismatch { expected: 2, actual: 1 })`.
    fn validate_alpha_with_length_mismatch_returns_alpha_length_mismatch() {
        // Arrange
        let alpha = array![0.1_f64];
        let q = 2_usize;

        // Act
        let result = validate_alpha(alpha.view(), q);

        // Assert
        match result {
            Err(ParamError::AlphaLengthMismatch { expected, actual }) => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            }
            other => panic!("expected AlphaLengthMismatch error, got: {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_alpha` rejects negative or non-finite coordinates with InvalidAlpha.
    //
    // Given
    // -----
    // - `alpha = [0.1, -0.2]`, `q = 2`.
    //
    // Expect
    // ------
    // - `Err(ParamError::InvalidAlpha { index: 1, value: -0.2 })`.
    fn validate_alpha_with_negative_or_non_finite_value_returns_invalid_alpha() {
        // Arrange
        let alpha = array![0.1_f64, -0.2_f64];
        let q = 2_usize;

        // Act
        let result = validate_alpha(alpha.view(), q);

        // Assert
        match result {
            Err(ParamError::InvalidAlpha { index, value }) => {
                assert_eq!(index, 1);
                assert_eq!(value, -0.2_f64);
            }
            other => panic!("expected InvalidAlpha error, got: {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_beta` accepts correct length and finite, non-negative entries.
    //
    // Given
    // -----
    // - `beta = [0.3]`, `p = 1`.
    //
    // Expect
    // ------
    // - `Ok(())` is returned.
    fn validate_beta_with_correct_length_and_non_negative_returns_ok() {
        // Arrange
        let beta = array![0.3_f64];
        let p = 1_usize;

        // Act
        let result = validate_beta(beta.view(), p);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    // Purpose
    // -------
    // `validate_beta` rejects length mismatches with BetaLengthMismatch.
    //
    // Given
    // -----
    // - `beta = [0.3]`, `p = 2`.
    //
    // Expect
    // ------
    // - `Err(ParamError::BetaLengthMismatch { expected: 2, actual: 1 })`.
    fn validate_beta_with_length_mismatch_returns_beta_length_mismatch() {
        // Arrange
        let beta = array![0.3_f64];
        let p = 2_usize;

        // Act
        let result = validate_beta(beta.view(), p);

        // Assert
        match result {
            Err(ParamError::BetaLengthMismatch { expected, actual }) => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            }
            other => panic!("expected BetaLengthMismatch error, got: {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_beta` rejects negative or non-finite coordinates with InvalidBeta.
    //
    // Given
    // -----
    // - `beta = [NaN]`, `p = 1`.
    //
    // Expect
    // ------
    // - `Err(ParamError::InvalidBeta { index: 0, .. })`.
    fn validate_beta_with_negative_or_non_finite_value_returns_invalid_beta() {
        // Arrange
        let beta = array![f64::NAN];
        let p = 1_usize;

        // Act
        let result = validate_beta(beta.view(), p);

        // Assert
        match result {
            Err(ParamError::InvalidBeta { index, value }) => {
                assert_eq!(index, 0);
                assert!(value.is_nan());
            }
            other => panic!("expected InvalidBeta error, got: {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_alpha_beta_lengths` accepts matching (len(alpha), len(beta))
    // against (q, p).
    //
    // Given
    // -----
    // - `alpha = [0.1, 0.2]`, `beta = [0.3]`, `q = 2`, `p = 1`.
    //
    // Expect
    // ------
    // - `Ok(())` is returned.
    fn validate_alpha_beta_lengths_with_matching_lengths_returns_ok() {
        // Arrange
        let alpha = array![0.1_f64, 0.2_f64];
        let beta = array![0.3_f64];
        let q = 2_usize;
        let p = 1_usize;

        // Act
        let result = validate_alpha_beta_lengths(alpha.view(), beta.view(), q, p);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    // Purpose
    // -------
    // `validate_alpha_beta_lengths` rejects mismatched α length.
    //
    // Given
    // -----
    // - `alpha.len() = 2`, `q = 3`.
    //
    // Expect
    // ------
    // - `Err(ParamError::AlphaLengthMismatch { expected: 3, actual: 2 })`.
    fn validate_alpha_beta_lengths_with_alpha_mismatch_returns_alpha_length_mismatch() {
        // Arrange
        let alpha = array![0.1_f64, 0.2_f64];
        let beta = array![0.3_f64];
        let q = 3_usize;
        let p = 1_usize;

        // Act
        let result = validate_alpha_beta_lengths(alpha.view(), beta.view(), q, p);

        // Assert
        match result {
            Err(ParamError::AlphaLengthMismatch { expected, actual }) => {
                assert_eq!(expected, 3);
                assert_eq!(actual, 2);
            }
            other => panic!("expected AlphaLengthMismatch error, got: {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_alpha_beta_lengths` rejects mismatched β length.
    //
    // Given
    // -----
    // - `beta.len() = 1`, `p = 2`.
    //
    // Expect
    // ------
    // - `Err(ParamError::BetaLengthMismatch { expected: 2, actual: 1 })`.
    fn validate_alpha_beta_lengths_with_beta_mismatch_returns_beta_length_mismatch() {
        // Arrange
        let alpha = array![0.1_f64, 0.2_f64];
        let beta = array![0.3_f64];
        let q = 2_usize;
        let p = 2_usize;

        // Act
        let result = validate_alpha_beta_lengths(alpha.view(), beta.view(), q, p);

        // Assert
        match result {
            Err(ParamError::BetaLengthMismatch { expected, actual }) => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            }
            other => panic!("expected BetaLengthMismatch error, got: {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_stationarity_and_slack` accepts α, β, and slack that satisfy
    // the sum condition with STATIONARITY_MARGIN.
    //
    // Given
    // -----
    // - α and β with small positive entries, and slack chosen so that
    //   sum(α) + sum(β) + slack = 1 - STATIONARITY_MARGIN (up to rounding).
    //
    // Expect
    // ------
    // - `Ok(())` is returned.
    fn validate_stationarity_and_slack_with_consistent_sum_returns_ok() {
        // Arrange
        let alpha = array![0.2_f64];
        let beta = array![0.3_f64];
        let coeff_sum = alpha.sum() + beta.sum();
        let slack = 1.0_f64 - STATIONARITY_MARGIN - coeff_sum;

        // Act
        let result = validate_stationarity_and_slack(alpha.view(), beta.view(), slack);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    // Purpose
    // -------
    // `validate_stationarity_and_slack` rejects negative or non-finite slack
    // with InvalidSlack.
    //
    // Given
    // -----
    // - α, β valid; `slack = -0.1`.
    //
    // Expect
    // ------
    // - `Err(ParamError::InvalidSlack { value: -0.1 })`.
    fn validate_stationarity_and_slack_with_invalid_slack_returns_invalid_slack() {
        // Arrange
        let alpha = array![0.2_f64];
        let beta = array![0.3_f64];
        let slack = -0.1_f64;

        // Act
        let result = validate_stationarity_and_slack(alpha.view(), beta.view(), slack);

        // Assert
        match result {
            Err(ParamError::InvalidSlack { value }) => {
                assert_eq!(value, slack);
            }
            other => panic!("expected InvalidSlack error, got: {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_stationarity_and_slack` rejects combinations where the sum
    // deviates from `1 - STATIONARITY_MARGIN` beyond the tolerance.
    //
    // Given
    // -----
    // - α, β valid; slack artificially perturbed to move the sum outside
    //   the allowed tolerance.
    //
    // Expect
    // ------
    // - `Err(ParamError::StationarityViolated { coeff_sum })`.
    fn validate_stationarity_and_slack_with_inconsistent_sum_returns_stationarity_violated() {
        // Arrange
        let alpha = array![0.2_f64];
        let beta = array![0.3_f64];
        let coeff_sum = alpha.sum() + beta.sum();
        // First compute a valid slack, then perturb it beyond the tolerance.
        let valid_slack = 1.0_f64 - STATIONARITY_MARGIN - coeff_sum;
        let slack = valid_slack + 1e-6; // larger than SUM_TOL = 1e-10 internally

        // Act
        let result = validate_stationarity_and_slack(alpha.view(), beta.view(), slack);

        // Assert
        match result {
            Err(ParamError::StationarityViolated { coeff_sum: reported_sum }) => {
                assert!((reported_sum - coeff_sum).abs() < 1e-12);
            }
            other => panic!("expected StationarityViolated error, got: {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_loglik_params` accepts strictly positive, finite x and ψ.
    //
    // Given
    // -----
    // - `x = 1.0`, `psi = 0.5`.
    //
    // Expect
    // ------
    // - `Ok(())` is returned.
    fn validate_loglik_params_with_positive_finite_values_returns_ok() {
        // Arrange
        let x = 1.0_f64;
        let psi = 0.5_f64;

        // Act
        let result = validate_loglik_params(x, psi);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    // Purpose
    // -------
    // `validate_loglik_params` rejects invalid x values with InvalidLogLikInput.
    //
    // Given
    // -----
    // - x in {0.0, -1.0, NaN, ±∞}, ψ fixed positive.
    //
    // Expect
    // ------
    // - `Err(ACDError::InvalidLogLikInput { value: x })` for each case.
    fn validate_loglik_params_with_invalid_x_returns_invalid_loglik_input() {
        // Arrange
        let psi = 0.5_f64;
        let invalid_x = [0.0_f64, -1.0_f64, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

        // Act & Assert
        for &x in &invalid_x {
            let result = validate_loglik_params(x, psi);
            match result {
                Err(ACDError::InvalidLogLikInput { value }) => {
                    if x.is_nan() {
                        assert!(value.is_nan());
                    } else {
                        assert_eq!(value, x);
                    }
                }
                other => panic!("expected InvalidLogLikInput for x={x:?}, got: {other:?}"),
            }
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_loglik_params` rejects invalid ψ values with InvalidPsiLogLik.
    //
    // Given
    // -----
    // - ψ in {0.0, -0.5, NaN, ±∞}, x fixed positive.
    //
    // Expect
    // ------
    // - `Err(ACDError::InvalidPsiLogLik { value: psi })` for each case.
    fn validate_loglik_params_with_invalid_psi_returns_invalid_psi_loglik() {
        // Arrange
        let x = 1.0_f64;
        let invalid_psi = [0.0_f64, -0.5_f64, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

        // Act & Assert
        for &psi in &invalid_psi {
            let result = validate_loglik_params(x, psi);
            match result {
                Err(ACDError::InvalidPsiLogLik { value }) => {
                    if psi.is_nan() {
                        assert!(value.is_nan());
                    } else {
                        assert_eq!(value, psi);
                    }
                }
                other => panic!("expected InvalidPsiLogLik for psi={psi:?}, got: {other:?}"),
            }
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_theta` accepts correct length and finite entries.
    //
    // Given
    // -----
    // - `p = 1`, `q = 1`, θ length = 3 with finite values.
    //
    // Expect
    // ------
    // - `Ok(())` is returned.
    fn validate_theta_with_correct_length_and_finite_values_returns_ok() {
        // Arrange
        let p = 1_usize;
        let q = 1_usize;
        let theta = array![0.0_f64, 1.0_f64, -2.0_f64];

        // Act
        let result = validate_theta(theta.view(), p, q);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    // Purpose
    // -------
    // `validate_theta` rejects length mismatches with ThetaLengthMismatch.
    //
    // Given
    // -----
    // - `p = 1`, `q = 1`, expected θ length = 3, actual length = 2.
    //
    // Expect
    // ------
    // - `Err(ParamError::ThetaLengthMismatch { expected: 3, actual: 2 })`.
    fn validate_theta_with_length_mismatch_returns_theta_length_mismatch() {
        // Arrange
        let p = 1_usize;
        let q = 1_usize;
        let theta = array![0.0_f64, 1.0_f64];

        // Act
        let result = validate_theta(theta.view(), p, q);

        // Assert
        match result {
            Err(ParamError::ThetaLengthMismatch { expected, actual }) => {
                assert_eq!(expected, 1 + p + q);
                assert_eq!(actual, theta.len());
            }
            other => panic!("expected ThetaLengthMismatch error, got: {other:?}"),
        }
    }

    #[test]
    // Purpose
    // -------
    // `validate_theta` rejects non-finite coordinates with InvalidThetaInput.
    //
    // Given
    // -----
    // - θ length = 3, with a NaN at index 1.
    //
    // Expect
    // ------
    // - `Err(ParamError::InvalidThetaInput { index: 1, .. })`.
    fn validate_theta_with_non_finite_value_returns_invalid_theta_input() {
        // Arrange
        let p = 1_usize;
        let q = 1_usize;
        let theta = array![0.0_f64, f64::NAN, 2.0_f64];

        // Act
        let result = validate_theta(theta.view(), p, q);

        // Assert
        match result {
            Err(ParamError::InvalidThetaInput { index, value }) => {
                assert_eq!(index, 1);
                assert!(value.is_nan());
            }
            other => panic!("expected InvalidThetaInput error at index 1, got: {other:?}"),
        }
    }
}
