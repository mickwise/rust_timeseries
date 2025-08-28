//! Validation helpers for ACD duration models.
//!
//! This module centralizes small, reusable checks used across the duration
//! stack—distribution parameters, pre-sample lags, and model-space parameter
//! validity (ω, α, β, slack, stationarity).
//!
//! **Convention (Wikipedia-style):**
//! - `q` = number of **α** terms (lags on durations τ).
//! - `p` = number of **β** terms (lags on conditional means ψ).
//!
//! **Stationarity margin:** we enforce strict stationarity by reserving a small
//! buffer `1e-6`, so `sum(α) + sum(β) + slack = 1 - margin`. Staying off the
//! boundary avoids numerical blow-ups during recursion/likelihood.
use crate::duration::errors::{ACDError, ACDResult, ParamError, ParamResult};
use ndarray::Array1;

/// Safety margin for strict stationarity in ACD models.
///
/// In an ACD(p, q), the stability condition requires
///   sum(alpha) + sum(beta) < 1.
/// This margin enforces the inequality *strictly* by reserving a small
/// buffer (default = 1e-6). Practically, the recursion always runs inside
/// the stable region, avoiding borderline cases that can cause blow-ups
/// in likelihood evaluation.
const STATIONARITY_MARGIN: f64 = 1e-6;

/// Validate one Weibull parameter (scale or shape).
///
/// Requires the value to be finite and strictly positive.
///
/// # Errors
/// Returns [`ACDError::InvalidWeibullParam`] with a descriptive reason on failure.
pub fn verify_weibull_param(param: f64) -> ACDResult<f64> {
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

/// Validate one Generalized-Gamma parameter (scale or shape).
///
/// Requires the value to be finite and strictly positive.
///
/// # Errors
/// Returns [`ACDError::InvalidGenGammaParam`] with a descriptive reason on failure.
pub fn verify_gamma_param(param: f64) -> ACDResult<f64> {
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

/// Validate pre-sample **duration** lags (length **Q**).
///
/// Checks length equals `q` and every entry is finite and strictly positive.
///
/// # Errors
/// - [`ACDError::InvalidDurationLength`] on length mismatch,
/// - [`ACDError::InvalidDurationLags`] if any element is non-finite or ≤ 0.
pub fn verify_duration_lags(duration_lags: &Array1<f64>, q: usize) -> ACDResult<()> {
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

/// Validate pre-sample **ψ** lags (length **P**).
///
/// Checks length equals `p` and every entry is finite and strictly positive.
///
/// # Errors
/// - [`ACDError::InvalidPsiLength`] on length mismatch,
/// - [`ACDError::InvalidPsiLags`] if any element is non-finite or ≤ 0.
pub fn verify_psi_lags(psi_lags: &Array1<f64>, p: usize) -> ACDResult<()> {
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
pub fn validate_omega(omega: f64) -> ParamResult<()> {
    if omega <= 0.0 || !omega.is_finite() {
        return Err(ParamError::InvalidOmega { value: omega });
    }
    Ok(())
}

/// Validate the **α** vector (lags on **durations**, length **q**).
///
/// Ensures length equals `q` and all entries are finite and non-negative.
///
/// # Errors
/// - [`ParamError::AlphaLengthMismatch`] on length mismatch,
/// - [`ParamError::InvalidAlpha`] if any element is < 0 or non-finite.
pub fn validate_alpha(alpha: &Array1<f64>, q: usize) -> ParamResult<()> {
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

/// Validate the **β** vector (lags on **ψ**, length **p**).
///
/// Ensures length equals `p` and all entries are finite and non-negative.
///
/// # Errors
/// - [`ParamError::BetaLengthMismatch`] on length mismatch,
/// - [`ParamError::InvalidBeta`] if any element is < 0 or non-finite.
pub fn validate_beta(beta: &Array1<f64>, p: usize) -> ParamResult<()> {
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

/// Validate strict stationarity and the slack component.
///
/// Checks:
/// - `slack` is finite and ≥ 0,
/// - `sum(α) + sum(β) + slack ≈ 1 - STATIONARITY_MARGIN` (within tolerance).
///
/// # Errors
/// - [`ParamError::InvalidSlack`] if `slack` is negative or non-finite,
/// - [`ParamError::StationarityViolated`] if the equality is violated.
pub fn validate_stationarity_and_slack(
    alpha: &Array1<f64>, beta: &Array1<f64>, slack: f64,
) -> ParamResult<()> {
    if slack < 0.0 {
        return Err(ParamError::InvalidSlack { value: slack });
    }
    if !slack.is_finite() {
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
/// Checks:
/// - `x` (duration) is finite and ≥ 0,
/// - `psi` (conditional mean) is finite and > 0.
///
/// # Errors
/// - [`ACDError::InvalidLogLikInput`] if `x` is invalid,
/// - [`ACDError::InvalidPsiLogLik`] if `psi` is invalid.
pub fn validate_loglik_params(x: f64, psi: f64) -> ACDResult<()> {
    if !x.is_finite() || x < 0.0 {
        return Err(ACDError::InvalidLogLikInput { value: x });
    }
    if !psi.is_finite() || psi <= 0.0 {
        return Err(ACDError::InvalidPsiLogLik { value: psi });
    }
    Ok(())
}
