//! Innovation distributions for ACD(p, q) models.
//!
//! This module defines [`ACDInnovation`], which enumerates the supported
//! innovation (error) distributions used in Autoregressive Conditional Duration
//! models. Each distribution is parameterized and validated to enforce the
//! **unit-mean constraint** `E[ε_t] = 1`, which is required for stationarity
//! and identifiability.
//!
//! ## Supported distributions
//! - [`ACDInnovation::Exponential`]: canonical exponential(1) distribution (no parameters).
//! - [`ACDInnovation::Weibull`]: Weibull with shape `k` and scale `λ`
//!   constrained to satisfy unit mean.
//! - [`ACDInnovation::GeneralizedGamma`]: generalized gamma with scale `a`
//!   and shapes `d, p`, constrained to satisfy unit mean.
//!
//! ## Numerics
//! - Scale parameters are either computed automatically (given shapes) or
//!   validated if provided explicitly.
//! - Unit-mean checks use a small tolerance (`1e-10`).
//! - Log-gamma functions are used internally for numerical stability.
use crate::duration::{
    core::validation::{validate_loglik_params, verify_gamma_param, verify_weibull_param},
    errors::{ACDError, ACDResult},
};
use statrs::{
    distribution::{Continuous, Exp, Weibull},
    function::gamma,
};

// Constants
const UNIT_MEAN_TOL: f64 = 1e-10;

/// Innovation (error) distributions for ACD models.
///
/// Variants encode exponential, Weibull, and generalized gamma distributions.
/// All enforce the unit-mean condition `E[ε_t] = 1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ACDInnovation {
    /// Exponential(1) innovations (mean = 1).
    Exponential,
    /// Weibull innovations with scale λ > 0 and shape k > 0.
    Weibull {
        lambda: f64, // scale parameter
        k: f64,
    },
    /// Generalized gamma innovations with scale a > 0, shape d > 0, and power p > 0.
    GeneralizedGamma {
        a: f64, // scale parameter
        d: f64, // shape parameter
        p: f64,
    },
}

impl ACDInnovation {
    /// Exponential(1) innovations (unit mean).
    ///
    /// Returns the exponential distribution with mean = 1 (rate = 1).
    /// Requires no parameters and always succeeds.
    pub const fn exponential() -> Self {
        ACDInnovation::Exponential
    }

    /// Weibull innovations with **unit mean**, computed from shape `k > 0`.
    ///
    /// - Computes `λ = 1 / Γ(1 + 1/k)` so that `E[ε] = 1`.
    /// - Uses log-gamma internally for stability.
    ///
    /// # Errors
    /// Returns [`ACDError::InvalidWeibullParam`] if `k` is not finite or ≤ 0.
    pub fn weibull(k: f64) -> ACDResult<Self> {
        let k = verify_weibull_param(k)?;
        let lambda = (-gamma::ln_gamma(1.0 + 1.0 / k)).exp();
        Ok(ACDInnovation::Weibull { lambda, k })
    }

    /// Weibull innovations with user-provided `λ` and `k`, validated for unit mean.
    ///
    /// - Checks `λ Γ(1 + 1/k) ≈ 1` within tolerance (`1e-10`).
    /// - Normalizes λ to exactly satisfy the constraint if within tolerance.
    ///
    /// # Errors
    /// - [`ACDError::InvalidWeibullParam`] if `λ` or `k` are invalid.
    /// - [`ACDError::InvalidUnitMeanWeibull`] if the mean deviates from 1 beyond tolerance.
    pub fn weibull_with_lambda(lambda: f64, k: f64) -> ACDResult<Self> {
        let k = verify_weibull_param(k)?;
        let mut lambda = verify_weibull_param(lambda)?;
        let uncond_mean = lambda * gamma::ln_gamma(1.0 + 1.0 / k).exp();
        if (uncond_mean - 1.0).abs() > UNIT_MEAN_TOL {
            return Err(ACDError::InvalidUnitMeanWeibull { mean: uncond_mean });
        } else {
            lambda /= uncond_mean;
        }
        Ok(ACDInnovation::Weibull { lambda, k })
    }

    /// Generalized-gamma innovations with **unit mean**, given shapes `p, d > 0`.
    ///
    /// - Computes `a = Γ(d/p) / Γ((d+1)/p)` so that `E[ε] = 1`.
    /// - Uses log-gamma differences for stability.
    ///
    /// # Errors
    /// Returns [`ACDError::InvalidGenGammaParam`] if `p` or `d` are invalid.
    pub fn generalized_gamma(p: f64, d: f64) -> ACDResult<Self> {
        let p = verify_gamma_param(p)?;
        let d = verify_gamma_param(d)?;
        let a = (gamma::ln_gamma(d / p) - gamma::ln_gamma((d + 1.0) / p)).exp();
        Ok(ACDInnovation::GeneralizedGamma { a, d, p })
    }

    /// Generalized-gamma innovations with user-provided `a, p, d`, validated for unit mean.
    ///
    /// - Checks `a Γ(d/p) / Γ((d+1)/p) ≈ 1` within tolerance.
    /// - Normalizes `a` if the check passes.
    ///
    /// # Errors
    /// - [`ACDError::InvalidGenGammaParam`] if any parameter is invalid.
    /// - [`ACDError::InvalidUnitMeanGenGamma`] if the mean deviates from 1 beyond tolerance.
    pub fn generalized_gamma_with_a(a: f64, p: f64, d: f64) -> ACDResult<Self> {
        let mut a = verify_gamma_param(a)?;
        let p = verify_gamma_param(p)?;
        let d = verify_gamma_param(d)?;
        let uncond_mean = (a.ln() + gamma::ln_gamma((d + 1.0) / p) - gamma::ln_gamma(d / p)).exp();
        if (uncond_mean - 1.0).abs() > UNIT_MEAN_TOL {
            return Err(ACDError::InvalidUnitMeanGenGamma { mean: uncond_mean });
        } else {
            a /= uncond_mean;
        }
        Ok(ACDInnovation::GeneralizedGamma { a, d, p })
    }

    /// Evaluate the per-observation log-likelihood contribution for a duration `x`
    /// given conditional mean `psi` under this innovation law.
    ///
    /// The ACD model assumes `x = psi * ε` with `E[ε] = 1`. By the
    /// change-of-variables formula,
    /// `log f_X(x | psi) = log f_ε(x/psi) - log(psi)`.
    /// This method computes that quantity in a numerically stable manner by
    /// precomputing `ln(psi)` and evaluating the innovation log-pdf at
    /// `ε = x/psi`.
    ///
    /// # Arguments
    /// - `x`: Observed duration (must be finite and ≥ 0).
    /// - `psi`: Conditional mean for the same time point (must be finite and > 0).
    ///
    /// # Behavior
    /// - For `Exponential`, delegates to `statrs::distribution::Exp::ln_pdf`.
    /// - For `Weibull { lambda, k }`, delegates to
    ///   `statrs::distribution::Weibull::ln_pdf`
    ///   (ensure constructor argument order matches statrs: shape, scale).
    /// - For `GeneralizedGamma { a, d, p }`, uses a stable log-gamma formula
    ///   for `log f_ε(ε)`.
    ///
    /// # Returns
    /// The scalar log-likelihood contribution `log f_X(x | psi)`.
    ///
    /// # Errors
    /// - Returns an `ACDError` if inputs are invalid (non-finite or out of domain)
    ///   or if the underlying distribution constructor in `statrs` rejects its
    ///   parameters.
    ///
    /// # Notes
    /// - Using `log f_ε(x/psi) - log(psi)` avoids avoidable rounding error versus
    ///   computing `f_ε(x/psi)/psi` and then taking `log`.
    /// - If `x = 0`, the term may be `-∞` depending on the innovation law; this is
    ///   expected and propagates correctly in the total log-likelihood.
    pub fn log_pdf_duration(&self, x: f64, psi: f64) -> ACDResult<f64> {
        validate_loglik_params(x, psi)?;
        let ln_psi = psi.ln();
        let eps = x / psi;
        match self {
            ACDInnovation::Exponential => Ok(Exp::new(1.0)?.ln_pdf(eps) - ln_psi),
            ACDInnovation::Weibull { lambda, k } => {
                Ok(Weibull::new(*k, *lambda)?.ln_pdf(eps) - ln_psi)
            }
            ACDInnovation::GeneralizedGamma { a, d, p } => {
                let log_lik = d * (p.ln() - a.ln()) + (d - 1.0) * (eps).ln()
                    - ((eps) / a).powf(*p)
                    - gamma::ln_gamma(d / p);
                Ok(log_lik - ln_psi)
            }
        }
    }
}
