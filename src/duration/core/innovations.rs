//! ACD innovation distributions — parameterized, unit-mean error laws for durations.
//!
//! Purpose
//! -------
//! Provide a small, centralized abstraction over the innovation (error)
//! distributions used in ACD(p, q) models, enforcing the unit-mean constraint
//! `E[ε_t] = 1` for exponential, Weibull, and generalized gamma families and
//! exposing log-likelihood, gradient, and sampling utilities.
//!
//! Key behaviors
//! -------------
//! - Represent supported innovation laws via [`ACDInnovation`], including
//!   Exponential(1), Weibull, and generalized gamma distributions.
//! - Construct and validate distribution parameters so that the resulting
//!   innovations satisfy the unit-mean constraint within a tight tolerance.
//! - Provide per-observation log-likelihood, its one-dimensional gradient with
//!   respect to the conditional mean ψ, and random draws from the innovation
//!   law, delegating to `statrs` where appropriate.
//!
//! Invariants & assumptions
//! ------------------------
//! - All innovation distributions are parameterized to satisfy `E[ε_t] = 1`
//!   within a small numerical tolerance (`UNIT_MEAN_TOL`).
//! - Scale/shape parameters for Weibull and generalized gamma families must be
//!   finite and strictly positive; invalid inputs are rejected via [`ACDError`].
//! - Callers must pass valid (finite, in-domain) duration and ψ values to
//!   `log_pdf_duration` and `one_d_loglik_grad`; domain checks are performed by
//!   `validate_loglik_params`.
//!
//! Conventions
//! -----------
//! - The ACD observation model is `x_t = ψ_t · ε_t` with `E[ε_t] = 1`, so ψ_t is
//!   both the conditional mean and the scaling parameter in the likelihood.
//! - Log-likelihood contributions are computed as
//!   `log f_X(x_t | ψ_t) = log f_ε(x_t / ψ_t) - log ψ_t`.
//! - All numerics are in `f64`; gamma-related formulas use log-gamma functions
//!   from `statrs::function::gamma` for stability.
//! - Invalid inputs return [`ACDError`] rather than panicking; panics are
//!   reserved for programming errors in upstream code, not user input.
//!
//! Downstream usage
//! ----------------
//! - Construct an [`ACDInnovation`] variant at model setup time (e.g., from
//!   configuration or Python API arguments) using the associated constructors.
//! - Pass the innovation object into likelihood evaluation and gradient code,
//!   calling `log_pdf_duration` and `one_d_loglik_grad` at each time step.
//! - Use `draw_innovation` when simulating ACD paths from a fitted model.
//!
//! Testing notes
//! -------------
//! - Unit tests in this module:
//!   - verify that `weibull` / `generalized_gamma` produce unit-mean
//!     distributions within `UNIT_MEAN_TOL`,
//!   - check that `*_with_lambda` / `*_with_a` reject invalid parameters and
//!     out-of-tolerance means with the correct [`ACDError`] variants,
//!   - validate `log_pdf_duration` against `statrs` baselines for exponential
//!     and Weibull, and against a direct generalized-gamma formula.
//!   - compare `one_d_loglik_grad` to finite-difference approximations of
//!     `log_pdf_duration`.
//! - Simulation behavior of `draw_innovation` can be smoke-tested here
//!   (distributional checks, positivity) and more extensively tested in
//!   higher-level simulation modules.
use crate::duration::{
    core::validation::{validate_gamma_param, validate_loglik_params, validate_weibull_param},
    errors::{ACDError, ACDResult},
};
use rand::{prelude::Distribution, rngs::StdRng};
use statrs::{
    distribution::{Continuous, Exp, Gamma, Weibull},
    function::gamma,
};

// Constants
const UNIT_MEAN_TOL: f64 = 1e-10;

/// ACDInnovation — unit-mean innovation distributions for ACD models.
///
/// Purpose
/// -------
/// Represent the family of innovation (error) distributions used in ACD(p, q)
/// models under the constraint `E[ε_t] = 1`, including exponential, Weibull,
/// and generalized gamma laws, each with a concrete parameterization.
///
/// Key behaviors
/// -------------
/// - Encodes which innovation family is used (Exponential, Weibull,
///   GeneralizedGamma) along with its parameters.
/// - Guarantees that, when constructed via the provided constructors, the
///   resulting innovations satisfy the unit-mean constraint up to a small
///   numerical tolerance.
/// - Serves as the single entry point for computing log-likelihood
///   contributions, their gradients with respect to ψ, and random draws
///   from the innovation law.
///
/// Parameters
/// ----------
/// Constructed via:
/// - `ACDInnovation::exponential()`
///   Exponential(1) innovations (mean = 1).
/// - `ACDInnovation::weibull(k: f64)`
///   Weibull innovations with shape `k > 0`, scale λ computed to enforce
///   unit mean.
/// - `ACDInnovation::weibull_with_lambda(lambda: f64, k: f64)`
///   Weibull innovations with user-provided `lambda` and `k`, validated and
///   normalized to enforce unit mean.
/// - `ACDInnovation::generalized_gamma(p: f64, d: f64)`
///   Generalized gamma innovations with shapes `p, d > 0`, scale `a` computed
///   to enforce unit mean.
/// - `ACDInnovation::generalized_gamma_with_a(a: f64, p: f64, d: f64)`
///   Generalized gamma innovations with user-provided `a, p, d`, validated and
///   normalized to enforce unit mean.
///
/// Variants
/// --------
/// - `Exponential`
///   Canonical exponential(1) innovations with mean 1 (no parameters).
/// - `Weibull { lambda, k }`
///   Weibull innovations with scale `lambda > 0` and shape `k > 0`, already
///   parameterized to satisfy `E[ε_t] = 1`.
/// - `GeneralizedGamma { a, d, p }`
///   Generalized gamma innovations with scale `a > 0`, shape `d > 0`,
///   and power `p > 0`, parameterized to satisfy `E[ε_t] = 1`.
///
/// Invariants
/// ----------
/// - For all variants constructed via this module’s constructors:
///   - Parameters are finite and strictly positive.
///   - The implied innovation mean is 1 within `UNIT_MEAN_TOL`.
/// - The representation is stable: fields (`lambda`, `k`, `a`, `d`, `p`)
///   are stored exactly as used in log-likelihood and simulation.
///
/// Notes
/// -----
/// - This enum is intended as a public configuration surface for ACD models
///   (Python API, config files, etc.).
/// - Downstream code should pattern-match exhaustively on `ACDInnovation`
///   so that the compiler flags missing cases if new distributions are added.
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
    /// Construct Exponential(1) innovation law with unit mean.
    ///
    /// Parameters
    /// ----------
    /// - *(none)*
    ///   This is a parameter-free constructor; the exponential rate is fixed
    ///   to 1 so that `E[ε_t] = 1`.
    ///
    /// Returns
    /// -------
    /// ACDInnovation
    ///   The `ACDInnovation::Exponential` variant representing an exponential
    ///   distribution with unit mean.
    ///
    /// Errors
    /// ------
    /// - Never returns an error; this is a plain `const` constructor.
    ///
    /// Panics
    /// ------
    /// - Never panics.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; no additional guarantees are required.
    ///
    /// Notes
    /// -----
    /// - Use this constructor when you want the canonical exponential innovation
    ///   law, which is the baseline choice in many ACD specifications.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::innovation::ACDInnovation;
    /// let innovation = ACDInnovation::exponential();
    /// assert_eq!(innovation, ACDInnovation::Exponential);
    /// ```
    pub const fn exponential() -> Self {
        ACDInnovation::Exponential
    }

    /// Construct a Weibull innovation law with unit mean from the shape `k`.
    ///
    /// Parameters
    /// ----------
    /// - `k`: `f64`
    ///   Weibull shape parameter (must be finite and strictly positive).
    ///
    /// Returns
    /// -------
    /// ACDResult<ACDInnovation>
    ///   - `Ok(ACDInnovation::Weibull { lambda, k })` where `lambda` is computed
    ///     so that `E[ε_t] = 1`.
    ///   - `Err(ACDError::InvalidWeibullParam { .. })` if `k` is not finite or
    ///     not strictly positive.
    ///
    /// Errors
    /// ------
    /// - `ACDError::InvalidWeibullParam`
    ///   Returned when `k` fails basic validation (non-finite or ≤ 0).
    ///
    /// Panics
    /// ------
    /// - Never panics; invalid parameters are reported via [`ACDError`].
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; callers must only ensure that `k` is sensible for
    ///   their modeling context.
    ///
    /// Notes
    /// -----
    /// - The scale `lambda` is computed as `lambda = 1 / Γ(1 + 1/k)` so that
    ///   the implied innovation mean is 1.
    /// - Internally, a log-gamma call is used for numerical stability.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::innovation::ACDInnovation;
    /// let innovation = ACDInnovation::weibull(1.5).unwrap();
    /// if let ACDInnovation::Weibull { lambda, k } = innovation {
    ///     assert!(lambda > 0.0);
    ///     assert!((k - 1.5).abs() < 1e-12);
    /// }
    /// ```
    pub fn weibull(k: f64) -> ACDResult<Self> {
        let k = validate_weibull_param(k)?;
        let lambda = (-gamma::ln_gamma(1.0 + 1.0 / k)).exp();
        Ok(ACDInnovation::Weibull { lambda, k })
    }

    /// Construct a Weibull innovation law from user-provided `lambda` and `k`,
    /// validated and normalized to enforce unit mean.
    ///
    /// Parameters
    /// ----------
    /// - `lambda`: `f64`
    ///   Scale parameter candidate, must be finite and strictly positive.
    /// - `k`: `f64`
    ///   Shape parameter, must be finite and strictly positive.
    ///
    /// Returns
    /// -------
    /// ACDResult<ACDInnovation>
    ///   - `Ok(ACDInnovation::Weibull { lambda, k })` when both parameters are
    ///     valid and the implied mean is within `UNIT_MEAN_TOL` of 1. The internal
    ///     `lambda` is rescaled to satisfy the constraint exactly.
    ///   - `Err(ACDError::InvalidWeibullParam { .. })` if either parameter is invalid.
    ///   - `Err(ACDError::InvalidUnitMeanWeibull { mean })` if the implied mean
    ///     deviates from 1 by more than `UNIT_MEAN_TOL`.
    ///
    /// Errors
    /// ------
    /// - `ACDError::InvalidWeibullParam`
    ///   Returned when `lambda` or `k` is non-finite or ≤ 0.
    /// - `ACDError::InvalidUnitMeanWeibull`
    ///   Returned when `lambda * Γ(1 + 1/k)` is not within tolerance of 1.
    ///
    /// Panics
    /// ------
    /// - Never panics; all invalid inputs are surfaced as [`ACDError`] values.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; callers must only supply parameters they consider
    ///   meaningful for their application.
    ///
    /// Notes
    /// -----
    /// - When the mean is within tolerance, `lambda` is renormalized by dividing
    ///   by the implied mean so that the stored parameters enforce unit mean
    ///   exactly.
    /// - This is useful when you calibrate a Weibull offline and then want to
    ///   snap it onto the unit-mean constraint for ACD.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::innovation::ACDInnovation;
    /// let candidate_lambda = 0.8;
    /// let k = 1.5;
    /// let innovation = ACDInnovation::weibull_with_lambda(candidate_lambda, k);
    /// ```
    pub fn weibull_with_lambda(lambda: f64, k: f64) -> ACDResult<Self> {
        let k = validate_weibull_param(k)?;
        let mut lambda = validate_weibull_param(lambda)?;
        let uncond_mean = lambda * gamma::ln_gamma(1.0 + 1.0 / k).exp();
        if (uncond_mean - 1.0).abs() > UNIT_MEAN_TOL {
            return Err(ACDError::InvalidUnitMeanWeibull { mean: uncond_mean });
        } else {
            lambda /= uncond_mean;
        }
        Ok(ACDInnovation::Weibull { lambda, k })
    }

    /// Construct a generalized gamma innovation law with unit mean from shapes `p` and `d`.
    ///
    /// Parameters
    /// ----------
    /// - `p`: `f64`
    ///   Power parameter (must be finite and strictly positive).
    /// - `d`: `f64`
    ///   Shape parameter (must be finite and strictly positive).
    ///
    /// Returns
    /// -------
    /// ACDResult<ACDInnovation>
    ///   - `Ok(ACDInnovation::GeneralizedGamma { a, d, p })` where the scale `a`
    ///     is computed to enforce unit mean.
    ///   - `Err(ACDError::InvalidGenGammaParam { .. })` if either shape is invalid.
    ///
    /// Errors
    /// ------
    /// - `ACDError::InvalidGenGammaParam`
    ///   Returned when `p` or `d` is non-finite or ≤ 0.
    ///
    /// Panics
    /// ------
    /// - Never panics; invalid parameters are reported via [`ACDError`].
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; callers must only ensure that `p` and `d` are
    ///   sensible in their modeling context.
    ///
    /// Notes
    /// -----
    /// - The scale `a` is chosen as `a = Γ(d/p) / Γ((d + 1)/p)` so that
    ///   `E[ε_t] = 1` under the generalized gamma law.
    /// - Log-gamma differences are used internally for numerical stability.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::innovation::ACDInnovation;
    /// let innovation = ACDInnovation::generalized_gamma(2.0, 3.0).unwrap();
    /// if let ACDInnovation::GeneralizedGamma { a, d, p } = innovation {
    ///     assert!(a > 0.0);
    ///     assert!((p - 2.0).abs() < 1e-12);
    ///     assert!((d - 3.0).abs() < 1e-12);
    /// }
    /// ```
    pub fn generalized_gamma(p: f64, d: f64) -> ACDResult<Self> {
        let p = validate_gamma_param(p)?;
        let d = validate_gamma_param(d)?;
        let a = (gamma::ln_gamma(d / p) - gamma::ln_gamma((d + 1.0) / p)).exp();
        Ok(ACDInnovation::GeneralizedGamma { a, d, p })
    }

    /// Construct a generalized gamma innovation law from user-provided `a, p, d`,
    /// validated and normalized to enforce unit mean.
    ///
    /// Parameters
    /// ----------
    /// - `a`: `f64`
    ///   Scale parameter candidate (finite, strictly positive).
    /// - `p`: `f64`
    ///   Power parameter (finite, strictly positive).
    /// - `d`: `f64`
    ///   Shape parameter (finite, strictly positive).
    ///
    /// Returns
    /// -------
    /// ACDResult<ACDInnovation>
    ///   - `Ok(ACDInnovation::GeneralizedGamma { a, d, p })` when parameters are
    ///     valid and the implied mean is within `UNIT_MEAN_TOL` of 1. The scale
    ///     `a` is renormalized to make the mean exactly 1.
    ///   - `Err(ACDError::InvalidGenGammaParam { .. })` if any parameter is invalid.
    ///   - `Err(ACDError::InvalidUnitMeanGenGamma { mean })` if the implied mean
    ///     deviates from 1 beyond `UNIT_MEAN_TOL`.
    ///
    /// Errors
    /// ------
    /// - `ACDError::InvalidGenGammaParam`
    ///   Returned when `a`, `p`, or `d` is non-finite or ≤ 0.
    /// - `ACDError::InvalidUnitMeanGenGamma`
    ///   Returned when the implied mean from `(a, p, d)` is not within tolerance.
    ///
    /// Panics
    /// ------
    /// - Never panics; all invalid inputs are surfaced as [`ACDError`] values.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; callers need only supply numerically reasonable
    ///   parameters.
    ///
    /// Notes
    /// -----
    /// - When the implied mean is within tolerance, `a` is divided by that mean
    ///   so that the stored parameters enforce unit mean exactly.
    /// - This constructor is useful for reusing external generalized gamma fits
    ///   under the ACD unit-mean constraint.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::innovation::ACDInnovation;
    /// let innovation = ACDInnovation::generalized_gamma_with_a(1.0, 2.0, 3.0);
    /// ```
    pub fn generalized_gamma_with_a(a: f64, p: f64, d: f64) -> ACDResult<Self> {
        let mut a = validate_gamma_param(a)?;
        let p = validate_gamma_param(p)?;
        let d = validate_gamma_param(d)?;
        let uncond_mean = (a.ln() + gamma::ln_gamma((d + 1.0) / p) - gamma::ln_gamma(d / p)).exp();
        if (uncond_mean - 1.0).abs() > UNIT_MEAN_TOL {
            return Err(ACDError::InvalidUnitMeanGenGamma { mean: uncond_mean });
        } else {
            a /= uncond_mean;
        }
        Ok(ACDInnovation::GeneralizedGamma { a, d, p })
    }

    /// Compute `log f_X(x | psi)` for an ACD duration under this innovation law.
    ///
    /// Parameters
    /// ----------
    /// - `x`: `f64`
    ///   Observed duration at a given time step. Must be finite and ≥ 0; the
    ///   domain is checked by `validate_loglik_params`.
    /// - `psi`: `f64`
    ///   Conditional mean duration `ψ_t` at the same time step, used as the
    ///   scaling parameter. Must be finite and strictly positive.
    ///
    /// Returns
    /// -------
    /// ACDResult<f64>
    ///   - `Ok(value)` containing the per-observation log-likelihood
    ///     contribution `log f_X(x | psi)`.
    ///   - `Err(ACDError)` if `x` or `psi` fails domain checks or if the
    ///     underlying `statrs` distribution rejects its parameters.
    ///
    /// Errors
    /// ------
    /// - `ACDError::InvalidLogLikInput`
    ///   Returned when `x` is non-finite or negative.
    /// - `ACDError::InvalidPsiLogLik`
    ///   Returned when `psi` is non-finite or non-positive.
    /// - Distribution-specific `ACDError` variants propagated from `statrs`
    ///   constructors if internal parameters are invalid.
    ///
    /// Panics
    /// ------
    /// - Never panics under normal operation; all invalid inputs are checked and
    ///   converted into [`ACDError`] values.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; callers must simply respect the input domains.
    ///
    /// Notes
    /// -----
    /// - Uses the identity `log f_X(x | psi) = log f_ε(x/psi) - log(psi)` to keep
    ///   computation numerically stable.
    /// - For exponential and Weibull innovations, the implementation delegates
    ///   to `statrs` distributions; for generalized gamma, a hand-coded log-pdf
    ///   formula is used with log-gamma for stability.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::innovation::ACDInnovation;
    /// let innovation = ACDInnovation::exponential();
    /// let loglik = innovation.log_pdf_duration(1.0, 1.0).unwrap();
    /// ```
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
                let log_lik = p.ln() - d * a.ln() + (d - 1.0) * (eps).ln()
                    - ((eps) / a).powf(*p)
                    - gamma::ln_gamma(d / p);
                Ok(log_lik - ln_psi)
            }
        }
    }

    /// Compute the gradient of `log f_X(x | psi)` with respect to `psi`.
    ///
    /// Parameters
    /// ----------
    /// - `x`: `f64`
    ///   Observed duration (finite, ≥ 0), checked via `validate_loglik_params`.
    /// - `psi`: `f64`
    ///   Conditional mean `ψ_t` at the same time step (finite, > 0).
    ///
    /// Returns
    /// -------
    /// ACDResult<f64>
    ///   - `Ok(value)` containing `∂/∂ψ log f_X(x | psi)` for the current
    ///     innovation law.
    ///   - `Err(ACDError)` if `x` or `psi` fails domain checks.
    ///
    /// Errors
    /// ------
    /// - `ACDError::InvalidLogLikInput` or `ACDError::InvalidPsiLogLik`
    ///   Propagated from `validate_loglik_params` when inputs are out of domain.
    ///
    /// Panics
    /// ------
    /// - Never panics; domain violations are reported as [`ACDError`] values.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; callers must only respect the input domains.
    ///
    /// Notes
    /// -----
    /// - For the supported innovations, closed-form expressions are used:
    ///   - Exponential(1): `(ε - 1) / psi`, where `ε = x / psi`.
    ///   - Weibull(k, lambda): `k * ((ε/lambda)^k - 1) / psi`.
    ///   - GeneralizedGamma(a, d, p): `p * ((ε/a)^p - d) / psi`.
    /// - This method is intended for use in 1D ψ-gradients when building full
    ///   score vectors or Hessians for ACD models.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::innovation::ACDInnovation;
    /// let innovation = ACDInnovation::exponential();
    /// let grad = innovation.one_d_loglik_grad(1.0, 1.0).unwrap();
    /// ```
    pub fn one_d_loglik_grad(&self, x: f64, psi: f64) -> ACDResult<f64> {
        validate_loglik_params(x, psi)?;
        let eps = x / psi;
        match self {
            ACDInnovation::Exponential => Ok(eps / psi - 1.0 / psi),
            ACDInnovation::Weibull { lambda, k } => {
                Ok((k * ((k * (eps.ln() - lambda.ln())).exp() - 1.0)) / psi)
            }
            ACDInnovation::GeneralizedGamma { a, d, p } => {
                Ok((p * (p * (eps.ln() - a.ln())).exp() - d) / psi)
            }
        }
    }

    /// Draw a single unit-mean innovation ε from this law using a caller-provided RNG.
    ///
    /// Parameters
    /// ----------
    /// - `rng`: `&mut StdRng`
    ///   Random number generator used as the entropy source. Passing the RNG
    ///   from the caller allows reproducible and controllable simulation flows.
    ///
    /// Returns
    /// -------
    /// ACDResult<f64>
    ///   - `Ok(value)` containing a simulated innovation `ε > 0` with unit mean
    ///     under the parameterization enforced by this [`ACDInnovation`] variant.
    ///   - `Err(ACDError)` if the underlying `statrs` distribution constructor
    ///     rejects the current parameters.
    ///
    /// Errors
    /// ------
    /// - Distribution-related `ACDError` variants propagated from `statrs`
    ///   (`InvalidExpParam`, `ScaleInvalid`, `ShapeInvalid`, or generalized gamma
    ///   parameter errors).
    ///
    /// Panics
    /// ------
    /// - Never panics under normal conditions; distribution parameter issues are
    ///   converted to [`ACDError`] first.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; callers must only ensure that the RNG is in a valid
    ///   state and that the innovation parameters were constructed successfully.
    ///
    /// Notes
    /// -----
    /// - For generalized gamma, the implementation draws `Z ~ Gamma(d/p, 1)`
    ///   and returns `ε = a * Z^{1/p}`, consistent with the parameterization
    ///   used in the unit-mean construction.
    /// - This function is meant for simulation and bootstrapping rather than
    ///   likelihood evaluation.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rand::SeedableRng;
    /// # use rand::rngs::StdRng;
    /// # use rust_timeseries::duration::core::innovation::ACDInnovation;
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let innovation = ACDInnovation::exponential();
    /// let eps = innovation.draw_innovation(&mut rng).unwrap();
    /// assert!(eps > 0.0);
    /// ```
    pub fn draw_innovation(&self, rng: &mut StdRng) -> ACDResult<f64> {
        match self {
            ACDInnovation::Exponential => {
                let exp = Exp::new(1.0)?;
                Ok(exp.sample(rng))
            }
            ACDInnovation::Weibull { lambda, k } => {
                let weibull = Weibull::new(*k, *lambda)?;
                Ok(weibull.sample(rng))
            }
            ACDInnovation::GeneralizedGamma { a, d, p } => {
                let gamma = Gamma::new(d / p, 1.0)?;
                let samp = gamma.sample(rng);
                Ok(a * samp.powf(1.0 / p))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duration::errors::ACDError;
    use approx::assert_relative_eq;
    use rand::{SeedableRng, rngs::StdRng};
    use statrs::distribution::{Continuous, Exp, Weibull};

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Construction and validation of ACDInnovation variants
    //   (exponential, Weibull, generalized gamma).
    // - Enforcement of the unit-mean constraint via constructors
    //   and *_with_lambda / *_with_a helpers.
    // - Per-observation log-likelihood evaluation for durations and
    //   its one-dimensional gradient with respect to psi.
    // - Basic simulation behavior of draw_innovation (positivity, mean ~ 1).
    //
    // They intentionally DO NOT cover:
    // - Full ACD recursion or parameter estimation behavior (those are tested
    //   at higher levels in the duration stack).
    // - Multi-observation log-likelihood aggregation or optimizer wiring.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Ensure that the exponential() constructor yields the Exponential variant.
    //
    // Given
    // -----
    // - No parameters, calling ACDInnovation::exponential().
    //
    // Expect
    // ------
    // - The returned innovation equals ACDInnovation::Exponential.
    fn acdinnovation_exponential_constructor_yields_exponential_variant() {
        // Arrange & Act
        let innovation = ACDInnovation::exponential();

        // Assert
        assert_eq!(innovation, ACDInnovation::Exponential);
    }

    #[test]
    // Purpose
    // -------
    // Verify that Weibull constructor rejects invalid shape parameters.
    //
    // Given
    // -----
    // - Shape k <= 0 or non-finite.
    //
    // Expect
    // ------
    // - ACDInnovation::weibull returns Err(ACDError::InvalidWeibullParam).
    fn weibull_constructor_with_invalid_shape_returns_error() {
        // Arrange
        let invalid_ks = [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

        // Act & Assert
        for k in invalid_ks {
            let result = ACDInnovation::weibull(k);
            assert!(matches!(result, Err(ACDError::InvalidWeibullParam { .. })));
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that Weibull constructor with valid shape produces unit-mean
    // innovations to within a small Monte Carlo tolerance.
    //
    // Given
    // -----
    // - A valid shape k > 0.
    // - A fixed RNG seed to simulate many draws from the resulting law.
    //
    // Expect
    // ------
    // - The sample mean of epsilon draws is approximately 1.0.
    fn weibull_constructor_with_valid_shape_has_unit_mean() {
        // Arrange
        let k = 1.5;
        let innovation = ACDInnovation::weibull(k).unwrap();
        let num_samples: usize = 20_000;
        let mut rng = StdRng::seed_from_u64(12345);

        // Act
        let mut sum_eps = 0.0;
        for _ in 0..num_samples {
            let eps = innovation.draw_innovation(&mut rng).unwrap();
            assert!(eps > 0.0);
            sum_eps += eps;
        }
        let mean_eps = sum_eps / num_samples as f64;

        // Assert
        assert_relative_eq!(mean_eps, 1.0, max_relative = 1e-2);
    }

    #[test]
    // Purpose
    // -------
    // Verify that generalized_gamma constructor with valid shapes
    // produces unit-mean innovations to within a small Monte Carlo tolerance.
    //
    // Given
    // -----
    // - Valid shape parameters p > 0 and d > 0.
    // - A fixed RNG seed to simulate many draws.
    //
    // Expect
    // ------
    // - The sample mean of epsilon draws is approximately 1.0.
    fn generalized_gamma_constructor_with_valid_shapes_has_unit_mean() {
        // Arrange
        let p = 2.0;
        let d = 3.0;
        let innovation = ACDInnovation::generalized_gamma(p, d).unwrap();
        let num_samples: usize = 20_000;
        let mut rng = StdRng::seed_from_u64(67890);

        // Act
        let mut sum_eps = 0.0;
        for _ in 0..num_samples {
            let eps = innovation.draw_innovation(&mut rng).unwrap();
            assert!(eps > 0.0);
            sum_eps += eps;
        }
        let mean_eps = sum_eps / num_samples as f64;

        // Assert
        assert_relative_eq!(mean_eps, 1.0, max_relative = 1e-2);
    }

    #[test]
    // Purpose
    // -------
    // Ensure that weibull_with_lambda rejects parameter combinations that
    // induce a mean far from 1.
    //
    // Given
    // -----
    // - A valid lambda and shape k such that lambda * Γ(1 + 1/k) is not ~ 1.
    //
    // Expect
    // ------
    // - ACDInnovation::weibull_with_lambda returns
    //   Err(ACDError::InvalidUnitMeanWeibull).
    fn weibull_with_lambda_out_of_tolerance_returns_invalid_unit_mean_error() {
        // Arrange
        let lambda = 1.0;
        let k = 1.5;

        // Act
        let result = ACDInnovation::weibull_with_lambda(lambda, k);

        // Assert
        assert!(matches!(result, Err(ACDError::InvalidUnitMeanWeibull { .. })));
    }

    #[test]
    // Purpose
    // -------
    // Ensure that generalized_gamma_with_a rejects parameter combinations that
    // induce a mean far from 1.
    //
    // Given
    // -----
    // - A valid (a, p, d) triple whose implied mean is not approximately 1.
    //
    // Expect
    // ------
    // - ACDInnovation::generalized_gamma_with_a returns
    //   Err(ACDError::InvalidUnitMeanGenGamma).
    fn generalized_gamma_with_a_out_of_tolerance_returns_invalid_unit_mean_error() {
        // Arrange
        let a = 1.0;
        let p = 2.0;
        let d = 3.0;

        // Act
        let result = ACDInnovation::generalized_gamma_with_a(a, p, d);

        // Assert
        assert!(matches!(result, Err(ACDError::InvalidUnitMeanGenGamma { .. })));
    }

    #[test]
    // Purpose
    // -------
    // Verify that log_pdf_duration for the exponential innovation matches
    // the statrs baseline log f_X(x | psi) = log f_ε(x / psi) - log psi.
    //
    // Given
    // -----
    // - x > 0, psi > 0, and Exponential(1) innovation.
    //
    // Expect
    // ------
    // - The implementation matches the statrs formula to within tight tolerance.
    fn log_pdf_duration_exponential_matches_statrs_baseline() {
        // Arrange
        let innovation = ACDInnovation::exponential();
        let x = 1.3;
        let psi: f64 = 0.7;
        let eps = x / psi;
        let ln_psi = psi.ln();
        let baseline = Exp::new(1.0).unwrap().ln_pdf(eps) - ln_psi;

        // Act
        let result = innovation.log_pdf_duration(x, psi).unwrap();

        // Assert
        assert_relative_eq!(result, baseline, epsilon = 1e-12);
    }

    #[test]
    // Purpose
    // -------
    // Verify that log_pdf_duration for Weibull innovations matches the
    // corresponding statrs Weibull baseline.
    //
    // Given
    // -----
    // - x > 0, psi > 0, and a Weibull innovation constructed via weibull(k).
    //
    // Expect
    // ------
    // - The implementation matches statrs::Weibull ln_pdf(x/psi) - ln psi
    //   to within tight tolerance.
    fn log_pdf_duration_weibull_matches_statrs_baseline() {
        // Arrange
        let k = 1.7;
        let innovation = ACDInnovation::weibull(k).unwrap();
        let x = 0.9;
        let psi: f64 = 1.3;
        let eps = x / psi;
        let ln_psi = psi.ln();

        let lambda = match innovation {
            ACDInnovation::Weibull { lambda, k: k_inner } => {
                assert_relative_eq!(k_inner, k, epsilon = 1e-12);
                lambda
            }
            _ => panic!("expected Weibull innovation"),
        };

        let baseline = Weibull::new(k, lambda).unwrap().ln_pdf(eps) - ln_psi;

        // Act
        let result = innovation.log_pdf_duration(x, psi).unwrap();

        // Assert
        assert_relative_eq!(result, baseline, epsilon = 1e-12);
    }

    #[test]
    // Purpose
    // -------
    // Ensure that log_pdf_duration rejects invalid inputs (negative x or
    // non-positive psi) with the appropriate ACDError variants.
    //
    // Given
    // -----
    // - x < 0 with psi > 0.
    // - x >= 0 with psi <= 0.
    //
    // Expect
    // ------
    // - Errors ACDError::InvalidLogLikInput and ACDError::InvalidPsiLogLik
    //   respectively.
    fn log_pdf_duration_with_invalid_inputs_returns_validation_errors() {
        // Arrange
        let innovation = ACDInnovation::exponential();

        // Act
        let result_x_neg = innovation.log_pdf_duration(-0.1, 1.0);
        let result_psi_non_pos = innovation.log_pdf_duration(0.1, 0.0);

        // Assert
        assert!(matches!(result_x_neg, Err(ACDError::InvalidLogLikInput { .. })));
        assert!(matches!(result_psi_non_pos, Err(ACDError::InvalidPsiLogLik { .. })));
    }

    #[test]
    // Purpose
    // -------
    // Verify that one_d_loglik_grad matches a central finite-difference
    // approximation of log_pdf_duration with respect to psi for all
    // supported innovation families.
    //
    // Given
    // -----
    // - A set of innovation variants (Exponential, Weibull, GeneralizedGamma).
    // - Fixed x > 0, psi > 0, and a small finite-difference step h.
    //
    // Expect
    // ------
    // - The analytical gradient matches the finite-difference estimate
    //   within reasonable numerical tolerance.
    fn one_d_loglik_grad_matches_finite_difference_for_all_variants() {
        // Arrange
        let innovations = [
            ACDInnovation::exponential(),
            ACDInnovation::weibull(1.5).unwrap(),
            ACDInnovation::generalized_gamma(2.0, 3.0).unwrap(),
        ];
        let x = 1.2;
        let psi = 0.9;
        let h = 1e-6;

        for innovation in innovations.iter() {
            // Act
            let grad = innovation.one_d_loglik_grad(x, psi).unwrap();
            let loglik_plus = innovation.log_pdf_duration(x, psi + h).unwrap();
            let loglik_minus = innovation.log_pdf_duration(x, psi - h).unwrap();
            let fd_grad = (loglik_plus - loglik_minus) / (2.0 * h);

            // Assert
            assert_relative_eq!(grad, fd_grad, epsilon = 1e-5);
        }
    }

    #[test]
    // Purpose
    // -------
    // Smoke-test draw_innovation for positivity and unit-mean behavior
    // across all supported innovation families.
    //
    // Given
    // -----
    // - Exponential, Weibull, and GeneralizedGamma innovations constructed
    //   via the unit-mean constructors.
    // - A fixed RNG seed and a moderate number of samples.
    //
    // Expect
    // ------
    // - All draws are strictly positive.
    // - The sample mean is close to 1.0 for each variant.
    fn draw_innovation_produces_positive_unit_mean_samples() {
        // Arrange
        let innovations = [
            ACDInnovation::exponential(),
            ACDInnovation::weibull(1.3).unwrap(),
            ACDInnovation::generalized_gamma(2.5, 4.0).unwrap(),
        ];
        let num_samples: usize = 10_000;
        let base_seed = 4242_u64;

        for (idx, innovation) in innovations.iter().enumerate() {
            let mut rng = StdRng::seed_from_u64(base_seed + idx as u64);
            let mut sum_eps = 0.0;

            // Act
            for _ in 0..num_samples {
                let eps = innovation.draw_innovation(&mut rng).unwrap();
                assert!(eps > 0.0);
                sum_eps += eps;
            }
            let mean_eps = sum_eps / num_samples as f64;

            // Assert
            assert_relative_eq!(mean_eps, 1.0, max_relative = 2e-2);
        }
    }
}
