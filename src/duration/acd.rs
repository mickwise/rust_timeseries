use crate::duration::duration_errors::{ACDError, ACDResult};
use ndarray::Array1;
use statrs::function::gamma;

// Constants
const UNIT_MEAN_ATOL: f64 = 1e-10;

/// Units of measurement for durations in an ACD model.
///
/// These specify the time granularity assumed by the data and used in
/// likelihood evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ACDUnit {
    Microseconds,
    Milliseconds,
    Seconds,
}

/// Policy for handling zeros that arise during **internal computations**
/// (not in raw input).
///
/// - `ClipToEpsilon`: replace internal 0 with `epsilon_floor`.
/// - `Error`: fail immediately.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZeroPolicy {
    ClipToEpsilon(f64),
    Error,
}

impl ZeroPolicy {
    // Return an error if encountering a zero during computations.
    pub const fn error() -> Self {
        ZeroPolicy::Error
    }

    /// Create a `ZeroPolicy` that replaces internal zeros with a positive floor.
    ///
    /// The floor `epsilon_floor` must be **finite and strictly positive**.
    /// This does not affect raw input values; it only guards intermediate
    /// computations (e.g., avoiding `log(0)` or division by zero).
    ///
    /// # Errors
    /// Returns [`ACDError::InvalidEpsilonFloor`] if `epsilon_floor` is not finite or nonpositive.
    pub fn clip_to_epsilon(epsilon_floor: f64) -> ACDResult<Self> {
        if !epsilon_floor.is_finite() || epsilon_floor <= 0.0 {
            return Err(ACDError::InvalidEpsilonFloor {
                value: epsilon_floor,
            });
        }
        Ok(ZeroPolicy::ClipToEpsilon(epsilon_floor))
    }
}

/// Initialization policy for the ACD ψ recursion.
///
/// Controls how the initial state (and/or initial guess) is set before
/// maximizing the likelihood. The only variant that accepts a parameter is
/// [`Init::Fixed`]; it must be **finite and strictly positive**.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Init {
    /// Initialize ψ at the unconditional mean implied by the model.
    UncondMean,
    /// Initialize ψ at the sample mean of the data.
    SampleMean,
    /// Initialize ψ at a fixed value.
    Fixed(f64),
}

impl Init {
    /// Use the unconditional-mean initialization for ψ.
    ///
    /// This is typically a safe, fast default that avoids dependence on the
    /// sample path in the first few iterations.
    pub const fn uncond_mean() -> Self {
        Init::UncondMean
    }

    /// Use the sample-mean initialization for ψ.
    ///
    /// Useful when you believe the estimation window is representative and want
    /// ψ₀ anchored to the data rather than the model-implied mean.
    pub const fn sample_mean() -> Self {
        Init::SampleMean
    }

    /// Use a fixed value to initialize ψ.
    ///
    /// The value must be **finite and strictly positive**.
    ///
    /// # Errors
    /// Returns [`ACDError::InvalidInitFixed`] if `value` is not finite or `value <= 0.0`.
    pub fn fixed(value: f64) -> ACDResult<Self> {
        if !value.is_finite() || value <= 0.0 {
            return Err(ACDError::InvalidInitFixed { value });
        }
        Ok(Init::Fixed(value))
    }
}

/// Innovation (error) distributions for the ACD model.
///
/// All variants are parameterized so that the **unit-mean** constraint
/// `E[ε_t] = 1` holds. Methods on `ACDInnovation` either:
/// - compute the missing scale from provided shape(s) to enforce unit mean, or
/// - validate user-provided parameters against the unit-mean constraint.
///
/// **Conventions**
/// - `Exponential` is the special case with mean 1 by definition.
/// - `Weibull{ lambda, k }`: `lambda` is the scale, `k` the shape (> 0).
/// - `GeneralizedGamma{ a, d, p }`: `a` is the scale (> 0); `d, p` are shapes (> 0).
/// - These follow Wikipedia conventions for Weibull and generalized gamma.
///
/// **Numerics**
/// Constructors use `ln_gamma` differences for stability. Unit-mean
/// checks use a small absolute tolerance (currently `1e-10`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ACDInnovation {
    Exponential,

    Weibull {
        lambda: f64, // scale parameter
        k: f64,
    },

    GeneralizedGamma {
        a: f64, // scale parameter
        d: f64, // shape parameter
        p: f64,
    },
}

impl ACDInnovation {
    /// Exponential(1) innovations (unit mean).
    ///
    /// Returns the exponential distribution with rate = 1 (mean = 1),
    /// requiring no parameters and no validation.
    /// This is the canonical EACD innovation choice.
    pub const fn exponential() -> Self {
        ACDInnovation::Exponential
    }

    /// Weibull innovations with **unit mean**.
    ///
    /// Computes the scale `lambda` from the provided shape `k > 0` so that
    /// `E[ε] = lambda * Γ(1 + 1/k) = 1`, i.e.,
    /// `lambda = 1 / Γ(1 + 1/k)`.
    ///
    /// # Errors
    /// Returns `InvalidWeibullParam` if `k` is not finite or `k <= 0.0`.
    ///
    /// # Notes
    /// Uses `ln_gamma` internally for numerical stability.
    pub fn weibull(k: f64) -> ACDResult<Self> {
        let k = verify_weibull_param(k)?;
        let lambda = (-gamma::ln_gamma(1.0 + 1.0 / k)).exp();
        Ok(ACDInnovation::Weibull { lambda, k })
    }

    /// Weibull innovations with user-provided `lambda` and `k`, **validated** for unit mean.
    ///
    /// Accepts both scale `lambda > 0` and shape `k > 0`, then checks the
    /// unit-mean constraint `lambda * Γ(1 + 1/k) ≈ 1` with a small absolute
    /// tolerance (`1e-10`).
    ///
    /// # Errors
    /// - `InvalidWeibullParam` if either parameter is not finite or non-positive.
    /// - `InvalidUnitMeanWeibull { mean }` if the resulting mean deviates from 1
    ///   by more than the tolerance (the computed `mean` is included).
    ///
    /// # Notes
    /// Uses `ln_gamma` internally for numerical stability.
    /// Normalizes `lambda` so that the unconditional mean is 1 if it is within the tolerance.
    pub fn weibull_with_lambda(lambda: f64, k: f64) -> ACDResult<Self> {
        let k = verify_weibull_param(k)?;
        let mut lambda = verify_weibull_param(lambda)?;
        let uncond_mean = lambda * gamma::ln_gamma(1.0 + 1.0 / k).exp();
        if (uncond_mean - 1.0).abs() > UNIT_MEAN_ATOL {
            return Err(ACDError::InvalidUnitMeanWeibull { mean: uncond_mean });
        } else {
            lambda = lambda / uncond_mean;
        }
        Ok(ACDInnovation::Weibull { lambda, k })
    }

    /// Generalized-Gamma innovations with **unit mean**, given shapes `p` and `d`.
    ///
    /// Computes the scale `a` from the provided shape parameters `p > 0`, `d > 0`
    /// so that `E[ε] = a * Γ(d/p) / Γ((d+1)/p) = 1`, i.e.,
    /// `a = Γ(d/p) / Γ((d + 1)/p)`.
    ///
    /// # Errors
    /// Returns `InvalidGenGammaParam` if either shape is not finite or non-positive.
    ///
    /// # Notes
    /// Uses `exp(lnΓ(d/p) - lnΓ((d+1)/p))` for numerical stability.
    pub fn generalized_gamma(p: f64, d: f64) -> ACDResult<Self> {
        let p = verify_gamma_param(p)?;
        let d = verify_gamma_param(d)?;
        let a = (gamma::ln_gamma(d / p) - gamma::ln_gamma((d + 1.0) / p)).exp();
        Ok(ACDInnovation::GeneralizedGamma { a, d, p })
    }

    /// Generalized-Gamma innovations with user-provided `a, p, d`, **validated** for unit mean.
    ///
    /// Accepts scale `a > 0` and shapes `p > 0`, `d > 0`, then checks the
    /// unit-mean constraint `a * Γ(d/p) / Γ((d+1)/p) ≈ 1` with a small absolute
    /// tolerance (`1e-10`).
    ///
    /// # Errors
    /// - `InvalidGenGammaParam` if any parameter is not finite or non-positive.
    /// - `InvalidUnitMeanGenGamma { mean }` if the resulting mean deviates from 1
    ///   by more than the tolerance (the computed `mean` is included).
    ///
    /// # Notes
    /// Uses `exp(ln(a) + lnΓ(d/p) - lnΓ((d+1)/p))` for numerical stability.
    /// Normalizes `a` so that the unconditional mean is 1 if it is within the tolerance.
    pub fn generalized_gamma_with_a(a: f64, p: f64, d: f64) -> ACDResult<Self> {
        let mut a = verify_gamma_param(a)?;
        let p = verify_gamma_param(p)?;
        let d = verify_gamma_param(d)?;
        let uncond_mean = (a.ln() + gamma::ln_gamma((d + 1.0) / p) - gamma::ln_gamma(d / p)).exp();
        if (uncond_mean - 1.0).abs() > UNIT_MEAN_ATOL {
            return Err(ACDError::InvalidUnitMeanGenGamma { mean: uncond_mean });
        } else {
            a = a / uncond_mean;
        }
        Ok(ACDInnovation::GeneralizedGamma { a, d, p })
    }
}

/// Order of the ACD(p, q) model.
///
/// - `p`: number of lagged durations (α terms)
/// - `q`: number of lagged conditional means ψ (β terms)
///
/// At least one of `p` or `q` must be > 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ACDShape {
    pub p: usize,
    pub q: usize,
}

impl ACDShape {
    /// Construct an [`ACDShape`] = ACD(p, q).
    ///
    /// # Errors
    /// Returns [`ACDError::InvalidModelShape`] if `p == 0 && q == 0`
    /// (a degenerate model with no dynamics).
    pub fn new(p: usize, q: usize) -> ACDResult<Self> {
        if q == 0 && p == 0 {
            return Err(ACDError::InvalidModelShape { param: p as f64 });
        }
        Ok(ACDShape { p, q })
    }
}

/// Bounds for the ψ recursion during estimation.
///
/// These guards enforce a valid operating range for ψ to prevent divergence
/// during likelihood maximization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsiGuards {
    pub min: f64,
    pub max: f64,
}

impl PsiGuards {
    /// Construct new ψ bounds.
    ///
    /// # Errors
    /// Returns [`ACDError::InvalidPsiGuards`] if:
    /// - `min >= max`
    /// - either bound is not finite
    /// - `min <= 0.0`
    pub fn new(value: (f64, f64)) -> ACDResult<Self> {
        if value.0 >= value.1 {
            return Err(ACDError::InvalidPsiGuards {
                min: value.0,
                max: value.1,
                reason: "Psi guards must have min < max.",
            });
        }

        if !value.0.is_finite() || !value.1.is_finite() {
            return Err(ACDError::InvalidPsiGuards {
                min: value.0,
                max: value.1,
                reason: "Psi guards must be finite.",
            });
        }

        if value.0 <= 0.0 {
            return Err(ACDError::InvalidPsiGuards {
                min: value.0,
                max: value.1,
                reason: "Psi guards must be strictly positive.",
            });
        }

        Ok(PsiGuards {
            min: value.0,
            max: value.1,
        })
    }
}

/// Container for input duration data and associated metadata.
///
/// This ensures the data is strictly positive, finite, and internally
/// consistent with the provided metadata and origin time (`t0`).
#[derive(Debug, Clone, PartialEq)]
pub struct ACDData {
    pub data: Array1<f64>,
    pub t0: Option<usize>,
    pub meta: ACDMeta,
}

impl ACDData {
    /// Construct a validated [`ACDData`] instance.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The series is empty.
    /// - Any element is non-finite or nonpositive.
    /// - `t0` is specified but out of range.
    /// - Metadata validation fails.
    pub fn new(data: Array1<f64>, t0: Option<usize>, meta: ACDMeta) -> ACDResult<Self> {
        if data.is_empty() {
            return Err(ACDError::EmptySeries);
        }

        for (index, &value) in data.iter().enumerate() {
            if !value.is_finite() {
                return Err(ACDError::NonFiniteData { index, value });
            }
            if value <= 0.0 {
                return Err(ACDError::NonPositiveData { index, value });
            }
        }

        if let Some(t0_val) = t0 {
            if t0_val >= data.len() {
                return Err(ACDError::T0OutOfRange {
                    t0: t0_val,
                    len: data.len(),
                });
            }
        }

        Ok(ACDData { data, t0, meta })
    }
}

/// Metadata describing how the duration data should be interpreted.
///
/// Includes the time unit, zero-handling policy, scaling, and epsilon floor for
/// numerical stability.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDMeta {
    pub unit: ACDUnit,
    pub zero_policy: ZeroPolicy,

    /// Optional internal scaling factor used for numerical stability; not
    /// user-specified and does not alter raw input values.
    pub scale: Option<f64>,
    pub diurnal_adjusted: bool,
}

impl ACDMeta {
    /// Construct a new [`ACDMeta`] instance.
    ///
    /// Does not currently validate options beyond their constructors; assumes
    /// provided arguments are already consistent.
    pub fn new(
        unit: ACDUnit,
        zero_policy: ZeroPolicy,
        scale: Option<f64>,
        diurnal_adjusted: bool,
    ) -> ACDMeta {
        ACDMeta {
            unit,
            zero_policy,
            scale,
            diurnal_adjusted,
        }
    }
}

/// Configuration options for estimating an ACD model.
///
/// Bundles initialization, numerical guards, randomization, and output
/// preferences into a single struct.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDOptions {
    pub init: Init,
    // pub optimizer: MleOptions,
    pub psi_guards: PsiGuards,
    pub random_seed: Option<u64>,
    pub return_norm_resid: bool,
    pub compute_hessian: bool,
}

impl ACDOptions {
    /// Construct a new [`ACDOptions`] instance.
    ///
    /// Does not currently validate options beyond their constructors; assumes
    /// provided arguments are already consistent.
    pub fn new(
        init: Init,
        psi_guards: PsiGuards,
        random_seed: Option<u64>,
        return_norm_resid: bool,
        compute_hessian: bool,
    ) -> ACDOptions {
        ACDOptions {
            init,
            // optimizer: MleOptions::default(),
            psi_guards,
            random_seed,
            return_norm_resid,
            compute_hessian,
        }
    }
}

/// Complete specification of an ACD model *excluding data*.
///
/// This bundles:
/// - [`ACDShape`]: the (p, q) order of the ACD model,
/// - [`ACDInnovation`]: the innovation distribution (parameterized to unit mean),
/// - [`ACDOptions`]: estimation and numerical options.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDModelSpec {
    /// Model order: ACD(p, q).
    pub shape: ACDShape,
    /// Innovation (error) distribution with unit-mean parameterization.
    pub innovation: ACDInnovation,
    /// Estimation and numerical options (initialization, guards, etc.).
    pub options: ACDOptions,
}

impl ACDModelSpec {
    /// Construct an [`ACDModelSpec`] from its components.
    ///
    /// This function assumes each component has already been validated by its
    /// own constructor (e.g., [`ACDShape::new`], `ACDInnovation::*`, [`ACDOptions::new`]).
    /// No additional checks are performed here.
    pub fn new(shape: ACDShape, innovation: ACDInnovation, options: ACDOptions) -> ACDModelSpec {
        ACDModelSpec {
            shape,
            innovation,
            options,
        }
    }
}

/// ---- Helper methods ----

/// Validate a single Weibull parameter (scale or shape).
///
/// Ensures the value is finite and strictly positive.
///
/// # Errors
/// Returns `InvalidWeibullParam` with a descriptive `reason` if the check fails.
fn verify_weibull_param(param: f64) -> ACDResult<f64> {
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

/// Validate a single Generalized-Gamma parameter (scale or shape).
///
/// Ensures the value is finite and strictly positive.
///
/// # Errors
/// Returns `InvalidGenGammaParam` with a descriptive `reason` if the check fails.
fn verify_gamma_param(param: f64) -> ACDResult<f64> {
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
