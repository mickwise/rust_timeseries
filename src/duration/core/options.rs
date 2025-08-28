//! Estimation options for ACD models.
//!
//! This module defines [`ACDOptions`], a single struct that bundles together
//! all configuration needed to estimate an ACD(p, q) model: initialization
//! policy, model shape, innovation distribution, optimizer options,
//! numerical guards for the ψ recursion, and a few convenience flags.
//!
//! Design goals:
//! - Keep all estimation knobs in one place.
//! - Avoid hidden defaults in the core (validation lives in the individual
//!   option constructors like `Init::…`, `ACDShape::new`, `MLEOptions::new`, etc.).
//!
//! Notes:
//! - `psi_guards` protect the ψ recursion from numerical pathologies (e.g. log(0)).
//! - `mle_opts` are passed through to the L-BFGS/line-search backend.
use crate::{
    duration::core::{guards::PsiGuards, init::Init, innovations::ACDInnovation, shape::ACDShape},
    optimization::loglik_optimizer::MLEOptions,
};

/// Configuration options for estimating an ACD model.
///
/// Bundles initialization, model shape, innovation distribution, optimizer
/// options, ψ-recursion guards, and a few convenience flags.
///
/// Fields:
/// - `init`: how to seed pre-sample ψ and duration lags (see [`Init`]).
/// - `shape`: the ACD(p, q) order (see [`ACDShape`]).
/// - `innovation`: innovation (error) distribution with unit-mean enforcement
///   (see [`ACDInnovation`]).
/// - `mle_opts`: optimizer settings (tolerances, line search, verbosity).
/// - `psi_guards`: lower/upper bounds for ψ to ensure numerical stability.
/// - `random_seed`: optional seed for any stochastic procedures (if used).
/// - `return_norm_resid`: whether to request normalized residuals downstream.
/// - `compute_hessian`: whether to request a Hessian/observed information downstream.
///
/// This type does not perform cross-field validation beyond what each component
/// constructor enforces; supply already-validated parts.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDOptions {
    /// Initialization policy for pre-sample ψ and duration lags.
    pub init: Init,
    /// ACD(p, q) model order.
    pub shape: ACDShape,
    /// Innovation distribution with unit-mean parametrization.
    pub innovation: ACDInnovation,
    /// Maximum-likelihood optimizer options (L-BFGS + line search).
    pub mle_opts: MLEOptions,
    /// Bounds for ψ during recursion to prevent divergence.
    pub psi_guards: PsiGuards,
    /// Request normalized residuals in outputs (if supported).
    pub return_norm_resid: bool,
    /// Request Hessian/observed information (if supported).
    pub compute_hessian: bool,
}

impl ACDOptions {
    /// Construct a new [`ACDOptions`] instance.
    ///
    /// This constructor assumes its arguments were already validated by their
    /// respective builders (e.g., [`Init`], [`ACDShape`], [`ACDInnovation`],
    /// [`MLEOptions`], and [`PsiGuards`]). No additional cross-field checks
    /// are performed here.
    pub fn new(
        init: Init, shape: ACDShape, innovation: ACDInnovation, mle_opts: MLEOptions,
        psi_guards: PsiGuards, return_norm_resid: bool, compute_hessian: bool,
    ) -> ACDOptions {
        ACDOptions {
            init,
            shape,
            innovation,
            mle_opts,
            psi_guards,
            return_norm_resid,
            compute_hessian,
        }
    }
}
