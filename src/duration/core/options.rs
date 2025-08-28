//! Estimation options for ACD models.
//!
//! This module defines [`ACDOptions`], a single struct that bundles the
//! configuration used during estimation of an ACD(p, q) model: the
//! initialization policy, optimizer options, and numerical guards for the
//! ψ-recursion, plus a couple of convenience flags.
//!
//! Design goals:
//! - Keep all estimation knobs in one place.
//! - Avoid hidden defaults in the core (validation lives in the individual
//!   constructors like `Init::…`, `MLEOptions::new`, `PsiGuards::new`, etc.).
//!
//! Notes:
//! - `psi_guards` protect the ψ-recursion from numerical pathologies (e.g., log(0)).
//! - `mle_opts` are passed through to the L-BFGS/line-search backend.
use crate::{
    duration::core::{guards::PsiGuards, init::Init},
    optimization::loglik_optimizer::MLEOptions,
};

/// Configuration options for estimating an ACD model.
///
/// Bundles the initialization policy, optimizer options, and ψ-recursion
/// guards, plus convenience flags that control what outputs are produced.
///
/// Fields:
/// - `init`: how to seed pre-sample ψ and duration lags (see [`Init`]).
/// - `mle_opts`: optimizer settings (tolerances, line-search choice, verbosity).
/// - `psi_guards`: lower/upper bounds for ψ to ensure numerical stability.
/// - `return_norm_resid`: whether to request normalized residuals downstream.
/// - `compute_hessian`: whether to request a Hessian/observed information downstream.
///
/// This type does not perform cross-field validation beyond what each component
/// enforces; supply already-validated parts.

#[derive(Debug, Clone, PartialEq)]
pub struct ACDOptions {
    /// Initialization policy for pre-sample ψ and duration lags.
    pub init: Init,
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
    /// Assumes its arguments were already validated by their respective builders
    /// (e.g., [`Init`], [`MLEOptions`], [`PsiGuards`]). No additional cross-field
    /// checks are performed here.
    pub fn new(
        init: Init, mle_opts: MLEOptions, psi_guards: PsiGuards, return_norm_resid: bool,
        compute_hessian: bool,
    ) -> ACDOptions {
        ACDOptions { init, mle_opts, psi_guards, return_norm_resid, compute_hessian }
    }
}
