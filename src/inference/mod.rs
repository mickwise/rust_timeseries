//! # Inference module (SEs, HAC, and sandwich estimators)
//!
//! Tools for post-estimation uncertainty on top of a fitted model. This crate
//! provides classical (observed-information) standard errors and robust
//! (sandwich) corrections that account for serial dependence via HAC kernels.
//!
//! ## What this module provides
//! - **HAC score covariance**: [`calculate_avg_scores`] builds the p×p score
//!   covariance using IID/OPG or HAC with Bartlett/Parzen/QS kernels and a
//!   (possibly plug-in) bandwidth.
//! - **Kernels & bandwidths**: [`KernelType`] window weights and
//!   data-driven (Andrews-style) bandwidth selection.
//! - **Hessian-based SEs**: [`calc_standard_errors`] to compute classical
//!   or robust (sandwich) standard errors using the observed information.
//! - **Errors**: a unified [`InferenceError`] for stationarity/degeneracy guards,
//!   plus the [`InferenceResult`] alias.
//!
//! ## Usage sketch
//! 1. Fit your model to obtain θ̂ and (optionally) per-observation score rows.
//! 2. If robust SEs are desired, pick a kernel/bandwidth via [`HACOptions`]
//!    and build the score covariance with [`calculate_avg_scores`].
//! 3. Call [`calc_standard_errors`] with (i) θ̂ and (ii) `Some(score_cov)` for
//!    robust SEs, or `None` for classical SEs.
//!
//! ## Defaults & guards
//! - Default HAC kernel: Bartlett; bandwidth uses a safe rule if plug-in fails.
//!  Small-sample correction follows conventional `n` vs `n−k` scaling.
//! - Bandwidth plug-in uses AR(1) fits per column and rejects near-nonstationary
//!   estimates; it falls back to a numerical default when denominators are tiny.
//!
//! ## Re-exports
//! This module re-exports the most commonly used types and functions so
//! downstream code can `use inference::*;` and get a compact surface area.

pub mod errors;
pub mod hac;
pub mod hessian;
pub mod kernel;

// ---- Re-exports (primary surface) ----
pub use self::errors::{InferenceError, InferenceResult};
pub use self::hac::{HACOptions, calculate_avg_scores_cov};
pub use self::hessian::calc_covariance;
pub use self::kernel::KernelType;

// Optional convenience prelude for downstream crates:
// use `inference::prelude::*;`
pub mod prelude {
    pub use super::errors::{InferenceError, InferenceResult};
    pub use super::hac::{HACOptions, calculate_avg_scores_cov};
    pub use super::hessian::calc_covariance;
    pub use super::kernel::KernelType;
}
