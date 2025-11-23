//! inference — standard errors, HAC, and sandwich covariance for fitted models.
//!
//! Purpose
//! -------
//! Provide tools for post-estimation uncertainty quantification on top of a
//! fitted model. This module computes classical (observed-information) standard
//! errors and robust (sandwich / HAC) covariance estimators from
//! per-observation scores, all expressed in the unconstrained optimizer
//! parameter space `θ`.
//!
//! Key behaviors
//! -------------
//! - Define a unified error and result type, [`InferenceError`] and
//!   [`InferenceResult`], for inference-specific failures (bandwidth,
//!   kernel, stationarity, and numerical issues).
//! - Configure HAC behavior via [`HACOptions`], including kernel choice,
//!   bandwidth regime (fixed vs plug-in), centering policy, and
//!   small-sample corrections.
//! - Enumerate HAC kernel families with [`KernelType`] and expose
//!   Andrews-style plug-in bandwidth rules via
//!   [`KernelType::optimal_bandwidth`].
//! - Build IID / HAC covariance matrices of average per-observation scores
//!   via [`calculate_avg_scores_cov`].
//! - Convert score-level covariance into parameter-space covariance for
//!   `θ` using [`calc_covariance`], supporting both classical and robust
//!   sandwich estimators.
//!
//! Invariants & assumptions
//! ------------------------
//! - Per-observation scores are supplied on the **average log-likelihood
//!   scale** (i.e., scores of `ℓ̄(θ) = (1/n) Σ_i ℓ_i(θ)`), with shape
//!   `n × p` where `p` is the number of free parameters.
//! - HAC plug-in bandwidth selection enforces a strict stationarity margin
//!   on AR(1) coefficients and falls back to conservative defaults when
//!   plug-in formulas are unstable or ill-conditioned.
//! - Covariance matrices are treated as `p × p` symmetric objects; helper
//!   routines assume inputs and outputs match the model’s parameter
//!   dimension.
//! - All numerical routines return [`InferenceError`] on failure rather
//!   than panicking; callers are expected to handle these errors explicitly.
//!
//! Conventions
//! -----------
//! - Parameters `θ` live in **unconstrained optimizer space**. Any mapping
//!   from constrained model parameters to `θ` is handled upstream in the
//!   estimation code.
//! - Score arrays and covariance matrices use row-major indexing with rows
//!   corresponding to observations and columns to parameters.
//! - HAC bandwidths are expressed on the same time index as the original
//!   series (lag units); kernel implementations follow standard
//!   econometric definitions (e.g., Bartlett, Parzen).
//! - All functions are pure with respect to I/O: no logging, no global
//!   state, and no `unsafe` code paths. Failures are reported via
//!   [`InferenceResult`] only.
//!
//! Downstream usage
//! ----------------
//! - After fitting a model and obtaining an estimate `θ̂` in unconstrained
//!   space, callers:
//!   - construct an `n × p` matrix of per-observation scores (for robust
//!     SEs), and
//!   - choose an [`HACOptions`] configuration and optional [`KernelType`].
//! - Use [`calculate_avg_scores_cov`] to obtain a score covariance matrix
//!   `S` on the average-score scale (IID or HAC).
//! - Call [`calc_covariance`] with:
//!   - a gradient callback at `θ̂`,
//!   - `θ̂` itself, and
//!   - `Some(S)` for robust covariance or `None` for classical
//!     observed-information covariance.
//! - Downstream code typically re-exports this module as:
//!   `use rust_timeseries::inference::*;` to surface the main inference
//!   types and routines as a compact, curated API.
//!
//! Testing notes
//! -------------
//! - Unit tests in this module cover:
//!   - [`HACOptions`] construction and default behavior,
//!   - [`KernelType`] bandwidth rules and kernel-specific edge cases,
//!   - score covariance construction via [`calculate_avg_scores_cov`],
//!   - covariance mapping into parameter space via [`calc_covariance`], and
//!   - error-path behavior for bandwidth, kernel, and stationarity failures.
//! - Integration tests elsewhere exercise the full pipeline on toy models,
//!   from simulated scores through HAC covariance to final standard errors,
//!   ensuring that IID vs HAC settings and small-sample corrections behave
//!   as expected in end-to-end workflows.

pub mod errors;
pub mod hac;
pub mod hessian;
pub mod kernel;

// ---- Re-exports (primary surface) -----------------------------------------

pub use self::errors::{InferenceError, InferenceResult};
pub use self::hac::{HACOptions, calculate_avg_scores_cov};
pub use self::hessian::calc_covariance;
pub use self::kernel::KernelType;

// ---- Optional convenience prelude for downstream crates ------------------
//
// Downstream crates can `use rust_timeseries::inference::prelude::*;` to
// import the primary inference surface in a single line.

pub mod prelude {
    pub use super::errors::{InferenceError, InferenceResult};
    pub use super::hac::{HACOptions, calculate_avg_scores_cov};
    pub use super::hessian::calc_covariance;
    pub use super::kernel::KernelType;
}
