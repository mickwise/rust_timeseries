//! loglik_optimizer — MLE-friendly, argmin-powered log-likelihood optimizer.
//!
//! Purpose
//! -------
//! Provide a high-level, Argmin-backed optimization layer for **maximizing
//! log-likelihoods** `ℓ(θ)` from Rust or Python. Callers implement a single
//! trait, [`LogLikelihood`], and invoke [`maximize`] to run L-BFGS with a
//! configurable line search, tolerances, and finite-difference fallbacks.
//!
//! Key behaviors
//! -------------
//! - Convert user-supplied log-likelihoods `ℓ(θ)` into Argmin-compatible
//!   cost functions `c(θ) = -ℓ(θ)` via [`adapter::ArgMinAdapter`].
//! - Expose a single, user-facing entrypoint [`maximize`] that:
//!   - validates the initial guess with [`LogLikelihood::check`],
//!   - selects an L-BFGS solver via [`builders`] based on [`traits::LineSearcher`],
//!   - executes the solver via [`run::run_lbfgs`], and
//!   - normalizes results into an [`OptimOutcome`].
//! - Provide robust finite-difference helpers in [`finite_diff`] for
//!   gradients and Hessians when analytic derivatives are missing, with
//!   post-hoc validation and error capture.
//! - Centralize optimizer configuration ([`Tolerances`], [`MLEOptions`]) and
//!   validation logic ([`validation`]) so downstream code can assume sane,
//!   finite inputs.
//!
//! Invariants & assumptions
//! ------------------------
//! - The optimizer **always maximizes** a log-likelihood `ℓ(θ)` by minimizing
//!   a cost `c(θ) = -ℓ(θ)`; user code must implement `ℓ(θ)` and `∇ℓ(θ)`
//!   (when available), **never** the cost directly.
//! - [`LogLikelihood::value`] and [`LogLikelihood::grad`] must treat invalid
//!   inputs as recoverable [`OptError`] values, not panics.
//! - Vectors and matrices use the canonical aliases [`Theta`], [`Grad`],
//!   [`types::Hessian`]; all are assumed finite whenever optimization proceeds.
//! - Configuration types ([`Tolerances`], [`MLEOptions`]) are validated on
//!   construction and are treated as internally consistent by the solver
//!   layer.
//!
//! Conventions
//! -----------
//! - Parameters live in an unconstrained optimizer space as [`Theta`]
//!   (`Array1<f64>`). Any mapping from constrained → unconstrained space
//!   happens in the model layer.
//! - Cost is always `c(θ) = -ℓ(θ)` internally; all user-facing APIs and
//!   diagnostics (including [`OptimOutcome::value`]) are expressed in terms
//!   of the log-likelihood `ℓ`.
//! - Gradients exposed by [`LogLikelihood::grad`] are for the log-likelihood
//!   (`∇ℓ(θ)`); the adapter is responsible for flipping signs to obtain the
//!   cost gradient (`∇c(θ) = -∇ℓ(θ)`).
//! - Errors bubble up as [`OptResult<T>`] / [`OptError`]; this module and its
//!   children never intentionally panic or use `unsafe`.
//!
//! Downstream usage
//! ----------------
//! - Model crates implement [`LogLikelihood`] for their types, then call
//!   [`maximize`] with:
//!   - a model instance `&M`,
//!   - an initial parameter vector [`Theta`],
//!   - a data payload `&M::Data`, and
//!   - an [`MLEOptions`] configuration (tolerances, line search, L-BFGS
//!     memory).
//! - Higher-level front-ends (Python bindings, CLI tools) are expected to
//!   interact only with the re-exported surface:
//!   [`maximize`], [`LogLikelihood`], [`MLEOptions`], [`Tolerances`],
//!   [`OptimOutcome`], plus numeric aliases from [`types`].
//! - Internal optimizer code:
//!   - uses [`adapter`] to bridge into Argmin,
//!   - uses [`builders`] to construct L-BFGS solvers with the chosen
//!     line search,
//!   - delegates execution to [`run::run_lbfgs`], and
//!   - relies on [`finite_diff`] and [`validation`] for derivative and
//!     state checks.
//!
//! Testing notes
//! -------------
//! - Unit tests in submodules cover:
//!   - sign conventions and gradient handling in [`adapter`],
//!   - solver construction and tolerance wiring in [`builders`],
//!   - finite-difference + validation behavior in [`finite_diff`] and
//!     [`validation`],
//!   - configuration and outcome invariants in [`traits`].
//! - Integration tests exercise [`maximize`] implicitly on toy
//!   log-likelihoods (by fitting an ACD model), verifying that:
//!   - line-search choices are respected,
//!   - finite-difference fallbacks behave as expected, and
//!   - [`OptimOutcome`] reports sensible values and diagnostics.

pub mod adapter;
pub mod api;
pub mod builders;
pub mod finite_diff;
pub mod run;
pub mod traits;
pub mod types;
pub mod validation;

// ---- Re-exports (primary public surface) ----------------------------------

pub use self::api::maximize;
pub use self::traits::{LogLikelihood, MLEOptions, OptimOutcome, Tolerances};
pub use self::types::{Cost, DEFAULT_LBFGS_MEM, FnEvalMap, Grad, Theta};

// ---- Optional convenience prelude for downstream crates -------------------
//
// Downstream crates can write
//
//     use rust_timeseries::optimization::loglik_optimizer::prelude::*;
//
// to import the main optimizer surface in a single line.

pub mod prelude {
    pub use super::api::maximize;
    pub use super::traits::{LogLikelihood, MLEOptions, OptimOutcome, Tolerances};
    pub use super::types::{Cost, Grad, Theta};
}
