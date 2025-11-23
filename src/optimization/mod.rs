//! optimization — MLE stack, numerical helpers, and unified error surface.
//!
//! Purpose
//! -------
//! Provide a cohesive optimization layer for model fitting, combining an
//! Argmin-backed log-likelihood optimizer, numerically stable parameter
//! transforms, and a single error/result surface. Callers implement a
//! log-likelihood, choose tolerances, and obtain fitted parameters and
//! diagnostics without touching backend solver details.
//!
//! Key behaviors
//! -------------
//! - Expose a high-level API for **maximizing log-likelihoods** `ℓ(θ)`
//!   (`loglik_optimizer`), including configuration of solvers and stopping
//!   criteria.
//! - Supply shared numerical primitives (`numerical_stability`) for mapping
//!   unconstrained parameters into model space and propagating covariance
//!   information via the delta method.
//! - Normalize configuration issues, numerical failures, and backend solver
//!   errors into a single enum (`errors::OptError`) with a common result
//!   alias (`OptResult<T>`).
//!
//! Invariants & assumptions
//! ------------------------
//! - Optimizers operate in an unconstrained parameter space `θ` and assume
//!   that inputs are finite once validation has passed; invalid states are
//!   reported as `OptError`, not panics.
//! - Log-likelihood implementations are expected to treat domain violations
//!   (e.g., non-positive durations, non-stationary parameters) as recoverable
//!   errors surfaced through the optimization layer.
//! - Stationarity, positivity, and dimension checks for model parameters are
//!   enforced via shared validation and error conversions, so downstream code
//!   can assume that accepted parameters satisfy basic domain constraints.
//!
//! Conventions
//! -----------
//! - All solvers conceptually maximize a log-likelihood `ℓ(θ)` by minimizing
//!   an internal cost `c(θ) = -ℓ(θ)`; user-facing APIs and outcomes are
//!   expressed in terms of `ℓ`.
//! - Parameters, gradients, and Hessians are represented using `ndarray`-
//!   based aliases (`Theta`, `Grad`, `Hessian` types); any mapping between
//!   unconstrained θ-space and structured model parameters (e.g., ACD
//!   `(ω, α, β)`) is handled by numerical-stability helpers.
//! - Public optimization entrypoints that can fail return `OptResult<T>`;
//!   callers never see raw Argmin errors or model-specific error enums.
//! - This module and its submodules avoid I/O and logging; higher layers
//!   (Python bindings, CLI tools, notebooks) are responsible for reporting
//!   progress and diagnostics.
//!
//! Downstream usage
//! ----------------
//! - Model crates implement `LogLikelihood` for their types and call
//!   `maximize` with a parameter guess, data payload, and `MLEOptions` to
//!   obtain an `OptimOutcome` (via `loglik_optimizer`).
//! - Duration and inference code use `numerical_stability` for stable
//!   transforms, stationarity-aware mappings, and delta-method covariance
//!   propagation when interpreting optimizer output.
//! - Front-ends typically import the curated surface via
//!   `optimization::prelude::*`, which forwards the submodule preludes and
//!   the core error types, or they depend directly on
//!   `loglik_optimizer::prelude` / `numerical_stability::prelude` when they
//!   want a more fine-grained split.
//!
//! Testing notes
//! -------------
//! - Unit tests in the submodules focus on local concerns:
//!   - `loglik_optimizer`: solver wiring, tolerance handling, and basic
//!     MLE behavior on toy models.
//!   - `numerical_stability`: agreement with naïve formulas on safe grids,
//!     well-behaved tails, and delta-method consistency checks.
//!   - `errors`: conversions from backend/model errors into `OptError` and
//!     basic invariants of the error surface.
//! - Higher-level integration tests exercise end-to-end MLE workflows,
//!   verifying that configuration mistakes, numerical problems, and backend
//!   failures all surface as sensible `OptError` values and that successful
//!   runs produce stable `OptimOutcome`s.

pub mod errors;
pub mod loglik_optimizer;
pub mod numerical_stability;

// ---- Optional convenience prelude for downstream crates -------------------
//
// Downstream crates can write
//
//     use rust_timeseries::optimization::prelude::*;
//
// to import the main optimization surface in a single line.

pub mod prelude {
    pub use super::errors::{OptError, OptResult};
    pub use super::loglik_optimizer::prelude::*;
    pub use super::numerical_stability::prelude::*;
}
