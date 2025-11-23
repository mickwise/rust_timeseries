//! models — high-level ACD(p, q) duration models and inference helpers.
//!
//! Purpose
//! -------
//! Collect user-facing ACD(p, q) model APIs (fitting, forecasting, inference)
//! plus low-level observation-walking helpers used for standard errors and
//! diagnostics. This layer sits on top of `duration::core`, wiring together
//! ψ-recursions, innovation families, and the generic log-likelihood optimizer.
//!
//! Key behaviors
//! -------------
//! - Expose a complete ACD(p, q) model type [`ACDModel`] that implements
//!   [`LogLikelihood`] and provides `fit`, `forecast`, and
//!   `covariance_matrix` methods.
//! - Centralize in-sample traversal and score construction in
//!   [`model_internals`], including per-observation score matrices
//!   ([`calculate_scores`]) and θ̂ extraction ([`extract_theta`]).
//! - Reuse shared workspaces and scratch buffers from `duration::core` to
//!   keep likelihood / gradient evaluations allocation-free in ψ-space.
//! - Provide a light-weight prelude so downstream code can import the main
//!   ACD model surface in a single line.
//!
//! Invariants & assumptions
//! ------------------------
//! - Duration data are carried in validated [`ACDData`] instances: finite,
//!   strictly positive, and consistent with any burn-in index `t0`.
//! - Model orders `(p, q)` are validated via [`ACDShape::new`]; scratch
//!   buffers in [`ACDScratch`] match both the shape and in-sample length `n`.
//! - Unconstrained optimizer vectors θ always have length `1 + p + q` and
//!   finite entries; this is enforced by [`LogLikelihood::check`] via
//!   `duration::core::validation::validate_theta`.
//! - Parameter-space invariants (`ω, slack > 0`, `αᵢ ≥ 0`, `βⱼ ≥ 0`,
//!   `∑α + ∑β + slack< 1 − margin`) are enforced by [`WorkSpace`] +
//!   `duration::core::validation`; invalid θ is surfaced as [`ACDError`] /
//!   [`OptResult`] instead of panics.
//! - Internal ψ and derivative buffers are treated as single-owner,
//!   non-thread-safe scratch; concurrent access to the same [`ACDModel`]
//!   instance is not supported.
//!
//! Conventions
//! -----------
//! - Optimization is performed in unconstrained θ-space with the layout
//!   `θ = (θ₀, θ_α[0..q), θ_β[0..p))`:
//!   - `ω = softplus(θ₀)` via a numerically stable logistic / softplus pair,
//!   - `(α, β, slack)` from a scaled softmax over `θ[1..]`.
//! - Time indexing in ψ / derivative buffers is 0-based; the first `p` rows
//!   store pre-sample ψ-lags, and in-sample observations live at indices
//!   `p..p + n`.
//! - The *effective* sample size `n_eff` is defined as `data.len() - t0`,
//!   independent of `p`; lag burn-in is handled purely by indexing via
//!   [`start_idx`].
//! - Errors are reported as [`ACDResult`] / [`OptResult`]; panics indicate
//!   programming errors (e.g., shape mismatches, borrow violations), not
//!   bad user data or bad θ.
//!
//! Downstream usage
//! ----------------
//! - Construct an [`ACDShape`] from `(p, q, n)` and an [`ACDOptions`] bundle
//!   (initialization policy, optimizer tolerances, ψ-guards).
//! - Build an [`ACDModel`] via `ACDModel::new(shape, innovation, options, n)`,
//!   then call `fit(theta0, &data)` to perform MLE in θ-space.
//! - After a successful fit:
//!   - use `forecast(horizon, &data)` for ψ/τ forecasts,
//!   - use `covariance_matrix(&data, hac_opts)` for classical / HAC-robust
//!     covariance in parameter space, and
//!   - use [`calculate_scores`] when an explicit per-observation score matrix
//!     is needed (e.g., sandwich variance, diagnostics).
//! - For advanced inference or diagnostics, use helpers from
//!   [`model_internals`] (e.g., [`walk_observations`], [`n_eff`]) to traverse
//!   ψ/derivative paths in lockstep with the data.
//! - Front-ends (Python bindings, CLI tools) are expected to depend mainly on
//!   the items re-exported below or via the [`prelude`].
//!
//! Testing notes
//! -------------
//! - Unit tests in [`acd`] cover:
//!   - [`LogLikelihood`] conformance of [`ACDModel`] (`check`, `value`,
//!     `grad` vs. finite differences),
//!   - `fit` / `forecast` behavior on synthetic data, and
//!   - `covariance_matrix` shape, finiteness, and symmetry.
//! - Unit tests in [`model_internals`] cover:
//!   - alignment between [`n_eff`], [`start_idx`], and `likelihood_driver`,
//!   - correct wiring of [`with_workspace`] (positive, finite ψ) and
//!   - traversal behavior of [`walk_observations`].
//! - Higher-level integration / Python tests exercise full pipelines
//!   (data → fit → forecast → covariance / scores) through the public
//!   [`ACDModel`] API.

pub mod acd;
pub mod model_internals;

// ---- Re-exports (primary public surface) ----------------------------------

pub use self::acd::ACDModel;
pub use self::model_internals::{calculate_scores, n_eff, start_idx, walk_observations};

// ---- Optional convenience prelude for downstream crates -------------------
//
// Downstream crates can write
//
//     use rust_timeseries::duration::models::prelude::*;
//
// to import the main ACD model surface in a single line.

pub mod prelude {
    pub use super::acd::ACDModel;
    pub use super::model_internals::calculate_scores;
}
