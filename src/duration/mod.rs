//! duration — ACD(p, q) duration stack: core numerics, models, and errors.
//!
//! Purpose
//! -------
//! Provide a cohesive ACD(p, q) duration layer that bundles core data /
//! parameter types, ψ-recursions, model-level fitting / forecasting /
//! inference, and shared error types under a single namespace. This is the
//! main entry point for Engle–Russell ACD models in the crate, and is the
//! surface most consumers (including Python bindings) should depend on.
//!
//! Key behaviors
//! -------------
//! - Collect core numerical and structural building blocks in [`core`]:
//!   duration data containers, units, shapes, ψ-recursions, innovation
//!   families, initialization policies, validation, and workspaces.
//! - Expose a user-facing ACD(p, q) model API in [`models`] via [`ACDModel`],
//!   including MLE in θ-space, forecasting, and covariance estimation, plus
//!   helpers like [`calculate_scores`] for per-observation scores.
//! - Centralize ACD-specific error types in [`errors`] (`ACDError`,
//!   `ParamError`, and the `ACDResult` / `ParamResult` aliases) so callers see
//!   a uniform error surface across the duration stack.
//! - Re-export the core “everyday” types (data, shapes, options, parameters,
//!   model, and errors) directly from this module and via [`prelude`] for
//!   ergonomic imports in downstream crates and bindings.
//!
//! Invariants & assumptions
//! ------------------------
//! - Duration data are carried in validated [`ACDData`] instances: finite,
//!   strictly positive, and consistent with any burn-in index `t0` and chosen
//!   [`ACDUnit`].
//! - ACD orders `(p, q)` follow the Engle–Russell convention (q = α-lags on
//!   τ, p = β-lags on ψ) and are validated via [`ACDShape::new`], ensuring
//!   that scratch buffers and workspaces are shape-consistent.
//! - Unconstrained optimizer vectors θ have length `1 + q + p` and finite
//!   entries; parameter mappings to `(ω, α, β, slack)` enforce positivity and
//!   stationarity margins via validation helpers in [`core::validation`].
//! - ψ-paths, parameters, and score matrices are treated as finite `f64`
//!   arrays; guard rails (`PsiGuards`, recursion checks) clamp or error out on
//!   pathological values rather than letting NaNs / infinities propagate.
//! - Internal scratch buffers are single-owner and not thread-safe; concurrent
//!   use of the same [`ACDModel`] instance is not supported.
//!
//! Conventions
//! -----------
//! - Indexing is 0-based throughout. Duration series store the oldest
//!   observation at index 0, newest at the end. ψ / derivative buffers use
//!   the first `p` entries/rows for pre-sample ψ-lags, with in-sample entries
//!   at indices `p..p + n`.
//! - The *effective* sample size used in likelihood / score calculations is
//!   `n_eff = data.len() - t0`; lag burn-in is handled purely via indexing
//!   (`start_idx = p + t0`), not by shortening the sample.
//! - Optimization is performed in unconstrained θ-space with layout
//!   `θ = (θ₀, θ_α[0..q), θ_β[0..p))`, mapped to parameter space using
//!   numerically stable softplus / softmax-based transforms with an explicit
//!   stationarity margin.
//! - The duration stack itself performs no I/O and no logging; callers
//!   orchestrate data loading / logging. Error conditions are surfaced as
//!   [`ACDResult`] / [`ParamResult`] and, at higher layers, as optimizer
//!   result types; panics indicate programming errors such as shape
//!   mismatches or `RefCell` borrow violations.
//!
//! Downstream usage
//! ----------------
//! - Typical end-to-end flow:
//!   1. Construct [`ACDData`] / [`ACDMeta`] in a chosen [`ACDUnit`].
//!   2. Build an [`ACDShape`] (p, q) and [`ACDOptions`] (initialization
//!      policy [`Init`], optimizer tolerances, ψ-guards).
//!   3. Choose an [`ACDInnovation`] family, then construct an [`ACDModel`]
//!      via `ACDModel::new(shape, innovation, options, n)` with `n = data.len()`.
//!   4. Fit by MLE in θ-space using `ACDModel::fit(theta0, &data)`.
//!   5. After a successful fit, use:
//!      - `forecast(horizon, &data)` for ψ/τ forecasts,
//!      - `covariance_matrix(&data, hac_opts)` for classical / HAC-robust
//!        covariances in parameter space, and
//!      - [`calculate_scores`] whenever a per-observation score matrix is
//!        needed (e.g., sandwich variance, diagnostics).
//! - Python bindings are expected to import from this module (or its
//!   [`prelude`]) and rely on `ACDError` / `ParamError` conversions into
//!   `PyErr` defined in [`errors`].
//! - Advanced callers can work directly with submodules (e.g., `core::psi`,
//!   `models::model_internals`) when they need lower-level control over
//!   ψ-recursions, workspaces, or observation traversal.
//!
//! Testing notes
//! -------------
//! - Unit tests in [`core`] cover:
//!   - data / units round-trips and `t0` handling,
//!   - ψ-recursion behavior and guard rails (in-sample and out-of-sample),
//!   - innovation configuration and validation routines,
//!   - parameter / θ validation and [`WorkSpace`] mapping.
//! - Unit tests in [`models`] cover:
//!   - [`LogLikelihood`] conformance of [`ACDModel`] (`check`, `value`,
//!     `grad` vs. finite differences),
//!   - `fit` / `forecast` behavior and `ModelNotFitted` error paths,
//!   - covariance-matrix shape, finiteness, and symmetry, and
//!   - alignment of observation traversal / score matrices with
//!     `likelihood_driver` via `model_internals`.
//! - Unit tests in [`errors`] cover concrete variant mappings, `Display`
//!   behavior, and conversions from `ParamError`, `OptError`, and `statrs`
//!   errors into [`ACDError`]. Higher-level integration / Python tests
//!   exercise full pipelines through the public [`duration`] API.

pub mod core;
pub mod errors;
pub mod models;

// ---- Re-exports (primary public surface) ----------------------------------
//
// These are the “everyday” types most users need. More specialized items
// (validation helpers, low-level ψ-recursions, simulation options, etc.)
// remain under their respective submodules.

pub use self::core::{
    ACDData, ACDForecastResult, ACDInnovation, ACDMeta, ACDOptions, ACDParams, ACDShape, ACDUnit,
    Init, PsiGuards,
};

pub use self::errors::{ACDError, ACDResult, ParamError, ParamResult};

pub use self::models::{ACDModel, calculate_scores};

// ---- Optional convenience prelude for downstream crates -------------------
//
// Downstream crates can write
//
//     use rust_timeseries::duration::prelude::*;
//
// to import the main duration-model surface in a single line, without pulling
// in lower-level internals.

pub mod prelude {
    pub use super::{
        ACDData, ACDError, ACDForecastResult, ACDInnovation, ACDMeta, ACDModel, ACDOptions,
        ACDParams, ACDResult, ACDShape, ACDUnit, Init, ParamError, ParamResult, PsiGuards,
        calculate_scores,
    };
}
