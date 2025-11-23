//! core — shared ACD(p, q) data, parameters, and ψ-recursions.
//!
//! Purpose
//! -------
//! Collect the core building blocks for Engle–Russell ACD(p, q) duration
//! models: data containers, parameter shapes, ψ-recursions (in-sample and
//! out-of-sample), initialization policies, innovation families, and
//! validation helpers. Higher-level duration models and optimizers build on
//! top of these primitives.
//!
//! Key behaviors
//! -------------
//! - Define model configuration and shape types ([`ACDShape`], [`ACDUnit`],
//!   [`ACDOptions`]) plus owned parameter containers ([`ACDParams`],
//!   [`ACDScratch`]) and a zero-copy optimizer workspace ([`WorkSpace`]).
//! - Implement the ACD(p, q) ψ-recursion and its guards for in-sample paths
//!   ([`compute_psi`], [`PsiGuards`]) and provide allocation-free
//!   out-of-sample forecasting ([`ACDForecastResult`], [`forecast_recursion`]).
//! - Encapsulate innovation distributions ([`ACDInnovation`] and helpers),
//!   initialization policies ([`Init`]), and validation routines for
//!   parameters, lags, and θ-vectors.
//! - Track duration data and metadata ([`ACDData`], [`ACDMeta`]) with explicit
//!   units and scaling so downstream code can assume well-formed inputs.
//!
//! Invariants & assumptions
//! ------------------------
//! - Duration observations stored in [`ACDData`] are strictly positive after
//!   any unit conversion or scaling; zero/negative values are treated as
//!   invalid upstream.
//! - ACD orders follow the Engle–Russell convention: `q` counts α-lags on
//!   durations τ, `p` counts β-lags on conditional means ψ; shapes and buffer
//!   lengths are checked via [`ACDShape`] and validation helpers.
//! - Parameter mappings (`θ → (ω, α, β, slack)`) enforce positivity and
//!   stationarity margins via validation functions (e.g.,
//!   [`validate_omega`], [`validate_alpha`], [`validate_beta`],
//!   [`validate_stationarity_and_slack`]); callers can assume that
//!   successfully constructed parameters satisfy these constraints.
//! - ψ-paths and parameters are treated as finite `f64` values; guard rails
//!   ([`PsiGuards`], [`guard_psi`]) are used to clamp extreme values rather
//!   than allowing NaNs/inf to propagate through the stack.
//! - Length relationships between shapes, lags, and θ-vectors are enforced
//!   by functions like [`validate_alpha_beta_lengths`] and
//!   [`validate_theta`]; mismatches are surfaced as `ParamError` / `ACDError`
//!   rather than silently truncated.
//!
//! Conventions
//! -----------
//! - Indexing is 0-based throughout. Lag arrays and buffers use the
//!   convention “oldest at index 0, newest at the end”; ψ-forecasts are
//!   stored as `psi_forecast[i] = ψ̂_{T+i+1}`.
//! - Units are represented explicitly via [`ACDUnit`]; conversions to a
//!   working scale happen in the data layer so the core recursion only deals
//!   with dimensionless, positive durations.
//! - ACD parameters are organized as `(ω, α, β, slack)` with orders `(p, q)`
//!   carried by [`ACDShape`]; unconstrained optimizer parameters `θ` have
//!   length `1 + q + p`.
//! - This module avoids I/O and logging; it operates purely
//!   on `ndarray` containers and scalar values. Error conditions are reported
//!   via `ACDResult` / `ParamResult`; panics are reserved for logic bugs such
//!   as irreconcilable length mismatches.
//!
//! Downstream usage
//! ----------------
//! - Data preparation code constructs [`ACDData`] / [`ACDMeta`] in a chosen
//!   [`ACDUnit`], then derives an [`ACDShape`] and [`ACDOptions`] describing
//!   `(p, q)`, innovation family ([`ACDInnovation`]), and initialization
//!   policy ([`Init`]) for an ACD model.
//! - Optimizer-facing code allocates α/β buffers and a [`WorkSpace`], then
//!   repeatedly calls `WorkSpace::update` with candidate `θ` vectors and
//!   uses [`compute_psi`] plus log-likelihood logic elsewhere to evaluate
//!   fits, with validation handled by functions in [`validation`].
//! - After fitting, callers build owned parameters ([`ACDParams`]) and use
//!   [`ACDForecastResult`] + [`forecast_recursion`] to generate out-of-sample
//!   ψ- and duration forecasts, reusing buffers to avoid allocations.
//! - Higher-level APIs (e.g., `duration::models`, Python bindings) are
//!   expected to depend primarily on the types and functions re-exported
//!   below or via the [`prelude`] rather than reaching into submodules
//!   directly.
//!
//! Testing notes
//! -------------
//! - Unit tests in submodules cover: data/units round-trips, ψ-recursion
//!   behavior and guarding, innovation configuration, parameter/θ validation,
//!   workspace updates, and out-of-sample forecasting behavior under simple
//!   configurations.
//! - Integration tests at the duration-model and optimization layers exercise
//!   full pipelines (data → parameters → ψ-recursion → likelihood → fit),
//!   treating this module as the underlying numerical and structural
//!   “core” for ACD(p, q) models.

pub mod data;
pub mod forecasts;
pub mod guards;
pub mod init;
pub mod innovations;
pub mod options;
pub mod params;
pub mod psi;
pub mod shape;
pub mod units;
pub mod validation;
pub mod workspace;

// ---- Re-exports (primary public surface) ----------------------------------

pub use self::data::{ACDData, ACDMeta};
pub use self::forecasts::{ACDForecastResult, forecast_recursion};
pub use self::guards::PsiGuards;
pub use self::init::Init;
pub use self::innovations::ACDInnovation;
pub use self::options::{ACDOptions, SimOpts, SimStart};
pub use self::params::{ACDParams, ACDScratch};
pub use self::psi::{compute_psi, guard_psi};
pub use self::shape::ACDShape;
pub use self::units::ACDUnit;
pub use self::validation::{
    validate_alpha, validate_alpha_beta_lengths, validate_beta, validate_duration_lags,
    validate_gamma_param, validate_loglik_params, validate_omega, validate_psi_lags,
    validate_stationarity_and_slack, validate_theta, validate_weibull_param,
};
pub use self::workspace::WorkSpace;

// ---- Optional convenience prelude for downstream crates -------------------
//
// Downstream crates can write
//
//     use rust_timeseries::duration::core::prelude::*;
//
// to import the main ACD core surface in a single line.

pub mod prelude {
    pub use super::data::{ACDData, ACDMeta};
    pub use super::forecasts::{ACDForecastResult, forecast_recursion};
    pub use super::guards::PsiGuards;
    pub use super::init::Init;
    pub use super::innovations::ACDInnovation;
    pub use super::options::{ACDOptions, SimOpts, SimStart};
    pub use super::params::{ACDParams, ACDScratch};
    pub use super::psi::compute_psi;
    pub use super::shape::ACDShape;
    pub use super::units::ACDUnit;
    pub use super::workspace::WorkSpace;
}
