//! ACD options — configuration for estimation and simulation workflows.
//!
//! Purpose
//! -------
//! Collect configuration knobs for ACD(p, q) estimation and simulation in one
//! place, making the workflow explicit and reproducible. This includes
//! estimation-time options (initialization, optimizer, ψ-guards) and
//! simulation-time options (RNG seed, warm/cold start behavior, and which
//! series to return).
//!
//! Key behaviors
//! -------------
//! - Represent estimation configuration via [`ACDOptions`], bundling the
//!   initialization policy, maximum-likelihood optimizer options, and ψ-guards
//!   used during recursion.
//! - Represent simulation configuration via [`SimOpts`] and [`SimStart`],
//!   controlling RNG seeding, whether to perform a warm or cold start, and
//!   which internal series (ε, ψ) should be returned.
//! - Keep cross-cutting configuration out of low-level recursion code, so call
//!   sites pass explicit, validated options instead of ad-hoc flags.
//!
//! Invariants & assumptions
//! ------------------------
//! - [`ACDOptions`] assumes that its components (`Init`, `MLEOptions`,
//!   `PsiGuards`) have already been validated by their own builders; it does
//!   not impose additional cross-field checks.
//! - [`SimOpts`] and [`SimStart`] describe *intent* (seed, warm vs cold start,
//!   optional ψ-lags); enforcement of length/positivity constraints on data
//!   and lags is performed in the simulation engine, not here.
//! - Simulation assumes unit-mean innovations (E[ε] = 1) as enforced by the
//!   chosen [`ACDInnovation`].
//!
//! Conventions
//! -----------
//! - ψ denotes the conditional mean duration process; τ denotes observed
//!   durations (inter-arrival times).
//! - Estimation and simulation use `f64` throughout and rely on internal
//!   validation (e.g., [`PsiGuards::new`], `Init::fixed`, etc.) to enforce
//!   positivity and finiteness.
//! - This module provides plain data carriers and builders that never panic;
//!   any invalid configuration is expected to be rejected earlier by the
//!   component types (`Init`, `MLEOptions`, `PsiGuards`, and the simulator).
//!
//! Downstream usage
//! ----------------
//! - At model setup time, construct an [`ACDOptions`] with the desired
//!   initialization policy, optimizer settings, and ψ-guards, and pass it to
//!   the ACD estimation entry point.
//! - For simulation, build a [`SimOpts`] (or start from `SimOpts::default()`),
//!   choosing whether to perform a warm start or a cold start via [`SimStart`],
//!   and pass it into the ACD simulator.
//! - Treat this module as the public configuration surface for tuning ACD
//!   estimation and simulation; low-level code should depend on these types
//!   rather than on ad-hoc arguments.
//!
//! Testing notes
//! -------------
//! - Unit tests for this module:
//!   - verify that `ACDOptions::new` preserves its inputs without mutation,
//!   - verify that `SimOpts::new` and `SimOpts::default` set fields as
//!     documented (seed, flags, and `SimStart` variant).
//! - Behavioral tests (e.g., enforcing burn-in semantics, validating ψ/τ lag
//!   lengths, applying ψ-guards during recursion) are covered by integration
//!   tests in the estimation and simulation modules rather than here.
use crate::{
    duration::core::{guards::PsiGuards, init::Init},
    optimization::loglik_optimizer::MLEOptions,
};
use ndarray::Array1;

/// ACDOptions — estimation-time configuration for ACD models.
///
/// Purpose
/// -------
/// Bundle the key configuration components required to fit an ACD(p, q) model:
/// the pre-sample initialization policy, maximum-likelihood optimizer options,
/// and ψ-guards used to protect the recursion from numerical pathologies.
///
/// Key behaviors
/// -------------
/// - Holds the chosen initialization policy (`init`), which determines how
///   pre-sample ψ and τ lags are seeded before the recursion starts.
/// - Carries optimizer options (`mle_opts`) used by the L-BFGS / line-search
///   backend during likelihood maximization.
/// - Exposes ψ-guards (`psi_guards`) that bound the ψ recursion and are
///   consumed by downstream code when updating ψ_t.
///
/// Parameters
/// ----------
/// Constructed via:
/// - `ACDOptions::new(init: Init, mle_opts: MLEOptions, psi_guards: PsiGuards)`
///   Provide already-validated components from their respective builders.
///   No additional cross-field validation is performed here.
///
/// Fields
/// ------
/// - `init`: [`Init`]
///   Initialization policy for pre-sample ψ and duration lags (e.g.,
///   unconditional mean, sample mean, fixed scalar, or explicit vectors).
/// - `mle_opts`: [`MLEOptions`]
///   Optimizer configuration (tolerances, maximum iterations, line-search
///   strategy, logging controls) used during MLE.
/// - `psi_guards`: [`PsiGuards`]
///   Lower/upper bounds applied to ψ_t at each recursion step to prevent
///   collapse toward zero or explosion to numerically unsafe values.
///
/// Invariants
/// ----------
/// - Each field is assumed to have been constructed via its own validated
///   builder or default; `ACDOptions` does not enforce extra constraints.
/// - Callers should ensure that `psi_guards` are compatible with the scale of
///   durations and ψ implied by the model and data.
///
/// Performance
/// -----------
/// - Struct is small and `Clone`/`PartialEq`, making it cheap to pass by value
///   or store as part of a larger configuration object.
///
/// Notes
/// -----
/// - This type is intended to be the primary estimation-configuration handle
///   for ACD models. Public APIs should accept `ACDOptions` rather than
///   separate `init`, `mle_opts`, and `psi_guards` parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ACDOptions {
    /// Initialization policy for pre-sample ψ and duration lags.
    pub init: Init,
    /// Maximum-likelihood optimizer options (L-BFGS + line search).
    pub mle_opts: MLEOptions,
    /// Bounds for ψ during recursion to prevent divergence.
    pub psi_guards: PsiGuards,
}

impl ACDOptions {
    /// Construct a new [`ACDOptions`] from already-validated components.
    ///
    /// Parameters
    /// ----------
    /// - `init`: `Init`
    ///   Initialization policy for pre-sample ψ and duration lags. Must be
    ///   constructed via a valid `Init` builder (e.g., `uncond_mean`, `sample_mean`,
    ///   `fixed`, or explicit lag vectors).
    /// - `mle_opts`: `MLEOptions`
    ///   Maximum-likelihood optimizer configuration (tolerances, iteration caps,
    ///   line-search strategy, optional L-BFGS memory). Must be a validated
    ///   instance created via `MLEOptions::new` or `MLEOptions::default`.
    /// - `psi_guards`: `PsiGuards`
    ///   Lower and upper bounds for ψ during recursion, constructed via
    ///   `PsiGuards::new` to enforce positivity and prevent numerical blow-ups.
    ///
    /// Returns
    /// -------
    /// `ACDOptions`
    ///   A configuration struct bundling the provided initialization, optimizer
    ///   options, and ψ-guards, with no additional transformation applied.
    ///
    /// Errors
    /// ------
    /// - Never returns an error.
    ///   All validation is expected to have been performed by `Init`,
    ///   `MLEOptions`, and `PsiGuards` constructors before this function is called.
    ///
    /// Panics
    /// ------
    /// - Never panics.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; no additional caller guarantees are required
    ///   beyond supplying valid components.
    ///
    /// Notes
    /// -----
    /// - This is a thin convenience constructor: it simply packages the three
    ///   components into a single `ACDOptions` value without performing
    ///   cross-field checks.
    /// - Callers are responsible for ensuring that the scale of `psi_guards`
    ///   is compatible with the model parameters and data.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::options::ACDOptions;
    /// # use rust_timeseries::duration::core::{guards::PsiGuards, init::Init};
    /// # use rust_timeseries::optimization::loglik_optimizer::{MLEOptions, Tolerances, LineSearcher};
    ///
    /// let init = Init::uncond_mean();
    /// let tols = Tolerances::new(Some(1e-6), None, Some(300)).unwrap();
    /// let mle_opts = MLEOptions::new(tols, LineSearcher::MoreThuente, None).unwrap();
    /// let psi_guards = PsiGuards::new(1e-6, 1e6).unwrap();
    ///
    /// let opts = ACDOptions::new(init, mle_opts, psi_guards);
    /// # assert!(opts.psi_guards.min() > 0.0);
    /// ```
    pub fn new(init: Init, mle_opts: MLEOptions, psi_guards: PsiGuards) -> ACDOptions {
        ACDOptions { init, mle_opts, psi_guards }
    }
}

/// SimOpts — simulation-time configuration for synthetic ACD paths.
///
/// Purpose
/// -------
/// Control how synthetic ACD(p, q) durations are simulated: RNG seeding,
/// whether to perform a warm or cold start, and which internal series
/// (innovations and ψ) should be returned alongside simulated durations.
///
/// Key behaviors
/// -------------
/// - Encapsulates RNG seeding (`seed`) for reproducible simulations.
/// - Indicates whether the simulator should return innovation draws (ε) and/or
///   the conditional mean path (ψ) in addition to simulated durations.
/// - Delegates the choice between warm and cold starts to [`SimStart`],
///   which encodes how ψ-lags are initialized before producing output.
///
/// Parameters
/// ----------
/// Constructed via:
/// - `SimOpts::new(seed: Option<u64>, return_eps: bool, return_psi: bool, sim_start: SimStart)`
///   Build an explicit configuration from the provided arguments.
/// - `SimOpts::default()`
///   Start from a conservative, reproducible default and override fields as
///   needed for a given experiment.
///
/// Fields
/// ------
/// - `seed`: `Option<u64>`
///   Optional RNG seed. `Some(seed)` yields reproducible runs; `None`
///   delegates to upstream system entropy. The simulator may record the
///   effective seed in its output metadata.
/// - `return_eps`: `bool`
///   Whether to return the path of innovations ε_t (unit-mean shocks).
/// - `return_psi`: `bool`
///   Whether to return the conditional mean path ψ_t alongside simulated
///   durations.
/// - `sim_start`: [`SimStart`]
///   Strategy for seeding the ψ–τ recursion (warm vs cold start).
///
/// Invariants
/// ----------
/// - `SimOpts` itself does not enforce constraints on lag lengths or data;
///   those checks are the responsibility of the simulation engine that
///   consumes these options.
/// - When `sim_start` is `SimStart::Cold { burn_in, psi_init }`, the simulator
///   is expected to validate that `psi_init` has length at least `p` and that
///   any provided duration lags are compatible with the model’s `q`.
///
/// Performance
/// -----------
/// - Struct is lightweight and `Clone`/`PartialEq`. Copies are cheap and it
///   can be stored or passed around as part of a larger simulation config.
///
/// Notes
/// -----
/// - The default implementation is geared toward stable, reproducible warm-start
///   experiments:
///   - `seed = Some(42)`
///   - `return_eps = false`
///   - `return_psi = false`
///   - `sim_start = SimStart::Warm { burn_in: 2000 }`
/// - Callers typically construct `SimOpts` once per simulation experiment and
///   pass it into the ACD simulator entry point.
#[derive(Debug, Clone, PartialEq)]
pub struct SimOpts {
    /// Optional RNG seed for reproducibility.
    pub seed: Option<u64>,
    /// Whether to return the innovation path ε.
    pub return_eps: bool,
    /// Whether to return the conditional-mean path ψ.
    pub return_psi: bool,
    /// Mode for simulation start: warm or cold (with provided lags).
    pub sim_start: SimStart,
}

impl SimOpts {
    /// Construct a new [`SimOpts`] instance from explicit simulation settings.
    ///
    /// Parameters
    /// ----------
    /// - `seed`: `Option<u64>`
    ///   Optional RNG seed for reproducibility. Use `Some(seed)` to obtain
    ///   deterministic simulations; use `None` to delegate seeding to upstream
    ///   system entropy or a higher-level driver.
    /// - `return_eps`: `bool`
    ///   If `true`, request that the simulator return the path of innovations
    ///   ε_t in addition to simulated durations.
    /// - `return_psi`: `bool`
    ///   If `true`, request that the simulator return the conditional mean path
    ///   ψ_t in addition to simulated durations.
    /// - `sim_start`: [`SimStart`]
    ///   High-level start mode, either warm (with a configured `burn_in`) or
    ///   cold (with explicit ψ-lags via `SimStart::Cold { psi_init }`).
    ///
    /// Returns
    /// -------
    /// `SimOpts`
    ///   A configuration struct encapsulating all simulation-time options:
    ///   RNG seeding, which internal series to return, and how to seed the
    ///   ψ–τ recursion.
    ///
    /// Errors
    /// ------
    /// - Never returns an error.
    ///   Any validation of `psi_init` length/values or data lag requirements
    ///   is performed by the simulation engine that consumes `SimOpts`.
    ///
    /// Panics
    /// ------
    /// - Never panics.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; callers must only ensure that `sim_start` is
    ///   meaningful for their intended simulation (e.g., `psi_init` length
    ///   matches the ACD order when using `Cold`).
    ///
    /// Notes
    /// -----
    /// - This constructor does not inspect or validate any associated data;
    ///   it is purely a container for user intent.
    /// - Typical usage is to call `SimOpts::new` at the start of a simulation
    ///   experiment and pass the resulting value into an ACD simulator entry
    ///   point.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use ndarray::array;
    /// # use rust_timeseries::duration::core::options::{SimOpts, SimStart};
    ///
    /// let psi_init = array![1.0, 1.0, 1.0];
    /// let sim_start = SimStart::Cold { psi_init };
    ///
    /// let opts = SimOpts::new(
    ///     Some(123_u64),
    ///     true,   // return_eps
    ///     true,   // return_psi
    ///     sim_start,
    /// );
    ///
    /// # assert!(opts.seed.is_some());
    /// # assert!(opts.return_eps && opts.return_psi);
    /// ```
    pub fn new(
        seed: Option<u64>, return_eps: bool, return_psi: bool, sim_start: SimStart,
    ) -> SimOpts {
        SimOpts { seed, return_eps, return_psi, sim_start }
    }
}

impl Default for SimOpts {
    /// Construct conservative, reproducible default simulation options.
    ///
    /// Parameters
    /// ----------
    /// - *(none)*
    ///   This is the `Default` implementation; it takes no arguments and
    ///   returns a baseline configuration for ACD simulations.
    ///
    /// Returns
    /// -------
    /// `SimOpts`
    ///   A simulation configuration with:
    ///   - `seed = Some(42)`
    ///   - `return_eps = false`
    ///   - `return_psi = false`
    ///   - `sim_start = SimStart::Warm { burn_in: 2000 }`
    ///
    /// Errors
    /// ------
    /// - Never returns an error.
    ///
    /// Panics
    /// ------
    /// - Never panics.
    ///
    /// Safety
    /// ------
    /// - Not an `unsafe fn`; there are no special caller obligations.
    ///
    /// Notes
    /// -----
    /// - These defaults are geared toward stable, reproducible warm-start
    ///   experiments where only the duration path is required.
    /// - Callers are expected to use this as a base and override fields as
    ///   needed (e.g., enable `return_psi` or switch to a cold start).
    ///
    /// Examples
    /// --------
    /// ```rust
    /// # use rust_timeseries::duration::core::options::{SimOpts, SimStart};
    ///
    /// let opts = SimOpts::default();
    ///
    /// assert_eq!(opts.seed, Some(42));
    /// assert!(!opts.return_eps);
    /// assert!(!opts.return_psi);
    ///
    /// match opts.sim_start {
    ///     SimStart::Warm { burn_in } => assert_eq!(burn_in, 2000),
    ///     _ => panic!("expected warm-start default"),
    /// }
    /// ```
    fn default() -> Self {
        SimOpts {
            seed: Some(42),
            return_eps: false,
            return_psi: false,
            sim_start: SimStart::Warm { burn_in: (2000) },
        }
    }
}

/// SimStart — warm vs cold start strategies for ACD simulation.
///
/// Purpose
/// -------
/// Describe how the ACD ψ–τ recursion should be initialized before producing
/// simulated durations, distinguishing between a warm start (letting the model
/// reach its steady-state behavior) and a cold start from explicit ψ-lags.
///
/// Variants
/// --------
/// - `Warm { burn_in }`
///   Use the model’s configured initialization (e.g., unconditional mean or
///   sample mean) and perform a warm-up phase before collecting output. The
///   exact warm-up length is determined by the simulator or higher-level
///   options; this variant signals that no explicit ψ-lags are supplied.
/// - `Cold { psi_init }`
///   Start from explicit ψ-lag values and bypass any internal warm-up. The
///   simulator uses `psi_init` to seed ψ.
///
/// Invariants
/// ----------
/// - For `Cold`:
///   - `psi_init` must contain finite, strictly positive values and have
///     length at least `p` for the ACD(p, q) model in use.
///   - ψ-lags are produced internally (e.g., via recursive
///     application of the model), not taken from user input.
/// - For `Warm`, `burn_in` is interpreted as a non-negative integer; the simulator is
///     responsible for enforcing any minimum length requirements.
///
/// Notes
/// -----
/// - This enum is used by [`SimOpts`] to encode the high-level simulation
///   start mode. The actual mechanics (how `burn_in` is applied, where ψ- and
///   τ-lags are read from) are implemented in the simulation engine.
/// - Downstream code should pattern-match on `SimStart` exhaustively so that
///   the compiler flags missing cases if new variants are added later.
#[derive(Debug, Clone, PartialEq)]
pub enum SimStart {
    Cold { psi_init: Array1<f64> },
    Warm { burn_in: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duration::core::{guards::PsiGuards, init::Init};
    use crate::optimization::loglik_optimizer::traits::{LineSearcher, MLEOptions, Tolerances};
    use ndarray::array;

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - That `ACDOptions::new` preserves its inputs without modification.
    // - That `SimOpts::new` and `SimOpts::default` set fields as documented.
    // - That `SimStart` variants store their payloads correctly.
    //
    // They intentionally DO NOT cover:
    // - The behavior of the optimizer (L-BFGS), which is tested in the
    //   optimization module.
    // - The semantics of warm vs cold simulation in the ACD simulator; those
    //   are covered by higher-level integration tests.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Verify that `ACDOptions::new` preserves its input components exactly.
    //
    // Given
    // -----
    // - A simple `Init` policy, a default `MLEOptions`, and valid `PsiGuards`.
    //
    // Expect
    // ------
    // - The returned `ACDOptions` contains the same values in each field and
    //   does not mutate or reconstruct its inputs.
    fn acdoptions_new_preserves_fields() {
        // Arrange
        let init = Init::uncond_mean();
        let tols = Tolerances::new(Some(1e-6), None, Some(100)).unwrap();
        let mle_opts = MLEOptions::new(tols, LineSearcher::MoreThuente, Some(5)).unwrap();
        let psi_guards = PsiGuards::new((1e-6, 1e6)).unwrap();

        // Act
        let opts = ACDOptions::new(init.clone(), mle_opts.clone(), psi_guards.clone());

        // Assert
        assert_eq!(opts.init, init);
        assert_eq!(opts.mle_opts, mle_opts);
        assert_eq!(opts.psi_guards, psi_guards);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `SimOpts::new` sets all fields according to its arguments.
    //
    // Given
    // -----
    // - An explicit seed, flags for returning ε and ψ, and a cold-start
    //   `SimStart` variant with a non-empty `psi_init`.
    //
    // Expect
    // ------
    // - The resulting `SimOpts` struct mirrors those inputs exactly.
    fn simopts_new_sets_fields_as_provided() {
        // Arrange
        let seed = Some(123_u64);
        let return_eps = true;
        let return_psi = true;
        let psi_init = array![1.0, 1.1, 1.2];
        let sim_start = SimStart::Cold { psi_init: psi_init.clone() };

        // Act
        let opts = SimOpts::new(seed, return_eps, return_psi, sim_start.clone());

        // Assert
        assert_eq!(opts.seed, seed);
        assert_eq!(opts.return_eps, return_eps);
        assert_eq!(opts.return_psi, return_psi);
        assert_eq!(opts.sim_start, sim_start);
    }

    #[test]
    // Purpose
    // -------
    // Verify that `SimOpts::default` matches the documented default values.
    //
    // Given
    // -----
    // - The `Default` implementation for `SimOpts`.
    //
    // Expect
    // ------
    // - `seed = Some(42)`,
    // - `return_eps = false`,
    // - `return_psi = false`,
    // - `sim_start = SimStart::Warm { burn_in: 2000 }`.
    fn simopts_default_matches_documented_defaults() {
        // Arrange + Act
        let opts = SimOpts::default();

        // Assert
        assert_eq!(opts.seed, Some(42));
        assert!(!opts.return_eps);
        assert!(!opts.return_psi);

        match opts.sim_start {
            SimStart::Warm { burn_in } => {
                assert_eq!(burn_in, 2000);
            }
            other => panic!("expected SimStart::Warm {{ burn_in: 2000 }}, got {:?}", other),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that the `SimStart::Cold` variant stores `psi_init` as provided.
    //
    // Given
    // -----
    // - A `SimStart::Cold` constructed with a known `psi_init` vector.
    //
    // Expect
    // ------
    // - Pattern-matching on the variant yields an identical `psi_init`.
    fn simstart_cold_stores_psi_init() {
        // Arrange
        let psi_init = array![0.9, 1.0, 1.1];
        let sim_start = SimStart::Cold { psi_init: psi_init.clone() };

        // Act + Assert
        match sim_start {
            SimStart::Cold { psi_init: stored } => {
                assert_eq!(stored, psi_init);
            }
            _ => panic!("expected SimStart::Cold variant"),
        }
    }

    #[test]
    // Purpose
    // -------
    // Verify that the `SimStart::Warm` variant stores `burn_in` as provided.
    //
    // Given
    // -----
    // - A `SimStart::Warm` constructed with a specific `burn_in` length.
    //
    // Expect
    // ------
    // - Pattern-matching on the variant yields the same `burn_in` value.
    fn simstart_warm_stores_burn_in() {
        // Arrange
        let burn_in = 500_usize;
        let sim_start = SimStart::Warm { burn_in };

        // Act + Assert
        match sim_start {
            SimStart::Warm { burn_in: stored } => {
                assert_eq!(stored, burn_in);
            }
            _ => panic!("expected SimStart::Warm variant"),
        }
    }
}
