//! # Optimization module (MLE-friendly, argmin-powered)
//!
//! High-level, ergonomic wrapper around `argmin` for **maximizing log-likelihoods**
//! from Rust or Python. Users implement a single trait, [`LogLikelihood`], and call
//! [`maximize`]. Internally we **minimize −ℓ(θ)** using L-BFGS with a configurable
//! line search.
//!
//! ## Sign conventions (important)
//! - You implement **ℓ(θ)** and **∇ℓ(θ)** *as-is* (no negation).
//! - The optimizer internally minimizes **−ℓ(θ)** and flips signs as needed.
//!
//! ## Gradients
//! - If you implement `grad`, it’s used directly (and validated).
//! - If you don’t, a finite-difference gradient of the **cost** (−ℓ) is computed.
//! - If your analytic gradient fails validation, we **error** (we do *not* silently
//!   fall back to FD).
//!
//! ## Tolerances & defaults
//! See [`Tolerances`] and [`MLEOptions`]. Sensible defaults are provided to make
//! Python interop trivial.
//!
//! ## Function evaluation counts
//! We expose `func_counts` from `argmin`’s state so you can inspect how often the
//! cost, gradient, etc. were evaluated. Keys follow the pattern
//! `"<thing>_count"`—for example: `"cost_count"`, `"gradient_count"`,
//! `"hessian_count"`, `"jacobian_count"`, `"apply_count"`, `"anneal_count"`.
//!
//! ## Verbose mode
//! With `verbose = true`, observers print per-iteration progress.
//! The **initial** ℓ(θ₀) and ‖∇ℓ(θ₀)‖ are also logged once before iteration 1 to
//! diagnose bad initializations.
//!
//! ## What you get back
//! [`OptimOutcome`] standardizes solver output: best parameters θ̂, best value ℓ(θ̂),
//! gradient norm (if available), iteration counts, and a clear convergence/termination
//! status string.
//!
//! ---
//! Re-exports: [`maximize`], [`LogLikelihood`], [`LineSearcher`], [`Tolerances`],
//! [`MLEOptions`], [`OptimOutcome`].

pub mod adapter;
pub mod api;
pub mod builders;
pub mod run;
pub mod traits;
pub mod types;
pub mod validation;

// ---- Re-exports ----
pub use self::api::maximize;
pub use self::traits::{LogLikelihood, MLEOptions, OptimOutcome, Tolerances};
pub use self::types::{Cost, FnEvalMap, Grad, LBFGS_MEM, Theta};

// Optional convenience prelude for downstream crates:
// use `loglik_optimizer::prelude::*;`
pub mod prelude {
    pub use super::api::maximize;
    pub use super::traits::{LogLikelihood, MLEOptions, OptimOutcome, Tolerances};
    pub use super::types::{Cost, Grad, Theta};
}
