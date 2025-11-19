//! loglik_optimizer::types — shared numeric aliases and solver wiring.
//!
//! Purpose
//! -------
//! Centralize the core numeric types and solver aliases used by the
//! log-likelihood optimizer. By defining these in one place, the rest of
//! the optimization code can stay agnostic to `ndarray` and Argmin
//! generics and can more easily evolve if the backend changes.
//!
//! Key behaviors
//! -------------
//! - Define canonical aliases for parameter vectors, gradients,
//!   Hessians, and scalar costs (`Theta`, `Grad`, `Hessian`, `Cost`).
//! - Provide a standard map type for Argmin function-evaluation counters
//!   (`FnEvalMap`).
//! - Expose pre-wired L-BFGS solver aliases for different line-search
//!   strategies, using the common `(Theta, Grad, Cost)` numeric shapes.
//!
//! Invariants & assumptions
//! ------------------------
//! - All optimizer vectors and matrices are represented as `ndarray`
//!   containers over `f64`.
//! - `Cost` is always a scalar `f64` in log-likelihood space; higher
//!   layers handle any sign flips between cost and log-likelihood.
//! - The line-search aliases assume Argmin’s three-parameter forms
//!   `(Param, Gradient, Float)` as of the pinned Argmin version.
//!
//! Conventions
//! -----------
//! - `Theta` and `Grad` are treated conceptually as column vectors with
//!   length equal to the number of free parameters.
//! - `Hessian` is a dense square matrix with dimension
//!   `theta.len() × theta.len()` when used.
//! - `DEFAULT_LBFGS_MEM` encodes the typical history size for L-BFGS;
//!   callers may override this via per-run options.
//! - This module defines no runtime behavior beyond what `ndarray` and
//!   Argmin require when these types are instantiated elsewhere.
//!
//! Downstream usage
//! ----------------
//! - Other optimizer modules import these aliases instead of referring
//!   directly to `ndarray` or Argmin generics.
//! - High-level APIs use [`Theta`] and [`Grad`] as the standard parameter
//!   and gradient types when building log-likelihood adapters.
//! - Solver wrappers construct concrete L-BFGS instances via the
//!   provided solver aliases (e.g., [`LbfgsHagerZhang`]) based on a
//!   chosen line search.
//!
//! Testing notes
//! -------------
//! - This module only defines type aliases and constants; there are no
//!   dedicated unit tests.
//! - Correctness is exercised indirectly by tests in the surrounding
//!   optimizer modules that instantiate solvers and operate on these
//!   aliases.
use argmin::solver::{
    linesearch::{HagerZhangLineSearch, MoreThuenteLineSearch},
    quasinewton::LBFGS,
};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Parameter vector `θ` for log-likelihood optimization.
///
/// Alias for `ndarray::Array1<f64>`, used as the canonical parameter type
/// throughout the optimizer.
pub type Theta = Array1<f64>;

/// Gradient vector `∇ℓ(θ)` or `∇c(θ)` for optimization.
///
/// Alias for `ndarray::Array1<f64>`, matching the shape of `Theta`.
pub type Grad = Array1<f64>;

/// Dense Hessian matrix for second-order information.
///
/// Alias for `ndarray::Array2<f64>`; `n × n` for `n = Theta.len()`.
pub type Hessian = Array2<f64>;

/// Scalar objective value used by the optimizer.
///
/// In this crate, this is the cost `c(θ) = -ℓ(θ)` derived from a
/// log-likelihood `ℓ(θ)`.
pub type Cost = f64;

/// Function-evaluation counters as reported by the solver.
///
/// Maps human-readable counter names (e.g., `"cost_count"`) to counts.
pub type FnEvalMap = HashMap<String, u64>;

/// Default history size (`m`) for L-BFGS runs.
pub const DEFAULT_LBFGS_MEM: usize = 7;

/// Hager–Zhang line search specialized to this crate’s numeric types.
pub type HagerZhangLS = HagerZhangLineSearch<Theta, Grad, Cost>;

/// More–Thuente line search specialized to this crate’s numeric types.
pub type MoreThuenteLS = MoreThuenteLineSearch<Theta, Grad, Cost>;

/// L-BFGS solver wired to the Hager–Zhang line search.
pub type LbfgsHagerZhang = LBFGS<HagerZhangLS, Theta, Grad, Cost>;

/// L-BFGS solver wired to the More–Thuente line search.
pub type LbfgsMoreThuente = LBFGS<MoreThuenteLS, Theta, Grad, Cost>;
