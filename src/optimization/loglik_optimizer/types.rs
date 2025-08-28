//! Common type aliases for the log-likelihood optimizer.
//!
//! Note: These line-search aliases match argmin 0.10’s 3-parameter forms
//! (Param, Gradient, Float).

use argmin::solver::{
    linesearch::{HagerZhangLineSearch, MoreThuenteLineSearch},
    quasinewton::LBFGS,
};
use ndarray::Array1;
use std::collections::HashMap;

// Core numeric types
pub type Theta = Array1<f64>;
pub type Grad = Array1<f64>;
pub type Cost = f64;

// Function-evaluation counts as returned by Argmin
pub type FnEvalMap = HashMap<String, u64>;

// Typical L-BFGS memory (a.k.a. history size)
pub const LBFGS_MEM: usize = 7;

// Line-search flavors
pub type HZ = HagerZhangLineSearch<Theta, Grad, Cost>;
pub type MT = MoreThuenteLineSearch<Theta, Grad, Cost>;

// L-BFGS configured with each line search
pub type LbfgsHz = LBFGS<HZ, Theta, Grad, Cost>;
pub type LbfgsMt = LBFGS<MT, Theta, Grad, Cost>;
