//! Inference utilities for Hessians: symmetrization, variance estimation, and SEs.
//!
//! What this module provides
//! - Helpers to symmetrize numerically computed Hessians.
//! - Conversion between `ndarray` Hessians and `nalgebra` matrices for decompositions.
//! - Variance/SE estimation from the observed information:
//!   - Classical: inverse Hessian (or pseudoinverse if singular).
//!   - Robust: combine pseudoinverse with a score covariance matrix.
//!
//! Implementation notes
//! - Decompositions use `nalgebra::DMatrix` (column-major).
//! - For singular Hessians, eigenvalues below `EIGEN_EPS` are clipped to zero.
use crate::optimization::{
    errors::OptResult, loglik_optimizer::finite_diff::compute_hessian,
    numerical_stability::transformations::EIGEN_EPS,
};
use nalgebra::DMatrix;
use ndarray::{Array1, Array2};

/// Compute standard errors from the observed information, with optional robust (sandwich) correction.
///
/// Strategy:
/// - Always build the observed information `J(θ̂)` using `compute_hessian`
///   (based on the average log-likelihood).
/// - Symmetrize and copy into a `nalgebra::DMatrix` for stable linear algebra.
/// - Compute SEs using one of two paths:
///
///   *Classical (no scores provided)*
///   - Eigendecompose `J = Q Λ Qᵀ`.
///   - For each parameter `i`, set
///     `Var(θ̂_i) = Σ_{k: λ_k > EIGEN_EPS} Q[i,k]^2 / λ_k`.
///   - This is the diagonal of `J⁻¹` if PD, or of the Moore–Penrose pseudoinverse `J⁺` if singular.
///
///   *Robust (scores provided)*
///   - Use the same eigendecomposition to form pseudoinverse vectors
///     `w_i = J⁺ e_i`.
///   - With the user-supplied score covariance matrix `S`, compute
///     `Var(θ̂_i) = w_iᵀ S w_i` (the sandwich variance).
///   - This accommodates model misspecification and serial dependence (e.g., via HAC).
///
/// Returns:
/// - A length-`n` vector of standard errors.
///
/// Notes:
/// - `scores` must be the *p×p score covariance matrix* (OPG or HAC), not raw per-observation scores.
/// - Eigenvalues below `EIGEN_EPS` are treated as zero.
/// - No explicit matrix inverse is formed; all solves use eigendecomposition.
/// - Conservative: weakly identified parameters yield inflated variances.
pub fn calc_standard_errors<F: Fn(&Array1<f64>) -> Array1<f64>>(
    f: &F, theta_hat: &Array1<f64>, scores: Option<&Array2<f64>>,
) -> OptResult<Array1<f64>> {
    let n = theta_hat.len();
    let obs_info = compute_hessian(f, theta_hat)?;
    let mut obs_info_nalg = DMatrix::<f64>::zeros(obs_info.nrows(), obs_info.ncols());
    fill_dmatrix(&obs_info, &mut obs_info_nalg);
    match scores {
        Some(s) => Ok(solve_for_se_robust(obs_info_nalg, s, n)),
        None => Ok(solve_for_se(obs_info_nalg, n)),
    }
}

// ---- Helper methods ----

/// Copy a symmetric `ndarray` matrix into a `nalgebra::DMatrix` with column-first writes.
///
/// This routine fills the destination in **column-major order**, which matches
/// the internal storage of `nalgebra::DMatrix`. Writing down each column before
/// moving to the next improves cache locality and avoids unnecessary memory
/// jumps during the copy. The result is more efficient than a naive row-first
/// traversal, especially when `n` grows.
///
/// For each column `j`:
/// - Copy the diagonal entry once (`(j, j)`).
/// - For each row `i > j`, copy the off-diagonal element into both `(i, j)` and `(j, i)`.
fn fill_dmatrix(obs_info: &Array2<f64>, obs_info_nalg: &mut DMatrix<f64>) -> () {
    let n = obs_info.ncols();
    for j in 0..n {
        for i in j..n {
            if j == i {
                obs_info_nalg[(i, i)] = obs_info[[i, i]];
            } else {
                obs_info_nalg[(i, j)] = obs_info[[i, j]];
                obs_info_nalg[(j, i)] = obs_info[[j, i]];
            }
        }
    }
}

/// Compute standard errors from a (possibly singular) observed information matrix.
///
/// Uses the spectral form:
/// - `J = Q Λ Qᵀ` (symmetric eigen).
/// - `Var(θ̂_i) = Σ_{k: λ_k > EIGEN_EPS} Q[i,k]^2 / λ_k`.
///
/// This implements the diagonal of the Moore–Penrose pseudoinverse when
/// some `λ_k` are numerically nonpositive. Returns the vector of SEs.
///
/// Notes:
/// - Ensure indexing uses the original eigen indices: enumerate **before**
///   filtering by `EIGEN_EPS`.
fn solve_for_se(obs_info_nalg: DMatrix<f64>, n: usize) -> Array1<f64> {
    let eigen_decomp = obs_info_nalg.symmetric_eigen();
    let mut se = Array1::<f64>::zeros(n);
    let q = eigen_decomp.eigenvectors;
    let eigenvals = eigen_decomp.eigenvalues;
    for i in 0..n {
        se[i] = eigenvals
            .iter()
            .enumerate()
            .filter(|(_, lambda)| **lambda > EIGEN_EPS)
            .map(|(k, &lambda)| q[(i, k)] * q[(i, k)] / lambda)
            .sum();
        se[i] = se[i].sqrt();
    }
    se
}

/// Robust (sandwich) variances using eigen/pseudoinverse of the observed information.
///
/// Inputs:
/// - `obs_info_nalg`: symmetric observed information H (of the average log-likelihood).
/// - `scores`: score covariance S (p×p) for the average score (IID OPG or HAC), **not** raw T×p scores.
/// - `n`: parameter dimension.
///
/// For each parameter i:
///   1) Build w_i = H^+ e_i via the eigendecomposition H = Q Λ Qᵀ with λ_k ≤ ε clipped.
///   2) var_i = w_iᵀ S w_i.
/// Returns the vector of variances (sqrt externally if SEs are desired).
fn solve_for_se_robust(obs_info_nalg: DMatrix<f64>, scores: &Array2<f64>, n: usize) -> Array1<f64> {
    let eigen_decomp = obs_info_nalg.symmetric_eigen();
    let mut se = Array1::<f64>::zeros(n);
    let q = eigen_decomp.eigenvectors;
    let eigenvals = eigen_decomp.eigenvalues;
    for i in 0..n {
        let mut w_i = Array1::zeros(n);
        for (k, &lambda) in eigenvals.iter().enumerate() {
            if lambda > EIGEN_EPS {
                let coeff = q[(i, k)] / lambda;
                for j in 0..n {
                    w_i[j] += coeff * q[(j, k)];
                }
            }
        }
        se[i] = w_i.t().dot(scores).dot(&w_i);
        se[i] = se[i].sqrt();
    }
    se
}
