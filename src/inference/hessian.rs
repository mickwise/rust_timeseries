//! inference::hessian — Hessian-based variance and standard error utilities.
//!
//! Purpose
//! -------
//! Provide a thin wrapper around finite-difference Hessians that converts
//! them into numerically stable variance and standard error estimates.
//! This module handles conversion between `ndarray` and `nalgebra` types
//! and supports both classical and robust (sandwich) SEs built from the
//! observed information and a score covariance matrix.
//!
//! Key behaviors
//! -------------
//! - Call [`compute_hessian`] on an average log-likelihood gradient to
//!   obtain the observed information matrix `J(θ̂)`.
//! - Copy the resulting `ndarray` Hessian into a `nalgebra::DMatrix`
//!   (`fill_dmatrix`) for eigen-based linear algebra.
//! - Compute classical standard errors from the Moore–Penrose
//!   pseudoinverse of `J(θ̂)`.
//! - Compute robust (sandwich) standard errors when supplied with a
//!   score covariance matrix `S` (IID OPG or HAC) on the average-score
//!   scale.
//!
//! Invariants & assumptions
//! ------------------------
//! - [`compute_hessian`] returns a finite, square `n×n` matrix with
//!   `n = θ̂.len()`. Symmetry is already enforced upstream via
//!   `symmetrize_hess`; this module does **not** re-symmetrize.
//! - The matrix passed into [`solve_for_se`] and
//!   [`solve_for_se_robust`] is treated as symmetric for the purposes
//!   of `symmetric_eigen`.
//! - When provided, `scores` is an `n×n` symmetric score covariance
//!   matrix for the *average* score, with `n = θ̂.len()`.
//! - Eigenvalues with magnitude at most [`EIGEN_EPS`] are treated as
//!   numerically nonpositive and ignored when constructing pseudoinverse
//!   directions.
//!
//! Conventions
//! -----------
//! - All Hessians are on the **average log-likelihood** scale (not the
//!   sum), so the resulting variances/SEs correspond to that scaling.
//! - Standard errors are returned as the square roots of diagonal
//!   variances; no full covariance matrix is currently exposed by this
//!   module.
//! - No explicit matrix inverse is formed; all computations use
//!   symmetric eigendecomposition with eigenvalue truncation.
//! - Errors are reported via [`OptResult<T>`].
//!
//! Downstream usage
//! ----------------
//! - Model layers (e.g., ACD models) call [`calc_standard_errors`] after
//!   fitting to obtain classical or robust SEs at the MLE.
//! - Robust SEs use a score covariance matrix `S` produced by
//!   `inference::hac` (e.g., via
//!   `inference::hac::calculate_avg_scores_cov`) and threaded through
//!   model-side inference options.
//! - Helper routines [`fill_dmatrix`], [`solve_for_se`], and
//!   [`solve_for_se_robust`] are internal utilities; library users
//!   should not need to invoke them directly.
//!
//! Testing notes
//! -------------
//! - Unit tests in this module cover:
//!   - Correct copying of Hessians from `ndarray` into `DMatrix`
//!     without altering symmetry.
//!   - Agreement between classical SEs and the diagonal of an analytic
//!     `J⁺` for simple quadratic objectives.
//!   - Robust SE behavior when `scores` is chosen to inflate variances
//!     relative to the classical case.
//! - Integration tests at the model layer verify that:
//!   - SEs behave sensibly under reparameterizations.
//!   - Weak identification (near-zero eigenvalues) produces inflated
//!     SEs along those directions.
use crate::optimization::{
    errors::OptResult, loglik_optimizer::finite_diff::compute_hessian,
    numerical_stability::transformations::EIGEN_EPS,
};
use nalgebra::DMatrix;
use ndarray::{Array1, Array2};

/// calc_standard_errors — standard errors from observed information.
///
/// Purpose
/// -------
/// Compute classical or robust (sandwich) standard errors from the observed
/// information matrix `J(θ̂)`, using eigen-based pseudoinverses. The observed
/// information is built via finite-difference Hessians of an average
/// log-likelihood gradient, then decomposed to produce per-parameter SEs.
///
/// Parameters
/// ----------
/// - `f`: `&F`
///   Gradient map of the **average** log-likelihood or negative
///   average log-likelihood, `f: θ ↦ g(θ)`. f must be C¹ in a
///   neighborhood of `theta_hat` so that [`compute_hessian`] can
///   succeed.
/// - `theta_hat`: `&Array1<f64>`
///   Parameter vector `θ̂` at which the observed information is
///   evaluated. Its length `n` determines the dimension of the Hessian
///   and of the returned SE vector.
/// - `scores`: `Option<&Array2<f64>>`
///   Optional `n×n` score covariance matrix `S` on the *average-score*
///   scale. When `None`, classical SEs are computed from `J(θ̂)` alone.
///   When `Some(S)`, robust (sandwich) SEs are computed via
///   `Var(θ̂_i) = w_iᵀ S w_i`, where `w_i` is the `i`-th column of the
///   pseudoinverse `J⁺`.
///
/// Returns
/// -------
/// `OptResult<Array1<f64>>`
///   On success, a length-`n` vector of standard errors corresponding to
///   the entries of `theta_hat`. On failure, propagates the error from
///   [`compute_hessian`] (e.g., invalid Hessian, non-finite entries).
///
/// Errors
/// ------
/// - `OptError`  
///   Any error that [`compute_hessian`] may return, such as Hessian
///   dimension mismatches or non-finite entries detected by validation.
///
/// Panics
/// ------
/// - Never panics under the documented invariants. Any internal panic
///   would indicate a programming error (e.g., dimension mismatch
///   between `theta_hat` and `scores` in upstream code).
///
/// Safety
/// ------
/// - No `unsafe` code is used. Callers must ensure that, when provided,
///   `scores` is `n×n` with `n = theta_hat.len()` and matches the same
///   parameter ordering as `theta_hat`.
///
/// Notes
/// -----
/// - The sign convention of `f` (log-likelihood vs negative
///   log-likelihood) is absorbed into `compute_hessian`; `J(θ̂)` is
///   interpreted as the observed information matrix on the average
///   log-likelihood scale.
/// - Eigenvalues with magnitude at most [`EIGEN_EPS`] are treated as
///   zero when forming pseudoinverse directions, inflating SEs along
///   weakly identified directions.
/// - The robust branch does **not** change the point estimate `θ̂`; it
///   only modifies the uncertainty quantification around it.
///
/// Examples
/// --------
/// ```rust
/// # use ndarray::array;
/// # use rust_timeseries::inference::hessian::calc_standard_errors;
/// # use rust_timeseries::optimization::errors::OptResult;
/// #
/// // Simple quadratic: g(θ) = A θ, where A is PD.
/// let a = array![[4.0, 0.0],
///                [0.0, 1.0]];
/// let f = |theta: &ndarray::Array1<f64>| -> ndarray::Array1<f64> {
///     a.dot(theta)
/// };
/// let theta_hat = array![1.0, -1.0];
///
/// let se: OptResult<ndarray::Array1<f64>> =
///     calc_standard_errors(&f, &theta_hat, None);
/// assert!(se.is_ok());
/// let se = se.unwrap();
/// assert_eq!(se.len(), 2);
/// // For diagonal A, classical SEs are ~[1/sqrt(4), 1/sqrt(1)].
/// assert!((se[0] - 0.5).abs() < 1e-6);
/// assert!((se[1] - 1.0).abs() < 1e-6);
/// ```
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

/// fill_dmatrix — copy an `ndarray` Hessian into a `nalgebra::DMatrix`.
///
/// Purpose
/// -------
/// Bridge between `ndarray` and `nalgebra` by copying a square observed
/// information matrix into a `DMatrix<f64>` using column-major writes.
/// This helper does **not** modify symmetry; it assumes that the input
/// has already been symmetrized upstream (e.g., by [`compute_hessian`]).
///
/// Parameters
/// ----------
/// - `obs_info`: `&Array2<f64>`
///   Square `n×n` observed information matrix in `ndarray` form. Expected
///   to be symmetric up to numerical precision.
/// - `obs_info_nalg`: `&mut DMatrix<f64>`
///   Preallocated `n×n` `DMatrix` that will receive the contents of
///   `obs_info`. Must have the same dimensions as `obs_info`.
///
/// Returns
/// -------
/// `()`
///   Mutates `obs_info_nalg` in place; no value is returned.
///
/// Errors
/// ------
/// - `None`  
///   This helper does not return a `Result`. Dimension mismatches are
///   considered programmer errors.
///
/// Panics
/// ------
/// - May panic if `obs_info` and `obs_info_nalg` have inconsistent shapes,
///   due to out-of-bounds indexing.
///
/// Safety
/// ------
/// - No `unsafe` code is used. Callers must ensure that both matrices are
///   `n×n` with matching `n`.
///
/// Notes
/// -----
/// - The copy proceeds column by column, matching the internal storage of
///   `DMatrix` (column-major) and improving cache locality compared to a
///   row-major traversal.
/// - No symmetrization is performed here; any asymmetry present in
///   `obs_info` will be preserved in `obs_info_nalg`.
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

/// solve_for_se — classical standard errors from observed information.
///
/// Purpose
/// -------
/// Compute classical standard errors from a symmetric observed information
/// matrix `J(θ̂)` using symmetric eigendecomposition and eigenvalue
/// truncation. This corresponds to taking the square root of the diagonal
/// of the Moore–Penrose pseudoinverse `J⁺`.
///
/// Parameters
/// ----------
/// - `obs_info_nalg`: `DMatrix<f64>`
///   Symmetric `n×n` observed information matrix, typically produced by
///   [`fill_dmatrix`]. Consumed by the eigendecomposition.
/// - `n`: `usize`
///   Parameter dimension (length of `θ̂`). Must match the dimension of
///   `obs_info_nalg`.
///
/// Returns
/// -------
/// `Array1<f64>`
///   Length-`n` vector of classical standard errors `SE(θ̂_i)` for each
///   parameter.
///
/// Errors
/// ------
/// - `None`  
///   This helper does not return a `Result`. Numerical breakdown (e.g.,
///   NaNs from `symmetric_eigen`) would indicate a deeper issue upstream.
///
/// Panics
/// ------
/// - May panic if `obs_info_nalg` is not square or if its dimension does
///   not match `n`. Such mismatches are considered programmer errors.
///
/// Safety
/// ------
/// - No `unsafe` code is used.
///
/// Notes
/// -----
/// - Eigenvalues `λ_k` with `λ_k ≤ EIGEN_EPS` are treated as zero and
///   excluded from the variance sum, which inflates SEs along weakly
///   identified directions.
/// - The implemented formula is:
///   `Var(θ̂_i) = Σ_{k: λ_k > EIGEN_EPS} Q[i,k]^2 / λ_k`, and the return
///   value is `sqrt(Var(θ̂_i))` for each `i`.
/// - Here Q is the orthonormal matrix of eigenvectors from the
///   symmetric eigendecomposition J = Q Λ Qᵀ
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

/// solve_for_se_robust — robust (sandwich) standard errors.
///
/// Purpose
/// -------
/// Compute robust (sandwich) standard errors by combining the pseudoinverse
/// of the observed information matrix `H = J(θ̂)` with a score covariance
/// matrix `S` on the average-score scale. For each parameter `i`, the
/// routine constructs `w_i = H⁺ e_i` and computes
/// `Var(θ̂_i) = w_iᵀ S w_i`, returning `sqrt(Var(θ̂_i))`.
///
/// Parameters
/// ----------
/// - `obs_info_nalg`: `DMatrix<f64>`
///   Symmetric `n×n` observed information matrix on the average
///   log-likelihood scale. Consumed by the eigendecomposition.
/// - `scores`: `&Array2<f64>`
///   `n×n` score covariance matrix `S` on the *average-score* scale
///   (IID OPG or HAC). Must be symmetric and conformable with
///   `obs_info_nalg`.
/// - `n`: `usize`
///   Parameter dimension; must match the sizes of `obs_info_nalg` and
///   `scores`.
///
/// Returns
/// -------
/// `Array1<f64>`
///   Length-`n` vector of robust standard errors `SE_robust(θ̂_i)`.
///
/// Errors
/// ------
/// - `None`  
///   This helper does not return a `Result`. Numerical breakdown manifests
///   as NaNs or panics inside the underlying linear algebra and indicates
///   a deeper issue upstream.
///
/// Panics
/// ------
/// - May panic if `scores` does not have shape `n×n` or if
///   `obs_info_nalg` is not `n×n`. Such cases are considered programmer
///   errors.
///
/// Safety
/// ------
/// - No `unsafe` code is used. Callers are responsible for providing
///   conformable, finite matrices.
///
/// Notes
/// -----
/// - Eigenvalues `λ_k ≤ EIGEN_EPS` are discarded when constructing
///   pseudoinverse columns `w_i`, which protects against division by very
///   small eigenvalues while inflating variances along nearly flat
///   directions.
/// - This function computes **standard errors directly** (square roots of
///   variances); callers should not apply an additional square root.
/// - In well-specified cases where `S` is proportional to `J⁻¹`, robust
///   SEs coincide with classical SEs up to that proportionality constant.
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    use ndarray::{Array1, Array2, array};

    // -------------------------------------------------------------------------
    // Scope
    // -----
    // These tests cover:
    // - Correct copying of Hessians from `ndarray` into `DMatrix`.
    // - Classical SEs for simple quadratic objectives with known analytic
    //   information matrices.
    // - Robust SEs when the score covariance inflates variances relative
    //   to the classical case.
    //
    // They intentionally DO NOT cover:
    // - End-to-end ACD model inference or HAC bandwidth selection.
    // - Pathological cases where `compute_hessian` itself fails.
    // -------------------------------------------------------------------------

    #[test]
    // Purpose
    // -------
    // Verify that `fill_dmatrix` copies entries from an `ndarray` Hessian
    // into a `nalgebra::DMatrix` without altering values or symmetry.
    //
    // Given
    // -----
    // - A small 2×2 symmetric `Array2<f64>` with distinct entries.
    //
    // Expect
    // ------
    // - The corresponding `DMatrix` has identical entries at all positions.
    fn fill_dmatrix_copies_ndarray_into_dmatrix_without_modification() {
        // Arrange
        let obs_info: Array2<f64> = array![[2.0, 0.5], [0.5, 1.0]];
        let mut obs_info_nalg = DMatrix::<f64>::zeros(2, 2);

        // Act
        fill_dmatrix(&obs_info, &mut obs_info_nalg);

        // Assert
        assert_eq!(obs_info_nalg[(0, 0)], 2.0);
        assert_eq!(obs_info_nalg[(0, 1)], 0.5);
        assert_eq!(obs_info_nalg[(1, 0)], 0.5);
        assert_eq!(obs_info_nalg[(1, 1)], 1.0);
    }

    #[test]
    // Purpose
    // -------
    // Check that `calc_standard_errors` produces classical SEs equal to the
    // diagonal of the analytic pseudoinverse for a simple diagonal quadratic.
    //
    // Given
    // -----
    // - A diagonal information matrix A = diag(4, 1) encoded via a linear
    //   gradient map g(θ) = A θ.
    // - A generic θ̂ (its value is irrelevant for a constant Hessian).
    //
    // Expect
    // ------
    // - Classical SEs are approximately [1/sqrt(4), 1/sqrt(1)] = [0.5, 1.0].
    fn calc_standard_errors_diagonal_quadratic_matches_analytic_se() {
        // Arrange
        let a = array![[4.0, 0.0], [0.0, 1.0]];
        let f = |theta: &Array1<f64>| -> Array1<f64> { a.dot(theta) };
        let theta_hat = array![1.0, -1.0];

        // Act
        let se_res: OptResult<Array1<f64>> = calc_standard_errors(&f, &theta_hat, None);

        // Assert
        assert!(se_res.is_ok());
        let se = se_res.unwrap();
        assert_eq!(se.len(), 2);
        assert!((se[0] - 0.5).abs() < 1e-6);
        assert!((se[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    // Purpose
    // -------
    // Verify that providing a score covariance matrix with larger variance
    // inflates robust SEs relative to classical SEs.
    //
    // Given
    // -----
    // - An identity information matrix H = I_2 encoded directly as a
    //   `DMatrix<f64>`.
    // - A score covariance S = 2 I_2, representing doubled variance.
    //
    // Expect
    // ------
    // - Classical SEs are approximately [1.0, 1.0].
    // - Robust SEs are approximately [sqrt(2), sqrt(2)] > classical SEs.
    fn solve_for_se_robust_inflates_se_relative_to_classical_when_scores_are_larger() {
        // Arrange
        let h = DMatrix::<f64>::from_diagonal(&DVector::from_vec(vec![1.0, 1.0]));
        let scores: Array2<f64> = array![[2.0, 0.0], [0.0, 2.0]];
        let n = 2;

        // Classical SEs with H = I_2
        let se_classical = solve_for_se(h.clone(), n);

        // Robust SEs with S = 2 I_2
        let se_robust = solve_for_se_robust(h, &scores, n);

        // Assert
        assert_eq!(se_classical.len(), 2);
        assert_eq!(se_robust.len(), 2);

        // Classical: Var = 1, SE = 1
        assert!((se_classical[0] - 1.0).abs() < 1e-8);
        assert!((se_classical[1] - 1.0).abs() < 1e-8);

        // Robust: Var = 2, SE = sqrt(2)
        let sqrt2 = 2.0_f64.sqrt();
        assert!((se_robust[0] - sqrt2).abs() < 1e-8);
        assert!((se_robust[1] - sqrt2).abs() < 1e-8);

        // And robust SEs should strictly dominate classical in this setup.
        assert!(se_robust[0] > se_classical[0]);
        assert!(se_robust[1] > se_classical[1]);
    }
}
