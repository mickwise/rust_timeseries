#[cfg(feature = "python-bindings")]
use ndarray::Array1;

#[cfg(feature = "python-bindings")]
use pyo3::{exceptions::PyValueError, prelude::*, types::PyAny};

#[cfg(feature = "python-bindings")]
use crate::{
    duration::{
        core::{
            data::ACDData, guards::PsiGuards, init::Init, options::ACDOptions, shape::ACDShape,
        },
        models::acd::ACDModel,
    },
    inference::{hac::HACOptions, kernel::KernelType},
    optimization::loglik_optimizer::traits::{LineSearcher, MLEOptions, Tolerances},
};

#[cfg(feature = "python-bindings")]
use numpy::{
    IntoPyArray,    // Vec → PyArray
    PyArrayMethods, // .readonly()
    PyReadonlyArray1,
};

#[cfg(feature = "python-bindings")]
#[inline]
pub fn extract_f64_array<'py>(
    py: Python<'py>, raw_data: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, f64>> {
    if let Ok(arr_ro) = raw_data.extract::<PyReadonlyArray1<f64>>() {
        if arr_ro.as_slice().is_ok() {
            return Ok(arr_ro);
        }
    }

    if let Ok(obj) = raw_data.call_method("to_numpy", (false,), None) {
        if let Ok(series_ro) = obj.extract::<PyReadonlyArray1<f64>>() {
            if series_ro.as_slice().is_ok() {
                return Ok(series_ro);
            }
        }
    }

    let vec: Vec<f64> = raw_data.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "expected a 1-D numpy.ndarray, pandas.Series, or sequence of float64",
        )
    })?;
    Ok(vec.into_pyarray(py).readonly())
}

#[cfg(feature = "python-bindings")]
pub fn build_acd_model<'py>(
    py: Python<'py>, data_length: usize,
    innovation: crate::duration::core::innovations::ACDInnovation, p: Option<usize>,
    q: Option<usize>, init: Option<&str>, init_fixed: Option<f64>,
    init_psi_lags: Option<&Bound<'py, PyAny>>, init_durations_lags: Option<&Bound<'py, PyAny>>,
    tol_grad: Option<f64>, tol_cost: Option<f64>, max_iter: Option<usize>,
    line_searcher: Option<&str>, lbfgs_mem: Option<usize>, psi_guards: Option<(f64, f64)>,
) -> PyResult<ACDModel> {
    let p_val = p.unwrap_or(1);
    let q_val = q.unwrap_or(1);

    // Validate shape against in-sample length.
    let shape = ACDShape::new(p_val, q_val, data_length)?;

    // Init policy.
    let init_policy =
        extract_init(py, init, init_fixed, init_psi_lags, init_durations_lags, p_val, q_val)?;

    // Optimizer options.
    let mle_opts = extract_mle_opts(tol_grad, tol_cost, max_iter, line_searcher, lbfgs_mem)?;

    // Psi guards with sensible default.
    let guards_tuple = psi_guards.unwrap_or((1e-6, 1e6));
    let guards = PsiGuards::new(guards_tuple)?;

    let opts = ACDOptions::new(init_policy, mle_opts, guards);

    Ok(ACDModel::new(shape, innovation, opts, data_length))
}

#[cfg(feature = "python-bindings")]
fn extract_init<'py>(
    py: Python<'py>, init: Option<&str>, init_fixed: Option<f64>,
    init_psi_lags: Option<&Bound<'py, PyAny>>, init_durations_lags: Option<&Bound<'py, PyAny>>,
    p: usize, q: usize,
) -> PyResult<Init> {
    use crate::utils::extract_f64_array; // already defined in this module

    let init_str = init.unwrap_or("uncond_mean");

    let policy = match init_str {
        "uncond_mean" => Init::uncond_mean(),
        "sample_mean" => Init::sample_mean(),
        "fixed" => {
            let val = init_fixed.ok_or_else(|| {
                PyValueError::new_err("init_fixed must be provided when init='fixed'")
            })?;
            Init::fixed(val)?
        }
        "fixed_vector" => {
            let psi_any = init_psi_lags.ok_or_else(|| {
                PyValueError::new_err("init_psi_lags must be provided when init='fixed_vector'")
            })?;
            let dur_any = init_durations_lags.ok_or_else(|| {
                PyValueError::new_err(
                    "init_durations_lags must be provided when init='fixed_vector'",
                )
            })?;

            let psi_arr = extract_f64_array(py, psi_any)?;
            let dur_arr = extract_f64_array(py, dur_any)?;

            let psi_slice = psi_arr.as_slice().map_err(|_| {
                PyValueError::new_err(
                    "init_psi_lags must be a 1-D contiguous float64 array or sequence",
                )
            })?;
            let dur_slice = dur_arr.as_slice().map_err(|_| {
                PyValueError::new_err(
                    "init_durations_lags must be a 1-D contiguous float64 array or sequence",
                )
            })?;

            let psi_vec = Array1::from(psi_slice.to_vec());
            let dur_vec = Array1::from(dur_slice.to_vec());

            Init::fixed_vector(psi_vec, dur_vec, p, q)?
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "invalid init policy {:?} (expected 'uncond_mean', 'sample_mean', 'fixed', or 'fixed_vector')",
                other
            )));
        }
    };

    Ok(policy)
}

#[cfg(feature = "python-bindings")]
fn extract_mle_opts(
    tol_grad: Option<f64>, tol_cost: Option<f64>, max_iter: Option<usize>,
    line_searcher: Option<&str>, lbfgs_mem: Option<usize>,
) -> PyResult<MLEOptions> {
    use std::str::FromStr;

    use crate::duration::errors::ACDError;

    // Tolerances::new -> OptResult<Tolerances> -> ACDError -> PyErr
    let tols = Tolerances::new(tol_grad, tol_cost, max_iter).map_err(ACDError::from)?;

    // LineSearcher::from_str -> OptResult<LineSearcher> -> ACDError -> PyErr
    let ls = match line_searcher {
        Some(name) => LineSearcher::from_str(name).map_err(ACDError::from)?,
        None => LineSearcher::MoreThuente,
    };

    // MLEOptions::new -> OptResult<MLEOptions> -> ACDError -> PyErr
    let opts = MLEOptions::new(tols, ls, lbfgs_mem).map_err(ACDError::from)?;

    Ok(opts)
}

#[cfg(feature = "python-bindings")]
pub fn extract_acd_data<'py>(
    py: Python<'py>, durations: &Bound<'py, PyAny>, unit: Option<&str>, t0: Option<usize>,
    diurnal_adjusted: Option<bool>,
) -> PyResult<ACDData> {
    use crate::duration::core::{data::ACDMeta, units::ACDUnit};

    let dur_arr = extract_f64_array(py, durations)?;
    let dur_slice = dur_arr.as_slice().map_err(|_| {
        PyValueError::new_err("durations must be a 1-D contiguous float64 array or sequence")
    })?;
    let dur_vec = Array1::from(dur_slice.to_vec());
    let unit_str = unit.unwrap_or("seconds").to_lowercase();
    let acd_unit = match unit_str.as_str() {
        "seconds" | "s" => ACDUnit::Seconds,
        "milliseconds" | "ms" => ACDUnit::Milliseconds,
        "microseconds" | "us" => ACDUnit::Microseconds,
        other => {
            return Err(PyValueError::new_err(format!(
                "invalid unit {:?} (expected 'seconds', 'milliseconds', or 'microseconds')",
                other
            )));
        }
    };
    let diurnal_flag = diurnal_adjusted.unwrap_or(false);
    let meta = ACDMeta::new(acd_unit, None, diurnal_flag);
    let model = ACDData::new(dur_vec, t0, meta);
    match model {
        Ok(data) => Ok(data),
        Err(e) => Err(e.into()),
        
    }
}

#[cfg(feature = "python-bindings")]
pub fn extract_hac_options(
    kernel: Option<&str>, bandwidth: Option<usize>, center: Option<bool>,
    small_sample_correction: Option<bool>,
) -> PyResult<HACOptions> {
    let kernel_str = kernel.unwrap_or("bartlett").to_lowercase();
    let kernel_type = match kernel_str.as_str() {
        "iid" => KernelType::IID,
        "bartlett" | "newey_west" => KernelType::Bartlett,
        "parzen" => KernelType::Parzen,
        "quadratic_spectral" | "quadraticspectral" | "qs" => KernelType::QuadraticSpectral,
        other => {
            return Err(PyValueError::new_err(format!(
                "invalid HAC kernel {:?} (expected 'iid', 'bartlett', 'parzen', or 'quadratic_spectral')",
                other
            )));
        }
    };

    let center_val = center.unwrap_or(false);
    let ssc_val = small_sample_correction.unwrap_or(true);

    Ok(HACOptions::new(bandwidth, kernel_type, center_val, ssc_val))
}
