use numpy::{
    IntoPyArray,    // Vec → PyArray
    PyArrayMethods, // .readonly()
    PyReadonlyArray1,
};
use pyo3::{prelude::*, types::PyAny};

/// Always returns a *contiguous* `float64` NumPy column vector.
/// Copies only when contiguity cannot be guaranteed.
#[inline]
pub fn extract_f64_array<'py>(
    py: Python<'py>,
    raw_data: &Bound<'py, PyAny>,
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
