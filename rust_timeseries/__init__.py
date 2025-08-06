# rust_timeseries/__init__.py

# Import the extension module (maturin will place a file named rust_timeseries.cpython-*.so here)
from . import rust_timeseries as _ext

# Make the submodule visible (we expose a real Python file below)
from . import statistical_tests

__all__ = ["statistical_tests"]
