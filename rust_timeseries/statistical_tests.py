# rust_timeseries/statistical_tests.py

# Grab the compiled extension module…
from . import rust_timeseries as _ext

# …then pull EscancianoLobato out of its `statistical_tests` submodule
EscancianoLobato = _ext.statistical_tests.EscancianoLobato

__all__ = ["EscancianoLobato"]
