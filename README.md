# rust_timeseries

[PyPI](https://img.shields.io/pypi/v/rust_timeseries)
[CI](https://github.com/your-org/rust_timeseries/actions/workflows/ci.yml/badge.svg)

**rust_timeseries** is a high-performance **Python** library for time-series diagnostics.  
The first release implements the Escanciano–Lobato (2009) robust automatic portmanteau test for serial dependence.  
Heavy lifting is handled in Rust (via [PyO3]) so you get C-level speed with a pure-Python interface.

---

## ✨ Highlights

| Why it matters           | What you get                                                                    |
|--------------------------|---------------------------------------------------------------------------------|
| **Native-code core**     | Tight Rust loops compiled to machine code                                       |
| **Zero-copy I/O**        | `numpy.ndarray` / `pandas.Series` buffers are viewed directly—no copying ever   |
| **Heteroskedastic-robust** | τ̂-adjusted autocorrelations maintain validity under conditional heteroskedasticity |
| **Automatic lag choice** | Data-driven \(p̃\) maximises the penalised statistic \(L_p\)                    |
| **Friendly errors**      | Clear `ValueError` / `OSError` when inputs are invalid                          |

---

## 📦 Installation

```bash
pip install rust_timeseries
```

Binary wheels are provided for Python 3.9–3.13 on Linux x86-64, macOS (Intel & Apple Silicon) and Windows 64-bit.  
If no wheel matches your platform a source install will build automatically—just have **Rust 1.76+** on `PATH`.

---

## 🚀 Quick start

```python
import rust_timeseries as rts
import numpy as np

y = np.random.randn(500)

test = rts.statistical_tests.EscancianoLobato(y, q=3.5)  # d defaults to ⌊n**0.2⌋
print(f"Q*      = {test.statistic:.3f}")
print(f"p̃       = {test.p_tilde}")
print(f"p-value = {test.pvalue:.4f}")
```

### API snapshot

| Object               | Attribute     | Meaning                                       |
|----------------------|---------------|-----------------------------------------------|
| `EscancianoLobato`   | `.statistic`  | Robust Box–Pierce statistic \(Q^{*}_{p̃}\)    |
|                      | `.pvalue`     | Asymptotic χ² (1) tail probability            |
|                      | `.p_tilde`    | Data-driven lag \(p̃\)                         |

Constructor signature

```
EscancianoLobato(data, /, *, q=2.4, d=None)
```

---

## ⚙️ How it works

All numerics live in safe Rust (`src/`), compiled into a shared library and imported by Python.  
The Rust crate is **internal**; no stable Rust API is promised.

---

## 🛠 Development setup

```bash
git clone https://github.com/your-org/rust_timeseries
cd rust_timeseries
python -m venv .venv && source .venv/bin/activate
pip install -U pip maturin
maturin develop --features pyo3/extension-module
pytest
```

---

## 📜 License

Released under the **MIT License** – free for commercial and academic use.

---

## 📖 Reference

Escanciano, J. C. & Lobato, I. N. (2009). *Testing serial correlation in time series with missing observations.* **Journal of Econometrics 150**, 209–225.

[PyO3]: https://pyo3.rs
