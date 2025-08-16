# rust_timeseries
[[PyPI version](https://img.shields.io/pypi/v/rust-timeseries.svg)](https://pypi.org/project/rust-timeseries/)
[[License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[[CI](https://github.com/mickwise/rust_timeseries/actions/workflows/ci.yml/badge.svg)](https://github.com/mickwise/rust_timeseries/actions/workflows/ci.yml)

**rust_timeseries** is a high-performance **Python** library for time-series diagnostics.  
The first release implements the Escanciano–Lobato (2009) robust automatic portmanteau test for serial dependence.  
Heavy lifting is handled in Rust (via [PyO3]) so you get C-level speed with a pure-Python interface.

---

## 📦 Installation

```bash
pip install rust_timeseries
```

Binary wheels are provided for Python 3.11–3.13 on Linux x86-64, macOS (Intel & Apple Silicon) and Windows 64-bit.  
If no wheel matches your platform a source install will build automatically—just have **Rust 1.76+** on `PATH`.

---

## 🚀 Quick start

```python
import rust_timeseries as rts
import numpy as np

y = np.random.randn(500)

test = rts.statistical_tests.EscancianoLobato(y)  # q defaults to 2.4, d defaults to ⌊n**0.2⌋
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
git clone https://github.com/mickwise/rust_timeseries
cd rust_timeseries
python -m venv .venv && source .venv/bin/activate
pip install -U pip maturin pytest
maturin develop --release
```

---

## 📜 License

Released under the **MIT License** – free for commercial and academic use.

---

## Examples
- **EL test + GARCH (Intel 1973–2008)**  
  Notebook: `examples/Escanciano-Lobato/Escanciano-Lobato-example.ipynb`  
  PDF: `examples/Escanciano-Lobato/Escanciano-Lobato-example.pdf`