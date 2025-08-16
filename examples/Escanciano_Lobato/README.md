#### **EL-specific `examples/Escanciano_Lobato/README.md`**

# Escanciano–Lobato Test Example

This folder contains a demonstration of the **Escanciano–Lobato (2009)** automatic Portmanteau test for serial correlation, implemented in Rust and exposed to Python via `rust_timeseries`.

---

## Contents
- `Escanciano_Lobato_example.ipynb` – Jupyter notebook with simulation and usage.
- `Escanciano_Lobato_example.pdf` – PDF export of the notebook.
- `escanciano_lobato_histogram.png` – Histogram of empirical test size.

---

## Summary of Results
- Under conditional heteroskedasticity, the Escanciano–Lobato test maintains size ≈ 5% at 5% nominal.
- In contrast, Ljung–Box inflates size across common lag choices.
- Automatic lag selection \( \tilde{p} \) and heteroskedasticity-robust variance \( \hat{\tau} \) make the test reliable in practice.

---

## Highlights

| Why it matters           | What you get                                                                    |
|--------------------------|---------------------------------------------------------------------------------|
| **Native-code core**     | Tight Rust loops compiled to machine code                                       |
| **Zero-copy I/O**        | `numpy.ndarray` / `pandas.Series` buffers are viewed directly—no copying ever   |
| **Heteroskedastic-robust** | τ̂-adjusted autocorrelations maintain validity under conditional heteroskedasticity |
| **Automatic lag choice** | Data-driven \(p̃\) maximises the penalised statistic \(L_p\)                    |
| **Friendly errors**      | Clear `ValueError` / `OSError` when inputs are invalid                          |

---

## Limitations

- **Power at higher lags:** The test has no power against Pitman-local alternatives where serial correlation appears only at lag j > 1; asymptotically it behaves like a lag-1 test under those alternatives.  
- **Assumptions:** Theory requires strict stationarity, finite fourth moments, and a martingale-difference structure. Extremely heavy-tailed data or processes outside this framework are not covered.  
- **Residuals:** Applying the test to residuals (rather than raw series) requires additional theoretical care.  
- **Finite samples:** Simulations show mild size distortions at small–moderate n under GARCH; these vanish at larger n.  
- **Tuning constant:** The AIC/BIC switch uses q = 2.4, chosen via simulation. Other nearby values perform similarly, but the choice is not formally unique.


---


## References
Escanciano, J.C. & Lobato, I.N. (2009). *An automatic Portmanteau test for serial correlation*. Journal of Econometrics.