# PINNthermo: Input Convex Neural Network (ICNN) Helmholtz Free Energy & CAPE-OPEN PR-EOS Engine

[![CI](https://github.com/dhrubchatterjee16-lang/pinn-thermo/actions/workflows/ci.yml/badge.svg)](https://github.com/dhrubchatterjee16-lang/pinn-thermo/actions)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhrubchatterjee16-lang/pinn-thermo/blob/main/notebooks/quickstart.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


[![CI Tests](https://github.com/dhrubchatterjee16-lang/pinn-thermo/actions/workflows/ci.yml/badge.svg)](https://github.com/dhrubchatterjee16-lang/pinn-thermo)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

**PINNthermo** is an open-source, thermodynamically consistent Scientific Machine Learning (SciML) framework and process simulator socket. It bridges microscopic molecular potentials and macroscopic chemical process engineering by embedding fundamental thermodynamic laws directly into neural architectures.

---

## 🔬 Mathematical & Thermodynamic Foundations

### 1. Input Convex Neural Network (ICNN) Helmholtz Free Energy
Standard black-box neural networks violate the Second Law of Thermodynamics and mechanical stability conditions, generating artificial phase inversions or negative pressures. PINNthermo guarantees thermodynamic convexity by parameterizing the Helmholtz free energy functional $A(T, V, \mathbf{N})$:

$$A(T, V, \mathbf{N}) = A_{\text{ideal}}(T, V, \mathbf{N}) + A_{\text{residual}}^{\text{ICNN}}(T, V, \mathbf{N})$$

where the residual neural operator satisfies:

$$z_{k+1} = \sigma\left(W_k^{(z)} z_k + W_k^{(u)} u + b_k\right), \quad \text{with } W_k^{(z)} \ge 0$$

Under non-negative weight projections ($W_k^{(z)} \ge 0$) and convex, non-decreasing activation functions $\sigma$, the network is unconditionally convex with respect to volume $V$:

$$\frac{\partial^2 A}{\partial V^2} \ge 0 \quad \forall (T, V, \mathbf{N})$$

### 2. Exact Thermodynamic Derivatives & Maxwell Relations
Through exact analytical and automatic differentiation, state variables and chemical potentials are computed:

- **Thermodynamic Pressure:**
  $$P = -\left(\frac{\partial A}{\partial V}\right)_{T, \mathbf{N}}$$

- **Entropy:**
  $$S = -\left(\frac{\partial A}{\partial T}\right)_{V, \mathbf{N}}$$

- **Chemical Potentials:**
  $$\mu_i = \left(\frac{\partial A}{\partial N_i}\right)_{T, V, N_{j \ne i}}$$

- **Gibbs Free Energy & Gibbs-Duhem Consistency:**
  $$G = A + PV = \sum_{i} N_i \mu_i \implies \sum_i N_i d\mu_i + S dT - V dP = 0$$

### 3. High-Pressure Peng-Robinson Cubic Equation of State
For non-ideal gas and vapor-liquid equilibrium (VLE) regimes:

$$P = \frac{R T}{v - b} - \frac{a(T)}{v(v + b) + b(v - b)}$$

In terms of compressibility factor $Z = \frac{P v}{R T}$:

$$Z^3 - (1 - B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0$$

where:
$$A = \frac{a(T) P}{(R T)^2}, \quad B = \frac{b P}{R T}$$

Vapor fugacity coefficients $\phi_i$ are solved via:

$$\ln \phi = Z - 1 - \ln(Z - B) - \frac{A}{2\sqrt{2}B} \ln\left[\frac{Z + (1 + \sqrt{2})B}{Z + (1 - \sqrt{2})B}\right]$$

### 4. CAPE-OPEN 1.1 Compliant Rachford-Rice PT-Flash
The isothermal-isobaric flash split is solved across arbitrary multicomponent mixtures:

$$f(\beta) = \sum_{i=1}^{C} \frac{z_i (K_i - 1)}{1 + \beta(K_i - 1)} = 0$$

with quadratic Newton-Raphson updates:

$$\beta^{(k+1)} = \beta^{(k)} - \frac{f(\beta^{(k)})}{f'(\beta^{(k)})}, \quad f'(\beta) = -\sum_{i=1}^{C} \frac{z_i (K_i - 1)^2}{[1 + \beta(K_i - 1)]^2}$$

---

## ⚡ Installation

```bash
git clone https://github.com/dhrubchatterjee16-lang/pinn-thermo.git
cd pinn-thermo
pip install -e .
```

---

## 💻 CLI Quickstart

### 1. Solve Peng-Robinson Cubic EOS
```bash
pinn-thermo eos --temp 300 --press 2.5e6 --tc 190.56 --pc 45.99e5 --omega 0.011
```

### 2. Evaluate ICNN Helmholtz Convexity & Stability
```bash
pinn-thermo icnn --temp 300 --vol 0.024 --moles 0.6,0.4
```

### 3. Run CAPE-OPEN Multicomponent Flash
```bash
pinn-thermo flash --temp 220 --press 3e6 --compounds Methane,Ethane --feed 0.6,0.4
```

---

## 🐍 Python API Usage

```python
from pinn_thermo import CapeOpenPropertyPackage, InputConvexHelmholtzNetwork, ICNNState

# 1. CAPE-OPEN Property Package
pkg = CapeOpenPropertyPackage(compounds=["Methane", "Ethane"])
flash = pkg.calc_pt_flash(t_k=220.0, p_pa=3.0e6, z_feed=[0.6, 0.4])

print(f"Vapor Fraction: {flash.vapor_fraction:.4f}")
print(f"Liquid Composition (x): {flash.x_liquid}")
print(f"Vapor Composition (y): {flash.y_vapor}")

# 2. Input Convex Neural Network Mechanical Stability
icnn = InputConvexHelmholtzNetwork(n_components=2)
state = ICNNState(temperature_k=300.0, volume_m3=0.024, mole_numbers=[0.6, 0.4])
derivs = icnn.evaluate(state)

print(f"Pressure: {derivs.pressure_pa:.2f} Pa")
print(f"Mechanically Stable (d2A/dV2 >= 0): {derivs.is_mechanically_stable}")
print(f"Gibbs-Duhem Residual: {derivs.gibbs_duhem_residual:.2e}")
```

---

## 🧪 Verification & Unit Tests

Run the complete physics and numerical test suite:

```bash
python run_tests.py
```

Test coverage includes:
- Unconditional mechanical stability ($\frac{\partial^2 A}{\partial V^2} \ge 0$) across multi-temperature grids
- Ideal gas asymptotic limits ($Z \to 1.0, \phi \to 1.0$ as $P \to 0$)
- Subcritical vapor-liquid root separation
- CAPE-OPEN Rachford-Rice PT-flash mass balance convergence

---

## 📜 Citation & License

```bibtex
@software{chatterjee2026pinnthermo,
  author = {Chatterjee, Dhrubajyoti},
  title = {PINNthermo: Thermodynamically Consistent Input Convex Neural Networks & CAPE-OPEN Engine},
  year = {2026},
  url = {https://github.com/dhrubchatterjee16-lang/pinn-thermo}
}
```

Licensed under the Apache License 2.0.