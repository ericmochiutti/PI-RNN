# **Physics-Informed Echo State Networks (PI-ESN) for the Van der Pol Oscillator**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Built with uv](https://img.shields.io/badge/Environment-uv%20%F0%9F%94%8E-green)]()
[![Slidev Presentation](https://img.shields.io/badge/Slides-Slidev-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> **Reference:**
>
> This project is an implementation based on the methodology described in:
> **E. Mochiutti, E. A. Antonelo, E. Camponogara. *Physics-informed echo state networks for hybrid dynamical modeling*. Neurocomputing, 2025.**
> [https://doi.org/10.1016/j.neucom.2025.130251](https://doi.org/10.1016/j.neucom.2025.130251)

---

## 1. Overview & Objective

This repository provides a modular, reproducible implementation of **Physics-Informed Echo State Networks (PI-ESN)** applied to the **Van der Pol oscillator**, a classic benchmark for non-linear dynamics.

### **The Problem**
Standard data-driven models (like vanilla ESNs) are "black boxes". They learn patterns solely from observed data. However, in dynamical systems:
1.  Data may be scarce.
2.  The model may violate fundamental physical laws (e.g., energy conservation) when predicting outside the training range.
3.  Long-term predictions tend to diverge (explode or vanish) as errors accumulate.

### **The Solution: PI-ESN**
The **PI-ESN** architecture creates a hybrid model. It uses the flexible "Reservoir Computing" framework to learn data patterns but constrains the training process using the known differential equations of the system.

**Key Objectives:**
* **Data-Efficiency:** Learn accurate dynamics with fewer training samples.
* **Physical Consistency:** Ensure the model respects the vector field defined by the Van der Pol equations.
* **Long-Term Stability:** Reduce error accumulation in free-running (autonomous) prediction modes.

---

## 2. Mathematical Framework

### **2.1 The Van der Pol Oscillator**
The system is a non-conservative oscillator with non-linear damping. It is defined by the second-order ODE:

$$
\ddot{x} - \mu(1 - x^2)\dot{x} + x = u(t)
$$

Converted to State-Space form (used in this project):
$$ \dot{x}_1 = x_2 $$
$$ \dot{x}_2 &= \mu(1 - x\_1^2)x_2 - x\_1 + u(t)$$

* Where $\mu > 0$ controls the non-linearity and damping strength.
* The system exhibits a stable **limit cycle**.

### **2.2 Echo State Network (ESN)**
The ESN is a Recurrent Neural Network (RNN) where the hidden layer (reservoir) is fixed and random. Only the readout weights are trained.

**State Update Equation:**
$$
\mathbf{x}(k+1) = (1-\alpha)\mathbf{x}(k) + \alpha \tanh(\mathbf{W}\mathbf{x}(k) + \mathbf{W}_{\text{in}}\mathbf{u}(k) + \mathbf{W}_{\text{fb}}\mathbf{y}(k))
$$

* $\mathbf{x}(k)$: Reservoir state vector.
* $\alpha$: Leaking rate (controls the "memory" speed of the reservoir).
* $\mathbf{W}$: Recurrent weight matrix (sparse, fixed).
* $\rho(\mathbf{W})$: Spectral radius, scaled to ensure the "Echo State Property".

### **2.3 Physics-Informed Training (PI-ESN)**
In a standard ESN, we minimize the Mean Squared Error (MSE) of the data. In PI-ESN, we minimize a composite loss function:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \gamma \mathcal{L}_{\text{physics}}
$$

1.  **Data Loss ($\mathcal{L}_{\text{data}}$):** Measures the fit to the observed training trajectory.
2.  **Physics Loss ($\mathcal{L}_{\text{physics}}$):** Measures the residual of the differential equation.

$$
\mathcal{L}_{\text{physics}} = \frac{1}{N} \sum_{k=1}^{N} \left\| \dot{\hat{\mathbf{y}}} - f(\hat{\mathbf{y}}, u) \right\|^2
$$

By using automatic differentiation (via PyTorch) to compute $\dot{\hat{\mathbf{y}}}$, we force the network output to satisfy the Van der Pol vector field $f(\cdot)$.

---

## 3. Results & Discussion

The architecture implemented in this repository demonstrates distinct advantages over standard ESNs:

    * The PI-ESN successfully captures the **limit cycle** of the Van der Pol oscillator, even when trained on partial data.
    * Standard ESNs often drift away from the limit cycle during long-term recursive prediction.
    * By embedding physics, the model generalizes better to unseen input signals.

### **Performance Comparison**

![ESN vs PI-ESN Prediction](assets/result_preview.png)
*Figure 1: Comparison of free-running prediction. Notice how the PI-ESN (green) maintains the phase and amplitude, while the standard ESN (red) may drift or dampen over time.
------------------------------------------------------------------------
## 4. Project Structure 
```
.
├── PI_ESN
│   ├── experiments # ML experiment outputs
│   ├── model_classes
│   │   ├── ESN.py
│   │   ├── ESN_torch.py
│   │   └── PIESN.py
│   ├── pi_esn_vanderpol.ipynb # ESN/PI-ESN on Van der Pol system
│   ├── simulation
│   │   ├── input_signal_generator.py
│   │   ├── split_data.py
│   │   └── vdp_simulator.py
│   └── utils # Helper scripts for experiment management
│       ├── create_experiment_directory.py
│       └── save_experiment_config.py
├── pi-esn-slides # Slidev project for presentation
│   ├── background
│   ├── components
│   ├── netlify.toml
│   ├── package.json
│   ├── pages
│   ├── pnpm-lock.yaml
│   ├── public
│   ├── slides.md
│   ├── snippets
│   └── vercel.json
├── pyproject.toml
└── uv.lock
```
---

## 5. Environment Setup

This project uses **`uv`**, a modern Python package manager.

### **Step 1: Install `uv` (if not installed)**
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```
### **Step 2: Sync Environment
Clone the repo and sync dependencies exactly as specified in uv.lock.
```bash
uv sync
```

## 6. Slidev Presentation

This repository includes a full Slidev presentation explaining:

-   Model architecture
-   Training workflow
-   Simulation details
-   Comparative results (ESN vs PI-ESN)
-   Interactive Plotly figures

Location:

    pi-esn-slides/
        slides.md
        public/

### ✔ Install Slidev

Requires Node.js:

``` bash
npm install -g @slidev/cli
```

or with pnpm:

``` bash
pnpm add -g @slidev/cli
```

### ✔ Run the presentation

``` bash
slidev slides.md --port 3030
```

Alternatively through VSCode:

``` bash
sudo npm exec -c 'slidev "slides.md" --port 3030'
```