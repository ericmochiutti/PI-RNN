# ![Project Icon](./assets/icon.png)

# **Physics-Informed Echo State Networks (PI-ESN) for the Van der Pol Oscillator**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Built with
uv](https://img.shields.io/badge/Environment-uv%20%F0%9F%94%8E-green)]()
[![Slidev
Presentation](https://img.shields.io/badge/Slides-Slidev-orange)]()
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> **Reference:**\
> For a detailed description of the methodology, see:\
> **E. Mochiutti, E. A. Antonelo, E. Camponogara. *Physics-informed echo state networks for
> hybrid dynamical modeling*. Neurocomputing, 2025.**\
> https://doi.org/10.1016/j.neucom.2025.130251

------------------------------------------------------------------------

## 1. Overview

This repository provides a complete and modular implementation of
**Physics-Informed Echo State Networks (PI-ESN)** applied to the **Van
der Pol oscillator**.\
It includes signal generation, nonlinear dynamic simulation, ESN/PI-ESN training, long-term prediction analysis, artifact storage, interactive Plotly visualizations, and a Slidev-based presentation.

The repository implements two main models:

### **• Standard ESN (Echo State Network)**

A recurrent reservoir model trained using ridge regression.

### **• Physics-Informed ESN (PI-ESN)**

A hybrid model that incorporates the differential equation of the Van der Pol oscillator directly into the loss function, improving
long-horizon predictive stability.

The complete workflow includes:

1.  Generation of excitation signals (PRBS, step, sine, mixed).
2.  Numerical simulation of the Van der Pol oscillator.
3.  Dataset creation, normalization, and splitting.
4.  ESN and PI-ESN training.
5.  Long-term prediction and rollout evaluation.
6.  Error metrics (MSE, RMSE) and diagnostics.
7.  Automatic saving of experiment configurations and artifacts.
8.  Generation of interactive Plotly HTML visualizations.
9.  Slidev presentation for documentation and results.

------------------------------------------------------------------------

## 2. Environment Setup (Using `uv`)

This project uses **`uv`**, a fast Python package manager and
environment manager.

### ✔ Create and sync the environment

``` bash
uv sync
```

This command:

-   Creates a Python environment
-   Installs all dependencies from `uv.lock`
-   Ensures reproducibility

------------------------------------------------------------------------
## 3. Project Structure 
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
## 4. Van der Pol Simulation

The Van der Pol system is defined as:

$$
\dot{x} = y 
$$
$$
\dot{y} = \mu(1 - x^2)y - x + u(t)
$$

Simulation code is located in:

    simulation/
        vdp_simulator.py
        signal_generator.py

------------------------------------------------------------------------

## 5. Echo State Network (ESN)

The ESN state update equation is:

$$
\mathbf{x}(k+1) = (1-\alpha)\mathbf{x}(k) + \alpha \tanh(\mathbf{W}\mathbf{x}(k) + \mathbf{W}_{\text{in}}\mathbf{u}(k))
$$

Training uses ridge regression:

$$
\mathbf{W}_{\text{out}} = \mathbf{Y}\mathbf{X}^{\top} (\mathbf{X}\mathbf{X}^{\top} + \lambda \mathbf{I})^{-1}
$$

Configurable parameters include:

-   Reservoir size\
-   Spectral radius\
-   Leaking rate\
-   Input/reservoir/bias scaling\
-   Regularization\
-   Washout length\
-   Random seed for reproducibility

------------------------------------------------------------------------

## 6. Physics-Informed ESN (PI-ESN)

The PI-ESN introduces a hybrid loss function that combines data fitting
with physics consistency.\
The physics-based term enforces the Van der Pol dynamics:

$$
\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_{\text{phys}} \| \mathbf{\dot{{z}}} - f(\mathbf{{z}}, u) \|^2
$$

## 7. Slidev Presentation

This repository includes a full Slidev presentation explaining:

-   Model architecture\
-   Training workflow\
-   Simulation details\
-   Comparative results (ESN vs PI-ESN)\
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