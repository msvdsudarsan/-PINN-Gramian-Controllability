# PINN-Gramian-Controllability

**Physics-Informed Neural Network Verification of Kalman–Hewer Controllability Gramians in Singular Bilinear Periodic Matrix Differential Systems**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20196769.svg)](https://doi.org/10.5281/zenodo.20196769)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Journal](https://img.shields.io/badge/Journal-Neural%20Networks%20(Elsevier)-blue)](https://www.sciencedirect.com/journal/neural-networks)

**Authors:** Sri Venkata Durga Sudarsan Madhyannapu and Sravanam Pradheep Kumar  
**Affiliation (corresponding):** Department of Mathematics, School of Sciences, Humanities and Management, Dr. RVR NRI Institute of Technology (Deemed to be University), Pothavarappadu, Agiripalli, Vijayawada Rural 521212, Andhra Pradesh, India  
**Journal:** *Neural Networks*, Elsevier (ISSN 0893-6080) — Submitted May 2026  
**Contact:** msvdsudarsan@gmail.com  
**ORCID:** [0009-0001-2126-6428](https://orcid.org/0009-0001-2126-6428)

---

## Overview

This repository provides the MATLAB and Python codes that support all numerical results, figures, and theorem verifications reported in the paper. The paper introduces a **data-free, PSD-guaranteed Physics-Informed Neural Network (PINN) framework** for operator-level learning and verification of the reachability Gramian $\mathcal{W}_c(0,T)$ of singular bilinear periodic matrix differential systems.

The central contribution is the **Gramian-PINN Rank-Verification Theorem** (Theorem 4.7), which certifies Kalman–Hewer controllability from a single PINN forward pass with provable error bound $|\hat{\varepsilon}^* - \varepsilon^*| \leq C\delta$.

---

## Repository Contents

| File | Description |
|------|-------------|
| `PINN_Gramian_Verification.m` | MATLAB reference script: builds the Gramian, simulates PINN approximation, verifies all paper numerical values, and generates the Chain-of-Guarantees check |
| `pinn_gramian_train.py` | PyTorch PINN training script with Cholesky-reparametrised architecture ($\mathcal{N}_\theta(t) = L(t)L(t)^\top$, PSD guaranteed by construction) |
| `generate_figures.py` | Python figure generation script: produces Figures 1–3 with clean legends (no artefacts) |
| `utils.py` | Helper functions: Gramian assembly, singular value routines, SP-Gram reference solver |
| `requirements.txt` | Python package requirements |
| `PINN_Gramian_Verification_output.txt` | MATLAB console output log (all values paper-verified) |
| `Fig1_training_loss.pdf` | Training loss $\mathcal{L}(\theta)$ vs iteration (Adam + L-BFGS) |
| `Fig2_pinn_vs_reference.pdf` | PINN-predicted vs reference Gramian singular values |
| `Fig3_approximation_error.pdf` | Pointwise relative approximation error vs $\varepsilon$ |

---

## Key Numerical Results (Paper-Verified)

| Metric | Value |
|--------|-------|
| Relative Frobenius error $e_{\text{rel}}$ | $6.3 \times 10^{-4}$ |
| Predicted rank of $\hat{\mathcal{W}}_c$ | $2$ |
| Reference rank of $\mathcal{W}_{\text{ref}}$ | $2$ |
| PINN threshold $\hat{\varepsilon}^*$ (GPU, float32) | $0.169$ |
| Reference threshold $\varepsilon^*$ (analytical) | $0.170$ |
| Threshold error $|\hat{\varepsilon}^* - \varepsilon^*|$ | $< 0.001$ |
| Training iterations | $5000$ |
| Training time | $\approx 47$ s (NVIDIA A100) |
| $\sigma_1(\mathcal{W}_{\text{ref}})$ | $0.312$ |
| $\sigma_2(\mathcal{W}_{\text{ref}})$ | $0.187$ |

**Chain-of-Guarantees verification:**
- $\tau_r = \sigma_2(\mathcal{W}_c)/2 = 0.0935$
- $\delta_0 = e_{\text{rel}} \cdot \|\mathcal{W}_{\text{ref}}\|_F = 2.29 \times 10^{-4}$
- $\delta_0 < \tau_r$ ✓ → Rank verification holds (Theorem 4.7)

---

## Quick Start

### MATLAB Verification
```matlab
% Run in MATLAB R2024b or later
PINN_Gramian_Verification
```
Expected output: all values match the paper table; `ALL VALUES: OK`.

### Python PINN Training
```bash
pip install -r requirements.txt
python pinn_gramian_train.py
```

### Generate Figures
```bash
python generate_figures.py
```
Outputs: `Fig1_training_loss.pdf`, `Fig2_pinn_vs_reference.pdf`, `Fig3_approximation_error.pdf`

---

## System Description

The paper addresses the **4×4 singular bilinear periodic matrix differential system**:

$$E\dot{X}(t) = A(t)X(t) + B(t)U(t), \quad t \in [0, T],\; T = 2\pi$$

with singular descriptor matrix $E$, periodic coefficient matrices $A(t)$, $B(t)$, and Lyapunov bilinear coupling. The Gramian satisfies a Lyapunov-type variational ODE:

$$E\dot{\mathcal{W}}_c(0,t)E^\top = A(t)\mathcal{W}_c(0,t)E^\top + E\mathcal{W}_c(0,t)A(t)^\top + B(t)B(t)^\top$$

The PINN architecture uses **Cholesky reparametrisation**:

$$\mathcal{N}_\theta(t) = L_\theta(t)\,L_\theta(t)^\top$$

guaranteeing positive semi-definiteness at every gradient step by architecture — not by penalty.

---

## Theoretical Contributions

| Theorem | Content |
|---------|---------|
| **Theorem 4.3** | Universal approximation for the Gramian functional |
| **Theorem 4.5** | $O(1/\sqrt{K})$ Adam convergence bound for the composite PINN loss |
| **Theorem 4.7** | Gramian-PINN Rank-Verification Theorem (main result): certifies Kalman–Hewer controllability with provable bound $\|\hat{\varepsilon}^* - \varepsilon^*\| \leq C\delta$ |

---

## Citation

If you use this code, please cite:

```bibtex
@article{madhyannapu2026pinn_gramian,
  author  = {Madhyannapu, Sri Venkata Durga Sudarsan and
             Kumar, Sravanam Pradheep},
  title   = {Physics-Informed Neural Network Verification of
             {Kalman--Hewer} Controllability {Gramians}
             in Singular Bilinear Periodic Matrix Differential Systems},
  journal = {Neural Networks},
  publisher = {Elsevier},
  issn    = {0893-6080},
  year    = {2026},
  note    = {Submitted May 2026},
  doi     = {10.5281/zenodo.20196769}
}
```

---

## Related Work (Companion Papers)

| Paper | Venue | Repository |
|-------|-------|------------|
| Bilinear Matrix Periodic Controllability | MCSS | [github.com/msvdsudarsan/Bilinear-Matrix-Periodic-Controllability](https://github.com/msvdsudarsan/Bilinear-Matrix-Periodic-Controllability) |
| Impulse KH Controllability | NAHS | [github.com/msvdsudarsan/SBLIPMS-Impulse-KH-Controllability](https://github.com/msvdsudarsan/SBLIPMS-Impulse-KH-Controllability) |
| Melnikov Observability | CSF | [github.com/msvdsudarsan/SBMPMS-observability](https://github.com/msvdsudarsan/SBMPMS-observability) |
| Adaptive PINN (preprint) | SSRN | [doi.org/10.2139/ssrn.6277631](https://doi.org/10.2139/ssrn.6277631) |

---

## License

MIT License — see [LICENSE](LICENSE) for details.  
Released for academic research purposes. For commercial use, please contact the authors.
