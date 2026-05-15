## -PINN-Gramian-Controllability

**Physics-Informed Neural Network Verification of Kalman–Hewer Controllability Gramians in Singular Bilinear Periodic Matrix Differential Systems**
DOI: https://doi.org/10.5281/zenodo.20196769

**Authors:** Sri Venkata Durga Sudarsan Madhyannapu and Pradheep Kumar S.  
**Journal:** *Neural Networks*, Elsevier (Submitted April 2026)  
**SSRN Preprint:** Not yet deposited

---

## Repository Contents

| File | Description |
|------|-------------|
| `Paper3_PINN_Gramian_Verification.m` | MATLAB reference Gramian computation and verification of all paper numerical values |
| `pinn_gramian_train.py` | Python/PyTorch PINN training script (Cholesky-reparametrised architecture) |
| `generate_figures.py` | Figure generation routines for Figures 1–3 |
| `utils.py` | Helper functions: Gramian assembly, singular value routines |
| `requirements.txt` | Python package requirements |

---

## Key Numerical Results (Paper-Verified)

| Metric | Value |
|--------|-------|
| Relative Frobenius error `e_rel` | `6.3 × 10⁻⁴` |
| Predicted rank of `Ŵ_c` | `2` |
| Reference rank of `W_ref` | `2` |
| PINN threshold `ε̂*` | `0.169` |
| Reference threshold `ε*` | `0.170` |
| Threshold error | `< 0.001` |
| Training iterations | `5000` |
| Training time (GPU) | `~47 seconds` |
| σ₁(W_ref) | `0.312` |
| σ₂(W_ref) | `0.187` |

---

## Quick Start

### MATLAB Verification
```matlab
% Run full paper verification
Paper3_PINN_Gramian_Verification
```

### Python PINN Training
```bash
pip install -r requirements.txt
python pinn_gramian_train.py
```

### Generate All Figures
```bash
python generate_figures.py
```

---

## System Description

The paper addresses the **4×4 singular bilinear periodic matrix differential system**:

```
E Ẋ(t) = A(t)X(t) + B(t)U(t),   t ∈ [0, T], T = 2π
```

with singular descriptor matrix `E`, Lyapunov bilinear coupling, and
periodic coefficient matrices `A(t)`, `B(t)`.

The PINN architecture uses **Cholesky reparametrisation**:
```
N_θ(t) = L(t) L(t)ᵀ
```
guaranteeing positive semi-definiteness by construction.

---

## Citation

If you use this code, please cite:

```bibtex
@article{sudarsan2026pinn_gramian_nn,
  author  = {Madhyannapu, Sri Venkata Durga Sudarsan and
             {Pradheep Kumar}, S.},
  title   = {Physics-Informed Neural Network Verification of
             {Kalman--Hewer} Controllability {Gramians}
             in Singular Bilinear Periodic Matrix Differential Systems},
  journal = {Neural Networks},
  year    = {2026},
  note    = {Submitted}
}
```

---

## Related Work

- **Bilinear Controllability (MCSS):** `github.com/msvdsudarsan/Bilinear-Matrix-Periodic-Controllability`
- **Impulse KH Controllability (NAHS):** `github.com/msvdsudarsan/SBLIPMS-Impulse-KH-Controllability`
- **Melnikov Observability (CSF):** `github.com/msvdsudarsan/SBMPMS-observability`
- **Adaptive PINN (SSRN):** `https://doi.org/10.2139/ssrn.6277631`

---

## License

This code is released for academic research purposes.
Contact: `msvdsudarsan@gmail.com`
