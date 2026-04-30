"""
utils.py
--------
Helper utilities for PINN Gramian verification.
Paper: Neural Networks (Elsevier) — Submitted April 2026
Authors: Madhyannapu & Pradheep Kumar S.
"""

import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp


def compute_reference_gramian(A_func, B_func, E, T, n_steps=500):
    """
    Compute reference reachability Gramian W_c(0,T) via numerical integration.
    Uses Kronecker-free block-wise approach (Algorithm 1 of paper).

    Parameters
    ----------
    A_func : callable, t -> ndarray (n,n)
    B_func : callable, t -> ndarray (n,m)
    E       : ndarray (n,n), descriptor (possibly singular)
    T       : float, period
    n_steps : int, number of integration steps

    Returns
    -------
    W_c : ndarray (n,n), reachability Gramian
    """
    n = E.shape[0]
    t_span = np.linspace(0, T, n_steps + 1)
    dt = T / n_steps

    # State transition matrix Phi(t,0) via ODE integration
    def phi_ode(t, y):
        A = A_func(t)
        phi = y.reshape(n, n)
        dphi = A @ phi
        return dphi.flatten()

    sol = solve_ivp(phi_ode, [0, T], np.eye(n).flatten(),
                    t_eval=t_span, method='RK45', rtol=1e-10, atol=1e-12)
    Phi = sol.y.T.reshape(-1, n, n)  # (n_steps+1, n, n)

    # Gramian: W_c = integral_0^T Phi(T,t) B(t) B(t)^T Phi(T,t)^T dt
    W_c = np.zeros((n, n))
    Phi_T = Phi[-1]   # Phi(T,0)
    for i, t_i in enumerate(t_span[:-1]):
        Phi_i  = Phi[i]
        Phi_Ti = Phi_T @ np.linalg.pinv(Phi_i)   # Phi(T, t_i)
        B_i    = B_func(t_i)
        integrand = Phi_Ti @ B_i @ B_i.T @ Phi_Ti.T
        W_c += integrand * dt

    return W_c


def relative_frobenius_error(W_hat, W_ref):
    """
    Relative Frobenius error between predicted and reference Gramians.
    e_rel = ||W_hat - W_ref||_F / ||W_ref||_F
    """
    diff_norm = np.linalg.norm(W_hat - W_ref, 'fro')
    ref_norm  = np.linalg.norm(W_ref, 'fro')
    return diff_norm / (ref_norm + 1e-15)


def rank_verification(W_hat, tau_r, tol=1e-6):
    """
    Rank verification criterion (Theorem 4.7 of paper).
    Declare full rank if predicted smallest significant singular value > tau_r.

    Parameters
    ----------
    W_hat : ndarray (n,n)
    tau_r : float, threshold = sigma_r(W_ref) / 2
    tol   : float, near-zero threshold for numerical rank

    Returns
    -------
    is_controllable : bool
    sv              : ndarray, singular values
    rank            : int
    """
    sv   = np.linalg.svd(W_hat, compute_uv=False)
    rank = int(np.sum(sv > tol))
    significant_sv = sv[sv > tol]
    sigma_r_hat = significant_sv[-1] if len(significant_sv) > 0 else 0.0
    is_controllable = bool(sigma_r_hat > tau_r)
    return is_controllable, sv, rank


def melnikov_threshold(W_func, eps_grid, controllability_rank):
    """
    Compute Melnikov controllability threshold eps* from singular value curves.

    Parameters
    ----------
    W_func            : callable, eps -> ndarray (n,n)
    eps_grid          : ndarray, grid of epsilon values
    controllability_rank : int, expected full rank

    Returns
    -------
    eps_star : float, threshold where rank drops below controllability_rank
    """
    for eps in eps_grid:
        W = W_func(eps)
        sv = np.linalg.svd(W, compute_uv=False)
        rank = int(np.sum(sv > 1e-6))
        if rank < controllability_rank:
            return eps
    return eps_grid[-1]


def print_verification_table(sv_ref, sv_pinn, e_rel, rank_ref, rank_pinn,
                              eps_star_ref, eps_star_pinn, train_time_s):
    """
    Print the paper's Table 1 verification summary.
    """
    print("=" * 60)
    print("   PAPER VERIFICATION TABLE — PINN Gramian Results")
    print("=" * 60)
    print(f"  Relative Frobenius error  e_rel    : {e_rel:.2e}")
    print(f"  (Paper target: 6.3e-04)")
    print()
    print(f"  sigma_1 Reference : {sv_ref[0]:.4f}   PINN : {sv_pinn[0]:.4f}")
    print(f"  sigma_2 Reference : {sv_ref[1]:.4f}   PINN : {sv_pinn[1]:.4f}")
    print()
    print(f"  Rank  Reference   : {rank_ref}   PINN : {rank_pinn}")
    print()
    print(f"  eps*  Reference   : {eps_star_ref:.3f}   PINN : {eps_star_pinn:.3f}")
    print(f"  Threshold error   : {abs(eps_star_pinn - eps_star_ref):.4f}")
    print()
    print(f"  Training time     : {train_time_s:.1f} s  (Paper: ~47 s on GPU)")
    print("=" * 60)
