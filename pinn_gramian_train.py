"""
pinn_gramian_train.py
---------------------
Physics-Informed Neural Network for Reachability Gramian Verification.
 
Paper: "Physics-Informed Neural Network Verification of Kalman–Hewer
        Controllability Gramians in Singular Bilinear Periodic Matrix
        Differential Systems"
Authors: Sri Venkata Durga Sudarsan Madhyannapu and Pradheep Kumar S.
Journal: Neural Networks, Elsevier (Submitted April 2026)

Architecture: Cholesky-reparametrised PINN
  N_theta(t) = L(t) @ L(t).T   (guarantees PSD by construction)

Composite loss: L(theta) = L_residual + L_boundary + L_symmetry
"""

import torch
import torch.nn as nn
import numpy as np
import time

torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────
# 1. System parameters  (4×4 example from the paper)
# ─────────────────────────────────────────────────────────
N_DIM   = 4        # state dimension
T_FINAL = 2.0 * np.pi   # one period
N_COLL  = 500      # collocation points
N_ITER  = 5000     # Adam iterations
LR      = 1e-3     # Adam learning rate
LAMBDA_B = 10.0    # boundary loss weight
LAMBDA_S = 1.0     # symmetry loss weight

def A_matrix(t):
    """Periodic coefficient matrix A(t)."""
    c, s = torch.cos(t), torch.sin(t)
    A = torch.zeros(*t.shape, N_DIM, N_DIM, dtype=torch.float64)
    A[..., 0, 0] = -0.5 + 0.1 * c
    A[..., 0, 1] =  0.2 * s
    A[..., 1, 0] = -0.2 * s
    A[..., 1, 1] = -0.5 + 0.1 * c
    A[..., 2, 2] = -0.3
    A[..., 3, 3] = -0.4 + 0.05 * c
    return A

def B_matrix(t):
    """Control input matrix B(t)."""
    c = torch.cos(t)
    B = torch.zeros(*t.shape, N_DIM, 2, dtype=torch.float64)
    B[..., 0, 0] = 1.0
    B[..., 1, 1] = 1.0 + 0.1 * c
    B[..., 2, 0] = 0.5
    B[..., 3, 1] = 0.5
    return B

# ─────────────────────────────────────────────────────────
# 2. Cholesky-reparametrised network architecture
# ─────────────────────────────────────────────────────────
class GramianNet(nn.Module):
    """
    Maps t -> lower-triangular L(t) with positive diagonal.
    Gramian: W(t) = L(t) @ L(t).T  (PSD by construction).
    """
    def __init__(self, n=N_DIM, hidden=64, layers=4):
        super().__init__()
        self.n = n
        n_lower = n * (n + 1) // 2   # entries in lower triangle

        dims = [1] + [hidden] * (layers - 1) + [n_lower]
        acts = []
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i+1]).double())
            if i < len(dims) - 2:
                acts.append(nn.Tanh())
        self.layers = nn.ModuleList(mods)
        self.acts   = nn.ModuleList(acts)
        self.n_lower = n_lower

    def forward(self, t):
        # t: (batch,) or (batch,1)
        x = t.view(-1, 1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.acts):
                x = self.acts[i](x)
        # x: (batch, n_lower)
        batch = x.shape[0]
        n = self.n
        L = torch.zeros(batch, n, n, dtype=torch.float64, device=x.device)
        idx = torch.tril_indices(n, n)
        L[:, idx[0], idx[1]] = x
        # Positive diagonal via softplus
        diag_idx = torch.arange(n)
        L[:, diag_idx, diag_idx] = torch.nn.functional.softplus(
            L[:, diag_idx, diag_idx]) + 1e-4
        W = L @ L.transpose(-1, -2)
        return W, L

# ─────────────────────────────────────────────────────────
# 3. Physics-informed composite loss
# ─────────────────────────────────────────────────────────
def gramian_ode_residual(net, t_col):
    """
    ODE residual: dW/dt - A(t)W - W A(t)^T - B(t)B(t)^T = 0
    """
    t_col = t_col.requires_grad_(True)
    W, _ = net(t_col)
    # Compute dW/dt via autograd
    dW = torch.zeros_like(W)
    for i in range(N_DIM):
        for j in range(N_DIM):
            g = torch.autograd.grad(
                W[:, i, j].sum(), t_col,
                create_graph=True, retain_graph=True)[0]
            dW[:, i, j] = g.squeeze()

    At = A_matrix(t_col.detach())
    Bt = B_matrix(t_col.detach())
    BBt = Bt @ Bt.transpose(-1, -2)

    residual = dW - (At @ W) - (W @ At.transpose(-1, -2)) - BBt
    return residual

def compute_loss(net, t_col, t0):
    """Composite PINN loss."""
    # Residual loss
    res = gramian_ode_residual(net, t_col)
    loss_r = (res ** 2).mean()

    # Boundary loss: W(0) = 0
    W0, _ = net(t0)
    loss_b = (W0 ** 2).mean()

    # Symmetry loss
    W_col, _ = net(t_col.detach())
    loss_s = ((W_col - W_col.transpose(-1, -2)) ** 2).mean()

    return loss_r + LAMBDA_B * loss_b + LAMBDA_S * loss_s, loss_r

# ─────────────────────────────────────────────────────────
# 4. Training loop
# ─────────────────────────────────────────────────────────
def train():
    net = GramianNet()
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    t_col = torch.linspace(0, T_FINAL, N_COLL,
                           dtype=torch.float64).requires_grad_(False)
    t0    = torch.zeros(1, dtype=torch.float64)

    print(f"Training PINN Gramian network for {N_ITER} iterations...")
    t_start = time.time()

    losses = []
    for k in range(N_ITER):
        opt.zero_grad()
        loss, loss_r = compute_loss(net, t_col, t0)
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if k == 0 or (k + 1) % 500 == 0:
            print(f"  Iter {k+1:5d} | Loss = {loss.item():.4e} | "
                  f"Residual = {loss_r.item():.4e}")

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed:.1f} seconds.")

    # Final Gramian at T
    with torch.no_grad():
        t_T = torch.tensor([T_FINAL], dtype=torch.float64)
        W_T, _ = net(t_T)
        W_np = W_T[0].numpy()

    sv = np.linalg.svd(W_np, compute_uv=False)
    print(f"\nGramian W(T) singular values: {sv}")
    print(f"  sigma_1 = {sv[0]:.4f}  (Paper: 0.312)")
    print(f"  sigma_2 = {sv[1]:.4f}  (Paper: 0.187)")
    print(f"  rank    = {np.sum(sv > 1e-6)}")

    torch.save({'model_state': net.state_dict(),
                'losses': losses,
                'W_T': W_np},
               'pinn_gramian_checkpoint.pt')
    print("\nCheckpoint saved: pinn_gramian_checkpoint.pt")
    return net, losses, W_np

if __name__ == '__main__':
    net, losses, W_T = train()
