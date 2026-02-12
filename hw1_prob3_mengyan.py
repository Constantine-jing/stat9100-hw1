# hw1_prob3_conformal.py
# Split-Conformal inference for parameter uncertainty (Problem 3)
# Score: r_{i,j} = |theta_hat_j(y_i) - theta_{i,j}|
# Interval: theta_hat_j(y_obs) +/- q_j where q_j is conformal quantile from calibration residuals

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# 0) Settings
# -----------------------
torch.manual_seed(42)
np.random.seed(42)

alpha = 0.05          # 95% conformal
N = 60000             # total simulated pairs (theta, y)
cal_frac = 0.20       # split conformal: 80% train, 20% calibration
sigma = 0.1           # noise sd
epochs = 20
batch_size = 512
lr = 1e-3

os.makedirs("results", exist_ok=True)

# fixed x grid
x_grid = torch.linspace(0, 5, 50)  # [50]

# -----------------------
# 1) Simulator + Prior (same as Prob1)
# -----------------------
def simulate_nonlinear(theta, x_grid, sigma=0.1):
    beta1, gamma1 = theta[:, 0:1], theta[:, 1:2]
    beta2, gamma2 = theta[:, 2:3], theta[:, 3:4]
    f_x = beta1 * torch.exp(-gamma1 * x_grid) + beta2 * torch.exp(-gamma2 * x_grid)
    return f_x + torch.randn_like(f_x) * sigma

def sample_prior(n):
    # Enforce gamma1 < gamma2 for identifiability
    betas = torch.rand(n, 2) * 2.0 + 0.5      # [0.5, 2.5]
    gammas_raw = torch.rand(n, 2) * 1.5 + 0.1 # [0.1, 1.6]
    gammas, _ = torch.sort(gammas_raw, dim=1)
    return torch.cat([betas[:, 0:1], gammas[:, 0:1], betas[:, 1:2], gammas[:, 1:2]], dim=1)

# -----------------------
# 2) ThetaNet (Bayes estimator) — paste from Prob1
# -----------------------
# TODO: Paste your Prob1 ThetaNet here (exact same class).
# It should take y_z [batch,50] and output theta_hat [batch,4] with positivity + sorted gammas.

class ThetaNet(nn.Module):
    def __init__(self, d_in=50, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4)
        )

    def forward(self, y_z):
        raw = self.net(y_z)

        beta_raw  = raw[:, [0, 2]]
        gamma_raw = raw[:, [1, 3]]

        beta_pos  = F.softplus(beta_raw)
        gamma_pos = F.softplus(gamma_raw)

        gammas_sorted, _ = torch.sort(gamma_pos, dim=1)

        beta1 = beta_pos[:, [0]]
        beta2 = beta_pos[:, [1]]
        gamma1 = gammas_sorted[:, [0]]
        gamma2 = gammas_sorted[:, [1]]

        theta_hat = torch.cat([beta1, gamma1, beta2, gamma2], dim=1)
        return theta_hat

# -----------------------
# 3) Simulate dataset + standardize y
# -----------------------
theta_all = sample_prior(N)                          # [N,4]
y_all = simulate_nonlinear(theta_all, x_grid, sigma) # [N,50]

y_mean = y_all.mean()
y_std  = y_all.std()
y_all_z = (y_all - y_mean) / y_std                   # [N,50]

# -----------------------
# 4) Split: Train / Calibration (split conformal)
# -----------------------
perm = torch.randperm(N)
n_cal = int(cal_frac * N)
cal_idx = perm[:n_cal]
tr_idx  = perm[n_cal:]

y_tr, th_tr = y_all_z[tr_idx], theta_all[tr_idx]
y_cal, th_cal = y_all_z[cal_idx], theta_all[cal_idx]

print(f"Train size={y_tr.shape[0]}, Cal size={y_cal.shape[0]}")

# -----------------------
# 5) Train ThetaNet on TRAIN split only
# -----------------------
net = ThetaNet(d_in=50, hidden=128)
opt = torch.optim.Adam(net.parameters(), lr=lr)

# TODO: If you want, paste your exact Prob1 training loop here.
# Below is a clean minimal training loop (MSE loss).
for ep in range(1, epochs + 1):
    net.train()
    perm2 = torch.randperm(y_tr.shape[0])
    total = 0.0

    for i in range(0, y_tr.shape[0], batch_size):
        idx = perm2[i:i+batch_size]
        yb = y_tr[idx]
        tb = th_tr[idx]

        pred = net(yb)
        loss = torch.mean((pred - tb) ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * yb.shape[0]

    if ep % 5 == 0 or ep == 1:
        net.eval()
        with torch.no_grad():
            pred_cal = net(y_cal)
            cal_mse = torch.mean((pred_cal - th_cal) ** 2).item()
        print(f"Epoch {ep:02d} | train MSE={total/y_tr.shape[0]:.6f} | cal MSE={cal_mse:.6f}")

# -----------------------
# 6) Calibration residuals: r_{i,j} = |theta_hat_j(y_i) - theta_{i,j}|
# -----------------------
net.eval()
with torch.no_grad():
    th_hat_cal = net(y_cal)                          # [n_cal,4]
resid = torch.abs(th_hat_cal - th_cal)               # [n_cal,4]

# Conformal quantile with +1 correction:
# k = ceil((n_cal + 1) * (1 - alpha))  (1-indexed rank)
k = int(np.ceil((n_cal + 1) * (1 - alpha)))
q = torch.kthvalue(resid, k, dim=0).values           # [4]
q_np = q.cpu().numpy()

print("Conformal q (95%) =", q_np)

# -----------------------
# 7) Apply to one observed curve y_obs
# -----------------------
true_theta_test = torch.tensor([[1.8, 0.4, 1.2, 1.1]])

# fix seed so y_obs is reproducible
torch.manual_seed(999)
y_obs_raw = simulate_nonlinear(true_theta_test, x_grid, sigma)       # [1,50]
y_obs_z = (y_obs_raw - y_mean) / y_std                               # [1,50]

with torch.no_grad():
    th_hat_obs = net(y_obs_z).squeeze(0)                             # [4]

lower = (th_hat_obs - q).cpu().numpy()
upper = (th_hat_obs + q).cpu().numpy()
est   = th_hat_obs.cpu().numpy()
truth = true_theta_test.squeeze(0).cpu().numpy()

print("theta_hat(y_obs) =", est)
print("lower95 =", lower)
print("upper95 =", upper)

# -----------------------
# 8) Save CSV for Rmd
# -----------------------
names = ["beta1","gamma1","beta2","gamma2"]
df = pd.DataFrame({
    "Parameter": names,
    "Estimate": est,
    "Truth": truth,
    "q_conformal": q_np,
    "Lower95": lower,
    "Upper95": upper
})
df.to_csv("results/prob3_conformal.csv", index=False)
print("Saved: results/prob3_conformal.csv")
