import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ThetaNet(nn.Module):
    def __init__(self, d_in=50, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4)   # raw outputs
        )

    def forward(self, y_z):
        # y_z: [batch, 50]
        raw = self.net(y_z)

        # Enforce positivity with softplus
        beta_raw  = raw[:, [0, 2]]
        gamma_raw = raw[:, [1, 3]]

        beta_pos  = F.softplus(beta_raw)         # > 0
        gamma_pos = F.softplus(gamma_raw)        # > 0

        # Enforce gamma1 < gamma2 by sorting
        gammas_sorted, _ = torch.sort(gamma_pos, dim=1)

        beta1 = beta_pos[:, [0]]
        beta2 = beta_pos[:, [1]]
        gamma1 = gammas_sorted[:, [0]]
        gamma2 = gammas_sorted[:, [1]]

        theta_hat = torch.cat([beta1, gamma1, beta2, gamma2], dim=1)
        return theta_hat

x_grid = torch.linspace(0, 5, 50)  # [50]

def simulate_nonlinear(theta, x_grid):
    beta1, gamma1 = theta[:, 0:1], theta[:, 1:2]
    beta2, gamma2 = theta[:, 2:3], theta[:, 3:4]
    f_x = beta1 * torch.exp(-gamma1 * x_grid) + beta2 * torch.exp(-gamma2 * x_grid)
    return f_x + torch.randn_like(f_x) * 0.1

def sample_prior(n):
    betas = torch.rand(n, 2) * 2.0 + 0.5
    gammas_raw = torch.rand(n, 2) * 1.5 + 0.1
    gammas, _ = torch.sort(gammas_raw, dim=1)
    return torch.cat([betas[:, 0:1], gammas[:, 0:1], betas[:, 1:2], gammas[:, 1:2]], dim=1)

# --- simulate training set ---
n_train = 60000
theta_train = sample_prior(n_train)             # [n,4]
y_train = simulate_nonlinear(theta_train, x_grid)  # [n,50]

# --- standardize y (global mean/std, as in handout) ---
y_mean = y_train.mean()
y_std  = y_train.std()
y_train_z = (y_train - y_mean) / y_std          # [n,50]

# --- one observed curve (test) ---
true_theta_test = torch.tensor([[1.8, 0.4, 1.2, 1.1]])
y_obs_raw = simulate_nonlinear(true_theta_test, x_grid)   # [1,50]
y_obs_z = (y_obs_raw - y_mean) / y_std                    # [1,50]




net = ThetaNet(d_in=50, hidden=128)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# Split train/val quickly
n = y_train_z.shape[0]
perm = torch.randperm(n)
n_val = int(0.1 * n)
val_idx = perm[:n_val]
tr_idx  = perm[n_val:]

y_tr, th_tr = y_train_z[tr_idx], theta_train[tr_idx]
y_va, th_va = y_train_z[val_idx], theta_train[val_idx]

batch_size = 512
n_epochs = 20

for ep in range(1, n_epochs+1):
    net.train()
    # mini-batch
    perm2 = torch.randperm(y_tr.shape[0])
    total = 0.0
    for i in range(0, y_tr.shape[0], batch_size):
        idx = perm2[i:i+batch_size]
        yb = y_tr[idx]
        tb = th_tr[idx]

        pred = net(yb)
        loss = torch.mean((pred - tb)**2)

        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * yb.shape[0]

    # validation
    net.eval()
    with torch.no_grad():
        pred_va = net(y_va)
        val_loss = torch.mean((pred_va - th_va)**2).item()

    print(f"Epoch {ep:02d} | train MSE={total/y_tr.shape[0]:.6f} | val MSE={val_loss:.6f}")

net.eval()
with torch.no_grad():
    theta_hat = net(y_obs_z).cpu().numpy().ravel()

print("theta_hat [beta1, gamma1, beta2, gamma2] =", theta_hat)
print("true_theta =", true_theta_test.numpy().ravel())

import matplotlib.pyplot as plt

def f_np(x, beta1, gamma1, beta2, gamma2):
    return beta1*np.exp(-gamma1*x) + beta2*np.exp(-gamma2*x)

x_np = x_grid.cpu().numpy()
y_obs_np = y_obs_raw.cpu().numpy().ravel()

true_np = true_theta_test.numpy().ravel()
y_true = f_np(x_np, *true_np)
y_hat  = f_np(x_np, *theta_hat)

plt.figure()
plt.plot(x_np, y_obs_np, "o", label="data y_obs")
plt.plot(x_np, y_true, "-", linewidth=2, label="true f(x)")
plt.plot(x_np, y_hat,  "--", linewidth=2, label="estimated f(x)")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()
plt.savefig("figs/prob1_fit.png", dpi=200, bbox_inches="tight")
plt.close()




