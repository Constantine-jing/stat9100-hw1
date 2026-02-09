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

# ----- Bootstrap for parameter uncertainty -----
B = 100

# fitted mean curve under theta_hat
x_np = x_grid.cpu().numpy()
def f_np(x, beta1, gamma1, beta2, gamma2):
    return beta1*np.exp(-gamma1*x) + beta2*np.exp(-gamma2*x)

y_obs_np = y_obs_raw.cpu().numpy().ravel()
y_fit = f_np(x_np, *theta_hat)

# residuals from fitted curve
resid = y_obs_np - y_fit

theta_boot = np.zeros((B, 4))
yhat_boot  = np.zeros((B, len(x_np)))  # optional: for bands

rng = np.random.default_rng(123)

for b in range(B):
    # resample residuals with replacement
    resid_b = rng.choice(resid, size=resid.shape[0], replace=True)
    y_b = y_fit + resid_b

    # standardize using training y_mean, y_std
    y_b_t = torch.tensor(y_b, dtype=torch.float32).unsqueeze(0)
    y_b_z = (y_b_t - y_mean) / y_std

    # predict theta for bootstrap sample
    net.eval()
    with torch.no_grad():
        th_b = net(y_b_z).cpu().numpy().ravel()

    theta_boot[b, :] = th_b
    yhat_boot[b, :]  = f_np(x_np, *th_b)  # optional

# summaries
theta_mean = theta_boot.mean(axis=0)
theta_sd   = theta_boot.std(axis=0, ddof=1)
theta_ci   = np.quantile(theta_boot, [0.025, 0.975], axis=0)

print("\nBootstrap summary (B=100)")
names = ["beta1","gamma1","beta2","gamma2"]
for j, nm in enumerate(names):
    print(f"{nm:6s}: mean={theta_mean[j]:.3f}, sd={theta_sd[j]:.3f}, "
          f"CI95=[{theta_ci[0,j]:.3f}, {theta_ci[1,j]:.3f}]")


# 90% band for reconstructed response (optional but nice)
lower = np.quantile(yhat_boot, 0.05, axis=0)
upper = np.quantile(yhat_boot, 0.95, axis=0)

plt.figure()
plt.plot(x_np, y_obs_np, "o", label="data y_obs")
plt.plot(x_np, y_true, "-", linewidth=2, label="true f(x)")
plt.plot(x_np, y_hat,  "--", linewidth=2, label="estimated f(x)")
plt.fill_between(x_np, lower, upper, alpha=0.2, label="bootstrap 90% band")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()
plt.savefig("figs/prob1_fit_band.png", dpi=200, bbox_inches="tight")
plt.close()

np.savetxt("results/prob1_theta_boot.csv", theta_boot, delimiter=",",
           header="beta1,gamma1,beta2,gamma2", comments="")
           
           
# ----- Save a clean summary table for the report -----
summary = np.column_stack([
    theta_hat,                 # point estimate from y_obs
    theta_mean,                # bootstrap mean
    theta_sd,                  # bootstrap sd
    theta_ci[0, :],            # 2.5%
    theta_ci[1, :]             # 97.5%
])

np.savetxt(
    "results/prob1_summary.csv",
    summary,
    delimiter=",",
    header="theta_hat,boot_mean,boot_sd,ci2.5,ci97.5",
    comments=""
)

np.savetxt(
    "results/prob1_param_names.txt",
    np.array(names),
    fmt="%s"
)

