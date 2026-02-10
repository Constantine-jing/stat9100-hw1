import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# --- warm start from Problem 1 (if available) ---
theta_init_np = np.loadtxt("results/prob1_summary.csv", delimiter=",", skiprows=1)[:, 0]  # theta_hat column
theta_init = torch.tensor(theta_init_np, dtype=torch.float32).unsqueeze(0)  # [1,4]
print("Warm start theta_init from Prob1 =", theta_init_np)

torch.manual_seed(42)
np.random.seed(42)


# fixed x grid
x_grid = torch.linspace(0, 5, 50)

def simulate_nonlinear(theta, x_grid, sigma=0.1):
    beta1, gamma1 = theta[:, 0:1], theta[:, 1:2]
    beta2, gamma2 = theta[:, 2:3], theta[:, 3:4]
    f_x = beta1 * torch.exp(-gamma1 * x_grid) + beta2 * torch.exp(-gamma2 * x_grid)
    return f_x + torch.randn_like(f_x) * sigma

def sample_prior(n):
    betas = torch.rand(n, 2) * 2.0 + 0.5       # [0.5, 2.5]
    gammas_raw = torch.rand(n, 2) * 1.5 + 0.1  # [0.1, 1.6]
    gammas, _ = torch.sort(gammas_raw, dim=1)  # enforce gamma1 < gamma2
    return torch.cat([betas[:, 0:1], gammas[:, 0:1], betas[:, 1:2], gammas[:, 1:2]], dim=1)

# scale theta to [0,1]-ish for stable classifier
def theta_scale(theta):
    beta1 = (theta[:,0:1] - 0.5) / 2.0
    gamma1 = (theta[:,1:2] - 0.1) / 1.5
    beta2 = (theta[:,2:3] - 0.5) / 2.0
    gamma2 = (theta[:,3:4] - 0.1) / 1.5
    return torch.cat([beta1,gamma1,beta2,gamma2], dim=1)

class LikelihoodClassifier(nn.Module):
    def __init__(self, d_in=54, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)  # logits
        )
    def forward(self, y_z, theta_s):
        x = torch.cat([y_z, theta_s], dim=1)
        return self.net(x).squeeze(1)

# --- simulate training data ---
N = 60000
theta = sample_prior(N)
y = simulate_nonlinear(theta, x_grid, sigma=0.1)

y_mean = y.mean()
y_std  = y.std()
y_z = (y - y_mean) / y_std

theta_s = theta_scale(theta)
theta_perm_s = theta_s[torch.randperm(N)]

# paired vs permuted
Y_all = torch.cat([y_z, y_z], dim=0)                # [2N,50]
T_all = torch.cat([theta_s, theta_perm_s], dim=0)   # [2N,4]
L_all = torch.cat([torch.ones(N), torch.zeros(N)], dim=0)  # [2N]

# shuffle + split
M = Y_all.shape[0]
perm = torch.randperm(M)
Y_all, T_all, L_all = Y_all[perm], T_all[perm], L_all[perm]

n_val = int(0.1 * M)
Y_va, T_va, L_va = Y_all[:n_val], T_all[:n_val], L_all[:n_val]
Y_tr, T_tr, L_tr = Y_all[n_val:], T_all[n_val:], L_all[n_val:]

print("Label mean:", L_all.float().mean().item(), "(should be ~0.5)")

# --- train classifier ---
net = LikelihoodClassifier()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
crit = nn.BCEWithLogitsLoss()

batch_size = 512
epochs = 8

for ep in range(1, epochs+1):
    net.train()
    perm2 = torch.randperm(Y_tr.shape[0])
    total = 0.0
    for i in range(0, Y_tr.shape[0], batch_size):
        idx = perm2[i:i+batch_size]
        yb, tb, lb = Y_tr[idx], T_tr[idx], L_tr[idx]

        logits = net(yb, tb)
        loss = crit(logits, lb)

        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * yb.shape[0]

    net.eval()
    with torch.no_grad():
        val_logits = net(Y_va, T_va)
        val_prob = torch.sigmoid(val_logits)
        val_acc = ((val_prob > 0.5).float() == L_va).float().mean().item()
        val_loss = crit(val_logits, L_va).item()

    print(f"Epoch {ep:02d} | train loss={total/Y_tr.shape[0]:.4f} | val loss={val_loss:.4f} | val acc={val_acc:.4f}")


net.eval()
for p in net.parameters():
    p.requires_grad_(False)
true_theta_test = torch.tensor([[1.8, 0.4, 1.2, 1.1]])


torch.manual_seed(999)
y_obs_raw = simulate_nonlinear(true_theta_test, x_grid, sigma=0.1)
y_obs_z = (y_obs_raw - y_mean) / y_std

                            # [1,50]
def u_to_theta(u):
    s = torch.sigmoid(u)
    beta1 = 0.5 + 2.0 * s[:, 0:1]
    beta2 = 0.5 + 2.0 * s[:, 2:3]
    g1 = 0.1 + 1.5 * s[:, 1:2]
    g2 = 0.1 + 1.5 * s[:, 3:4]
    gammas, _ = torch.sort(torch.cat([g1, g2], dim=1), dim=1)
    return torch.cat([beta1, gammas[:,0:1], beta2, gammas[:,1:2]], dim=1)

def theta_to_u(theta):
    beta1, gamma1, beta2, gamma2 = theta[:,0:1], theta[:,1:2], theta[:,2:3], theta[:,3:4]
    s0 = (beta1 - 0.5) / 2.0
    s1 = (gamma1 - 0.1) / 1.5
    s2 = (beta2 - 0.5) / 2.0
    s3 = (gamma2 - 0.1) / 1.5
    s = torch.cat([s0, s1, s2, s3], dim=1).clamp(1e-5, 1-1e-5)
    return torch.log(s) - torch.log(1 - s)


def neural_loglik(theta):
    # theta: [1,4] in original scale
    theta_s = theta_scale(theta)          # [1,4] scaled
    logit = net(y_obs_z, theta_s)         # scalar
    return logit
best = {"ll": -1e9, "theta": None}

R = 50
steps = 400
lr = 0.03


for r in range(R):
    if r == 0:
        # warm start from Prob1 theta_hat
        # make sure gammas are sorted
        theta_ws = theta_init.clone()
        g = theta_ws[:, [1,3]]
        g_sorted, _ = torch.sort(g, dim=1)
        theta_ws[:,1] = g_sorted[:,0]
        theta_ws[:,3] = g_sorted[:,1]

        u0 = theta_to_u(theta_ws)
        u = (u0 + 0.05 * torch.randn_like(u0)).detach().requires_grad_(True)
    else:
        u = torch.randn(1, 4, requires_grad=True)

    opt_u = torch.optim.Adam([u], lr=lr)

    for t in range(steps):
        theta = u_to_theta(u)
        ll = neural_loglik(theta)
        loss = -ll

        opt_u.zero_grad()
        loss.backward()
        opt_u.step()

    with torch.no_grad():
        theta = u_to_theta(u)
        ll = neural_loglik(theta).item()
        if ll > best["ll"]:
            best["ll"] = ll
            best["theta"] = theta.detach().cpu().numpy().ravel()

print("\nApprox MLE from neural likelihood:")
print("theta_mle [beta1, gamma1, beta2, gamma2] =", best["theta"])
print("true_theta =", true_theta_test.numpy().ravel())
print("best logit =", best["ll"])


theta_mle = best["theta"]  # [beta1,gamma1,beta2,gamma2] from your print

def f_np(x, beta1, gamma1, beta2, gamma2):
    return beta1*np.exp(-gamma1*x) + beta2*np.exp(-gamma2*x)

x_np = x_grid.numpy()
y_obs_np = y_obs_raw.numpy().ravel()

y_true = f_np(x_np, *true_theta_test.numpy().ravel())
y_mle  = f_np(x_np, *theta_mle)

plt.figure()
plt.plot(x_np, y_obs_np, "o", label="data y_obs")
plt.plot(x_np, y_true, "-", linewidth=2, label="true f(x)")
plt.plot(x_np, y_mle,  "--", linewidth=2, label="neural-lik MLE fit")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()
plt.savefig("figs/prob2_fit.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figs/prob2_fit.png")

# ====== 2D likelihood surface for (gamma1, gamma2) ======


beta1_hat, gamma1_hat, beta2_hat, gamma2_hat = theta_mle
gamma_true1, gamma_true2 = 0.4, 1.1

g = np.linspace(0.1, 1.6, 80)
G1, G2 = np.meshgrid(g, g)
mask = G1 < G2

theta_grid = np.zeros((mask.sum(), 4), dtype=np.float32)
theta_grid[:, 0] = beta1_hat
theta_grid[:, 2] = beta2_hat
theta_grid[:, 1] = G1[mask].ravel()
theta_grid[:, 3] = G2[mask].ravel()

theta_t = torch.tensor(theta_grid, dtype=torch.float32)
theta_s = theta_scale(theta_t)

# repeat y_obs_z for batch eval
y_rep = y_obs_z.repeat(theta_t.shape[0], 1)

with torch.no_grad():
    logits = net(y_rep, theta_s).cpu().numpy().ravel()

LL = np.full(G1.shape, np.nan, dtype=float)
LL[mask] = logits

LL_max = np.nanmax(LL)
with torch.no_grad():
    ll_true = neural_loglik(true_theta_test).item()  # logit at truth
    ll_hat  = best["ll"]                             # logit at MLE
    LR_true = 2 * (ll_hat - ll_true)

print("LR at truth =", LR_true, " (<= 5.99 means truth inside 95% region)")

import pandas as pd


names = ["beta1","gamma1","beta2","gamma2"]

# theta table: estimate + truth
truth = np.array([1.8, 0.4, 1.2, 1.1], dtype=float)
df_theta = pd.DataFrame({
    "Parameter": names,
    "Estimate": theta_mle,
    "Truth": truth
})
df_theta.to_csv("results/prob2_theta.csv", index=False)

# scalar metrics
df_metrics = pd.DataFrame({
    "best_logit": [best["ll"]],
    "LR_at_truth": [LR_true],
    "LR95_thresh_df2": [5.99]
})
df_metrics.to_csv("results/prob2_metrics.csv", index=False)

print("Saved: results/prob2_theta.csv, results/prob2_metrics.csv")

LR = 2 * (LL_max - LL)   # likelihood-ratio statistic on grid

# 95% LR contour in 2D uses chi-square(2) ≈ 5.99
lr95 = 5.99

plt.figure()
cf = plt.contourf(G1, G2, LL, levels=30)
plt.colorbar(cf, label="neural log-likelihood (logit)")

# 95% region boundary
plt.contour(G1, G2, LR, levels=[lr95], colors="k", linewidths=2)

# mark MLE and truth
plt.plot(gamma1_hat, gamma2_hat, "wo", markersize=8, markeredgecolor="k", label="MLE")
plt.plot(gamma_true1, gamma_true2, "r*", markersize=12, label="Truth")

plt.xlabel("gamma1")
plt.ylabel("gamma2")
plt.title("Neural likelihood surface for (gamma1, gamma2)")
plt.legend()
plt.savefig("figs/prob2_gamma_surface.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figs/prob2_gamma_surface.png")

# optional: report projected ranges of the 95% region
inside = (LR <= lr95) & mask
g1_min, g1_max = np.nanmin(G1[inside]), np.nanmax(G1[inside])
g2_min, g2_max = np.nanmin(G2[inside]), np.nanmax(G2[inside])
print(f"LR95 projected ranges: gamma1 in [{g1_min:.3f}, {g1_max:.3f}], gamma2 in [{g2_min:.3f}, {g2_max:.3f}]")
