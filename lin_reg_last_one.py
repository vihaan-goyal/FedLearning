import itertools
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# ================================================================
# Config (feel free to tweak)
# ================================================================
NUM_USERS = 100
K = 2                 # <- generic subset size
ROUNDS = 2000               # make 2k if you want; 1k is faster to iterate
LOCAL_EPOCHS = 5            # local steps per selection
DIM = 100
SAMPLES_PER_USER = 3000 // NUM_USERS
LR = 0.01
SEEDS = [0, 1, 2, 3, 4]     # five seeds
NUM_SAMPLES_FOR_Q = 1_000_000   # Monte Carlo to estimate q over K-subsets
seed=0
IPFP_TOL = 1e-12
IPFP_MAX_ITERS = 10000

# ================================================================
# Utilities: masked IPFP (unchanged in spirit, generalized for any K)
# ================================================================
def build_mask(n, subsets):
    """
    Boolean mask M (n x m) with 1-based subset entries:
    M[i, j] = True iff (i+1) is in subset j; rows are 0..n-1, subsets are tuples in {1,..,n}.
    """
    m = len(subsets)
    M = np.zeros((n, m), dtype=bool)
    for j, Aj in enumerate(subsets):
        for i in Aj:       # i is 1..n
            M[i-1, j] = True
    return M

def initialize_Y(p, q, M, mode="uniform"):
    n, m = M.shape
    Y = np.zeros((n, m), dtype=float)
    for j in range(m):
        rows = np.where(M[:, j])[0]
        if len(rows) == 0:
            if q[j] > 0:
                raise ValueError(f"Column j={j} has empty subset but q[j]={q[j]}>0.")
            else:
                continue
        if q[j] == 0.0:
            continue
        if mode == "uniform":
            Y[rows, j] = q[j] / len(rows)
        elif mode == "p_proportional":
            pj = p[rows]
            s = pj.sum()
            if s > 0:
                Y[rows, j] = q[j] * (pj / s)
            else:
                Y[rows, j] = q[j] / len(rows)
        else:
            raise ValueError("Unknown init mode")
    return Y

def ipfp_masked(p, q, M, tol=1e-10, max_iter=10000, verbose=False):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    n, m = M.shape

    if p.ndim != 1 or q.ndim != 1:
        raise ValueError("p and q must be 1D arrays")
    if not np.all(p >= 0) or not np.all(q >= 0):
        raise ValueError("p and q must be nonnegative")
    if abs(p.sum() - q.sum()) > 1e-12:
        raise ValueError(f"Totals must match: sum(p)={p.sum()} vs sum(q)={q.sum()}")

    Y = initialize_Y(p, q, M, mode="uniform")
    rows_idx = [np.where(M[i, :])[0] for i in range(n)]
    cols_idx = [np.where(M[:, j])[0] for j in range(m)]

    for it in range(max_iter):
        # Row scaling
        row_sums = Y.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            row_scale = np.ones(n)
            need = (row_sums > 0)
            row_scale[need] = p[need] / row_sums[need]
        infeasible_rows = (row_sums == 0) & (p > 0)
        if np.any(infeasible_rows):
            i = np.where(infeasible_rows)[0][0]
            raise ValueError(f"Infeasible: row i={i} has no support but p[i]={p[i]}>0")
        for i in range(n):
            js = rows_idx[i]
            if js.size:
                Y[i, js] *= row_scale[i]

        # Column scaling
        col_sums = Y.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            col_scale = np.ones(m)
            needc = (col_sums > 0)
            col_scale[needc] = q[needc] / col_sums[needc]
        infeasible_cols = (col_sums == 0) & (q > 0)
        if np.any(infeasible_cols):
            j = np.where(infeasible_cols)[0][0]
            raise ValueError(f"Infeasible: column j={j} has no support but q[j]={q[j]}>0")
        for j in range(m):
            is_ = cols_idx[j]
            if is_.size:
                Y[is_, j] *= col_scale[j]

        row_err = np.linalg.norm(Y.sum(axis=1) - p, ord=1)
        col_err = np.linalg.norm(Y.sum(axis=0) - q, ord=1)
        if verbose and it % 100 == 0:
            print(f"[{it}] row_err={row_err:.3e}, col_err={col_err:.3e}")
        if max(row_err, col_err) <= tol:
            return Y, {"iters": it+1, "row_err": row_err, "col_err": col_err}

    return Y, {"iters": max_iter, "row_err": row_err, "col_err": col_err, "warning": "max_iter reached"}

def recover_T_from_Y(Y, q, M, p, strategy_for_q0="p_restricted"):
    n, m = Y.shape
    q = np.asarray(q, dtype=float)
    T = np.zeros_like(Y)

    pos = q > 0
    T[:, pos] = Y[:, pos] / q[pos]

    zero_cols = np.where(~pos)[0]
    for j in zero_cols:
        rows = np.where(M[:, j])[0]
        if rows.size == 0:
            raise ValueError(f"Column j={j} has empty subset; cannot define T.")
        if strategy_for_q0 == "p_restricted":
            w = p[rows]
            s = w.sum()
            if s > 0:
                T[rows, j] = w / s
            else:
                T[rows, j] = 1.0 / rows.size
        elif strategy_for_q0 == "uniform":
            T[rows, j] = 1.0 / rows.size
        else:
            raise ValueError("Unknown strategy_for_q0")

    T[~M] = 0.0
    col_sums = T.sum(axis=0)
    for j in range(m):
        if col_sums[j] != 0:
            T[:, j] /= col_sums[j]
    return T

def solve_T_with_given_subsets(p, q, subsets, tol=1e-10, max_iter=10000, verbose=False):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    n = p.size
    m = len(subsets)
    if q.size != m:
        raise ValueError(f"Length of q ({q.size}) must equal number of subsets ({m}).")

    M = build_mask(n, subsets)  # 1-based subsets
    if abs(p.sum() - q.sum()) > 1e-12:
        p = p / p.sum()
        q = q / q.sum()

    Y, info = ipfp_masked(p, q, M, tol=tol, max_iter=max_iter, verbose=verbose)
    T = recover_T_from_Y(Y, q, M, p, strategy_for_q0="p_restricted")

    col_err = np.max(np.abs(T.sum(axis=0) - 1))
    recon_p = T @ q
    p_err = np.max(np.abs(recon_p - p))
    info.update({"T_col_err_inf": float(col_err), "p_match_err_inf": float(p_err)})
    return T, subsets, M, info

def column_users_and_weights(T, subsets, j):
    """
    Return (0-based user ids, normalized weights) for column j (subset of size K).
    """
    Aj_1 = subsets[j]                      # 1-based subset tuple
    rows = np.array([i-1 for i in Aj_1])   # to 0-based
    w = T[rows, j]
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        w = np.full_like(w, 1.0 / len(w))
    else:
        w = w / s
    return rows, w

# ================================================================
# Linear regression helpers
# ================================================================
def local_train_lr(X, y, w_init, epochs=1, lr=0.01):
    w = w_init.copy()
    n = len(y)
    for _ in range(epochs):
        preds = X @ w
        grad = (X.T @ (preds - y)) / n
        w -= lr * grad
    return w

def global_weighted_mse(w, user_data, p):
    tot = 0.0
    for i, (Xi, yi) in enumerate(user_data):
        preds = Xi @ w
        loss = mean_squared_error(yi, preds)
        tot += p[i] * loss
    return tot

# ================================================================
# Build skewed p, r and K-subsets; estimate q via Monte Carlo
# ================================================================
def make_skew_distributions(N):
    idx_1 = np.arange(1, N + 1, dtype=float)
    # r_i ∝ exp(+i) (prior for availability)
    r = np.power(idx_1, .8)
    r /= r.sum()
    # p_i ∝ exp(-i) (importance for objective)
    p = np.power(idx_1[::-1], .8)
    p /= p.sum()
    return p, r

def all_K_subsets_1based(N, K):
    return [tuple(c) for c in combinations(range(1, N + 1), K)]

def estimate_q_by_mc(subsets, r, N, K, num_samples=1_000_000, rng=None):
    """
    Estimate q over the provided 'subsets' (1-based tuples of size K)
    by Monte Carlo: S ~ Choice(N, K, p=r, replace=False).
    Efficiently tallies unique rows using np.unique.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    # map subset tuple -> index
    subset_to_idx = {s: j for j, s in enumerate(subsets)}
    # sample many subsets
    draws = rng.choice(N, size=(num_samples, K), replace=True, p=r)
    draws.sort(axis=1)  # canonical order
    # convert to 1-based to match 'subsets'
    draws_1b = draws + 1
    # unique counts
    uniq, counts = np.unique(draws_1b, axis=0, return_counts=True)
    q_counts = np.zeros(len(subsets), dtype=np.int64)
    # assign counts to indices
    for row, c in zip(uniq, counts):
        tup = tuple(row.tolist())
        j = subset_to_idx.get(tup, None)
        if j is not None:
            q_counts[j] += c
    q = q_counts.astype(float)
    q /= q.sum()
    return q

# ================================================================
# Main experiment: FedAVOT(K) vs FedAvg(K, EXACT) vs FedAvg(full)
# ================================================================
all_losses_FedAVOT = []
all_losses_faK   = []
all_losses_full  = []

print(f"\n[Seed {seed}]  (K={K})")
rng = np.random.RandomState(seed)
np.random.seed(seed)

# --- Data ---
X_full, y_full = make_regression(n_samples=3000, n_features=DIM, noise=0.1, random_state=seed)
X_full = StandardScaler().fit_transform(X_full)

# --- Partition to users ---
perm = np.random.permutation(len(X_full))
X = X_full[perm]
y = y_full[perm]
user_data = [
    (X[i * SAMPLES_PER_USER:(i + 1) * SAMPLES_PER_USER],
        y[i * SAMPLES_PER_USER:(i + 1) * SAMPLES_PER_USER])
    for i in range(NUM_USERS)
]

# --- Distributions and subset family ---
p, r = make_skew_distributions(NUM_USERS)
subsets_K = all_K_subsets_1based(NUM_USERS, K)
print(f"  Total subsets of size K: C({NUM_USERS},{K}) = {len(subsets_K)}")

# --- Estimate q via Monte Carlo sampling from r ---
q = estimate_q_by_mc(subsets_K, r, NUM_USERS, K, num_samples=NUM_SAMPLES_FOR_Q, rng=rng)

# --- Solve T via masked IPFP for this subset family and q ---
T, subsets_used, M_mask, info = solve_T_with_given_subsets(
    p, q, subsets_K, tol=IPFP_TOL, max_iter=IPFP_MAX_ITERS, verbose=False
)
print(f"  IPFP iters={info.get('iters')} | T col err∞={info.get('T_col_err_inf'):.2e} | p match err∞={info.get('p_match_err_inf'):.2e}")



for seed in SEEDS:
    
    print(f"\n[Seed {seed}]  (K={K})")
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    # --- Initialize globals for the three methods ---
    d = user_data[0][0].shape[1]
    w_FedAVOT = np.zeros(d)
    w_faK   = np.zeros(d)
    w_full  = np.zeros(d)

    losses_ot   = []
    losses_faK  = []
    losses_full = []

    q_cum = np.cumsum(q)

    # --- Precompute local data references for speed ---
    Xs = [ud[0] for ud in user_data]
    ys = [ud[1] for ud in user_data]

    # --- Training loop ---
    for r_idx in range(ROUNDS):
        # sample a subset j ~ q
        u = rng.random()
        j = int(np.searchsorted(q_cum, u, side="right"))

        # members of subset j for this round
        users_S, weights_T = column_users_and_weights(T, subsets_used, j)  # length K

        # ------------------ FedAVOT(K) ------------------
        local_models_ot = []
        for uid in users_S:
            theta_i = local_train_lr(Xs[uid], ys[uid], w_FedAVOT, epochs=LOCAL_EPOCHS, lr=LR)
            local_models_ot.append(theta_i)
        # convex combination by T column weights
        w_next_ot = np.zeros_like(w_FedAVOT)
        for coeff, theta in zip(weights_T, local_models_ot):
            w_next_ot += coeff * theta
        w_FedAVOT = w_next_ot
        losses_ot.append(global_weighted_mse(w_FedAVOT, user_data, p))

        # ---------------- FedAvg(K), EXACT: sum_{i in S} (N/K)*p_i * theta_i ----------------
        local_models_faK = []
        for uid in users_S:
            theta_i = local_train_lr(Xs[uid], ys[uid], w_faK, epochs=LOCAL_EPOCHS, lr=LR)
            local_models_faK.append((uid, theta_i))
        w_next_faK = np.zeros_like(w_faK)
        scale = NUM_USERS / float(K)
        # pees = np.asarray(list(p[uid] for (uid, theta_i) in local_models_faK)).sum()
        # scale = pees
        for uid, theta_i in local_models_faK:
            w_next_faK += (p[uid] * scale) * theta_i
            # w_next_faK += (p[uid] / scale) * theta_i
        w_faK = w_next_faK
        losses_faK.append(global_weighted_mse(w_faK, user_data, p))

        # ---------------- FedAvg (full): sum_i p_i * theta_i ----------------
        local_models_full = []
        for uid in range(NUM_USERS):
            theta_i = local_train_lr(Xs[uid], ys[uid], w_full, epochs=LOCAL_EPOCHS, lr=LR)
            local_models_full.append((uid, theta_i))
        w_next_full = np.zeros_like(w_full)
        for uid, theta_i in local_models_full:
            w_next_full += p[uid] * theta_i
        w_full = w_next_full
        losses_full.append(global_weighted_mse(w_full, user_data, p))

        if (r_idx + 1) % 200 == 0:
            print(f"  Round {r_idx+1}/{ROUNDS}: FedAVOT={losses_ot[-1]:.6f} | "
                  f"FedAvg(K)={losses_faK[-1]:.6f} | FedAvg(full)={losses_full[-1]:.6f}")

    all_losses_FedAVOT.append(np.array(losses_ot))
    all_losses_faK.append(np.array(losses_faK))
    all_losses_full.append(np.array(losses_full))

# ================================================================
# Aggregate across seeds
# ================================================================
all_losses_FedAVOT = np.vstack(all_losses_FedAVOT)
all_losses_faK   = np.vstack(all_losses_faK)
all_losses_full  = np.vstack(all_losses_full)

loss_FedAVOT_mean = all_losses_FedAVOT.mean(axis=0)
loss_FedAVOT_std  = all_losses_FedAVOT.std(axis=0)

loss_faK_mean = all_losses_faK.mean(axis=0)
loss_faK_std  = all_losses_faK.std(axis=0)

loss_full_mean = all_losses_full.mean(axis=0)
loss_full_std  = all_losses_full.std(axis=0)

# For the bar plots (use last-seed p and r — they are deterministic anyway)
p, r = make_skew_distributions(NUM_USERS)

# ================================================================
# Plot: loss curves + bars for p and r
# Colors: FedAVOT=blue, FedAvg(K)=orange, FedAvg(full)=red
# ================================================================
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[2.2, 1.0])

# --- (1) Loss curves (span top row) ---
ax0 = fig.add_subplot(gs[0, :])
x = np.arange(ROUNDS)

ax0.plot(x, loss_FedAVOT_mean, label=f"FedAVOT (K={K})", color="tab:blue", linewidth=1.8)
ax0.fill_between(x, loss_FedAVOT_mean - loss_FedAVOT_std, loss_FedAVOT_mean + loss_FedAVOT_std,
                 alpha=0.15, color="tab:blue")

ax0.plot(x, loss_faK_mean, label=f"FedAvg (K={K})", color="tab:orange", linewidth=1.8)
ax0.fill_between(x, loss_faK_mean - loss_faK_std, loss_faK_mean + loss_faK_std,
                 alpha=0.15, color="tab:orange")

ax0.plot(x, loss_full_mean, label="FedAvg (full devices)", color="tab:red", linewidth=1.8)
ax0.fill_between(x, loss_full_mean - loss_full_std, loss_full_mean + loss_full_std,
                 alpha=0.15, color="tab:red")

# --- (1) Loss plot ---
ax0.set_yscale("log")  # optional
ax0.set_xlabel("Communication round", fontsize=14)
ax0.set_ylabel(r"Global loss MSE loss - Log Scale", fontsize=14)
ax0.set_title(f"Linear Regression — FedAVOT vs FedAvg (K={K}) vs FedAvg(full) — mean ± std over {len(SEEDS)} seeds", fontsize=14)
ax0.tick_params(axis="both", which="major", labelsize=18)
ax0.grid(True, alpha=0.3, which="both", linestyle="--")
ax0.legend(fontsize=14)  # make legend larger

# --- (2) Bar plot for p ---
ax1 = fig.add_subplot(gs[1, 0])
ax1.bar(np.arange(1, NUM_USERS+1), p, color="gray", edgecolor="black", linewidth=0.6)
ax1.set_title("Importance Distribution $p$", fontsize=14)
ax1.set_xlabel("User index", fontsize=14)
ax1.set_ylabel("p", fontsize=14)
ax1.tick_params(axis="both", which="major", labelsize=18)
ax1.grid(axis="y", alpha=0.3)

# --- (3) Bar plot for r ---
ax2 = fig.add_subplot(gs[1, 1])
ax2.bar(np.arange(1, NUM_USERS+1), r, color="gray", edgecolor="black", linewidth=0.6)
ax2.set_title("Availability Distribution $r", fontsize=14)
ax2.set_xlabel("User index", fontsize=14)
ax2.set_ylabel("r", fontsize=14)
ax2.tick_params(axis="both", which="major", labelsize=18)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"linreg_K{K}_FedAVOT_vs_FedAvg_full_with_priors.png", dpi=300, bbox_inches="tight")
plt.savefig(f"linreg_K{K}_FedAVOT_vs_FedAvg_full_with_priors.pdf", bbox_inches="tight")
plt.show()