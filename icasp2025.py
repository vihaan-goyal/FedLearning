# ================================================================
# FedAVOT vs FedAvg(K) vs FedAvg(full) on MNIST (Generic K, Monte Carlo q from r)
# + normalized FedAvg(K): weights p_i / sum_{j in S} p_j
# Colors: FedAVOT=blue, FedAvg(K)=orange, normalized FedAvg(K)=green, FedAvg(full)=red
# ================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Experiment Configuration
# -----------------------------
NUM_USERS = 100
NUM_CLASSES = 10
K = 2                          # generic K (size of each participating subset)
ROUNDS = 2000                  # 2000 is fine too; 1000 for quicker runs
LOCAL_EPOCHS = 3               # local steps per selection
DIM = 784
TOTAL_SAMPLES = 3000           # total examples used from MNIST
SAMPLES_PER_USER = TOTAL_SAMPLES // NUM_USERS
LR = 0.1
SEEDS = [0, 1, 2, 3, 4]        # replicate for mean±std
IPFP_TOL = 1e-10
IPFP_MAX_ITERS = 4000
NUM_SAMPLES_FOR_Q = 1_000_000  # MC draws to estimate q over K-subsets
seed=0

# -----------------------------
# Fetch & preprocess MNIST once
# -----------------------------
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X_full, y_full = mnist.data[:TOTAL_SAMPLES] / 255.0, mnist.target.astype(int)[:TOTAL_SAMPLES]
X_full = StandardScaler().fit_transform(X_full)

# ================================================================
# Masked IPFP on an arbitrary K-subset family (1-based subset tuples)
# ================================================================
def build_mask(n, subsets):
    m = len(subsets)
    M = np.zeros((n, m), dtype=bool)
    for j, Aj in enumerate(subsets):
        for i in Aj:       # i in 1..n
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
        if verbose and it % 200 == 0:
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

    M = build_mask(n, subsets)  # 1-based
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
    Aj_1 = subsets[j]                      # 1-based subset tuple
    rows = np.array([i-1 for i in Aj_1])   # to 0-based indices
    w = T[rows, j]
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        w = np.full_like(w, 1.0 / len(w))
    else:
        w = w / s
    return rows, w

# ================================================================
# Distributions and subset family helpers
# ================================================================
def make_skew_distributions(N):
    idx_1 = np.arange(1, N + 1, dtype=float)
    # r_i ∝ exp(+i) (availability prior)
    r = np.ones(N)
    r /= r.sum()
    # p_i ∝ exp(-i) (importance)
    p = np.power(idx_1[::-1], 4)
    p /= p.sum()
    print(r)
    print(p)
    return p, r

def all_K_subsets_1based(N, K):
    return [tuple(c) for c in combinations(range(1, N + 1), K)]

def estimate_q_by_mc(subsets, r, N, K, num_samples=1_000_000, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    subset_to_idx = {s: j for j, s in enumerate(subsets)}
    draws = rng.choice(N, size=(num_samples, K), replace=True, p=r)
    draws.sort(axis=1)
    draws_1b = draws + 1
    uniq, counts = np.unique(draws_1b, axis=0, return_counts=True)
    q_counts = np.zeros(len(subsets), dtype=np.int64)
    for row, c in zip(uniq, counts):
        tup = tuple(row.tolist())
        j = subset_to_idx.get(tup, None)
        if j is not None:
            q_counts[j] += c
    q = q_counts.astype(float)
    q /= q.sum()
    return q

# ================================================================
# Multinomial logistic regression (NumPy)
# ================================================================
def init_theta(d, C):
    return {"W": np.zeros((d, C)), "b": np.zeros(C)}

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def forward(X, theta):
    return X @ theta["W"] + theta["b"]

def ce_loss_and_grad(X, y, theta, l2=0.0):
    n, d = X.shape
    C = theta["W"].shape[1]
    scores = forward(X, theta)
    P = softmax(scores)
    Y = np.zeros((n, C))
    Y[np.arange(n), y] = 1.0
    eps = 1e-12
    loss = -np.sum(Y * np.log(P + eps)) / n
    loss += 0.5 * l2 * np.sum(theta["W"] ** 2)
    G = (P - Y) / n
    dW = X.T @ G + l2 * theta["W"]
    db = np.sum(G, axis=0)
    return loss, {"W": dW, "b": db}

def local_train(theta, X, y, epochs=1, lr=0.1, l2=0.0):
    th = {"W": theta["W"].copy(), "b": theta["b"].copy()}
    for _ in range(epochs):
        _, grads = ce_loss_and_grad(X, y, th, l2=l2)
        th["W"] -= lr * grads["W"]
        th["b"] -= lr * grads["b"]
    return th

def weighted_average_thetas(thetas, weights):
    W = np.sum([w * t["W"] for t, w in zip(thetas, weights)], axis=0)
    b = np.sum([w * t["b"] for t, w in zip(thetas, weights)], axis=0)
    return {"W": W, "b": b}

def global_loss(theta, user_data, p, l2=0.0):
    tot = 0.0
    for i, (Xi, yi) in enumerate(user_data):
        li, _ = ce_loss_and_grad(Xi, yi, theta, l2=l2)
        tot += p[i] * li
    return tot

# ================================================================
# Data partitioning
# ================================================================
def make_user_data(X_full, y_full, N):
    perm = np.random.permutation(len(X_full))
    X = X_full[perm]
    y = y_full[perm]
    return [(X[i*SAMPLES_PER_USER:(i+1)*SAMPLES_PER_USER],
             y[i*SAMPLES_PER_USER:(i+1)*SAMPLES_PER_USER]) for i in range(N)]

# ================================================================
# Run the methods per seed
# ================================================================
all_losses_FedAVOT = []
all_losses_faK   = []
all_losses_full  = []

print(f"\n[Seed {seed}]  (K={K})")
rng = np.random.RandomState(seed)
np.random.seed(seed)

# user datasets
user_data = make_user_data(X_full, y_full, NUM_USERS)

# skewed p and r
p, r = make_skew_distributions(NUM_USERS)

# K-subsets and q via Monte Carlo from r
subsets_K = all_K_subsets_1based(NUM_USERS, K)
print(f"  Total subsets size K: C({NUM_USERS},{K}) = {len(subsets_K)}")
q = estimate_q_by_mc(subsets_K, r, NUM_USERS, K, num_samples=NUM_SAMPLES_FOR_Q, rng=rng)

# Solve T for this family
T, subsets_used, M_mask, info = solve_T_with_given_subsets(
    p, q, subsets_K, tol=IPFP_TOL, max_iter=IPFP_MAX_ITERS, verbose=False
)
print(f"  IPFP iters={info.get('iters')} | T col err∞={info.get('T_col_err_inf'):.2e} | p match err∞={info.get('p_match_err_inf'):.2e}")

for seed in SEEDS:
    print(f"\n[Seed {seed}]  (K={K})")
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    # initialize global models
    theta_FedAVOT   = init_theta(DIM, NUM_CLASSES)
    theta_faK     = init_theta(DIM, NUM_CLASSES)
    theta_full    = init_theta(DIM, NUM_CLASSES)

    losses_ot, losses_faK, losses_full = [], [], []

    # Precompute for speed
    Xs = [ud[0] for ud in user_data]
    ys = [ud[1] for ud in user_data]
    q_cum = np.cumsum(q)

    for t in range(ROUNDS):
        # sample subset j ~ q
        u = rng.random()
        j = int(np.searchsorted(q_cum, u, side="right"))
        users_S, weights_T = column_users_and_weights(T, subsets_used, j)  # indices (0-based) and normalized weights

        # ------------------ FedAVOT(K) ------------------
        local_thetas = []
        for uid in users_S:
            ti = local_train(theta_FedAVOT, Xs[uid], ys[uid], epochs=LOCAL_EPOCHS, lr=LR, l2=0.0)
            local_thetas.append(ti)
        theta_FedAVOT = weighted_average_thetas(local_thetas, weights_T)
        losses_ot.append(global_loss(theta_FedAVOT, user_data, p, l2=0.0))

        # ---------------- FedAvg(K) EXACT: sum_{i in S} (N/K)*p_i * theta_i ----------------
        local_thetas_fak = []
        for uid in users_S:
            ti = local_train(theta_faK, Xs[uid], ys[uid], epochs=LOCAL_EPOCHS, lr=LR, l2=0.0)
            local_thetas_fak.append((uid, ti))
        scale = NUM_USERS / float(K)
        # weighted sum without normalization (EXACT rule)
        W_acc = np.zeros_like(theta_faK["W"])
        b_acc = np.zeros_like(theta_faK["b"])
        for uid, ti in local_thetas_fak:
            coeff = p[uid] * scale
            W_acc += coeff * ti["W"]
            b_acc += coeff * ti["b"]
        theta_faK = {"W": W_acc, "b": b_acc}
        losses_faK.append(global_loss(theta_faK, user_data, p, l2=0.0))


        # ---------------- FedAvg (full): sum_i p_i * theta_i ----------------
        W_acc = np.zeros_like(theta_full["W"])
        b_acc = np.zeros_like(theta_full["b"])
        for uid in range(NUM_USERS):
            ti = local_train(theta_full, Xs[uid], ys[uid], epochs=LOCAL_EPOCHS, lr=LR, l2=0.0)
            W_acc += p[uid] * ti["W"]
            b_acc += p[uid] * ti["b"]
        theta_full = {"W": W_acc, "b": b_acc}
        losses_full.append(global_loss(theta_full, user_data, p, l2=0.0))

        if (t + 1) % 200 == 0:
            print(f"  round {t+1}/{ROUNDS}  FedAVOT={losses_ot[-1]:.4f} | "
                  f"FedAvg(K)={losses_faK[-1]:.4f} | "
                  f"FedAvg(full)={losses_full[-1]:.4f}")

    all_losses_FedAVOT.append(np.array(losses_ot))
    all_losses_faK.append(np.array(losses_faK))
    all_losses_full.append(np.array(losses_full))

# ================================================================
# Aggregate and plot (loss curves + p/r bars)
# ================================================================
L_ot    = np.vstack(all_losses_FedAVOT)
L_faK   = np.vstack(all_losses_faK)
L_full  = np.vstack(all_losses_full)

mean_ot,   std_ot   = L_ot.mean(axis=0),   L_ot.std(axis=0)
mean_faK,  std_faK  = L_faK.mean(axis=0),  L_faK.std(axis=0)
mean_full, std_full = L_full.mean(axis=0), L_full.std(axis=0)

# p and r (deterministic functions of N)
p, r = make_skew_distributions(NUM_USERS)

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[2.2, 1.0])

# --- Top row: loss curves
ax0 = fig.add_subplot(gs[0, :])
x = np.arange(ROUNDS)

# FedAVOT (blue)
ax0.plot(x, mean_ot, label=f"FedAVOT (K={K})", color="tab:blue", linewidth=1.8)
ax0.fill_between(x, mean_ot - std_ot, mean_ot + std_ot, alpha=0.15, color="tab:blue")

# FedAvg(K) (orange)
ax0.plot(x, mean_faK, label=f"FedAvg (K={K})", color="tab:orange", linewidth=1.8)
ax0.fill_between(x, mean_faK - std_faK, mean_faK + std_faK, alpha=0.15, color="tab:orange")

# FedAvg (full) (red)
ax0.plot(x, mean_full, label="FedAvg (full devices)", color="tab:red", linewidth=1.8)
ax0.fill_between(x, mean_full - std_full, mean_full + std_full, alpha=0.15, color="tab:red")

# --- Top: Loss plot
# ax0.set_yscale("log")  # optional
ax0.set_xlabel("Communication round", fontsize=14)
ax0.set_ylabel(r"Global Cross Entropy Loss", fontsize=14)
ax0.set_title(f"MNIST — FedAVOT vs FedAvg (K={K}) vs FedAvg(full) — mean ± std over {len(SEEDS)} seeds", fontsize=14)
ax0.tick_params(axis="both", which="major", labelsize=18)
ax0.grid(True, alpha=0.3, which="both", linestyle="--")
ax0.legend(fontsize=14)

# --- Bottom left: p bar plot
ax1 = fig.add_subplot(gs[1, 0])
ax1.bar(np.arange(1, NUM_USERS+1), p, color="gray", edgecolor="black", linewidth=0.6)
ax1.set_title("Importance Distribution $p$", fontsize=14)
ax1.set_xlabel("User index", fontsize=14)
ax1.set_ylabel("p", fontsize=14)
ax1.tick_params(axis="both", which="major", labelsize=18)
ax1.grid(axis="y", alpha=0.3)

# --- Bottom right: r bar plot
ax2 = fig.add_subplot(gs[1, 1])
ax2.bar(np.arange(1, NUM_USERS+1), r, color="gray", edgecolor="black", linewidth=0.6)
ax2.set_title("Selection Distribution $r$", fontsize=14)
ax2.set_xlabel("User index", fontsize=14)
ax2.set_ylabel("r", fontsize=14)
ax2.tick_params(axis="both", which="major", labelsize=18)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"mnist_K{K}_FedAVOT_vs_FedAvg_full_with_priors_and_normalized.png", dpi=300, bbox_inches="tight")
plt.savefig(f"mnist_K{K}_FedAVOT_vs_FedAvg_full_with_priors_and_normalized.pdf", bbox_inches="tight")
plt.show()