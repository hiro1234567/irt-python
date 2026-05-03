"""
T10: GRMの母数推定とMCMC：段階反応モデルの識別力・境界パラメータをベイズ推定する

記事のPythonコードを統合したファイル。
記事URL: https://bigdata-analytics.jp/analytics/irt-grm-mcmc-estimation/

Author: Hiroyuki Matsumoto (Digital Boy LLC)
Repository: https://github.com/hiro1234567/irt-python
"""

# ============================================================
# Code Block 1
# ============================================================

import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 220,
    'axes.titlesize': 20,
    'axes.labelsize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 13,
    'font.family': 'Hiragino Sans',
})

# ============================================================
# Code Block 2
# ============================================================

def grm_cumulative_prob(theta, alpha, delta_k):
    """カテゴリk以上の累積確率"""
    return expit(alpha * (theta - delta_k))

def grm_category_prob(theta, alpha, deltas):
    """各カテゴリの反応確率"""
    m = len(deltas) + 1
    probs = np.zeros(m)
    cum_probs = np.ones(m + 1)
    cum_probs[-1] = 0.0
    for k in range(1, m):
        cum_probs[k] = grm_cumulative_prob(theta, alpha, deltas[k - 1])
    for k in range(m):
        probs[k] = cum_probs[k] - cum_probs[k + 1]
    return probs

# ============================================================
# Code Block 3
# ============================================================

np.random.seed(42)
N, J, K = 300, 5, 5

true_theta = np.random.normal(0, 1, N)
true_alpha = np.array([1.2, 0.8, 1.5, 2.0, 1.0])
true_deltas = [
    np.array([-1.5, -0.5, 0.5, 1.5]),
    np.array([-2.0, -1.0, 0.0, 1.0]),
    np.array([-1.0, 0.0, 1.0, 2.0]),
    np.array([-1.5, -0.5, 0.5, 1.5]),
    np.array([-2.0, -0.5, 0.5, 2.0]),
]

responses = np.zeros((N, J), dtype=int)
for i in range(N):
    for j in range(J):
        probs = grm_category_prob(true_theta[i], true_alpha[j], true_deltas[j])
        probs = np.clip(probs, 0, None)
        probs /= probs.sum()
        responses[i, j] = np.random.choice(K, p=probs)

# ============================================================
# Code Block 4
# ============================================================

def log_prior_alpha(alpha, mu=0.0, sigma=1.0):
    if alpha <= 0:
        return -np.inf
    return -0.5 * ((np.log(alpha) - mu) / sigma) ** 2 - np.log(alpha)

def log_prior_delta(delta, sigma=5.0):
    return -0.5 * (delta / sigma) ** 2

def log_prior_theta(theta, sigma=1.0):
    return -0.5 * (theta / sigma) ** 2

def log_likelihood_person_grm(theta_i, alphas, deltas_list, responses_i):
    ll = 0.0
    for j in range(len(alphas)):
        probs = grm_category_prob(theta_i, alphas[j], deltas_list[j])
        k = responses_i[j]
        if probs[k] > 1e-15:
            ll += np.log(probs[k])
        else:
            ll += np.log(1e-15)
    return ll

def log_likelihood_item_grm(alpha_j, deltas_j, thetas, responses_j):
    ll = 0.0
    for i in range(len(thetas)):
        probs = grm_category_prob(thetas[i], alpha_j, deltas_j)
        k = responses_j[i]
        if probs[k] > 1e-15:
            ll += np.log(probs[k])
        else:
            ll += np.log(1e-15)
    return ll

# ============================================================
# Code Block 5
# ============================================================

n_iter = 3000
burn_in = 1000

alpha_chain = np.zeros((n_iter, J))
delta_chain = np.zeros((n_iter, J, K - 1))
theta_chain = np.zeros((n_iter, N))

# 初期値
alpha_chain[0] = np.ones(J)
for j in range(J):
    delta_chain[0, j] = np.linspace(-1.5, 1.5, K - 1)
theta_chain[0] = np.zeros(N)

prop_sd_alpha = 0.1
prop_sd_delta = 0.15
prop_sd_theta = 0.4

np.random.seed(123)

for t in range(1, n_iter):
    cur_alpha = alpha_chain[t - 1].copy()
    cur_delta = [delta_chain[t - 1, j].copy() for j in range(J)]
    cur_theta = theta_chain[t - 1].copy()

    # --- θ の更新 ---
    for i in range(N):
        cand = cur_theta[i] + np.random.normal(0, prop_sd_theta)
        log_a = (log_likelihood_person_grm(cand, cur_alpha, cur_delta, responses[i])
                 + log_prior_theta(cand)
                 - log_likelihood_person_grm(cur_theta[i], cur_alpha, cur_delta, responses[i])
                 - log_prior_theta(cur_theta[i]))
        if np.log(np.random.uniform()) < log_a:
            cur_theta[i] = cand

    # --- α の更新 ---
    for j in range(J):
        cand_alpha = cur_alpha[j] + np.random.normal(0, prop_sd_alpha)
        if cand_alpha > 0:
            log_a = (log_likelihood_item_grm(cand_alpha, cur_delta[j], cur_theta, responses[:, j])
                     + log_prior_alpha(cand_alpha)
                     - log_likelihood_item_grm(cur_alpha[j], cur_delta[j], cur_theta, responses[:, j])
                     - log_prior_alpha(cur_alpha[j]))
            if np.log(np.random.uniform()) < log_a:
                cur_alpha[j] = cand_alpha

    # --- δ の更新（順序制約あり）---
    for j in range(J):
        for k_idx in range(K - 1):
            cand_d = cur_delta[j].copy()
            cand_d[k_idx] += np.random.normal(0, prop_sd_delta)
            # 順序制約のチェック
            if k_idx > 0 and cand_d[k_idx] <= cand_d[k_idx - 1]:
                continue
            if k_idx < K - 2 and cand_d[k_idx] >= cand_d[k_idx + 1]:
                continue
            log_a = (log_likelihood_item_grm(cur_alpha[j], cand_d, cur_theta, responses[:, j])
                     + log_prior_delta(cand_d[k_idx])
                     - log_likelihood_item_grm(cur_alpha[j], cur_delta[j], cur_theta, responses[:, j])
                     - log_prior_delta(cur_delta[j][k_idx]))
            if np.log(np.random.uniform()) < log_a:
                cur_delta[j] = cand_d

    alpha_chain[t] = cur_alpha
    for j in range(J):
        delta_chain[t, j] = cur_delta[j]
    theta_chain[t] = cur_theta

# ============================================================
# Code Block 6
# ============================================================

alpha_post = alpha_chain[burn_in:]
delta_post = delta_chain[burn_in:]

# ============================================================
# Code Block 7
# ============================================================

def grm_item_information(theta_arr, alpha, deltas, dx=1e-5):
    """項目情報関数 I_j(θ) = Σ (p'_k)^2 / p_k"""
    info = np.zeros(len(theta_arr))
    for i, t in enumerate(theta_arr):
        p = grm_category_prob(t, alpha, deltas)
        p_plus = grm_category_prob(t + dx, alpha, deltas)
        p_minus = grm_category_prob(t - dx, alpha, deltas)
        dp = (p_plus - p_minus) / (2 * dx)
        for k in range(len(p)):
            if p[k] > 1e-10:
                info[i] += dp[k] ** 2 / p[k]
    return info

def grm_category_information(theta_arr, alpha, deltas, dx=1e-5):
    """カテゴリ情報関数 (p'_k)^2 / p_k を各カテゴリについて返す"""
    m = len(deltas) + 1
    cat_info = np.zeros((len(theta_arr), m))
    for i, t in enumerate(theta_arr):
        p = grm_category_prob(t, alpha, deltas)
        p_plus = grm_category_prob(t + dx, alpha, deltas)
        p_minus = grm_category_prob(t - dx, alpha, deltas)
        dp = (p_plus - p_minus) / (2 * dx)
        for k in range(m):
            if p[k] > 1e-10:
                cat_info[i, k] = dp[k] ** 2 / p[k]
    return cat_info

theta_range = np.linspace(-4, 4, 500)
alpha_ex = 1.5
deltas_ex = np.array([-1.5, -0.5, 0.5, 1.5])

cat_info = grm_category_information(theta_range, alpha_ex, deltas_ex)
item_info = grm_item_information(theta_range, alpha_ex, deltas_ex)

fig, ax = plt.subplots(figsize=(10, 6))
colors5 = ['#E53935', '#FB8C00', '#43A047', '#1E88E5', '#8E24AA']
for k in range(5):
    ax.plot(theta_range, cat_info[:, k], color=colors5[k], linewidth=1.5,
            linestyle='--', label=f'カテゴリ k={k+1}')
ax.plot(theta_range, item_info, color='black', linewidth=2.5, label='項目情報関数（合計）')
ax.set_xlabel('θ（能力）')
ax.set_ylabel('I(θ)')
ax.set_title(f'カテゴリ情報関数と項目情報関数（α = {alpha_ex}）')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('fig11_07_category_information.png')
plt.close()

# ============================================================
# Code Block 8
# ============================================================

theta_range = np.linspace(-4, 4, 500)

info_2pl = grm_item_information(theta_range, 1.5, np.array([0.0]))
info_3cat = grm_item_information(theta_range, 1.5, np.array([-1.0, 1.0]))
info_5cat = grm_item_information(theta_range, 1.5, np.array([-1.5, -0.5, 0.5, 1.5]))
info_7cat = grm_item_information(theta_range, 1.5,
                                  np.array([-2.0, -1.2, -0.4, 0.4, 1.2, 2.0]))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta_range, info_2pl, linewidth=2, label='2カテゴリ（2値）', color='#9E9E9E', linestyle=':')
ax.plot(theta_range, info_3cat, linewidth=2, label='3カテゴリ', color='#FB8C00')
ax.plot(theta_range, info_5cat, linewidth=2, label='5カテゴリ', color='#43A047')
ax.plot(theta_range, info_7cat, linewidth=2, label='7カテゴリ', color='#1E88E5')
ax.set_xlabel('θ（能力）')
ax.set_ylabel('I(θ)')
ax.set_title('カテゴリ数を増やすと情報量は増加する（α = 1.5）')
ax.legend()
plt.tight_layout()
plt.savefig('fig11_08_category_count_info.png')
plt.close()

# ============================================================
# Code Block 9
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

test_info = np.zeros_like(theta_range)
est_alpha = alpha_post.mean(axis=0)
est_deltas = [delta_post[:, j].mean(axis=0) for j in range(J)]

for j in range(J):
    item_inf = grm_item_information(theta_range, est_alpha[j], est_deltas[j])
    ax.plot(theta_range, item_inf, linewidth=1.5, linestyle='--',
            label=f'項目{j+1}（α = {est_alpha[j]:.2f}）', alpha=0.7)
    test_info += item_inf

ax.plot(theta_range, test_info, color='black', linewidth=2.5, label='テスト情報関数')
ax.set_xlabel('θ（能力）')
ax.set_ylabel('I(θ)')
ax.set_title('テスト情報関数と標準誤差')
ax.legend(fontsize=9, loc='upper left')
plt.tight_layout()
plt.savefig('fig11_09_test_information.png')
plt.close()

