"""
T09: 段階反応モデル（GRM）の数理｜多値IRTの累積確率アプローチをPythonで導出

記事のPythonコードを統合したファイル。
記事URL: https://bigdata-analytics.jp/analytics/irt-grm-theory/

Author: Hiroyuki Matsumoto (Digital Boy LLC)
Repository: https://github.com/hiro1234567/irt-python
"""

# ============================================================
# Code Block 1
# ============================================================

import numpy as np
from scipy.special import expit  # ロジスティック関数
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 220,
    'axes.titlesize': 20,
    'axes.labelsize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'font.family': 'Hiragino Sans',
})

# ============================================================
# Code Block 2
# ============================================================

def grm_cumulative_prob(theta, alpha, delta_k):
    """GRMの累積確率 P*(X >= k | theta)"""
    return expit(alpha * (theta - delta_k))

def grm_category_prob(theta, alpha, deltas):
    """GRMのカテゴリ確率 P(X = k | theta)を全カテゴリぶん計算"""
    m = len(deltas) + 1  # カテゴリ数
    probs = np.zeros(m)

    # 累積確率を計算（P*_1 = 1, P*_{m+1} = 0）
    cum_probs = np.ones(m + 1)
    cum_probs[-1] = 0.0
    for k in range(1, m):
        cum_probs[k] = grm_cumulative_prob(theta, alpha, deltas[k - 1])

    # カテゴリ確率 = 累積確率の差分
    for k in range(m):
        probs[k] = cum_probs[k] - cum_probs[k + 1]

    return probs

# 数値例：α = 1.5, δ = [-1.5, -0.5, 0.5, 1.5]
alpha = 1.5
deltas = np.array([-1.5, -0.5, 0.5, 1.5])

# θ = 0.0 と θ = 2.0 での確認
for theta_val in [0.0, 2.0]:
    probs = grm_category_prob(theta_val, alpha, deltas)
    print(f"θ = {theta_val:.1f}: {[f'{p:.3f}' for p in probs]}, 合計 = {sum(probs):.3f}")

# ============================================================
# Code Block 3
# ============================================================

theta_range = np.linspace(-4, 4, 500)

# 全θに対してカテゴリ確率を計算
cat_probs = np.array([grm_category_prob(t, alpha, deltas) for t in theta_range])

fig, ax = plt.subplots(figsize=(10, 6))
category_labels = ['k=1', 'k=2', 'k=3', 'k=4', 'k=5']
colors = ['#E53935', '#FB8C00', '#43A047', '#1E88E5', '#8E24AA']

for k in range(5):
    ax.plot(theta_range, cat_probs[:, k], color=colors[k],
            linewidth=2.5, label=category_labels[k])

# 境界位置を破線で表示
for d in deltas:
    ax.axvline(d, color='gray', linestyle='--', alpha=0.4)

ax.set_xlabel('θ（能力）')
ax.set_ylabel('P(X = k | θ)')
ax.set_title(f'GRMカテゴリ確率曲線（α = {alpha}, δ = {list(deltas)}）')
ax.legend(loc='upper left')
ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.savefig('fig10_01_category_probability.png')
plt.close()

# ============================================================
# Code Block 4
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
alpha_values = [0.5, 1.5, 3.0]
alpha_labels = ['低識別力', '中識別力', '高識別力']

for idx, (a_val, a_label) in enumerate(zip(alpha_values, alpha_labels)):
    cat_probs_a = np.array([grm_category_prob(t, a_val, deltas) for t in theta_range])
    for k in range(5):
        axes[idx].plot(theta_range, cat_probs_a[:, k], color=colors[k],
                       linewidth=2, label=category_labels[k])
    axes[idx].set_xlabel('θ')
    axes[idx].set_ylabel('P(X = k | θ)')
    axes[idx].set_title(f'{a_label}（α = {a_val}）')
    axes[idx].legend(fontsize=10, loc='upper left')
    axes[idx].set_ylim(-0.02, 1.02)

plt.tight_layout()
plt.savefig('fig10_02_alpha_comparison.png')
plt.close()

# ============================================================
# Code Block 5
# ============================================================

cat_probs_neg = np.array([grm_category_prob(t, -1.5, deltas) for t in theta_range])

fig, ax = plt.subplots(figsize=(10, 6))
for k in range(5):
    ax.plot(theta_range, cat_probs_neg[:, k], color=colors[k],
            linewidth=2.5, label=category_labels[k])
ax.set_xlabel('θ')
ax.set_ylabel('P(X = k | θ)')
ax.set_title('α = −1.5 の場合（逆転項目）')
ax.legend(loc='upper right')
ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.savefig('fig10_03_negative_alpha.png')
plt.close()

# ============================================================
# Code Block 6
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))
for k_idx, dk in enumerate(deltas):
    cum_p = expit(alpha * (theta_range - dk))
    ax.plot(theta_range, cum_p, linewidth=2.5,
            label=f'P*(X >= {k_idx + 2} | θ), δ = {dk}')

ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('θ（能力）')
ax.set_ylabel('P*(X ≥ k | θ)')
ax.set_title(f'GRM累積確率曲線（α = {alpha}）')
ax.legend(loc='lower right')
ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.savefig('fig10_04_cumulative_curves.png')
plt.close()

# ============================================================
# Code Block 7
# ============================================================

def grm_item_information(theta_arr, alpha, deltas, dx=1e-5):
    """GRMの項目情報関数を数値微分で計算"""
    info = np.zeros_like(theta_arr)
    for i, t in enumerate(theta_arr):
        p = grm_category_prob(t, alpha, deltas)
        p_plus = grm_category_prob(t + dx, alpha, deltas)
        p_minus = grm_category_prob(t - dx, alpha, deltas)

        # 1階微分（中心差分）
        dp = (p_plus - p_minus) / (2 * dx)

        # 項目情報 = Σ (dp_k^2 / p_k) — ただし p_k > 0 の項のみ
        mask = p > 1e-10
        info[i] = np.sum(dp[mask] ** 2 / p[mask])

    return info

# ============================================================
# Code Block 8
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))
for a_val, a_label, col in zip([0.5, 1.5, 3.0],
                                ['α = 0.5', 'α = 1.5', 'α = 3.0'],
                                ['#FB8C00', '#43A047', '#1E88E5']):
    info_vals = grm_item_information(theta_range, a_val, deltas)
    ax.plot(theta_range, info_vals, linewidth=2.5, label=a_label, color=col)

ax.set_xlabel('θ（能力）')
ax.set_ylabel('I(θ)')
ax.set_title('GRM項目情報関数：識別力αの効果')
ax.legend()
plt.tight_layout()
plt.savefig('fig10_05_item_information.png')
plt.close()

# ============================================================
# Code Block 9
# ============================================================

from scipy.optimize import minimize_scalar

def grm_log_likelihood(theta, responses, alphas, deltas_list):
    """GRMの対数尤度を計算"""
    log_lik = 0.0
    for j, (resp, a_j, d_j) in enumerate(zip(responses, alphas, deltas_list)):
        probs = grm_category_prob(theta, a_j, d_j)
        k = resp - 1  # 0-indexed
        if probs[k] > 1e-15:
            log_lik += np.log(probs[k])
        else:
            log_lik += np.log(1e-15)
    return log_lik

def grm_map_estimate(responses, alphas, deltas_list, prior_mu=0.0, prior_sigma=1.0):
    """MAP推定で θ を求める"""
    def neg_log_posterior(theta):
        log_lik = grm_log_likelihood(theta, responses, alphas, deltas_list)
        log_prior = -0.5 * ((theta - prior_mu) / prior_sigma) ** 2
        return -(log_lik + log_prior)

    result = minimize_scalar(neg_log_posterior, bounds=(-4, 4), method='bounded')
    return result.x

# テスト：5項目の応答パターン [3, 4, 2, 5, 3] に対するMAP推定
# （5件法アンケートに5問回答した1人のデータを想定）
alphas_test = [1.5, 1.0, 2.0, 0.8, 1.2]
deltas_test = [
    np.array([-1.5, -0.5, 0.5, 1.5]),
    np.array([-1.0, 0.0, 1.0, 2.0]),
    np.array([-2.0, -1.0, 0.0, 1.0]),
    np.array([-0.5, 0.5, 1.5, 2.5]),
    np.array([-1.5, -0.5, 0.5, 1.5]),
]
responses_test = [3, 4, 2, 5, 3]

theta_map = grm_map_estimate(responses_test, alphas_test, deltas_test)
theta_mle_approx = minimize_scalar(
    lambda t: -grm_log_likelihood(t, responses_test, alphas_test, deltas_test),
    bounds=(-4, 4), method='bounded'
).x

print(f"MAP推定: θ = {theta_map:.3f}")
print(f"MLE推定: θ = {theta_mle_approx:.3f}")

# ============================================================
# Code Block 10
# ============================================================

theta_grid = np.linspace(-3, 3, 300)
log_liks = [grm_log_likelihood(t, responses_test, alphas_test, deltas_test)
            for t in theta_grid]
log_prior = -0.5 * theta_grid ** 2
log_posterior = np.array(log_liks) + log_prior

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta_grid, log_liks - max(log_liks), color='#1E88E5',
        linewidth=2.5, label='対数尤度（MLE）')
ax.plot(theta_grid, log_posterior - max(log_posterior), color='#E53935',
        linewidth=2.5, label='対数事後確率（MAP）')
ax.plot(theta_grid, log_prior - max(log_prior), color='gray',
        linewidth=1.5, linestyle='--', label='事前分布 N(0,1)')
ax.axvline(theta_mle_approx, color='#1E88E5', linestyle=':', alpha=0.7)
ax.axvline(theta_map, color='#E53935', linestyle=':', alpha=0.7)
ax.set_xlabel('θ')
ax.set_ylabel('正規化した対数確率')
ax.set_title('MLE vs MAP 推定の比較')
ax.legend()
plt.tight_layout()
plt.savefig('fig10_06_mle_vs_map.png')
plt.close()

# ============================================================
# Code Block 11
# ============================================================

np.random.seed(42)

N_persons = 500
N_items = 10
N_categories = 5

# 真値の設定
true_theta = np.random.normal(0, 1, N_persons)
true_alpha = np.array([0.8, 1.0, 1.2, 1.5, 2.0, 0.6, 1.8, 1.3, 2.5, 0.9])
true_deltas = [
    np.array([-1.5, -0.5, 0.5, 1.5]),   # バランス型
    np.array([-2.0, -1.0, 0.0, 1.0]),   # やや甘め
    np.array([-1.0, 0.0, 1.0, 2.0]),    # やや厳しめ
    np.array([-1.5, -0.5, 0.5, 1.5]),
    np.array([-2.0, -1.0, 0.0, 1.0]),
    np.array([-1.5, -0.5, 0.5, 1.5]),
    np.array([-1.0, 0.0, 1.0, 2.0]),
    np.array([-2.0, -1.0, 0.0, 1.0]),
    np.array([-1.5, -0.5, 0.5, 1.5]),
    np.array([-1.0, 0.0, 1.0, 2.0]),
]

# 応答データの生成
responses = np.zeros((N_persons, N_items), dtype=int)
for i in range(N_persons):
    for j in range(N_items):
        probs = grm_category_prob(true_theta[i], true_alpha[j], true_deltas[j])
        responses[i, j] = np.random.choice(range(1, N_categories + 1), p=probs)

print(f"応答データの形状: {responses.shape}")
print(f"カテゴリ分布: {dict(zip(*np.unique(responses, return_counts=True)))}")

# ============================================================
# Code Block 12
# ============================================================

theta_map_estimates = np.array([
    grm_map_estimate(responses[i], true_alpha, true_deltas)
    for i in range(N_persons)
])

# 真値との相関
corr = np.corrcoef(true_theta, theta_map_estimates)[0, 1]
print(f"θの真値 vs MAP推定の相関: r = {corr:.3f}")

# ============================================================
# Code Block 13
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(true_theta, theta_map_estimates, alpha=0.3, s=15, color='#1E88E5')
ax.plot([-3.5, 3.5], [-3.5, 3.5], 'r--', linewidth=1.5, label='y = x（完全一致）')
ax.set_xlabel('θ 真値')
ax.set_ylabel('θ MAP推定値')
ax.set_title(f'GRM MAP推定の精度（r = {corr:.3f}）')
ax.legend()
ax.set_aspect('equal')
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
plt.tight_layout()
plt.savefig('fig10_07_theta_recovery.png')
plt.close()

