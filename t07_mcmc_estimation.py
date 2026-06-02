"""
T07: MCMCによるIRT母数推定：ギブスサンプラーとPyMCでベイズIRTをPythonで実装

記事のPythonコードを統合したファイル。
記事URL: https://bigdata-analytics.jp/analytics/irt-mcmc-estimation/

Author: Hiroyuki Matsumoto (Digital Boy LLC)
Repository: https://github.com/hiro1234567/irt-python

実行方法:
  python t07_mcmc_estimation.py
    → CLIで一気に実行。グラフはウィンドウで順次表示される
  MPLBACKEND=Agg python t07_mcmc_estimation.py
    → グラフ表示せず数値だけ確認
  Jupyterで Code Block を1つずつコピペして対話的に試すのも可
"""

# ============================================================
# Code Block 1
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

SAVE_FIGS = False  # True にすると plt.savefig() で画像保存される

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

# 事前分布×尤度→事後分布の視覚化

# 既知の項目パラメータ（ラッシュモデル: α = 1）
delta = np.array([-2.0, -0.5, 0.5, 1.0, 2.0])
# この受験者の応答パターン: 易しい3問に正答、難しい2問に不正答
x = np.array([1, 1, 1, 0, 0])

# θの範囲を設定
theta_grid = np.linspace(-4, 4, 500)

# 事前分布: N(0, 1)
prior = norm.pdf(theta_grid, 0, 1)

# 尤度関数: 各θでの応答パターンの生起確率
def likelihood(theta, delta, x):
    """応答パターンの尤度を計算"""
    p = 1 / (1 + np.exp(-(theta - delta)))
    return np.prod(p**x * (1 - p)**(1 - x))

lik = np.array([likelihood(t, delta, x) for t in theta_grid])

# 事後分布（正規化前）
posterior_unnorm = prior * lik
# 正規化
posterior = posterior_unnorm / np.trapezoid(posterior_unnorm, theta_grid)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta_grid, prior, '--', color='#2196F3', linewidth=2, label='事前分布 N(0,1)')
ax.plot(theta_grid, lik / np.max(lik) * np.max(posterior),
        ':', color='#FF9800', linewidth=2, label='尤度（スケール調整済み）')
ax.plot(theta_grid, posterior, '-', color='#4CAF50', linewidth=2.5, label='事後分布')
ax.axvline(theta_grid[np.argmax(posterior)], color='gray', linestyle='-.', alpha=0.5)
ax.set_xlabel('θ')
ax.set_ylabel('密度')
ax.set_title('ベイズ推定：事前分布 × 尤度 → 事後分布')
ax.legend(loc='upper right')
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig07_01_prior_likelihood_posterior.png')
plt.close()

# ============================================================
# Code Block 3
# ============================================================

# MH法の簡単な例: N(2, 0.5^2) からのサンプリング
np.random.seed(42)

def target_log_prob(x):
    """目標分布の対数密度: N(2, 0.5^2)"""
    return -0.5 * ((x - 2) / 0.5) ** 2

n_samples = 10000
samples = np.zeros(n_samples)
samples[0] = 0.0  # 初期値（目標分布から離れた位置）
proposal_sd = 0.5  # 提案分布の標準偏差
n_accepted = 0

for t in range(1, n_samples):
    # 1. 候補を提案: 現在値の周りの正規分布から
    candidate = samples[t-1] + np.random.normal(0, proposal_sd)
    # 2. 受容確率を計算（対数スケールで）
    log_a = target_log_prob(candidate) - target_log_prob(samples[t-1])
    # 3. 受容/棄却
    if np.log(np.random.uniform()) < log_a:
        samples[t] = candidate
        n_accepted += 1
    else:
        samples[t] = samples[t-1]

acceptance_rate = n_accepted / (n_samples - 1)

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左: トレースプロット（最初の500サンプル）
axes[0].plot(samples[:500], linewidth=0.5, color='#1976D2')
axes[0].axhline(2.0, color='red', linestyle='--', alpha=0.7, label='真の平均 = 2.0')
axes[0].set_xlabel('反復回数')
axes[0].set_ylabel('θ')
axes[0].set_title(f'トレースプロット（受容率 {acceptance_rate:.1%}）')
axes[0].legend()

# 右: ヒストグラム vs 真の分布
burn_in = 1000
axes[1].hist(samples[burn_in:], bins=50, density=True, alpha=0.7, color='#42A5F5', label='MCMCサンプル')
x_range = np.linspace(0, 4, 200)
axes[1].plot(x_range, norm.pdf(x_range, 2, 0.5), 'r-', linewidth=2, label='真の分布 N(2, 0.5²)')
axes[1].set_xlabel('θ')
axes[1].set_ylabel('密度')
axes[1].set_title('事後分布の近似')
axes[1].legend()

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig07_02_mh_simple_example.png')
plt.close()

# ============================================================
# Code Block 4
# ============================================================

# シミュレーションデータの生成
np.random.seed(123)

N = 200  # 受験者数
J = 5    # 項目数
true_theta = np.random.normal(0, 1, N)  # 受験者の真の能力
true_delta = np.array([-2.0, -0.5, 0.5, 1.0, 2.0])  # 項目の真の位置

# ラッシュモデルで応答行列を生成
prob = 1 / (1 + np.exp(-(true_theta[:, None] - true_delta[None, :])))
X = (np.random.uniform(size=(N, J)) < prob).astype(int)

print(f"応答行列のサイズ: {X.shape}")
print(f"各項目の正答率: {X.mean(axis=0).round(3)}")
print(f"各受験者の平均得点: {X.mean(axis=1).mean():.3f}")

# ============================================================
# Code Block 5
# ============================================================

def log_prior_theta(theta):
    """θの事前分布: N(0, 1) の対数密度"""
    return -0.5 * theta**2

def log_prior_delta(delta):
    """δの事前分布: N(0, 5) の対数密度"""
    return -0.5 * (delta / 5)**2

def log_likelihood_person(theta_i, delta, x_i):
    """受験者iの対数尤度（ラッシュモデル）"""
    logit = theta_i - delta
    log_p = -np.logaddexp(0, -logit)  # log(p)
    log_q = -np.logaddexp(0, logit)   # log(1-p)
    return np.sum(x_i * log_p + (1 - x_i) * log_q)

def log_likelihood_item(theta, delta_j, x_j):
    """項目jの対数尤度（ラッシュモデル）"""
    logit = theta - delta_j
    log_p = -np.logaddexp(0, -logit)
    log_q = -np.logaddexp(0, logit)
    return np.sum(x_j * log_p + (1 - x_j) * log_q)

# MCMCの設定
n_iter = 5000
burn_in = 1000
proposal_sd_theta = 0.5  # θの提案分布の標準偏差
proposal_sd_delta = 0.3  # δの提案分布の標準偏差

# パラメータの初期値
theta_chain = np.zeros((n_iter, N))
delta_chain = np.zeros((n_iter, J))
theta_chain[0] = np.zeros(N)  # θの初期値: 0
delta_chain[0] = np.zeros(J)  # δの初期値: 0

# 受容回数の記録
n_accept_theta = np.zeros(N)
n_accept_delta = np.zeros(J)

np.random.seed(42)

for t in range(1, n_iter):
    current_theta = theta_chain[t-1].copy()
    current_delta = delta_chain[t-1].copy()

    # --- θの更新（受験者ごと） ---
    for i in range(N):
        # 候補を提案
        candidate = current_theta[i] + np.random.normal(0, proposal_sd_theta)
        # 対数受容確率を計算
        log_a = (log_likelihood_person(candidate, current_delta, X[i])
                 + log_prior_theta(candidate)
                 - log_likelihood_person(current_theta[i], current_delta, X[i])
                 - log_prior_theta(current_theta[i]))
        # 受容/棄却
        if np.log(np.random.uniform()) < log_a:
            current_theta[i] = candidate
            n_accept_theta[i] += 1

    # --- δの更新（項目ごと） ---
    for j in range(J):
        candidate = current_delta[j] + np.random.normal(0, proposal_sd_delta)
        log_a = (log_likelihood_item(current_theta, candidate, X[:, j])
                 + log_prior_delta(candidate)
                 - log_likelihood_item(current_theta, current_delta[j], X[:, j])
                 - log_prior_delta(current_delta[j]))
        if np.log(np.random.uniform()) < log_a:
            current_delta[j] = candidate
            n_accept_delta[j] += 1

    theta_chain[t] = current_theta
    delta_chain[t] = current_delta

    if (t + 1) % 1000 == 0:
        print(f"反復 {t+1}/{n_iter} 完了")

# 受容率
print(f"\nθの平均受容率: {(n_accept_theta / (n_iter - 1)).mean():.1%}")
print(f"δの平均受容率: {(n_accept_delta / (n_iter - 1)).mean():.1%}")

# ============================================================
# Code Block 6
# ============================================================

# δのトレースプロット
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for j in range(J):
    row, col = j // 3, j % 3
    axes[row, col].plot(delta_chain[:, j], linewidth=0.3, color='#1976D2')
    axes[row, col].axhline(true_delta[j], color='red', linestyle='--',
                            linewidth=1.5, label=f'真値 = {true_delta[j]}')
    axes[row, col].axvline(burn_in, color='orange', linestyle=':', alpha=0.7, label='バーンイン')
    axes[row, col].set_title(f'項目{j+1}（δ{j+1}）')
    axes[row, col].set_xlabel('反復回数')
    axes[row, col].set_ylabel('δ')
    axes[row, col].legend(fontsize=10)

# 空のパネルを非表示
axes[1, 2].set_visible(False)

plt.suptitle('項目パラメータ δ のトレースプロット', fontsize=18, y=1.02)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig07_03_trace_plot_delta.png')
plt.close()

# ============================================================
# Code Block 7
# ============================================================

# バーンイン後のサンプル
delta_post = delta_chain[burn_in:]
theta_post = theta_chain[burn_in:]

# δの事後統計量
print("項目パラメータ δ の推定結果:")
print(f"{'項目':>4} {'真値':>8} {'事後平均':>8} {'事後SD':>8} {'95%CI下限':>10} {'95%CI上限':>10}")
for j in range(J):
    post_mean = delta_post[:, j].mean()
    post_sd = delta_post[:, j].std()
    ci_low = np.percentile(delta_post[:, j], 2.5)
    ci_high = np.percentile(delta_post[:, j], 97.5)
    print(f"{j+1:>4} {true_delta[j]:>8.2f} {post_mean:>8.3f} {post_sd:>8.3f} {ci_low:>10.3f} {ci_high:>10.3f}")

# ============================================================
# Code Block 8
# ============================================================

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for j in range(J):
    axes[j].hist(delta_post[:, j], bins=40, density=True, alpha=0.7, color='#42A5F5')
    axes[j].axvline(true_delta[j], color='red', linewidth=2, linestyle='--', label=f'真値 = {true_delta[j]}')
    axes[j].axvline(delta_post[:, j].mean(), color='green', linewidth=2, linestyle='-', label=f'事後平均 = {delta_post[:, j].mean():.2f}')
    axes[j].set_title(f'項目{j+1}')
    axes[j].set_xlabel('δ')
    if j == 0:
        axes[j].set_ylabel('密度')
    axes[j].legend(fontsize=8)

plt.suptitle('項目パラメータ δ の事後分布', fontsize=18, y=1.05)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig07_04_posterior_delta.png')
plt.close()

# ============================================================
# Code Block 9
# ============================================================

import pymc as pm
import arviz as az
print(f"PyMC {pm.__version__}, ArviZ {az.__version__}")

# ============================================================
# Code Block 10
# ============================================================

import pymc as pm
import arviz as az

# 2PLモデル用のデータ生成
np.random.seed(456)
N_2pl = 300
J_2pl = 8
true_theta_2pl = np.random.normal(0, 1, N_2pl)
true_alpha_2pl = np.array([0.8, 1.0, 1.2, 0.6, 1.5, 1.0, 0.9, 1.3])
true_delta_2pl = np.array([-1.5, -1.0, -0.3, 0.0, 0.5, 1.0, 1.5, 2.0])

# 応答データ生成
logit_2pl = true_alpha_2pl[None, :] * (true_theta_2pl[:, None] - true_delta_2pl[None, :])
prob_2pl = 1 / (1 + np.exp(-logit_2pl))
X_2pl = (np.random.uniform(size=(N_2pl, J_2pl)) < prob_2pl).astype(int)

print(f"2PLモデル用データ: {N_2pl}人 × {J_2pl}項目")
print(f"各項目の正答率: {X_2pl.mean(axis=0).round(3)}")

# ============================================================
# Code Block 11
# ============================================================

# PyMCによる2PLモデルの定義
with pm.Model() as irt_2pl:
    # 事前分布の定義
    # 受験者の能力θ: 標準正規分布
    theta = pm.Normal('theta', mu=0, sigma=1, shape=N_2pl)
    # 項目の識別力α: 対数正規分布（正値制約）
    alpha = pm.LogNormal('alpha', mu=0, sigma=0.5, shape=J_2pl)
    # 項目の位置δ: 弱情報事前分布
    delta = pm.Normal('delta', mu=0, sigma=5, shape=J_2pl)

    # 2PLモデルの正答確率
    logit_p = alpha[None, :] * (theta[:, None] - delta[None, :])
    p = pm.math.sigmoid(logit_p)

    # 尤度（ベルヌーイ分布）
    obs = pm.Bernoulli('obs', p=p, observed=X_2pl)

    # NUTSサンプラーでMCMCを実行
    trace = pm.sample(2000, tune=1000, chains=2, cores=1,
                       random_seed=42, return_inferencedata=True)

# ============================================================
# Code Block 12
# ============================================================

# 項目パラメータの推定結果
print("=== 項目パラメータの推定結果（2PLモデル） ===\n")
delta_summary = az.summary(trace, var_names=['delta'])
alpha_summary = az.summary(trace, var_names=['alpha'])

print("δ（項目位置）:")
print(f"{'項目':>4} {'真値':>8} {'事後平均':>8} {'事後SD':>8} {'95%HDI下限':>10} {'95%HDI上限':>10} {'R-hat':>8}")
for j in range(J_2pl):
    row = delta_summary.iloc[j]
    print(f"{j+1:>4} {true_delta_2pl[j]:>8.2f} {row['mean']:>8.3f} {row['sd']:>8.3f} "
          f"{row['hdi_3%']:>10.3f} {row['hdi_97%']:>10.3f} {row['r_hat']:>8.3f}")

print(f"\nα（識別力）:")
print(f"{'項目':>4} {'真値':>8} {'事後平均':>8} {'事後SD':>8} {'95%HDI下限':>10} {'95%HDI上限':>10} {'R-hat':>8}")
for j in range(J_2pl):
    row = alpha_summary.iloc[j]
    print(f"{j+1:>4} {true_alpha_2pl[j]:>8.2f} {row['mean']:>8.3f} {row['sd']:>8.3f} "
          f"{row['hdi_3%']:>10.3f} {row['hdi_97%']:>10.3f} {row['r_hat']:>8.3f}")

# ============================================================
# Code Block 13
# ============================================================

# トレースプロット（δとα）
az.plot_trace(trace, var_names=['delta', 'alpha'], figsize=(14, 16))
plt.suptitle('2PLモデルの収束診断：トレースプロット', fontsize=16, y=1.01)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig07_05_pymc_trace_2pl.png')
plt.close()

# ============================================================
# Code Block 14
# ============================================================

# 真値 vs 事後平均の比較プロット
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# δの比較
delta_means = [delta_summary.iloc[j]['mean'] for j in range(J_2pl)]
delta_ci_low = [delta_summary.iloc[j]['hdi_3%'] for j in range(J_2pl)]
delta_ci_high = [delta_summary.iloc[j]['hdi_97%'] for j in range(J_2pl)]
axes[0].errorbar(true_delta_2pl, delta_means,
                  yerr=[np.array(delta_means) - np.array(delta_ci_low),
                        np.array(delta_ci_high) - np.array(delta_means)],
                  fmt='o', color='#1976D2', capsize=5, markersize=8, label='事後平均 ± 95%HDI')
axes[0].plot([-2.5, 2.5], [-2.5, 2.5], 'r--', alpha=0.7, label='y = x（完全一致）')
axes[0].set_xlabel('真の δ')
axes[0].set_ylabel('推定された δ（事後平均）')
axes[0].set_title('項目位置 δ の推定精度')
axes[0].legend()
axes[0].set_aspect('equal')

# αの比較
alpha_means = [alpha_summary.iloc[j]['mean'] for j in range(J_2pl)]
alpha_ci_low = [alpha_summary.iloc[j]['hdi_3%'] for j in range(J_2pl)]
alpha_ci_high = [alpha_summary.iloc[j]['hdi_97%'] for j in range(J_2pl)]
axes[1].errorbar(true_alpha_2pl, alpha_means,
                  yerr=[np.array(alpha_means) - np.array(alpha_ci_low),
                        np.array(alpha_ci_high) - np.array(alpha_means)],
                  fmt='o', color='#E65100', capsize=5, markersize=8, label='事後平均 ± 95%HDI')
axes[1].plot([0.3, 2.0], [0.3, 2.0], 'r--', alpha=0.7, label='y = x（完全一致）')
axes[1].set_xlabel('真の α')
axes[1].set_ylabel('推定された α（事後平均）')
axes[1].set_title('識別力 α の推定精度')
axes[1].legend()
axes[1].set_aspect('equal')

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig07_06_true_vs_estimated.png')
plt.close()

