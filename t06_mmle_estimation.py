"""
T06: IRTの周辺最尤推定法（MMLE）：EMアルゴリズムとEAP推定をPythonで実装

記事のPythonコードを統合したファイル。
記事URL: https://bigdata-analytics.jp/analytics/irt-mmle-estimation/

Author: Hiroyuki Matsumoto (Digital Boy LLC)
Repository: https://github.com/hiro1234567/irt-python

実行方法:
  python t06_mmle_estimation.py
    → CLIで一気に実行。グラフはウィンドウで順次表示される
  MPLBACKEND=Agg python t06_mmle_estimation.py
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

from numpy.polynomial.hermite import hermgauss

def get_quadrature(R=21):
    """ガウス=エルミート求積点と重みを取得する"""
    points, weights = hermgauss(R)
    # Hermite多項式の求積点をN(0,1)のスケールに変換
    X_r = points * np.sqrt(2)
    A_r = weights / np.sqrt(np.pi)
    return X_r, A_r

# 21点の求積点と重み
X_r, A_r = get_quadrature(R=21)

print(f"求積点の数: {len(X_r)}")
print(f"求積点の範囲: [{X_r[0]:.3f}, {X_r[-1]:.3f}]")
print(f"重みの合計: {A_r.sum():.6f}（1に近いはず）")

# ============================================================
# Code Block 3
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

theta_range = np.linspace(-4.5, 4.5, 500)
pdf = norm.pdf(theta_range, 0, 1)
ax.plot(theta_range, pdf, 'k-', linewidth=2, label='母集団分布 g(θ) = N(0, 1)')

# 棒の高さは重みに比例
bar_heights = A_r / A_r.max() * pdf.max() * 0.95
for r in range(len(X_r)):
    ax.bar(X_r[r], bar_heights[r], width=0.45, alpha=0.45,
           color='steelblue', edgecolor='navy', linewidth=0.8)
    ax.plot(X_r[r], -0.005, 'v', color='darkred', markersize=8)

ax.set_xlabel('θ')
ax.set_ylabel('密度 / 求積重み')
ax.set_title('ガウス求積法：連続分布を離散点で近似する')
ax.legend(loc='upper right')
ax.set_xlim(-4.5, 4.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig01_quadrature.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 4
# ============================================================

def rasch_prob(theta, delta):
    """ラッシュモデルの正答確率を計算する"""
    diff = theta - delta
    return 1.0 / (1.0 + np.exp(-diff))


def mmle_rasch_em(X, R=21, max_cycles=100, tol=1e-6):
    """
    ラッシュモデルのMMLE推定（EMアルゴリズム）
    X: 応答行列（N×J）。全正答・全不正答の行は除外済みであること
    R: 求積点の数
    max_cycles: EMサイクルの最大回数
    tol: 収束判定の閾値（対数尤度の変化量）
    """
    N, J = X.shape

    # ガウス求積点と重みの取得
    X_r, A_r = get_quadrature(R)

    # δの初期値
    item_scores = X.sum(axis=0)
    deltas = -np.log((item_scores + 0.5) / (N - item_scores + 0.5))

    delta_history = [deltas.copy()]
    ll_history = []

    for cycle in range(max_cycles):
        # --- E-step ---
        # 各求積点での正答確率 P_rj: (R, J)
        P_rj = rasch_prob(X_r[:, np.newaxis], deltas[np.newaxis, :])

        # 各受験者×各求積点での尤度 L_ir: (N, R)
        L_ir = np.ones((N, R))
        for r in range(R):
            for j in range(J):
                L_ir[:, r] *= P_rj[r, j]**X[:, j] * (1 - P_rj[r, j])**(1 - X[:, j])

        # 事後確率: posterior_ir = L_ir * A_r / Σ(L_ir * A_r)
        posterior = L_ir * A_r[np.newaxis, :]
        posterior_sum = posterior.sum(axis=1, keepdims=True)
        posterior_sum = np.clip(posterior_sum, 1e-300, None)
        posterior = posterior / posterior_sum

        # 周辺対数尤度
        ll = np.sum(np.log(np.clip(posterior_sum.flatten(), 1e-300, None)))
        ll_history.append(ll)

        # 期待人数 n_r: (R,) = 各求積点の期待人数
        n_r = posterior.sum(axis=0)

        # 期待得点 c_rj: (R, J)
        c_rj = np.zeros((R, J))
        for j in range(J):
            c_rj[:, j] = (X[:, j:j+1] * posterior).sum(axis=0)

        # --- M-step: Newton-Raphsonでδを更新 ---
        for j in range(J):
            for _ in range(10):
                p_r = rasch_prob(X_r, deltas[j])
                # 勾配: -Σ(c_rj - n_r * p_r)
                gradient = -np.sum(c_rj[:, j] - n_r * p_r)
                # ヘシアン: -Σ(n_r * p_r * (1 - p_r))
                hessian = -np.sum(n_r * p_r * (1 - p_r))
                if abs(hessian) < 1e-10:
                    break
                update = gradient / hessian
                deltas[j] -= update
                if abs(update) < 1e-6:
                    break

        delta_history.append(deltas.copy())

        # 収束判定
        if cycle > 0 and abs(ll_history[-1] - ll_history[-2]) < tol:
            print(f"収束しました（{cycle + 1}サイクル）")
            break
    else:
        print(f"最大サイクル数（{max_cycles}）に達しました")

    return deltas, delta_history, ll_history

# ============================================================
# Code Block 5
# ============================================================

# シミュレーションデータの生成
np.random.seed(42)

# テスト設定
N_sim = 500
J_sim = 8
true_thetas = np.random.normal(0, 1, N_sim)
true_deltas = np.array([-2.0, -1.2, -0.5, 0.0, 0.3, 0.8, 1.5, 2.2])

print("=== テスト設定 ===")
print(f"受験者数: {N_sim}人")
print(f"項目数: {J_sim}問")
print(f"真のδ: {true_deltas}")

# ============================================================
# Code Block 6
# ============================================================

# 応答データの生成
P_true = rasch_prob(true_thetas[:, np.newaxis], true_deltas[np.newaxis, :])
X_sim = (np.random.rand(N_sim, J_sim) < P_true).astype(int)

# 全正答・全不正答を除外
row_sums = X_sim.sum(axis=1)
valid_mask = (row_sums > 0) & (row_sums < J_sim)
X_valid = X_sim[valid_mask]
print(f"有効な受験者数: {X_valid.shape[0]}人")

# MMLE推定
est_deltas, delta_hist, ll_hist = mmle_rasch_em(X_valid)

print(f"\n=== 推定結果 ===")
print(f"推定δ: {np.round(est_deltas, 3)}")
print(f"真のδ: {true_deltas}")
print(f"RMSE: {np.sqrt(np.mean((est_deltas - true_deltas)**2)):.4f}")

# ============================================================
# Code Block 7
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# δの収束過程
ax = axes[0]
delta_history = np.array(delta_hist)
for j in range(J_sim):
    ax.plot(delta_history[:, j], marker='o', markersize=3,
            label=f'δ_{j+1}（真値={true_deltas[j]:.1f}）')
    ax.axhline(y=true_deltas[j], color=f'C{j}', linestyle='--', alpha=0.3)
ax.set_xlabel('EMサイクル')
ax.set_ylabel('δの推定値')
ax.set_title('EMアルゴリズムによるδの収束')
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

# 周辺対数尤度の推移
ax = axes[1]
ax.plot(ll_hist, 'b-o', markersize=4)
ax.set_xlabel('EMサイクル')
ax.set_ylabel('周辺対数尤度')
ax.set_title('周辺対数尤度の推移')
ax.grid(True, alpha=0.3)

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig03_em_convergence.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 8
# ============================================================

def eap_theta(X, deltas, R=21):
    """
    EAPによる個人θの推定
    X: 応答行列（N×J）
    deltas: 推定済みの項目パラメータ
    R: 求積点の数
    """
    N, J = X.shape
    X_r, A_r = get_quadrature(R)

    eap_estimates = np.zeros(N)
    psd_estimates = np.zeros(N)

    for i in range(N):
        # 各求積点での尤度
        L_r = np.ones(R)
        for j in range(J):
            p = rasch_prob(X_r, deltas[j])
            p = np.clip(p, 1e-15, 1 - 1e-15)
            L_r *= p**X[i, j] * (1 - p)**(1 - X[i, j])

        # 事後分布（正規化前）
        posterior = L_r * A_r
        posterior_sum = np.sum(posterior)

        if posterior_sum > 0:
            # EAP = 事後分布の平均
            eap_estimates[i] = np.sum(X_r * posterior) / posterior_sum
            # PSD = 事後分布の標準偏差
            psd_estimates[i] = np.sqrt(
                np.sum((X_r - eap_estimates[i])**2 * posterior) / posterior_sum
            )
        else:
            eap_estimates[i] = 0.0
            psd_estimates[i] = np.inf

    return eap_estimates, psd_estimates

# ============================================================
# Code Block 9
# ============================================================

# EAP推定
eap_thetas, psd_thetas = eap_theta(X_valid, est_deltas)

print("=== EAP推定結果 ===")
print(f"θ̂の平均: {eap_thetas.mean():.4f}")
print(f"θ̂の標準偏差: {eap_thetas.std():.4f}")
print(f"PSDの平均: {psd_thetas.mean():.4f}")

# ============================================================
# Code Block 10
# ============================================================

theta_range = np.linspace(-4, 6, 500)

# Prior: N(0, 1)
prior = norm.pdf(theta_range, 0, 1)

# Likelihood: x = [1,1,0] に対する尤度（δ = [-1.0, 0.5, 1.5]）
deltas_demo = np.array([-1.0, 0.5, 1.5])
x_demo = np.array([1, 1, 0])
likelihood = np.ones_like(theta_range)
for j in range(len(deltas_demo)):
    p = rasch_prob(theta_range, deltas_demo[j])
    likelihood *= p**x_demo[j] * (1 - p)**(1 - x_demo[j])

# Posterior ∝ Prior × Likelihood
posterior = prior * likelihood
posterior = posterior / np.trapezoid(posterior, theta_range)

# 正規化して見やすくする
prior_norm = prior / prior.max()
likelihood_norm = likelihood / likelihood.max()
posterior_norm = posterior / posterior.max()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(theta_range, prior_norm, 'b--', linewidth=2, label='事前分布 g(θ) = N(0, 1)')
ax.plot(theta_range, likelihood_norm, 'g:', linewidth=2, label='尤度関数 L(x|θ)')
ax.plot(theta_range, posterior_norm, 'r-', linewidth=2.5,
        label='事後分布 ∝ Prior × Likelihood')

eap = np.trapezoid(theta_range * posterior, theta_range)
mle = theta_range[np.argmax(likelihood)]
ax.axvline(x=eap, color='darkred', linestyle='--', alpha=0.6, label=f'EAP = {eap:.2f}')
ax.axvline(x=mle, color='darkgreen', linestyle=':', alpha=0.6, label=f'MLE = {mle:.2f}')

ax.set_xlabel('θ')
ax.set_ylabel('正規化された値')
ax.set_title('事前分布 × 尤度関数 → 事後分布')
ax.legend(loc='upper right')
ax.set_xlim(-4, 6)
ax.grid(True, alpha=0.3)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig02_bayesian.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 11
# ============================================================

patterns = [
    np.array([1, 0, 0, 0, 0, 0, 0, 0]),  # 1問正答
    np.array([1, 1, 1, 1, 0, 0, 0, 0]),  # 4問正答
    np.array([1, 1, 1, 1, 1, 1, 1, 0]),  # 7問正答
]
labels = ['1問正答（X=1）', '4問正答（X=4）', '7問正答（X=7）']
colors = ['steelblue', 'coral', 'forestgreen']

theta_range = np.linspace(-5, 5, 500)

fig, ax = plt.subplots(figsize=(10, 5))

for x, label, color in zip(patterns, labels, colors):
    prior = norm.pdf(theta_range, 0, 1)
    likelihood = np.ones_like(theta_range)
    for j in range(J_sim):
        p = rasch_prob(theta_range, est_deltas[j])
        p = np.clip(p, 1e-15, 1 - 1e-15)
        likelihood *= p**x[j] * (1 - p)**(1 - x[j])

    posterior = prior * likelihood
    area = np.trapezoid(posterior, theta_range)
    if area > 0:
        posterior = posterior / area

    eap = np.trapezoid(theta_range * posterior, theta_range)
    psd = np.sqrt(np.trapezoid((theta_range - eap)**2 * posterior, theta_range))

    ax.plot(theta_range, posterior, linewidth=2, color=color,
            label=f'{label}: EAP={eap:.2f}, PSD={psd:.2f}')
    ax.axvline(x=eap, color=color, linestyle='--', alpha=0.4)

ax.set_xlabel('θ')
ax.set_ylabel('事後密度')
ax.set_title('応答パターンごとの事後分布')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 5)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig05_eap_posteriors.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 12
# ============================================================

# 各受験者のMLE推定
def mle_theta_person(x, deltas, max_iter=50):
    theta = 0.0
    for _ in range(max_iter):
        p = rasch_prob(theta, deltas)
        grad = np.sum(x - p)
        info = np.sum(p * (1 - p))
        if info < 1e-8:
            break
        update = grad / info
        theta += update
        if abs(update) < 1e-6:
            break
    return theta

mle_thetas = np.array([mle_theta_person(X_valid[i], est_deltas)
                        for i in range(X_valid.shape[0])])

true_thetas_valid = true_thetas[valid_mask]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# EAP vs 真のθ
ax = axes[0]
ax.scatter(true_thetas_valid, eap_thetas, s=15, alpha=0.4, color='steelblue')
lim = 4
ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.4)
r_eap = np.corrcoef(true_thetas_valid, eap_thetas)[0, 1]
ax.set_xlabel('真のθ')
ax.set_ylabel('EAP推定値')
ax.set_title(f'EAP vs 真のθ（r = {r_eap:.4f}）')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# MLE vs 真のθ
ax = axes[1]
ax.scatter(true_thetas_valid, mle_thetas, s=15, alpha=0.4, color='coral')
ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.4)
r_mle = np.corrcoef(true_thetas_valid, mle_thetas)[0, 1]
ax.set_xlabel('真のθ')
ax.set_ylabel('MLE推定値')
ax.set_title(f'MLE vs 真のθ（r = {r_mle:.4f}）')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig04_eap_vs_mle.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 13
# ============================================================

# JMLE推定（JMLEの実装を再利用）
def jmle_rasch(X, max_iter=100, tol=1e-4):
    N, J = X.shape
    raw_scores = X.sum(axis=1)
    item_scores = X.sum(axis=0)
    thetas = np.log((raw_scores + 0.5) / (J - raw_scores + 0.5))
    deltas = -np.log((item_scores + 0.5) / (N - item_scores + 0.5))

    for iteration in range(max_iter):
        P = rasch_prob(thetas[:, np.newaxis], deltas[np.newaxis, :])
        residuals_theta = (X - P).sum(axis=1)
        info_theta = (P * (1 - P)).sum(axis=1)
        theta_update = residuals_theta / np.clip(info_theta, 1e-8, None)
        thetas = thetas + theta_update

        # Person centering
        thetas = thetas - thetas.mean()

        P = rasch_prob(thetas[:, np.newaxis], deltas[np.newaxis, :])
        residuals_delta = (P - X).sum(axis=0)
        info_delta = (P * (1 - P)).sum(axis=0)
        delta_update = residuals_delta / np.clip(info_delta, 1e-8, None)
        deltas = deltas + delta_update

        max_change = max(np.max(np.abs(theta_update)),
                         np.max(np.abs(delta_update)))
        if max_change < tol:
            break

    return thetas, deltas

# JMLE実行
jmle_thetas, jmle_deltas = jmle_rasch(X_valid)

# 比較
r = np.corrcoef(jmle_deltas, est_deltas)[0, 1]

print("=== JMLE vs MMLE ===")
print(f"{'項目':>4} {'真のδ':>8} {'JMLE δ̂':>8} {'MMLE δ̂':>8}")
print("-" * 36)
for j in range(J_sim):
    print(f"  {j+1:>2}  {true_deltas[j]:>8.3f}"
          f" {jmle_deltas[j]:>8.3f} {est_deltas[j]:>8.3f}")
print(f"\n相関係数: r = {r:.4f}")

# ============================================================
# Code Block 14
# ============================================================

# 散布図
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(jmle_deltas, est_deltas, s=80, color='steelblue',
           edgecolors='navy', linewidths=0.5, zorder=3)

lim = max(abs(jmle_deltas).max(), abs(est_deltas).max()) + 0.3
ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.4, label='完全一致線')

for j in range(J_sim):
    ax.annotate(f'δ_{j+1}', (jmle_deltas[j], est_deltas[j]),
                textcoords='offset points', xytext=(8, 5), fontsize=10)

ax.set_xlabel('JMLE δ̂')
ax.set_ylabel('MMLE δ̂')
ax.set_title(f'JMLE vs MMLE の項目パラメータ比較（r = {r:.4f}）')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig06_jmle_vs_mmle.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 15
# ============================================================

theta_range = np.linspace(-4, 4, 500)

tcf = np.zeros_like(theta_range)
for j in range(J_sim):
    tcf += rasch_prob(theta_range, est_deltas[j])

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(theta_range, tcf, 'b-', linewidth=2.5)

# 例：θ = 1.0 の受験者
theta_ex = 1.0
t_ex = sum(rasch_prob(theta_ex, d) for d in est_deltas)
ax.plot([theta_ex, theta_ex], [0, t_ex], 'r--', linewidth=1.5)
ax.plot([-4, theta_ex], [t_ex, t_ex], 'r--', linewidth=1.5)
ax.plot(theta_ex, t_ex, 'ro', markersize=8)
ax.annotate(f'θ = {theta_ex:.1f} → T = {t_ex:.2f}',
            xy=(theta_ex, t_ex), xytext=(theta_ex + 0.5, t_ex - 1),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=12, color='red')

ax.set_xlabel('θ（能力）')
ax.set_ylabel('T（期待得点）')
ax.set_title('テスト特性関数（TCF）')
ax.set_xlim(-4, 4)
ax.set_ylim(0, J_sim + 0.3)
ax.axhline(y=J_sim, color='gray', linestyle=':', alpha=0.4)
ax.grid(True, alpha=0.3)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig07_tcf.png', dpi=150, bbox_inches='tight')
plt.show()

