"""
T14: 多次元IRT（MIRT）のPython実装と推定：情報関数・回転不定性・尺度変換を完全解説

記事のPythonコードを統合したファイル。
記事URL: https://bigdata-analytics.jp/analytics/irt-mestimation/

Author: Hiroyuki Matsumoto (Digital Boy LLC)
Repository: https://github.com/hiro1234567/irt-python
"""

# ============================================================
# Code Block 1
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

# 図の見栄えを統一するための共通設定
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

def m2pl_prob(theta, alpha, gamma):
    """M2PLモデルの正答確率"""
    logit = np.dot(alpha, theta) + gamma
    return 1 / (1 + np.exp(-logit))

def mirt_item_info(theta, alpha, gamma, omega_deg):
    """多次元項目情報関数（方向依存）"""
    p = m2pl_prob(theta, alpha, gamma)
    omega_rad = np.radians(omega_deg)
    # 2次元の場合：方向余弦
    cos_omega = np.array([np.cos(omega_rad), np.sin(omega_rad)])
    projection = np.dot(alpha, cos_omega)
    return p * (1 - p) * projection**2

# パラメータ設定（de Ayala Ch.10の例に対応）
alpha = np.array([2.0, 0.5])
gamma = 2.0
theta_origin = np.array([0.0, 0.0])

# 0°〜360°の全方向で情報量を計算
angles = np.linspace(0, 360, 361)
info_values = [mirt_item_info(theta_origin, alpha, gamma, ang) for ang in angles]

# αベクトルの方向を計算
A = np.linalg.norm(alpha)
alpha_angle = np.degrees(np.arccos(alpha[0] / A))

print(f'多次元識別力 A = {A:.3f}')
print(f'αベクトルの方向 = {alpha_angle:.2f}°')
print(f'最大情報量（α方向） = {max(info_values):.4f}')
print(f'直交方向の情報量 = {mirt_item_info(theta_origin, alpha, gamma, alpha_angle + 90):.6f}')

# ============================================================
# Code Block 3
# ============================================================

# 能力空間のグリッドを作成
theta1 = np.linspace(-4, 4, 100)
theta2 = np.linspace(-4, 4, 100)
T1, T2 = np.meshgrid(theta1, theta2)

# パラメータ設定
alpha = np.array([2.0, 0.5])
gamma = 2.0
A = np.linalg.norm(alpha)

# αベクトル方向の情報曲面を計算
P = 1 / (1 + np.exp(-(alpha[0] * T1 + alpha[1] * T2 + gamma)))
Info_max = P * (1 - P) * A**2

# 3Dプロット
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T1, T2, Info_max, cmap='hot', alpha=0.85, edgecolor='none')
ax.set_xlabel('theta1')
ax.set_ylabel('theta2')
ax.set_zlabel('情報量 I')
ax.set_title('M2PL 多次元項目情報曲面\n（αベクトル方向、α=(2.0, 0.5), γ=2.0）')
ax.view_init(elev=25, azim=-60)
plt.tight_layout()
plt.savefig('irt_mirt_est_info_surface.png', dpi=220, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 4
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

levels = np.linspace(0, 1.1, 12)
cs = ax.contourf(T1, T2, Info_max, levels=levels, cmap='hot')
plt.colorbar(cs, ax=ax, label='情報量 I')

# 変曲線（P=0.5）を重ねて描画
P_line = 1 / (1 + np.exp(-(alpha[0] * T1 + alpha[1] * T2 + gamma)))
ax.contour(T1, T2, P_line, levels=[0.5], colors='cyan', linewidths=2, linestyles='--')

ax.set_xlabel('theta1')
ax.set_ylabel('theta2')
ax.set_title('項目情報の等高線\n（シアン破線 = P=0.5の変曲線）')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('irt_mirt_est_info_contour.png', dpi=220, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 5
# ============================================================

# 元のパラメータ
alpha_orig = np.array([2.0, 0.5])
gamma_orig = 2.0
theta_orig = np.array([1.5, -1.0])

# 元の正答確率
logit_orig = np.dot(alpha_orig, theta_orig) + gamma_orig
p_orig = 1 / (1 + np.exp(-logit_orig))
print(f'回転前: logit = {logit_orig:.4f}, P = {p_orig:.4f}')

# 25°の回転行列
angle = np.radians(25)
R = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle),  np.cos(angle)]
])

# 回転後のパラメータ
alpha_rot = R @ alpha_orig      # α* = Rα
theta_rot = R @ theta_orig      # θ* = Rθ
gamma_rot = gamma_orig          # γは回転の影響を受けない

# 回転後の正答確率
logit_rot = np.dot(alpha_rot, theta_rot) + gamma_rot
p_rot = 1 / (1 + np.exp(-logit_rot))
print(f'回転後: logit = {logit_rot:.4f}, P = {p_rot:.4f}')
print(f'回転前α = {alpha_orig}, 回転後α* = {np.round(alpha_rot, 4)}')
print(f'回転前θ = {theta_orig}, 回転後θ* = {np.round(theta_rot, 4)}')

# ============================================================
# Code Block 6
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 回転前
ax = axes[0]
ax.annotate('', xy=alpha_orig, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
ax.text(alpha_orig[0]+0.1, alpha_orig[1]+0.1, 'alpha', fontsize=14, color='blue')
ax.plot(theta_orig[0], theta_orig[1], 'ro', markersize=10)
ax.text(theta_orig[0]+0.1, theta_orig[1]+0.1, 'theta', fontsize=14, color='red')
ax.set_xlim(-2, 3)
ax.set_ylim(-1.5, 2)
ax.set_aspect('equal')
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.grid(True, alpha=0.3)
ax.set_title(f'回転前\nP = {p_orig:.4f}')
ax.set_xlabel('theta1 / alpha1')
ax.set_ylabel('theta2 / alpha2')

# 回転後
ax = axes[1]
ax.annotate('', xy=alpha_rot, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
ax.text(alpha_rot[0]+0.1, alpha_rot[1]+0.1, 'alpha*', fontsize=14, color='blue')
ax.plot(theta_rot[0], theta_rot[1], 'ro', markersize=10)
ax.text(theta_rot[0]+0.1, theta_rot[1]+0.1, 'theta*', fontsize=14, color='red')
ax.set_xlim(-2, 3)
ax.set_ylim(-1.5, 2)
ax.set_aspect('equal')
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.grid(True, alpha=0.3)
ax.set_title(f'25°回転後\nP = {p_rot:.4f}')
ax.set_xlabel('theta1* / alpha1*')
ax.set_ylabel('theta2* / alpha2*')

plt.tight_layout()
plt.savefig('irt_mirt_est_rotation.png', dpi=220, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 7
# ============================================================

# 元のパラメータ
alpha = np.array([2.0, 0.5])
gamma = 2.0
theta = np.array([1.5, -1.0])

# 変換行列とシフトベクトル
Z = np.array([[2, 1], [1, 2]])
kappa = np.array([1.5, 1.5])

# 元の正答確率
p_orig = m2pl_prob(theta, alpha, gamma)

# 変換後のパラメータ
Z_inv = np.linalg.inv(Z)
alpha_star = Z_inv.T @ alpha      # α* = (Z^{-1})' α
gamma_star = gamma - alpha @ Z_inv @ kappa  # γ* = γ - α'Z^{-1}κ
theta_star = Z @ theta + kappa     # θ* = Zθ + κ

# 変換後の正答確率
p_star = m2pl_prob(theta_star, alpha_star, gamma_star)

print(f'元のパラメータ: α={alpha}, γ={gamma}, θ={theta}')
print(f'変換後: α*={np.round(alpha_star, 4)}, γ*={gamma_star:.4f}, θ*={np.round(theta_star, 4)}')
print(f'元の確率: P = {p_orig:.6f}')
print(f'変換後の確率: P* = {p_star:.6f}')
print(f'差: {abs(p_orig - p_star):.10f}')

# ============================================================
# Code Block 8
# ============================================================

np.random.seed(42)

# 真のパラメータ
n_persons = 500
n_items = 10
n_dims = 2

# 項目パラメータ（真値）
alpha_true = np.array([
    [1.8, 0.2], [1.5, 0.3], [1.2, 0.4], [2.0, 0.1], [1.0, 0.5],
    [0.3, 1.6], [0.2, 1.8], [0.4, 1.3], [0.1, 2.0], [0.5, 1.0],
])
gamma_true = np.array([0.5, -0.3, -0.8, 0.2, -1.0, 0.3, -0.5, -0.2, 0.8, -0.6])

# 能力パラメータ（相関あり2変量正規分布）
Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
theta_true = np.random.multivariate_normal([0, 0], Sigma, n_persons)

# 応答データの生成
logits = theta_true @ alpha_true.T + gamma_true  # (500, 10)
probs = 1 / (1 + np.exp(-logits))
responses = (np.random.rand(n_persons, n_items) < probs).astype(int)

print(f'応答データの形状: {responses.shape}')
print(f'正答率: {responses.mean(axis=0).round(3)}')

# ============================================================
# Code Block 9
# ============================================================

from scipy.stats import multivariate_normal

def m2pl_mmle(responses, n_dims=2, n_quad=15, max_iter=100, tol=1e-4):
    """M2PLのMMLEによる推定（2次元、EMアルゴリズム）"""
    n_persons, n_items = responses.shape

    # 求積点の設定（ガウス＝エルミート）
    from numpy.polynomial.hermite import hermgauss
    points_1d, weights_1d = hermgauss(n_quad)
    points_1d = points_1d * np.sqrt(2)  # 標準正規分布にスケーリング
    weights_1d = weights_1d / np.sqrt(np.pi)

    # 2次元の格子点を生成
    T1, T2 = np.meshgrid(points_1d, points_1d)
    quad_points = np.column_stack([T1.ravel(), T2.ravel()])  # (Q^2, 2)
    quad_weights = np.outer(weights_1d, weights_1d).ravel()  # (Q^2,)
    n_quad_total = len(quad_weights)

    # パラメータの初期値
    alpha_est = np.random.randn(n_items, n_dims) * 0.5 + 0.5
    gamma_est = np.zeros(n_items)

    for iteration in range(max_iter):
        alpha_old = alpha_est.copy()

        # E-step: 各求積点での尤度を計算
        # logit = quad_points @ alpha_est.T + gamma_est  → (Q^2, n_items)
        logits = quad_points @ alpha_est.T + gamma_est
        probs_at_quad = 1 / (1 + np.exp(-logits))  # (Q^2, n_items)

        # 各受験者の各求積点での尤度
        log_lik_at_quad = np.zeros((n_persons, n_quad_total))
        for q in range(n_quad_total):
            p_q = probs_at_quad[q]  # (n_items,)
            log_lik_at_quad[:, q] = np.sum(
                responses * np.log(p_q + 1e-10) +
                (1 - responses) * np.log(1 - p_q + 1e-10),
                axis=1
            )

        # 事後確率の計算
        lik_at_quad = np.exp(log_lik_at_quad) * quad_weights  # (n_persons, Q^2)
        marginal_lik = lik_at_quad.sum(axis=1, keepdims=True)  # (n_persons, 1)
        posterior = lik_at_quad / (marginal_lik + 1e-300)  # (n_persons, Q^2)

        # M-step: 項目パラメータの更新
        for j in range(n_items):
            # 期待十分統計量
            r_j = posterior.T @ responses[:, j]  # (Q^2,): 求積点ごとの期待正答数
            n_j = posterior.sum(axis=0)           # (Q^2,): 求積点ごとの期待受験者数

            # ニュートン・ラフソン法で各項目のα, γを更新
            params = np.concatenate([alpha_est[j], [gamma_est[j]]])

            def neg_expected_loglik(params):
                a = params[:n_dims]
                g = params[n_dims]
                logit_q = quad_points @ a + g
                p_q = 1 / (1 + np.exp(-logit_q))
                return -np.sum(r_j * np.log(p_q + 1e-10) +
                              (n_j - r_j) * np.log(1 - p_q + 1e-10))

            result = minimize(neg_expected_loglik, params, method='L-BFGS-B')
            alpha_est[j] = result.x[:n_dims]
            gamma_est[j] = result.x[n_dims]

        # 収束判定
        change = np.max(np.abs(alpha_est - alpha_old))
        if change < tol:
            print(f'収束: {iteration+1}回目, 最大変化量 = {change:.6f}')
            break

    return alpha_est, gamma_est

# 推定実行
alpha_hat, gamma_hat = m2pl_mmle(responses, n_dims=2, n_quad=11, max_iter=50)

# 真値との比較
print('\n=== 推定結果 vs 真値 ===')
print(f'{"項目":>4} {"α1真":>6} {"α1推定":>7} {"α2真":>6} {"α2推定":>7} {"γ真":>5} {"γ推定":>6}')
for j in range(n_items):
    print(f'{j+1:>4} {alpha_true[j,0]:>6.2f} {alpha_hat[j,0]:>7.3f} '
          f'{alpha_true[j,1]:>6.2f} {alpha_hat[j,1]:>7.3f} '
          f'{gamma_true[j]:>5.2f} {gamma_hat[j]:>6.3f}')

