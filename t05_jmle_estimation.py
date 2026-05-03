"""
T05: IRTの同時最尤推定法（JMLE）：項目と受験者を同時に推定するアルゴリズムをPythonで実装

記事のPythonコードを統合したファイル。
記事URL: https://bigdata-analytics.jp/analytics/irt-jmle-estimation/

Author: Hiroyuki Matsumoto (Digital Boy LLC)
Repository: https://github.com/hiro1234567/irt-python
"""

# ============================================================
# Code Block 1
# ============================================================

import numpy as np
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

# 5人×3項目のシミュレーション応答行列
# 行: 受験者（θ = -1.5, -0.5, 0.0, 0.8, 1.5）
# 列: 項目（δ = -1.0, 0.5, 1.5）
response_matrix = np.array([
    [1, 0, 0],  # 受験者1: 低能力（θ=-1.5）。易しい問題だけ正答
    [1, 0, 0],  # 受験者2: やや低（θ=-0.5）
    [1, 1, 0],  # 受験者3: 平均的（θ=0.0）。中程度まで正答
    [1, 1, 0],  # 受験者4: やや高（θ=0.8）
    [1, 1, 1],  # 受験者5: 高能力（θ=1.5）。全問正答
])

print("応答行列 X:")
print(response_matrix)
print(f"行（受験者）: {response_matrix.shape[0]}人")
print(f"列（項目）: {response_matrix.shape[1]}項目")
print(f"各受験者の合計点: {response_matrix.sum(axis=1)}")
print(f"各項目の正答者数: {response_matrix.sum(axis=0)}")

# ============================================================
# Code Block 3
# ============================================================

def rasch_prob(theta, delta):
    """ラッシュモデルの正答確率を計算する"""
    diff = theta - delta
    return 1.0 / (1.0 + np.exp(-diff))

def joint_log_likelihood(thetas, deltas, X):
    """
    結合対数尤度を計算する
    thetas: 受験者の能力ベクトル（N,）
    deltas: 項目位置ベクトル（J,）
    X: 応答行列（N×J）
    """
    N, J = X.shape
    ll = 0.0
    for i in range(N):
        for j in range(J):
            p = rasch_prob(thetas[i], deltas[j])
            # 数値的に安定させるためにclip
            p = np.clip(p, 1e-10, 1 - 1e-10)
            ll += X[i, j] * np.log(p) + (1 - X[i, j]) * np.log(1 - p)
    return ll

# ============================================================
# Code Block 4
# ============================================================

def joint_log_likelihood_vectorized(thetas, deltas, X):
    """結合対数尤度のベクトル化版"""
    # thetas: (N,), deltas: (J,) → P: (N, J)
    P = rasch_prob(thetas[:, np.newaxis], deltas[np.newaxis, :])
    P = np.clip(P, 1e-10, 1 - 1e-10)
    return np.sum(X * np.log(P) + (1 - X) * np.log(1 - P))

# テスト: 真のパラメータで対数尤度を計算
true_thetas = np.array([-1.5, -0.5, 0.0, 0.8, 1.5])
true_deltas = np.array([-1.0, 0.5, 1.5])

ll = joint_log_likelihood_vectorized(true_thetas, true_deltas, response_matrix)
print(f"真のパラメータでの対数尤度: {ll:.4f}")

# ============================================================
# Code Block 5
# ============================================================

# 1人×1項目の対数尤度等高線
theta_range = np.linspace(-3, 3, 200)
delta_range = np.linspace(-3, 3, 200)
THETA, DELTA = np.meshgrid(theta_range, delta_range)

# x=1（正答）の場合の対数尤度
P = rasch_prob(THETA, DELTA)
P = np.clip(P, 1e-10, 1 - 1e-10)
LL = np.log(P)  # x=1なのでln(p)だけ

fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(THETA, DELTA, LL, levels=30, cmap='RdYlBu_r')
plt.colorbar(contour, ax=ax, label='対数尤度')
ax.set_xlabel('受験者の能力 θ')
ax.set_ylabel('項目位置 δ')
ax.set_title('正答(x=1)時の対数尤度')

# θ - δ = 0 の線（正答確率0.5）
ax.plot(theta_range, theta_range, 'k--', alpha=0.5, label='θ = δ（確率0.5）')
ax.legend()
plt.tight_layout()
plt.savefig('fig01_loglik_contour.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 6
# ============================================================

def jmle_rasch(X, max_iter=100, tol=1e-4):
    """
    ラッシュモデルのJMLE推定
    X: 応答行列（N×J）。全正答・全不正答の行は除外済みであること
    max_iter: 最大反復回数
    tol: 収束判定の閾値（パラメータ変化量の最大値）
    """
    N, J = X.shape

    # 初期値: 素点から大まかな推定値を設定
    raw_scores = X.sum(axis=1)  # 各受験者の合計点
    item_scores = X.sum(axis=0)  # 各項目の正答者数

    # 素点からの初期値（対数オッズ変換、PROXに相当）
    thetas = np.log((raw_scores + 0.5) / (J - raw_scores + 0.5))
    deltas = -np.log((item_scores + 0.5) / (N - item_scores + 0.5))

    history = {'thetas': [thetas.copy()], 'deltas': [deltas.copy()]}

    for iteration in range(max_iter):
        # Step 1: δ固定でθを更新
        P = rasch_prob(thetas[:, np.newaxis], deltas[np.newaxis, :])
        residuals_theta = (X - P).sum(axis=1)      # Σ(x_ij - p_ij)
        info_theta = (P * (1 - P)).sum(axis=1)      # Σ p_ij(1-p_ij)
        theta_update = residuals_theta / np.clip(info_theta, 1e-8, None)
        thetas = thetas + theta_update

        # 尺度の固定: person centering（θの平均を0に保つ）
        thetas = thetas - thetas.mean()

        # Step 2: θ固定でδを更新
        P = rasch_prob(thetas[:, np.newaxis], deltas[np.newaxis, :])
        residuals_delta = (P - X).sum(axis=0)      # Σ(p_ij - x_ij)（符号反転）
        info_delta = (P * (1 - P)).sum(axis=0)      # Σ p_ij(1-p_ij)
        delta_update = residuals_delta / np.clip(info_delta, 1e-8, None)
        deltas = deltas + delta_update

        history['thetas'].append(thetas.copy())
        history['deltas'].append(deltas.copy())

        # 収束判定
        max_change = max(np.max(np.abs(theta_update)),
                         np.max(np.abs(delta_update)))
        if max_change < tol:
            print(f"収束しました（{iteration + 1}回の反復）")
            break
    else:
        print(f"最大反復回数（{max_iter}）に達しました")

    return thetas, deltas, history

# ============================================================
# Code Block 7
# ============================================================

# シミュレーションデータの生成
np.random.seed(42)

# テスト設定
true_thetas_sim = np.array([-2.0, -1.2, -0.8, -0.3, 0.0, 0.3, 0.7, 1.0, 1.5, 2.0])
true_deltas_sim = np.array([-1.5, -0.5, 0.0, 0.8, 1.5])

N_sim = len(true_thetas_sim)
J_sim = len(true_deltas_sim)

print("=== テスト設定 ===")
print(f"受験者数: {N_sim}人")
print(f"項目数: {J_sim}問")
print(f"真のθ: {true_thetas_sim}")
print(f"真のδ: {true_deltas_sim}")

# ============================================================
# Code Block 8
# ============================================================

# 応答データの生成
P_true = rasch_prob(true_thetas_sim[:, np.newaxis],
                     true_deltas_sim[np.newaxis, :])
X_sim = (np.random.rand(N_sim, J_sim) < P_true).astype(int)

# 全正答・全不正答の受験者がいないか確認
row_sums = X_sim.sum(axis=1)
print(f"\n各受験者の合計点: {row_sums}")
print(f"全正答の受験者: {(row_sums == J_sim).sum()}人")
print(f"全不正答の受験者: {(row_sums == 0).sum()}人")

# 全正答・全不正答がいたら除外（セクション5で詳述）
valid_mask = (row_sums > 0) & (row_sums < J_sim)
X_valid = X_sim[valid_mask]
true_thetas_valid = true_thetas_sim[valid_mask]

print(f"\n有効な受験者数: {X_valid.shape[0]}人")
print(f"\n応答行列:")
print(X_valid)

# ============================================================
# Code Block 9
# ============================================================

# JMLE実行
est_thetas, est_deltas, hist = jmle_rasch(X_valid)

print(f"\n=== 推定結果 ===")
print(f"推定θ: {np.round(est_thetas, 3)}")
print(f"推定δ: {np.round(est_deltas, 3)}")
print(f"真のδ: {true_deltas_sim}")

# ============================================================
# Code Block 10
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# δの収束過程
ax = axes[0]
deltas_history = np.array(hist['deltas'])
for j in range(J_sim):
    ax.plot(deltas_history[:, j], marker='o', markersize=3,
            label=f'δ_{j+1}（真値={true_deltas_sim[j]:.1f}）')
    ax.axhline(y=true_deltas_sim[j], color=f'C{j}', linestyle='--', alpha=0.3)
ax.set_xlabel('反復回数')
ax.set_ylabel('δの推定値')
ax.set_title('δの収束過程')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# θの収束過程
ax = axes[1]
thetas_history = np.array(hist['thetas'])
n_valid = thetas_history.shape[1]
for i in range(min(n_valid, 5)):  # 最大5人まで表示
    ax.plot(thetas_history[:, i], marker='o', markersize=3,
            label=f'θ_{i+1}')
ax.set_xlabel('反復回数')
ax.set_ylabel('θの推定値')
ax.set_title('θの収束過程')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig02_convergence.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 11
# ============================================================

# 不定性の実証: θとδに同じ定数を加えても対数尤度は変化しない
c_values = np.linspace(-3, 3, 50)
ll_values = []

for c in c_values:
    shifted_thetas = est_thetas + c
    shifted_deltas = est_deltas + c
    ll = joint_log_likelihood_vectorized(shifted_thetas, shifted_deltas, X_valid)
    ll_values.append(ll)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(c_values, ll_values, 'b-', linewidth=2)
ax.set_xlabel('シフト量 c')
ax.set_ylabel('対数尤度')
ax.set_title('θとδに同じ定数を加えても対数尤度は不変')
ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='c = 0（推定値）')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig03_indeterminacy.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"対数尤度の範囲: {min(ll_values):.6f} 〜 {max(ll_values):.6f}")
print(f"変化幅: {max(ll_values) - min(ll_values):.10f}")

# ============================================================
# Code Block 12
# ============================================================

# Person centering と Item centering の比較
# Item centering版のJMLE
def jmle_rasch_item_centering(X, max_iter=100, tol=1e-4):
    """Item centering版JMLE"""
    N, J = X.shape
    raw_scores = X.sum(axis=1)
    item_scores = X.sum(axis=0)
    thetas = np.log((raw_scores + 0.5) / (J - raw_scores + 0.5))
    deltas = -np.log((item_scores + 0.5) / (N - item_scores + 0.5))

    for iteration in range(max_iter):
        P = rasch_prob(thetas[:, np.newaxis], deltas[np.newaxis, :])
        residuals_theta = (X - P).sum(axis=1)
        info_theta = (P * (1 - P)).sum(axis=1)
        thetas = thetas + residuals_theta / np.clip(info_theta, 1e-8, None)

        P = rasch_prob(thetas[:, np.newaxis], deltas[np.newaxis, :])
        residuals_delta = (P - X).sum(axis=0)
        info_delta = (P * (1 - P)).sum(axis=0)
        deltas = deltas + residuals_delta / np.clip(info_delta, 1e-8, None)

        # Item centering: δの平均を0に固定
        deltas = deltas - deltas.mean()

        max_change = max(np.max(np.abs(residuals_theta / np.clip(info_theta, 1e-8, None))),
                         np.max(np.abs(residuals_delta / np.clip(info_delta, 1e-8, None))))
        if max_change < tol:
            break

    return thetas, deltas

est_thetas_ic, est_deltas_ic = jmle_rasch_item_centering(X_valid)

print("=== Person Centering ===")
print(f"θ: {np.round(est_thetas, 3)}, 平均: {est_thetas.mean():.6f}")
print(f"δ: {np.round(est_deltas, 3)}, 平均: {est_deltas.mean():.3f}")

print("\n=== Item Centering ===")
print(f"θ: {np.round(est_thetas_ic, 3)}, 平均: {est_thetas_ic.mean():.3f}")
print(f"δ: {np.round(est_deltas_ic, 3)}, 平均: {est_deltas_ic.mean():.6f}")

# 差の構造が同じことを確認
print(f"\nδの差（Person Centering）: {np.round(np.diff(est_deltas), 3)}")
print(f"δの差（Item Centering）:   {np.round(np.diff(est_deltas_ic), 3)}")

# ============================================================
# Code Block 13
# ============================================================

# extreme scoreでのMLE発散のデモ
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 5項目・δ=[-1.5, -0.5, 0.0, 0.8, 1.5] のテスト
deltas_demo = np.array([-1.5, -0.5, 0.0, 0.8, 1.5])
theta_range = np.linspace(-5, 8, 500)

# (a) 全問正答 x = [1,1,1,1,1]
ax = axes[0]
ll_perfect = np.zeros_like(theta_range)
for theta_val, idx in zip(theta_range, range(len(theta_range))):
    probs = rasch_prob(theta_val, deltas_demo)
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    ll_perfect[idx] = np.sum(np.log(probs))  # 全正答
ax.plot(theta_range, ll_perfect, 'r-', linewidth=2)
ax.set_xlabel('θ')
ax.set_ylabel('対数尤度')
ax.set_title('全問正答（X=5）の対数尤度')
ax.annotate('θ → +∞ で\n対数尤度が増加し続ける',
            xy=(6, ll_perfect[-50]), fontsize=11,
            arrowprops=dict(arrowstyle='->', color='red'),
            xytext=(3, ll_perfect[-50] - 1.5))
ax.grid(True, alpha=0.3)

# (b) 通常パターン x = [1,1,0,0,0]（合計点2）
ax = axes[1]
x_normal = np.array([1, 1, 0, 0, 0])
ll_normal = np.zeros_like(theta_range)
for theta_val, idx in zip(theta_range, range(len(theta_range))):
    probs = rasch_prob(theta_val, deltas_demo)
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    ll_normal[idx] = np.sum(x_normal * np.log(probs) +
                            (1 - x_normal) * np.log(1 - probs))

theta_mle = theta_range[np.argmax(ll_normal)]
ax.plot(theta_range, ll_normal, 'b-', linewidth=2)
ax.axvline(x=theta_mle, color='r', linestyle='--', alpha=0.7,
           label=f'MLE: θ = {theta_mle:.2f}')
ax.set_xlabel('θ')
ax.set_ylabel('対数尤度')
ax.set_title('通常パターン（X=2）の対数尤度')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig04_extreme_score.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 14
# ============================================================

def compute_fit_statistics(X, thetas, deltas):
    """INFIT・OUTFITを計算する"""
    N, J = X.shape
    P = rasch_prob(thetas[:, np.newaxis], deltas[np.newaxis, :])
    P = np.clip(P, 1e-10, 1 - 1e-10)

    # 残差と分散
    residuals = X - P                   # x_ij - p_ij
    variance = P * (1 - P)              # p_ij(1 - p_ij)
    z_squared = residuals**2 / variance  # 標準化残差の2乗

    # 項目ごとのINFIT・OUTFIT
    item_infit = (residuals**2).sum(axis=0) / variance.sum(axis=0)
    item_outfit = z_squared.mean(axis=0)

    # 受験者ごとのINFIT・OUTFIT
    person_infit = (residuals**2).sum(axis=1) / variance.sum(axis=1)
    person_outfit = z_squared.mean(axis=1)

    return {
        'item_infit': item_infit,
        'item_outfit': item_outfit,
        'person_infit': person_infit,
        'person_outfit': person_outfit,
    }

# ============================================================
# Code Block 15
# ============================================================

np.random.seed(123)

# テスト設定
N_fit = 200
J_fit = 10
thetas_fit = np.random.normal(0, 1, N_fit)
deltas_fit = np.linspace(-2, 2, J_fit)

# 全項目α=1（ラッシュモデル）で応答生成…ただし項目5だけα=0.3（識別力が低い）
alphas_fit = np.ones(J_fit)
alphas_fit[4] = 0.3  # 項目5の識別力を低く設定

P_fit = 1.0 / (1.0 + np.exp(-alphas_fit[np.newaxis, :] *
               (thetas_fit[:, np.newaxis] - deltas_fit[np.newaxis, :])))
X_fit = (np.random.rand(N_fit, J_fit) < P_fit).astype(int)

# 全正答・全不正答を除外
row_sums = X_fit.sum(axis=1)
valid = (row_sums > 0) & (row_sums < J_fit)
X_fit_valid = X_fit[valid]
thetas_fit_valid = thetas_fit[valid]

# JMLE推定
est_thetas_fit, est_deltas_fit, _ = jmle_rasch(X_fit_valid, max_iter=200)

# 適合度計算
fit_stats = compute_fit_statistics(X_fit_valid, est_thetas_fit, est_deltas_fit)

print("=== 項目の適合度 ===")
print(f"{'項目':>4} {'INFIT':>8} {'OUTFIT':>8} {'備考':>12}")
print("-" * 40)
for j in range(J_fit):
    note = "← α=0.3" if j == 4 else ""
    print(f"  {j+1:>2}  {fit_stats['item_infit'][j]:>8.3f}"
          f" {fit_stats['item_outfit'][j]:>8.3f}  {note}")

# ============================================================
# Code Block 16
# ============================================================

# 適合度のプロット
fig, ax = plt.subplots(figsize=(10, 6))

items = np.arange(1, J_fit + 1)
width = 0.35
ax.bar(items - width/2, fit_stats['item_infit'], width, label='INFIT', color='steelblue')
ax.bar(items + width/2, fit_stats['item_outfit'], width, label='OUTFIT', color='coral')

# 期待値=1の基準線
ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Smith et al. (1998) の目安: 0.7〜1.3
ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.3)
ax.axhline(y=1.3, color='gray', linestyle='--', alpha=0.3)
ax.fill_between(items, 0.7, 1.3, alpha=0.05, color='green')

# 項目5を強調
ax.annotate('α = 0.3\n（識別力が低い）', xy=(5, fit_stats['item_outfit'][4]),
            xytext=(7, fit_stats['item_outfit'][4] + 0.3),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=11, color='red')

ax.set_xlabel('項目番号')
ax.set_ylabel('MNSQ')
ax.set_title('項目の適合度（INFIT・OUTFIT）')
ax.set_xticks(items)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig05_fit_statistics.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 17
# ============================================================

np.random.seed(456)

# 大きめのデータ: 1000人 × 8項目
N_inv = 1000
J_inv = 8
thetas_inv = np.random.normal(0, 1, N_inv)
deltas_inv = np.array([-2.0, -1.2, -0.5, 0.0, 0.3, 0.8, 1.5, 2.2])

P_inv = rasch_prob(thetas_inv[:, np.newaxis], deltas_inv[np.newaxis, :])
X_inv = (np.random.rand(N_inv, J_inv) < P_inv).astype(int)

# 全正答/全不正答を除外
row_sums = X_inv.sum(axis=1)
valid_inv = (row_sums > 0) & (row_sums < J_inv)
X_inv = X_inv[valid_inv]
N_valid = X_inv.shape[0]

# ランダムに2分割
indices = np.random.permutation(N_valid)
half = N_valid // 2
X_sub1 = X_inv[indices[:half]]
X_sub2 = X_inv[indices[half:]]

print(f"サブサンプル1: {X_sub1.shape[0]}人")
print(f"サブサンプル2: {X_sub2.shape[0]}人")

# 各サブサンプルで独立にJMLE
_, deltas_sub1, _ = jmle_rasch(X_sub1, max_iter=200)
_, deltas_sub2, _ = jmle_rasch(X_sub2, max_iter=200)

# 相関係数
r = np.corrcoef(deltas_sub1, deltas_sub2)[0, 1]
print(f"\n2つのサブサンプルのδ推定値の相関: r = {r:.4f}")

# ============================================================
# Code Block 18
# ============================================================

# 散布図
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(deltas_sub1, deltas_sub2, s=80, color='steelblue',
           edgecolors='navy', linewidths=0.5, zorder=3)

# 45度線
lim = max(abs(deltas_sub1).max(), abs(deltas_sub2).max()) + 0.3
ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.4, label='完全一致線')

# 各点にラベル
for j in range(J_inv):
    ax.annotate(f'δ_{j+1}', (deltas_sub1[j], deltas_sub2[j]),
                textcoords='offset points', xytext=(8, 5), fontsize=10)

ax.set_xlabel('サブサンプル1のδ推定値')
ax.set_ylabel('サブサンプル2のδ推定値')
ax.set_title(f'δの不変性検証（r = {r:.4f}）')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig06_invariance.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 19
# ============================================================

# Variable Map: 1000人データで作成
est_thetas_all, est_deltas_all, _ = jmle_rasch(X_inv, max_iter=200)

fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True,
                          gridspec_kw={'width_ratios': [3, 1]})

# 左: 受験者の分布
ax = axes[0]
ax.hist(est_thetas_all, bins=30, orientation='horizontal',
        color='lightsteelblue', edgecolor='steelblue', alpha=0.8)
ax.set_xlabel('受験者数')
ax.set_title('受験者の分布')
ax.invert_xaxis()

# 右: 項目の位置
ax = axes[1]
for j, d in enumerate(est_deltas_all):
    ax.plot(0.5, d, 'D', color='coral', markersize=12, markeredgecolor='darkred')
    ax.annotate(f'項目{j+1}', (0.5, d), textcoords='offset points',
                xytext=(15, 0), fontsize=11, va='center')
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_title('項目位置')

# 共通のy軸ラベル
axes[0].set_ylabel('ロジット尺度（θ / δ）')

plt.suptitle('Variable Map（受験者-項目マップ）', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('fig07_variable_map.png', dpi=150, bbox_inches='tight')
plt.show()

