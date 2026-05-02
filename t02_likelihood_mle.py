"""
T2: IRTの尤度関数と最尤推定 — θをどう求めるか
https://bigdata-analytics.jp/analytics/irt-likelihood-mle/

de Ayala (2022) Ch.2後半に対応。
尤度関数 → 対数尤度 → MLE → 全問正答/誤答問題 → SEE・Fisher情報量
→ 項目情報関数 → テスト情報関数 → テスト設計への応用
"""
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 150,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# --- 共通パラメータ（独自設定） ---
alpha = 1.2
deltas = np.array([-1.5, -0.5, 0.0, 0.8, 1.8])
responses = np.array([1, 0, 1, 1, 0])
theta_grid = np.linspace(-4, 4, 500)


def irf(theta, alpha, delta):
    """1PLモデルの項目反応関数"""
    return 1 / (1 + np.exp(-alpha * (theta - delta)))


def likelihood(theta, alpha, deltas, responses):
    """θを与えたときの尤度を計算"""
    L = 1.0
    for j in range(len(deltas)):
        p = irf(theta, alpha, deltas[j])
        if responses[j] == 1:
            L *= p
        else:
            L *= (1 - p)
    return L


def log_likelihood(theta, alpha, deltas, responses):
    """θを与えたときの対数尤度を計算"""
    ll = 0.0
    for j in range(len(deltas)):
        p = irf(theta, alpha, deltas[j])
        p = np.clip(p, 1e-15, 1 - 1e-15)
        if responses[j] == 1:
            ll += np.log(p)
        else:
            ll += np.log(1 - p)
    return ll


def item_information(theta, alpha, delta):
    """1PLモデルの項目情報関数"""
    p = irf(theta, alpha, delta)
    return alpha**2 * p * (1 - p)


def test_information(theta, alpha, deltas):
    """テスト情報関数：全項目の情報量の和"""
    total_info = np.zeros_like(theta, dtype=float)
    for delta_j in deltas:
        total_info += item_information(theta, alpha, delta_j)
    return total_info


# ============================================================
# 1. 尤度関数 L(θ)
# ============================================================
print("=== 1. 尤度関数 ===")
L_values = np.array([likelihood(t, alpha, deltas, responses) for t in theta_grid])
theta_hat = theta_grid[np.argmax(L_values)]
print(f"尤度最大の θ̂ ≈ {theta_hat:.2f}")

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(theta_grid, L_values, 'b-', linewidth=2)
ax.axvline(theta_hat, color='red', linestyle='--', linewidth=1.5,
           label=f'尤度最大の θ̂ = {theta_hat:.2f}')
ax.fill_between(theta_grid, L_values, alpha=0.15, color='blue')
ax.set_title('尤度関数 L(θ)：応答パターン (1, 0, 1, 1, 0)')
ax.set_xlabel('能力 θ')
ax.set_ylabel('尤度 L(θ)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 2. 対数尤度と最尤推定（MLE）
# ============================================================
print("\n=== 2. 対数尤度とMLE ===")
ll_values = np.array([log_likelihood(t, alpha, deltas, responses) for t in theta_grid])

result = minimize_scalar(
    lambda t: -log_likelihood(t, alpha, deltas, responses),
    bounds=(-4, 4), method='bounded'
)
theta_mle = result.x
print(f"最尤推定値 θ̂ = {theta_mle:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].plot(theta_grid, ll_values, 'b-', linewidth=2)
axes[0].axvline(theta_mle, color='red', linestyle='--', linewidth=1.5,
                label=f'MLE θ̂ = {theta_mle:.3f}')
axes[0].set_title('対数尤度関数 ln L(θ)')
axes[0].set_xlabel('能力 θ')
axes[0].set_ylabel('対数尤度 ln L(θ)')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(theta_grid, L_values, 'b-', linewidth=2)
axes[1].axvline(theta_mle, color='red', linestyle='--', linewidth=1.5,
                label=f'MLE θ̂ = {theta_mle:.3f}')
axes[1].fill_between(theta_grid, L_values, alpha=0.15, color='blue')
axes[1].set_title('尤度関数 L(θ)（比較用）')
axes[1].set_xlabel('能力 θ')
axes[1].set_ylabel('尤度 L(θ)')
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 3. 全問正答・全問誤答の問題
# ============================================================
print("\n=== 3. 全問正答・全問誤答の問題 ===")
patterns = {
    '全問正答 (1,1,1,1,1)': np.array([1, 1, 1, 1, 1]),
    '全問誤答 (0,0,0,0,0)': np.array([0, 0, 0, 0, 0]),
    '混合パターン (1,0,1,1,0)': np.array([1, 0, 1, 1, 0]),
}

fig, ax = plt.subplots(figsize=(10, 7))
colors = ['#e74c3c', '#3498db', '#2ecc71']

for (label, resp), color in zip(patterns.items(), colors):
    ll_vals = [log_likelihood(t, alpha, deltas, resp) for t in theta_grid]
    ax.plot(theta_grid, ll_vals, color=color, linewidth=2, label=label)

ax.set_title('対数尤度関数：全問正答・全問誤答ではピークがない')
ax.set_xlabel('能力 θ')
ax.set_ylabel('対数尤度 ln L(θ)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 4. 項目情報関数
# ============================================================
print("\n=== 4. 項目情報関数 ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for j, delta_j in enumerate(deltas):
    info_j = item_information(theta_grid, alpha, delta_j)
    axes[0].plot(theta_grid, info_j, linewidth=2,
                 label=f'項目{j+1}（δ={delta_j}）')
    print(f"項目{j+1}（δ={delta_j}）: ピーク情報量 = {info_j.max():.4f}")

axes[0].set_title('項目情報関数 I_j(θ)')
axes[0].set_xlabel('能力 θ')
axes[0].set_ylabel('情報量 I_j(θ)')
axes[0].legend(fontsize=12)
axes[0].grid(alpha=0.3)

for alpha_val in [0.5, 1.0, 1.5, 2.0]:
    info_alpha = item_information(theta_grid, alpha_val, 0.0)
    axes[1].plot(theta_grid, info_alpha, linewidth=2,
                 label=f'α = {alpha_val}')

axes[1].set_title('識別力αと項目情報関数の関係（δ=0固定）')
axes[1].set_xlabel('能力 θ')
axes[1].set_ylabel('情報量 I_j(θ)')
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 5. テスト情報関数とSEE
# ============================================================
print("\n=== 5. テスト情報関数とSEE ===")
test_info = test_information(theta_grid, alpha, deltas)
see = 1.0 / np.sqrt(np.maximum(test_info, 1e-10))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for j, delta_j in enumerate(deltas):
    info_j = item_information(theta_grid, alpha, delta_j)
    axes[0].fill_between(theta_grid, info_j, alpha=0.15)
    axes[0].plot(theta_grid, info_j, linewidth=1, alpha=0.6, label=f'項目{j+1}')
axes[0].plot(theta_grid, test_info, 'k-', linewidth=2.5,
             label='テスト情報関数 I(θ)')
axes[0].set_title('テスト情報関数と各項目の情報量')
axes[0].set_xlabel('能力 θ')
axes[0].set_ylabel('情報量')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

axes[1].plot(theta_grid, see, 'r-', linewidth=2)
axes[1].set_title('推定標準誤差 SEE(θ)')
axes[1].set_xlabel('能力 θ')
axes[1].set_ylabel('SEE(θ)')
axes[1].set_ylim(0, 3)
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 6. 信頼帯の構成
# ============================================================
print("\n=== 6. 信頼帯 ===")
info_at_mle = test_information(np.array([theta_mle]), alpha, deltas)[0]
see_at_mle = 1.0 / np.sqrt(info_at_mle)
ci_lower = theta_mle - 1.96 * see_at_mle
ci_upper = theta_mle + 1.96 * see_at_mle

print(f"MLE θ̂ = {theta_mle:.4f}")
print(f"SEE = {see_at_mle:.4f}")
print(f"95%信頼区間: [{ci_lower:.4f}, {ci_upper:.4f}]")

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(theta_grid, ll_values, 'b-', linewidth=2, label='対数尤度 ln L(θ)')
ax.axvline(theta_mle, color='red', linestyle='--', linewidth=1.5,
           label=f'MLE θ̂ = {theta_mle:.3f}')
ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='orange',
           label=f'95%信頼区間 [{ci_lower:.2f}, {ci_upper:.2f}]')
ax.set_title(f'最尤推定値と95%信頼帯（SEE = {see_at_mle:.3f}）')
ax.set_xlabel('能力 θ')
ax.set_ylabel('対数尤度 ln L(θ)')
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 7. テスト設計の比較
# ============================================================
print("\n=== 7. テスト設計の比較 ===")
deltas_focused = np.array([-0.5, -0.2, 0.0, 0.2, 0.5,
                           -0.3, 0.1, -0.1, 0.3, 0.4])
deltas_spread = np.array([-3.0, -2.0, -1.0, -0.5, 0.0,
                           0.5, 1.0, 2.0, 2.5, 3.0])
alpha_common = 1.2

info_focused = test_information(theta_grid, alpha_common, deltas_focused)
info_spread = test_information(theta_grid, alpha_common, deltas_spread)

see_focused = 1.0 / np.sqrt(np.maximum(info_focused, 1e-10))
see_spread = 1.0 / np.sqrt(np.maximum(info_spread, 1e-10))

idx_zero = np.argmin(np.abs(theta_grid))
print(f"設計A（集中）θ=0でのSEE: {see_focused[idx_zero]:.3f}")
print(f"設計B（分散）θ=0でのSEE: {see_spread[idx_zero]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].plot(theta_grid, info_focused, 'b-', linewidth=2,
             label='設計A：θ=0付近に集中（10項目）')
axes[0].plot(theta_grid, info_spread, 'r-', linewidth=2,
             label='設計B：広範囲に分散（10項目）')
axes[0].axvline(0, color='gray', linestyle=':', alpha=0.5)
axes[0].set_title('テスト情報関数の比較')
axes[0].set_xlabel('能力 θ')
axes[0].set_ylabel('情報量 I(θ)')
axes[0].legend(fontsize=12)
axes[0].grid(alpha=0.3)

axes[1].plot(theta_grid, see_focused, 'b-', linewidth=2, label='設計A：集中')
axes[1].plot(theta_grid, see_spread, 'r-', linewidth=2, label='設計B：分散')
axes[1].axvline(0, color='gray', linestyle=':', alpha=0.5)
axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.3)
axes[1].set_title('推定標準誤差 SEE(θ) の比較')
axes[1].set_xlabel('能力 θ')
axes[1].set_ylabel('SEE(θ)')
axes[1].set_ylim(0, 3)
axes[1].legend(fontsize=12)
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()
