"""
T12: テスト情報関数とテスト設計：最適な問題セットをPythonで設計する
https://bigdata-analytics.jp/analytics/irt-test-information-design/

記事内コードブロックの統合版。

実行方法:
  python t11_test_information_design.py
    → CLIで一気に実行。グラフはウィンドウで順次表示される
  MPLBACKEND=Agg python t11_test_information_design.py
    → グラフ表示せず数値だけ確認
  Jupyterで Code Block を1つずつコピペして対話的に試すのも可
"""

# ============================================================
# Code Block 1: import と matplotlib 共通設定
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

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
# Code Block 2: 共通関数とテスト設定（10項目・2PL）
# ============================================================

def irf_2pl(theta, alpha, delta):
    """2PLモデルのIRF（項目反応関数）"""
    return 1.0 / (1.0 + np.exp(-alpha * (theta - delta)))

def info_2pl(theta, alpha, delta):
    """2PLモデルの項目情報関数"""
    p = irf_2pl(theta, alpha, delta)
    return alpha**2 * p * (1 - p)

# テスト設定（10項目・2PLモデル）
alphas = np.array([1.2, 1.5, 0.8, 2.0, 1.0, 1.8, 1.3, 0.9, 1.6, 1.1])
deltas = np.array([-2.0, -1.2, -0.5, -0.1, 0.3, 0.7, 1.0, 1.5, 2.0, 2.5])
n_items = len(alphas)

theta = np.linspace(-4, 4, 500)

# ============================================================
# Code Block 3: テスト特性曲線（TCF）の可視化
# ============================================================

tcf = np.zeros_like(theta)
for a, d in zip(alphas, deltas):
    tcf += irf_2pl(theta, a, d)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(theta, tcf, color='#4e79a7', linewidth=2.5)
ax.axhline(n_items / 2, color='gray', linestyle=':', alpha=0.5, label=f'期待得点 = {n_items/2:.0f}（半分正答）')
ax.set_xlabel('能力 θ')
ax.set_ylabel('期待得点 T(θ)')
ax.set_title('テスト特性曲線（TCF）：10項目・2PLモデル')
ax.set_ylim(0, n_items + 0.5)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# ============================================================
# Code Block 4: テスト情報関数と推定標準誤差 SEE
# ============================================================

test_info = np.zeros_like(theta)
for a, d in zip(alphas, deltas):
    test_info += info_2pl(theta, a, d)

# SEE = 1/√I(θ)（情報量が極端に小さい端ではclip）
see = 1.0 / np.sqrt(np.maximum(test_info, 1e-10))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：テスト情報関数（項目情報の積み上げ）
colors = plt.cm.tab10(np.linspace(0, 1, n_items))
for i, (a, d) in enumerate(zip(alphas, deltas)):
    item_info = info_2pl(theta, a, d)
    axes[0].fill_between(theta, 0, item_info, alpha=0.15, color=colors[i])
    axes[0].plot(theta, item_info, color=colors[i], linewidth=0.8,
                 label=f'$j_{{{i+1}}}$: α={a}, δ={d}')
axes[0].plot(theta, test_info, 'k-', linewidth=2.5, label='テスト情報関数 I(θ)')
axes[0].set_xlabel('能力 θ')
axes[0].set_ylabel('情報量')
axes[0].set_title('項目情報関数とテスト情報関数')
axes[0].legend(fontsize=8, ncol=2, loc='upper right')

# 右：SEE
axes[1].plot(theta, see, color='#e15759', linewidth=2.5)
axes[1].axhline(0.3, color='green', linestyle='--', alpha=0.4, label='SEE = 0.3（高精度ライン）')
axes[1].axhline(0.5, color='orange', linestyle='--', alpha=0.4, label='SEE = 0.5（中精度ライン）')
axes[1].set_xlabel('能力 θ')
axes[1].set_ylabel('SEE(θ)')
axes[1].set_title('推定標準誤差 SEE(θ)')
axes[1].set_ylim(0, 2.5)
axes[1].legend()

plt.tight_layout()
plt.show()

# ============================================================
# Code Block 5: 項目プール（40項目）の生成と分布可視化
# ============================================================

np.random.seed(42)

# 項目プール：40項目（αは0.6〜2.5、δは−3〜+3の範囲でランダム生成）
pool_size = 40
pool_alphas = np.round(np.random.uniform(0.6, 2.5, pool_size), 2)
pool_deltas = np.round(np.random.uniform(-3.0, 3.0, pool_size), 2)
n_select = 10  # テストに選ぶ項目数

# 項目プール分布の可視化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(pool_alphas, bins=12, color='#4e79a7', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('識別力 α')
axes[0].set_ylabel('項目数')
axes[0].set_title('項目プールのα分布')

axes[1].hist(pool_deltas, bins=12, color='#e15759', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('項目位置 δ')
axes[1].set_ylabel('項目数')
axes[1].set_title('項目プールのδ分布')
plt.tight_layout()
plt.show()

# ============================================================
# Code Block 6: peaked 型（合否判定）の項目選択
# ============================================================

theta_c = 0.0  # カットポイント（合格ライン）

# 各項目のカットポイントでの情報量を計算
info_at_cut = np.array([info_2pl(np.array([theta_c]), a, d)[0]
                         for a, d in zip(pool_alphas, pool_deltas)])

# 情報量の大きい順に10項目を選択
peaked_indices = np.argsort(info_at_cut)[::-1][:n_select]
peaked_alphas = pool_alphas[peaked_indices]
peaked_deltas = pool_deltas[peaked_indices]

print("peaked型テスト（θ_c = 0.0 での情報量Top10）:")
print(f"  α = {peaked_alphas}")
print(f"  δ = {peaked_deltas}")
print(f"  θ_c での情報量 = {info_at_cut[peaked_indices].round(3)}")

# ============================================================
# Code Block 7: rectangular 型（等精度）の項目選択（貪欲法）
# ============================================================

theta_L, theta_U = -2.0, 2.0  # 等精度を狙うθの範囲
theta_eval = np.linspace(theta_L, theta_U, 200)

# 目標情報関数：指定範囲内で一定値
target_info_value = 2.0
target_info = np.full_like(theta_eval, target_info_value)

# 貪欲法で項目を選択
remaining = list(range(pool_size))
selected_rect = []

for _ in range(n_select):
    best_idx = None
    best_error = np.inf

    for idx in remaining:
        # 候補を追加したときのテスト情報関数を計算
        trial = selected_rect + [idx]
        trial_info = np.zeros_like(theta_eval)
        for i in trial:
            trial_info += info_2pl(theta_eval, pool_alphas[i], pool_deltas[i])

        # 目標情報関数との二乗誤差
        error = np.sum((trial_info - target_info) ** 2)
        if error < best_error:
            best_error = error
            best_idx = idx

    selected_rect.append(best_idx)
    remaining.remove(best_idx)

rect_alphas = pool_alphas[selected_rect]
rect_deltas = pool_deltas[selected_rect]

print("\nrectangular型テスト（θ ∈ [−2, 2] で等精度を狙う）:")
print(f"  α = {rect_alphas}")
print(f"  δ = {rect_deltas}")

# ============================================================
# Code Block 8: peaked vs rectangular の比較プロット
# ============================================================

# peaked型のテスト情報関数
info_peaked = np.zeros_like(theta)
for a, d in zip(peaked_alphas, peaked_deltas):
    info_peaked += info_2pl(theta, a, d)

# rectangular型のテスト情報関数
info_rect = np.zeros_like(theta)
for a, d in zip(rect_alphas, rect_deltas):
    info_rect += info_2pl(theta, a, d)

# SEE
see_peaked = 1.0 / np.sqrt(np.maximum(info_peaked, 1e-10))
see_rect = 1.0 / np.sqrt(np.maximum(info_rect, 1e-10))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：テスト情報関数の比較
axes[0].plot(theta, info_peaked, color='#4e79a7', linewidth=2.5, label='peaked型（合否判定）')
axes[0].plot(theta, info_rect, color='#e15759', linewidth=2.5, label='rectangular型（等精度）')
axes[0].axvline(theta_c, color='gray', linestyle=':', alpha=0.5, label=f'カットポイント θ = {theta_c}')
axes[0].fill_between(theta, 0, target_info_value,
                     where=(theta >= theta_L) & (theta <= theta_U),
                     alpha=0.08, color='red', label=f'目標: I(θ) = {target_info_value}')
axes[0].set_xlabel('能力 θ')
axes[0].set_ylabel('テスト情報量 I(θ)')
axes[0].set_title('テスト情報関数の比較')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# 右：SEEの比較
axes[1].plot(theta, see_peaked, color='#4e79a7', linewidth=2.5, label='peaked型')
axes[1].plot(theta, see_rect, color='#e15759', linewidth=2.5, label='rectangular型')
axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.3)
axes[1].set_xlabel('能力 θ')
axes[1].set_ylabel('SEE(θ)')
axes[1].set_title('推定標準誤差の比較')
axes[1].set_ylim(0, 3.0)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# Code Block 9: 総合情報面積指標 I_A の計算
# ============================================================

# 数値積分でテスト情報関数の面積を計算
area_peaked_numerical = trapezoid(info_peaked, theta)
area_rect_numerical = trapezoid(info_rect, theta)

# 近似式 I_A = Σα_j（2PLモデル）
ia_peaked_approx = np.sum(peaked_alphas)
ia_rect_approx = np.sum(rect_alphas)

print("=== 総合情報面積指標 I_A ===")
print(f"\npeaked型テスト:")
print(f"  数値積分（面積）: {area_peaked_numerical:.3f}")
print(f"  近似式 Σα_j:     {ia_peaked_approx:.3f}")

print(f"\nrectangular型テスト:")
print(f"  数値積分（面積）: {area_rect_numerical:.3f}")
print(f"  近似式 Σα_j:     {ia_rect_approx:.3f}")

# ============================================================
# Code Block 10: 相対効率 RE プロット
# ============================================================

# RE = peaked / rectangular
re_peaked_vs_rect = np.where(info_rect > 1e-10,
                              info_peaked / info_rect, np.nan)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(theta, re_peaked_vs_rect, color='#4e79a7', linewidth=2.5)
ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
ax.axvline(theta_c, color='red', linestyle='--', alpha=0.5, label=f'カットポイント θ = {theta_c}')

# RE > 1 の領域をハイライト
ax.fill_between(theta, 1.0, re_peaked_vs_rect,
                where=re_peaked_vs_rect > 1.0,
                alpha=0.15, color='blue', label='peaked型が有利')
ax.fill_between(theta, re_peaked_vs_rect, 1.0,
                where=re_peaked_vs_rect < 1.0,
                alpha=0.15, color='red', label='rectangular型が有利')

ax.set_xlabel('能力 θ')
ax.set_ylabel('相対効率 RE(peaked / rectangular)')
ax.set_title('相対効率プロット：peaked型 vs rectangular型')
ax.set_ylim(0, 5.0)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# ============================================================
# Code Block 11: 項目数 vs SEE の関係
# ============================================================

n_items_list = [5, 10, 15, 20]
see_at_cut = []

# 項目プールから情報量順にn_select個選ぶ（peaked型）
for n_sel in n_items_list:
    indices = np.argsort(info_at_cut)[::-1][:n_sel]
    test_info_at_cut = np.sum(info_at_cut[indices])
    see_val = 1.0 / np.sqrt(test_info_at_cut)
    see_at_cut.append(see_val)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：項目数ごとのテスト情報関数
colors_n = ['#bab0ac', '#e15759', '#4e79a7', '#59a14f']
for n_sel, color in zip(n_items_list, colors_n):
    indices = np.argsort(info_at_cut)[::-1][:n_sel]
    test_info_n = np.zeros_like(theta)
    for idx in indices:
        test_info_n += info_2pl(theta, pool_alphas[idx], pool_deltas[idx])
    axes[0].plot(theta, test_info_n, color=color, linewidth=2,
                 label=f'{n_sel}項目')
axes[0].axvline(theta_c, color='gray', linestyle=':', alpha=0.5)
axes[0].set_xlabel('能力 θ')
axes[0].set_ylabel('テスト情報量 I(θ)')
axes[0].set_title('項目数を増やしたときのテスト情報関数（peaked型）')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 右：項目数 vs SEE
axes[1].plot(n_items_list, see_at_cut, 'o-', color='#4e79a7',
             linewidth=2.5, markersize=10)
axes[1].axhline(0.3, color='green', linestyle='--', alpha=0.4, label='SEE = 0.3 目標')
for n_sel, see_val in zip(n_items_list, see_at_cut):
    axes[1].annotate(f'SEE={see_val:.3f}', (n_sel, see_val),
                     textcoords='offset points', xytext=(10, 10), fontsize=12)
axes[1].set_xlabel('テストの項目数')
axes[1].set_ylabel('SEE(θ=0)')
axes[1].set_title('項目数とカットポイントでの推定精度')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
