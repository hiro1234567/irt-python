"""
T1: IRTの数理的基礎 — ロジスティックモデルとICCの導出
https://bigdata-analytics.jp/analytics/irt-mathematical-foundations/

de Ayala (2022) Ch.1-2前半に対応。
CTTの限界 → ラッシュモデル → 1PLモデル → α,δの幾何学的意味 → 3つの仮定
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 150,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

theta_range = np.linspace(-4, 4, 500)


# ============================================================
# 1. CTTの限界：テストが変われば順位が変わる
# ============================================================
def generate_responses(theta, delta, alpha=1.0):
    """ロジスティックモデルに基づいて0/1の応答を生成する"""
    prob = 1 / (1 + np.exp(-alpha * (theta[:, None] - delta[None, :])))
    return (np.random.rand(len(theta), len(delta)) < prob).astype(int)


np.random.seed(42)
n_persons = 100
n_items = 20

theta = np.random.normal(0, 1, n_persons)
delta_easy = np.random.normal(-1.0, 0.5, n_items)
delta_hard = np.random.normal(1.0, 0.5, n_items)

resp_easy = generate_responses(theta, delta_easy)
resp_hard = generate_responses(theta, delta_hard)

rank_easy = np.argsort(np.argsort(-resp_easy.sum(axis=1)))
rank_hard = np.argsort(np.argsort(-resp_hard.sum(axis=1)))
rank_diff = np.abs(rank_easy - rank_hard)

print(f"=== CTTの限界 ===")
print(f"順位変動の平均: {rank_diff.mean():.1f}位")
print(f"順位変動が10位以上の人数: {(rank_diff >= 10).sum()}人 / {n_persons}人")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(rank_easy, rank_hard, alpha=0.6, s=40, c='#3498db')
axes[0].plot([0, 100], [0, 100], 'r--', alpha=0.5, linewidth=1.5)
axes[0].set_title('テスト変更による順位変動')
axes[0].set_xlabel('易しいテストでの順位')
axes[0].set_ylabel('難しいテストでの順位')
axes[0].grid(alpha=0.3)

axes[1].hist(rank_diff, bins=20, color='#e74c3c', alpha=0.7, edgecolor='white')
axes[1].axvline(rank_diff.mean(), color='black', linestyle='--', linewidth=2,
                label=f'平均変動: {rank_diff.mean():.1f}位')
axes[1].set_title('順位変動の分布')
axes[1].set_xlabel('順位変動（絶対値）')
axes[1].set_ylabel('人数')
axes[1].legend(fontsize=14)
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 2. ラッシュモデル：最もシンプルなIRT
# ============================================================
def rasch_model(theta, delta):
    """ラッシュモデル：能力θと項目位置δの差から正答確率を計算"""
    return np.exp(theta - delta) / (1 + np.exp(theta - delta))


fig, ax = plt.subplots(figsize=(10, 7))
for delta_val in [-2.0, -1.0, 0.0, 1.0, 2.0]:
    p = rasch_model(theta_range, delta_val)
    ax.plot(theta_range, p, label=f'δ = {delta_val}', linewidth=2)

ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_title('ラッシュモデル：項目位置δの効果')
ax.set_xlabel('能力 θ')
ax.set_ylabel('正答確率 p(θ)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 3. 1PLモデル：共通識別力αの導入
# ============================================================
def one_pl(theta, alpha, delta):
    """1PLモデル：能力θ、識別力α、項目位置δから正答確率を計算"""
    return 1 / (1 + np.exp(-alpha * (theta - delta)))


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for delta_val in [-2, -1, 0, 1, 2]:
    axes[0].plot(theta_range, one_pl(theta_range, 1.5, delta_val),
                 label=f'δ = {delta_val}', linewidth=2)
axes[0].set_title('項目位置 δ の効果（α = 1.5 固定）')
axes[0].set_xlabel('能力 θ')
axes[0].set_ylabel('正答確率 p(θ)')
axes[0].legend()
axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
axes[0].grid(alpha=0.3)

for alpha_val in [0.5, 1.0, 1.5, 2.0]:
    axes[1].plot(theta_range, one_pl(theta_range, alpha_val, 0),
                 label=f'α = {alpha_val}', linewidth=2)
axes[1].set_title('識別力 α の効果（δ = 0.0 固定）')
axes[1].set_xlabel('能力 θ')
axes[1].set_ylabel('正答確率 p(θ)')
axes[1].legend()
axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 4. αの幾何学的意味：変曲点の接線
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))
delta_val = 0.0

for alpha_val in [-1.0, 0.5, 1.0, 2.0]:
    p = one_pl(theta_range, alpha_val, delta_val)
    label = f'α = {alpha_val}' + (' （逆転項目）' if alpha_val < 0 else '')
    ax.plot(theta_range, p, label=label, linewidth=2)

    slope = alpha_val / 4
    tangent = 0.5 + slope * (theta_range - delta_val)
    mask = (tangent > -0.1) & (tangent < 1.1)
    ax.plot(theta_range[mask], tangent[mask], '--', alpha=0.6, linewidth=1.5)

ax.axhline(0.5, color='gray', linestyle=':', alpha=0.4)
ax.axvline(delta_val, color='gray', linestyle=':', alpha=0.4)
ax.set_title('識別力 α と変曲点の接線（α<0 は逆転項目）')
ax.set_xlabel('能力 θ')
ax.set_ylabel('正答確率 p(θ)')
ax.legend()
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3)
plt.show()

print(f"\n=== 変曲点の傾き α/4 ===")
for a in [0.5, 1.0, 1.5, 2.0]:
    print(f"α = {a:.1f} → 傾き = {a/4:.3f}")


# ============================================================
# 5. 5項目のIRFを重ね描き
# ============================================================
items = [
    {'alpha': 0.6, 'delta': -2.0, 'label': '問1（易・低識別）'},
    {'alpha': 1.5, 'delta': -1.0, 'label': '問2（やや易・高識別）'},
    {'alpha': 1.0, 'delta': 0.0, 'label': '問3（標準）'},
    {'alpha': 1.8, 'delta': 0.5, 'label': '問4（やや難・高識別）'},
    {'alpha': 0.8, 'delta': 2.0, 'label': '問5（難・低識別）'},
]

fig, ax = plt.subplots(figsize=(10, 7))
for item in items:
    p = one_pl(theta_range, item['alpha'], item['delta'])
    ax.plot(theta_range, p, label=item['label'], linewidth=2)

ax.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.4)
ax.set_title('5項目のIRF重ね描き')
ax.set_xlabel('能力 θ')
ax.set_ylabel('正答確率 p(θ)')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.show()
