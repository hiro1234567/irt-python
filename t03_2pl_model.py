"""
T03: IRTの2パラメータモデル：識別力αを項目ごとに解放するとIRFはどう変わるか

記事のPythonコードを統合したファイル。
記事URL: https://bigdata-analytics.jp/analytics/irt-2pl-model/

Author: Hiroyuki Matsumoto (Digital Boy LLC)
Repository: https://github.com/hiro1234567/irt-python

実行方法:
  python t03_2pl_model.py
    → CLIで一気に実行。グラフはウィンドウで順次表示される
  MPLBACKEND=Agg python t03_2pl_model.py
    → グラフ表示せず数値だけ確認
  Jupyterで Code Block を1つずつコピペして対話的に試すのも可
"""

# ============================================================
# Code Block 1
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

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

theta = np.linspace(-4, 4, 1000)

def irf_2pl(th, alpha, delta):
    """2PLモデルの項目反応関数"""
    return 1 / (1 + np.exp(-alpha * (th - delta)))

alphas = [0.5, 1.0, 1.5, 2.0, 3.0]
delta_common = 0.0
colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']

fig, ax = plt.subplots(figsize=(8, 5))
for a, c in zip(alphas, colors):
    ax.plot(theta, irf_2pl(theta, a, delta_common), label=f'α = {a}', color=c, linewidth=2)
ax.set_xlabel('能力 θ')
ax.set_ylabel('正答確率 P(θ)')
ax.set_title('識別力αを変えたときのIRF（δ = 0.0）')
ax.legend()
ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
ax.axvline(0.0, color='gray', linestyle=':', linewidth=0.8)
ax.set_ylim(-0.02, 1.02)
fig.tight_layout()
plt.show()

# ============================================================
# Code Block 3
# ============================================================

items = [
    {'alpha': 0.8, 'delta': -1.0, 'label': '項目A（α=0.8, δ=−1.0）'},
    {'alpha': 2.0, 'delta':  0.5, 'label': '項目B（α=2.0, δ=0.5）'},
    {'alpha': 1.5, 'delta': -0.5, 'label': '項目C（α=1.5, δ=−0.5）'},
]

fig, ax = plt.subplots(figsize=(8, 5))
for item, c in zip(items, colors[:3]):
    ax.plot(theta, irf_2pl(theta, item['alpha'], item['delta']),
            label=item['label'], color=c, linewidth=2)
ax.set_xlabel('能力 θ')
ax.set_ylabel('正答確率 P(θ)')
ax.set_title('2PLモデルでのIRFの交差')
ax.legend(fontsize=11)
ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
ax.set_ylim(-0.02, 1.02)
fig.tight_layout()
plt.show()

# ============================================================
# Code Block 4
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(theta, irf_2pl(theta, 1.5, 0.0), label='α = 1.5（正常）', color='#4e79a7', linewidth=2)
ax.plot(theta, irf_2pl(theta, -1.0, 0.0), label='α = −1.0（異常）', color='#e15759', linewidth=2, linestyle='--')
ax.set_xlabel('能力 θ')
ax.set_ylabel('正答確率 P(θ)')
ax.set_title('αが負の場合：能力が高いほど正答率が下がる')
ax.legend()
ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
ax.axvline(0.0, color='gray', linestyle=':', linewidth=0.8)
ax.set_ylim(-0.02, 1.02)
fig.tight_layout()
plt.show()

# ============================================================
# Code Block 5
# ============================================================

def info_2pl(th, alpha, delta):
    """2PLモデルの項目情報関数"""
    p = irf_2pl(th, alpha, delta)
    return alpha**2 * p * (1 - p)

fig, ax = plt.subplots(figsize=(8, 5))
for a, c in zip(alphas, colors):
    ax.plot(theta, info_2pl(theta, a, delta_common), label=f'α = {a}', color=c, linewidth=2)
ax.set_xlabel('能力 θ')
ax.set_ylabel('情報量 I(θ)')
ax.set_title('αごとの項目情報関数（δ = 0.0）')
ax.legend()
ax.set_ylim(0, 2.4)
fig.tight_layout()
plt.show()

# ============================================================
# Code Block 6
# ============================================================

items_2pl = [
    {'alpha': 1.2, 'delta': -2.0},
    {'alpha': 2.0, 'delta': -0.5},
    {'alpha': 1.6, 'delta': -0.2},
    {'alpha': 1.5, 'delta':  0.3},
    {'alpha': 1.0, 'delta':  0.6},
]

test_info_2pl = np.zeros_like(theta)
for item in items_2pl:
    test_info_2pl += info_2pl(theta, item['alpha'], item['delta'])

alpha_mean = np.mean([it['alpha'] for it in items_2pl])
test_info_1pl = np.zeros_like(theta)
for item in items_2pl:
    test_info_1pl += info_2pl(theta, alpha_mean, item['delta'])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(theta, test_info_2pl, label='2PLモデル', color='#4e79a7', linewidth=2.5)
ax.plot(theta, test_info_1pl, label=f'1PLモデル（α={alpha_mean:.2f}共通）',
        color='#e15759', linewidth=2, linestyle='--')
ax.set_xlabel('能力 θ')
ax.set_ylabel('テスト情報量 I(θ)')
ax.set_title('テスト情報関数の比較（1PL vs 2PL）')
ax.legend()
fig.tight_layout()
plt.show()

# ============================================================
# Code Block 7
# ============================================================

re = np.where(test_info_1pl > 1e-10, test_info_2pl / test_info_1pl, np.nan)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(theta, re, color='#4e79a7', linewidth=2.5)
ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('能力 θ')
ax.set_ylabel('相対効率 RE(2PL / 1PL)')
ax.set_title('相対効率プロット（2PL / 1PL）')
ax.set_ylim(0, 2.0)
fig.tight_layout()
plt.show()

# ============================================================
# Code Block 8
# ============================================================

from mpl_toolkits.mplot3d import Axes3D

# α-δの探索グリッドを生成
alpha_grid = np.linspace(0.3, 3.5, 80)
delta_grid = np.linspace(-3, 3, 80)
A, D = np.meshgrid(alpha_grid, delta_grid)

# 真のパラメータでシミュレーションデータを生成
np.random.seed(42)
n_persons = 500
theta_true = np.random.normal(0, 1, n_persons)  # 500人の能力値（標準正規分布）
delta_true = 0.5
alpha_true = 1.8
responses = (np.random.rand(n_persons) < irf_2pl(theta_true, alpha_true, delta_true)).astype(int)

# グリッド上の各(α, δ)で対数尤度を計算
lnL = np.zeros_like(A)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        p = irf_2pl(theta_true, A[i, j], D[i, j])
        p = np.clip(p, 1e-10, 1 - 1e-10)
        lnL[i, j] = np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p))

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, D, lnL, cmap='viridis', alpha=0.85, edgecolor='none')
ax.set_xlabel('α（識別力）', labelpad=10)
ax.set_ylabel('δ（項目位置）', labelpad=10)
ax.set_zlabel('対数尤度 ln L', labelpad=10)
ax.set_title('2PLモデルの対数尤度曲面')
ax.view_init(elev=30, azim=-60)
fig.tight_layout()
plt.show()

# ============================================================
# Code Block 9
# ============================================================

items_demo = [
    {'alpha': 2.0, 'delta': -1.5, 'label': '項目1'},
    {'alpha': 1.5, 'delta': -0.5, 'label': '項目2'},
    {'alpha': 1.0, 'delta':  0.0, 'label': '項目3'},
    {'alpha': 0.8, 'delta':  0.5, 'label': '項目4'},
    {'alpha': 0.5, 'delta':  1.5, 'label': '項目5'},
]

pattern_a = np.array([1, 1, 1, 0, 0])
pattern_b = np.array([0, 0, 1, 1, 1])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax_i, (pattern, label) in enumerate(
    [(pattern_a, '応答パターン (1,1,1,0,0)'), (pattern_b, '応答パターン (0,0,1,1,1)')]):
    ax = axes[ax_i]
    theta_fine = np.linspace(-4, 4, 500)
    log_lik = np.zeros_like(theta_fine)
    for th_idx, th in enumerate(theta_fine):
        ll = 0
        for j, item in enumerate(items_demo):
            p = irf_2pl(th, item['alpha'], item['delta'])
            p = np.clip(p, 1e-10, 1 - 1e-10)
            ll += pattern[j] * np.log(p) + (1 - pattern[j]) * np.log(1 - p)
        log_lik[th_idx] = ll
    ax.plot(theta_fine, log_lik, color='#4e79a7', linewidth=2.5)
    mle_idx = np.argmax(log_lik)
    ax.axvline(theta_fine[mle_idx], color='#e15759', linestyle='--', linewidth=1.5,
               label=f'MLE θ̂ = {theta_fine[mle_idx]:.2f}')
    ax.set_xlabel('能力 θ')
    ax.set_ylabel('対数尤度 ln L(θ)')
    ax.set_title(label)
    ax.legend()
fig.suptitle('同じ合計点（X=3）でも推定されるθが異なる', fontsize=16, y=1.02)
fig.tight_layout()
plt.show()

# ============================================================
# Code Block 10
# ============================================================

def irf_2pl(th, alpha, delta):
    return 1 / (1 + np.exp(-alpha * (th - delta)))

# ============================================================
# Code Block 11
# ============================================================

def info_2pl(th, alpha, delta):
    p = irf_2pl(th, alpha, delta)
    return alpha**2 * p * (1 - p)

