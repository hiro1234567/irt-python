"""
T08: 部分得点モデル（PCM）と評定尺度モデル（RSM）｜Rasch系多値IRTの数理をPythonで導出

記事のPythonコードを統合したファイル。
記事URL: https://bigdata-analytics.jp/analytics/irt-polytomous-pcm-rsm/

Author: Hiroyuki Matsumoto (Digital Boy LLC)
Repository: https://github.com/hiro1234567/irt-python

実行方法:
  python t08_polytomous_pcm_rsm.py
    → CLIで一気に実行。グラフはウィンドウで順次表示される
  MPLBACKEND=Agg python t08_polytomous_pcm_rsm.py
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
    'legend.fontsize': 15,
    'font.family': 'Hiragino Sans',
})

# ============================================================
# Code Block 2
# ============================================================

def pcm_category_prob(theta, deltas):
    """
    PCM（部分得点モデル）のカテゴリ確率を計算する。
    
    Parameters
    ----------
    theta : float
        受験者の能力
    deltas : array-like
        遷移位置パラメータ [δ_j1, δ_j2, ...]
    
    Returns
    -------
    probs : numpy.ndarray
        各カテゴリの確率 [p_0, p_1, ..., p_m]
    """
    deltas = np.array(deltas)
    m = len(deltas)       # 遷移位置の数 = 最大カテゴリ値
    n_cat = m + 1         # カテゴリ数
    
    # 各カテゴリの累積和 Σ(θ - δ_jh) を計算
    cum_logits = np.zeros(n_cat)
    for k in range(1, n_cat):
        cum_logits[k] = cum_logits[k - 1] + (theta - deltas[k - 1])
    
    # 数値安定性のためにmax値を引いてからexpを取る
    cum_logits -= np.max(cum_logits)
    exp_vals = np.exp(cum_logits)
    
    return exp_vals / np.sum(exp_vals)

# ============================================================
# Code Block 3
# ============================================================

# 数値例の検証
deltas_example = [-1.0, 1.0]
theta_example = 0.5

probs = pcm_category_prob(theta_example, deltas_example)
print(f"θ = {theta_example}, δ = {deltas_example}")
for k, p in enumerate(probs):
    print(f"  p_{k} = {p:.3f}")

# ============================================================
# Code Block 4
# ============================================================

# PCMのカテゴリ反応関数（ORF）を描画
theta_range = np.linspace(-4, 4, 200)
deltas_1 = [-1.0, 1.0]  # 遷移位置が順序通り

fig, ax = plt.subplots(figsize=(8, 5))

probs_all = np.array([pcm_category_prob(t, deltas_1) for t in theta_range])

labels = ['カテゴリ 0', 'カテゴリ 1', 'カテゴリ 2']
colors = ['#2196F3', '#FF9800', '#4CAF50']
for k in range(probs_all.shape[1]):
    ax.plot(theta_range, probs_all[:, k], label=labels[k],
            color=colors[k], linewidth=2)

# 遷移位置を縦線で表示
for i, d in enumerate(deltas_1):
    ax.axvline(d, color='gray', linestyle='--', alpha=0.5)
    ax.text(d, 0.95, f'$\\delta_{{j{i+1}}}$ = {d}',
            ha='center', fontsize=12, color='gray')

ax.set_xlabel('能力 θ')
ax.set_ylabel('確率')
ax.set_title('PCM：カテゴリ反応関数（ORF）\n'
             f'δ = {deltas_1}')
ax.legend()
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig01_pcm_orf_ordered.png', bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 5
# ============================================================

# 遷移位置が逆転するケース
deltas_reversed = [1.0, -1.0]  # δ_j1 > δ_j2（逆転）

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：順序通り
probs_ord = np.array([pcm_category_prob(t, deltas_1) for t in theta_range])
for k in range(3):
    axes[0].plot(theta_range, probs_ord[:, k], label=labels[k],
                 color=colors[k], linewidth=2)
axes[0].set_title(f'順序通り：δ = {deltas_1}')
axes[0].set_xlabel('能力 θ')
axes[0].set_ylabel('確率')
axes[0].legend()
axes[0].set_ylim(0, 1.02)
axes[0].grid(True, alpha=0.3)

# 右：逆転
probs_rev = np.array([pcm_category_prob(t, deltas_reversed)
                       for t in theta_range])
for k in range(3):
    axes[1].plot(theta_range, probs_rev[:, k], label=labels[k],
                 color=colors[k], linewidth=2)
axes[1].set_title(f'逆転：δ = {deltas_reversed}')
axes[1].set_xlabel('能力 θ')
axes[1].set_ylabel('確率')
axes[1].legend()
axes[1].set_ylim(0, 1.02)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig02_pcm_orf_reversed.png', bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 6
# ============================================================

# 項目ごとにカテゴリ数が異なるケース
items = [
    {'name': '計算問題 (0-1点)', 'deltas': [0.0]},
    {'name': '応用問題 (0-2点)', 'deltas': [-0.5, 1.5]},
    {'name': '論述問題 (0-4点)', 'deltas': [-1.5, -0.5, 0.5, 2.0]},
]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
cmap = plt.cm.tab10

for idx, item in enumerate(items):
    ax = axes[idx]
    deltas = item['deltas']
    n_cat = len(deltas) + 1
    probs = np.array([pcm_category_prob(t, deltas) for t in theta_range])
    
    for k in range(n_cat):
        ax.plot(theta_range, probs[:, k], label=f'x={k}',
                color=cmap(k), linewidth=2)
    
    ax.set_title(item['name'], fontsize=14)
    ax.set_xlabel('能力 θ')
    if idx == 0:
        ax.set_ylabel('確率')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig03_pcm_variable_categories.png', bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 7
# ============================================================

def rsm_category_prob(theta, delta_j, taus):
    """
    RSM（評定尺度モデル）のカテゴリ確率を計算する。
    
    Parameters
    ----------
    theta : float
        受験者の能力
    delta_j : float
        項目位置パラメータ
    taus : array-like
        閾値パラメータ [τ_1, τ_2, ...]（全項目共通）
    
    Returns
    -------
    probs : numpy.ndarray
        各カテゴリの確率 [p_0, p_1, ..., p_m]
    """
    taus = np.array(taus)
    m = len(taus)         # 閾値の数 = 最大カテゴリ値
    n_cat = m + 1         # カテゴリ数
    
    # カテゴリ係数 κ を計算（κ_0 = 0, κ_k = -Στ_h）
    kappa = np.zeros(n_cat)
    for k in range(1, n_cat):
        kappa[k] = kappa[k - 1] - taus[k - 1]
    
    # 各カテゴリのlogit値を計算
    logits = np.zeros(n_cat)
    for k in range(n_cat):
        logits[k] = kappa[k] + k * (theta - delta_j)
    
    # 数値安定性のためにmax値を引く
    logits -= np.max(logits)
    exp_vals = np.exp(logits)
    
    return exp_vals / np.sum(exp_vals)

# ============================================================
# Code Block 8
# ============================================================

# RSMの2項目比較
taus_common = [-0.30, -0.02, 0.32]  # 全項目共通の閾値
delta_items = [-0.98, 0.70]         # 2つの項目位置

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
labels_4cat = ['全くそう思わない', 'そう思わない',
               'そう思う', '非常にそう思う']
colors_4 = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

for idx, d_j in enumerate(delta_items):
    ax = axes[idx]
    probs = np.array([rsm_category_prob(t, d_j, taus_common)
                       for t in theta_range])
    
    for k in range(4):
        ax.plot(theta_range, probs[:, k], label=labels_4cat[k],
                color=colors_4[k], linewidth=2)
    
    # 項目位置を縦線で表示
    ax.axvline(d_j, color='black', linestyle='--', alpha=0.5)
    ax.text(d_j + 0.1, 0.95, f'$\\delta_{idx+1}$ = {d_j}',
            fontsize=12, color='black')
    
    # 遷移位置を点線で表示
    for h, tau in enumerate(taus_common):
        loc = d_j + tau
        ax.axvline(loc, color='gray', linestyle=':', alpha=0.3)
    
    ax.set_title(f'項目{idx + 1}（$\\delta$ = {d_j}）', fontsize=15)
    ax.set_xlabel('能力 θ')
    if idx == 0:
        ax.set_ylabel('確率')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

plt.suptitle('RSM：項目位置が異なる2項目のORF\n'
             f'共通閾値 τ = {taus_common}', fontsize=16, y=1.04)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig04_rsm_two_items.png', bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 9
# ============================================================

def pcm_item_information(theta_arr, deltas):
    """
    PCM項目情報関数 I_j(θ) = Var(X_j | θ) を計算する。
    
    Parameters
    ----------
    theta_arr : numpy.ndarray
        θの配列
    deltas : array-like
        遷移位置パラメータ
    
    Returns
    -------
    info : numpy.ndarray
        各θでの情報量
    """
    deltas = np.array(deltas)
    m = len(deltas)
    n_cat = m + 1
    info = np.zeros(len(theta_arr))
    
    for i, t in enumerate(theta_arr):
        p = pcm_category_prob(t, deltas)
        k_vals = np.arange(n_cat)
        ex = np.sum(k_vals * p)        # E[X]
        ex2 = np.sum(k_vals**2 * p)    # E[X²]
        info[i] = ex2 - ex**2           # Var(X) = E[X²] - E[X]²
    
    return info

# ============================================================
# Code Block 10
# ============================================================

# カテゴリ数の違いによる情報量の比較
items_info = [
    {'label': '2カテゴリ (0-1)', 'deltas': [0.0]},
    {'label': '3カテゴリ (0-2)', 'deltas': [-0.5, 0.5]},
    {'label': '5カテゴリ (0-4)', 'deltas': [-1.5, -0.5, 0.5, 1.5]},
]

fig, ax = plt.subplots(figsize=(8, 5))
colors_info = ['#2196F3', '#FF9800', '#4CAF50']

for idx, item in enumerate(items_info):
    info = pcm_item_information(theta_range, item['deltas'])
    ax.plot(theta_range, info, label=item['label'],
            color=colors_info[idx], linewidth=2)

ax.set_xlabel('能力 θ')
ax.set_ylabel('情報量 $I_j(\\theta)$')
ax.set_title('カテゴリ数による情報量の違い（PCM）')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig05_pcm_info_categories.png', bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 11
# ============================================================

def rsm_item_information(theta_arr, delta_j, taus):
    """RSM項目情報関数"""
    m = len(taus)
    n_cat = m + 1
    info = np.zeros(len(theta_arr))
    
    for i, t in enumerate(theta_arr):
        p = rsm_category_prob(t, delta_j, taus)
        k_vals = np.arange(n_cat)
        ex = np.sum(k_vals * p)
        ex2 = np.sum(k_vals**2 * p)
        info[i] = ex2 - ex**2
    
    return info

# RSMの情報関数：項目位置の違い
taus_common = [-0.30, -0.02, 0.32]
deltas_rsm = [-1.0, 0.0, 1.0]

fig, ax = plt.subplots(figsize=(8, 5))
colors_rsm = ['#2196F3', '#FF9800', '#4CAF50']

for idx, d_j in enumerate(deltas_rsm):
    info = rsm_item_information(theta_range, d_j, taus_common)
    ax.plot(theta_range, info,
            label=f'項目{idx+1}（$\\delta$ = {d_j}）',
            color=colors_rsm[idx], linewidth=2)

# テスト情報関数（合計）
test_info = sum(rsm_item_information(theta_range, d, taus_common)
                for d in deltas_rsm)
ax.plot(theta_range, test_info, label='テスト情報関数',
        color='black', linewidth=2.5, linestyle='--')

ax.set_xlabel('能力 θ')
ax.set_ylabel('情報量 $I(\\theta)$')
ax.set_title('RSMの項目情報関数とテスト情報関数\n'
             f'共通閾値 τ = {taus_common}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig06_rsm_info_functions.png', bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 12
# ============================================================

# テスト特性曲線（TCF）の描画
items_test = [
    {'deltas': [-1.5, -0.5]},   # 易しい項目
    {'deltas': [-0.5, 0.5]},    # 中程度の項目
    {'deltas': [0.5, 1.5]},     # 難しい項目
    {'deltas': [-1.0, 0.0]},    # やや易しい項目
    {'deltas': [0.0, 1.0]},     # やや難しい項目
]

# 各θでの期待テスト得点を計算
expected_scores = np.zeros(len(theta_range))
for item in items_test:
    for i, t in enumerate(theta_range):
        p = pcm_category_prob(t, item['deltas'])
        k_vals = np.arange(len(item['deltas']) + 1)
        expected_scores[i] += np.sum(k_vals * p)

max_score = sum(len(item['deltas']) for item in items_test)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(theta_range, expected_scores, color='#2196F3', linewidth=2.5)
ax.set_xlabel('能力 θ')
ax.set_ylabel('期待テスト得点 $T$')
ax.set_title(f'テスト特性曲線（PCM、{len(items_test)}項目、'
             f'最大得点 = {max_score}点）')
ax.set_ylim(0, max_score + 0.5)
ax.axhline(max_score / 2, color='gray', linestyle=':', alpha=0.5)
ax.text(-3.5, max_score / 2 + 0.2, f'得点 = {max_score/2:.1f}',
        fontsize=11, color='gray')
ax.grid(True, alpha=0.3)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('fig07_pcm_tcf.png', bbox_inches='tight')
plt.show()

