"""
T17: Multilevel IRT × POS分析：購買データから顧客ロイヤルティを測定する

記事のPythonコードを統合したファイル。
記事URL: https://bigdata-analytics.jp/analytics/irt-multilevel-pos-analysis/

Author: Hiroyuki Matsumoto (Digital Boy LLC)
Repository: https://github.com/hiro1234567/irt-python

実行方法:
  python t17_multilevel_pos_analysis.py
    → CLIで一気に実行。グラフはウィンドウで順次表示される
  MPLBACKEND=Agg python t17_multilevel_pos_analysis.py
    → グラフ表示せず数値だけ確認
  Jupyterで Code Block を1つずつコピペして対話的に試すのも可
"""

# ============================================================
# Code Block 1
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')

SAVE_FIGS = False  # True にすると plt.savefig() で画像保存される

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

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# テストデータ（2値）
np.random.seed(42)
n_students, n_items = 8, 6
theta_test = np.random.randn(n_students) * 0.8
delta_test = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
prob_test = expit(theta_test[:, None] - delta_test[None, :])
responses_test = (np.random.rand(n_students, n_items) < prob_test).astype(int)

ax = axes[0]
im = ax.imshow(responses_test, cmap='Blues', aspect='auto', vmin=0, vmax=1)
ax.set_xlabel('項目')
ax.set_ylabel('受験者')
ax.set_title('テストデータ（2値：正答/誤答）')
ax.set_xticks(range(n_items))
ax.set_xticklabels([f'項目{j+1}' for j in range(n_items)], fontsize=11)
ax.set_yticks(range(n_students))
ax.set_yticklabels([f'受験者{i+1}' for i in range(n_students)], fontsize=11)
for i in range(n_students):
    for j in range(n_items):
        ax.text(j, i, str(responses_test[i, j]),
                ha='center', va='center', fontsize=13,
                color='white' if responses_test[i, j] == 1 else 'black')

# POSデータ（カウント）
n_customers, n_categories = 8, 6
theta_pos = np.random.randn(n_customers) * 0.6 + 2.0
delta_pos = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
lambda_pos = np.exp(theta_pos[:, None] - delta_pos[None, :])
purchases = np.random.poisson(lambda_pos)

ax = axes[1]
im2 = ax.imshow(purchases, cmap='Oranges', aspect='auto', vmin=0, vmax=purchases.max())
ax.set_xlabel('商品カテゴリ')
ax.set_ylabel('顧客')
ax.set_title('POSデータ（カウント：購買回数）')
ax.set_xticks(range(n_categories))
ax.set_xticklabels([f'カテゴリ{j+1}' for j in range(n_categories)], fontsize=11)
ax.set_yticks(range(n_customers))
ax.set_yticklabels([f'顧客{i+1}' for i in range(n_customers)], fontsize=11)
for i in range(n_customers):
    for j in range(n_categories):
        ax.text(j, i, str(purchases[i, j]),
                ha='center', va='center', fontsize=13,
                color='white' if purchases[i, j] > purchases.max()/2 else 'black')

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('t17_fig01_test_vs_pos.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 3
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

linear_pred = np.linspace(-3, 3, 200)

# logitリンク：確率（0〜1に収まる）
ax = axes[0]
prob = expit(linear_pred)
ax.plot(linear_pred, prob, 'b-', linewidth=2.5)
ax.set_xlabel('線形予測子 θ − δ')
ax.set_ylabel('正答確率 p')
ax.set_title('logitリンク（ベルヌーイ分布）')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

# logリンク：期待値（指数関数的に増加）
ax = axes[1]
lam = np.exp(linear_pred)
ax.plot(linear_pred, lam, 'r-', linewidth=2.5)
ax.set_xlabel('線形予測子 θ − δ')
ax.set_ylabel('期待購買回数 λ')
ax.set_title('logリンク（ポアソン分布）')
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylim(-0.5, 20)
ax.grid(True, alpha=0.3)

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('t17_fig02_link_functions.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 4
# ============================================================

# パラメータの具体例
theta_A = 1.5
deltas = {'生鮮食品': 0.5, '日配品': 1.0, '加工食品': 1.5, '嗜好品': 2.0, '日用品': 2.5}

print("=== 顧客A（θ = 1.5）の期待購買回数 ===\n")
print(f"{'カテゴリ':<10} {'δ':>5} {'θ-δ':>6} {'λ（期待回数）':>12}")
print("-" * 40)
for cat, d in deltas.items():
    lam = np.exp(theta_A - d)
    print(f"{cat:<10} {d:>5.1f} {theta_A - d:>6.1f} {lam:>12.2f}")

# 2つの顧客の比較
print("\n\n=== 顧客A（θ=1.5）vs 顧客B（θ=0.0）の比較 ===\n")
theta_B = 0.0
print(f"{'カテゴリ':<10} {'λ_A':>8} {'λ_B':>8} {'比率(A/B)':>10}")
print("-" * 40)
for cat, d in deltas.items():
    lam_A = np.exp(theta_A - d)
    lam_B = np.exp(theta_B - d)
    print(f"{cat:<10} {lam_A:>8.2f} {lam_B:>8.2f} {lam_A/lam_B:>10.2f}")
print(f"\n※ 比率は全カテゴリで一定: exp(θ_A - θ_B) = exp({theta_A - theta_B}) = {np.exp(theta_A - theta_B):.2f}")

# ============================================================
# Code Block 5
# ============================================================

# 重み行列Qの定義
categories = ['生鮮食品', 'パン・米', '惣菜', '加工食品', '飲料', '菓子', '酒類', '日用品']
Q = np.array([
    [0, 0],  # 生鮮食品
    [0, 0],  # パン・米
    [0, 1],  # 惣菜
    [0, 1],  # 加工食品
    [0, 1],  # 飲料
    [1, 1],  # 菓子
    [1, 1],  # 酒類
    [1, 1],  # 日用品
])
component_names = ['高価格帯', '低日常必需度']
L_cat = len(categories)
S_comp = Q.shape[1]

# 真のηパラメータ
eta_true = np.array([0.8, 0.5])  # 高価格帯の効果、低必需度の効果

# Q行列からδを計算
delta_lltm = Q @ eta_true

print("=== Q行列によるδの構造的分解 ===\n")
print(f"η_1（高価格帯の効果）= {eta_true[0]:.1f}")
print(f"η_2（低日常必需度の効果）= {eta_true[1]:.1f}\n")
print(f"{'カテゴリ':<8} {'q_1':>4} {'q_2':>4} {'δ':>6}")
print("-" * 28)
for j, cat in enumerate(categories):
    print(f"{cat:<8} {Q[j, 0]:>4} {Q[j, 1]:>4} {delta_lltm[j]:>6.1f}")

# ============================================================
# Code Block 6
# ============================================================

# LLTM+εのシミュレーション
np.random.seed(123)

N_sim = 300
sigma_theta = 0.7
sigma_eps = 0.4

# 顧客のランダム効果
theta_sim = np.random.randn(N_sim) * sigma_theta

# 商品カテゴリの構造（LLTM部分） + ランダム効果（ε部分）
delta_structural = Q @ eta_true
eps_j = np.random.randn(L_cat) * sigma_eps
delta_total = delta_structural + eps_j

# 期待購買回数と実現値
log_lambda = theta_sim[:, None] - delta_total[None, :]
lambda_sim = np.exp(log_lambda)
purchases_sim = np.random.poisson(lambda_sim)

# LLTMとLLTM+εのδの比較を可視化
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(L_cat)
width = 0.35

bars1 = ax.bar(x_pos - width/2, delta_structural, width, label='LLTM（構造部分のみ）',
               color='steelblue', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, delta_total, width, label='LLTM+ε（構造＋商品固有効果）',
               color='coral', alpha=0.8)

ax.set_xlabel('商品カテゴリ')
ax.set_ylabel('δ（購買されにくさ）')
ax.set_title('LLTMとLLTM+εによるδの推定の違い')
ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontsize=11, rotation=20, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('t17_fig03_lltm_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n商品カテゴリ固有効果（ε）の分散: {eps_j.var():.4f}（真の値: {sigma_eps**2:.2f}）")
print(f"LLTMのδとLLTM+εのδの相関: {np.corrcoef(delta_structural, delta_total)[0,1]:.4f}")

# ============================================================
# Code Block 7
# ============================================================

# 3レベルモデルのシミュレーション
np.random.seed(789)

G_stores = 5        # 店舗数
N_per_store = 100   # 店舗あたりの顧客数
N_total = G_stores * N_per_store
L_items = 6         # 商品カテゴリ数

sigma_store = 0.5   # 店舗間SD
sigma_person = 0.7  # 店舗内個人差SD
delta_3lev = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])

# 店舗効果
store_effects = np.random.randn(G_stores) * sigma_store
store_labels = np.repeat(np.arange(G_stores), N_per_store)

# 店舗内の個人差
person_effects = np.random.randn(N_total) * sigma_person

# 顧客ロイヤルティ = 店舗効果 + 個人差
theta_3lev = store_effects[store_labels] + person_effects

# 期待購買回数と実現値
log_lambda_3lev = theta_3lev[:, None] - delta_3lev[None, :]
lambda_3lev = np.exp(log_lambda_3lev)
purchases_3lev = np.random.poisson(lambda_3lev)

# 店舗ごとのθ分布を可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：店舗ごとのθの分布
ax = axes[0]
store_names = [f'店舗{g+1}' for g in range(G_stores)]
colors = plt.cm.Set2(np.linspace(0, 1, G_stores))

for g in range(G_stores):
    mask = store_labels == g
    ax.hist(theta_3lev[mask], bins=20, alpha=0.5, color=colors[g],
            label=f'{store_names[g]}（効果={store_effects[g]:.2f}）', density=True)

ax.set_xlabel('顧客ロイヤルティ θ')
ax.set_ylabel('密度')
ax.set_title('店舗ごとの顧客ロイヤルティ分布')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 右：2レベル vs 3レベルの分散分解
ax = axes[1]
var_total = theta_3lev.var()
var_between = store_effects[store_labels].var()
var_within = person_effects.var()

labels = ['全分散', '店舗間分散', '店舗内分散']
values = [var_total, var_between, var_within]
bar_colors = ['gray', 'steelblue', 'coral']

bars = ax.bar(labels, values, color=bar_colors, alpha=0.8, edgecolor='black')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=13)

ax.set_ylabel('分散')
ax.set_title('顧客ロイヤルティの分散分解')
ax.grid(True, alpha=0.3, axis='y')
icc = var_between / var_total
ax.text(0.95, 0.95, f'ICC = {icc:.3f}', transform=ax.transAxes,
        ha='right', va='top', fontsize=14,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('t17_fig04_three_level.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n=== 分散分解 ===")
print(f"全分散: {var_total:.4f}")
print(f"店舗間分散（σ²_g）: {var_between:.4f}（設定値: {sigma_store**2:.2f}）")
print(f"店舗内分散（σ²_θ）: {var_within:.4f}（設定値: {sigma_person**2:.2f}）")
print(f"ICC（級内相関係数）: {icc:.4f}")
print(f"→ 顧客ロイヤルティの分散のうち{icc*100:.1f}%が店舗間の差で説明される")

# ============================================================
# Code Block 8
# ============================================================

import pymc as pm
import arviz as az

# シミュレーションデータの生成
np.random.seed(42)

N_persons = 200
L_items_est = 8
sigma_theta_true = 0.8

# 真のパラメータ
theta_true = np.random.randn(N_persons) * sigma_theta_true
delta_true = np.array([-1.0, -0.5, 0.0, 0.3, 0.7, 1.0, 1.5, 2.0])

# 期待購買回数と実現値
log_lambda_true = theta_true[:, None] - delta_true[None, :]
lambda_true = np.exp(log_lambda_true)
y_obs = np.random.poisson(lambda_true)

# スタックドデータに変換
person_idx = np.repeat(np.arange(N_persons), L_items_est)
item_idx = np.tile(np.arange(L_items_est), N_persons)
y_stacked = y_obs.flatten()

print(f"データサイズ: {N_persons}人 × {L_items_est}カテゴリ = {len(y_stacked)}観測")
print(f"購買回数の範囲: {y_stacked.min()} 〜 {y_stacked.max()}")
print(f"購買回数の平均: {y_stacked.mean():.2f}")
print(f"購買回数が0の割合: {(y_stacked == 0).mean()*100:.1f}%")

# ============================================================
# Code Block 9
# ============================================================

# PyMCモデルの構築
with pm.Model() as poisson_irt_2level:
    # 事前分布
    sigma_theta_est = pm.HalfNormal('sigma_theta', sigma=1.0)
    theta_offset = pm.Normal('theta_offset', mu=0, sigma=1, shape=N_persons)
    theta_est = pm.Deterministic('theta', sigma_theta_est * theta_offset)
    
    # 項目パラメータ（固定効果）
    delta_est = pm.Normal('delta', mu=0, sigma=2, shape=L_items_est)
    
    # 線形予測子
    log_lambda_est = theta_est[person_idx] - delta_est[item_idx]
    
    # 尤度（ポアソン分布）
    y_like = pm.Poisson('y', mu=pm.math.exp(log_lambda_est), observed=y_stacked)
    
    # MCMC推定
    trace_2level = pm.sample(2000, tune=1000, cores=2, random_seed=42,
                             target_accept=0.9)

# ============================================================
# Code Block 10
# ============================================================

# パラメータ回復の確認
delta_posterior = trace_2level.posterior['delta'].mean(dim=['chain', 'draw']).values
theta_posterior = trace_2level.posterior['theta'].mean(dim=['chain', 'draw']).values
sigma_posterior = trace_2level.posterior['sigma_theta'].mean(dim=['chain', 'draw']).values

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# δの回復
ax = axes[0]
ax.scatter(delta_true, delta_posterior, c='steelblue', s=80, edgecolors='black', zorder=3)
lims = [min(delta_true.min(), delta_posterior.min()) - 0.3,
        max(delta_true.max(), delta_posterior.max()) + 0.3]
ax.plot(lims, lims, 'k--', alpha=0.5, label='完全回復ライン')
r_delta = np.corrcoef(delta_true, delta_posterior)[0, 1]
ax.set_xlabel('真のδ')
ax.set_ylabel('推定δ')
ax.set_title(f'δ（商品カテゴリ）の回復（r = {r_delta:.4f}）')
ax.legend()
ax.grid(True, alpha=0.3)

# θの回復
ax = axes[1]
ax.scatter(theta_true, theta_posterior, c='coral', s=15, alpha=0.5, edgecolors='none')
lims_t = [min(theta_true.min(), theta_posterior.min()) - 0.3,
          max(theta_true.max(), theta_posterior.max()) + 0.3]
ax.plot(lims_t, lims_t, 'k--', alpha=0.5, label='完全回復ライン')
r_theta = np.corrcoef(theta_true, theta_posterior)[0, 1]
ax.set_xlabel('真のθ')
ax.set_ylabel('推定θ')
ax.set_title(f'θ（顧客ロイヤルティ）の回復（r = {r_theta:.4f}）')
ax.legend()
ax.grid(True, alpha=0.3)

# σ_θの回復
ax = axes[2]
sigma_samples = trace_2level.posterior['sigma_theta'].values.flatten()
ax.hist(sigma_samples, bins=40, density=True, color='seagreen', alpha=0.7, edgecolor='black')
ax.axvline(x=sigma_theta_true, color='red', linestyle='--', linewidth=2,
           label=f'真の値 ({sigma_theta_true:.2f})')
ax.axvline(x=sigma_posterior, color='blue', linestyle='-', linewidth=2,
           label=f'事後平均 ({sigma_posterior:.3f})')
ax.set_xlabel('σ_θ')
ax.set_ylabel('事後密度')
ax.set_title('σ_θ の事後分布')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('t17_fig05_parameter_recovery.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n=== パラメータ回復の結果 ===")
print(f"δの相関: r = {r_delta:.4f}")
print(f"θの相関: r = {r_theta:.4f}")
print(f"σ_θ: 真の値 = {sigma_theta_true:.2f}, 推定値 = {sigma_posterior:.3f}")

# ============================================================
# Code Block 11
# ============================================================

# 3レベルのシミュレーションデータ
np.random.seed(456)

G_est = 5
N_per_store_est = 60
N_est = G_est * N_per_store_est
L_est = 6

sigma_store_true = 0.5
sigma_person_true = 0.7
delta_3lev_true = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])

# 階層的なデータ生成
store_effects_true = np.random.randn(G_est) * sigma_store_true
store_idx_est = np.repeat(np.arange(G_est), N_per_store_est)
person_effects_true = np.random.randn(N_est) * sigma_person_true
theta_3lev_true = store_effects_true[store_idx_est] + person_effects_true

# 購買データ生成
log_lambda_3 = theta_3lev_true[:, None] - delta_3lev_true[None, :]
y_3lev = np.random.poisson(np.exp(log_lambda_3))

# スタックドデータ
person_idx_3 = np.repeat(np.arange(N_est), L_est)
item_idx_3 = np.tile(np.arange(L_est), N_est)
store_idx_3 = store_idx_est[person_idx_3]
y_stacked_3 = y_3lev.flatten()

print(f"データ構造: {G_est}店舗 × {N_per_store_est}顧客/店舗 × {L_est}カテゴリ")
print(f"= {len(y_stacked_3)}観測")

# ============================================================
# Code Block 12
# ============================================================

# 3レベルPyMCモデル
with pm.Model() as poisson_irt_3level:
    # Level-3: 店舗効果
    sigma_store_est = pm.HalfNormal('sigma_store', sigma=1.0)
    store_offset = pm.Normal('store_offset', mu=0, sigma=1, shape=G_est)
    store_effect = pm.Deterministic('store_effect',
                                     sigma_store_est * store_offset)
    
    # Level-2: 店舗内の個人差
    sigma_person_est = pm.HalfNormal('sigma_person', sigma=1.0)
    person_offset = pm.Normal('person_offset', mu=0, sigma=1, shape=N_est)
    person_effect = pm.Deterministic('person_effect',
                                      sigma_person_est * person_offset)
    
    # θ = 店舗効果 + 個人差
    theta_3 = store_effect[store_idx_est] + person_effect
    
    # 項目パラメータ
    delta_3 = pm.Normal('delta', mu=0, sigma=2, shape=L_est)
    
    # 線形予測子
    log_lambda_3est = theta_3[person_idx_3] - delta_3[item_idx_3]
    
    # 尤度
    y_like_3 = pm.Poisson('y', mu=pm.math.exp(log_lambda_3est),
                           observed=y_stacked_3)
    
    # MCMC推定
    trace_3level = pm.sample(2000, tune=1000, cores=2, random_seed=456,
                             target_accept=0.95)

# ============================================================
# Code Block 13
# ============================================================

# 3レベルモデルの結果
store_post = trace_3level.posterior['store_effect'].mean(dim=['chain', 'draw']).values
sigma_store_post = trace_3level.posterior['sigma_store'].mean(dim=['chain', 'draw']).values
sigma_person_post = trace_3level.posterior['sigma_person'].mean(dim=['chain', 'draw']).values
delta_3_post = trace_3level.posterior['delta'].mean(dim=['chain', 'draw']).values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 店舗効果の回復
ax = axes[0]
ax.scatter(store_effects_true, store_post, c='steelblue', s=120,
           edgecolors='black', zorder=3)
for g in range(G_est):
    ax.annotate(f'店舗{g+1}', (store_effects_true[g], store_post[g]),
                textcoords='offset points', xytext=(8, 8), fontsize=11)
lims_s = [min(store_effects_true.min(), store_post.min()) - 0.3,
          max(store_effects_true.max(), store_post.max()) + 0.3]
ax.plot(lims_s, lims_s, 'k--', alpha=0.5)
r_store = np.corrcoef(store_effects_true, store_post)[0, 1]
ax.set_xlabel('真の店舗効果')
ax.set_ylabel('推定店舗効果')
ax.set_title(f'店舗効果の回復（r = {r_store:.4f}）')
ax.grid(True, alpha=0.3)

# 分散パラメータの回復
ax = axes[1]
labels_var = ['σ_store', 'σ_person']
true_vals = [sigma_store_true, sigma_person_true]
est_vals = [sigma_store_post, sigma_person_post]

x_var = np.arange(len(labels_var))
width_var = 0.35
ax.bar(x_var - width_var/2, true_vals, width_var, label='真の値',
       color='steelblue', alpha=0.8)
ax.bar(x_var + width_var/2, est_vals, width_var, label='推定値',
       color='coral', alpha=0.8)
ax.set_xticks(x_var)
ax.set_xticklabels(labels_var, fontsize=14)
ax.set_ylabel('標準偏差')
ax.set_title('分散パラメータの回復')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for i, (t, e) in enumerate(zip(true_vals, est_vals)):
    ax.text(i - width_var/2, t + 0.02, f'{t:.2f}', ha='center', fontsize=11)
    ax.text(i + width_var/2, e + 0.02, f'{e:.3f}', ha='center', fontsize=11)

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('t17_fig06_three_level_recovery.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 14
# ============================================================

# θの推定値による顧客セグメンテーション
np.random.seed(42)
N_seg = 500
L_seg = 8
sigma_seg = 0.8
theta_seg = np.random.randn(N_seg) * sigma_seg
delta_seg = np.array([-1.0, -0.5, 0.0, 0.3, 0.7, 1.0, 1.5, 2.0])
y_seg = np.random.poisson(np.exp(theta_seg[:, None] - delta_seg[None, :]))

# ここではθの真の値を使用（PyMCによる推定はセクション6参照）

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：θの分布とセグメント
ax = axes[0]
q25 = np.percentile(theta_seg, 25)
q75 = np.percentile(theta_seg, 75)

colors_seg = np.where(theta_seg > q75, 'tab:red',
             np.where(theta_seg < q25, 'tab:blue', 'tab:gray'))

ax.hist(theta_seg[theta_seg <= q25], bins=20, alpha=0.7, color='tab:blue',
        label=f'ライト層（θ ≤ {q25:.2f}）', density=True)
ax.hist(theta_seg[(theta_seg > q25) & (theta_seg <= q75)], bins=20, alpha=0.7,
        color='tab:gray', label='ミドル層', density=True)
ax.hist(theta_seg[theta_seg > q75], bins=20, alpha=0.7, color='tab:red',
        label=f'ヘビー層（θ > {q75:.2f}）', density=True)
ax.set_xlabel('顧客ロイヤルティ θ')
ax.set_ylabel('密度')
ax.set_title('θによる顧客セグメンテーション')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 右：セグメント別の購買パターン（IRTの強み）
ax = axes[1]
cat_labels = ['生鮮', 'パン米', '惣菜', '加工', '飲料', '菓子', '酒類', '日用品']

heavy_mask = theta_seg > q75
light_mask = theta_seg <= q25

heavy_mean = y_seg[heavy_mask].mean(axis=0)
light_mean = y_seg[light_mask].mean(axis=0)

x_cat = np.arange(L_seg)
w = 0.35
ax.bar(x_cat - w/2, heavy_mean, w, color='tab:red', alpha=0.8, label='ヘビー層')
ax.bar(x_cat + w/2, light_mean, w, color='tab:blue', alpha=0.8, label='ライト層')
ax.set_xticks(x_cat)
ax.set_xticklabels(cat_labels, fontsize=11)
ax.set_ylabel('平均購買回数')
ax.set_title('セグメント別の購買パターン')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('t17_fig07_segmentation.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 15
# ============================================================

# δによる商品カテゴリの購買されにくさマップ
fig, ax = plt.subplots(figsize=(10, 5))

sorted_idx = np.argsort(delta_seg)
sorted_delta = delta_seg[sorted_idx]
sorted_labels = [cat_labels[i] for i in sorted_idx]

colors_delta = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, L_seg))
bars = ax.barh(range(L_seg), sorted_delta, color=colors_delta, edgecolor='black', alpha=0.8)

ax.set_yticks(range(L_seg))
ax.set_yticklabels(sorted_labels, fontsize=13)
ax.set_xlabel('δ（購買されにくさ）')
ax.set_title('商品カテゴリの購買されにくさランキング')
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3, axis='x')

for i, (bar, val) in enumerate(zip(bars, sorted_delta)):
    ax.text(val + 0.05 if val >= 0 else val - 0.05, i, f'{val:.1f}',
            va='center', ha='left' if val >= 0 else 'right', fontsize=12)

ax.annotate('← 購買されやすい', xy=(sorted_delta.min() - 0.2, -0.8),
            fontsize=12, color='green')
ax.annotate('購買されにくい →', xy=(sorted_delta.max() - 0.5, -0.8),
            fontsize=12, color='red')

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('t17_fig08_delta_ranking.png', dpi=150, bbox_inches='tight')
plt.show()

