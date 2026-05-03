"""
T12: IRTの等化とリンキング：異なるテストの得点を同じ物差しで比較する数理をPythonで実装

記事のPythonコードを統合したファイル。
記事URL: https://bigdata-analytics.jp/analytics/irt-linking-equating/

Author: Hiroyuki Matsumoto (Digital Boy LLC)
Repository: https://github.com/hiro1234567/irt-python
"""

# ============================================================
# Code Block 1
# ============================================================

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

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

# 変換係数
zeta = 0.8
kappa = 0.22

# フォーム2の項目パラメータ
delta_form2 = 1.0
alpha_form2 = 1.5

# フォーム1のメトリックに変換
delta_star = zeta * delta_form2 + kappa
alpha_star = alpha_form2 / zeta

print(f"δ の変換: {delta_form2:.2f} → {delta_star:.2f}")
print(f"α の変換: {alpha_form2:.2f} → {alpha_star:.2f}")

# ============================================================
# Code Block 3
# ============================================================

np.random.seed(42)

n_unique = 10   # 各フォームの固有項目数
n_anchor = 10   # 共通（アンカー）項目数
n_items = n_unique + n_anchor  # フォームあたりの合計項目数
n_persons_1 = 1000  # フォーム1の受験者数
n_persons_2 = 1000  # フォーム2の受験者数

# --- 真のアンカー項目パラメータ（共通のメトリック上） ---
alpha_anchor_true = np.random.uniform(0.8, 2.5, n_anchor)
delta_anchor_true = np.random.uniform(-2.0, 2.0, n_anchor)

# --- フォーム1の固有項目パラメータ ---
alpha_unique1 = np.random.uniform(0.8, 2.5, n_unique)
delta_unique1 = np.random.uniform(-2.0, 2.0, n_unique)

# --- フォーム2の固有項目パラメータ ---
alpha_unique2 = np.random.uniform(0.8, 2.5, n_unique)
delta_unique2 = np.random.uniform(-2.0, 2.0, n_unique)

# --- フォーム1の全項目パラメータ ---
alpha_form1 = np.concatenate([alpha_unique1, alpha_anchor_true])
delta_form1 = np.concatenate([delta_unique1, delta_anchor_true])

# --- フォーム2の全項目パラメータ ---
alpha_form2 = np.concatenate([alpha_unique2, alpha_anchor_true])
delta_form2 = np.concatenate([delta_unique2, delta_anchor_true])

# --- 受験者の能力パラメータ ---
theta_group1 = np.random.normal(0.0, 1.0, n_persons_1)
theta_group2 = np.random.normal(0.3, 1.1, n_persons_2)

print(f"フォーム1：{n_items}項目、{n_persons_1}人")
print(f"フォーム2：{n_items}項目、{n_persons_2}人")
print(f"アンカー項目：{n_anchor}項目")
print(f"グループ1の平均能力：{theta_group1.mean():.3f}")
print(f"グループ2の平均能力：{theta_group2.mean():.3f}")

# ============================================================
# Code Block 4
# ============================================================

def irf_2pl(theta, alpha, delta):
    """2PLモデルの正答確率を計算する"""
    return 1.0 / (1.0 + np.exp(-alpha * (theta - delta)))

def generate_responses(theta, alpha, delta):
    """応答データを生成する（0/1の行列）"""
    n_persons = len(theta)
    n_items = len(alpha)
    # 各受験者×各項目の正答確率を計算
    prob = irf_2pl(theta[:, np.newaxis], alpha[np.newaxis, :], delta[np.newaxis, :])
    # 一様乱数と比較して0/1の応答を生成
    responses = (np.random.rand(n_persons, n_items) < prob).astype(int)
    return responses

# 応答データの生成
resp1 = generate_responses(theta_group1, alpha_form1, delta_form1)
resp2 = generate_responses(theta_group2, alpha_form2, delta_form2)

print(f"フォーム1の応答データ：{resp1.shape}（受験者×項目）")
print(f"フォーム2の応答データ：{resp2.shape}（受験者×項目）")
print(f"フォーム1の平均正答率：{resp1.mean():.3f}")
print(f"フォーム2の平均正答率：{resp2.mean():.3f}")

# ============================================================
# Code Block 5
# ============================================================

# メトリックのずれをシミュレーション
# フォーム2の推定値が「フォーム1とは異なるメトリック」に乗っている状態を再現
# 真の変換係数を zeta_true=0.85, kappa_true=0.25 と設定
zeta_true = 0.85
kappa_true = 0.25

# フォーム1の推定値 ≈ 真値 + わずかな推定誤差
alpha_est1 = alpha_form1 * (1 + np.random.normal(0, 0.03, n_items))
delta_est1 = delta_form1 + np.random.normal(0, 0.05, n_items)

# フォーム2の推定値は「別のメトリック」に乗っている
# 真値を逆変換して別メトリックにし、推定誤差を加える
alpha_est2_raw = alpha_form2 * zeta_true  # α は ζ倍される（逆変換）
delta_est2_raw = (delta_form2 - kappa_true) / zeta_true  # δ は逆変換
alpha_est2 = alpha_est2_raw * (1 + np.random.normal(0, 0.03, n_items))
delta_est2 = delta_est2_raw + np.random.normal(0, 0.05, n_items)

# アンカー項目のインデックス（各フォームの後半10項目）
anchor_idx = np.arange(n_unique, n_items)

print("アンカー項目の推定値（フォーム1のメトリック）：")
print(f"  α平均: {alpha_est1[anchor_idx].mean():.3f},  δ平均: {delta_est1[anchor_idx].mean():.3f}")
print("アンカー項目の推定値（フォーム2のメトリック）：")
print(f"  α平均: {alpha_est2[anchor_idx].mean():.3f},  δ平均: {delta_est2[anchor_idx].mean():.3f}")

# ============================================================
# Code Block 6
# ============================================================

def mean_mean_method(alpha1_anchor, delta1_anchor, alpha2_anchor, delta2_anchor):
    """Mean-mean法で変換係数を推定する
    
    Parameters
    ----------
    alpha1_anchor : array - フォーム1のアンカー項目のα推定値
    delta1_anchor : array - フォーム1のアンカー項目のδ推定値
    alpha2_anchor : array - フォーム2のアンカー項目のα推定値
    delta2_anchor : array - フォーム2のアンカー項目のδ推定値
    
    Returns
    -------
    zeta, kappa : float - 変換係数
    """
    zeta = np.mean(alpha1_anchor) / np.mean(alpha2_anchor)
    kappa = np.mean(delta1_anchor) - zeta * np.mean(delta2_anchor)
    return zeta, kappa

# アンカー項目の推定値を取り出す
alpha1_anc = alpha_est1[anchor_idx]
delta1_anc = delta_est1[anchor_idx]
alpha2_anc = alpha_est2[anchor_idx]
delta2_anc = delta_est2[anchor_idx]

# Mean-mean法の実行
zeta_mm, kappa_mm = mean_mean_method(alpha1_anc, delta1_anc, alpha2_anc, delta2_anc)
print(f"Mean-mean法: ζ = {zeta_mm:.4f}, κ = {kappa_mm:.4f}")
print(f"真の値:       ζ = {zeta_true:.4f}, κ = {kappa_true:.4f}")

# ============================================================
# Code Block 7
# ============================================================

def mean_sigma_method(delta1_anchor, delta2_anchor):
    """Mean-sigma法で変換係数を推定する
    
    Parameters
    ----------
    delta1_anchor : array - フォーム1のアンカー項目のδ推定値
    delta2_anchor : array - フォーム2のアンカー項目のδ推定値
    
    Returns
    -------
    zeta, kappa : float - 変換係数
    """
    zeta = np.std(delta1_anchor, ddof=1) / np.std(delta2_anchor, ddof=1)
    kappa = np.mean(delta1_anchor) - zeta * np.mean(delta2_anchor)
    return zeta, kappa

# Mean-sigma法の実行
zeta_ms, kappa_ms = mean_sigma_method(delta1_anc, delta2_anc)
print(f"Mean-sigma法: ζ = {zeta_ms:.4f}, κ = {kappa_ms:.4f}")
print(f"真の値:        ζ = {zeta_true:.4f}, κ = {kappa_true:.4f}")

# ============================================================
# Code Block 8
# ============================================================

def tcf(theta, alpha, delta):
    """テスト特性関数：能力θに対する期待得点"""
    prob = irf_2pl(theta[:, np.newaxis], alpha[np.newaxis, :], delta[np.newaxis, :])
    return prob.sum(axis=1)

def stocking_lord_loss(params, theta_grid, alpha1_anchor, delta1_anchor,
                       alpha2_anchor, delta2_anchor):
    """Stocking-Lord損失関数
    
    Parameters
    ----------
    params : array [zeta, kappa] - 最適化する変換係数
    theta_grid : array - 能力尺度上の評価点
    alpha1_anchor, delta1_anchor : フォーム1のアンカー項目パラメータ
    alpha2_anchor, delta2_anchor : フォーム2のアンカー項目パラメータ
    
    Returns
    -------
    loss : float - TCFの差の二乗和
    """
    zeta, kappa = params
    
    # フォーム2のパラメータを変換
    alpha2_star = alpha2_anchor / zeta
    delta2_star = zeta * delta2_anchor + kappa
    
    # 両フォームのアンカー項目によるTCFを計算
    tcf1 = tcf(theta_grid, alpha1_anchor, delta1_anchor)
    tcf2_star = tcf(theta_grid, alpha2_star, delta2_star)
    
    # 差の二乗和
    loss = np.sum((tcf1 - tcf2_star) ** 2)
    return loss

# 評価点（-4から4までの50点）
theta_grid = np.linspace(-4, 4, 50)

# 初期値をMean-mean法の結果にする
x0 = [zeta_mm, kappa_mm]

# Stocking-Lord法の最適化
result_sl = optimize.minimize(
    stocking_lord_loss, x0,
    args=(theta_grid, alpha1_anc, delta1_anc, alpha2_anc, delta2_anc),
    method='Nelder-Mead'
)

zeta_sl, kappa_sl = result_sl.x
print(f"Stocking-Lord法: ζ = {zeta_sl:.4f}, κ = {kappa_sl:.4f}")
print(f"真の値:           ζ = {zeta_true:.4f}, κ = {kappa_true:.4f}")
print(f"最適化の収束: {result_sl.success}")
print(f"損失関数の最終値: {result_sl.fun:.6f}")

# ============================================================
# Code Block 9
# ============================================================

def haebara_loss(params, theta_grid, alpha1_anchor, delta1_anchor,
                 alpha2_anchor, delta2_anchor):
    """Haebara損失関数（項目レベルの差の二乗和）"""
    zeta, kappa = params
    
    # フォーム2のパラメータを変換
    alpha2_star = alpha2_anchor / zeta
    delta2_star = zeta * delta2_anchor + kappa
    
    # 各項目の正答確率の差の二乗を、項目×評価点で合算
    loss = 0.0
    for j in range(len(alpha1_anchor)):
        p1 = irf_2pl(theta_grid, alpha1_anchor[j], delta1_anchor[j])
        p2_star = irf_2pl(theta_grid, alpha2_star[j], delta2_star[j])
        loss += np.sum((p1 - p2_star) ** 2)
    
    return loss

# Haebara法の最適化
result_hb = optimize.minimize(
    haebara_loss, x0,
    args=(theta_grid, alpha1_anc, delta1_anc, alpha2_anc, delta2_anc),
    method='Nelder-Mead'
)

zeta_hb, kappa_hb = result_hb.x
print(f"Haebara法:        ζ = {zeta_hb:.4f}, κ = {kappa_hb:.4f}")
print(f"Stocking-Lord法:  ζ = {zeta_sl:.4f}, κ = {kappa_sl:.4f}")
print(f"真の値:            ζ = {zeta_true:.4f}, κ = {kappa_true:.4f}")

# ============================================================
# Code Block 10
# ============================================================

methods = ['Mean-mean', 'Mean-sigma', 'Stocking-Lord', 'Haebara', '真の値']
zetas = [zeta_mm, zeta_ms, zeta_sl, zeta_hb, zeta_true]
kappas = [kappa_mm, kappa_ms, kappa_sl, kappa_hb, kappa_true]

print(f"{'手法':<16} {'ζ':>8} {'κ':>8}")
print("-" * 34)
for m, z, k in zip(methods, zetas, kappas):
    print(f"{m:<16} {z:>8.4f} {k:>8.4f}")

# ============================================================
# Code Block 11
# ============================================================

theta_plot = np.linspace(-4, 4, 200)

# 等化前のTCF
tcf1_values = tcf(theta_plot, alpha1_anc, delta1_anc)
tcf2_values = tcf(theta_plot, alpha2_anc, delta2_anc)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta_plot, tcf1_values, 'b-', linewidth=2, label='フォーム1')
ax.plot(theta_plot, tcf2_values, 'r--', linewidth=2, label='フォーム2（変換前）')
ax.set_xlabel('θ')
ax.set_ylabel('期待得点')
ax.set_title('等化前：アンカー項目のTCF')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig01_tcf_before_equating.png', bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 12
# ============================================================

# フォーム2のアンカー項目パラメータを変換
alpha2_anc_star = alpha2_anc / zeta_sl
delta2_anc_star = zeta_sl * delta2_anc + kappa_sl

# 等化後のTCF
tcf2_star_values = tcf(theta_plot, alpha2_anc_star, delta2_anc_star)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta_plot, tcf1_values, 'b-', linewidth=2, label='フォーム1')
ax.plot(theta_plot, tcf2_star_values, 'r--', linewidth=2, label='フォーム2（等化後）')
ax.set_xlabel('θ')
ax.set_ylabel('期待得点')
ax.set_title('等化後：アンカー項目のTCF')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig02_tcf_after_equating.png', bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 13
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左：等化前
axes[0].plot(theta_plot, tcf1_values, 'b-', linewidth=2, label='フォーム1')
axes[0].plot(theta_plot, tcf2_values, 'r--', linewidth=2, label='フォーム2（変換前）')
axes[0].set_xlabel('θ')
axes[0].set_ylabel('期待得点')
axes[0].set_title('等化前')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右：等化後
axes[1].plot(theta_plot, tcf1_values, 'b-', linewidth=2, label='フォーム1')
axes[1].plot(theta_plot, tcf2_star_values, 'r--', linewidth=2, label='フォーム2（等化後）')
axes[1].set_xlabel('θ')
axes[1].set_ylabel('期待得点')
axes[1].set_title('等化後（Stocking-Lord法）')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig03_tcf_before_after.png', bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 14
# ============================================================

# フォーム2受験者の能力推定値を変換
# （実務ではMMLEやEAP推定で得た値を変換する）
theta2_on_form2_metric = (theta_group2 - kappa_true) / zeta_true  # 「フォーム2メトリック」上の値
theta2_equated = zeta_sl * theta2_on_form2_metric + kappa_sl  # フォーム1メトリックに変換

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左：変換前の分布比較
axes[0].hist(theta_group1, bins=40, alpha=0.5, density=True, label='フォーム1受験者', color='blue')
axes[0].hist(theta2_on_form2_metric, bins=40, alpha=0.5, density=True,
             label='フォーム2受験者（変換前）', color='red')
axes[0].set_xlabel('θ')
axes[0].set_ylabel('密度')
axes[0].set_title('変換前：メトリックが異なる')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右：変換後の分布比較
axes[1].hist(theta_group1, bins=40, alpha=0.5, density=True, label='フォーム1受験者', color='blue')
axes[1].hist(theta2_equated, bins=40, alpha=0.5, density=True,
             label='フォーム2受験者（等化後）', color='red')
axes[1].set_xlabel('θ')
axes[1].set_ylabel('密度')
axes[1].set_title('等化後：同じメトリック')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig04_theta_distributions.png', bbox_inches='tight')
plt.show()

print(f"フォーム1受験者：平均 = {theta_group1.mean():.3f}, SD = {theta_group1.std():.3f}")
print(f"フォーム2受験者（変換前）：平均 = {theta2_on_form2_metric.mean():.3f}, SD = {theta2_on_form2_metric.std():.3f}")
print(f"フォーム2受験者（等化後）：平均 = {theta2_equated.mean():.3f}, SD = {theta2_equated.std():.3f}")

# ============================================================
# Code Block 15
# ============================================================

# 固定項目パラメータ法のコンセプトコード
# （完全な推定には反復最適化が必要。ここではロジックを示す）

def fixed_item_concept(alpha1_anchor, delta1_anchor, alpha2_all, delta2_all, anchor_idx):
    """固定項目パラメータ法のコンセプト
    
    アンカー項目のパラメータをフォーム1の値に固定し、
    フォーム2の固有項目だけを再推定する。
    """
    alpha2_fixed = alpha2_all.copy()
    delta2_fixed = delta2_all.copy()
    
    # アンカー項目はフォーム1の推定値で上書き（＝固定）
    alpha2_fixed[anchor_idx] = alpha1_anchor
    delta2_fixed[anchor_idx] = delta1_anchor
    
    # 固有項目は再推定が必要（ここでは省略）
    # 実務ではmirtのSTART/FIXEDオプションでこの操作を行う
    
    return alpha2_fixed, delta2_fixed

print("固定項目パラメータ法のポイント：")
print("- アンカー項目をフォーム1の値に固定 → 自動的にメトリックが揃う")
print("- 変換係数ζ, κを明示的に計算する必要がない")
print("- Rではmirtパッケージの START/FIXED オプションで実装可能")

# ============================================================
# Code Block 16
# ============================================================

# 併行キャリブレーション用のデータ構成を可視化

# データ行列の構造を示す
print("併行キャリブレーションのデータ構造：")
print()
print(f"{'':>12} {'F1固有(10項目)':>18} {'共通(10項目)':>16} {'F2固有(10項目)':>18}")
print("-" * 68)
print(f"{'グループ1':>12} {'回答データ':>18} {'回答データ':>16} {'欠測(NA)':>18}")
print(f"{'(1000人)':>12} {'':>18} {'':>16} {'':>18}")
print(f"{'グループ2':>12} {'欠測(NA)':>18} {'回答データ':>16} {'回答データ':>18}")
print(f"{'(1000人)':>12} {'':>18} {'':>16} {'':>18}")
print()
print("全30項目を同時に推定 → 自動的に全パラメータが同一メトリック上に乗る")

