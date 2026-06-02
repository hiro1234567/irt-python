"""
T16: Multilevel IRTの理論：GLMMフレームワークでIRTを再解釈する

記事のPythonコードを統合したファイル。
記事URL: https://bigdata-analytics.jp/analytics/irt-multilevel-theory/

Author: Hiroyuki Matsumoto (Digital Boy LLC)
Repository: https://github.com/hiro1234567/irt-python

実行方法:
  python t16_multilevel_theory.py
    → CLIで一気に実行。グラフはウィンドウで順次表示される
  MPLBACKEND=Agg python t16_multilevel_theory.py
    → グラフ表示せず数値だけ確認
  Jupyterで Code Block を1つずつコピペして対話的に試すのも可
"""

# ============================================================
# Code Block 1
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import expit  # ロジスティック関数
import matplotlib.pyplot as plt
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

np.random.seed(42)

# テスト設定
N = 500   # 受験者数
L = 5     # 項目数
delta = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # 項目位置
alpha = 1.0  # 識別力（Raschモデル）

# 受験者の能力をシミュレーション
theta = np.random.randn(N)

# 応答データの生成（Raschモデル）
prob = expit(alpha * (theta[:, None] - delta[None, :]))
responses = (np.random.rand(N, L) < prob).astype(int)

# wide形式のDataFrame
wide_df = pd.DataFrame(
    responses,
    columns=[f'item{j+1}' for j in range(L)]
)
wide_df.insert(0, 'person', np.arange(1, N + 1))

print("=== Wide形式（受験者×項目の行列）===")
print(wide_df.head(10))
print(f"サイズ: {wide_df.shape[0]}行 × {wide_df.shape[1]}列")

# ============================================================
# Code Block 3
# ============================================================

# long形式（スタックドデータ）への変換
stacked = wide_df.melt(
    id_vars=['person'],
    var_name='item',
    value_name='x'
)
stacked = stacked.sort_values(['person', 'item']).reset_index(drop=True)

print("\n=== Long形式（スタックドデータ）===")
print(stacked.head(15))
print(f"サイズ: {stacked.shape[0]}行 × {stacked.shape[1]}列")
print(f"\n500人 × 5項目 = {500 * 5}行のlong形式に展開されました")

# ============================================================
# Code Block 4
# ============================================================

from scipy.optimize import minimize

def rasch_neg_log_marginal(params, X, items, n_items, n_quad=21):
    """
    Raschモデルの周辺対数尤度（負値）。
    ガウス求積法でθを積分消去する。
    
    params: [delta_1, ..., delta_L, log_sigma_theta]
    """
    deltas = params[:n_items]
    log_sigma = params[n_items]
    sigma = np.exp(log_sigma)
    
    # ガウス求積点と重み
    quad_points, quad_weights = np.polynomial.hermite.hermgauss(n_quad)
    quad_points = quad_points * np.sqrt(2) * sigma
    quad_weights = quad_weights / np.sqrt(np.pi)
    
    persons = np.unique(items[:, 0])
    log_lik = 0.0
    
    for person in persons:
        mask = items[:, 0] == person
        x_p = X[mask]
        item_idx = items[mask, 1]
        delta_p = deltas[item_idx]
        
        # 各求積点でのログ尤度
        log_probs = np.zeros(n_quad)
        for k in range(n_quad):
            theta_k = quad_points[k]
            logit_p = theta_k - delta_p
            log_p = x_p * logit_p - np.log(1 + np.exp(logit_p))
            log_probs[k] = np.sum(log_p)
        
        # log-sum-exp for numerical stability
        max_lp = np.max(log_probs)
        log_marginal_p = max_lp + np.log(
            np.sum(quad_weights * np.exp(log_probs - max_lp))
        )
        log_lik += log_marginal_p
    
    return -log_lik

# 推定用データの準備
item_labels = stacked['item'].values
item_map = {f'item{j+1}': j for j in range(L)}
item_indices = np.array([item_map[lbl] for lbl in item_labels])
person_indices = stacked['person'].values - 1

items_array = np.column_stack([person_indices, item_indices])
X_resp = stacked['x'].values.astype(float)

# 初期値
init_params = np.zeros(L + 1)  # delta_1..5 + log_sigma
init_params[-1] = 0.0  # log(sigma) = 0 → sigma = 1

print("Raschモデル（周辺最尤推定）を推定中...")
result_rasch = minimize(
    rasch_neg_log_marginal,
    init_params,
    args=(X_resp, items_array, L),
    method='L-BFGS-B',
    options={'maxiter': 500, 'disp': False}
)

delta_hat = result_rasch.x[:L]
sigma_hat = np.exp(result_rasch.x[L])

print(f"\n=== Raschモデルの推定結果 ===")
print(f"項目位置 δ の推定値:")
for j in range(L):
    print(f"  item{j+1}: δ̂ = {delta_hat[j]:+.4f}  (真値: {delta[j]:+.1f})")
print(f"\nθの標準偏差 σ̂_θ = {sigma_hat:.4f}")
print(f"対数尤度 = {-result_rasch.fun:.2f}")

# ============================================================
# Code Block 5
# ============================================================

def glmm_neg_log_marginal(params, X, person_ids, item_dummies, n_quad=21):
    """
    GLMMの周辺対数尤度（負値）。
    固定効果: 切片(zeta_00) + 項目ダミー(zeta_q0)
    ランダム効果: 受験者の切片(theta'_0i)
    """
    n_fixed = item_dummies.shape[1] + 1  # 切片 + L-1個の項目ダミー
    zeta = params[:n_fixed]  # [zeta_00, zeta_10, zeta_20, ..., zeta_(L-1)0]
    log_sigma = params[n_fixed]
    sigma = np.exp(log_sigma)
    
    # 求積点と重み
    quad_points, quad_weights = np.polynomial.hermite.hermgauss(n_quad)
    quad_points = quad_points * np.sqrt(2) * sigma
    quad_weights = quad_weights / np.sqrt(np.pi)
    
    persons = np.unique(person_ids)
    log_lik = 0.0
    
    for person in persons:
        mask = person_ids == person
        x_p = X[mask]
        dummies_p = item_dummies[mask]
        
        # 固定効果の線形予測子
        linear_fixed = zeta[0] + dummies_p @ zeta[1:]
        
        log_probs = np.zeros(n_quad)
        for k in range(n_quad):
            theta_k = quad_points[k]
            logit_p = linear_fixed + theta_k
            log_p = x_p * logit_p - np.log(1 + np.exp(logit_p))
            log_probs[k] = np.sum(log_p)
        
        max_lp = np.max(log_probs)
        log_marginal_p = max_lp + np.log(
            np.sum(quad_weights * np.exp(log_probs - max_lp))
        )
        log_lik += log_marginal_p
    
    return -log_lik

# GLMMの設計行列（項目ダミー、item5が基準）
item_dummies_matrix = np.zeros((len(stacked), L - 1))
for q in range(L - 1):
    item_dummies_matrix[:, q] = (item_indices == q).astype(float)

# 初期値
init_glmm = np.zeros(L + 1)  # zeta_00, zeta_10..40, log_sigma

print("GLMMを推定中...")
result_glmm = minimize(
    glmm_neg_log_marginal,
    init_glmm,
    args=(X_resp, person_indices, item_dummies_matrix),
    method='L-BFGS-B',
    options={'maxiter': 500, 'disp': False}
)

zeta_hat = result_glmm.x[:L]
sigma_hat_glmm = np.exp(result_glmm.x[L])

# GLMMパラメータ → IRTパラメータへの変換
zeta_00 = zeta_hat[0]
zeta_q0 = zeta_hat[1:]  # item1〜item4の効果

# δ_j = -(zeta_q0 + zeta_00) for j=1..4
# δ_5 = -zeta_00 (基準項目)
delta_from_glmm = np.zeros(L)
for q in range(L - 1):
    delta_from_glmm[q] = -(zeta_q0[q] + zeta_00)
delta_from_glmm[L - 1] = -zeta_00

print(f"\n=== GLMM → IRT変換結果 ===")
print(f"全体平均 ζ̂_00 = {zeta_00:.4f}")
print(f"θの標準偏差 σ̂_θ = {sigma_hat_glmm:.4f}")
print(f"\n項目位置の比較:")
print(f"{'項目':<8} {'Rasch直接推定':>14} {'GLMM→IRT変換':>14} {'真値':>8} {'差':>8}")
print("-" * 55)
for j in range(L):
    diff = delta_hat[j] - delta_from_glmm[j]
    print(f"item{j+1:<4} {delta_hat[j]:>+14.4f} {delta_from_glmm[j]:>+14.4f} {delta[j]:>+8.1f} {diff:>+8.4f}")

# ============================================================
# Code Block 6
# ============================================================

# DIF付きシミュレーションデータの生成
np.random.seed(123)

N_dif = 1000
L_dif = 5
delta_ref = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # 参照グループの項目位置

# グループ割り当て（0: Reference, 1: Focal, 各500人）
group = np.repeat([0, 1], N_dif // 2)
theta_dif = np.random.randn(N_dif)

# DIFの設定: item3にのみ大きなDIFを仕込む
# Focalグループではitem3が0.8だけ難しくなる
dif_effect = np.zeros((N_dif, L_dif))
dif_effect[group == 1, 2] = 0.8  # item3, Focalグループ

delta_effective = delta_ref[None, :] + dif_effect

# 応答データ生成
prob_dif = expit(theta_dif[:, None] - delta_effective)
responses_dif = (np.random.rand(N_dif, L_dif) < prob_dif).astype(int)

# DataFrameに整形
dif_df = pd.DataFrame(responses_dif, columns=[f'item{j+1}' for j in range(L_dif)])
dif_df['person'] = np.arange(1, N_dif + 1)
dif_df['group'] = group

# long形式に変換
dif_long = dif_df.melt(
    id_vars=['person', 'group'],
    var_name='item',
    value_name='x'
)
dif_long = dif_long.sort_values(['person', 'item']).reset_index(drop=True)

# 各グループ×項目の正答率を確認
pivot = dif_long.groupby(['group', 'item'])['x'].mean().unstack()
print("=== グループ別の正答率 ===")
print(pivot.round(3))
print("\n差（Reference - Focal）:")
print((pivot.loc[0] - pivot.loc[1]).round(3))

# ============================================================
# Code Block 7
# ============================================================

def glmm_dif_test(X_resp, person_ids, item_indices, group_ids,
                  n_items, n_quad=15):
    """
    DIF検出のためのGLMM推定。
    Model 0（DIFなし）とModel 1（DIFあり）を比較する。
    """
    persons = np.unique(person_ids)
    
    def neg_log_lik_no_dif(params):
        """DIFなしモデル: 固定効果 = 切片 + item"""
        zeta_00 = params[0]
        zeta_q = params[1:n_items]  # L-1個の項目効果
        sigma = np.exp(params[n_items])
        
        quad_pts, quad_wts = np.polynomial.hermite.hermgauss(n_quad)
        quad_pts = quad_pts * np.sqrt(2) * sigma
        quad_wts = quad_wts / np.sqrt(np.pi)
        
        ll = 0.0
        for p in persons:
            mask = person_ids == p
            x_p = X_resp[mask]
            items_p = item_indices[mask]
            
            linear = np.zeros(mask.sum())
            linear += zeta_00
            for q in range(n_items - 1):
                linear[items_p == q] += zeta_q[q]
            
            lp = np.zeros(n_quad)
            for k in range(n_quad):
                logit = linear + quad_pts[k]
                lp[k] = np.sum(x_p * logit - np.log(1 + np.exp(logit)))
            
            mx = np.max(lp)
            ll += mx + np.log(np.sum(quad_wts * np.exp(lp - mx)))
        
        return -ll
    
    def neg_log_lik_dif(params):
        """DIFモデル: 固定効果 = 切片 + item + group + item×group"""
        n_f = 1 + (n_items - 1) + 1 + (n_items - 1)
        zeta_00 = params[0]
        zeta_q = params[1:n_items]
        zeta_01 = params[n_items]
        zeta_q1 = params[n_items + 1:2 * n_items]
        sigma = np.exp(params[2 * n_items])
        
        quad_pts, quad_wts = np.polynomial.hermite.hermgauss(n_quad)
        quad_pts = quad_pts * np.sqrt(2) * sigma
        quad_wts = quad_wts / np.sqrt(np.pi)
        
        ll = 0.0
        for p in persons:
            mask = person_ids == p
            x_p = X_resp[mask]
            items_p = item_indices[mask]
            g = group_ids[mask][0]
            
            linear = np.zeros(mask.sum())
            linear += zeta_00 + zeta_01 * g
            for q in range(n_items - 1):
                item_mask = items_p == q
                linear[item_mask] += zeta_q[q] + zeta_q1[q] * g
            
            lp = np.zeros(n_quad)
            for k in range(n_quad):
                logit = linear + quad_pts[k]
                lp[k] = np.sum(x_p * logit - np.log(1 + np.exp(logit)))
            
            mx = np.max(lp)
            ll += mx + np.log(np.sum(quad_wts * np.exp(lp - mx)))
        
        return -ll
    
    # Model 0の推定
    init0 = np.zeros(n_items + 1)
    res0 = minimize(neg_log_lik_no_dif, init0, method='L-BFGS-B',
                    options={'maxiter': 300})
    
    # Model 1の推定
    init1 = np.zeros(2 * n_items + 1)
    res1 = minimize(neg_log_lik_dif, init1, method='L-BFGS-B',
                    options={'maxiter': 300})
    
    return res0, res1

# DIF用のインデックスを作成
dif_item_labels = dif_long['item'].values
dif_item_map = {f'item{j+1}': j for j in range(L_dif)}
dif_item_idx = np.array([dif_item_map[lbl] for lbl in dif_item_labels])
dif_person_idx = dif_long['person'].values - 1
dif_group_idx = dif_long['group'].values
dif_X = dif_long['x'].values.astype(float)

print("DIF分析を実行中（少々時間がかかります）...")
res_null, res_dif = glmm_dif_test(
    dif_X, dif_person_idx, dif_item_idx, dif_group_idx, L_dif
)

# 尤度比検定
G2_null = 2 * res_null.fun
G2_dif = 2 * res_dif.fun
delta_G2 = G2_null - G2_dif
df_diff = L_dif  # group効果(1) + item×group交互作用(L-1)
p_value = 1 - stats.chi2.cdf(delta_G2, df_diff)

print(f"\n=== DIF検出結果（尤度比検定）===")
print(f"Model 0（DIFなし）: -2lnL = {G2_null:.2f}")
print(f"Model 1（DIF あり）: -2lnL = {G2_dif:.2f}")
print(f"ΔG² = {delta_G2:.2f}, df = {df_diff}, p = {p_value:.6f}")

if p_value < 0.05:
    print("→ 有意（p < 0.05）：DIFの存在が示唆されます")
else:
    print("→ 非有意：DIFは検出されませんでした")

# DIFモデルの交互作用項を確認
dif_params = res_dif.x
zeta_q1_hat = dif_params[L_dif + 1:2 * L_dif]
print(f"\n=== 項目×グループ交互作用の推定値 ===")
print(f"（値が大きい項目にDIFが存在）")
for q in range(L_dif - 1):
    marker = " ← DIF仕込み" if q == 2 else ""
    print(f"  item{q+1}×group: ζ̂_q1 = {zeta_q1_hat[q]:+.4f}{marker}")
print(f"  item{L_dif}（基準項目）: ζ̂_q1 = 0（定義）")

# ============================================================
# Code Block 8
# ============================================================

# 共変量付きシミュレーション
np.random.seed(456)

N_cov = 500
L_cov = 5
delta_cov = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

# 共変量: 教育水準（13〜21年、平均17年）
edlevel = np.random.randint(13, 22, size=N_cov).astype(float)
edlevel_centered = edlevel - edlevel.mean()

# 真のパラメータ
zeta_01_true = 0.10  # 教育水準の効果（1年あたり）
sigma_theta_true = 0.8

# θを教育水準と残差に分解
theta_residual = np.random.randn(N_cov) * sigma_theta_true
theta_full = zeta_01_true * edlevel_centered + theta_residual

# 応答データ生成
prob_cov = expit(theta_full[:, None] - delta_cov[None, :])
responses_cov = (np.random.rand(N_cov, L_cov) < prob_cov).astype(int)

# 合計点の計算
total_scores = responses_cov.sum(axis=1)

# 同じ合計点をとった受験者で、教育水準による推定能力の違いを確認
target_score = 3  # 合計点3の受験者に注目
mask_score3 = total_scores == target_score

if mask_score3.sum() > 10:
    ed_levels_score3 = edlevel[mask_score3]
    theta_full_score3 = theta_full[mask_score3]
    theta_resid_score3 = theta_residual[mask_score3]
    
    # 教育水準が高いグループと低いグループに分割
    median_ed = np.median(ed_levels_score3)
    high_ed = theta_full_score3[ed_levels_score3 > median_ed]
    low_ed = theta_full_score3[ed_levels_score3 <= median_ed]
    
    print(f"=== 合計点 {target_score} の受験者（{mask_score3.sum()}人）===")
    print(f"\n教育水準が高いグループ（> {median_ed}年, {len(high_ed)}人）:")
    print(f"  真のθ平均 = {high_ed.mean():.4f}")
    print(f"\n教育水準が低いグループ（≤ {median_ed}年, {len(low_ed)}人）:")
    print(f"  真のθ平均 = {low_ed.mean():.4f}")
    print(f"\n差 = {high_ed.mean() - low_ed.mean():.4f}")
    print("→ 同じ合計点でも、教育水準が高い人ほど推定能力が高くなります")

