"""
T15: 項目機能差異（DIF）の検出：テストの公平性をPythonで統計的に検証する

記事のPythonコードを統合したファイル。
記事URL: https://bigdata-analytics.jp/analytics/irt-dif-detection/

Author: Hiroyuki Matsumoto (Digital Boy LLC)
Repository: https://github.com/hiro1234567/irt-python
"""

# ============================================================
# Code Block 1
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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

def irf(theta, alpha, delta):
    """2PLモデルの項目反応関数"""
    return 1.0 / (1.0 + np.exp(-alpha * (theta - delta)))

theta = np.linspace(-4, 4, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 一様DIF：αは同じ、δが異なる
ax = axes[0]
ax.plot(theta, irf(theta, 1.5, -0.5), 'b-', lw=2.5, label='基準群（Reference）')
ax.plot(theta, irf(theta, 1.5, 0.8), 'r--', lw=2.5, label='焦点群（Focal）')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$p(x=1)$')
ax.set_title('一様DIF（uniform）')
ax.legend(loc='lower right')
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3)

# 非一様DIF：αもδも異なる → IRFが交差
ax = axes[1]
ax.plot(theta, irf(theta, 2.0, 0.0), 'b-', lw=2.5, label='基準群（Reference）')
ax.plot(theta, irf(theta, 0.7, 0.0), 'r--', lw=2.5, label='焦点群（Focal）')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$p(x=1)$')
ax.set_title('非一様DIF（nonuniform）')
ax.legend(loc='lower right')
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('irt_dif_01_types.png', bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 3
# ============================================================

np.random.seed(42)

n_ref = 1000   # 基準群の人数
n_foc = 1000   # 焦点群の人数
n_items = 10   # 項目数

# 能力パラメータ：両群とも標準正規分布
theta_ref = np.random.randn(n_ref)
theta_foc = np.random.randn(n_foc)

# 項目パラメータ（基準群）
alpha_ref = np.array([1.2, 0.8, 1.5, 1.0, 1.3, 0.9, 2.0, 1.1, 1.4, 0.7])
delta_ref = np.array([-1.0, -0.5, 0.0, 0.5, -0.3, 0.8, 0.0, -0.7, 1.0, -1.5])

# 項目パラメータ（焦点群）：項目3と項目7のみ異なる
alpha_foc = alpha_ref.copy()
delta_foc = delta_ref.copy()
delta_foc[2] = 1.2     # 項目3：δが1.2に（一様DIF）
alpha_foc[6] = 0.6     # 項目7：αが0.6に（非一様DIF）

# 応答データの生成
def generate_responses(theta, alpha, delta):
    """2PLモデルに従う二値応答データを生成する"""
    n_persons = len(theta)
    n_items = len(alpha)
    prob = irf(theta[:, np.newaxis], alpha[np.newaxis, :], delta[np.newaxis, :])
    responses = (np.random.rand(n_persons, n_items) < prob).astype(int)
    return responses

resp_ref = generate_responses(theta_ref, alpha_ref, delta_ref)
resp_foc = generate_responses(theta_foc, alpha_foc, delta_foc)

# 全データの結合
responses = np.vstack([resp_ref, resp_foc])
group = np.array([0] * n_ref + [1] * n_foc)  # 0=基準群, 1=焦点群

print(f'基準群: {n_ref}人, 焦点群: {n_foc}人')
print(f'項目数: {n_items}')
print(f'全体の応答行列: {responses.shape}')
print(f'基準群の平均正答率: {resp_ref.mean():.3f}')
print(f'焦点群の平均正答率: {resp_foc.mean():.3f}')

# ============================================================
# Code Block 4
# ============================================================

def mantel_haenszel(responses, group, item_idx):
    """
    1つの項目に対するMantel-Haenszel検定を実行する。
    
    Parameters
    ----------
    responses : ndarray, shape (n_persons, n_items)
    group : ndarray, shape (n_persons,)  0=基準群, 1=焦点群
    item_idx : int  検定対象の項目インデックス
    
    Returns
    -------
    dict : MH χ², p値, 共通オッズ比, log odds ratio, ETS Delta, ETSカテゴリ
    """
    n_items = responses.shape[1]
    
    # 合計得点を計算（対象項目を除外して計算する場合もあるが、ここでは含める）
    X = responses.sum(axis=1)
    
    # 対象項目の応答
    item_resp = responses[:, item_idx]
    
    # 合計得点の範囲（0と最大値を除く）
    score_min = 1
    score_max = n_items - 1
    
    sum_A_E = 0.0    # Σ(A_t - E(A_t))
    sum_var = 0.0    # Σvar(A_t)
    sum_AD_n = 0.0   # Σ(A_t * D_t / n_t)
    sum_BC_n = 0.0   # Σ(B_t * C_t / n_t)
    
    for t in range(score_min, score_max + 1):
        # 合計得点がtの人を抽出
        mask = (X == t)
        if mask.sum() == 0:
            continue
        
        ref_mask = mask & (group == 0)
        foc_mask = mask & (group == 1)
        
        n_Rt = ref_mask.sum()
        n_Ft = foc_mask.sum()
        n_t = n_Rt + n_Ft
        
        if n_t <= 1 or n_Rt == 0 or n_Ft == 0:
            continue
        
        # 2×2分割表のセル度数
        A_t = (item_resp[ref_mask] == 1).sum()  # 基準群・正答
        B_t = (item_resp[ref_mask] == 0).sum()  # 基準群・誤答
        C_t = (item_resp[foc_mask] == 1).sum()  # 焦点群・正答
        D_t = (item_resp[foc_mask] == 0).sum()  # 焦点群・誤答
        
        n_1t = A_t + C_t  # 正答合計
        n_0t = B_t + D_t  # 誤答合計
        
        # 期待値と分散
        E_At = n_Rt * n_1t / n_t
        var_At = n_Rt * n_Ft * n_1t * n_0t / (n_t**2 * (n_t - 1))
        
        sum_A_E += (A_t - E_At)
        sum_var += var_At
        sum_AD_n += A_t * D_t / n_t
        sum_BC_n += B_t * C_t / n_t
    
    # MH χ²（Yatesの連続性補正付き）
    if sum_var > 0:
        mh_chi2 = (abs(sum_A_E) - 0.5)**2 / sum_var
    else:
        mh_chi2 = 0.0
    
    # p値（χ²分布, df=1）
    p_value = 1.0 - stats.chi2.cdf(mh_chi2, df=1)
    
    # 共通オッズ比
    if sum_BC_n > 0:
        alpha_MH = sum_AD_n / sum_BC_n
    else:
        alpha_MH = np.inf
    
    # 対数オッズ比
    if alpha_MH > 0 and np.isfinite(alpha_MH):
        beta_MH = np.log(alpha_MH)
    else:
        beta_MH = np.inf
    
    # ETS Delta尺度
    D_MH = -2.35 * beta_MH if np.isfinite(beta_MH) else np.inf
    
    # ETSカテゴリ
    abs_D = abs(D_MH) if np.isfinite(D_MH) else np.inf
    if p_value >= 0.05 or abs_D < 1.0:
        ets_category = 'A'
    elif abs_D <= 1.5:
        ets_category = 'B'
    else:
        ets_category = 'C'
    
    return {
        'MH_chi2': mh_chi2,
        'p_value': p_value,
        'alpha_MH': alpha_MH,
        'beta_MH': beta_MH,
        'D_MH': D_MH,
        'ETS_category': ets_category,
    }

# ============================================================
# Code Block 5
# ============================================================

print(f"{'項目':>4} {'MH χ²':>10} {'p値':>10} {'α_MH':>8} {'β_MH':>8} {'D_MH':>8} {'ETS':>4} {'真のDIF':>10}")
print('-' * 75)

true_dif = {3: '一様DIF', 7: '非一様DIF'}

for j in range(n_items):
    result = mantel_haenszel(responses, group, j)
    dif_label = true_dif.get(j + 1, 'なし')
    sig = '***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else ''
    print(f"{j+1:>4} {result['MH_chi2']:>10.3f} {result['p_value']:>10.4f}{sig:3s} "
          f"{result['alpha_MH']:>8.4f} {result['beta_MH']:>8.4f} {result['D_MH']:>8.4f} "
          f"{result['ETS_category']:>4} {dif_label:>10}")

# ============================================================
# Code Block 6
# ============================================================

def logistic_regression_dif(responses, group, item_idx):
    """
    1つの項目に対するロジスティック回帰DIF検定を実行する。
    """
    n = responses.shape[0]
    X = responses.sum(axis=1)  # 合計得点
    y = responses[:, item_idx]  # 対象項目の応答
    g = group.copy()
    
    # フルモデル：y ~ X + group + X*group
    # 縮約モデル1：y ~ X + group（一様DIFモデル）
    # 縮約モデル2：y ~ X（DIFなしモデル）
    
    def neg_log_likelihood(beta, X_design, y):
        """ロジスティック回帰の負の対数尤度"""
        z = X_design @ beta
        # 数値安定性のためクリッピング
        z = np.clip(z, -500, 500)
        ll = np.sum(y * z - np.log(1.0 + np.exp(z)))
        return -ll
    
    # デザイン行列の構築
    ones = np.ones(n)
    interaction = X * g
    
    X_full = np.column_stack([ones, X, g, interaction])   # フルモデル
    X_red1 = np.column_stack([ones, X, g])                 # 縮約モデル1
    X_red2 = np.column_stack([ones, X])                    # 縮約モデル2
    X_null = ones.reshape(-1, 1)                           # 切片のみモデル
    
    def fit_model(X_design, y):
        """モデルをフィットして-2lnLとNagelkerke R²を返す"""
        k = X_design.shape[1]
        beta0 = np.zeros(k)
        result = minimize(neg_log_likelihood, beta0, args=(X_design, y),
                         method='BFGS')
        neg_ll = result.fun
        deviance = 2 * neg_ll  # -2 ln L
        return deviance, result.x
    
    # 各モデルをフィット
    dev_full, beta_full = fit_model(X_full, y)
    dev_red1, beta_red1 = fit_model(X_red1, y)
    dev_red2, beta_red2 = fit_model(X_red2, y)
    dev_null, _ = fit_model(X_null, y)
    
    # 非一様DIFの検定：フルモデル vs 縮約モデル1
    dG2_nonuniform = dev_red1 - dev_full
    p_nonuniform = 1 - stats.chi2.cdf(dG2_nonuniform, df=1)
    
    # 一様DIFの検定：縮約モデル1 vs 縮約モデル2
    dG2_uniform = dev_red2 - dev_red1
    p_uniform = 1 - stats.chi2.cdf(dG2_uniform, df=1)
    
    # 全体のDIF検定：フルモデル vs 縮約モデル2（df=2）
    dG2_total = dev_red2 - dev_full
    p_total = 1 - stats.chi2.cdf(dG2_total, df=2)
    
    # Nagelkerke R²の計算
    def nagelkerke_r2(deviance_model, deviance_null, n):
        cox_snell = 1 - np.exp((deviance_model - deviance_null) / n)
        max_cox_snell = 1 - np.exp(-deviance_null / n)
        if max_cox_snell > 0:
            return cox_snell / max_cox_snell
        return 0.0
    
    r2_full = nagelkerke_r2(dev_full, dev_null, n)
    r2_red1 = nagelkerke_r2(dev_red1, dev_null, n)
    r2_red2 = nagelkerke_r2(dev_red2, dev_null, n)
    
    # ΔR²
    dr2_nonuniform = r2_full - r2_red1
    dr2_uniform = r2_red1 - r2_red2
    dr2_total = r2_full - r2_red2
    
    # Jodoin & Gierl基準
    def jg_category(dr2):
        if dr2 >= 0.070:
            return 'C'
        elif dr2 >= 0.035:
            return 'B'
        else:
            return 'A'
    
    return {
        'dG2_nonuniform': dG2_nonuniform,
        'p_nonuniform': p_nonuniform,
        'dG2_uniform': dG2_uniform,
        'p_uniform': p_uniform,
        'dG2_total': dG2_total,
        'p_total': p_total,
        'dr2_nonuniform': dr2_nonuniform,
        'dr2_uniform': dr2_uniform,
        'dr2_total': dr2_total,
        'JG_nonuniform': jg_category(dr2_nonuniform),
        'JG_uniform': jg_category(dr2_uniform),
        'JG_total': jg_category(dr2_total),
        'beta_full': beta_full,
        'beta_red1': beta_red1,
    }

# ============================================================
# Code Block 7
# ============================================================

print(f"{'項目':>4} {'ΔG²(非一様)':>12} {'p値':>8} {'ΔG²(一様)':>12} {'p値':>8} {'ΔR²(全体)':>12} {'JG':>4} {'真のDIF':>10}")
print('-' * 85)

for j in range(n_items):
    result = logistic_regression_dif(responses, group, j)
    dif_label = true_dif.get(j + 1, 'なし')
    sig_nu = '***' if result['p_nonuniform'] < 0.001 else '**' if result['p_nonuniform'] < 0.01 else '*' if result['p_nonuniform'] < 0.05 else ''
    sig_u = '***' if result['p_uniform'] < 0.001 else '**' if result['p_uniform'] < 0.01 else '*' if result['p_uniform'] < 0.05 else ''
    print(f"{j+1:>4} {result['dG2_nonuniform']:>12.3f} {result['p_nonuniform']:>7.4f}{sig_nu:3s} "
          f"{result['dG2_uniform']:>12.3f} {result['p_uniform']:>7.4f}{sig_u:3s} "
          f"{result['dr2_total']:>12.4f} {result['JG_total']:>4} {dif_label:>10}")

# ============================================================
# Code Block 8
# ============================================================

def irf_2pl(theta, alpha, delta):
    """2PLの項目反応関数"""
    z = alpha * (theta - delta)
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def neg_log_lik_2pl(params, theta, responses):
    """
    2PLモデルの負の対数尤度。
    params: [alpha_1, delta_1, alpha_2, delta_2, ..., alpha_J, delta_J]
    theta: shape (n,) — 受験者の能力（既知として扱う）
    responses: shape (n, J)
    """
    J = responses.shape[1]
    total = 0.0
    for j in range(J):
        alpha_j = params[2 * j]
        delta_j = params[2 * j + 1]
        p = irf_2pl(theta, alpha_j, delta_j)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        total += np.sum(responses[:, j] * np.log(p) + (1 - responses[:, j]) * np.log(1 - p))
    return -total

def tsw_dif_test(responses, group, theta_est, item_idx):
    """
    TSW-ΔG²によるDIF検定（簡易版：θを既知として扱う）。
    """
    n_items = responses.shape[1]
    
    ref_mask = (group == 0)
    foc_mask = (group == 1)
    
    resp_ref = responses[ref_mask]
    resp_foc = responses[foc_mask]
    theta_r = theta_est[ref_mask]
    theta_f = theta_est[foc_mask]
    
    # モデル1（自由推定）：対象項目は群ごとに別パラメータ
    # モデル2（等価制約）：全項目で共通パラメータ
    
    # まず全項目の共通推定（等価制約モデル）
    params0 = np.ones(2 * n_items)
    params0[1::2] = 0.0  # deltaの初期値を0に
    
    result_eq = minimize(
        lambda p: neg_log_lik_2pl(p, theta_r, resp_ref) + neg_log_lik_2pl(p, theta_f, resp_foc),
        params0, method='L-BFGS-B'
    )
    G2_eq = 2 * result_eq.fun  # -2 ln L（等価制約）
    
    # 自由推定モデル：対象項目だけ群ごとに別パラメータ
    # 基準群と焦点群で対象項目のパラメータを分離
    # params: [共通α_1, 共通δ_1, ..., 基準群α_target, 基準群δ_target, ..., 焦点群α_target, 焦点群δ_target]
    n_shared = n_items - 1
    n_params_free = 2 * n_shared + 4  # 共通(n_shared項目×2) + 基準群(2) + 焦点群(2)
    
    def neg_ll_free(params_free):
        shared_params = params_free[:2 * n_shared]
        ref_target = params_free[2 * n_shared:2 * n_shared + 2]
        foc_target = params_free[2 * n_shared + 2:2 * n_shared + 4]
        
        # 基準群のパラメータ配列を構築
        ref_params = np.zeros(2 * n_items)
        foc_params = np.zeros(2 * n_items)
        
        k = 0
        for j in range(n_items):
            if j == item_idx:
                ref_params[2 * j] = ref_target[0]
                ref_params[2 * j + 1] = ref_target[1]
                foc_params[2 * j] = foc_target[0]
                foc_params[2 * j + 1] = foc_target[1]
            else:
                ref_params[2 * j] = shared_params[2 * k]
                ref_params[2 * j + 1] = shared_params[2 * k + 1]
                foc_params[2 * j] = shared_params[2 * k]
                foc_params[2 * j + 1] = shared_params[2 * k + 1]
                k += 1
        
        return neg_log_lik_2pl(ref_params, theta_r, resp_ref) + neg_log_lik_2pl(foc_params, theta_f, resp_foc)
    
    params0_free = np.ones(n_params_free)
    params0_free[1::2] = 0.0
    
    result_free = minimize(neg_ll_free, params0_free, method='L-BFGS-B')
    G2_free = 2 * result_free.fun  # -2 ln L（自由推定）
    
    # TSW-ΔG²
    delta_G2 = G2_eq - G2_free
    p_value = 1 - stats.chi2.cdf(delta_G2, df=2)  # α,δ両方をテスト → df=2
    
    return {
        'delta_G2': delta_G2,
        'p_value': p_value,
        'G2_eq': G2_eq,
        'G2_free': G2_free,
    }

# ============================================================
# Code Block 9
# ============================================================

# θの近似値：合計得点を標準化
X_total = responses.sum(axis=1)
theta_approx = (X_total - X_total.mean()) / X_total.std()

print(f"{'項目':>4} {'TSW-ΔG²':>10} {'p値':>10} {'真のDIF':>10}")
print('-' * 45)

for j in range(n_items):
    result = tsw_dif_test(responses, group, theta_approx, j)
    dif_label = true_dif.get(j + 1, 'なし')
    sig = '***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else ''
    print(f"{j+1:>4} {result['delta_G2']:>10.3f} {result['p_value']:>10.4f}{sig:3s} {dif_label:>10}")

# ============================================================
# Code Block 10
# ============================================================

def iterative_purification_mh(responses, group, alpha_level=0.05, max_iter=10):
    """
    MH検定の反復的浄化を実行する。
    """
    n_items = responses.shape[1]
    excluded = set()
    history = []
    
    for iteration in range(max_iter):
        # 合計得点を計算（除外された項目を含めない）
        included = [j for j in range(n_items) if j not in excluded]
        resp_included = responses[:, included]
        X_purified = resp_included.sum(axis=1)
        
        # 各項目のMH検定
        results = {}
        for j in range(n_items):
            item_resp = responses[:, j]
            
            # MH検定をX_purifiedで条件付けて実行
            sum_A_E = 0.0
            sum_var = 0.0
            sum_AD_n = 0.0
            sum_BC_n = 0.0
            
            score_vals = np.unique(X_purified)
            for t in score_vals:
                if t == 0 or t == len(included):
                    continue
                mask = (X_purified == t)
                ref_mask = mask & (group == 0)
                foc_mask = mask & (group == 1)
                
                n_Rt = ref_mask.sum()
                n_Ft = foc_mask.sum()
                n_t = n_Rt + n_Ft
                
                if n_t <= 1 or n_Rt == 0 or n_Ft == 0:
                    continue
                
                A_t = (item_resp[ref_mask] == 1).sum()
                B_t = (item_resp[ref_mask] == 0).sum()
                C_t = (item_resp[foc_mask] == 1).sum()
                D_t = (item_resp[foc_mask] == 0).sum()
                n_1t = A_t + C_t
                n_0t = B_t + D_t
                
                E_At = n_Rt * n_1t / n_t
                var_At = n_Rt * n_Ft * n_1t * n_0t / (n_t**2 * (n_t - 1))
                
                sum_A_E += (A_t - E_At)
                sum_var += var_At
                sum_AD_n += A_t * D_t / n_t
                sum_BC_n += B_t * C_t / n_t
            
            mh_chi2 = (abs(sum_A_E) - 0.5)**2 / sum_var if sum_var > 0 else 0.0
            p_value = 1.0 - stats.chi2.cdf(mh_chi2, df=1)
            alpha_MH = sum_AD_n / sum_BC_n if sum_BC_n > 0 else np.inf
            beta_MH = np.log(alpha_MH) if alpha_MH > 0 and np.isfinite(alpha_MH) else np.inf
            D_MH = -2.35 * beta_MH if np.isfinite(beta_MH) else np.inf
            
            results[j] = {
                'MH_chi2': mh_chi2, 'p_value': p_value,
                'alpha_MH': alpha_MH, 'D_MH': D_MH,
            }
        
        # DIF項目を特定
        sig_items = {j: r for j, r in results.items()
                     if r['p_value'] < alpha_level and j not in excluded
                     and abs(r['D_MH']) >= 1.0}
        
        history.append({
            'iteration': iteration + 1,
            'excluded': excluded.copy(),
            'results': results,
            'sig_items': sig_items,
        })
        
        if not sig_items:
            break
        
        # 最大の|D_MH|を持つ項目を除外
        worst = max(sig_items, key=lambda j: abs(sig_items[j]['D_MH']))
        excluded.add(worst)
        
        print(f"反復 {iteration + 1}: 項目{worst + 1}を除外 (|D_MH| = {abs(results[worst]['D_MH']):.3f})")
    
    return history, excluded

print("=== 反復的浄化（MH検定）===")
history, excluded = iterative_purification_mh(responses, group)
print(f"\n最終的に除外された項目: {sorted([j+1 for j in excluded])}")

# ============================================================
# Code Block 11
# ============================================================

# 3手法の結果をまとめる
print(f"\n{'='*90}")
print(f"{'項目':>4} {'MH χ²':>8} {'MH-p':>8} {'LR-ΔG²(全体)':>14} {'LR-p':>8} {'TSW-ΔG²':>10} {'TSW-p':>8} {'真のDIF':>10}")
print(f"{'='*90}")

for j in range(n_items):
    mh = mantel_haenszel(responses, group, j)
    lr = logistic_regression_dif(responses, group, j)
    tsw = tsw_dif_test(responses, group, theta_approx, j)
    dif_label = true_dif.get(j + 1, 'なし')
    
    print(f"{j+1:>4} {mh['MH_chi2']:>8.2f} {mh['p_value']:>8.4f} "
          f"{lr['dG2_total']:>14.2f} {lr['p_total']:>8.4f} "
          f"{tsw['delta_G2']:>10.2f} {tsw['p_value']:>8.4f} {dif_label:>10}")

# ============================================================
# Code Block 12
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
theta_plot = np.linspace(-4, 4, 500)

# 項目3（一様DIF）
ax = axes[0]
ax.plot(theta_plot, irf(theta_plot, alpha_ref[2], delta_ref[2]), 'b-', lw=2.5,
        label=f'基準群 (α={alpha_ref[2]}, δ={delta_ref[2]})')
ax.plot(theta_plot, irf(theta_plot, alpha_foc[2], delta_foc[2]), 'r--', lw=2.5,
        label=f'焦点群 (α={alpha_foc[2]}, δ={delta_foc[2]})')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$p(x=1)$')
ax.set_title('項目3：一様DIF')
ax.legend(loc='lower right', fontsize=13)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3)

# 項目7（非一様DIF）
ax = axes[1]
ax.plot(theta_plot, irf(theta_plot, alpha_ref[6], delta_ref[6]), 'b-', lw=2.5,
        label=f'基準群 (α={alpha_ref[6]}, δ={delta_ref[6]})')
ax.plot(theta_plot, irf(theta_plot, alpha_foc[6], delta_foc[6]), 'r--', lw=2.5,
        label=f'焦点群 (α={alpha_foc[6]}, δ={delta_foc[6]})')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$p(x=1)$')
ax.set_title('項目7：非一様DIF')
ax.legend(loc='lower right', fontsize=13)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('irt_dif_05_irf_comparison.png', bbox_inches='tight')
plt.show()

