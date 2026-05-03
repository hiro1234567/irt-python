"""
T04: IRTの3パラメータモデル：当て推量χと異常反応パターンの検出をPythonで実装

記事のPythonコードを統合したファイル。
記事URL: https://bigdata-analytics.jp/analytics/irt-3pl-person-fit/

Author: Hiroyuki Matsumoto (Digital Boy LLC)
Repository: https://github.com/hiro1234567/irt-python
"""

# ============================================================
# Code Block 1
# ============================================================

import numpy as np
from scipy import stats
from scipy.optimize import minimize
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

np.random.seed(42)

def irf_2pl(theta, alpha, delta):
    """2PLモデルのIRF"""
    z = alpha * (theta - delta)
    return 1 / (1 + np.exp(-z))

def irf_3pl(theta, alpha, delta, chi):
    """3PLモデルのIRF"""
    z = alpha * (theta - delta)
    return chi + (1 - chi) / (1 + np.exp(-z))

n_persons = 5000
theta = np.random.normal(0, 1, n_persons)

alpha_true = 1.5
delta_true = 0.0
chi_true = 0.20

prob_3pl = irf_3pl(theta, alpha_true, delta_true, chi_true)
responses = (np.random.rand(n_persons) < prob_3pl).astype(int)

theta_sorted = np.sort(theta)
n_bins = 15
bin_edges = np.linspace(theta_sorted[0], theta_sorted[-1], n_bins + 1)
bin_centers = []
bin_proportions = []

for i in range(n_bins):
    mask = (theta >= bin_edges[i]) & (theta < bin_edges[i + 1])
    if mask.sum() > 0:
        bin_centers.append(theta[mask].mean())
        bin_proportions.append(responses[mask].mean())

theta_line = np.linspace(-4, 4, 300)
p_2pl = irf_2pl(theta_line, alpha_true, delta_true)
p_3pl = irf_3pl(theta_line, alpha_true, delta_true, chi_true)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(bin_centers, bin_proportions, s=60, zorder=5,
           color='#2196F3', edgecolors='white', linewidth=0.8,
           label='実データの正答率（ビン集計）')
ax.plot(theta_line, p_2pl, '--', color='#FF9800', linewidth=2,
        label=f'2PLモデル (α={alpha_true}, δ={delta_true})')
ax.plot(theta_line, p_3pl, '-', color='#4CAF50', linewidth=2,
        label=f'3PLモデル (α={alpha_true}, δ={delta_true}, χ={chi_true})')
ax.axhline(y=chi_true, color='gray', linestyle=':', linewidth=1,
           label=f'下方漸近線 χ = {chi_true}')
ax.set_xlabel('能力 θ')
ax.set_ylabel('正答確率 p(θ)')
ax.set_title('2PLモデル vs 3PLモデル：当て推量の「床」')
ax.legend(loc='upper left', fontsize=11)
ax.set_xlim(-4, 4)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig('irt_3pl_floor_comparison.png', bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 3
# ============================================================

theta_range = np.linspace(-4, 4, 300)
alpha, delta = 1.5, 0.0
chi_values = [0.0, 0.10, 0.20, 0.30]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

fig, ax = plt.subplots(figsize=(8, 5))

for chi, color in zip(chi_values, colors):
    p = irf_3pl(theta_range, alpha, delta, chi)
    label = f'χ = {chi:.2f}' if chi > 0 else 'χ = 0（2PL相当）'
    ax.plot(theta_range, p, color=color, linewidth=2, label=label)
    ax.axhline(y=chi, color=color, linestyle=':', linewidth=0.8, alpha=0.5)

ax.axvline(x=delta, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('能力 θ')
ax.set_ylabel('正答確率 p(θ)')
ax.set_title(f'χの値によるIRFの変化（α={alpha}, δ={delta}）')
ax.legend(loc='upper left', fontsize=11)
ax.set_xlim(-4, 4)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig('irt_3pl_chi_comparison.png', bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 4
# ============================================================

theta_persons = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
responses_example = np.array([1, 0, 1, 1, 1])
delta_fixed = 0.0

alpha_grid = np.linspace(0.3, 3.0, 100)
chi_grid = np.linspace(0.01, 0.45, 100)
A, C = np.meshgrid(alpha_grid, chi_grid)
loglik = np.zeros_like(A)

for i in range(len(theta_persons)):
    p = C + (1 - C) / (1 + np.exp(-A * (theta_persons[i] - delta_fixed)))
    p = np.clip(p, 1e-10, 1 - 1e-10)
    if responses_example[i] == 1:
        loglik += np.log(p)
    else:
        loglik += np.log(1 - p)

fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(A, C, loglik, levels=30, cmap='RdYlBu_r')
fig.colorbar(contour, ax=ax, label='対数尤度')
ax.set_xlabel('識別力 α')
ax.set_ylabel('当て推量 χ')
ax.set_title('3PLの対数尤度：α-χ平面（δ=0 固定）')
plt.tight_layout()
plt.savefig('irt_3pl_loglik_surface.png', bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 5
# ============================================================

def item_info_3pl(theta, alpha, delta, chi):
    """3PLモデルの項目情報関数"""
    p = irf_3pl(theta, alpha, delta, chi)
    numerator = (p - chi) ** 2 / (1 - chi) ** 2
    denominator = (1 - p) / p
    return alpha ** 2 * numerator * denominator

theta_range = np.linspace(-4, 4, 300)
alpha, delta = 1.5, 0.0

info_2pl = item_info_3pl(theta_range, alpha, delta, 0.0)
info_3pl_02 = item_info_3pl(theta_range, alpha, delta, 0.20)
info_3pl_03 = item_info_3pl(theta_range, alpha, delta, 0.30)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(theta_range, info_2pl, color='#2196F3', linewidth=2,
        label='χ = 0（2PL相当）')
ax.plot(theta_range, info_3pl_02, color='#FF9800', linewidth=2,
        label='χ = 0.20')
ax.plot(theta_range, info_3pl_03, color='#E91E63', linewidth=2,
        label='χ = 0.30')
ax.axvline(x=delta, color='gray', linestyle='--', linewidth=0.8,
           alpha=0.5, label=f'δ = {delta}')
ax.set_xlabel('能力 θ')
ax.set_ylabel('情報量 I(θ)')
ax.set_title(f'項目情報関数の比較（α={alpha}, δ={delta}）')
ax.legend(loc='upper right', fontsize=11)
ax.set_xlim(-4, 4)
plt.tight_layout()
plt.savefig('irt_3pl_info_comparison.png', bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 6
# ============================================================

items = [
    {'alpha': 2.0, 'delta': -1.0, 'chi': 0.15},
    {'alpha': 1.5, 'delta': -0.3, 'chi': 0.20},
    {'alpha': 1.8, 'delta':  0.0, 'chi': 0.18},
    {'alpha': 2.2, 'delta':  0.5, 'chi': 0.12},
    {'alpha': 1.2, 'delta':  1.0, 'chi': 0.25},
]

theta_range = np.linspace(-4, 4, 300)

test_info_3pl = np.zeros_like(theta_range)
test_info_2pl = np.zeros_like(theta_range)
for item in items:
    test_info_3pl += item_info_3pl(
        theta_range, item['alpha'], item['delta'], item['chi'])
    test_info_2pl += item_info_3pl(
        theta_range, item['alpha'], item['delta'], 0.0)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(theta_range, test_info_2pl, '--', color='#2196F3', linewidth=2,
        label='2PL（χ = 0）')
ax.plot(theta_range, test_info_3pl, '-', color='#E91E63', linewidth=2,
        label='3PL（χ > 0）')
ax.fill_between(theta_range, test_info_3pl, test_info_2pl,
                alpha=0.15, color='#E91E63')
ax.set_xlabel('能力 θ')
ax.set_ylabel('テスト情報量 I(θ)')
ax.set_title('テスト情報関数の比較：2PL vs 3PL（5項目）')
ax.legend(loc='upper right', fontsize=11)
ax.set_xlim(-4, 4)
plt.tight_layout()
plt.savefig('irt_3pl_test_info.png', bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 7
# ============================================================

np.random.seed(123)

test_items = [
    {'alpha': 1.8, 'delta': -1.5, 'chi': 0.15},
    {'alpha': 2.0, 'delta': -1.0, 'chi': 0.18},
    {'alpha': 1.5, 'delta': -0.5, 'chi': 0.20},
    {'alpha': 1.7, 'delta': -0.2, 'chi': 0.16},
    {'alpha': 2.2, 'delta':  0.0, 'chi': 0.14},
    {'alpha': 1.4, 'delta':  0.3, 'chi': 0.22},
    {'alpha': 1.9, 'delta':  0.5, 'chi': 0.17},
    {'alpha': 1.6, 'delta':  0.8, 'chi': 0.19},
    {'alpha': 2.1, 'delta':  1.2, 'chi': 0.13},
    {'alpha': 1.3, 'delta':  1.5, 'chi': 0.25},
]

n_normal = 2000
theta_normal = np.random.normal(0, 1, n_normal)

n_items = len(test_items)
responses_normal = np.zeros((n_normal, n_items), dtype=int)
for j, item in enumerate(test_items):
    p = irf_3pl(theta_normal, item['alpha'], item['delta'], item['chi'])
    responses_normal[:, j] = (np.random.rand(n_normal) < p).astype(int)

n_aberrant = 50
theta_aberrant = np.random.normal(0, 0.8, n_aberrant)
responses_aberrant = np.zeros((n_aberrant, n_items), dtype=int)
for j, item in enumerate(test_items):
    p = irf_3pl(theta_aberrant, item['alpha'], item['delta'], item['chi'])
    p_flipped = 1 - p
    responses_aberrant[:, j] = (
        np.random.rand(n_aberrant) < p_flipped).astype(int)

theta_all = np.concatenate([theta_normal, theta_aberrant])
responses_all = np.vstack([responses_normal, responses_aberrant])
labels = np.array([0] * n_normal + [1] * n_aberrant)

# ============================================================
# Code Block 8
# ============================================================

def compute_lz(theta_i, responses_i, items):
    """
    受験者1人分のl_zを計算する。
    theta_i: 受験者の能力推定値（スカラー）
    responses_i: 応答ベクトル（長さL）
    items: 項目パラメータのリスト
    """
    ln_L = 0.0
    E_ln_L = 0.0
    Var_ln_L = 0.0

    for j, item in enumerate(items):
        p = irf_3pl(theta_i, item['alpha'], item['delta'], item['chi'])
        p = np.clip(p, 1e-10, 1 - 1e-10)

        if responses_i[j] == 1:
            ln_L += np.log(p)
        else:
            ln_L += np.log(1 - p)

        E_ln_L += p * np.log(p) + (1 - p) * np.log(1 - p)
        Var_ln_L += p * (1 - p) * (np.log(p / (1 - p))) ** 2

    if Var_ln_L <= 0:
        return 0.0
    return (ln_L - E_ln_L) / np.sqrt(Var_ln_L)

lz_scores = np.array([
    compute_lz(theta_all[i], responses_all[i], test_items)
    for i in range(len(theta_all))
])

# ============================================================
# Code Block 9
# ============================================================

lz_normal = lz_scores[labels == 0]
lz_aberrant = lz_scores[labels == 1]

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(lz_normal, bins=50, alpha=0.6, color='#2196F3',
        label='正常応答（n=2000）', density=True)
ax.hist(lz_aberrant, bins=20, alpha=0.7, color='#E91E63',
        label='異常応答（n=50）', density=True)

cutoff = -2.0
ax.axvline(x=cutoff, color='red', linestyle='--', linewidth=2,
           label=f'スクリーニング値 = {cutoff}')
ax.set_xlabel('l_z')
ax.set_ylabel('密度')
ax.set_title('Person Fit 指標 l_z の分布')
ax.legend(loc='upper left', fontsize=11)
plt.tight_layout()
plt.savefig('irt_3pl_lz_distribution.png', bbox_inches='tight')
plt.show()

flagged_normal = (lz_normal < cutoff).sum()
flagged_aberrant = (lz_aberrant < cutoff).sum()
print(f'正常応答者のうちフラグ：{flagged_normal}/{len(lz_normal)}'
      f' ({flagged_normal/len(lz_normal)*100:.1f}%)')
print(f'異常応答者のうちフラグ：{flagged_aberrant}/{len(lz_aberrant)}'
      f' ({flagged_aberrant/len(lz_aberrant)*100:.1f}%)')

# ============================================================
# Code Block 10
# ============================================================

item_deltas = np.array([item['delta'] for item in test_items])
item_order = np.argsort(item_deltas)

normal_idx = np.where(labels == 0)[0]
aberrant_idx = np.where(labels == 1)[0]

example_normal = normal_idx[np.argmin(np.abs(lz_scores[normal_idx]))]
example_aberrant = aberrant_idx[np.argmin(lz_scores[aberrant_idx])]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_i, (idx, title_label) in enumerate([
    (example_normal, '正常応答者'),
    (example_aberrant, '異常応答者'),
]):
    ax = axes[ax_i]
    theta_i = theta_all[idx]
    resp_i = responses_all[idx]

    ordered_deltas = item_deltas[item_order]
    ordered_responses = resp_i[item_order]

    p_expected = np.array([
        irf_3pl(theta_i, test_items[j]['alpha'],
                test_items[j]['delta'], test_items[j]['chi'])
        for j in item_order
    ])

    ax.plot(range(n_items), p_expected, 'o-', color='#2196F3',
            linewidth=2, markersize=6, label='モデル予測 p(θ)')
    for k in range(n_items):
        color = '#4CAF50' if ordered_responses[k] == 1 else '#E91E63'
        marker = '^' if ordered_responses[k] == 1 else 'v'
        ax.plot(k, ordered_responses[k], marker=marker, color=color,
                markersize=10, zorder=5)

    ax.set_xticks(range(n_items))
    ax.set_xticklabels(
        [f'δ={ordered_deltas[k]:.1f}' for k in range(n_items)],
        rotation=45, fontsize=9)
    ax.set_ylabel('正答確率 / 実際の応答')
    ax.set_title(f'{title_label}（l_z = {lz_scores[idx]:.2f}, '
                 f'θ = {theta_i:.2f}）')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig('irt_3pl_prf_comparison.png', bbox_inches='tight')
plt.show()

# ============================================================
# Code Block 11
# ============================================================

def loglik_model(theta, responses, items, model='3PL'):
    """
    モデルの対数尤度を計算する。
    model: '1PL', '2PL', '3PL' のいずれか
    """
    n_persons, n_items_local = responses.shape
    ll = 0.0
    alpha_common = np.mean([item['alpha'] for item in items])

    for j in range(n_items_local):
        if model == '1PL':
            p = irf_3pl(theta, alpha_common, items[j]['delta'], 0.0)
        elif model == '2PL':
            p = irf_3pl(theta, items[j]['alpha'], items[j]['delta'], 0.0)
        else:
            p = irf_3pl(theta, items[j]['alpha'],
                        items[j]['delta'], items[j]['chi'])
        p = np.clip(p, 1e-10, 1 - 1e-10)
        ll += np.sum(
            responses[:, j] * np.log(p)
            + (1 - responses[:, j]) * np.log(1 - p)
        )
    return ll

theta_normal_data = theta_normal
resp_normal_data = responses_normal

ll_1pl = loglik_model(theta_normal_data, resp_normal_data,
                      test_items, '1PL')
ll_2pl = loglik_model(theta_normal_data, resp_normal_data,
                      test_items, '2PL')
ll_3pl = loglik_model(theta_normal_data, resp_normal_data,
                      test_items, '3PL')

n_items = 10
n_obs = n_normal

k_1pl = n_items + 1
k_2pl = n_items * 2
k_3pl = n_items * 3

aic_1pl = -2 * ll_1pl + 2 * k_1pl
aic_2pl = -2 * ll_2pl + 2 * k_2pl
aic_3pl = -2 * ll_3pl + 2 * k_3pl

bic_1pl = -2 * ll_1pl + k_1pl * np.log(n_obs)
bic_2pl = -2 * ll_2pl + k_2pl * np.log(n_obs)
bic_3pl = -2 * ll_3pl + k_3pl * np.log(n_obs)

delta_g2_1vs2 = (-2 * ll_1pl) - (-2 * ll_2pl)
df_1vs2 = k_2pl - k_1pl
p_1vs2 = 1 - stats.chi2.cdf(delta_g2_1vs2, df_1vs2)

delta_g2_2vs3 = (-2 * ll_2pl) - (-2 * ll_3pl)
df_2vs3 = k_3pl - k_2pl
p_2vs3 = 1 - stats.chi2.cdf(delta_g2_2vs3, df_2vs3)

print('=' * 65)
print(f'{"モデル":>8} {"パラメータ数":>10} {"-2lnL":>12} '
      f'{"AIC":>12} {"BIC":>12}')
print('-' * 65)
print(f'{"1PL":>8} {k_1pl:>10} {-2*ll_1pl:>12.1f} '
      f'{aic_1pl:>12.1f} {bic_1pl:>12.1f}')
print(f'{"2PL":>8} {k_2pl:>10} {-2*ll_2pl:>12.1f} '
      f'{aic_2pl:>12.1f} {bic_2pl:>12.1f}')
print(f'{"3PL":>8} {k_3pl:>10} {-2*ll_3pl:>12.1f} '
      f'{aic_3pl:>12.1f} {bic_3pl:>12.1f}')
print('=' * 65)
print(f'\n尤度比検定：1PL vs 2PL  ΔG² = {delta_g2_1vs2:.1f}, '
      f'df = {df_1vs2}, p = {p_1vs2:.4f}')
print(f'尤度比検定：2PL vs 3PL  ΔG² = {delta_g2_2vs3:.1f}, '
      f'df = {df_2vs3}, p = {p_2vs3:.4f}')

# ============================================================
# Code Block 12
# ============================================================

models = ['1PL', '2PL', '3PL']
aic_vals = [aic_1pl, aic_2pl, aic_3pl]
bic_vals = [bic_1pl, bic_2pl, bic_3pl]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, aic_vals, width, label='AIC',
               color='#2196F3', alpha=0.8)
bars2 = ax.bar(x + width/2, bic_vals, width, label='BIC',
               color='#FF9800', alpha=0.8)
ax.set_ylabel('情報量規準')
ax.set_title('モデル比較：AIC と BIC')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

y_min = min(min(aic_vals), min(bic_vals)) * 0.9999
y_max = max(max(aic_vals), max(bic_vals)) * 1.0001
ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('irt_3pl_model_comparison.png', bbox_inches='tight')
plt.show()

