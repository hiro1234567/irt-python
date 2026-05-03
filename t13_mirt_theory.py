"""
T13: 多次元IRT（MIRT）の理論 — 複数能力を同時に測定するモデルをPythonで可視化する

M2PLモデルの定式化、項目応答曲面（IRS）の3Dプロット・等高線・条件付きトレースライン、
多次元識別力A・項目位置Δ・方向余弦、項目ベクトルグラフ、M3PLモデルの可視化。

記事: https://bigdata-analytics.jp/analytics/irt-mirt-theory/
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    'figure.dpi': 220,
    'axes.titlesize': 20,
    'axes.labelsize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'font.family': 'Hiragino Sans',
})


# --- M2PLモデル ---

def m2pl_prob(theta, alpha, gamma):
    """M2PLモデルの正答確率を計算する

    Parameters
    ----------
    theta : array-like, shape (m,)
        能力ベクトル
    alpha : array-like, shape (m,)
        識別力ベクトル
    gamma : float
        切片パラメータ

    Returns
    -------
    float
        正答確率 P(x=1|theta)
    """
    logit = np.dot(alpha, theta) + gamma
    return 1 / (1 + np.exp(-logit))


def m2pl_prob_grid(T1, T2, alpha, gamma):
    """グリッド上でM2PLの正答確率を一括計算する"""
    return 1 / (1 + np.exp(-(alpha[0] * T1 + alpha[1] * T2 + gamma)))


# --- M3PLモデル ---

def m3pl_prob(theta, alpha, gamma, chi):
    """M3PLモデルの正答確率を計算する

    Parameters
    ----------
    chi : float
        疑似推測パラメータ（下方漸近線）
    """
    logit = np.dot(alpha, theta) + gamma
    p_star = 1 / (1 + np.exp(-logit))
    return chi + (1 - chi) * p_star


# --- 多次元パラメータの計算 ---

def mirt_params(alpha, gamma):
    """多次元識別力A、項目位置Δ、方向余弦を計算する"""
    A = np.linalg.norm(alpha)
    Delta = -gamma / A
    cos_omega = alpha / A
    omega_deg = np.degrees(np.arccos(cos_omega))
    return A, Delta, cos_omega, omega_deg


# --- デモ実行 ---

if __name__ == '__main__':

    # パラメータ設定
    alpha = np.array([1.5, 0.8])
    gamma = -0.5

    # 4人の受験者の正答確率
    print('=== M2PL 正答確率の計算例 ===')
    examinees = {
        'A (両方得意)': np.array([1.0, 1.0]),
        'B (theta1だけ得意)': np.array([1.5, -0.5]),
        'C (theta2だけ得意)': np.array([-0.5, 2.0]),
        'D (両方苦手)': np.array([-1.0, -1.0]),
    }
    for name, theta in examinees.items():
        prob = m2pl_prob(theta, alpha, gamma)
        logit = np.dot(alpha, theta) + gamma
        print(f'  {name}: theta={theta}, logit+gamma={logit:.2f}, P={prob:.3f}')

    # 多次元パラメータ
    print('\n=== 多次元パラメータ ===')
    A, Delta, cos_omega, omega_deg = mirt_params(alpha, gamma)
    print(f'  alpha = {alpha}')
    print(f'  A (多次元識別力) = {A:.3f}')
    print(f'  Delta (多次元項目位置) = {Delta:.3f}')
    print(f'  cos omega = {cos_omega}')
    print(f'  omega (角度) = {omega_deg} 度')

    # M3PLとの比較
    print('\n=== M2PL vs M3PL (低能力域) ===')
    chi = 0.25
    theta_low = np.array([-2.0, -2.0])
    print(f'  M2PL P(theta=(-2,-2)) = {m2pl_prob(theta_low, alpha, gamma):.3f}')
    print(f'  M3PL P(theta=(-2,-2)) = {m3pl_prob(theta_low, alpha, gamma, chi):.3f}')

    # --- IRS 3Dプロット ---
    theta1 = np.linspace(-3, 3, 80)
    theta2 = np.linspace(-3, 3, 80)
    T1, T2 = np.meshgrid(theta1, theta2)
    Z = m2pl_prob_grid(T1, T2, alpha, gamma)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T1, T2, Z, cmap='viridis', alpha=0.85, edgecolor='none')
    ax.set_xlabel('theta1')
    ax.set_ylabel('theta2')
    ax.set_zlabel('P')
    ax.set_title('M2PL IRS')
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    plt.savefig('t13_irs_3d.png', dpi=220, bbox_inches='tight')
    plt.show()

    # --- 等高線プロット ---
    T1f, T2f = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    Zf = m2pl_prob_grid(T1f, T2f, alpha, gamma)

    fig, ax = plt.subplots(figsize=(8, 7))
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cs = ax.contour(T1f, T2f, Zf, levels=levels, cmap='RdYlBu_r')
    ax.clabel(cs, inline=True, fontsize=13, fmt='%.1f')
    ax.contour(T1f, T2f, Zf, levels=[0.5], colors='red', linewidths=2.5)
    ax.set_xlabel('theta1')
    ax.set_ylabel('theta2')
    ax.set_title('M2PL Contour Plot')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('t13_contour.png', dpi=220, bbox_inches='tight')
    plt.show()

    # --- 項目ベクトルグラフ ---
    items = {
        'Item 1 (Math)': [2.0, 0.3],
        'Item 2 (English)': [0.2, 1.8],
        'Item 3 (Both)': [1.2, 1.0],
        'Item 4 (Weak)': [0.4, 0.3],
        'Item 5 (Math+)': [1.5, 0.8],
        'Item 6 (Eng+)': [0.6, 1.4],
    }

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(items)))
    for (name, a), color in zip(items.items(), colors):
        ax.annotate('', xy=(a[0], a[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        ax.text(a[0]+0.05, a[1]+0.05, name, fontsize=12, color=color, fontweight='bold')
    ax.set_xlabel('alpha1')
    ax.set_ylabel('alpha2')
    ax.set_title('Item Vector Graph')
    ax.set_xlim(-0.3, 2.5)
    ax.set_ylim(-0.3, 2.2)
    ax.set_aspect('equal')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('t13_vector_graph.png', dpi=220, bbox_inches='tight')
    plt.show()
