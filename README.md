# irt-python: PythonによるIRT（項目反応理論）の理論と実装

IRTの数理的基礎から推定・応用までを、数式とPythonコードの両面で体系的に実装するプロジェクトです。

## 解説記事シリーズ

各スクリプトは、以下の記事シリーズと対応しています。記事では数式の導出から実装の解説まで一気通貫で扱っています。

| # | テーマ | スクリプト | 記事 |
|---|---|---|---|
| P1 | **PythonでIRT完全ガイド（ピラーページ）** | — | [記事](https://bigdata-analytics.jp/analytics/irt-python-guide/) |
| T1 | IRTの数理的基礎：ロジスティックモデルとICCの導出 | `t01_mathematical_foundations.py` | [記事](https://bigdata-analytics.jp/analytics/irt-mathematical-foundations/) |
| T2 | IRTの尤度関数と最尤推定：θをどう求めるか | `t02_likelihood_mle.py` | [記事](https://bigdata-analytics.jp/analytics/irt-likelihood-mle/) |
| T3 | 2パラメータモデル：識別力αを項目ごとに解放する | `t03_2pl_model.py` | [記事](https://bigdata-analytics.jp/analytics/irt-2pl-model/) |
| T4 | 3パラメータモデル：当て推量χとPerson Fit | `t04_3pl_person_fit.py` | [記事](https://bigdata-analytics.jp/analytics/irt-3pl-person-fit/) |
| T5 | 同時最尤推定法（JMLE） | `t05_jmle_estimation.py` | [記事](https://bigdata-analytics.jp/analytics/irt-jmle-estimation/) |
| T6 | 周辺最尤推定法（MMLE）とEAP推定 | `t06_mmle_estimation.py` | [記事](https://bigdata-analytics.jp/analytics/irt-mmle-estimation/) |
| T7 | MCMCによるIRT母数推定 | `t07_mcmc_estimation.py` | [記事](https://bigdata-analytics.jp/analytics/irt-mcmc-estimation/) |
| T8 | 部分得点モデル（PCM）と評定尺度モデル（RSM） | `t08_polytomous_pcm_rsm.py` | [記事](https://bigdata-analytics.jp/analytics/irt-polytomous-pcm-rsm/) |
| T9 | 段階反応モデル（GRM）の数理 | `t09_grm_theory.py` | [記事](https://bigdata-analytics.jp/analytics/irt-grm-theory/) |
| T10 | GRMの母数推定とMCMC | `t10_grm_mcmc_estimation.py` | [記事](https://bigdata-analytics.jp/analytics/irt-grm-mcmc-estimation/) |
| T11 | テスト情報関数とテスト設計 | `t11_test_information_design.py` | [記事](https://bigdata-analytics.jp/analytics/irt-test-information-design/) |
| T12 | 等化とリンキング | `t12_linking_equating.py` | [記事](https://bigdata-analytics.jp/analytics/irt-linking-equating/) |
| T13 | 多次元IRT（MIRT）の理論 | `t13_mirt_theory.py` | [記事](https://bigdata-analytics.jp/analytics/irt-mirt-theory/) |
| T14 | 多次元IRT（MIRT）のPython実装と推定 | `t14_mirt_estimation.py` | [記事](https://bigdata-analytics.jp/analytics/irt-mirt-estimation/) |
| T15 | 項目機能差異（DIF）の検出 | `t15_dif_detection.py` | [記事](https://bigdata-analytics.jp/analytics/irt-dif-detection/) |
| T16 | Multilevel IRTの理論：GLMMフレームワーク | `t16_multilevel_theory.py` | [記事](https://bigdata-analytics.jp/analytics/irt-multilevel-theory/) |
| T17 | Multilevel IRT × POS分析 | `t17_multilevel_pos_analysis.py` | [記事](https://bigdata-analytics.jp/analytics/irt-multilevel-pos-analysis/) |

## ベース書籍

de Ayala, R. J. (2022). *The Theory and Practice of Item Response Theory*, 2nd Ed. Guilford Press.

IRTの理論と実装を体系的にカバーした世界標準テキスト（邦訳なし）。本プロジェクトはこの書籍の内容をPythonで独自に再構成したものです。

## セットアップ

```bash
pip install numpy scipy matplotlib
```

Python 3.9 以上を推奨。

## 記法

本プロジェクトでは de Ayala の記法に統一しています。

| 記号 | 意味 |
|---|---|
| α (alpha) | 識別力 |
| δ (delta) | 項目位置 |
| θ (theta) | 受験者の能力 |

Birnbaum 記法（a, b, D=1.702）は採用していません。

## 制作者

松本宏之（Hiroyuki Matsumoto）
合同会社デジタルボーイ代表（AI開発、データサイエンス、コンサルを行う会社です）
中小企業診断士 × データサイエンティスト × AIエンジニア

- Web: https://bigdata-analytics.jp/
- IRT・SEM・MCMCを専門とする計量心理学バックグラウンド（早稲田大学大学院 豊田秀樹研究室）
- 共著：『項目反応理論[理論編]』（朝倉書店）段階反応モデルの章を担当

## License

MIT
