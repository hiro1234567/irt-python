# irt-python: PythonによるIRT（項目反応理論）の理論と実装

IRTの数理的基礎から推定・応用までを、数式とPythonコードの両面で体系的に実装するプロジェクトです。

## 解説記事シリーズ

各スクリプトは、以下の記事シリーズと対応しています。記事では数式の導出から実装の解説まで一気通貫で扱っています。

| # | テーマ | スクリプト | 記事 |
|---|---|---|---|
| T1 | IRTの数理的基礎：ロジスティックモデルとICCの導出 | `t01_mathematical_foundations.py` | [記事](https://bigdata-analytics.jp/analytics/irt-mathematical-foundations/) |
| T2 | IRTの尤度関数と最尤推定：θをどう求めるか | `t02_likelihood_mle.py` | [記事](https://bigdata-analytics.jp/analytics/irt-likelihood-mle/) |
| T12 | テスト情報関数とテスト設計：最適な問題セットの設計法 | `t12_test_information_design.py` | [記事](https://bigdata-analytics.jp/analytics/irt-test-information-design/) |

今後、2パラメータモデル（T3）、3パラメータモデル（T4）、推定法（T5-T7）、多値モデル（T9-T11）、多次元IRT（T14-T15）、DIF（T16）、Multilevel IRT（T17-T18）と順次追加予定です。

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
