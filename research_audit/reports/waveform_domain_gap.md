# Waveform domain gap

## Diagnostic design

- 比较：213 个等距抽取的 `wumu simulation` cases vs 213 个 `wumu post0` records。
- 防泄漏：实验同 filename temperature 的 repeats 保持在同一 fold；5-fold deterministic group split。
- 分类器：从零实现的 L2-regularized logistic regression，仅为 gap 诊断。

## Results

| Representation | Accuracy | Balanced accuracy | AUC | F1 |
|---|---:|---:|---:|---:|
| Raw + relative features | 1.000 | 1.000 | 1.000 | 1.000 |
| Partial contract-relative features (no absolute amplitude) | 1.000 | 1.000 | 1.000 | 1.000 |

- Standardized-feature RBF MMD: 0.987。
- Mean standardized-feature Wasserstein diagnostic: 1.740。
- PCA explained variance PC1/PC2: 0.821/0.125；图中两域完全分离。

结论：gap **CRITICAL**。它不只是绝对幅值单位差；但由于 observable、sampling、sensor、gain 与 preprocessing 没有完全统一，当前不能把 gap 分解为“contract-only”与“physics-only”两部分。
