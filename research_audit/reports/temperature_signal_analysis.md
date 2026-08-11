# Temperature-sensitive signal analysis

## Wumu cross-domain evidence

| Feature | Simulation Spearman ρ | Experiment Spearman ρ | Direction |
|---|---:|---:|---|
| peak-time fraction | 1.000 | 0.294 | same, increasing |
| RMS | 0.999 | 0.327 | same, increasing |
| dominant frequency fraction | -0.003 | 0.197 | weak/mixed |
| spectral centroid fraction | 0.994 | -0.342 | inconsistent |
| spectral bandwidth fraction | 0.848 | -0.444 | inconsistent |

两域都保留 temperature-sensitive information，但实验相关性明显较弱。峰时/RMS 提供部分跨域一致趋势，频谱趋势不一致，不能据此假定同一 inverse mapping。

## Metal evidence

Simulation RMS 随温度强增，而 DLM experiment filename value 与 RMS 呈中等负相关。因为 metal 文件名单位/语义未知，此处只能报告“趋势不一致”，不能解释为材料物理矛盾。

## Limitation

这些是单变量关联，不是因果温度效应；batch、gain、coupling 与采集日期可能共同变化。所有数值可追溯至 `temperature_feature_correlations.csv`。
