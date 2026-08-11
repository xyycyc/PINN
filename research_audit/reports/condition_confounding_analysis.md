# Condition confounding analysis

- `wumu_post0`: 213 samples、6 acquisition dates、75 temperature groups；5 个温度只出现在单一日期。
- `metal_10times`: 375 samples、10 repetition folders、40 filename-value groups；2 个值只出现在单一 repetition。
- 多数 wumu 温度有约 3 个日期 repeats，多数 metal 常规温度有 10 repeats，这是有利条件；但并未保存 gain/coupling/sensor 状态，无法证明 repeats 仅改变随机噪声。
- 温度、日期、处理版本可能互相绑定。任何后续 split 必须以 temperature group + acquisition batch/physical run 为单位，并报告 leave-batch-out 结果。

Risk: **HIGH**。模型可能学习 acquisition-day/coupling identity，而非温度。
