# PINN reuse assessment

## Directly reusable

- case discovery 与 local config↔thermal CSV pairing validation
- receiver CSV parser、thermal-field CSV reader、manifest schema
- mesh fingerprint、deterministic fixed-node sampling、单位校验
- logical material catalog 与 artifact path conventions

## Useful reference, rewrite recommended

- dataset/split（需以 physical run、temperature group、batch 防泄漏重写）
- waveform preprocessing（需保留物理时间轴和 acquisition contract）
- fixed-node field representation 与 evaluation artifact layout
- CNN/LSTM engineering baselines

## Engineering-only

- GUI、routing、customer delivery compatibility、acceptance logic、checkpoint cleanup/incremental workflow

## Scientifically re-evaluate

- architecture、loss、`residual_pinn`、normalization、sim-real mixing、evaluation protocol
- `source=experiment_case`、experimental `temperature_k`、proxy field 与旧 manifest

结论：复用 parser/schema/sampling 经验，不继承论文模型与现有标签语义。
