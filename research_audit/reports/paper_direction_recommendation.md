# Paper direction recommendation

## Recommended scientific problem

**Condition-aware, weakly supervised simulation-to-real ultrasonic temperature inference under unpaired acquisition and partially unknown target labels**，但应把“真实二维场重构”设为待验证目标，而不是现有数据已经支持的结论。

## Recommended method family

第一优先级是 metadata/measurement-operator recovery + simulator calibration/new matched cases；建模阶段再考虑 simulation representation pretraining、real scalar/sparse supervision、condition-aware alignment 与重新定义的 physics constraint。

## Why

- simulation 有 1,400 个严格 waveform↔field pairs；real 有重复波形和名义温度弱标签。
- wumu 两域峰时/RMS 保留部分一致温度趋势，说明 representation transfer 有研究价值。
- 100% 域可分性说明 naive pooling 或 simulation-only direct deployment 风险极高。

## Why not the alternatives

- simulation-only：无法验证 real field。
- real-only full-field：没有 real field labels。
- naive sim+real mixing：单位、observable、geometry、sampling 与 label schema 不兼容。
- global CORAL/MMD/DANN：可能把温度/物理信息连同 domain nuisance 一起消除；且条件不配对。
- CycleGAN/diffusion translation：当前没有可验证的 physics-preservation target，科学风险与成本过高。
- multi-fidelity：缺少同配置 fidelity pairs。
- current PINN：只有 smoothness/graph Laplacian，不是 heat-equation physics。

## Method candidate ranking

| Method | Feasibility | Novelty | Cost | Data need | Scientific risk | Recommendation |
|---|---|---|---|---|---|---|
| simulation-only | High | Low | Low | current sim | Critical validation | baseline only |
| real-only scalar regression | High | Low | Low | current real | cannot infer field | diagnostic baseline |
| naive sim+real mixing | Medium | Low | Low | current | Critical | reject |
| sim pretrain + real fine-tune | Medium | Medium | Medium | confirmed real labels | High | conditional candidate |
| CORAL | Medium | Low | Low–Medium | harmonized contracts | High | baseline after harmonization |
| MMD | Medium | Low | Medium | harmonized contracts | High | baseline after harmonization |
| DANN | Medium | Low–Medium | Medium | balanced target conditions | High | not first choice |
| conditional DANN | Medium | Medium | Medium–High | reliable condition metadata | Medium–High | later baseline |
| temperature-conditioned alignment | Medium | Medium–High | Medium | confirmed temperature semantics | Medium–High | preferred alignment family |
| weakly supervised field reconstruction | Low now | High | High | known H(T) + sparse labels | Critical | blocked by labels |
| physics-constrained weak supervision | Low now | High | High | PDE/BC/material + H(T) | Critical | later candidate |
| domain/physics feature disentanglement | Medium | High | High | batch/condition metadata | High | exploratory candidate |
| CycleGAN waveform translation | Low | Medium | High | physics-preserving validation | Critical | defer |
| diffusion waveform translation | Very low now | Medium–High | Very high | large validated corpus | Critical | reject for first study |
| multi-fidelity learning | Low | Medium | High | matched configurations | Critical | reject now |
| simulator calibration/new simulation generation | Medium | High scientific value | Medium–High | minimum matched experiment | Medium | highest priority |

## Required additional data

- confirm temperature label meaning/unit and measurement device/position;
- specimen geometry/layers/material batch;
- transmitter/receiver model, locations, coupling, gain, excitation and raw sampling contract;
- known-coordinate sparse temperature measurements; calibrated IR subset if claiming 2D field accuracy;
- experiment-matched simulation cases or simulator calibration observations.

## Minimum additional experiment

在同一 specimen、固定 transducer/coupling/gain 下，至少做 5 个 simulation-overlap temperature conditions × 3 repeats；同步记录原始 waveform、完整 acquisition metadata 与不少于 6 个已知坐标温度点，并在至少 3 个条件获取 calibrated IR field 作为 held-out 2D validation。具体点数可随设备能力调整，但没有空间验证就不能声称 real full-field accuracy。

## Main scientific risk

**Real field is unidentifiable from current supervision**：实验温度语义未知且无 real spatial ground truth；任何漂亮的 real field 都可能只是 simulation prior，而非被实验验证的重构。

## Should a new research repository be created later?

**YES — after this audit is accepted and label/acquisition metadata are confirmed.** 当前不创建新仓库。

## Final recommendation

**Do not start direct domain adaptation or claim real full-field reconstruction with current artifacts. Proceed conditionally with an independent, condition-aware weak-supervision study only after metadata recovery, minimum spatial validation, and experiment-matched simulator calibration. Until then, simulation is suitable for representation pretraining and hypothesis generation only.**
