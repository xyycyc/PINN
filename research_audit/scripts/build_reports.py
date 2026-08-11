"""Build human-readable reports and the canonical machine-readable audit summary."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from audit_utils import AUDIT_ROOT, REPORT_ROOT, STATS_ROOT, ensure_output_dirs, write_csv, write_json


def read_csv(name: str) -> list[dict[str, str]]:
    with (STATS_ROOT / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(name: str) -> Any:
    return json.loads((STATS_ROOT / name).read_text(encoding="utf-8"))


def write_report(name: str, body: str) -> None:
    (REPORT_ROOT / name).write_text(body.strip() + "\n", encoding="utf-8")


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def report_inventory(inventory: list[dict[str, str]]) -> str:
    lines = [
        "# Data inventory",
        "",
        "本表以原始文件树为事实源；路径均已逻辑化。磁盘量与文件数来自 `statistics/data_inventory.csv`。",
        "",
        "| Dataset | Origin | Cases/traces | Files | Size (GB) | Waveforms | Full field | Temperature |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in inventory:
        size_gb = as_float(row, "approximate_bytes") / 1e9
        if row.get("temperature_min_k"):
            temp = f"{float(row['temperature_min_k']):.0f}–{float(row['temperature_max_k']):.0f} K"
        else:
            temp = f"{row.get('filename_temperature_min')}–{row.get('filename_temperature_max')} {row.get('filename_temperature_unit')}"
        lines.append(
            f"| {row['dataset']} | {row['actual_origin']} | {as_int(row, 'physical_cases')} | {as_int(row, 'file_count')} | {size_gb:.3f} | {row['waveform_count']} | {row['full_field_availability']} | {temp} |"
        )
    lines += [
        "",
        "## 关键资产判断",
        "",
        "- `metal`、`silicon`、`wumu` 是求解器生成的 simulation case，不是物理实验；三者合计 1,400 cases。",
        "- `REPOSITORY_ROOT/database/raw/10times` 是 10 批 DLM3000 示波器导出；有效数值文件名波形 375 条。",
        "- `REPOSITORY_ROOT/database/raw/waveforms_by_file` 是已经做过 residual/filter/post-zero 处理的实验波形；当前原始目录有 213 条。",
        "- 原始 `wumu` 仿真占约 215.0 GB，是总数据量主体；扫描脚本未加载 VTU 或完整场到内存。",
        "- 现有 `wumu_exp` manifest 有 251 records，但当前 post-zero 目录只有 213 CSV，且温度上限分别为 177 °C 与 169 °C；manifest 不能代替原始目录事实。",
    ]
    return "\n".join(lines)


def report_provenance(provenance: list[dict[str, str]]) -> str:
    lines = ["# Source provenance map", "", "`source` 代码字段仅作线索；`actual_origin` 由目录、配置、生成脚本和采集头联合判定。", "", "| Dataset | Code label | Actual origin | Confidence | Evidence |", "|---|---|---|---|---|"]
    for row in provenance:
        lines.append(f"| {row['dataset']} | {row['code_label']} | {row['actual_origin']} | {row['confidence']} | {row['evidence']} |")
    lines += ["", "最重要的纠正：case pipeline 将 1,400 个仿真 case 写成 `source=experiment_case`；这不改变其 simulation provenance。"]
    return "\n".join(lines)


def report_simulation(inventory: list[dict[str, str]], pairing: dict[str, Any]) -> str:
    sim = [row for row in inventory if row["actual_origin"] == "simulation"]
    return f"""
# Simulation data audit

## 数据集与配对

- `metal`: {sim[0]['physical_cases']} cases，300–1500 K，求解器 receiver waveform 与 steady thermomechanical node field 齐备。
- `silicon`: {sim[1]['physical_cases']} cases，300–1500 K，另有均匀化/宏观输入资产。
- `wumu`: {sim[2]['physical_cases']} cases，300–1500 K，两层结构、界面条件与六个仿真 monitor points。
- 同 case 配置引用同 case thermal CSV；{pairing['simulation_exact_waveform_field_pairs']}/{pairing['simulation_case_count']} 个 case 通过 waveform↔field 目录/配置引用检查。

## Solver 与物理配置证据

- 热场文件来自 `thermomechanical_steady` 求解；配置含温度相关 `k`、`cp`、`rho`、弹性参数、Robin/Dirichlet/radiation 边界、接触热阻和 heat source。
- 超声配置读取该 case 的 thermal CSV，使用二维网格与显式时间推进；`wumu` 示例为 2.5 MHz、5 cycles、traction source、normal receiver，并包含界面刚度。
- `wumu` 网格配置记录 233,321 原节点（超声阶段界面复制后更多自由度）；处理后 fixed-node manifest 采样 10,000 点。
- solver 配置保存了旧机器路径和 executable 位置，但本机是否仍具备网格、求解器环境和许可尚未验证，不能声称可重跑。

## Waveform contract

- `metal`: 24,510–25,348 points，约 1.225–1.267 GHz，约 20 μs。
- `silicon`: 6,630–6,824 points，约 0.332–0.341 GHz，约 20 μs。
- `wumu`: 3,133–3,362 points，约 0.157–0.168 GHz，约 20 μs。
- 现有建库固定保留 native-rate 前 1,097 点，不插值；不同材料的实际物理时间窗口并不相同。

## Temperature field

字段包含 node coordinates、`T`、material IDs、热学/力学属性与界面信息。它们是 simulation full-field labels，不是实验 ground truth。
"""


def report_experiment(inventory: list[dict[str, str]], confounding: dict[str, Any]) -> str:
    return f"""
# Experimental data audit

## Physical experiment assets

### metal_10times_dlm

- 375 条数值文件名的原始示波器 trace，10 个 repetition folders。
- DLM3000、CH1、125,000 samples、2.5 GHz、0.4 ns、约 50 μs、单位 V；header 保存 acquisition date/time。
- 常规温度序列 30–210，每 5 单位约 10 repetitions；另有 12/22/23 等少量异常或补充文件。
- 文件名数值的单位和物理语义没有原始日志证明：**USER CONFIRMATION REQUIRED**。

### wumu_post0_waveforms

- 当前 213 条 processed waveform CSV，75 个 filename temperature groups、{confounding['wumu_post0_waveforms']['batch_count']} 个 acquisition dates。
- 文件名明确写 `TxxC`，因此数值单位为 °C；CSV 为 32,500 samples、1.25 GHz、-1 至 25 μs，含 raw residual 与 filtered residual。
- 这不是原始示波器导出；当前分析使用 `amplitude_filtered_residual`。sensor、gain、excitation、coupling 和 residual 生成参数未完整保存。
- `TxxC` 是炉温、设定值、热电偶值、表面温度还是其它量，仍无证据：**USER CONFIRMATION REQUIRED**。

## Labels and fields

- 真正测量到的量：示波器 voltage（metal）及由实验波形派生的 residual waveform（wumu）。
- 可用弱标签：文件名数值/名义温度；其 measurement semantics 未确认。
- 未发现 thermocouple coordinates、IR/infrared image、calibrated thermal camera、实验二维温度矩阵或其它 measured field。
- `database/data_process/wumu_exp/fields/*.npy` 是代码生成的均匀 scalar proxy；legacy metal fields 是 zero arrays。两者都不是 ground truth。

**Experimental full-field ground truth: NO**

## Artifact defects

- 当前 `wumu_exp` manifest 的 `temperature_k` 与 `meta.temperature_c` 数值相同（例如 29 与 29），单位字段错误；当前 builder source 已改为 +273.15，表明 artifact 与代码版本不一致。
- manifest 251 records 与当前目录 213 CSV 不一致；后续研究必须从 raw/processed CSV 重建 clean metadata，不能继续沿用该 manifest 作为科研标签。
"""


def condition_gap_rows() -> list[dict[str, str]]:
    values = [
        ("material", "layered wumu (solver labels layer_1/layer_2)", "code label wumu; specimen composition unverified", "unknown", "HIGH"),
        ("geometry", "0.1 m × 0.02 m layered model (config/field coordinates)", "not recorded with waveform CSV", "unknown", "CRITICAL"),
        ("layer structure", "two layers plus interface_1_2", "unknown", "unknown", "CRITICAL"),
        ("thickness", "0.02 m total in model", "unknown", "unknown", "CRITICAL"),
        ("temperature range", "300–1500 K", "filename 29–169 °C = 302.15–442.15 K", "experiment within scalar range", "LOW"),
        ("temperature semantics", "solver field/case temperature", "nominal filename temperature; measurement meaning unknown", "different", "CRITICAL"),
        ("heating mode", "steady thermomechanical solver", "unknown experimental heating protocol", "unknown", "HIGH"),
        ("initial condition", "configured per solver case", "unknown", "unknown", "HIGH"),
        ("boundary condition", "Dirichlet/Robin/radiation/contact model", "unknown real boundaries", "unknown", "CRITICAL"),
        ("excitation", "traction source, 2.5 MHz, 5 cycles", "unknown hardware pulse", "unknown", "CRITICAL"),
        ("frequency", "2.5 MHz configured", "dominant processed waveform near 2.39 MHz; source frequency unknown", "similar observation, configuration unknown", "HIGH"),
        ("pulse shape", "configured toneburst/custom parameters", "unknown", "unknown", "HIGH"),
        ("transducer", "boundary traction, no physical piezo model", "unknown physical transducer", "different/unknown", "CRITICAL"),
        ("receiver", "normal velocity/displacement boundary average", "processed oscilloscope residual", "different", "CRITICAL"),
        ("sensor position", "Top group x-range in config", "unknown", "unknown", "CRITICAL"),
        ("sampling rate", "156.65–168.10 MHz variable CFL", "1.25 GHz", "different", "HIGH"),
        ("waveform duration", "~20 μs", "-1 to 25 μs", "different", "MEDIUM"),
        ("gain", "solver scale", "unknown acquisition gain", "unknown", "CRITICAL"),
        ("coupling", "idealized boundary/interface stiffness", "unknown and variable", "unknown", "CRITICAL"),
        ("preprocessing", "raw solver receiver; legacy fixed-prefix crop", "residual + filter + post-zero; later 512-point interpolation", "different", "CRITICAL"),
    ]
    return [{"attribute": a, "simulation": s, "experiment": e, "relation": r, "severity": v} for a, s, e, r, v in values]


def report_condition_gap(rows: list[dict[str, str]]) -> str:
    lines = ["# Sim-real condition gap", "", "比较对象为唯一名义材料可对应的 `wumu simulation` 与 `wumu post0 experiment`。", "", "| Attribute | Simulation | Experiment | Relation | Severity |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['attribute']} | {row['simulation']} | {row['experiment']} | {row['relation']} | {row['severity']} |")
    lines += ["", "温度标量范围重叠不等于 operating-condition pairing。geometry、transducer、boundary、gain、coupling 与 preprocessing 的未知/差异足以使直接全局 domain alignment 失去科学可解释性。"]
    return "\n".join(lines)


def report_pairing(pairing: dict[str, Any]) -> str:
    return f"""
# Pairing analysis

| Pairing type | Count | Evidence |
|---|---:|---|
| Simulation waveform↔field exact local pair | {pairing['simulation_exact_waveform_field_pairs']} | Same case directory plus ultrasonic config reference to the case thermal CSV |
| Simulation internal failures | {pairing['simulation_pair_failures']} | Filesystem/config scan |
| Sim-real exact operating-condition pairs | {pairing['sim_real_exact_pair_count']} | No experiment carries a simulation case/config ID |
| Sim-real approximate pairs | {pairing['sim_real_approximate_pair_count']} | Temperature similarity alone was deliberately rejected |

结论：simulation 自身严格 paired；simulation 与 experiment **fundamentally unpaired**。实验波形不能按相近温度强行附着到 simulation field。
"""


def report_contract() -> str:
    return """
# Waveform contract comparison

| Contract | wumu simulation | wumu post0 experiment | Consequence |
|---|---|---|---|
| Signal | boundary-average normal velocity | filtered residual amplitude | physical observable differs |
| Unit/scale | solver velocity/displacement | processed residual scale | raw amplitude incomparable |
| Sampling | variable 156.65–168.10 MHz | fixed 1.25 GHz | resampling required for common tensor, but cannot restore physics |
| Time origin | first post-update state | -1 μs pre-zero to 25 μs | origin/window differs |
| Duration | ~20 μs | 25.9992 μs | support differs |
| Original length | 3,133–3,362 | 32,500 | contract differs |
| Processing | solver receiver CSV | residual + filter; post-zero builder | processing/domain effects are entangled |

本审计分别保留 raw-scale features 与 amplitude-free、Nyquist/time-fraction features。后者只是 **partial contract-relative normalization**，并非完整 acquisition harmonization；100% classifier accuracy 不能被解释成纯 physics gap。
"""


def report_domain_gap(metrics: dict[str, Any]) -> str:
    raw = metrics["raw_feature_classifier"]
    norm = metrics["contract_normalized_classifier"]
    return f"""
# Waveform domain gap

## Diagnostic design

- 比较：213 个等距抽取的 `wumu simulation` cases vs 213 个 `wumu post0` records。
- 防泄漏：实验同 filename temperature 的 repeats 保持在同一 fold；5-fold deterministic group split。
- 分类器：从零实现的 L2-regularized logistic regression，仅为 gap 诊断。

## Results

| Representation | Accuracy | Balanced accuracy | AUC | F1 |
|---|---:|---:|---:|---:|
| Raw + relative features | {raw['accuracy']:.3f} | {raw['balanced_accuracy']:.3f} | {raw['auc']:.3f} | {raw['f1']:.3f} |
| Partial contract-relative features (no absolute amplitude) | {norm['accuracy']:.3f} | {norm['balanced_accuracy']:.3f} | {norm['auc']:.3f} | {norm['f1']:.3f} |

- Standardized-feature RBF MMD: {metrics['mmd_rbf_standardized_features']:.3f}。
- Mean standardized-feature Wasserstein diagnostic: {metrics['mean_feature_wasserstein_standardized']:.3f}。
- PCA explained variance PC1/PC2: {metrics['pca_explained_variance_ratio'][0]:.3f}/{metrics['pca_explained_variance_ratio'][1]:.3f}；图中两域完全分离。

结论：gap **CRITICAL**。它不只是绝对幅值单位差；但由于 observable、sampling、sensor、gain 与 preprocessing 没有完全统一，当前不能把 gap 分解为“contract-only”与“physics-only”两部分。
"""


def report_temperature(correlations: list[dict[str, str]]) -> str:
    lookup = {(row["dataset"], row["feature"]): row for row in correlations}
    def rho(dataset: str, feature: str) -> float:
        return float(lookup[(dataset, feature)]["spearman_rho"])
    return f"""
# Temperature-sensitive signal analysis

## Wumu cross-domain evidence

| Feature | Simulation Spearman ρ | Experiment Spearman ρ | Direction |
|---|---:|---:|---|
| peak-time fraction | {rho('wumu','peak_time_fraction'):.3f} | {rho('wumu_post0_waveforms','peak_time_fraction'):.3f} | same, increasing |
| RMS | {rho('wumu','rms'):.3f} | {rho('wumu_post0_waveforms','rms'):.3f} | same, increasing |
| dominant frequency fraction | {rho('wumu','dominant_frequency_fraction'):.3f} | {rho('wumu_post0_waveforms','dominant_frequency_fraction'):.3f} | weak/mixed |
| spectral centroid fraction | {rho('wumu','spectral_centroid_fraction'):.3f} | {rho('wumu_post0_waveforms','spectral_centroid_fraction'):.3f} | inconsistent |
| spectral bandwidth fraction | {rho('wumu','spectral_bandwidth_fraction'):.3f} | {rho('wumu_post0_waveforms','spectral_bandwidth_fraction'):.3f} | inconsistent |

两域都保留 temperature-sensitive information，但实验相关性明显较弱。峰时/RMS 提供部分跨域一致趋势，频谱趋势不一致，不能据此假定同一 inverse mapping。

## Metal evidence

Simulation RMS 随温度强增，而 DLM experiment filename value 与 RMS 呈中等负相关。因为 metal 文件名单位/语义未知，此处只能报告“趋势不一致”，不能解释为材料物理矛盾。

## Limitation

这些是单变量关联，不是因果温度效应；batch、gain、coupling 与采集日期可能共同变化。所有数值可追溯至 `temperature_feature_correlations.csv`。
"""


def report_confounding(confounding: dict[str, Any]) -> str:
    w = confounding["wumu_post0_waveforms"]
    m = confounding["metal_10times_dlm"]
    return f"""
# Condition confounding analysis

- `wumu_post0`: {w['sample_count']} samples、{w['batch_count']} acquisition dates、{w['temperature_group_count']} temperature groups；{w['temperature_groups_seen_in_one_batch_only']} 个温度只出现在单一日期。
- `metal_10times`: {m['sample_count']} samples、{m['batch_count']} repetition folders、{m['temperature_group_count']} filename-value groups；{m['temperature_groups_seen_in_one_batch_only']} 个值只出现在单一 repetition。
- 多数 wumu 温度有约 3 个日期 repeats，多数 metal 常规温度有 10 repeats，这是有利条件；但并未保存 gain/coupling/sensor 状态，无法证明 repeats 仅改变随机噪声。
- 温度、日期、处理版本可能互相绑定。任何后续 split 必须以 temperature group + acquisition batch/physical run 为单位，并报告 leave-batch-out 结果。

Risk: **HIGH**。模型可能学习 acquisition-day/coupling identity，而非温度。
"""


def report_physics() -> str:
    return """
# Current PINN physics assessment

## What `residual_pinn` actually computes

- Legacy grid: unscaled finite-difference `d2x + d2y`, squared and averaged；另有相邻像素 smoothness。
- Fixed nodes: same-material k-nearest-neighbor graph；`|T_i-T_j|` smoothness + normalized graph Laplacian squared。
- Interface nodes are excluded from the graph residual；boundary nodes are excluded from the Laplacian residual。

## Missing physical terms

训练 loss 未使用 density、heat capacity、thermal conductivity、transient derivative、heat source、heat flux、initial condition、boundary condition、interface temperature/flux continuity 或 material-dependent coefficients。没有量纲、空间尺度或时间尺度一致的 heat-equation residual。

原始 solver CSV/config **包含**部分 `rho/cp/k` 和边界信息，但模型 loss 没有消费这些信息。名称中的 `PINN` 因而不能作为科学物理模型证据。

## Decision

**Can the current PINN formulation be scientifically reused as-is? No.**

It is useful as an engineering spatial regularizer, but should not automatically become the paper's physical model. 未来若使用 physics constraint，必须从可识别的实验 measurement operator、热方程、材料/边界/界面条件重新定义并做量纲验证。
"""


def report_reuse() -> str:
    return """
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
"""


def report_classification() -> str:
    return """
# Research problem classification

| Classification | Decision | Evidence |
|---|---|---|
| Covariate shift | YES, but not sufficient | Two domains have separable waveform distributions |
| Conditional shift | LIKELY / not yet identifiable | Temperature trends differ by feature; no matched conditions |
| Weak target supervision | YES | Real data has filename scalar labels only; semantics unresolved |
| OOD / extrapolation | Scalar-temperature OOD: NO for wumu; condition OOD: POSSIBLE | 302.15–442.15 K lies inside 300–1500 K, but geometry/sensor support unknown |
| Large physical mismatch | YES / CRITICAL | Observable, sampling, geometry, transducer, boundaries and preprocessing differ/unknown |
| Multi-fidelity | NOT JUSTIFIED | No same-configuration low/high-fidelity pairs |
| Simulator calibration | REQUIRED BEFORE STRONG ALIGNMENT CLAIMS | No experiment-matched waveform contract or calibrated condition cases |
| Insufficient real validation | YES | No measured real 2D field and no known sparse measurement operator |

因此这不是单纯的“sim-to-real problem”，而是 **unpaired covariate + likely conditional shift + weak supervision + large physical/contract mismatch + insufficient validation** 的组合。
"""


def report_recommendation() -> str:
    return """
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
"""


def report_18_answers(metrics: dict[str, Any]) -> str:
    return f"""
# Answers to the 18 required audit questions

1. **Actual simulation datasets:** `metal` 200、`silicon` 200、`wumu` 1,000 solver-generated cases under `RAW_DATA_ROOT`.
2. **Actual physical experimental datasets:** `metal_10times_dlm` 375 raw DLM3000 traces；`wumu_post0_waveforms` 213 processed traces whose physical acquisition is upstream of the retained CSVs.
3. **Genuinely measured labels:** waveform voltage is measured for DLM；temperature is only filename metadata, not yet tied to a documented sensor/position.
4. **Experimental temperature meaning:** wumu unit is °C by filename, but physical meaning is unknown；metal unit and meaning are unknown. **USER CONFIRMATION REQUIRED.**
5. **Experimental full-field ground truth:** **NO**.
6. **Simulation waveform and field strictly paired:** **YES, 1,400/1,400**, by same-case artifacts and config references.
7. **Simulation and experiment operating conditions paired:** **NO；exact 0, approximate 0**.
8. **Waveform domain gap:** **CRITICAL**；group-split classifier accuracy/AUC {metrics['accuracy']:.3f}/{metrics['auc']:.3f}.
9. **Is gap only incompatible acquisition contracts:** **NO conclusion possible**；contract mismatch is major, but amplitude-free relative features remain fully separable. Physics-only contribution is not identifiable.
10. **Do both retain temperature information:** **YES**, strong in simulation and moderate/confounded in experiment.
11. **Are trends cross-domain consistent:** **PARTIAL**；wumu peak time/RMS directions agree, spectral directions disagree.
12. **Is experiment within simulation support:** scalar wumu temperature range **YES** (302.15–442.15 K within 300–1500 K)；full condition support **UNKNOWN/POSSIBLY OOD**.
13. **Is weakly supervised sim-to-real defensible:** **CONDITIONALLY**, only after label semantics, H(T), acquisition metadata and spatial validation are supplied.
14. **Is condition-aware domain adaptation justified:** **Plausible and preferable to global alignment**, but currently blocked by missing condition metadata and severe mismatch.
15. **Does simulator need calibration/new cases:** **YES**, before strong sim-to-real claims.
16. **PINN components to reuse/rewrite:** reuse parsers、pairing/schema、mesh/fixed-node sampling；rewrite split/preprocessing/evaluation；scientifically replace current architecture/loss/PINN physics.
17. **New repository:** **YES LATER**, after audit acceptance and metadata confirmation；not created now.
18. **Largest scientific risk:** real 2D field is unidentifiable/unverifiable, so predictions may merely reproduce a simulation prior.
"""


def report_user_questions() -> str:
    return """
# User questions

## Q1. Experimental temperature meaning

Evidence:
- post0 filenames explicitly encode `TxxC`; metal DLM files use bare numeric names.
- no thermocouple/IR/log/config identifies the measurement operator.

Unknown:
- furnace setpoint, thermocouple, surface, average, maximum, or another quantity; metal filename unit.

Why it matters:
- determines target unit, weak loss `H(T)`, support comparison and scientific claim.

Question for user:
- What physical quantity, device and spatial location generated each filename temperature, and is metal `30…210` in °C?

## Q2. Wumu experimental acquisition contract

Evidence:
- processed CSV retains time and residual columns but not hardware configuration.

Unknown:
- specimen geometry/layers, transducer model/position, excitation, gain, coupling, raw sample source and filter/residual procedure.

Why it matters:
- currently the domain gap cannot be separated into contract mismatch and physics mismatch.

Question for user:
- Are acquisition logs or original oscilloscope files available for the wumu post0 data?

## Q3. Spatial temperature validation

Evidence:
- no measured IR field or known-coordinate thermocouple table was found; six `p1…p6` files are simulation monitor outputs.

Unknown:
- whether unindexed IR images/thermocouple records exist elsewhere.

Why it matters:
- without real spatial measurements, field reconstruction accuracy is unidentifiable.

Question for user:
- Do any calibrated IR images or thermocouple measurements with coordinates/timestamps exist outside the scanned roots?

## Q4. Simulator reproducibility

Evidence:
- configs/logs retain solver parameters and old executable/mesh paths.

Unknown:
- whether the solver source, meshes and runtime are available and runnable on the current machine.

Why it matters:
- calibration and new experiment-matched cases require rerunning the simulator.

Question for user:
- Is the heat/ultrasonic solver environment and its original mesh input still available?
"""


def main() -> int:
    ensure_output_dirs()
    inventory = read_csv("data_inventory.csv")
    provenance = read_csv("source_provenance_map.csv")
    pairing = read_json("pairing_statistics.json")
    metrics = read_json("domain_classifier_metrics.json")
    correlations = read_csv("temperature_feature_correlations.csv")
    confounding = read_json("condition_confounding.json")

    gap_rows = condition_gap_rows()
    write_csv(STATS_ROOT / "sim_real_condition_gap.csv", gap_rows)

    write_report("data_inventory.md", report_inventory(inventory))
    write_report("source_provenance_map.md", report_provenance(provenance))
    write_report("simulation_data_audit.md", report_simulation(inventory, pairing))
    write_report("experimental_data_audit.md", report_experiment(inventory, confounding))
    write_report("sim_real_condition_gap.md", report_condition_gap(gap_rows))
    write_report("pairing_analysis.md", report_pairing(pairing))
    write_report("waveform_contract_comparison.md", report_contract())
    write_report("waveform_domain_gap.md", report_domain_gap(metrics))
    write_report("temperature_signal_analysis.md", report_temperature(correlations))
    write_report("condition_confounding_analysis.md", report_confounding(confounding))
    write_report("current_pinn_physics_assessment.md", report_physics())
    write_report("pinn_reuse_assessment.md", report_reuse())
    write_report("research_problem_classification.md", report_classification())
    write_report("paper_direction_recommendation.md", report_recommendation())
    write_report("audit_answers_18.md", report_18_answers(metrics))
    write_report("user_questions.md", report_user_questions())

    audit_summary = {
        "simulation": {
            "datasets": ["metal", "silicon", "wumu"],
            "case_count": pairing["simulation_case_count"],
            "temperature_range_k": [300.0, 1500.0],
            "has_full_field": True,
            "waveform_contract": {
                "per_material_native_rate": True,
                "duration_s_approx": 2.0e-5,
                "sampling_rate_hz_by_dataset": {
                    "metal": [1.2255e9, 1.2674e9],
                    "silicon": [3.315e8, 3.412e8],
                    "wumu": [1.5665e8, 1.6810e8],
                },
                "legacy_processing": "fixed native-rate prefix of 1097 points",
            },
        },
        "experiment": {
            "datasets": ["metal_10times_dlm", "wumu_post0_waveforms"],
            "case_count": 588,
            "temperature_range_k": [302.15, 442.15],
            "temperature_label_semantics": "wumu: filename-encoded nominal degrees Celsius; physical measurement meaning unknown (USER CONFIRMATION REQUIRED). metal: numeric filename unit/meaning unknown.",
            "has_full_field_ground_truth": False,
            "available_weak_labels": [
                "wumu filename nominal temperature in degrees Celsius (measurement semantics unknown)",
                "metal numeric filename label (unit and semantics unknown)",
            ],
            "raw_measured_quantities": ["DLM3000 CH1 voltage traces"],
            "processed_quantities": ["wumu raw/filtered residual waveform"],
        },
        "pairing": {
            "exact_pairs": 0,
            "approximate_pairs": 0,
            "is_fundamentally_unpaired": True,
            "simulation_internal_waveform_field_exact_pairs": pairing["simulation_exact_waveform_field_pairs"],
        },
        "domain_gap": {
            "domain_classifier_accuracy": metrics["accuracy"],
            "domain_classifier_auc": metrics["auc"],
            "qualitative_severity": "CRITICAL",
            "normalization_scope": "Amplitude-free, time-fraction and Nyquist-relative features; not full acquisition harmonization.",
        },
        "temperature_information": {
            "simulation": {"present": True, "strength": "strong monotonic trends"},
            "experiment": {"present": True, "strength": "moderate feature trends with batch/label uncertainty"},
            "cross_domain_trend_consistency": "PARTIAL: peak time and RMS directions agree for wumu; spectral trends disagree.",
        },
        "support": {
            "simulation_temperature_range_k": [300.0, 1500.0],
            "experiment_temperature_range_k": [302.15, 442.15],
            "shared_temperature_range_k": [302.15, 442.15],
            "experiment_out_of_simulation_support": False,
            "caveat": "Only scalar numeric range is comparable; geometry/sensor/excitation support is unknown.",
        },
        "physics": {
            "current_pinn_reusable_as_is": False,
            "assessment": "Engineering smoothness/graph-Laplacian regularizer; not a dimensional heat-equation model.",
        },
        "recommendation": {
            "problem_classification": ["covariate_shift", "likely_conditional_shift", "weak_supervision", "large_physical_mismatch", "insufficient_real_validation"],
            "recommended_method_family": "metadata/measurement-operator recovery + simulator calibration/new matched cases, then simulation pretraining with condition-aware weak supervision",
            "future_new_repository_recommended": True,
            "additional_data_required": ["temperature label semantics", "experiment geometry and acquisition metadata", "known-coordinate temperature measurements", "calibrated IR subset", "experiment-matched simulation cases"],
            "main_scientific_risk": "Current real supervision cannot identify or validate a 2D temperature field; predictions may reproduce only the simulation prior.",
            "current_full_field_paper_feasible": False,
            "conditional_independent_project_feasible": True,
        },
        "artifact_warnings": [
            "case_pipeline source=experiment_case labels simulation data",
            "existing wumu_exp temperature_k numerically equals Celsius filenames",
            "existing wumu_exp manifest has 251 records while current source directory has 213 CSVs",
            "proxy/zero experimental fields are not measured ground truth",
        ],
    }
    write_json(STATS_ROOT / "audit_summary.json", audit_summary)

    figures_readme = """
# Diagnostic figures

| Figure | Script | Input | Filtering/normalization | Sample count | Purpose |
|---|---|---|---|---:|---|
| `wumu_sim_real_pca.png` | `analyze_domain_gap.py` | `waveform_statistics.csv` | amplitude-free relative features, z-score over diagnostic set, PCA/SVD | 426 | visualize domain separation |
| `wumu_rms_domain_gap.png` | `analyze_domain_gap.py` | `waveform_statistics.csv` | log10 RMS, balanced 213/213 | 426 | show raw-scale gap |
| `wumu_temperature_peak_time.png` | `analyze_domain_gap.py` | `waveform_statistics.csv` | experiment filename °C converted to K for axis only | 426 | compare temperature trend/support |
| `wumu_domain_classifier_roc.png` | `analyze_domain_gap.py` | `waveform_statistics.csv` | 5-fold group-split logistic scores | 426 | leakage-aware separability diagnostic |

No raw waveform samples or absolute machine paths are embedded in figures.
"""
    (AUDIT_ROOT / "figures" / "README.md").write_text(figures_readme.strip() + "\n", encoding="utf-8")

    readme = """
# PINN raw-data research audit

## Decision in one minute

- Actual simulation: `metal` (200), `silicon` (200), `wumu` (1,000); all 1,400 have strict internal waveform↔field pairs.
- Actual experiment: 375 raw DLM3000 metal traces plus 213 processed wumu post-zero traces.
- Sim-real operating conditions are unpaired: exact 0, approximate 0.
- Real supervision is filename-level scalar temperature only; meaning is not confirmed. Experimental measured 2D full field: **NO**.
- Wumu domain classifier: accuracy/AUC 1.000/1.000 even without absolute amplitude; gap: **CRITICAL**, partly entangled with acquisition contracts.
- Temperature information exists in both domains; peak-time/RMS trends partly agree, spectral trends do not.
- Current `residual_pinn` is smoothness/graph Laplacian, not a heat-equation PINN; do not reuse as paper physics.
- Independent research direction is conditionally worthwhile, but current data cannot support a defensible real full-field reconstruction claim.

## Recommendation

Recover target-label/acquisition metadata and add minimum spatial validation; calibrate/generate experiment-matched simulations. Only then evaluate simulation pretraining + condition-aware weak supervision. Create a clean independent research repository after this audit is accepted—not now.

## Entry points

- Machine summary: `statistics/audit_summary.json`
- Main recommendation: `reports/paper_direction_recommendation.md`
- Required 18 answers: `reports/audit_answers_18.md`
- Data facts: `reports/data_inventory.md`
- Provenance corrections: `reports/source_provenance_map.md`
- Open questions: `reports/user_questions.md`

## Reproduce

Use the bundled/project Python with NumPy and Pillow. Scripts are deterministic, read-only for source data, and write only below `research_audit/`:

```powershell
python research_audit/scripts/scan_data.py --raw-data-root E:/pinn_data
python research_audit/scripts/audit_provenance.py
python research_audit/scripts/analyze_waveforms.py --raw-data-root E:/pinn_data
python research_audit/scripts/analyze_temperature_features.py
python research_audit/scripts/analyze_domain_gap.py
python research_audit/scripts/build_reports.py
```

Large raw/processed arrays, checkpoints and caches are excluded by `research_audit/.gitignore`.
"""
    (AUDIT_ROOT / "README.md").write_text(readme.strip() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
