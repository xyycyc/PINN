# Sim-real condition gap

比较对象为唯一名义材料可对应的 `wumu simulation` 与 `wumu post0 experiment`。

| Attribute | Simulation | Experiment | Relation | Severity |
|---|---|---|---|---|
| material | layered wumu (solver labels layer_1/layer_2) | code label wumu; specimen composition unverified | unknown | HIGH |
| geometry | 0.1 m × 0.02 m layered model (config/field coordinates) | not recorded with waveform CSV | unknown | CRITICAL |
| layer structure | two layers plus interface_1_2 | unknown | unknown | CRITICAL |
| thickness | 0.02 m total in model | unknown | unknown | CRITICAL |
| temperature range | 300–1500 K | filename 29–169 °C = 302.15–442.15 K | experiment within scalar range | LOW |
| temperature semantics | solver field/case temperature | nominal filename temperature; measurement meaning unknown | different | CRITICAL |
| heating mode | steady thermomechanical solver | unknown experimental heating protocol | unknown | HIGH |
| initial condition | configured per solver case | unknown | unknown | HIGH |
| boundary condition | Dirichlet/Robin/radiation/contact model | unknown real boundaries | unknown | CRITICAL |
| excitation | traction source, 2.5 MHz, 5 cycles | unknown hardware pulse | unknown | CRITICAL |
| frequency | 2.5 MHz configured | dominant processed waveform near 2.39 MHz; source frequency unknown | similar observation, configuration unknown | HIGH |
| pulse shape | configured toneburst/custom parameters | unknown | unknown | HIGH |
| transducer | boundary traction, no physical piezo model | unknown physical transducer | different/unknown | CRITICAL |
| receiver | normal velocity/displacement boundary average | processed oscilloscope residual | different | CRITICAL |
| sensor position | Top group x-range in config | unknown | unknown | CRITICAL |
| sampling rate | 156.65–168.10 MHz variable CFL | 1.25 GHz | different | HIGH |
| waveform duration | ~20 μs | -1 to 25 μs | different | MEDIUM |
| gain | solver scale | unknown acquisition gain | unknown | CRITICAL |
| coupling | idealized boundary/interface stiffness | unknown and variable | unknown | CRITICAL |
| preprocessing | raw solver receiver; legacy fixed-prefix crop | residual + filter + post-zero; later 512-point interpolation | different | CRITICAL |

温度标量范围重叠不等于 operating-condition pairing。geometry、transducer、boundary、gain、coupling 与 preprocessing 的未知/差异足以使直接全局 domain alignment 失去科学可解释性。
