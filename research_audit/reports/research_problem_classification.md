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
