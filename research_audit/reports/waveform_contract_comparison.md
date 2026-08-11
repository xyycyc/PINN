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
