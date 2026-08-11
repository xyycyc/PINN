# Diagnostic figures

| Figure | Script | Input | Filtering/normalization | Sample count | Purpose |
|---|---|---|---|---:|---|
| `wumu_sim_real_pca.png` | `analyze_domain_gap.py` | `waveform_statistics.csv` | amplitude-free relative features, z-score over diagnostic set, PCA/SVD | 426 | visualize domain separation |
| `wumu_rms_domain_gap.png` | `analyze_domain_gap.py` | `waveform_statistics.csv` | log10 RMS, balanced 213/213 | 426 | show raw-scale gap |
| `wumu_temperature_peak_time.png` | `analyze_domain_gap.py` | `waveform_statistics.csv` | experiment filename °C converted to K for axis only | 426 | compare temperature trend/support |
| `wumu_domain_classifier_roc.png` | `analyze_domain_gap.py` | `waveform_statistics.csv` | 5-fold group-split logistic scores | 426 | leakage-aware separability diagnostic |

No raw waveform samples or absolute machine paths are embedded in figures.
