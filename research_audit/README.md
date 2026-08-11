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
