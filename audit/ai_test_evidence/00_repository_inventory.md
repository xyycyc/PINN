# Repository inventory

- repository: https://github.com/xyycyc/PINN.git
- audit_branch: audit/ai-test-evidence
- base_branch: agent/fixed-node-temperature-field
- base_commit: 58bedc8461759f9afc96510bdc80dd01c1cfdd11
- audit_generated_at: 2026-07-15T15:29:48.294400+00:00
- tracked_file_count: 2727
- untracked_file_count: 4357
- dirty_worktree_at_scan: yes

## Scan scope

- Included: tracked files from `git ls-files`; untracked files from `git ls-files --others --exclude-standard`; JSON/CSV/MD/Python/Fortran/config/result metadata; existing checkpoint metadata.
- Skipped or limited: `.git` object internals; generated audit directory; file content hashing skipped for files larger than 20 MiB; no training, no inference, no data generation.

## Git status at scan

```text
## audit/ai-test-evidence
 M __pycache__/__init__.cpython-312.pyc
 M __pycache__/cli.cpython-310.pyc
 M __pycache__/cli.cpython-312.pyc
 M batch/__pycache__/batch_preprocess_test.cpython-310.pyc
 M batch/__pycache__/batch_test_modes.cpython-310.pyc
 M batch/__pycache__/plot_batch_test_loss_curves.cpython-310.pyc
 M batch/__pycache__/rerun_predict.cpython-310.pyc
 M batch/__pycache__/search_fixed_weights.cpython-310.pyc
 M config/__pycache__/__init__.cpython-312.pyc
 M config/__pycache__/config.cpython-310.pyc
 M config/__pycache__/config.cpython-312.pyc
 M config/__pycache__/materials.cpython-312.pyc
 M config/__pycache__/physics.cpython-312.pyc
 M data_process/__pycache__/__init__.cpython-310.pyc
 M data_process/__pycache__/__init__.cpython-312.pyc
 M data_process/__pycache__/builder.cpython-310.pyc
 M data_process/__pycache__/builder.cpython-312.pyc
 M data_process/__pycache__/case_pipeline.cpython-310.pyc
 M data_process/__pycache__/dataset.cpython-310.pyc
 M data_process/__pycache__/dataset.cpython-312.pyc
 M data_process/__pycache__/material_registry.cpython-312.pyc
 M data_process/__pycache__/mesh_sampling.cpython-310.pyc
 M data_process/__pycache__/preprocess.cpython-312.pyc
 M data_process/__pycache__/preprocess_cache.cpython-312.pyc
 M data_process/__pycache__/temperature_field.cpython-310.pyc
 M database/rule/material.csv
 M database/rule/trained_rules.csv
 M model/__pycache__/__init__.cpython-310.pyc
 M model/__pycache__/__init__.cpython-312.pyc
 M model/__pycache__/artifact_cleanup.cpython-310.pyc
 M model/__pycache__/artifact_cleanup.cpython-312.pyc
 M model/__pycache__/checkpoint_runtime.cpython-312.pyc
 M model/__pycache__/network.cpython-312.pyc
 M model/__pycache__/point_field.cpython-310.pyc
 M model/__pycache__/predict.cpython-310.pyc
 M model/__pycache__/predict.cpython-312.pyc
 M model/__pycache__/rule_registry.cpython-312.pyc
 M model/__pycache__/trainer.cpython-310.pyc
 M model/__pycache__/trainer.cpython-312.pyc
 M model/__pycache__/waveform_io.cpython-312.pyc
 M model/optimize/__pycache__/inversion.cpython-312.pyc
 M result/train/report/validation/requirement_3_3_report.json
 M tests/__pycache__/test_compatibility_baseline.cpython-310.pyc
 M tests/__pycache__/test_temperature_pipeline.cpython-310.pyc
 M window/__pycache__/__init__.cpython-312.pyc
 M window/__pycache__/app.cpython-312.pyc
 M window/__pycache__/rules.cpython-310.pyc
 M window/__pycache__/rules.cpython-312.pyc
 M window/__pycache__/runner.cpython-310.pyc
 M window/__pycache__/runner.cpython-312.pyc
 M window/__pycache__/settings.cpython-310.pyc
 M window/__pycache__/settings.cpython-312.pyc
 M window/__pycache__/widgets.cpython-310.pyc
 M window/__pycache__/widgets.cpython-312.pyc
 M window/tabs/__pycache__/__init__.cpython-312.pyc
 M window/tabs/__pycache__/base.cpython-310.pyc
 M window/tabs/__pycache__/base.cpython-312.pyc
 M window/tabs/__pycache__/batch_modes.cpython-310.pyc
 M window/tabs/__pycache__/batch_modes.cpython-312.pyc
 M window/tabs/__pycache__/batch_preprocess.cpython-310.pyc
 M window/tabs/__pycache__/batch_preprocess.cpython-312.pyc
 M window/tabs/__pycache__/build_db.cpython-310.pyc
 M window/tabs/__pycache__/build_db.cpython-312.pyc
 M window/tabs/__pycache__/demo.cpython-310.pyc
 M window/tabs/__pycache__/demo.cpython-312.pyc
 M window/tabs/__pycache__/manage_artifacts.cpython-310.pyc
 M window/tabs/__pycache__/manage_artifacts.cpython-312.pyc
 M window/tabs/__pycache__/online_update.cpython-310.pyc
 M window/tabs/__pycache__/online_update.cpython-312.pyc
 M window/tabs/__pycache__/path_settings.cpython-310.pyc
 M window/tabs/__pycache__/path_settings.cpython-312.pyc
 M window/tabs/__pycache__/plot_loss.cpython-312.pyc
 M window/tabs/__pycache__/predict.cpython-310.pyc
 M window/tabs/__pycache__/predict.cpython-312.pyc
 M window/tabs/__pycache__/rerun_predict.cpython-312.pyc
 M window/tabs/__pycache__/result_browser.cpython-312.pyc
 M window/tabs/__pycache__/search_weights.cpython-310.pyc
 M window/tabs/__pycache__/search_weights.cpython-312.pyc
 M window/tabs/__pycache__/train.cpython-310.pyc
 M window/tabs/__pycache__/train.cpython-312.pyc
 M window/tabs/__pycache__/validate.cpython-310.pyc
 M window/tabs/__pycache__/validate.cpython-312.pyc
?? audit/
?? database/data_process/metal_matrix_waveforms_post0/
?? database/data_process/wumu_case_temperature_field/
?? database/preprocess/clip_smooth_cq1_sw11/
?? database/raw/waveforms_by_file/build_post0_database.py
?? "database/raw/\345\267\262\345\244\204\347\220\206/"
?? database/smoke_runtime/
?? result/predict/inference/plot_smoke_signed_error/
?? result/predict/inference/two_steady_wumu_er_cli_2026_7_14_100_2026_7_14_131330/
?? result/predict/inference/two_steady_wumu_er_cli_2026_7_14_100_2026_7_14_135631/
?? result/predict/inference/two_steady_wumu_er_cli_2026_7_14_100_2026_7_14_142211/
?? result/predict/inference/two_steady_wumu_er_cli_2026_7_14_100_2026_7_14_143411/
?? result/predict/inference/two_steady_wumu_matrix_2026_7_14_1157_2026_7_14_151954/
?? result/train/report/two_steady_wumu_er_cli_2026_7_14_100/
?? result/train/report/two_steady_wumu_matrix_2026_7_14_1138/
?? result/train/report/two_steady_wumu_matrix_2026_7_14_1157/
```

## Recent Git history

```text
58bedc8 (HEAD -> audit/ai-test-evidence, origin/agent/fixed-node-temperature-field, agent/fixed-node-temperature-field) Complete fixed-node temperature field workflow
05bc281 Document current temperature field workflows
b99a3b6 Add fixed-node temperature field pipeline
c4126e4 (origin/master, master) Upload PINN project
a810780 loss修正
b7f35f9 PINN基础代码
```
