"""Regression fixtures are confined to pytest temporary directories."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ai_model.config import AIModelConfig
from ai_model.model.artifact_cleanup import (
    build_cleanup_plan,
    execute_cleanup,
    _write_rule_rows,
)
from ai_model.model.checkpoint_runtime import resolve_checkpoint_config_dict
from ai_model.model.trainer import OnlineUpdater, resolve_incremental_artifacts
from ai_model.window.settings import DEFAULT_SETTINGS, Settings, load_settings


def save_bundle(path: Path, **payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": {"weight": torch.tensor([1.0])}, **payload}, path)
    return path


@pytest.fixture
def cleanup_case(tmp_path):
    config = AIModelConfig(
        data_root=tmp_path / "data", result_root=tmp_path / "result", device="cpu"
    )
    base = save_bundle(
        config.train_checkpoint_root / "run" / "base.pt",
        config={"device": "cpu", "epochs": 7},
    )
    child = save_bundle(
        base.with_name("child.pt"),
        base_checkpoint=base.name,
        incremental_stamp="child_stamp",
    )
    grandchild = save_bundle(
        base.with_name("grandchild.pt"),
        base_checkpoint=child.name,
        incremental_stamp="grand_stamp",
    )
    for name in ("child_stamp", "grand_stamp"):
        report = (
            config.train_report_root / "run" / "Incremental" / name / "history.json"
        )
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("{}", encoding="utf-8")
    return config, base, child, grandchild


def cleanup_args(config, base):
    return dict(
        data_root=config.data_root, result_root=config.result_root, checkpoint_path=base
    )


def test_bad_numeric_settings_recover_only_bad_values_without_writing(tmp_path):
    path = tmp_path / "settings.json"
    payload = {
        "train": {
            "epochs": None,
            "early_stopping_patience": "12",
            "auto_manifest": True,
        },
        "common": {"clip_quantile": float("nan"), "data_root": "custom-data"},
        "extension": {"keep": [1, 2]},
    }
    path.write_text(json.dumps(payload), encoding="utf-8-sig")
    original = path.read_bytes()
    settings = load_settings(path)
    assert settings.get("train", "epochs") == DEFAULT_SETTINGS["train"]["epochs"]
    assert settings.get("train", "early_stopping_patience") == 12
    assert settings.get("train", "auto_manifest") is True
    assert settings.get("common", "data_root") == "custom-data"
    assert settings.data["extension"] == {"keep": [1, 2]}
    assert len(settings.recovery_warnings) == 2
    assert path.read_bytes() == original
    settings.save()
    assert not load_settings(path).recovery_warnings


@pytest.mark.parametrize(
    "value", [None, "", "wrong", float("inf"), float("nan"), True, 2.5]
)
def test_invalid_integer_save_leaves_file_and_memory_unchanged(tmp_path, value):
    settings = Settings(path=tmp_path / "settings.json")
    settings.save()
    original = settings.path.read_bytes()
    with pytest.raises(ValueError):
        settings.save_sections({"train": {"epochs": value}})
    assert settings.path.read_bytes() == original
    assert settings.get("train", "epochs") == 20


def test_bad_config_syntax_is_reported_and_reload_carries_warning(tmp_path):
    path = tmp_path / "settings.json"
    settings = Settings(path=path)
    path.write_text("[", encoding="utf-8")
    settings.reload()
    assert settings.recovery_warnings
    assert settings.get("train", "epochs") == 20
    assert path.read_text(encoding="utf-8") == "["


def test_cleanup_deletes_all_descendants_and_preserves_unrelated_artifacts(
    cleanup_case,
):
    config, base, child, grandchild = cleanup_case
    unrelated = save_bundle(base.with_name("other.pt"), config={"epochs": 1})
    plan = build_cleanup_plan(**cleanup_args(config, base))
    assert set(plan.dependent_incrementals) == {child, grandchild}
    with pytest.raises(ValueError):
        execute_cleanup(
            **cleanup_args(config, base),
            include_dependents=False,
            promote_dependents=False,
        )
    result = execute_cleanup(
        **cleanup_args(config, base), include_dependents=True, promote_dependents=False
    )
    assert {base, child, grandchild} <= set(result.removed_files)
    assert unrelated.is_file()
    assert not (config.train_report_root / "run" / "Incremental").exists()


def test_cleanup_promotes_whole_chain_with_inherited_config_and_reports(cleanup_case):
    config, base, child, grandchild = cleanup_case
    result = execute_cleanup(
        **cleanup_args(config, base), include_dependents=False, promote_dependents=True
    )
    assert not base.exists()
    assert len(result.promoted_incrementals) == 2
    for original, promoted in result.promoted_incrementals:
        bundle = torch.load(promoted, weights_only=False)
        assert bundle["base_checkpoint"] == ""
        assert bundle["config"]["epochs"] == 7
        assert torch.equal(bundle["model_state"]["weight"], torch.tensor([1.0]))
        plan = build_cleanup_plan(**cleanup_args(config, promoted))
        assert any(
            (report / "history.json").is_file() for report in plan.target_report_paths
        )
        assert not original.exists()


def test_cleanup_rejects_bad_report_stamp_before_any_deletion(cleanup_case):
    config, base, child, grandchild = cleanup_case
    save_bundle(grandchild, base_checkpoint=child.name, incremental_stamp="../..")
    sentinel = config.train_report_root / "unrelated.txt"
    sentinel.write_text("keep", encoding="utf-8")
    before = {path: path.read_bytes() for path in (base, child, grandchild, sentinel)}
    with pytest.raises(ValueError):
        execute_cleanup(
            **cleanup_args(config, base),
            include_dependents=True,
            promote_dependents=False,
        )
    assert all(path.read_bytes() == contents for path, contents in before.items())


def test_promotion_collision_cancels_before_moving_any_descendant(cleanup_case):
    config, base, child, grandchild = cleanup_case
    existing = (
        config.train_report_root / "one_steady_material_grand_stamp_inc" / "keep.txt"
    )
    existing.parent.mkdir(parents=True)
    existing.write_text("keep", encoding="utf-8")
    before = {path: path.read_bytes() for path in (base, child, grandchild, existing)}
    with pytest.raises(ValueError, match="已存在"):
        execute_cleanup(
            **cleanup_args(config, base),
            include_dependents=False,
            promote_dependents=True,
        )
    assert all(path.read_bytes() == contents for path, contents in before.items())


def test_cleanup_report_match_treats_brackets_literally(tmp_path):
    config = AIModelConfig(data_root=tmp_path / "data", result_root=tmp_path / "result")
    checkpoint = save_bundle(config.train_checkpoint_root / "run" / "model[1].pt")
    report = config.train_report_root / "run"
    report.mkdir(parents=True)
    own, other = report / "model[1]_history.json", report / "model1_history.json"
    own.write_text("{}", encoding="utf-8")
    other.write_text("{}", encoding="utf-8")
    execute_cleanup(
        **cleanup_args(config, checkpoint),
        include_dependents=False,
        promote_dependents=False,
    )
    assert not own.exists()
    assert other.is_file()


def test_dependency_cycle_is_finite_and_excludes_target(cleanup_case):
    config, base, child, grandchild = cleanup_case
    save_bundle(base, base_checkpoint=grandchild.name, incremental_stamp="base_stamp")
    assert set(
        build_cleanup_plan(**cleanup_args(config, base)).dependent_incrementals
    ) == {child, grandchild}


def test_registry_replace_failure_preserves_original_and_extension_columns(tmp_path):
    path = tmp_path / "rules.csv"
    _write_rule_rows(path, [{"parameter_name": "base.pt", "custom_note": "keep"}])
    before = path.read_bytes()
    with patch(
        "ai_model.model.artifact_cleanup.os.replace",
        side_effect=PermissionError("locked"),
    ):
        with pytest.raises(PermissionError):
            _write_rule_rows(path, [])
    assert path.read_bytes() == before
    assert b"custom_note" in before and b"keep" in before
    assert list(tmp_path.glob("*.tmp")) == []


def test_incremental_same_name_cannot_overwrite_base_or_existing_output(cleanup_case):
    config, base, child, grandchild = cleanup_case
    before = base.read_bytes()
    with pytest.raises(ValueError, match="基础 checkpoint"):
        OnlineUpdater(config).update(
            "missing-manifest.json", base, output_name=base.name
        )
    assert base.read_bytes() == before
    with pytest.raises(FileExistsError):
        resolve_incremental_artifacts(
            config, base, output_name=child.name, run_stamp="fresh"
        )
    report = config.train_report_root / "run" / "Incremental" / "child_stamp"
    with pytest.raises(FileExistsError):
        resolve_incremental_artifacts(config, base, run_stamp=report.name)


def test_legacy_checkpoint_numpy_metadata_and_relative_base_still_load(tmp_path):
    parent = save_bundle(
        tmp_path / "models" / "base.pt", config={"epochs": 3}, nodes=np.array([1, 2])
    )
    child = save_bundle(parent.with_name("incremental.pt"), base_checkpoint=parent.name)
    assert resolve_checkpoint_config_dict(child) == {"epochs": 3}


@pytest.mark.parametrize("value", [[], None, {"records": "wrong"}, {"records": [1]}])
def test_dataset_rejects_invalid_record_structure(tmp_path, value):
    from ai_model.data_process import AITemperatureDataset

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        AITemperatureDataset(path)


def test_manifest_discovery_skips_structurally_invalid_json(tmp_path):
    from ai_model.data_process import (
        latest_split_manifest,
        resolve_training_manifest_pair,
    )

    processed = tmp_path / "data_process"
    valid = processed / "valid"
    valid.mkdir(parents=True)
    manifest = valid / "train_manifest.json"
    manifest.write_text('{"records": [{}]}', encoding="utf-8")
    (valid / "split_config.json").write_text(
        '{"manifests": {"train": "train_manifest.json"}}', encoding="utf-8"
    )
    for name, config, contents in (
        ("bad_config", [], {}),
        ("bad_manifest", {"manifests": {"train": "train.json"}}, []),
    ):
        path = processed / name
        path.mkdir()
        (path / "split_config.json").write_text(json.dumps(config), encoding="utf-8")
        (path / "train.json").write_text(json.dumps(contents), encoding="utf-8")
    assert latest_split_manifest(tmp_path, "train") == manifest
    invalid = processed / "bad_manifest" / "train.json"
    with pytest.raises(ValueError, match="JSON"):
        resolve_training_manifest_pair(invalid)


@pytest.mark.parametrize(
    "flag", ["--fixed-weight-cnn", "--physics-residual-weight", "--clip-quantile"]
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_cli_rejects_non_finite_numbers_before_loading_data(flag, value):
    from ai_model.cli import _build_parser

    with pytest.raises(SystemExit) as error:
        _build_parser().parse_args(["train", f"{flag}={value}"])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "field",
    [
        "learning_rate",
        "physics_residual_weight",
        "fixed_weight_cnn",
        "fixed_weight_lstm",
    ],
)
def test_backend_rejects_non_finite_training_settings(tmp_path, field):
    from ai_model.model import ReconstructionTrainer

    config = AIModelConfig(
        data_root=tmp_path / "data",
        result_root=tmp_path / "result",
        device="cpu",
        **{field: float("nan")},
    )
    for trainer in (ReconstructionTrainer, OnlineUpdater):
        with pytest.raises(ValueError, match="finite"):
            trainer(config)
    assert not config.result_root.exists()


@pytest.mark.parametrize("owns_summary", [False, True])
def test_cleanup_preserves_similarly_named_model_reports(tmp_path, owns_summary):
    config = AIModelConfig(data_root=tmp_path / "data", result_root=tmp_path / "result")
    summary_payload = {"training_seconds": 42.5}
    checkpoint = save_bundle(
        config.train_checkpoint_root / "run" / "model.pt",
        **({"training_summary": summary_payload} if owns_summary else {}),
    )
    sibling = save_bundle(checkpoint.with_name("model_copy.pt"))
    reports = config.train_report_root / "run"
    reports.mkdir(parents=True)
    own = reports / "model_history.json"
    other = reports / "model_copy_history.json"
    summary = reports / "training_summary.json"
    for path in (own, other):
        path.write_text("{}", encoding="utf-8")
    summary.write_text(json.dumps(summary_payload), encoding="utf-8")
    execute_cleanup(
        **cleanup_args(config, checkpoint),
        include_dependents=False,
        promote_dependents=False,
    )
    assert sibling.is_file() and other.is_file()
    assert not checkpoint.exists() and not own.exists()
    assert summary.exists() is not owns_summary
