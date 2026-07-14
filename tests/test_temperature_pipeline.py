from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai_model.data_process.dataset import AITemperatureDataset, split_case_records, write_case_split_files
from ai_model.data_process.case_pipeline import build_case_dataset, crop_waveform_tail, discover_cases
from ai_model.data_process.mesh_sampling import deterministic_sample, read_mesh_csv
from ai_model.data_process.temperature_field import CaseKey, select_waveform, validate_temperature_field
from ai_model.data_process.training_input import resolve_training_manifest_pair
from ai_model.data_process.builder import DatabaseBuilder
from ai_model.data_process.material_collection import (
    build_material_collection,
    build_mixed_collection_manifest,
    discover_material_roots,
    sample_material_name,
    validate_material_split,
)
from ai_model.config import AIModelConfig
from ai_model.cli import _validate_material_collection
from ai_model.model import (
    MATERIAL_ROUTER_KIND,
    ReconstructionTrainer,
    predict_collection_with_checkpoint,
    predict_with_material_router,
)
from ai_model.model.point_field import (
    POINT_FIELD_CHECKPOINT_VERSION,
    DirectPointFieldModel,
    weighted_temperature_loss,
)
import json
import torch


class TemperatureContractTests(unittest.TestCase):
    def test_case_build_keeps_sample_material_separate_from_constituent_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "raw" / "case_0000_T0300p000K"
            configs = case / "configs"
            thermal_dir = case / "heat" / "thermomechanical_steady" / "csv"
            ultrasonic = case / "ultrasonic"
            configs.mkdir(parents=True)
            thermal_dir.mkdir(parents=True)
            ultrasonic.mkdir(parents=True)
            thermal_path = thermal_dir / "thermomechanical_steady_nodes.csv"
            thermal_path.write_text(
                "node_id,x,y,T,thermal_material_id,thermal_material_name\n"
                "1,0,0,300,1,layer_1\n"
                "2,1,0,301,1,layer_1\n"
                "3,0,1,302,2,layer_2\n"
                "4,1,1,303,2,layer_2\n",
                encoding="utf-8",
            )
            (configs / "case_ultrasonic_config.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "io": {
                                "temperature_csv_path": (
                                    "archive/worker_07/case_0000_T0300p000K/"
                                    "thermomechanical_steady_nodes.csv"
                                )
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (ultrasonic / "receiver_signal.csv").write_text(
                "step,time_s,signal\n"
                "0,0.0,0.0\n"
                "1,0.1,1.0\n"
                "2,0.2,2.0\n"
                "3,0.3,3.0\n",
                encoding="utf-8",
            )
            manifest_path = build_case_dataset(
                root / "raw",
                root / "output",
                target_points=4,
                waveform_crop_length=2,
                dataset_label="external_name_only",
                sample_material_key="wumu",
                sample_material_name="钨钼多层材料",
                train_manifest_name="fit.json",
                validation_manifest_name="tune.json",
                test_manifest_name="holdout.json",
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_label"], "external_name_only")
            self.assertEqual(payload["waveform_contract"], {
                "processing": "fixed_prefix_native_rate",
                "crop_length": 2,
                "resampled": False,
            })
            self.assertEqual(payload["records"][0]["case_key"], "worker_07/case_0000_T0300p000K")
            self.assertEqual(
                [item["material_name"] for item in payload["constituent_material_catalog"]],
                ["layer_1", "layer_2"],
            )
            self.assertEqual(
                payload["sample_material_catalog"],
                [{
                    "material_key": "wumu",
                    "material_name": "钨钼多层材料",
                    "case_count": 1,
                }],
            )
            self.assertEqual(payload["records"][0]["material_key"], "wumu")
            waveform = np.load(root / "output" / payload["records"][0]["waveform_path"])
            np.testing.assert_array_equal(waveform, np.array([0.0, 1.0], dtype=np.float32))
            self.assertEqual(payload["records"][0]["meta"]["waveform_crop_length"], 2)
            split_config = json.loads((root / "output" / "split_config.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(split_config["manifests"]["train"]).name, "fit.json")
            self.assertEqual(Path(split_config["manifests"]["validation"]).name, "tune.json")
            self.assertEqual(Path(split_config["manifests"]["test"]).name, "holdout.json")
            resolved_train, resolved_validation = resolve_training_manifest_pair(root / "output")
            self.assertEqual(resolved_train.name, "fit.json")
            self.assertIsNotNone(resolved_validation)
            self.assertEqual(resolved_validation.name, "tune.json")

    def test_case_discovery_supports_flat_and_worker_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flat = root / "case_0000_T0300p000K"
            nested = root / "worker_02" / "case_0025_T0330p030K"
            (flat / "configs").mkdir(parents=True)
            nested.mkdir(parents=True)
            (flat / "configs" / "case_ultrasonic_config.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "io": {
                                "temperature_csv_path": (
                                    "archive/worker_01/case_0000_T0300p000K/"
                                    "thermomechanical_steady_nodes.csv"
                                )
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(discover_cases(root), [flat, nested])
            self.assertEqual(CaseKey.from_case_dir(flat).value, "worker_01/case_0000_T0300p000K")
            self.assertEqual(CaseKey.from_case_dir(nested).value, "worker_02/case_0025_T0330p030K")

    def test_multi_material_discovery_uses_sibling_folders_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            (parent / "wumu" / "worker_01" / "case_0000_T0300p000K").mkdir(parents=True)
            (parent / "wumu" / "layer_1").mkdir()
            (parent / "wumu" / "layer_2").mkdir()
            (parent / "steel" / "case_0000_T0300p000K").mkdir(parents=True)
            roots = discover_material_roots(parent)
            self.assertEqual(list(roots), ["steel", "wumu"])
            self.assertNotIn("layer_1", roots)
            self.assertNotIn("layer_2", roots)
            self.assertEqual(sample_material_name("wumu"), "钨钼多层材料")
            self.assertEqual(validate_material_split(0.6, 0.1, 0.3), (0.6, 0.1, 0.3))

    def test_multi_material_build_applies_independent_splits_and_folder_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def write_case(material_key: str, index: int) -> None:
                case_name = f"case_{index:04d}_T0{300 + index:03d}p000K"
                case = root / "source" / material_key / case_name
                configs = case / "configs"
                thermal = case / "heat" / "thermomechanical_steady" / "csv"
                ultrasonic = case / "ultrasonic"
                configs.mkdir(parents=True)
                thermal.mkdir(parents=True)
                ultrasonic.mkdir(parents=True)
                (thermal / "thermomechanical_steady_nodes.csv").write_text(
                    "node_id,x,y,T,thermal_material_id,thermal_material_name\n"
                    f"1,0,0,{300 + index},1,layer_1\n"
                    f"2,1,0,{301 + index},1,layer_1\n"
                    f"3,0,1,{302 + index},2,layer_2\n"
                    f"4,1,1,{303 + index},2,layer_2\n",
                    encoding="utf-8",
                )
                (configs / "case_ultrasonic_config.json").write_text(
                    json.dumps({
                        "config": {"io": {"temperature_csv_path": (
                            f"archive/worker_01/{case_name}/thermomechanical_steady_nodes.csv"
                        )}}
                    }),
                    encoding="utf-8",
                )
                (ultrasonic / "receiver_signal.csv").write_text(
                    "step,time_s,signal\n0,0,0\n1,1,1\n2,2,2\n3,3,3\n",
                    encoding="utf-8",
                )

            for material_key in ("wumu", "steel"):
                for index in range(4):
                    write_case(material_key, index)
            collection_path = build_material_collection(
                root / "source",
                root / "built",
                dataset_label="collection_label_only",
                material_splits={
                    "wumu": (0.5, 0.25, 0.25),
                    "steel": (0.25, 0.25, 0.5),
                },
                target_points=4,
                waveform_crop_length=2,
            )
            collection = json.loads(collection_path.read_text(encoding="utf-8"))
            entries = {item["material_key"]: item for item in collection["materials"]}
            self.assertEqual(entries["wumu"]["ratios"], {
                "train": 0.5, "validation": 0.25, "test": 0.25,
            })
            self.assertEqual(entries["steel"]["ratios"], {
                "train": 0.25, "validation": 0.25, "test": 0.5,
            })
            for material_key, entry in entries.items():
                payload = json.loads(Path(entry["manifests"]["combined"]).read_text(encoding="utf-8"))
                self.assertEqual(
                    {record["material_key"] for record in payload["records"]},
                    {material_key},
                )
            self.assertEqual(
                entries["wumu"]["material_name"],
                "钨钼多层材料",
            )
            report = _validate_material_collection(
                AIModelConfig(data_root=root/"validation_db", result_root=root/"validation_result"),
                collection_path,
            )
            self.assertEqual(report["materials"], ["steel", "wumu"])
            self.assertEqual(report["material_count"], 2)
            self.assertFalse(report["meets_3_3_material_count"])

    def test_legacy_csv_import_rejects_unrelated_numbered_files_and_empty_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "p1.csv").write_text("time,value\n0,1\n", encoding="utf-8")
            (source / "thermomechanical_steady_nodes(1).csv").write_text(
                "node_id,x,T\n1,0,300\n", encoding="utf-8"
            )
            config = AIModelConfig(data_root=root / "database", result_root=root / "result", device="cpu")
            builder = DatabaseBuilder(config)
            with self.assertRaisesRegex(ValueError, "未从.*导入任何有效波形"):
                builder.import_experimental_csvs(source, "dataset_label")
            report = json.loads(
                (config.data_root / "experiment_manifest_import_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["records_imported"], 0)
            self.assertEqual(report["reason_counts"]["unsupported_temperature_filename"], 2)
            material_csv = (config.data_root / "rule" / "material.csv").read_text(encoding="utf-8")
            self.assertNotIn("dataset_label", material_csv)

    def test_legacy_numeric_dlm_csv_still_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "100.csv").write_text("0,1\n1,2\n2,3\n", encoding="utf-8")
            config = AIModelConfig(data_root=root / "database", result_root=root / "result", device="cpu")
            manifest = DatabaseBuilder(config).import_experimental_csvs(source, "legacy_material")
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["records"]), 1)
            self.assertEqual(payload["records"][0]["temperature_k"], 100.0)

    def test_mesh_constituent_catalog_is_read_from_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "mesh.csv"
            csv_path.write_text(
                "node_id,x,y,T,thermal_material_id,thermal_material_name\n"
                "1,0,0,300,1,layer_1\n"
                "2,1,0,301,1,layer_1\n"
                "3,0,1,302,2,layer_2\n"
                "4,1,1,303,2,layer_2\n",
                encoding="utf-8",
            )
            mesh = read_mesh_csv(csv_path)
            self.assertEqual(
                mesh["constituent_material_catalog"],
                [
                    {"material_id": 1, "material_name": "layer_1", "source_node_count": 2},
                    {"material_id": 2, "material_name": "layer_2", "source_node_count": 2},
                ],
            )
            sampled = deterministic_sample(mesh, target_points=4, seed=42)
            self.assertEqual(
                [item["sampled_point_count"] for item in sampled["constituent_material_catalog"]],
                [2, 2],
            )

    def test_mesh_material_catalog_rejects_conflicting_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "mesh.csv"
            csv_path.write_text(
                "node_id,x,y,T,thermal_material_id,thermal_material_name\n"
                "1,0,0,300,1,layer_1\n"
                "2,1,0,301,1,other_name\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "对应多个材料名称"):
                read_mesh_csv(csv_path)

    def test_validation_counts_sample_material_catalog_not_constituent_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "dataset_label": "external_name_only",
                        "sample_material_catalog": [
                            {"material_key": "wumu", "material_name": "钨钼多层材料"},
                            {"material_key": "steel", "material_name": "steel"},
                            {"material_key": "ceramic", "material_name": "ceramic"},
                        ],
                        "constituent_material_catalog": [
                            {"material_id": 1, "material_name": "layer_1"},
                            {"material_id": 2, "material_name": "layer_2"},
                        ],
                        "records": [
                            {
                                "source": "experiment_case",
                                "material_key": "wumu",
                                "temperature_k": 300.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = AIModelConfig(data_root=root / "database", result_root=root / "result")
            report = DatabaseBuilder(config).validate_requirement_33(manifest)
            self.assertEqual(report["materials"], ["ceramic", "steel", "wumu"])
            self.assertEqual(report["material_source"], "manifest.sample_material_catalog")
            self.assertEqual(report["dataset_label"], "external_name_only")
            self.assertTrue(report["meets_3_3_material_count"])

    def test_case_waveform_tail_crop_preserves_native_prefix(self):
        original = np.arange(10, dtype=np.float32)
        cropped = crop_waveform_tail(original, crop_length=3)
        np.testing.assert_array_equal(cropped, original[:3])
        self.assertEqual(cropped.dtype, np.float32)

    def test_case_waveform_tail_crop_rejects_invalid_length(self):
        with self.assertRaisesRegex(ValueError, "waveform_crop_length"):
            crop_waveform_tail(np.arange(10, dtype=np.float32), crop_length=0)

    def test_case_waveform_tail_crop_rejects_short_trace(self):
        with self.assertRaisesRegex(ValueError, "fewer than waveform_crop_length"):
            crop_waveform_tail(np.arange(10, dtype=np.float32), crop_length=11)

    def test_contract_rejects_missing_units_and_shape(self):
        base = {"temperature_k": np.ones(2), "coordinates_m": np.ones((2, 2)),
                "node_ids": np.arange(2), "material_ids": np.ones(2),
                "interface_side": np.zeros(2), "sample_weights": np.ones(2)}
        with self.assertRaisesRegex(ValueError, "单位"):
            validate_temperature_field(base)
        base.update(temperature_unit="K", coordinate_unit="m")
        self.assertEqual(validate_temperature_field(base), 2)

    def test_sampling_is_deterministic_and_preserves_interface_sides(self):
        x, y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
        xy = np.column_stack((x.ravel(), y.ravel()))
        xy = np.vstack((xy, [[.5, .5], [.5, .5]]))
        n = len(xy)
        mesh = {"node_ids": np.arange(1, n+1, dtype=np.int64), "coordinates_m": xy,
                "temperature_k": np.full(n, 300, np.float32), "material_ids": np.ones(n, np.int32),
                "interface_side": np.r_[np.zeros(n-2, np.int32), [1, 2]]}
        a, b = deterministic_sample(mesh, 150, 7), deterministic_sample(mesh, 150, 7)
        np.testing.assert_array_equal(a["node_ids"], b["node_ids"])
        self.assertTrue({n-1, n}.issubset(set(a["node_ids"].tolist())))

    def test_sampling_material_budget_and_weights_are_balanced(self):
        xy1 = np.column_stack((np.linspace(0, 1, 200), np.linspace(0, .5, 200)))
        xy2 = np.column_stack((np.linspace(0, 1, 200), np.linspace(.5, 1, 200)))
        xy = np.vstack((xy1, xy2)); n = len(xy)
        mesh = {"node_ids": np.arange(1, n+1, dtype=np.int64), "coordinates_m": xy,
                "temperature_k": np.full(n, 300, np.float32),
                "material_ids": np.r_[np.ones(200, np.int32), np.full(200, 2, np.int32)],
                "interface_side": np.zeros(n, np.int32)}
        sampled = deterministic_sample(mesh, 100, 42)
        counts = [int(np.sum(sampled["material_ids"] == k)) for k in (1, 2)]
        totals = [float(sampled["sample_weights"][sampled["material_ids"] == k].sum()) for k in (1, 2)]
        self.assertEqual(counts, [50, 50])
        self.assertAlmostEqual(totals[0], totals[1], delta=2.0)

    def test_waveform_policy_never_selects_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)/"worker_01"/"case_0001_T0300p000K"; ultra=case/"ultrasonic"; ultra.mkdir(parents=True)
            (ultra/"receiver_signal_rigid_motion.csv").write_text("bad", encoding="utf-8")
            with self.assertRaises(FileNotFoundError): select_waveform(case)
            (ultra/"receiver_signal.csv").write_text("ok", encoding="utf-8")
            self.assertEqual(select_waveform(case)[1], "raw")

    def test_case_split_has_no_leakage(self):
        records = [{"case_key": f"worker_01/case_{i:04d}_T0300p000K", "waveform_version": v}
                   for i in range(10) for v in ("raw", "rigid_corrected")]
        splits, report = split_case_records(records, seed=42)
        sets = [{x["case_key"] for x in splits[name]} for name in ("train", "validation", "test")]
        self.assertFalse((sets[0] & sets[1]) | (sets[0] & sets[2]) | (sets[1] & sets[2]))
        self.assertFalse(any(report["leakage"].values()))

    def test_point_dataset_and_model_end_to_end_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/"waveforms").mkdir(); (root/"temperature_fields").mkdir()
            node_ids=np.arange(1,5,dtype=np.int64); coords=np.column_stack((np.arange(4),np.zeros(4))).astype(np.float32)
            mesh_material_ids=np.array([1,1,2,2],np.int32)
            np.savez_compressed(root/"sampling_index.npz", node_ids=node_ids, coordinates_m=coords,
                material_ids=mesh_material_ids, interface_side=np.zeros(4,np.int32), sample_weights=np.ones(4,np.float32),
                metadata_json=np.array(json.dumps({"sampling_version":"test","source_mesh_fingerprint":"abc"})))
            records=[]
            for i,value in enumerate((300.,400.,500.)):
                np.save(root/"waveforms"/f"w{i}.npy",np.linspace(0,1,16,dtype=np.float32))
                np.savez_compressed(root/"temperature_fields"/f"f{i}.npz",temperature_k=(value+np.arange(4)).astype(np.float32),
                    coordinates_m=coords,node_ids=node_ids,material_ids=mesh_material_ids,interface_side=np.zeros(4,np.int32),
                    sample_weights=np.ones(4,np.float32),temperature_unit="K",coordinate_unit="m")
                records.append({"sample_id":f"s{i}","case_key":f"worker_01/case_{i:04d}_T0{i}K","source":"experiment_case",
                    "waveform_path":f"waveforms/w{i}.npy","field_path":f"temperature_fields/f{i}.npz",
                    "temperature_k":value,"material_key":"wumu","dimension":"2d","mode":"steady"})
            material_catalog=[
                {"material_id":1,"material_name":"layer_1","source_node_count":2,"sampled_point_count":2},
                {"material_id":2,"material_name":"layer_2","source_node_count":2,"sampled_point_count":2},
            ]
            manifest=root/"manifest.json"; manifest.write_text(json.dumps({"schema_version":2,"sampling_index":"sampling_index.npz",
                "dataset_name":"test","dataset_label":"composite",
                "sample_material_catalog":[{"material_key":"wumu","material_name":"钨钼多层材料","case_count":3}],
                "constituent_material_catalog":material_catalog,"records":records}),encoding="utf-8")
            split=write_case_split_files(manifest,seed=2,train_ratio=.34,validation_ratio=.33)
            ds=AITemperatureDataset(split["manifests"]["train"],AIModelConfig(data_root=root/"db",result_root=root/"out"))
            item=ds[0]; self.assertEqual(tuple(item["field"].shape),(4,)); self.assertEqual(float(item["field_mask"]),1.0)
            self.assertEqual(ds.waveform_length, 16)
            model=DirectPointFieldModel(4,hidden_dim=4,latent_dim=4,chunk_size=2)
            prediction=model(item["waveform"].unsqueeze(0)); self.assertEqual(tuple(prediction.shape),(1,4))
            loss=weighted_temperature_loss(prediction,item["field"].unsqueeze(0),item["sample_weights"].unsqueeze(0))
            self.assertTrue(torch.isfinite(loss))

            validation_config = AIModelConfig(
                data_root=root/"validation_db", result_root=root/"validation_result",
                epochs=5, early_stopping_patience=1, learning_rate=0.0,
                batch_size=1, hidden_dim=4, latent_dim=4, device="cpu",
            )
            validation_checkpoint = ReconstructionTrainer(validation_config).train(
                split["manifests"]["train"],
                validation_manifest_path=split["manifests"]["validation"],
                checkpoint_name="validated.pt",
                train_name="validated",
            )
            validation_bundle = torch.load(validation_checkpoint, map_location="cpu")
            self.assertEqual(len(validation_bundle["history"]), 2)
            self.assertTrue(all("validation_loss" in row for row in validation_bundle["history"]))
            self.assertIsNotNone(validation_bundle["training_summary"]["best_epoch"])
            self.assertEqual(validation_bundle["training_summary"]["validation_samples"], 1)
            self.assertTrue(validation_bundle["training_summary"]["stopped_early"])
            self.assertEqual(validation_bundle["training_summary"]["stopped_epoch"], 2)
            self.assertEqual(validation_bundle["training_summary"]["completed_epochs"], 2)
            self.assertEqual(validation_bundle["training_summary"]["requested_epochs"], 5)

            collection_entries = []
            for material_key in ("wumu", "steel"):
                material_root = root / "materials" / material_key
                material_root.mkdir(parents=True)
                material_records = []
                for record in records:
                    item = dict(record)
                    item["sample_id"] = f"{material_key}/{record['sample_id']}"
                    item["material_key"] = material_key
                    item["waveform_path"] = str((root / record["waveform_path"]).resolve())
                    item["field_path"] = str((root / record["field_path"]).resolve())
                    material_records.append(item)
                normalization = {
                    "version": 1,
                    "fit_split": "train",
                    "method": "dataset_train_zscore",
                    "mean_k": 351.5,
                    "std_k": 50.0,
                    "min_k": 300.0,
                    "max_k": 403.0,
                }
                common_payload = {
                    "schema_version": 2,
                    "sampling_index": str((root / "sampling_index.npz").resolve()),
                    "dataset_name": f"{material_key}_field",
                    "sample_material_catalog": [{
                        "material_key": material_key,
                        "material_name": material_key,
                        "case_count": 3,
                    }],
                    "constituent_material_catalog": material_catalog,
                    "normalization": normalization,
                }
                train_path = material_root / "train_manifest.json"
                test_path = material_root / "test_manifest.json"
                combined_path = material_root / "manifest.json"
                train_path.write_text(
                    json.dumps({**common_payload, "records": material_records[:2]}),
                    encoding="utf-8",
                )
                test_path.write_text(
                    json.dumps({**common_payload, "records": material_records[2:]}),
                    encoding="utf-8",
                )
                combined_path.write_text(
                    json.dumps({**common_payload, "records": material_records}),
                    encoding="utf-8",
                )
                collection_entries.append({
                    "material_key": material_key,
                    "material_name": material_key,
                    "manifests": {
                        "combined": str(combined_path.resolve()),
                        "train": str(train_path.resolve()),
                        "validation": str(test_path.resolve()),
                        "test": str(test_path.resolve()),
                    },
                })
            collection_path = root / "material_collection.json"
            collection_path.write_text(
                json.dumps({
                    "collection_kind": "sample_material_dataset_collection",
                    "collection_version": 1,
                    "dataset_label": "two_sample_materials",
                    "materials": collection_entries,
                }),
                encoding="utf-8",
            )
            mixed_manifest = build_mixed_collection_manifest(collection_path)
            mixed_payload = json.loads(mixed_manifest.read_text(encoding="utf-8"))
            mixed_validation_manifest = build_mixed_collection_manifest(
                collection_path,
                output_name="mixed_validation_manifest.json",
                split_kind="validation",
                normalization=mixed_payload["normalization"],
            )
            self.assertEqual(len(mixed_payload["records"]), 4)
            self.assertEqual(
                {record["material_key"] for record in mixed_payload["records"]},
                {"wumu", "steel"},
            )
            mixed_dataset = AITemperatureDataset(
                mixed_manifest,
                AIModelConfig(data_root=root/"mixed_db", result_root=root/"mixed_result"),
            )
            self.assertEqual((len(mixed_dataset), mixed_dataset.point_count), (4, 4))

            material_config=AIModelConfig(
                data_root=root/"material_db",
                result_root=root/"material_result",
                epochs=1,
                batch_size=2,
                hidden_dim=4,
                latent_dim=4,
                device="cpu",
            )
            mixed_checkpoint = ReconstructionTrainer(material_config).train(
                mixed_manifest,
                validation_manifest_path=mixed_validation_manifest,
                checkpoint_name="mixed.pt",
                train_name="mixed_materials",
            )
            mixed_metrics = predict_collection_with_checkpoint(
                material_config,
                mixed_checkpoint,
                collection_path,
                root/"mixed_predict",
            )
            self.assertEqual(
                mixed_metrics["routing_strategy"],
                "one_shared_checkpoint",
            )
            self.assertEqual(set(mixed_metrics["materials"]), {"wumu", "steel"})
            router=ReconstructionTrainer(material_config).train_material_checkpoints(
                collection_path,
                checkpoint_name="field.pt",
                train_name="material_checkpoints",
            )
            router_payload=json.loads(router.read_text(encoding="utf-8"))
            self.assertEqual(router_payload["router_kind"], MATERIAL_ROUTER_KIND)
            self.assertEqual(router_payload["routing_field"], "records[].material_key")
            self.assertEqual(len(router_payload["checkpoints"]), 2)
            self.assertNotIn("experts", router_payload)
            self.assertTrue(all(
                (router.parent/item["checkpoint"]).is_file()
                for item in router_payload["checkpoints"]
            ))
            for checkpoint_spec in router_payload["checkpoints"]:
                bundle = torch.load(
                    router.parent / checkpoint_spec["checkpoint"],
                    map_location="cpu",
                )
                self.assertEqual(bundle["checkpoint_version"], POINT_FIELD_CHECKPOINT_VERSION)
                self.assertEqual(bundle["full_point_count"], 4)
                self.assertIsNone(bundle["point_indices"])
                self.assertEqual(
                    bundle["sample_material_key"],
                    checkpoint_spec["material_key"],
                )
                torch.testing.assert_close(
                    bundle["model_state"]["weight_cnn"],
                    torch.tensor([material_config.fixed_weight_cnn], dtype=torch.float32),
                )
                torch.testing.assert_close(
                    bundle["model_state"]["weight_lstm"],
                    torch.tensor([material_config.fixed_weight_lstm], dtype=torch.float32),
                )

            metrics=predict_with_material_router(
                material_config,
                router,
                collection_path,
                root/"material_predict",
            )
            self.assertEqual(metrics["model_kind"], MATERIAL_ROUTER_KIND)
            self.assertEqual(metrics["routing_strategy"], "records[].material_key")
            self.assertEqual(set(metrics["materials"]), {"wumu", "steel"})
            for material_key in ("wumu", "steel"):
                with np.load(root/"material_predict"/material_key/"predictions.npz") as routed:
                    self.assertEqual(routed["prediction_temperature_k"].shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
