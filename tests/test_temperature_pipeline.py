from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai_model.data_process.dataset import AITemperatureDataset, split_case_records, write_case_split_files
from ai_model.data_process.mesh_sampling import deterministic_sample
from ai_model.data_process.temperature_field import CaseKey, select_waveform, validate_temperature_field
from ai_model.config import AIModelConfig
from ai_model.model.point_field import DirectPointFieldModel, weighted_temperature_loss
import json
import torch


class TemperatureContractTests(unittest.TestCase):
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
            np.savez_compressed(root/"sampling_index.npz", node_ids=node_ids, coordinates_m=coords,
                material_ids=np.ones(4,np.int32), interface_side=np.zeros(4,np.int32), sample_weights=np.ones(4,np.float32),
                metadata_json=np.array(json.dumps({"sampling_version":"test","source_mesh_fingerprint":"abc"})))
            records=[]
            for i,value in enumerate((300.,400.,500.)):
                np.save(root/"waveforms"/f"w{i}.npy",np.linspace(0,1,16,dtype=np.float32))
                np.savez_compressed(root/"temperature_fields"/f"f{i}.npz",temperature_k=(value+np.arange(4)).astype(np.float32),
                    coordinates_m=coords,node_ids=node_ids,material_ids=np.ones(4,np.int32),interface_side=np.zeros(4,np.int32),
                    sample_weights=np.ones(4,np.float32),temperature_unit="K",coordinate_unit="m")
                records.append({"sample_id":f"s{i}","case_key":f"worker_01/case_{i:04d}_T0{i}K","source":"experiment_case",
                    "waveform_path":f"waveforms/w{i}.npy","field_path":f"temperature_fields/f{i}.npz",
                    "temperature_k":value,"material_key":"case_mesh","dimension":"2d","mode":"steady"})
            manifest=root/"manifest.json"; manifest.write_text(json.dumps({"schema_version":1,"sampling_index":"sampling_index.npz",
                "dataset_name":"test","records":records}),encoding="utf-8")
            split=write_case_split_files(manifest,seed=2,train_ratio=.34,validation_ratio=.33)
            ds=AITemperatureDataset(split["manifests"]["train"],AIModelConfig(data_root=root/"db",result_root=root/"out"))
            item=ds[0]; self.assertEqual(tuple(item["field"].shape),(4,)); self.assertEqual(float(item["field_mask"]),1.0)
            model=DirectPointFieldModel(4,hidden_dim=4,latent_dim=4,chunk_size=2)
            prediction=model(item["waveform"].unsqueeze(0)); self.assertEqual(tuple(prediction.shape),(1,4))
            loss=weighted_temperature_loss(prediction,item["field"].unsqueeze(0),item["sample_weights"].unsqueeze(0))
            self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
