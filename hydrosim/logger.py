from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from pathlib import Path
import json
import h5py
import numpy as np


@dataclass
class CycleMeta:
    cycle_id: int
    mode: str
    duration_s: float
    payload_kg: float
    soil_factor: float
    aggressiveness: float
    faults_by_component: Dict[str, Dict[str, int]]
    faults_flat: Dict[str, int]


class H5Logger:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.h5_path = self.out_dir / "dataset.h5"
        self.meta_path = self.out_dir / "cycles_meta.jsonl"
        self.graph_path = self.out_dir / "graph.json"

        self.h5 = h5py.File(self.h5_path, "w")
        self.grp = self.h5.create_group("cycles")

        self._meta_f = open(self.meta_path, "w", encoding="utf-8")

    def write_graph(self):
        # Важно: node.id совпадает с keys faults_by_component
        graph = {
            "nodes": [
                {"id": "pump", "sensors": ["Ppump", "PLS", "pumpspeed", "Qpump", "Qrelief", "Qopen_center"]},
                {"id": "boom_cyl", "sensors": ["PboomA", "PboomB", "Xboom"]},
                {"id": "arm_cyl", "sensors": ["ParmA", "ParmB", "Xarm"]},
                {"id": "bucket_cyl", "sensors": ["PbucketA", "PbucketB", "Xbucket"]},
                {"id": "swing_motor", "sensors": ["PswingA", "PswingB", "Thetaswing"]},
                {"id": "oil", "sensors": ["Thydraulic"]},
            ],
            "edges": [
                ["pump", "boom_cyl"],
                ["pump", "arm_cyl"],
                ["pump", "bucket_cyl"],
                ["pump", "swing_motor"],
                ["pump", "oil"],
            ],
            "edge_type": "hydraulic_supply",
        }
        self.graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    def log_cycle(self, meta: CycleMeta, timeline: Dict[str, np.ndarray]):
        cid = f"cycle_{meta.cycle_id:06d}"
        g = self.grp.create_group(cid)

        for k, arr in timeline.items():
            g.create_dataset(k, data=np.asarray(arr, dtype=np.float32), compression="gzip", compression_opts=5)

        g.attrs["mode"] = meta.mode
        g.attrs["duration_s"] = meta.duration_s
        g.attrs["payload_kg"] = meta.payload_kg
        g.attrs["soil_factor"] = meta.soil_factor
        g.attrs["aggressiveness"] = meta.aggressiveness
        g.attrs["faults_by_component_json"] = json.dumps(meta.faults_by_component, ensure_ascii=False)
        g.attrs["faults_flat_json"] = json.dumps(meta.faults_flat, ensure_ascii=False)

        self._meta_f.write(json.dumps(meta.__dict__, ensure_ascii=False) + "\n")
        self._meta_f.flush()

    def close(self):
        self._meta_f.close()
        self.h5.close()
