from __future__ import annotations

from typing import Dict
from pathlib import Path
import json
import h5py
import numpy as np


class CycleMeta:
    def __init__(
        self,
        cycle_id: int,
        mode: str,
        duration_s: float,
        payload_kg: float = 0.0,
        soil_factor: float = 1.0,
        aggressiveness: float = 0.5,
        faults_flat: Dict[str, int] | None = None,
    ) -> None:
        self.cycle_id = cycle_id
        self.mode = mode
        self.duration_s = duration_s
        self.payload_kg = payload_kg
        self.soil_factor = soil_factor
        self.aggressiveness = aggressiveness
        self.faults_flat = faults_flat or {}


class H5Logger:
    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.h5_path = self.out_dir / "dataset.h5"
        self.meta_path = self.out_dir / "cycles_meta.jsonl"
        self.graph_path = self.out_dir / "graph.json"

        self.h5 = h5py.File(self.h5_path, "w")
        self.grp = self.h5.create_group("cycles")

        self._meta_f = open(self.meta_path, "w", encoding="utf-8")

    def write_graph(self) -> None:
        graph = {
            "nodes": [
                {"id": "pump", "sensors": ["p_pump", "p_ls", "q_pump", "q_relief"]},
                {"id": "boom_cyl", "sensors": ["p_boom_a", "p_boom_b", "x_boom"]},
                {"id": "arm_cyl", "sensors": ["p_arm_a", "p_arm_b", "x_arm"]},
                {"id": "bucket_cyl", "sensors": ["p_bucket_a", "p_bucket_b", "x_bucket"]},
            ],
            "edges": [
                ["pump", "boom_cyl"],
                ["pump", "arm_cyl"],
                ["pump", "bucket_cyl"],
            ],
            "edge_type": "hydraulic_supply",
        }
        self.graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    def log_cycle(self, meta: CycleMeta, timeline: Dict[str, np.ndarray]) -> None:
        cid = f"cycle_{meta.cycle_id:06d}"
        g = self.grp.create_group(cid)

        for k, arr in timeline.items():
            g.create_dataset(k, data=np.asarray(arr, dtype=np.float32), compression="gzip", compression_opts=5)

        g.attrs["mode"] = meta.mode
        g.attrs["duration_s"] = meta.duration_s
        g.attrs["payload_kg"] = meta.payload_kg
        g.attrs["soil_factor"] = meta.soil_factor
        g.attrs["aggressiveness"] = meta.aggressiveness
        g.attrs["faults_flat_json"] = json.dumps(meta.faults_flat, ensure_ascii=False)

        self._meta_f.write(json.dumps(meta.__dict__, ensure_ascii=False) + "\n")
        self._meta_f.flush()

    def close(self) -> None:
        self._meta_f.close()
        self.h5.close()
