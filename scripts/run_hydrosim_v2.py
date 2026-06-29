#!/usr/bin/env python
"""
Hydrosim_v2 CLI entrypoint: запуск симуляции экскаватора
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from hydrosim_v2.config import DEFAULT_MECHANICS_CONFIG
from hydrosim_v2.kinematics import forward_kinematics


def parse_args():
    parser = argparse.ArgumentParser(description="Hydrosim_v2: запуск forward-кинематики")
    parser.add_argument("--boom", type=float, default=2.0, help="Длина стрелы (м)")
    parser.add_argument("--arm", type=float, default=2.3, help="Длина рукояти (м)")
    parser.add_argument("--bucket", type=float, default=1.8, help="Длина ковша (м)")
    return parser.parse_args()


def main():
    args = parse_args()
    cyl_lengths = {
        "boom_cyl": args.boom,
        "arm_cyl": args.arm,
        "bucket_cyl": args.bucket,
    }
    cfg = DEFAULT_MECHANICS_CONFIG
    pts = forward_kinematics(cfg, cyl_lengths)
    print("\nРезультаты forward-кинематики:")
    for name, pos in pts.items():
        print(f"  {name}: {np.round(pos, 4)}")

if __name__ == "__main__":
    main()
