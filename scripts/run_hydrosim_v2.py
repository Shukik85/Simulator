#!/usr/bin/env python
"""
Hydrosim_v2 CLI: forward-кинематика или генерация датасета с live-графиками.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from hydrosim_v2.config import (
    ExcavatorConfig, DEFAULT_MECHANICS_CONFIG, LSConfig,
)
from hydrosim_v2.kinematics import forward_kinematics
from hydrosim_v2.data.generator import DatasetGenerator


def parse_args():
    p = argparse.ArgumentParser(description="Hydrosim v2")
    sp = p.add_subparsers(dest="mode", required=True)

    fk = sp.add_parser("fk", help="Forward kinematics")
    fk.add_argument("--boom", type=float, default=2.0)
    fk.add_argument("--arm", type=float, default=2.3)
    fk.add_argument("--bucket", type=float, default=1.8)

    gen = sp.add_parser("generate", help="Generate dataset")
    gen.add_argument("--cycles", type=int, default=200)
    gen.add_argument("--out", type=str, default="out_dataset")
    gen.add_argument("--no-plot", action="store_true", help="Disable live plotting")

    manual = sp.add_parser("manual", help="Manual control (numpad/arrows)")
    manual.add_argument("--out", type=str, default=None, help="Optional HDF5 output dir")

    return p.parse_args()


def cmd_fk(args):
    cyl_lengths = {
        "boom_cyl": args.boom,
        "arm_cyl": args.arm,
        "bucket_cyl": args.bucket,
    }
    pts = forward_kinematics(DEFAULT_MECHANICS_CONFIG, cyl_lengths)
    print("\nForward kinematics:")
    for name, pos in pts.items():
        print(f"  {name}: {np.round(pos, 4)}")


def cmd_generate(args):
    cfg = ExcavatorConfig(mechanics=DEFAULT_MECHANICS_CONFIG, hydraulics=LSConfig())
    gen = DatasetGenerator(
        cfg,
        out_dir=args.out,
        n_cycles=args.cycles,
        live_plot=not args.no_plot,
    )
    gen.run()


def cmd_manual(args):
    cfg = ExcavatorConfig(mechanics=DEFAULT_MECHANICS_CONFIG, hydraulics=LSConfig())
    out_dir = args.out or "out_manual"
    gen = DatasetGenerator(
        cfg,
        out_dir=out_dir,
        live_plot=True,
        live_control=True,
    )
    gen.run_manual()


def main():
    args = parse_args()
    if args.mode == "fk":
        cmd_fk(args)
    elif args.mode == "manual":
        cmd_manual(args)
    else:
        cmd_generate(args)


if __name__ == "__main__":
    main()
