from __future__ import annotations

import argparse
from hydrosim.config import SystemConfig
from hydrosim.generator import DatasetGenerator, GeneratorSettings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="out_dataset")
    ap.add_argument("--cycles", type=int, default=200)
    ap.add_argument("--p_faulty", type=float, default=0.65)
    args = ap.parse_args()

    cfg = SystemConfig()
    settings = GeneratorSettings(out_dir=args.out, n_cycles=args.cycles, p_faulty=args.p_faulty)

    gen = DatasetGenerator(cfg, settings)
    gen.run()


if __name__ == "__main__":
    main()
