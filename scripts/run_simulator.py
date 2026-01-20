#!/usr/bin/env python
"""
Run Hydraulic Simulator: Generate cycles and save to HDF5/CSV

Usage:
    python scripts/run_simulator.py [--cycles 1000] [--modes digging_light digging_medium]
    python -m simulator.generator  # Alternative (if installed as package)

Output:
    data/raw/simulator/simulation_YYYYMMDD_HHMMSS.h5 — Raw cycle data
    data/raw/simulator/simulation_YYYYMMDD_HHMMSS_metadata.csv — Summaries
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.generator import CycleGenerator
from simulator.config import SimulationConfig


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate hydraulic excavator simulation cycles for GNN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 cycles (quick test)
  python scripts/run_simulator.py --cycles 100

  # Generate 1000 cycles with specific modes
  python scripts/run_simulator.py --cycles 1000 --modes digging_light digging_medium digging_heavy

  # Full dataset (all modes)
  python scripts/run_simulator.py --cycles 5000
        """
    )
    
    parser.add_argument(
        "--cycles",
        type=int,
        default=1000,
        help="Number of cycles to generate (default: 1000)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/simulator",
        help="Output directory for HDF5 and CSV files (default: data/raw/simulator)"
    )
    
    parser.add_argument(
        "--modes",
        nargs="*",
        default=None,
        help="Operating modes to generate (default: all). Options: digging_light, digging_medium, digging_heavy, swing_only, boom_up, boom_down, combined_cycle, idle"
    )
    
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Simulation timestep in seconds (default: 0.01)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate cycles
    if args.cycles < 1:
        print("Error: --cycles must be >= 1")
        sys.exit(1)
    
    # Validate modes if provided
    if args.modes:
        valid_modes = {
            "digging_light", "digging_medium", "digging_heavy",
            "swing_only", "boom_up", "boom_down", "combined_cycle", "idle"
        }
        invalid_modes = set(args.modes) - valid_modes
        if invalid_modes:
            print(f"Error: Invalid modes: {invalid_modes}")
            print(f"Valid modes: {', '.join(valid_modes)}")
            sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Hydraulic Excavator Simulator - Cycle Generation")
    print(f"{'='*70}")
    print(f"Cycles to generate: {args.cycles}")
    print(f"Output directory: {output_dir}")
    print(f"Timestep: {args.dt*1000:.1f} ms")
    if args.modes:
        print(f"Modes: {', '.join(args.modes)}")
    print(f"{'='*70}\n")
    
    try:
        # Create generator
        generator = CycleGenerator(
            output_dir=str(output_dir),
            num_cycles=args.cycles
        )
        
        # Adjust timestep if provided
        if args.dt != 0.001:
            generator.dt = args.dt
        
        # Run simulation
        generator.run(
            num_cycles=args.cycles,
            modes=args.modes or None
        )
        
        print(f"\n✓ Simulation completed successfully!")
        print(f"Output files:")
        print(f"  - {output_dir}/simulation_*.h5")
        print(f"  - {output_dir}/simulation_*_metadata.csv")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
