#!/usr/bin/env python
"""
Preprocess HDF5 cycles into windowed dataset for GNN training

Converts raw simulation data from HDF5 into:
- Sliding windows (temporal chunks)
- Normalized per-sensor features
- Graph adjacency matrices
- PyTorch DataLoader-compatible format

Usage:
    python scripts/run_preprocessor.py --input data/raw/simulator/simulation_20260107_115800.h5
    python scripts/run_preprocessor.py  # Auto-find latest .h5
"""

import sys
import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.config import SensorConfig, DataLoaderConfig


class HydraulicDataPreprocessor:
    """Preprocess HDF5 cycles into windowed GNN-ready dataset."""
    
    def __init__(self, input_h5: Path, output_dir: Path,
                 sensor_config: SensorConfig = None,
                 loader_config: DataLoaderConfig = None):
        self.input_h5 = Path(input_h5)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sensor_config = sensor_config or SensorConfig()
        self.loader_config = loader_config or DataLoaderConfig()
        
        self.sensor_names = list(self.sensor_config.sensors.keys())
        print(f"✓ Sensors to process: {self.sensor_names}")
    
    def process(self):
        """Run full preprocessing pipeline."""
        
        print(f"\nOpening HDF5: {self.input_h5}")
        with h5py.File(self.input_h5, 'r') as h5_file:
            cycles_group = h5_file['cycles']
            num_cycles = len(cycles_group)
            
            print(f"Found {num_cycles} cycles")
            
            # Process each cycle
            all_windows = []
            all_labels = []
            
            for cycle_idx in range(num_cycles):
                cycle_key = f'cycle_{cycle_idx:05d}'
                cycle_group = cycles_group[cycle_key]
                
                # Load sensor data
                data_dict = {}
                for sensor_name in self.sensor_names:
                    if sensor_name in cycle_group:
                        data_dict[sensor_name] = np.array(cycle_group[sensor_name])
                
                if not data_dict:
                    continue
                
                # Get metadata
                mode = cycle_group.attrs.get('mode', 'unknown')
                payload = cycle_group.attrs.get('payload', 0)
                
                # Resample to common rate (100 Hz)
                resampled = self._resample_to_100hz(data_dict, cycle_group)
                
                # Create sliding windows
                windows = self._create_windows(resampled)
                
                # Create labels
                for _ in windows:
                    all_labels.append({
                        'cycle_id': cycle_idx,
                        'mode': mode,
                        'payload_kg': payload
                    })
                
                all_windows.extend(windows)
                
                if (cycle_idx + 1) % 100 == 0:
                    print(f"  Processed {cycle_idx + 1}/{num_cycles} cycles, "
                          f"Total windows: {len(all_windows)}")
        
        # Convert to arrays
        windows_array = np.array(all_windows)  # Shape: (n_windows, n_sensors, window_size)
        labels_df = pd.DataFrame(all_labels)
        
        # Normalize if configured
        if self.loader_config.normalize:
            windows_array = self._normalize(windows_array)
        
        # Save outputs
        self._save_data(windows_array, labels_df)
    
    def _resample_to_100hz(self, data_dict: dict, cycle_group) -> dict:
        """Resample all sensors to common 100 Hz rate."""
        
        # Get time vector from highest frequency sensor (100 Hz)
        time_100hz = np.array(cycle_group['time'])  # Assumed 100 Hz base
        
        resampled = {}
        for sensor_name, signal in data_dict.items():
            sensor_info = self.sensor_config.sensors[sensor_name]
            sensor_freq = sensor_info['freq']
            
            if sensor_freq == 100:
                # Already 100 Hz
                resampled[sensor_name] = signal
            elif sensor_freq == 10:
                # Upsample 10 Hz → 100 Hz (repeat each sample 10 times)
                resampled[sensor_name] = np.repeat(signal, 10)
            elif sensor_freq == 1:
                # Upsample 1 Hz → 100 Hz (repeat each sample 100 times)
                resampled[sensor_name] = np.repeat(signal, 100)
            else:
                # Generic linear interpolation
                old_t = np.linspace(0, 1, len(signal))
                new_t = np.linspace(0, 1, len(time_100hz))
                resampled[sensor_name] = np.interp(new_t, old_t, signal)
        
        return resampled
    
    def _create_windows(self, resampled: dict) -> list:
        """Create sliding windows from resampled data."""
        
        window_size = self.loader_config.window_size
        stride = self.loader_config.window_stride
        
        # All sensors should have same length after resampling
        data_length = len(next(iter(resampled.values())))
        
        windows = []
        for start_idx in range(0, data_length - window_size, stride):
            end_idx = start_idx + window_size
            
            window = []
            for sensor_name in self.sensor_names:
                signal = resampled[sensor_name]
                window_data = signal[start_idx:end_idx]
                window.append(window_data)
            
            windows.append(np.array(window))
        
        return windows
    
    def _normalize(self, windows_array: np.ndarray) -> np.ndarray:
        """Normalize windows per sensor (axis=2, across time and cycles)."""
        
        n_sensors, n_windows, window_size = windows_array.shape
        windows_reshaped = windows_array.reshape(n_sensors, -1)
        
        if self.loader_config.normalization_type == 'minmax':
            min_vals = windows_reshaped.min(axis=1, keepdims=True)
            max_vals = windows_reshaped.max(axis=1, keepdims=True)
            windows_reshaped = (windows_reshaped - min_vals) / (max_vals - min_vals + 1e-8)
        
        elif self.loader_config.normalization_type == 'zscore':
            mean_vals = windows_reshaped.mean(axis=1, keepdims=True)
            std_vals = windows_reshaped.std(axis=1, keepdims=True)
            windows_reshaped = (windows_reshaped - mean_vals) / (std_vals + 1e-8)
        
        return windows_reshaped.reshape(n_sensors, n_windows, window_size)
    
    def _save_data(self, windows_array: np.ndarray, labels_df: pd.DataFrame):
        """Save preprocessed data."""
        
        # Save windows as NPZ (efficient binary)
        windows_path = self.output_dir / "windows.npz"
        np.savez_compressed(windows_path, windows=windows_array)
        print(f"✓ Saved windows: {windows_path} ({windows_array.shape})")
        
        # Save labels as CSV
        labels_path = self.output_dir / "labels.csv"
        labels_df.to_csv(labels_path, index=False)
        print(f"✓ Saved labels: {labels_path}")
        
        # Save metadata
        metadata = {
            'n_windows': windows_array.shape[0],
            'n_sensors': windows_array.shape[1],
            'window_size': windows_array.shape[2],
            'window_size_seconds': windows_array.shape[2] / 100.0,
            'sensors': self.sensor_names,
            'normalized': self.loader_config.normalize,
            'normalization_type': self.loader_config.normalization_type,
        }
        
        metadata_path = self.output_dir / "metadata.txt"
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        print(f"✓ Saved metadata: {metadata_path}")


def find_latest_h5(search_dir: Path = None) -> Path:
    """Find the most recent HDF5 file in raw simulator directory."""
    
    if search_dir is None:
        search_dir = Path("data/raw/simulator")
    
    search_dir = Path(search_dir)
    h5_files = sorted(search_dir.glob("simulation_*.h5"))
    
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {search_dir}")
    
    latest = h5_files[-1]
    print(f"Found latest HDF5: {latest}")
    return latest


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess HDF5 simulation data into windowed GNN-ready dataset"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input HDF5 file (auto-detect if not provided)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/simulator",
        help="Output directory for processed data (default: data/processed/simulator)"
    )
    
    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Window size in timesteps (default: 100 = 1 second at 100 Hz)"
    )
    
    parser.add_argument(
        "--window-stride",
        type=int,
        default=50,
        help="Stride between windows (default: 50 = 50%% overlap)"
    )
    
    args = parser.parse_args()
    
    # Find input file
    if args.input:
        input_h5 = Path(args.input)
    else:
        input_h5 = find_latest_h5()
    
    if not input_h5.exists():
        print(f"Error: Input file not found: {input_h5}")
        sys.exit(1)
    
    try:
        print(f"\n{'='*70}")
        print(f"Hydraulic Simulator - Data Preprocessing")
        print(f"{'='*70}\n")
        
        preprocessor = HydraulicDataPreprocessor(
            input_h5=input_h5,
            output_dir=Path(args.output_dir)
        )
        
        preprocessor.loader_config.window_size = args.window_size
        preprocessor.loader_config.window_stride = args.window_stride
        
        preprocessor.process()
        
        print(f"\n✓ Preprocessing complete!")
        print(f"Output directory: {args.output_dir}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
