# Hydraulic Excavator Simulator for GNN Training — Quick Start Guide

Professional-grade offline hydraulic system simulator for generating realistic training datasets.

## 📋 What You Have

✅ **Core simulator modules:**
- `simulator_config.py` → rename to `simulator/config.py`
- `simulator_physics.py` → rename to `simulator/physics.py`
- `simulator_scenarios.py` → rename to `simulator/scenarios.py`
- `simulator_logger.py` → rename to `simulator/logger.py`
- `simulator_generator.py` → rename to `simulator/generator.py`
- `simulator_init.py` → rename to `simulator/__init__.py`

✅ **Scripts:**
- `run_simulator.py` → place in `scripts/run_simulator.py`
- `run_preprocessor.py` → place in `scripts/run_preprocessor.py`

✅ **Dependencies:**
- `requirements_simulator.txt` (updated, no `tables`)

---

## 🚀 Setup (5 minutes)

### Step 1: Directory Structure

```bash
# From /h/Simulator, create directories:
mkdir -p simulator scripts data/raw/simulator data/processed/simulator

# Move files:
mv simulator_config.py simulator/config.py
mv simulator_physics.py simulator/physics.py
mv simulator_scenarios.py simulator/scenarios.py
mv simulator_logger.py simulator/logger.py
mv simulator_generator.py simulator/generator.py
mv simulator_init.py simulator/__init__.py

mv run_simulator.py scripts/run_simulator.py
mv run_preprocessor.py scripts/run_preprocessor.py

# Verify:
ls -la simulator/
ls -la scripts/
```

### Step 2: Install Dependencies

```bash
python -m pip install -U pip setuptools wheel
pip install -r requirements_simulator.txt
```

Verify h5py works:
```bash
python -c "import h5py; print('✓ h5py OK')"
```

### Step 3: Quick Test (100 cycles)

```bash
# Generate 100 cycles (~1.6 hours)
python scripts/run_simulator.py --cycles 100 --output-dir data/raw/simulator
```

Expected output:
```
======================================================================
Hydraulic Excavator Simulator
======================================================================
Generating 100 cycles across 6 modes
Simulation step: 1.0 ms
Cycle duration: 60.0 s
======================================================================

Progress: 50/100 cycles (50.0%)
Progress: 100/100 cycles (100.0%)

✓ Logging complete. Files saved to data/raw/simulator
  - HDF5: simulation_20260107_145800.h5 (~8 MB)
  - CSV:  simulation_20260107_145800_metadata.csv
```

### Step 4: Inspect Generated Data

```bash
# Quick check
python -c "
import h5py
with h5py.File('data/raw/simulator/simulation_*.h5', 'r') as f:
    print(f'Cycles: {len(f[\"cycles\"])}')
    cycle_0 = f['cycles/cycle_00000']
    print(f'Sensors: {list(cycle_0.keys())}')
    print(f'P_pump shape: {cycle_0[\"P_pump\"].shape}')
"

# View CSV metadata
head data/raw/simulator/*_metadata.csv
```

### Step 5: Preprocess for GNN

```bash
# Convert HDF5 → windowed arrays + labels
python scripts/run_preprocessor.py

# Output:
# data/processed/simulator/windows.npz (100 sensors × 100 timesteps windows)
# data/processed/simulator/labels.csv (cycle modes + payload)
# data/processed/simulator/metadata.txt
```

---

## 📊 Full Dataset (Production)

For realistic training data, generate 1000+ cycles:

```bash
# ~16 hours runtime, ~800 MB disk
python scripts/run_simulator.py --cycles 1000

# Or in background (Linux/macOS):
nohup python scripts/run_simulator.py --cycles 5000 > sim.log 2>&1 &
```

---

## 🎯 Operating Modes

Each cycle simulates a different excavator scenario:

| Mode | Description | Frequency |
|------|-------------|-----------|
| `digging_light` | Loose soil, smooth operator | 15% |
| `digging_medium` | Sandy-clay, typical | 25% |
| `digging_heavy` | Compacted/rocky, aggressive | 20% |
| `swing_only` | Pure rotation (no digging) | 15% |
| `boom_up` | Raising boom (positive load) | 10% |
| `boom_down` | Lowering boom (regeneration) | 10% |
| `combined_cycle` | Full dig→swing→dump→return | 5% |

Generate specific modes:
```bash
python scripts/run_simulator.py --cycles 500 --modes digging_heavy boom_down
```

---

## 📈 Sensor Data

14 sensors at 3 sampling rates (like UCI dataset):

**Pressures (100 Hz):**
- `P_pump`: Main pump outlet (0–350 bar)
- `P_LS`: Load-sensing feedback (0–50 bar)
- `P_boom_A`, `P_boom_B`: Boom cylinder chambers (±50–350 bar)
- `P_arm_A`, `P_arm_B`: Arm cylinder chambers
- `P_bucket_A`, `P_bucket_B`: Bucket cylinder chambers
- `P_swing`: Swing motor supply

**Positions (10 Hz):**
- `X_boom`, `X_arm`, `X_bucket`: Cylinder positions (0–stroke mm)
- `Theta_swing`: Platform rotation (±180°)

**Temperature (1 Hz):**
- `T_hydraulic`: Bulk fluid temperature (0–80 °C)

---

## 🔧 Configuration

Edit `simulator/config.py` before running to customize:

```python
# Increase sensor noise
sensor_config.pressure_noise_pct = 1.0  # 1% of range

# More domain randomization
simulation_config.randomize_payload = True
simulation_config.randomize_soil = True
simulation_config.randomize_operator_style = True

# Adjust physical parameters
hydraulic_config.pump_pressure_relief = 350.0  # bar
mechanical_config.bucket_payload_max = 3000.0  # kg
```

---

## 📦 Integration with GNN Service

Once preprocessed, load data for training:

```python
import numpy as np

# Load preprocessed windows
data = np.load('data/processed/simulator/windows.npz')
windows = data['windows']  # Shape: (n_windows, 14 sensors, 100 timesteps)

# Load labels
import pandas as pd
labels = pd.read_csv('data/processed/simulator/labels.csv')

# Create PyTorch DataLoader (in your GNN service)
from torch.utils.data import DataLoader, TensorDataset
import torch

X = torch.FloatTensor(windows)
y = torch.LongTensor(pd.factorize(labels['mode'])[0])

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_x, batch_y in loader:
    # batch_x: (32, 14, 100)
    # batch_y: mode labels
    pass
```

---

## ⏱️ Performance

| Cycles | Runtime | Disk | Notes |
|--------|---------|------|-------|
| 100 | 1.6 hours | 80 MB | Test |
| 1000 | 16 hours | 800 MB | Standard |
| 5000 | 3–4 days | 4 GB | Large |

Speedups:
- Reduce timestep: `--dt 0.002` (2ms instead of 1ms) = 2x faster
- Disable noise: `sensor_config.pressure_noise_pct = 0`

---

## 🐛 Troubleshooting

**Q: "ModuleNotFoundError: No module named 'simulator'"**
- Ensure `simulator/__init__.py` exists and `cd` to `/h/Simulator` before running

**Q: HDF5 file grows large**
- Already compressed with gzip by default
- Reduce `cycle_duration` in config.py

**Q: Simulation looks "too clean"**
- Increase noise in config:
  ```python
  sensor_config.pressure_noise_pct = 2.0  # 2% of range
  ```

**Q: Slow simulation**
- Use larger timestep: `--dt 0.005`
- Disable sensor noise generation
- Run on SSD (not network drive)

---

## 📚 Files Structure After Setup

```
/h/Simulator/
├── simulator/
│   ├── __init__.py
│   ├── config.py
│   ├── physics.py
│   ├── scenarios.py
│   ├── logger.py
│   └── generator.py
├── scripts/
│   ├── run_simulator.py
│   └── run_preprocessor.py
├── data/
│   ├── raw/simulator/
│   │   ├── simulation_20260107_145800.h5
│   │   └── simulation_20260107_145800_metadata.csv
│   └── processed/simulator/
│       ├── windows.npz
│       ├── labels.csv
│       └── metadata.txt
├── requirements_simulator.txt
└── SIMULATOR_README.md
```

---

## 🎓 Next Steps

1. **Generate baseline data:**
   ```bash
   python scripts/run_simulator.py --cycles 1000
   ```

2. **Preprocess for GNN:**
   ```bash
   python scripts/run_preprocessor.py
   ```

3. **Integrate with `gnn_service`:**
   - Copy `data/processed/simulator/` to your ML pipeline
   - Load in `services/gnn_service/data_loader.py`
   - Train Universal Temporal GNN

4. **Sensor dropout experiments:**
   - Mask sensors in config: `sensor_config.sensor_presence_mask['P_pump'] = False`
   - Regenerate to create ablation dataset

---

**Questions?** Check `SIMULATOR_README.md` for detailed physics, configuration, and references.
