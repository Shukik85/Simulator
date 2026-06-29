---
name: excavator-simulation-v2
---
Use when working on hydrosim_v2 – a clean LS-based excavator simulator for
synthetic ML data generation. Covers kinematics (forward/backward), 4-bar
bucket linkage, LS axial-piston pump, valve compensators, cylinder dynamics,
fault simulation, digging load model, and data export for GNN training.
---

# hydrosim_v2 — LS excavator simulator

## Architecture

```
config/       — dataclass configs (mechanics + LS hydraulics), YAML load/save
core/         — Vec2, Vec3, unit converters
kinematics/   — forward_kinematics, backward_static, bucket_lever solver
hydraulics/   — LS pump, valve section with compensator, cylinder, system RHS
faults/       — fault definitions (continuous parameters, multi-label)
data/         — data generation, sensor model, export
```

## Key differences from v1
- No swing, no travel — only boom/arm/bucket
- LS (load-sensing) hydraulics instead of open-center
- Axial piston pump with swash plate control
- LS compensator per valve section (ΔP_LS = 14–20 bar)
- Multi-label faults: pump + 3 cylinders

## Units
- Pressure: Pa (config in bar: 1 bar = 1e5 Pa)
- Flow: m³/s (config in LPM)
- Length: m
- Force: N
- Angle: rad

## Testing
```bash
python -m pytest hydrosim_v2/tests/
```
