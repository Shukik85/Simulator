# Import policy (no legacy)

This branch aims for a clean architecture without any legacy module usage or mention.

## Rule: resolve ambiguous names by semantics

Some names are historically overloaded (e.g. `MechanicsConfig`). When updating imports:

- If the code needs **mechanism / geometry** data (link lengths, cylinder geometry, kinematics helpers), import from:
  - `hydrosim.config.mechanics`

- If the code needs **system-level physical parameters** (fluid, pump, valve bank, cylinders as hydraulic components, thermal, sensors, simulation defaults), import from:
  - `hydrosim.config.models`

## Canonical imports

- `SystemConfig`:
  - `from hydrosim.config.models import SystemConfig`

- Geometry config:
  - `from hydrosim.config.mechanics import MechanicsConfig, DEFAULT_MECHANICS_CONFIG`

- Mass/weight defaults (legacy-style "mechanics" parameters):
  - `from hydrosim.config.models import MassPropertiesConfig`

## What is forbidden

- Importing `SystemConfig` via `hydrosim.config` re-exports.
- Referring to or loading `hydrosim/config.py` or `hydrosim/physics.py` (flat legacy modules).

## Notes

This repository contains both:
- `hydrosim/config/mechanics.py` (geometry), and
- `hydrosim/config/models.py` (system configs)

Keep them separate to avoid cyclic dependencies and semantic confusion.
