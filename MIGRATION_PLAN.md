# 🔄 MIGRATION PLAN: Design to Implementation

**Status:** Planning Phase  
**Date:** 23 January 2026  
**Objective:** Integrate hydrosim design docs with Simulator implementation repo  

---

## 📊 CURRENT STATE

### Repository: `hydrosim`
✅ **Complete Design Package**
- 15 design documents (5,500+ lines)
- 72 code examples
- 30+ diagrams
- UML architecture
- 16-week roadmap
- **Status:** Ready for implementation
- **Purpose:** Reference & specifications

### Repository: `Simulator`
✅ **Existing Implementation**
- lever_system.py (17,892 bytes) - kinematics/mechanics
- check_h5.py (2,965 bytes) - data utilities
- QUICKSTART.md (8,135 bytes) - setup guide
- /hydrosim directory (structure started)
- /data directory (data storage)
- /scripts directory (utilities)
- /tests directory (testing)
- **Status:** Partial implementation started
- **Purpose:** Active development

---

## 🎯 STRATEGY

### Option A: Keep Separate (RECOMMENDED)
```
https://github.com/Shukik85/hydrosim (Design Specs)
  └─ docs/ (15 documents, 5,500+ lines)
  └─ README.md (overview)
  └─ .gitignore
  └─ DESIGN_DELIVERED.md

https://github.com/Shukik85/Simulator (Implementation)
  └─ hydrosim/ (actual code - Phase 0-7)
  └─ lever_system.py (existing mechanics)
  └─ tests/
  └─ QUICKSTART.md
  ├─ Link to design docs in README
  └─ Reference design in Phase roadmap
```

**Pros:**
- Clear separation: specs ≠ code
- Easy to maintain both
- Reference (hydrosim) stays unchanged
- Implementation (Simulator) stays clean
- Single source of truth for specs

**Cons:**
- Two repos to manage
- Must link between them

### Option B: Merge Into One Repo
```
https://github.com/Shukik85/hydrosim (All-in-one)
  ├─ docs/ (15 design documents)
  ├─ hydrosim/ (implementation code)
  ├─ tests/
  ├─ scripts/
  └─ README.md (points to both)
```

**Pros:**
- Everything in one place
- Single clone
- Easier CI/CD

**Cons:**
- Large repo (docs + code)
- Mixed concerns
- Harder to archive specs

---

## ✅ RECOMMENDED APPROACH (Option A)

### Structure

**Repository 1: `hydrosim` (Design Reference)**
```
https://github.com/Shukik85/hydrosim
├─ docs/
│  ├─ 00_READ_ME_FIRST.txt
│  ├─ 00_START_HERE.md
│  ├─ 01_FOUNDATIONAL_ARCHITECTURE.md
│  ├─ 02_REFERENCE_EXCAVATOR_SPEC.md
│  ├─ 03_MATHEMATICAL_SPECIFICATION.md
│  ├─ 04_SOFTWARE_ARCHITECTURE.md
│  ├─ 05_DEVELOPMENT_ROADMAP.md
│  ├─ QUICK_REFERENCE.md
│  ├─ INDEX.md
│  └─ (+ 5 more summary docs)
├─ README.md (project overview)
├─ DESIGN_DELIVERED.md
└─ .gitignore

Purpose: Professional specifications (5,500+ lines)
Size: ~200 KB (all text)
Frequency: Rarely updated (stable specs)
Access: Public reference
```

**Repository 2: `Simulator` (Implementation)**
```
https://github.com/Shukik85/Simulator
├─ hydrosim/
│  ├─ core/ (Phase 0-1: units, types, oil properties)
│  ├─ fluid/ (Phase 1: Walther, density, E)
│  ├─ hydraulics/ (Phase 2: pump, valve, cylinders)
│  ├─ mechanics/ (Phase 3: kinematics, dynamics)
│  ├─ thermal/ (Phase 4: heat, cooling)
│  ├─ control/ (Phase 5: Load Sensing)
│  ├─ simulator/ (ODE solver)
│  ├─ diagnostics/ (Phase 6: energy analysis)
│  └─ __init__.py
├─ tests/
│  ├─ test_core.py (Phase 0-1 tests)
│  ├─ test_fluid.py (Phase 1 tests)
│  └─ ... (one per phase)
├─ scripts/
│  ├─ run_phase.py (execute specific phase)
│  ├─ benchmark.py (vs CAT 320D)
│  └─ validate.py (energy conservation)
├─ lever_system.py (existing - keep for reference)
├─ check_h5.py (existing - keep for reference)
├─ QUICKSTART.md (updated with link to hydrosim)
├─ README.md (implementation guide)
├─ MIGRATION_PLAN.md (this file)
├─ requirements.txt (Python deps)
├─ setup.py (package setup)
└─ .gitignore

Purpose: Active implementation (code)
Size: ~10-50 MB (code + tests + data)
Frequency: Daily (development)
Access: Development repo
```

---

## 🔗 HOW TO LINK THEM

### In `Simulator/README.md`

```markdown
# Hydraulic Excavator Simulator - Implementation

**Status:** Active Development  
**Design Specs:** https://github.com/Shukik85/hydrosim  

## Quick Start

### 1. Read Design Specification
First, understand the design:
```bash
# Clone design specs
git clone https://github.com/Shukik85/hydrosim.git hydrosim-docs
cd hydrosim-docs

# Read entry point
cat docs/00_READ_ME_FIRST.txt
open docs/README_DESIGN_PACKAGE.md
```

### 2. Setup Implementation
Then, setup this repo for coding:
```bash
# Clone implementation
git clone https://github.com/Shukik85/Simulator.git
cd Simulator

# Create Python environment
python3.10 -m venv venv
source venv/bin/activate

# Install deps
pip install -r requirements.txt

# Run Phase 0 tests
pytest tests/test_core.py -v
```

## Implementation Timeline

See: https://github.com/Shukik85/hydrosim/blob/main/docs/05_DEVELOPMENT_ROADMAP.md

- Phase 0: Foundations (Week 1)
- Phase 1: Oil Properties (Weeks 2-3)
- Phase 2: Hydraulic Core (Weeks 4-6)
- ... (see full roadmap in hydrosim)

## Architecture

See: https://github.com/Shukik85/hydrosim/blob/main/docs/04_SOFTWARE_ARCHITECTURE.md

## Equations & Constants

See: https://github.com/Shukik85/hydrosim/blob/main/docs/QUICK_REFERENCE.md
```

### In Commits

```bash
# Reference design docs in commit messages
git commit -m "[PHASE-0] Implement units module\n\nSee: https://github.com/Shukik85/hydrosim/docs/03_MATHEMATICAL_SPECIFICATION.md\nFollowing Phase 0 from: https://github.com/Shukik85/hydrosim/docs/05_DEVELOPMENT_ROADMAP.md"
```

---

## 📋 ACTION ITEMS

### Step 1: Update `Simulator/README.md`
- [ ] Add link to hydrosim design docs
- [ ] Add quick start section pointing to design
- [ ] Add "Read First" section

### Step 2: Create `Simulator/LINKS.md`
```markdown
# Quick Links to Design Documentation

**Main Design Specs:** https://github.com/Shukik85/hydrosim

## By Document
- Start here: hydrosim/docs/00_READ_ME_FIRST.txt
- Overview: hydrosim/docs/00_START_HERE.md  
- Architecture: hydrosim/docs/04_SOFTWARE_ARCHITECTURE.md
- Equations: hydrosim/docs/03_MATHEMATICAL_SPECIFICATION.md
- Timeline: hydrosim/docs/05_DEVELOPMENT_ROADMAP.md
- Lookup: hydrosim/docs/QUICK_REFERENCE.md

## By Phase
- Phase 0: hydrosim/docs/05_DEVELOPMENT_ROADMAP.md#phase-0
- Phase 1: hydrosim/docs/05_DEVELOPMENT_ROADMAP.md#phase-1
- ... etc
```

### Step 3: Update `Simulator/.gitignore`
- [ ] Ensure it matches Python best practices
- [ ] Add /hydrosim-docs (design reference)
- [ ] Add *.h5, *.pkl (data files)

### Step 4: Create `Simulator/Phase_0_Checklist.md`
```markdown
# Phase 0: Foundations Checklist

Ref: https://github.com/Shukik85/hydrosim/docs/05_DEVELOPMENT_ROADMAP.md#phase-0

## Tasks
- [ ] Create hydrosim/core/__init__.py
- [ ] Implement hydrosim/core/units.py
- [ ] Implement hydrosim/core/types.py
- [ ] Implement hydrosim/core/validation.py
- [ ] Write tests/test_core.py (40+ tests)
- [ ] All tests passing
- [ ] Code coverage > 90%
- [ ] Commit: [PHASE-0] Foundations complete

## Success Criteria
✅ All 40 variables in state vector defined
✅ All units with correct dimensions
✅ 40+ unit tests passing
✅ Type hints on all code
```

---

## 📚 WHY SEPARATE REPOS?

### `hydrosim` = Specs (Reference)
- **Content:** Text specifications, equations, diagrams
- **Size:** ~200 KB (all text, no code)
- **Access:** Read-only (specs shouldn't change during implementation)
- **Audience:** Everyone (managers, architects, developers, QA)
- **Frequency:** Updated when design changes (rare)
- **Purpose:** Single source of truth for design

### `Simulator` = Code (Implementation)
- **Content:** Python code, tests, scripts
- **Size:** 10-50 MB (code, tests, data)
- **Access:** Read-write (active development)
- **Audience:** Developers, QA
- **Frequency:** Daily commits
- **Purpose:** Working implementation

**Separation Benefits:**
1. **Clean:** Code doesn't mix with specs
2. **Focused:** Each repo has one purpose
3. **Archivable:** Specs can be frozen, code evolves
4. **Referenceable:** Easy to link from code to specs
5. **Maintainable:** Easier to review and navigate

---

## 🚀 NEXT STEPS

### Week 1 (This week)
1. ✅ Create hydrosim repo with design docs
2. [ ] Update Simulator README.md
3. [ ] Create Simulator/LINKS.md
4. [ ] Create Simulator/Phase_0_Checklist.md

### Week 2 (Next week)
1. [ ] Begin Phase 0 implementation
2. [ ] Code hydrosim/core/ modules
3. [ ] Write 40+ unit tests
4. [ ] First commit: [PHASE-0] Foundations

---

## 📞 QUESTIONS?

**Q: Should I update both repos every commit?**  
A: No. Update `Simulator` for code. Update `hydrosim` only if design changes (rare).

**Q: Should I copy docs into Simulator?**  
A: No. Link to them instead. Single source of truth.

**Q: Can I clone both?**  
A: Yes, but not necessary. Just link between them.

**Q: What if design changes during implementation?**  
A: Update hydrosim/docs, then update Simulator code accordingly.

---

## ✅ DECISION

**RECOMMENDED:** Keep separate (Option A)
- `https://github.com/Shukik85/hydrosim` = Design Specs (reference)
- `https://github.com/Shukik85/Simulator` = Implementation (code)
- Link between them in READMEs and commits

**Status:** Ready to implement

---

**Next Action:** Update Simulator/README.md with link to hydrosim design docs