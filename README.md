# Simulator (Implementation)

Это репозиторий **реализации** симулятора (код, тесты, данные, эксперименты).  
Дизайн/спецификация живёт отдельно в репозитории **`hydrosim`**: [Shukik85/hydrosim](https://github.com/Shukik85/hydrosim).  

## Читать сначала

1. Дизайн-пакет (точка входа): [docs/00_READ_ME_FIRST.txt](https://github.com/Shukik85/hydrosim/blob/main/docs/00_READ_ME_FIRST.txt)
2. Роли и пути чтения: [docs/README_DESIGN_PACKAGE.md](https://github.com/Shukik85/hydrosim/blob/main/docs/README_DESIGN_PACKAGE.md)
3. Roadmap фаз: [docs/05_DEVELOPMENT_ROADMAP.md](https://github.com/Shukik85/hydrosim/blob/main/docs/05_DEVELOPMENT_ROADMAP.md)
4. Математика/формулы: [docs/03_MATHEMATICAL_SPECIFICATION.md](https://github.com/Shukik85/hydrosim/blob/main/docs/03_MATHEMATICAL_SPECIFICATION.md)
5. Архитектура ПО: [docs/04_SOFTWARE_ARCHITECTURE.md](https://github.com/Shukik85/hydrosim/blob/main/docs/04_SOFTWARE_ARCHITECTURE.md)

Короткие ссылки собраны в [LINKS.md](./LINKS.md).

## Быстрый старт (локально)

```bash
git clone https://github.com/Shukik85/Simulator.git
cd Simulator

python3 -m venv venv
source venv/bin/activate

pip install -U pip
pip install -r requirements_simulator.txt
pip install pytest

pytest -q
```

## Phase 0 (начато)

Phase 0 добавляет базовый слой ядра: `hydrosim/core` (единицы, типы, валидация) и минимальные тесты.  
Дальше код будем развивать по фазам из roadmap (см. ссылки выше).
