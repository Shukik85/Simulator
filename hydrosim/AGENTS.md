# hydrosim — контекст проекта

## Язык и окружение
- Python 3.14 (придерживаться новой типизации: `type hint`, `|` unions, `match`)
- Основные зависимости: numpy, (чистый Python без сторонних фреймворков)

## Структура проекта
- `mechanics/` — кинематика и динамика механизмов (stateless, numpy)
- `physics/` — гидравлическая модель (HydraulicModel, RK4)
- `config/` — конфигурационные датаклассы
- `*.py` в корне — утилиты (faults, sensors, loads, scenarios, generator)

## Конвенции
- Импорты: `from hydrosim.xxx import yyy` (абсолютные)
- Типы: `Vec2 = np.ndarray`, `Dict[str, float]`
- Углы в радианах
- Все функции кинематики stateless, чистые
- Имена: snake_case для функций/переменных, PascalCase для классов

## Тестирование
- `python -m hydrosim.mechanics.test_excavator` — проверка всей цепочки
