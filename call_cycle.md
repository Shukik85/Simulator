# Цикл вызовов при запуске генератора данных

Этот документ описывает последовательность вызовов и обращений, начиная с запуска скрипта `generate_dataset.py` до завершения генерации одного цикла и записи данных.

## 1. Запуск скрипта

```python
# hydrosim/generate_dataset.py
if __name__ == "__main__":
    main()
```

## 2. Обработка аргументов командной строки

```python
# hydrosim/generate_dataset.py
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="out_dataset")
    ap.add_argument("--cycles", type=int, default=200)
    ap.add_argument("--p_faulty", type=float, default=0.65)
    args = ap.parse_args()
```

## 3. Создание конфигурации и настроек генератора

```python
# hydrosim/generate_dataset.py
cfg = SystemConfig()
settings = GeneratorSettings(out_dir=args.out, n_cycles=args.cycles, p_faulty=args.p_faulty)
```

## 4. Инициализация генератора данных

```python
# hydrosim/generate_dataset.py
gen = DatasetGenerator(cfg, settings)
```

Конструктор `DatasetGenerator.__init__` создаёт и инициализирует все необходимые компоненты:

- `HydraulicModel(cfg)` - модель гидравлической системы
- `ScenarioGenerator(cfg, self.rng)` - генератор сценариев операций
- `LoadModel(cfg, self.rng)` - модель внешних нагрузок
- `SensorModel(cfg, self.rng)` - модель датчиков
- `H5Logger(settings.out_dir)` - логгер для записи данных
- `logger.write_graph()` - запись графа зависимостей в JSON

## 5. Запуск генерации данных

```python
# hydrosim/generate_dataset.py
gen.run()
```

## 6. Основной цикл генерации циклов

```python
# hydrosim/generator.py
def run(self):
    dt = self.cfg.sim.dt
    steps = int(self.cfg.sim.cycle_duration_s / dt)

    s = State()  # Начальное состояние
    # ... случайная температура масла ...

    for cid in range(self.settings.n_cycles):
        # Генерация одного цикла
```

## 7. Генерация профиля сценария

```python
# hydrosim/generator.py
mode = self._sample_mode()
prof = self.scenarios.sample_profile(mode)
```

```python
# hydrosim/scenarios.py
class ScenarioGenerator:
    def sample_profile(self, mode: str) -> ScenarioProfile:
        # Случайная масса полезного груза
        payload = float(self.rng.uniform(0.0, mech.payload_max)) if sim.randomize_payload else 0.0
        # Случайная агрессивность оператора
        aggress = float(self.rng.uniform(0.35, 0.95))
        # Случайный фактор сопротивления грунта
        soil_factor = float(1.0 + self.rng.uniform(-soil.randomness, soil.randomness)) if sim.randomize_soil else 1.0
        # ... возвращение ScenarioProfile с параметрами ...
```

## 8. Генерация неисправностей

```python
# hydrosim/generator.py
faults = FaultConfig.sample(self.rng, p_any=self.settings.p_faulty)
```

```python
# hydrosim/faults.py
@staticmethod
def sample(rng: np.random.Generator, p_any: float = 0.65) -> "FaultConfig":
    if rng.random() > p_any:
        return FaultConfig()  # Без неисправностей
    # Генерация различных неисправностей с определённой вероятностью
    pump_wear = float(rng.beta(2, 6)) if rng.random() < 0.45 else 0.0
    # ... другие неисправности ...
    return FaultConfig(...)
```

## 9. Основной временной цикл (интеграция)


```python
# hydrosim/generator.py
for i in range(steps):
    t = i * dt
    time[i] = t

    u = self.scenarios.command(prof, t)  # Команда управления
    ext = self.loads.external_loads(s, prof, u)  # Внешние нагрузки

    s, diag = self.model.rk4_step(s, u, ext, dt, faults)  # Интеграция

    rpm = float(u.get("pumpspeed", 0.0))
    obs = self.sensors.observe(s, rpm=rpm, diag=diag, faults=faults)  # Наблюдения с датчиков

    # Сохранение наблюдений
    for k in obs_buf.keys():
        obs_buf[k][i] = obs.get(k, np.nan)
```

### 9.1. Генерация команд управления

```python
# hydrosim/scenarios.py
def command(self, prof: ScenarioProfile, t: float) -> Dict[str, float]:
    T = max(prof.duration_s, 1e-9)
    tnorm = float(np.clip(t / T, 0.0, 1.0))
    ag = prof.aggressiveness
    u = {"boom": 0.0, "arm": 0.0, "bucket": 0.0, "swing": 0.0, "pumpspeed": 1800.0}

    if prof.mode.startswith("digging"):
        if tnorm < 0.18:
            # Фаза заглубления
            u["boom"] = -lerp(0.15, 0.45, k) - 0.15 * ag
            u["arm"] = -lerp(0.10, 0.55, k) - 0.10 * ag
            u["bucket"] = +lerp(0.10, 0.35, k) + 0.10 * ag
            u["pumpspeed"] = self._pumpspeed(2100.0, ag, tnorm)
        # ... другие фазы ...
    # ... другие режимы ...

    # Финальный клип значений
    for k in ("boom", "arm", "bucket", "swing"):
        u[k] = float(np.clip(u[k], -1.0, 1.0))
    u["pumpspeed"] = float(np.clip(u["pumpspeed"], 800.0, self.cfg.pump.max_speed_rpm))
    return u
```

### 9.2. Расчёт внешних нагрузок

```python
# hydrosim/loads.py
def external_loads(self, s: State, prof: ScenarioProfile, u: Dict[str, float]) -> Dict[str, float]:
    # Гравитационные нагрузки (зависят от положения)
    xb = s.xboom / max(self.cfg.boom_cyl.stroke_m, 1e-6)
    loads["boom"] += mech.boom_mass * g * (0.4 + 0.6 * (1.0 - xb))
    # ... другие звенья ...

    # Нагрузка от копания (если режим копания)
    if prof.mode.startswith("digging") or prof.mode == "combined":
        penetration = float(np.clip(xk, 0.0, 1.0))
        Fsoil = rnd * prof.soil_factor * (soil.base_resistance_N + soil.penetration_gain_N * penetration)
        Fvel = soil.vel_gain_N_per_m_s * abs(s.vbucket)
        Fpayload = prof.payload_kg * g
        loads["bucket"] += (Fsoil + Fvel + 0.25 * Fpayload)
        # Реакция на другие звенья
        loads["arm"] += 0.35 * loads["bucket"]
        loads["boom"] += 0.20 * loads["bucket"]
    # ... другие нагрузки ...
    return loads
```

### 9.3. Интеграция шага модели

```python
# hydrosim/physics.py
def rk4_step(self, s: State, u: Dict[str, float], ext: Dict[str, float], dt: float, faults: FaultConfig) -> Tuple[State, FlowDiagnostics]:
    k1, d1 = self.rhs(s, u, ext, faults)
    s2 = self._add(s, k1, 0.5 * dt)
    k2, _ = self.rhs(s2, u, ext, faults)
    s3 = self._add(s, k2, 0.5 * dt)
    k3, _ = self.rhs(s3, u, ext, faults)
    s4 = self._add(s, k3, dt)
    k4, _ = self.rhs(s4, u, ext, faults)

    # ... вычисление нового состояния ...
    # ... ограничения по физике ...
    # ... пересчёт Ppump ...

    return ns, diag
```

#### 9.3.1. Расчёт правой части системы ОДУ (rhs)


```python
# hydrosim/physics.py
def rhs(self, s: State, u: Dict[str, float], ext: Dict[str, float], faults: FaultConfig) -> Tuple[State, FlowDiagnostics]:
    # Целевое давление LS
    # ...

    # Расчёт расхода насоса
    Qpump = self.pump_flow(rpm, s.Ppump, faults)
    # ...
    # Предварительный расчёт запросов расхода
    qreq = {sec: self.req_flow_at_dp(sec, u.get(sec, 0.0), faults=faults) for sec in self.SECTIONS}
    qalloc = self.allocate_flow_sharing(qreq, Qpump)
    # ...
    # Расчёт Ppump через бисекцию
    # ...
    # Расчёт реальных расходов по секциям
    # ...
    # Обновление состояния цилиндров (давление, скорость)
    for sec in ("boom", "arm", "bucket"):
        # ...
        dPA = (self.K / VA) * (QA - Ahead * v - Qleak)
        dPB = (self.K / VB) * (QB + Aann * v + Qleak)
        # ...
    # ...
    # Обновление состояния поворотного механизма
    # ...
    # Обновление температуры масла
    # ...
    return ds, diag
```

### 9.4. Моделирование показаний датчиков

```python
# hydrosim/sensors.py
def observe(self, s: State, rpm: float, diag: FlowDiagnostics, faults: FaultConfig) -> Dict[str, float]:
    # Преобразование единиц
    out["Ppump"] = s.Ppump / PA_PER_BAR  # Pa -> bar
    out["Xboom"] = s.xboom * 1000.0  # m -> mm
    out["pumpspeed"] = float(rpm)
    out["Qpump"] = diag.Qpump * 60.0 * 1000.0  # m3/s -> lpm

    # Добавление шума
    if name.startswith("P"):
        std = (self._range_span(name) * sc.pressure_noise_pct / 100.0)
        v = value + self.rng.normal(0.0, std)
    # ...

    # Применение дрейфа (если есть неисправность)
    if faults.sensor_pressure_drift > 0:
        step = (0.002 * faults.sensor_pressure_drift) * self._range_span(name)
        self._drift_state[name] += float(self.rng.normal(0.0, step))
        v = value + self._drift_state[name]
    # ...

    # Возможность отказа датчика (dropout)
    if faults.sensor_dropout > 0:
        p = 0.02 * faults.sensor_dropout
        if self.rng.random() < p:
            return float("nan")
    # ...
    return out
```

## 10. Запись цикла данных

После завершения временного цикла для одного цикла операции:

```python
# hydrosim/generator.py
# Формирование timeline
timeline = {"time": time, **obs_buf}

# Преобразование неисправностей для метаданных
faults_by_component = faults.labels_by_component(thr=0.25)
faults_flat = faults.flat_labels(thr=0.25, prefix="fault__")

# Создание метаданных цикла
meta = CycleMeta(
    cycle_id=cid,
    mode=prof.mode,
    duration_s=prof.duration_s,
    payload_kg=prof.payload_kg,
    soil_factor=prof.soil_factor,
    aggressiveness=prof.aggressiveness,
    faults_by_component=faults_by_component,
    faults_flat=faults_flat,
)

# Запись цикла
self.logger.log_cycle(meta, timeline)
```

```python
# hydrosim/logger.py
def log_cycle(self, meta: CycleMeta, timeline: Dict[str, np.ndarray]):
    cid = f"cycle_{meta.cycle_id:06d}"
    g = self.grp.create_group(cid)


    # Запись временных рядов в HDF5
    for k, arr in timeline.items():
        g.create_dataset(k, data=np.asarray(arr, dtype=np.float32), compression="gzip", compression_opts=5)

    # Запись метаданных как атрибутов группы
    g.attrs["mode"] = meta.mode
    g.attrs["duration_s"] = meta.duration_s
    # ... другие атрибуты ...
    g.attrs["faults_by_component_json"] = json.dumps(meta.faults_by_component, ensure_ascii=False)
    g.attrs["faults_flat_json"] = json.dumps(meta.faults_flat, ensure_ascii=False)


    # Запись метаданных в JSONL файл
    self._meta_f.write(json.dumps(meta.__dict__, ensure_ascii=False) + "\n")
```

## 11. Завершение работы

После генерации всех циклов:

```python
# hydrosim/generator.py
self.logger.close()
print("Done. Output:", Path(self.settings.out_dir).resolve())
```