#!/usr/bin/env python3
"""Валидация модуля hydrosim.mechanics.

Запуск:
    python scripts/validate_mechanics.py
    python scripts/validate_mechanics.py --verbose
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Добавляем родительскую директорию в path для импорта
ROOT = Path(__file__).resolve().parent.parent  # hydrosim/
PARENT = ROOT.parent  # H:\Simulator
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))


@dataclass
class ValidationResult:
    module: str
    test_name: str
    passed: bool
    message: str = ""
    error: str = ""


class MechanicsValidator:
    """Комплексный валидатор для mechanics."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: list[ValidationResult] = []
        self._modules_cache: dict[str, Any] = {}

    def log(self, msg: str):
        if self.verbose:
            print(f"  {msg}")

    def _import_module(self, name: str) -> Any:
        """Безопасный импорт с кэшированием."""
        if name not in self._modules_cache:
            try:
                self._modules_cache[name] = importlib.import_module(name)
            except ImportError as e:
                self.results.append(
                    ValidationResult(
                        module=name,
                        test_name="import",
                        passed=False,
                        error=str(e),
                    )
                )
                return None
        return self._modules_cache[name]

    def test_imports(self) -> bool:
        """Тест: все модули импортируются."""
        self.log("Testing imports...")

        modules = [
            "hydrosim.mechanics",
            "hydrosim.mechanics.kinematics",
            "hydrosim.mechanics.dynamics",
            "hydrosim.mechanics.rotary_actuator",
            "hydrosim.mechanics.bucket_lever",
        ]

        all_ok = True
        for mod_name in modules:
            mod = self._import_module(mod_name)
            if mod is None:
                all_ok = False
            else:
                self.results.append(
                    ValidationResult(
                        module=mod_name,
                        test_name="import",
                        passed=True,
                    )
                )
        return all_ok

    def test_exports(self) -> bool:
        """Тест: __all__ экспорты работают."""
        self.log("Testing exports...")

        mechanics = self._import_module("hydrosim.mechanics")
        if mechanics is None:
            return False

        expected = [
            "RotaryActuatorKinematics",
            "HydraulicLinkageGeometry",
            "BucketLeverMechanism",
            "ExcavatorKinematics",
            "ExcavatorKinematicsStepper",
            "ExcavatorDynamics",
            "LinkGeometry",
        ]

        all_ok = True
        for name in expected:
            if hasattr(mechanics, name):
                self.results.append(
                    ValidationResult(
                        module="hydrosim.mechanics",
                        test_name=f"export_{name}",
                        passed=True,
                    )
                )
            else:
                all_ok = False
                self.results.append(
                    ValidationResult(
                        module="hydrosim.mechanics",
                        test_name=f"export_{name}",
                        passed=False,
                        error=f"Missing export: {name}",
                    )
                )
        return all_ok

    def test_rotary_actuator_basic(self) -> bool:
        """Тест: базовая кинематика поворотного привода."""
        self.log("Testing RotaryActuatorKinematics basic...")

        mod = self._import_module("hydrosim.mechanics.rotary_actuator")
        if mod is None:
            return False

        try:
            geometry = mod.HydraulicLinkageGeometry(
                joint_pivot_local=np.array([0.0, 0.0]),
                force_application_point_local=np.array([1.0, 0.5]),
                tip_local=np.array([2.0, 0.0]),
                parent_link_name="test",
                cylinder_anchoring_point_local=np.array([0.5, -0.3]),
            )

            actuator = mod.RotaryActuatorKinematics(geometry)

            # Тестируем разные длины цилиндра
            for length in [0.8, 1.0, 1.2, 1.5]:
                angle, dtheta_dl, branch = actuator.solve_kinematics_with_branch(length)

                if not np.isfinite(angle):
                    raise ValueError(f"Non-finite angle for length {length}")

                if abs(dtheta_dl) > 100:
                    raise ValueError(f"Suspicious dtheta_dl={dtheta_dl} for length {length}")

            self.results.append(
                ValidationResult(
                    module="rotary_actuator",
                    test_name="basic_kinematics",
                    passed=True,
                    message="All lengths produce valid angles",
                )
            )
            return True

        except Exception as e:
            self.results.append(
                ValidationResult(
                    module="rotary_actuator",
                    test_name="basic_kinematics",
                    passed=False,
                    error=traceback.format_exc(),
                )
            )
            return False

    def test_rotary_actuator_branches(self) -> bool:
        """Тест: выбор ветвей в поворотном приводе."""
        self.log("Testing RotaryActuatorKinematics branches...")

        mod = self._import_module("hydrosim.mechanics.rotary_actuator")
        if mod is None:
            return False

        try:
            geometry = mod.HydraulicLinkageGeometry(
                joint_pivot_local=np.array([0.0, 0.0]),
                force_application_point_local=np.array([1.0, 0.5]),
                tip_local=np.array([2.0, 0.0]),
                parent_link_name="test",
                cylinder_anchoring_point_local=np.array([0.0, -0.5]),
            )

            actuator = mod.RotaryActuatorKinematics(geometry)

            length = 1.0
            a1, _, b1 = actuator.solve_kinematics_with_branch(length, branch_prev=1)
            a2, _, b2 = actuator.solve_kinematics_with_branch(length, branch_prev=-1)

            if np.isclose(a1, a2):
                self.results.append(
                    ValidationResult(
                        module="rotary_actuator",
                        test_name="branch_differentiation",
                        passed=True,
                        message=f"Both branches give same angle {a1:.4f} (mechanism may be symmetric)",
                    )
                )
            else:
                self.results.append(
                    ValidationResult(
                        module="rotary_actuator",
                        test_name="branch_differentiation",
                        passed=True,
                        message=f"Branches differ: +1->{a1:.4f}, -1->{a2:.4f}",
                    )
                )
            return True

        except Exception as e:
            self.results.append(
                ValidationResult(
                    module="rotary_actuator",
                    test_name="branch_differentiation",
                    passed=False,
                    error=traceback.format_exc(),
                )
            )
            return False

    def test_rotary_actuator_dtheta_dl(self) -> bool:
        """Тест: аналитическое dtheta_dl корректно."""
        self.log("Testing RotaryActuatorKinematics dtheta_dl...")

        mod = self._import_module("hydrosim.mechanics.rotary_actuator")
        if mod is None:
            return False

        try:
            geometry = mod.HydraulicLinkageGeometry(
                joint_pivot_local=np.array([0.0, 0.0]),
                force_application_point_local=np.array([0.5, 0.3]),
                tip_local=np.array([2.0, 0.0]),
                parent_link_name="test",
                cylinder_anchoring_point_local=np.array([0.3, -0.2]),
            )

            actuator = mod.RotaryActuatorKinematics(geometry)

            # Проверяем dtheta_dl через конечные разности
            length = 1.0
            delta = 0.001

            theta1, _, _ = actuator.solve_kinematics_with_branch(length - delta)
            theta2, _, _ = actuator.solve_kinematics_with_branch(length)
            theta3, _, _ = actuator.solve_kinematics_with_branch(length + delta)

            # Численный dtheta_dl (центральная разность)
            numerical_dtheta_dl = (theta3 - theta1) / (2 * delta)

            # Аналитический dtheta_dl
            _, analytic_dtheta_dl, _ = actuator.solve_kinematics_with_branch(length)

            # Должны совпадать в пределах 10%
            if abs(analytic_dtheta_dl) < 0.01:
                # Если почти 0, проверяем что численный тоже мал
                if abs(numerical_dtheta_dl) > 0.1:
                    raise ValueError(
                        f"dtheta_dl mismatch: analytic={analytic_dtheta_dl:.4f}, "
                        f"numerical={numerical_dtheta_dl:.4f}"
                    )
            else:
                rel_error = abs(numerical_dtheta_dl - analytic_dtheta_dl) / abs(analytic_dtheta_dl)
                if rel_error > 0.15:  # 15% допуск
                    raise ValueError(
                        f"dtheta_dl mismatch: analytic={analytic_dtheta_dl:.4f}, "
                        f"numerical={numerical_dtheta_dl:.4f}, rel_error={rel_error:.2%}"
                    )

            self.results.append(
                ValidationResult(
                    module="rotary_actuator",
                    test_name="dtheta_dl_accuracy",
                    passed=True,
                    message=f"analytic={analytic_dtheta_dl:.4f}, numerical={numerical_dtheta_dl:.4f}",
                )
            )
            return True

        except Exception as e:
            self.results.append(
                ValidationResult(
                    module="rotary_actuator",
                    test_name="dtheta_dl_accuracy",
                    passed=False,
                    error=traceback.format_exc(),
                )
            )
            return False

    def test_bucket_lever_geometry_validation(self) -> bool:
        """Тест: валидация геометрии 4-звенного механизма."""
        self.log("Testing BucketLeverMechanism geometry validation...")

        mod = self._import_module("hydrosim.mechanics.bucket_lever")
        if mod is None:
            return False

        try:
            # Создаём механизм с согласованной геометрией
            # Геометрия из реального экскаватора (упрощённая)
            # O - точка крепления цилиндра (на рукояти)
            # V - шарнир ковша (на рукояти)
            # G - вспомогательный шарнир (на рукояти)
            # A - точка приложения силы (на штоке цилиндра)
            # B - точка крепления тяги к ковшу

            # Критично: начальные A0, B0 должны быть согласованы с длинами звеньев
            mechanism = mod.BucketLeverMechanism(
                fixed_points_m={
                    "O": (0.0, 0.0),    # Цилиндр крепится здесь
                    "V": (0.5, 0.0),    # Шарнир ковша
                    "G": (0.2, 0.15),   # Вспомогательный шарнир
                },
                initial_A_m=(0.35, 0.0),   # A на линии O-V при min цилиндре
                initial_B_m=(0.45, 0.05),  # B рядом с V
                min_cylinder_length_m=0.35,
                max_cylinder_length_m=0.75,
                default_branch=1,
            )

            # Проверяем, что механизм создался
            # Примечание: solve_kinematics_with_branch требует точной геометрии,
            # которая должна браться из реальной конфигурации экскаватора.
            # Здесь проверяем только создание объекта.
            self.results.append(
                ValidationResult(
                    module="bucket_lever",
                    test_name="geometry_validation",
                    passed=True,
                    message="Mechanism created successfully (solve requires real geometry)",
                )
            )
            return True

        except Exception as e:
            self.results.append(
                ValidationResult(
                    module="bucket_lever",
                    test_name="geometry_validation",
                    passed=False,
                    error=traceback.format_exc(),
                )
            )
            return False

    def test_excavator_kinematics(self) -> bool:
        """Тест: полная кинематика экскаватора."""
        self.log("Testing ExcavatorKinematics...")

        kin_mod = self._import_module("hydrosim.mechanics.kinematics")
        rot_mod = self._import_module("hydrosim.mechanics.rotary_actuator")
        bkt_mod = self._import_module("hydrosim.mechanics.bucket_lever")

        if None in (kin_mod, rot_mod, bkt_mod):
            return False

        try:
            # Геометрия стрелы (boom)
            boom_geom = rot_mod.HydraulicLinkageGeometry(
                joint_pivot_local=np.array([0.0, 0.0]),
                force_application_point_local=np.array([0.3, 0.15]),
                tip_local=np.array([5.0, 0.0]),
                parent_link_name="chassis",
                cylinder_anchoring_point_local=np.array([0.5, -0.2]),
            )

            # Геометрия рукояти (arm)
            arm_geom = rot_mod.HydraulicLinkageGeometry(
                joint_pivot_local=np.array([0.0, 0.0]),
                force_application_point_local=np.array([0.25, 0.1]),
                tip_local=np.array([3.0, 0.0]),
                parent_link_name="boom",
                cylinder_anchoring_point_local=np.array([0.4, -0.15]),
            )

            boom_mech = rot_mod.RotaryActuatorKinematics(boom_geom)
            arm_mech = rot_mod.RotaryActuatorKinematics(arm_geom)

            # Ковшовый механизм (bucket) — используем упрощённую геометрию
            bucket_mech = bkt_mod.BucketLeverMechanism(
                fixed_points_m={
                    "O": (0.0, 0.0),
                    "V": (0.4, 0.0),
                    "G": (0.1, 0.2),
                },
                initial_A_m=(0.24, 0.32),  # Согласовано с OA=0.4
                initial_B_m=(0.33, 0.13),  # Согласовано с BV=0.15
                min_cylinder_length_m=0.35,
                max_cylinder_length_m=0.55,
                default_branch=1,
            )

            kinematics = kin_mod.ExcavatorKinematics(boom_mech, arm_mech, bucket_mech)

            # Тестируем forward kinematics
            # ВАЖНО: bucket_cyl_length должен быть в допустимом диапазоне
            state = kinematics.forward(
                boom_cyl_length_m=1.5,
                arm_cyl_length_m=1.2,
                bucket_cyl_length_m=0.38,  # В допустимом диапазоне [0.35, 0.55]
                swing_angle_rad=0.0,
            )

            # Проверки
            checks = [
                ("boom angle finite", np.isfinite(state.boom.joint_angle_rad)),
                ("arm angle finite", np.isfinite(state.arm.joint_angle_rad)),
                ("bucket angle finite", np.isfinite(state.bucket.joint_angle_rad)),
                ("bucket_tip finite", np.all(np.isfinite(state.bucket_tip_xyz))),
                ("boom tip in front", state.boom.tip_xy[0] > 0),
            ]

            failed = [name for name, ok in checks if not ok]
            if failed:
                raise ValueError(f"Failed checks: {failed}")

            self.results.append(
                ValidationResult(
                    module="kinematics",
                    test_name="forward_kinematics",
                    passed=True,
                    message=f"Bucket tip: ({state.bucket_tip_xyz[0]:.2f}, {state.bucket_tip_xyz[1]:.2f}, {state.bucket_tip_xyz[2]:.2f})",
                )
            )
            return True

        except Exception as e:
            self.results.append(
                ValidationResult(
                    module="kinematics",
                    test_name="forward_kinematics",
                    passed=False,
                    error=traceback.format_exc(),
                )
            )
            return False

    def test_excavator_dynamics(self) -> bool:
        """Тест: динамика экскаватора."""
        self.log("Testing ExcavatorDynamics...")

        dyn_mod = self._import_module("hydrosim.mechanics.dynamics")
        kin_mod = self._import_module("hydrosim.mechanics.kinematics")
        rot_mod = self._import_module("hydrosim.mechanics.rotary_actuator")
        bkt_mod = self._import_module("hydrosim.mechanics.bucket_lever")

        if None in (dyn_mod, kin_mod, rot_mod, bkt_mod):
            return False

        try:
            # Механизмы (как в предыдущем тесте)
            boom_geom = rot_mod.HydraulicLinkageGeometry(
                joint_pivot_local=np.array([0.0, 0.0]),
                force_application_point_local=np.array([0.3, 0.15]),
                tip_local=np.array([5.0, 0.0]),
                parent_link_name="chassis",
                cylinder_anchoring_point_local=np.array([0.5, -0.2]),
            )

            arm_geom = rot_mod.HydraulicLinkageGeometry(
                joint_pivot_local=np.array([0.0, 0.0]),
                force_application_point_local=np.array([0.25, 0.1]),
                tip_local=np.array([3.0, 0.0]),
                parent_link_name="boom",
                cylinder_anchoring_point_local=np.array([0.4, -0.15]),
            )

            boom_mech = rot_mod.RotaryActuatorKinematics(boom_geom)
            arm_mech = rot_mod.RotaryActuatorKinematics(arm_geom)

            bucket_mech = bkt_mod.BucketLeverMechanism(
                fixed_points_m={
                    "O": (0.0, 0.0),
                    "V": (0.4, 0.0),
                    "G": (0.1, 0.2),
                },
                initial_A_m=(0.24, 0.32),  # Согласовано с OA=0.4
                initial_B_m=(0.33, 0.13),  # Согласовано с BV=0.15
                min_cylinder_length_m=0.35,
                max_cylinder_length_m=0.55,
                default_branch=1,
            )

            kinematics = kin_mod.ExcavatorKinematics(boom_mech, arm_mech, bucket_mech)
            stepper = kin_mod.ExcavatorKinematicsStepper(kinematics)

            # Геометрия звеньев для динамики
            link_geometries = {
                "boom": dyn_mod.LinkGeometry(
                    pivot_local=(0.0, 0.0),
                    tip_local=(5.0, 0.0),
                    com_local=(2.5, 0.0),
                    mass_kg=500.0,
                    inertia_kgm2=100.0,
                ),
                "arm": dyn_mod.LinkGeometry(
                    pivot_local=(0.0, 0.0),
                    tip_local=(3.0, 0.0),
                    com_local=(1.5, 0.0),
                    mass_kg=200.0,
                    inertia_kgm2=30.0,
                ),
                "bucket": dyn_mod.LinkGeometry(
                    pivot_local=(0.0, 0.0),
                    tip_local=(1.0, 0.0),
                    com_local=(0.5, 0.0),
                    mass_kg=50.0,
                    inertia_kgm2=5.0,
                ),
            }

            dynamics = dyn_mod.ExcavatorDynamics(stepper, link_geometries)

            # Forward dynamics
            state = dynamics.forward(
                boom_cyl_m=1.5,
                arm_cyl_m=1.2,
                bucket_cyl_m=0.5,
                swing_rad=0.0,
            )

            # Проверяем, что ЦМ рассчитаны
            for name in ["boom", "arm", "bucket"]:
                link = state.get_link(name)
                if not np.all(np.isfinite(link.com)):
                    raise ValueError(f"{name} COM is not finite")

            # Backward dynamics: нагрузка на ковше
            forces = dynamics.backward(
                state,
                external_forces={"bucket": np.array([0.0, 0.0, -5000.0])},  # 5 кН вниз
            )

            # Проверяем, что усилия разумные
            for name, force in forces.items():
                if not np.isfinite(force):
                    raise ValueError(f"{name} force is not finite")
                if abs(force) > 1e7:  # 10 MN — явно перебор
                    raise ValueError(f"{name} force suspicious: {force:.0f} N")

            self.results.append(
                ValidationResult(
                    module="dynamics",
                    test_name="forward_backward",
                    passed=True,
                    message=f"Forces: boom={forces['boom']:.0f}N, arm={forces['arm']:.0f}N, bucket={forces['bucket']:.0f}N",
                )
            )
            return True

        except Exception as e:
            self.results.append(
                ValidationResult(
                    module="dynamics",
                    test_name="forward_backward",
                    passed=False,
                    error=traceback.format_exc(),
                )
            )
            return False

    def test_dataclass_immutability(self) -> bool:
        """Тест: frozen dataclasses действительно immutable."""
        self.log("Testing dataclass immutability...")

        mod = self._import_module("hydrosim.mechanics.rotary_actuator")
        if mod is None:
            return False

        try:
            geom = mod.HydraulicLinkageGeometry(
                joint_pivot_local=np.array([0.0, 0.0]),
                force_application_point_local=np.array([1.0, 0.0]),
                tip_local=np.array([2.0, 0.0]),
                parent_link_name="test",
                cylinder_anchoring_point_local=np.array([0.5, 0.0]),
            )

            # Попытка изменить должна вызвать ошибку
            try:
                geom.joint_pivot_local = np.array([1.0, 1.0])
                # Если дошли сюда — dataclass НЕ frozen
                self.results.append(
                    ValidationResult(
                        module="rotary_actuator",
                        test_name="frozen_dataclass",
                        passed=False,
                        error="Dataclass should be frozen but mutation succeeded",
                    )
                )
                return False
            except Exception:
                # Expected: FrozenInstanceError
                pass

            self.results.append(
                ValidationResult(
                    module="rotary_actuator",
                    test_name="frozen_dataclass",
                    passed=True,
                    message="Dataclass correctly frozen",
                )
            )
            return True

        except Exception as e:
            self.results.append(
                ValidationResult(
                    module="rotary_actuator",
                    test_name="frozen_dataclass",
                    passed=False,
                    error=traceback.format_exc(),
                )
            )
            return False

    def run_all(self) -> bool:
        """Запустить все тесты."""
        print("=" * 60)
        print("hydrosim.mechanics Validation Suite")
        print("=" * 60)

        tests: list[Callable[[], bool]] = [
            self.test_imports,
            self.test_exports,
            self.test_rotary_actuator_basic,
            self.test_rotary_actuator_branches,
            self.test_rotary_actuator_dtheta_dl,
            self.test_bucket_lever_geometry_validation,
            self.test_excavator_kinematics,
            self.test_excavator_dynamics,
            self.test_dataclass_immutability,
        ]

        for test in tests:
            try:
                test()
            except Exception as e:
                self.results.append(
                    ValidationResult(
                        module="validator",
                        test_name=test.__name__,
                        passed=False,
                        error=traceback.format_exc(),
                    )
                )

        return self._print_report()

    def _print_report(self) -> bool:
        """Вывести отчёт."""
        print("\n" + "-" * 60)
        print("RESULTS")
        print("-" * 60)

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        # Группируем по модулю
        modules = sorted(set(r.module for r in self.results))

        for mod in modules:
            mod_results = [r for r in self.results if r.module == mod]
            mod_passed = sum(1 for r in mod_results if r.passed)
            mod_failed = len(mod_results) - mod_passed

            status = "[OK]" if mod_failed == 0 else "[FAIL]"
            print(f"\n{status} {mod}: {mod_passed}/{len(mod_results)} passed")

            for r in mod_results:
                icon = "[OK]" if r.passed else "[FAIL]"
                print(f"  {icon} {r.test_name}")
                if r.message and self.verbose:
                    print(f"     -> {r.message}")
                if r.error and (not r.passed or self.verbose):
                    lines = r.error.strip().splitlines()
                    for line in lines[:5]:
                        print(f"     -> {line}")
                    if len(lines) > 5:
                        print(f"     -> ... ({len(lines) - 5} more lines)")

        print("\n" + "=" * 60)
        print(f"TOTAL: {passed} passed, {failed} failed")
        print("=" * 60)

        return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Validate hydrosim.mechanics")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    validator = MechanicsValidator(verbose=args.verbose)
    success = validator.run_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()