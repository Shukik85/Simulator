import pytest

from hydrosim_v2.core.units import (
    bar_to_Pa, Pa_to_bar, LPM_to_m3s, m3s_to_LPM,
    rpm_to_rad_per_s, rad_per_s_to_rpm,
    meters_to_mm, mm_to_meters,
)
from hydrosim_v2.core.types import (
    Vec2, add_vec2, sub_vec2, mul_scalar_vec2, dot_vec2,
)


class TestUnits:
    def test_bar_to_Pa(self):
        assert bar_to_Pa(1.0) == 1e5
        assert bar_to_Pa(0.0) == 0.0

    def test_Pa_to_bar(self):
        assert Pa_to_bar(1e5) == 1.0
        assert Pa_to_bar(0.0) == 0.0

    def test_roundtrip_pressure(self):
        p = 150.0
        assert Pa_to_bar(bar_to_Pa(p)) == pytest.approx(p)

    def test_LPM_to_m3s(self):
        assert LPM_to_m3s(60.0) == pytest.approx(0.001)

    def test_m3s_to_LPM(self):
        assert m3s_to_LPM(0.001) == pytest.approx(60.0)

    def test_rpm_to_rad_per_s(self):
        assert rpm_to_rad_per_s(60.0) == pytest.approx(2 * 3.141592653589793)

    def test_mm_conversion(self):
        assert meters_to_mm(1.0) == 1000.0
        assert mm_to_meters(1000.0) == 1.0


class TestVec2:
    def test_add(self):
        assert add_vec2((1, 2), (3, 4)) == (4, 6)

    def test_sub(self):
        assert sub_vec2((5, 3), (2, 1)) == (3, 2)

    def test_mul_scalar(self):
        assert mul_scalar_vec2((2, 3), 2.0) == (4.0, 6.0)

    def test_dot(self):
        assert dot_vec2((1, 0), (0, 1)) == 0.0
        assert dot_vec2((2, 3), (4, 5)) == 23.0
