import pytest

from hydrosim.config.mechanics import CylinderGeometry, LinkGeometry


class TestLinkGeometry:
    def test_com_offset_m(self) -> None:
        link = LinkGeometry(name="boom", length_m=5.0, mass_kg=1200.0, com_ratio=0.4)
        assert link.com_offset_m == pytest.approx(2.0)

    def test_moment_of_inertia_pivot(self) -> None:
        m = 1000.0
        L = 5.0
        r = 0.4
        link = LinkGeometry(name="boom", length_m=L, mass_kg=m, com_ratio=r)
        expected = m * (L * L) * ((1.0 / 12.0) + (r * r))
        assert link.moment_of_inertia_pivot == pytest.approx(expected)

    @pytest.mark.parametrize(
        "length_m,mass_kg,com_ratio",
        [
            (0.0, 1.0, 0.4),
            (1.0, 0.0, 0.4),
            (1.0, 1.0, -0.1),
            (1.0, 1.0, 1.1),
        ],
    )
    def test_invariants(self, length_m: float, mass_kg: float, com_ratio: float) -> None:
        with pytest.raises(ValueError):
            LinkGeometry(name="boom", length_m=length_m, mass_kg=mass_kg, com_ratio=com_ratio)

    @pytest.mark.parametrize("com_ratio", [0.0, 1.0])
    def test_com_ratio_boundaries(self, com_ratio: float) -> None:
        link = LinkGeometry(name="boom", length_m=2.0, mass_kg=1.0, com_ratio=com_ratio)
        assert 0.0 <= link.com_offset_m <= link.length_m


class TestCylinderGeometry:
    def test_length_min_max(self) -> None:
        cyl = CylinderGeometry(
            name="boom_cyl",
            base_point_global=(0.0, 0.0),
            rod_attach_local=(1.0, 0.0),
            stroke_m=1.2,
            base_length_m=1.8,
        )
        assert cyl.length_min_m == pytest.approx(1.8)
        assert cyl.length_max_m == pytest.approx(3.0)

    def test_length_from_spool_position(self) -> None:
        cyl = CylinderGeometry(
            name="boom_cyl",
            base_point_global=(0.0, 0.0),
            rod_attach_local=(1.0, 0.0),
            stroke_m=1.2,
            base_length_m=1.8,
        )
        assert cyl.length_from_spool_position(-1.0) == pytest.approx(cyl.length_min_m)
        assert cyl.length_from_spool_position(1.0) == pytest.approx(cyl.length_max_m)
        assert cyl.length_from_spool_position(0.0) == pytest.approx(
            (cyl.length_min_m + cyl.length_max_m) / 2.0
        )

    def test_length_from_spool_position_clamps(self) -> None:
        cyl = CylinderGeometry(
            name="boom_cyl",
            base_point_global=(0.0, 0.0),
            rod_attach_local=(1.0, 0.0),
            stroke_m=1.2,
            base_length_m=1.8,
        )
        assert cyl.length_from_spool_position(-2.0) == pytest.approx(cyl.length_min_m)
        assert cyl.length_from_spool_position(2.0) == pytest.approx(cyl.length_max_m)

    @pytest.mark.parametrize(
        "stroke_m,base_length_m",
        [
            (0.0, 1.0),
            (1.0, 0.0),
        ],
    )
    def test_invariants(self, stroke_m: float, base_length_m: float) -> None:
        with pytest.raises(ValueError):
            CylinderGeometry(
                name="boom_cyl",
                base_point_global=(0.0, 0.0),
                rod_attach_local=(1.0, 0.0),
                stroke_m=stroke_m,
                base_length_m=base_length_m,
            )
