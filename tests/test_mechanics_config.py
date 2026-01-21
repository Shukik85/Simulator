import unittest

from hydrosim.config.mechanics import CylinderGeometry, LinkGeometry


class TestLinkGeometry(unittest.TestCase):
    def test_com_offset_m(self) -> None:
        link = LinkGeometry(name="boom", length_m=5.0, mass_kg=1200.0, com_ratio=0.4)
        self.assertAlmostEqual(link.com_offset_m, 2.0)

    def test_moment_of_inertia_pivot(self) -> None:
        m = 1000.0
        L = 5.0
        r = 0.4
        link = LinkGeometry(name="boom", length_m=L, mass_kg=m, com_ratio=r)
        expected = m * (L * L) * ((1.0 / 12.0) + (r * r))
        self.assertAlmostEqual(link.moment_of_inertia_pivot, expected)

    def test_invariants(self) -> None:
        with self.assertRaises(ValueError):
            LinkGeometry(name="boom", length_m=0.0, mass_kg=1.0)
        with self.assertRaises(ValueError):
            LinkGeometry(name="boom", length_m=1.0, mass_kg=0.0)
        with self.assertRaises(ValueError):
            LinkGeometry(name="boom", length_m=1.0, mass_kg=1.0, com_ratio=-0.1)
        with self.assertRaises(ValueError):
            LinkGeometry(name="boom", length_m=1.0, mass_kg=1.0, com_ratio=1.1)


class TestCylinderGeometry(unittest.TestCase):
    def test_length_min_max(self) -> None:
        cyl = CylinderGeometry(
            name="boom_cyl",
            base_point_global=(0.0, 0.0),
            rod_attach_local=(1.0, 0.0),
            stroke_m=1.2,
            base_length_m=1.8,
        )
        self.assertAlmostEqual(cyl.length_min_m, 1.8)
        self.assertAlmostEqual(cyl.length_max_m, 3.0)

    def test_length_from_spool_position(self) -> None:
        cyl = CylinderGeometry(
            name="boom_cyl",
            base_point_global=(0.0, 0.0),
            rod_attach_local=(1.0, 0.0),
            stroke_m=1.2,
            base_length_m=1.8,
        )
        self.assertAlmostEqual(cyl.length_from_spool_position(-1.0), cyl.length_min_m)
        self.assertAlmostEqual(cyl.length_from_spool_position(1.0), cyl.length_max_m)
        self.assertAlmostEqual(
            cyl.length_from_spool_position(0.0),
            (cyl.length_min_m + cyl.length_max_m) / 2.0,
        )

    def test_invariants(self) -> None:
        with self.assertRaises(ValueError):
            CylinderGeometry(
                name="boom_cyl",
                base_point_global=(0.0, 0.0),
                rod_attach_local=(1.0, 0.0),
                stroke_m=0.0,
                base_length_m=1.0,
            )
        with self.assertRaises(ValueError):
            CylinderGeometry(
                name="boom_cyl",
                base_point_global=(0.0, 0.0),
                rod_attach_local=(1.0, 0.0),
                stroke_m=1.0,
                base_length_m=0.0,
            )


if __name__ == "__main__":
    unittest.main()
