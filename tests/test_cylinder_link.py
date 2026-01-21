import pytest

from hydrosim.mechanics.cylinder_link import CylinderAttachment, CylinderLinkMechanism


@pytest.fixture()
def simple_mechanism() -> CylinderLinkMechanism:
    # Геометрия из PDF-примера:
    # O=(0,0), A=(0,1), B_local=(2,0)
    att = CylinderAttachment(
        pivot_point=(0.0, 0.0),
        base_point=(0.0, 1.0),
        link_point_local=(2.0, 0.0),
        link_length_m=3.0,
    )
    return CylinderLinkMechanism(att)


class TestCylinderLinkMechanism:
    def test_solve_angle_horizontal(self, simple_mechanism: CylinderLinkMechanism) -> None:
        # При θ=0 точка B=(2,0), длина AB = sqrt((2-0)^2 + (0-1)^2) = sqrt(5)
        theta, dtheta_dl = simple_mechanism.solve_angle(5.0**0.5)
        assert theta == pytest.approx(0.0, abs=0.15)
        assert dtheta_dl != 0.0

    def test_solve_angle_changes_with_cylinder_length(self, simple_mechanism: CylinderLinkMechanism) -> None:
        theta1, _ = simple_mechanism.solve_angle(2.0)
        theta2, _ = simple_mechanism.solve_angle(2.5)
        assert theta1 != pytest.approx(theta2)

    def test_solve_angle_singularity_handling(self) -> None:
        # Подбор геометрии, где cos(∠AOB) = -1 (∠=π, sin=0) при AB=OA+OB.
        # O=(0,0), A=(3,0) -> OA=3, B_local=(1,0) -> OB=1, AB=4.
        att = CylinderAttachment(
            pivot_point=(0.0, 0.0),
            base_point=(3.0, 0.0),
            link_point_local=(1.0, 0.0),
            link_length_m=2.0,
        )
        mech = CylinderLinkMechanism(att)

        theta, dtheta_dl = mech.solve_angle(4.0)
        assert theta == pytest.approx(3.141592653589793, abs=1e-6)
        assert dtheta_dl == 0.0

    def test_cylinder_force_to_moment_basic(self, simple_mechanism: CylinderLinkMechanism) -> None:
        moment = simple_mechanism.cylinder_force_to_moment(cyl_length_m=2.5, force_N=1000.0)
        assert moment != 0.0

    def test_cylinder_force_to_moment_zero_force(self, simple_mechanism: CylinderLinkMechanism) -> None:
        moment = simple_mechanism.cylinder_force_to_moment(cyl_length_m=2.5, force_N=0.0)
        assert moment == 0.0

    def test_cylinder_force_to_moment_sign_reversal(self, simple_mechanism: CylinderLinkMechanism) -> None:
        moment_pos = simple_mechanism.cylinder_force_to_moment(cyl_length_m=2.5, force_N=1000.0)
        moment_neg = simple_mechanism.cylinder_force_to_moment(cyl_length_m=2.5, force_N=-1000.0)
        if abs(moment_pos) > 1e-9:
            assert moment_pos * moment_neg < 0
