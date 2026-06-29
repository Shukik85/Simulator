from hydrosim_v2.core.units import bar_to_Pa, Pa_to_bar, LPM_to_m3s, m3s_to_LPM


class TestUnits:
    def test_bar_pa_roundtrip(self):
        assert Pa_to_bar(bar_to_Pa(150.0)) == 150.0

    def test_lpm_m3s_roundtrip(self):
        assert abs(m3s_to_LPM(LPM_to_m3s(60.0)) - 60.0) < 1e-12

    def test_known_value(self):
        assert abs(bar_to_Pa(350.0) - 3.5e7) < 1e-9
