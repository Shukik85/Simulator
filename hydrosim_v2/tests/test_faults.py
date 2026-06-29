import pytest

from hydrosim_v2.faults.params import (
    FaultConfig, PumpFaultParams, CylinderFaultParams,
    ValveFaultParams, SensorFaultParams,
)


class TestFaultConfig:
    def test_healthy_config(self):
        fc = FaultConfig()
        labels = fc.to_multilabel()
        assert not labels["pump"]
        assert not labels["boom"]
        assert not labels["arm"]
        assert not labels["bucket"]

    def test_pump_fault_detected(self):
        fc = FaultConfig(pump=PumpFaultParams(vol_eff_multiplier=0.85))
        labels = fc.to_multilabel()
        assert labels["pump"]
        assert not labels["boom"]

    def test_cylinder_fault_detected(self):
        fc = FaultConfig(cylinders={
            "boom": CylinderFaultParams(internal_leak_coeff=0.1),
        })
        labels = fc.to_multilabel()
        assert labels["boom"]
        assert not labels["arm"]

    def test_multi_fault(self):
        fc = FaultConfig(
            pump=PumpFaultParams(leakage_multiplier=2.0),
            cylinders={
                "arm": CylinderFaultParams(friction_multiplier=1.5),
                "bucket": CylinderFaultParams(external_leak_coeff=0.1),
            },
        )
        labels = fc.to_multilabel()
        assert labels["pump"]
        assert not labels["boom"]
        assert labels["arm"]
        assert labels["bucket"]


class TestSensorFault:
    def test_default_no_fault(self):
        sf = SensorFaultParams()
        assert sf.pressure_bias_pa == 0.0
        assert sf.pressure_scale == 1.0
