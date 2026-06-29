import math

def bar_to_Pa(bar: float) -> float:
    return bar * 1e5

def Pa_to_bar(Pa: float) -> float:
    return Pa / 1e5

def LPM_to_m3s(LPM: float) -> float:
    return LPM / 60_000.0

def m3s_to_LPM(m3s: float) -> float:
    return m3s * 60_000.0

def rpm_to_rad_per_s(rpm: float) -> float:
    return rpm * 2.0 * math.pi / 60.0

def rad_per_s_to_rpm(rad_per_s: float) -> float:
    return rad_per_s * 60.0 / (2.0 * math.pi)

# Metric utilities
def meters_to_mm(m: float) -> float:
    return m * 1000.0

def mm_to_meters(mm: float) -> float:
    return mm / 1000.0
