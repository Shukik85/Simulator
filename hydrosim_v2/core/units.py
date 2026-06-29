PA_PER_BAR = 1e5
LPM_PER_M3S = 1000.0 * 60.0  # 1 m³/s = 60000 LPM


def bar_to_Pa(bar: float) -> float:
    return bar * PA_PER_BAR


def Pa_to_bar(pa: float) -> float:
    return pa / PA_PER_BAR


def LPM_to_m3s(lpm: float) -> float:
    return lpm / LPM_PER_M3S


def m3s_to_LPM(m3s: float) -> float:
    return m3s * LPM_PER_M3S
