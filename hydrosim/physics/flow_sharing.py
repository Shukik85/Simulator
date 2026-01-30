from __future__ import annotations

from typing import Dict


DEFAULT_PRIORITIES: Dict[str, float] = {
    "swing": 1.0,
    "boom": 0.8,
    "arm": 0.6,
    "bucket": 0.4,
}


def allocate_flow_proportional(qreq: Dict[str, float], Qpump: float) -> Dict[str, float]:
    """Proportional flow-sharing (current baseline).

    Preserves sign of each request and scales magnitudes uniformly if total demand
    exceeds available pump flow.
    """

    qsum = sum(abs(q) for q in qreq.values())
    if qsum <= Qpump + 1e-18:
        return dict(qreq)
    alpha = Qpump / max(qsum, 1e-18)
    return {k: float(alpha * v) for k, v in qreq.items()}


def allocate_flow_priority(
    qreq: Dict[str, float],
    Qpump: float,
    priorities: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Priority-weighted allocator.

    Goals:
    - respect total available flow (sum(|qalloc|) <= Qpump)
    - prefer higher-priority axes when flow is limited
    - preserve sign of each requested flow

    Note: priorities are weights, not hard guarantees.
    """

    if priorities is None:
        priorities = DEFAULT_PRIORITIES

    # Demand magnitudes.
    demand = {k: max(0.0, abs(float(v))) for k, v in qreq.items()}
    if sum(demand.values()) <= Qpump + 1e-18:
        return dict(qreq)

    remaining_flow = float(max(Qpump, 0.0))
    remaining = {k for k, d in demand.items() if d > 0.0}
    alloc_mag = {k: 0.0 for k in qreq.keys()}

    # Iteratively distribute flow until all demand is satisfied or no flow remains.
    for _ in range(12):
        if remaining_flow <= 1e-18 or not remaining:
            break

        weights_sum = 0.0
        for k in remaining:
            w = float(priorities.get(k, 1.0))
            weights_sum += w * demand[k]

        if weights_sum <= 1e-18:
            break

        progress = 0.0
        for k in list(remaining):
            w = float(priorities.get(k, 1.0))
            share = remaining_flow * (w * demand[k]) / weights_sum
            take = min(demand[k], share)
            if take > 0.0:
                alloc_mag[k] += take
                demand[k] -= take
                remaining_flow -= take
                progress += take
            if demand[k] <= 1e-18:
                remaining.discard(k)

        if progress <= 1e-18:
            break

    # Restore sign.
    out: Dict[str, float] = {}
    for k, v in qreq.items():
        if v > 0:
            out[k] = float(alloc_mag.get(k, 0.0))
        elif v < 0:
            out[k] = float(-alloc_mag.get(k, 0.0))
        else:
            out[k] = 0.0

    return out


def allocate_flow(qreq: Dict[str, float], Qpump: float, mode: str = "proportional") -> Dict[str, float]:
    if mode == "priority":
        return allocate_flow_priority(qreq, Qpump)
    return allocate_flow_proportional(qreq, Qpump)


def effective_command(u: float, qreq: float, qalloc: float) -> float:
    """Map a requested command to an effective command under flow limitation.

    This matches the previous inlined logic in hydraulic_model.py:
    scale = min(1, |qalloc|/|qreq|), u_eff = sign(u) * min(1, |u| * scale)
    """

    uu = float(u)
    q0 = float(qreq)
    qa = float(qalloc)

    scale = 1.0 if abs(q0) < 1e-12 else min(1.0, abs(qa) / max(abs(q0), 1e-12))

    if uu > 0.0:
        return float(min(1.0, abs(uu) * scale))
    if uu < 0.0:
        return float(-min(1.0, abs(uu) * scale))
    return 0.0
