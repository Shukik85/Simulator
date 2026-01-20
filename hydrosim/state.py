from __future__ import annotations
from dataclasses import dataclass


@dataclass
class State:
    # давления (Pa)
    Ppump: float = 1.0e5
    PLS: float = 1.0e5

    PboomA: float = 1.0e5
    PboomB: float = 1.0e5
    ParmA: float = 1.0e5
    ParmB: float = 1.0e5
    PbucketA: float = 1.0e5
    PbucketB: float = 1.0e5
    PswingA: float = 1.0e5
    PswingB: float = 1.0e5

    Ptank: float = 1.0e5

    # кинематика цилиндров (m, m/s)
    xboom: float = 0.75
    vboom: float = 0.0
    xarm: float = 0.60
    varm: float = 0.0
    xbucket: float = 0.40
    vbucket: float = 0.0

    # swing (rad, rad/s)
    theta: float = 0.0
    omega: float = 0.0

    # температура (C)
    Thydraulic: float = 45.0

    def copy(self) -> "State":
        return State(**self.__dict__)
