from __future__ import annotations

import numpy as np
from aenum import EnumMeta

arrayishNone = (list, tuple, set, np.ndarray, type(None))
arrayish = (list, tuple, set, np.ndarray)


def elements(path: str) -> dict[str, list[str]]:
    with open(path) as fil:
        elements: dict[str, list[str]] = {}
        for lin in fil:
            line = lin.replace("\n", "")
            if "[" in line:
                section = line.replace("[", "").replace("]", "").lower().strip()
                elements[section] = []
                continue
            if len(line) > 0:
                elements[section].append(line)
    return elements


def _enum_get(enum: EnumMeta, name: str) -> int | None:
    try:
        return enum.__getitem__(name.upper())
    except KeyError:
        return None


def _enum_keys(enum: EnumMeta) -> list[str]:
    return list(map(lambda x: x.lower(), enum.__members__.keys()))
