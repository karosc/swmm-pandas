import numpy as np
from aenum import Enum
from typing import Dict, Tuple, List, Union

arrayishNone = (list, tuple, set, np.ndarray, type(None))
arrayish = (list, tuple, set, np.ndarray)


def elements(path: str) -> Dict[str, List[str]]:
    with open(path, "r") as fil:
        elements: Dict[str, List[str]] = {}
        for lin in fil:
            line = lin.replace("\n", "")
            if "[" in line:
                section = line.replace("[", "").replace("]", "").lower().strip()
                elements[section] = []
                continue
            if len(line) > 0:
                elements[section].append(line)
    return elements


def _enum_get(enum: Enum, name: str) -> Union[int, None]:
    try:
        return enum.__getitem__(name.upper())
    except KeyError:
        return None


def _enum_keys(enum: Enum) -> Tuple[str, ...]:
    return tuple(map(lambda x: x.lower(), enum.__members__.keys()))
