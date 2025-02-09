import numpy as np
from typing import List


def loosestack(arrays: List[np.ndarray]):
    SIZE = min([ s.shape[0] for s in arrays ])
    TIME = min([ s.shape[1] for s in arrays ])
    
    stack = [ s[:SIZE, :TIME] for s in arrays ]
    stack = np.array(stack)
    
    return stack


def lc_substring(s1: str, s2: str):
    print(s1 == s2)
    print(s1)
    matches = ""
    for a, b in zip(s1, s2):
        matches += "*" if a == b else " "
    print(matches)
    print(s2)


def parse_line(line: str):
    values = [eval(v) for v in line.split(" ")]
    assert " ".join([str(v) for v in values]) == line, "File not read right"
    return values


def loadfile(path: str):
    file = []
    
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            values = np.array(line.strip().split(" "), dtype=float)
            file.append(values)
    
    return file