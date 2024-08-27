import numpy as np
from typing import List
from pathlib import Path
from contextlib import ExitStack
from pandas import DataFrame
from random import randint

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
        while f.readable():
            values = parse_line(f.readline().strip())
            file.append(values)
    
    return file


class LineReader:
    
    params: List = []
    """ Contains parameters
    """
    
    series: List[DataFrame] = []
    """ Contains a timeseries for each parameter
    """
    
    def __init__(self, fparams: str | Path, fseries: List[str] | List[Path]) -> None:
        with ExitStack() as stack:
            pr = stack.enter_context(open(fparams, "r"))
            fp = [stack.enter_context(open(f, "r")) for f in fseries]
            
            while pr.readable():
                try:
                    params = np.array(parse_line(pr.readline().strip()))
                    series = np.array([parse_line(f.readline().strip()) for f in fp])
                    
                    self.params.append(params)
                    self.series.append(series)
                except:
                    print(Warning("Ingored 1 line"))
        
    def __getitem__(self, index):
        return self.params[index], self.series[index]
    
    def iter(self):
        ...
    
    def sample_params(self):
        IX = randint(0, len(self.params))
        return self[IX][0], IX
    
    def sample(self):
        IX = randint(0, len(self.params))
        return self[IX], IX
