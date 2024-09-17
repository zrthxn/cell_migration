import numpy as np
from typing import List
from pathlib import Path
from contextlib import ExitStack
from pandas import DataFrame
from random import randint, shuffle


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
    
    series: List[List] = []
    """ Contains a timeseries for each parameter
    """
    
    model: callable
    
    def __init__(self, fparams: str | Path, fseries: List[str] | List[Path], nlines: int, norm = False) -> None:                
        self.params = np.loadtxt(fparams)
        self.series = np.array([
            np.loadtxt(f) for f in fseries
        ])
    
                
    def __len__(self):
        return len(self.params)
        
    def __getitem__(self, index):
        return self.params[index], self.series[index]
    
    def train_val_split(self, frac: float):
        SPLIT = int(frac * len(self))
        ORDER = list(range(len(self)))
        shuffle(ORDER)
        
        train_x = self.params[ORDER][:SPLIT]
        train_y = self.series[ORDER][:SPLIT]
        
        val_x = self.params[ORDER][SPLIT:]
        val_y = self.series[ORDER][SPLIT:]
        
        return (train_x, train_y), (val_x, val_y)
    
    def sample_params(self):
        IX = randint(0, len(self.params) - 1)
        return self.params[IX]
    
    # def simulate_series(self, params: np.ndarray):
    #     if self.indexable_series:
    #         return self.series[params[0].astype(int)]
        
    #     if self.model:
    #         return self.model(params)
        
    #     raise RuntimeError("Cannot simulate without a model")
        
    def sample(self):
        IX = randint(0, len(self.params))
        return self[IX]
