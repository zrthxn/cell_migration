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
    
    series: List[List] = []
    """ Contains a timeseries for each parameter
    """
    
    indexable_series: bool
    """ Tells you whether or not the series are indexable 
        with the first [0th] parameter
    """
    
    model: callable
    
    def __init__(self, fparams: str | Path, fseries: List[str] | List[Path], nlines: int, norm = False, prepend_index = True) -> None:
        self.indexable_series = prepend_index
        
        with ExitStack() as stack:
            pr = stack.enter_context(open(fparams, "r"))
            fp = [stack.enter_context(open(f, "r")) for f in fseries]
            
            # while pr.readable():
            for lnum in range(nlines):
                params_line = pr.readline().strip()
                series_line = [f.readline().strip() for f in fp]
                
                try:
                    if prepend_index:
                        params_line = f"{lnum} {params_line}"
                        
                    params = np.array(parse_line(params_line))
                    series = [np.array(parse_line(l)) for l in series_line]
                    
                    
                    s_mean = [s.mean() for s in series]
                    s_std = [s.std() for s in series]
                    if norm:
                        series = [ (s - m)/z for s, m, z in zip(series, s_mean, s_std) ]
                    
                    slens = list(map(len, series))
                    pad_to = max(slens)
                    slice_to = min(slens)
                    series = [ s[:slice_to] for s in series ]
                    
                    self.params.append(params)
                    self.series.append(series)
                except:
                    print(Warning("Ingored 1 line"))
                
                assert len(self.params) == len(self.params)
                
    def __len__(self):
        return len(self.params)
        
    def __getitem__(self, index):
        return self.params[index][0 if self.indexable_series else 1:], self.series[index]
    
    def sample_params(self):
        IX = randint(0, len(self.params) - 1)
        return self.params[IX]
    
    def simulate_series(self, params: np.ndarray):
        if self.indexable_series:
            return self.series[params[0].astype(int)]
        
        if self.model:
            return self.model(params)
        
        raise RuntimeError("Cannot simulate without a model")
        
    def sample(self):
        IX = randint(0, len(self.params))
        return self[IX]
