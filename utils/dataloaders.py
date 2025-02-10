import numpy as np
from typing import List
from pathlib import Path
from random import randint, shuffle

from .arrays import loosestack, loadfile


def load_dataset(f_series: list, f_params: str = None, *, limit: int = None, method = "shortest"):
    """_summary_

    Args:
        f_series (list): _description_
        f_params (str, optional): _description_. Defaults to None.
        limit (int, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    print(f"Reading {len(f_series)} series from\n\t", ", \n\t".join(f_series))
    try:
        # Normally read series; all lines of same length
        series = [np.loadtxt(f) for f in f_series]

        # Since series can be different length, we slice them to the same length
        series = loosestack(series)

        # Put in to the right shape
        series = np.swapaxes(series, 0, 1)
    except ValueError:
        # Variable length data
        series = []
        for f in f_series:
            file = loadfile(f)
            series.append(file)
        
        assert len(set([ (nsamples := len(s)) for s in series ])) == 1, \
            "Number of samples in each series should be equal"
        
        nseries = len(series)
        varseries = []
        for i in range(nsamples):
            sample = []
            for n in range(nseries):
                sample.append(np.array(series[n][i]))
            # Reimplementation of loosestack
            LIMIT = min([ len(s) for s in sample ])
            sample = np.array([ s[:LIMIT] for s in sample ])
            
            # Put in to the right shape
            sample = np.swapaxes(sample, 0, 1)
            varseries.append(sample)
        
        assert method in [ "pad", "shortest", "slice" ], f"Unknown method {method}"
        
        if method == "pad":
            PADTO = max([ s.shape[0] for s in varseries ])
            for si, sample in enumerate(varseries):
                padded = []
                for j in range(sample.shape[1]):
                    padded.append(np.pad(sample[:, j], (0, PADTO - len(sample)), "edge"))
                varseries[si] = np.array(padded)
        elif method == "shortest":
            MINLEN = min([ s.shape[0] for s in varseries ])
            varseries = [s for s in varseries if s.shape[0] == MINLEN]
        
        series = np.array(varseries)
    
    # Apply limits
    series = series[:limit]

    if f_params:
        print("Reading parameters from", f_params)
        params = np.loadtxt(f_params)
        params = params[:limit]
    else:
        params = None
    
    return series, params


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
        
        series = [ np.loadtxt(f) for f in fseries ]
        SLICETO = min([ s.shape[1] for s in series ])
        series = [ s[:, :SLICETO] for s in series ]
        series = np.array(series)
        
        ndim, nrow, time = series.shape
        self.series = series.reshape((nrow, ndim, time))
    
                
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
    
    def simulate_series(self, params: np.ndarray):
        if self.indexable_series:
            return self.series[params[0].astype(int)]
        
        if self.model:
            return self.model(params)
        
        raise RuntimeError("Cannot simulate without a model")
        
    def sample(self):
        IX = randint(0, len(self.params))
        return self[IX]
