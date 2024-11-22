import numpy as np
from typing import List


def loosestack(arrays: List[np.ndarray]):
    SIZE = min([ s.shape[0] for s in arrays ])
    TIME = min([ s.shape[1] for s in arrays ])
    
    stack = [ s[:SIZE, :TIME] for s in arrays ]
    stack = np.array(stack)
    
    return stack