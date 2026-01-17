import numpy as np

def sigmoid(x) -> np.ndarray:
    """
    Vectorized sigmoid function.
    """
    x = np.array(x)
    result = 1 / (1 + np.exp(-x))
    return result