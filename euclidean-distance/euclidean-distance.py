import numpy as np

def euclidean_distance(x, y) -> float:
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return np.linalg.norm(x-y)