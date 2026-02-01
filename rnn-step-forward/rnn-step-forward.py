import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    @だと、np.dotと挙動が異なるケースがあるので注意。
    """
    h_t = np.tanh(np.dot(x_t, Wx) + np.dot(h_prev, Wh) + b)
    return h_t