import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X) # (n, m)
    w = np.zeros(X.shape[1])
    b = 0.0
    y = np.array(y)
    n_step = 0
    # loss = 10
    # loss_old = -10

    # while (n_step < steps) or (abs(loss_old - loss) < 0.01):
    for _ in range(steps):
        p = _sigmoid(np.dot(X, w) + b) # (n,m+1)@(m+1, 1)
        # loss_old = loss
        # loss = np.mean(y * np.log(p) + (1-y) * np.log(1-p))
        w = w - lr * (np.dot(X.T, (p-y)) / (X.shape[0]))
        b = b - lr * np.mean(p-y)
        n_step += 1
    return (w, b)