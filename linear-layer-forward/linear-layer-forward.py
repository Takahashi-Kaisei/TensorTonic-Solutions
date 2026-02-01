def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # X: i行, k列
    # W: k行, j列
    # b: j列

    # 転置
    W_T = []
    for i in range(len(W[0])):
        W_T.append([j[i] for j in W])

    y = []
    for X_i in X:
        y_i = []
        for W_j, b_j in zip(W_T, b):
            y_ij = 0
            for X_ik, W_kj in zip(X_i, W_j):
                y_ij += X_ik * W_kj
            y_ij = y_ij + b_j
            y_i.append(y_ij)
        y.append(y_i)

    return y