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

    y = [
        [
            sum(
                [X_ik * W_kj for X_ik, W_kj in zip(X_i, W_j)]
            ) + b_j for W_j, b_j in zip(W_T, b)
        ] for X_i in X
    ]
    return y