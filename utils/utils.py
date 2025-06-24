import random
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# def to_categorical(y, num_classes):
#     out = np.zeros(num_classes)
#     out[y] = 1
#     return out
def to_categorical(x, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with `categorical_crossentropy`.

    Args:
        x: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
            as `max(x) + 1`. Defaults to `None`.

    Returns:
        A binary matrix representation of the input as a NumPy array. The class
        axis is placed last.

    Example:

    >>> a = to_categorical([0, 1, 2, 3], num_classes=4)
    >>> print(a)
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    """
    x = np.array(x, dtype="int64")
    input_shape = x.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    x = x.reshape(-1)
    if not num_classes:
        num_classes = np.max(x) + 1
    batch_size = x.shape[0]
    categorical = np.zeros((batch_size, num_classes))
    categorical[np.arange(batch_size), x] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def show_graph(save=False):
    path = "../data/chain_mixed120_5.dot"

    G = nx.DiGraph(nx.nx_agraph.read_dot(path))

    options = {
        'node_size': 100,
        'arrowstyle': '-|>',
        'arrowsize': 12,
    }
    nx.draw_networkx(G, arrows=True, **options)

    if save == True:
        plt.savefig("./results/graph_plot.png")

    plt.show()


if __name__ == "__main__":
    show_graph()
