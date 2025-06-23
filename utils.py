import random
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def to_categorical(y, num_classes):
    out = np.zeros(num_classes)
    out[y] = 1
    return out


def show_graph(save=False):
    path = "./data/chain_mixed120_5.dot"

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
