import argparse
from pathlib import Path
import numpy as np
import networkx as nx
from sklearn.metrics import adjusted_rand_score

from datasets.Graph import GraphWalk
from models.SyncMap import SyncMap
from utils.utils import set_seed
from utils.draw_3d import *

def load_graph(dot_path):
    g = nx.DiGraph(nx.nx_agraph.read_dot(dot_path))
    idx = {n: i for i, n in enumerate(g.nodes())}
    A = np.zeros((len(idx), len(idx)), dtype=np.float32)
    for u, v, d in g.edges(data=True):
        A[idx[u], idx[v]] = float(d.get("weight", 1.0))
    return A, idx

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="imbalanced", help='Run which data.')
    parser.add_argument('--time_delay', type=int, default=10, help='Set the time delay.')
    parser.add_argument('--task_type', type=int, default=1,
                        help='Choose the task to run the model, e.g., 1 for GraphWalkTest, 2 for FixedChunkTest, 3 for GraphWalkTest with sequence2.dot, 4 for GraphWalkTest with sequence1.dot, 5 for LongChunkTest, 6 for OverlapChunkTest1, 7 for OverlapChunkTest2')
    parser.add_argument('--sequence_length', type=int, default=10_000, help='Set the sequence length to run the model.')
    parser.add_argument('--iter', type=int, default=1, help='Training time for each data.')
    parser.add_argument('--map_dimensions', type=int, default=3, help='Set the dimension of the map.')
    parser.add_argument('--adaptation_rate', type=float, default=0.001, help='Set the adaptation rate.')
    parser.add_argument('--seed', type=int, default=412, help='Random seed.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    root_dir = Path("./data")
    if args.dataset == "all":
        dot_files = list(root_dir.rglob("*.dot"))
        num = len(dot_files)

        score_all = 0.
        for dot_path in dot_files:
            # print("dot_path:", dot_path)
            env = GraphWalk(args.time_delay, dot_path)

            output_size = env.getOutputSize()
            input_sequence, input_class = env.getSequence(args.sequence_length)

            # 2. Train and Test
            number_of_nodes = output_size
            neuron_group = SyncMap(number_of_nodes, args.map_dimensions, args.adaptation_rate * output_size)

            neuron_group.input(input_sequence)
            labels = neuron_group.organize()

            ari = adjusted_rand_score(labels, env.trueLabel())
            score_all += ari
            print("Data:", dot_path)
            # print("Learned Labels: ", labels)
            # print("Correct Labels: ", env.trueLabel())
            print("ARI  =", ari)

        score_avg = score_all / num
        print("score_avg:", score_avg)

    else:
        file_path = "./data/graph.dot"
        env = GraphWalk(args.time_delay, file_path)

        output_size = env.getOutputSize()
        input_sequence, input_class = env.getSequence(args.sequence_length)

        # 2. Train and Test
        number_of_nodes = output_size
        neuron_group = SyncMap(number_of_nodes, args.map_dimensions, args.adaptation_rate * output_size)

        neuron_group.input(input_sequence)
        labels, draw_3d_nodes = neuron_group.organize()
        print("draw_3d_nodes:", draw_3d_nodes.shape)#(99990, 30, 3)

        ari = adjusted_rand_score(labels, env.trueLabel())
        print("ARI  =", ari)


        color = labels2colors(labels)
        print("color:", color)
        hex = convert_rgb_list_to_hex(color)
        print("hex:", hex)
        # create_scatter_gif_3d(draw_3d_nodes, hex)

        animate_3d_coords(draw_3d_nodes, hex)






