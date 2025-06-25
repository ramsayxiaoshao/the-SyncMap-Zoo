import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import warnings
from tqdm import tqdm
from collections import deque

from utils.utils import to_categorical


class GraphProcessor:
    def __init__(self, time_delay, dataset, state_memory=2):
        self.time_delay = time_delay
        self.time_counter = 0

        self.G = nx.DiGraph(nx.nx_agraph.read_dot(dataset))

        label = self.G.nodes(data="label")
        label = np.asarray(list(label))

        self.true_label = label[:, 1]

        self.true_label = [int(x) for x in self.true_label]
        self.true_label = np.asarray(self.true_label) - 1

        self.output_size = self.G.number_of_nodes()

        self.A = nx.adjacency_matrix(self.G)
        self.A = self.A.todense()
        self.A = np.array(self.A, dtype=np.float64)

        for i in range(self.output_size):
            accum = self.A[i].sum()
            if accum != 0:
                self.A[i] = self.A[i] / accum
            else:
                print("ERROR: Node ", i, " without connections from found")
                exit()
            # print(self.A[i])

        # random start
        self.output_class = np.random.randint(self.output_size)  #
        self.previous_output_class = None
        self.previous_previous_output_class = None

        # Store memory: set the number of activated nodes
        self.working_memory = deque(maxlen=state_memory)

        # self.plotGraph()

    def trueLabel(self):
        return self.true_label

    def getOutputSize(self):
        return self.output_size

    def updateTimeDelay(self):
        self.time_counter += 1
        if self.time_counter > self.time_delay:
            self.time_counter = 0
            self.previous_previous_output_class = self.previous_output_class
            self.previous_output_class = self.output_class
            return True
        else:
            return False

    # create an input pattern for the system
    def getInput(self, reset=False):
        update = self.updateTimeDelay()

        if update == True:
            self.previous_output_class = self.output_class
            # print(self.A)#(600, 600)
            # print(self.A[self.output_class].shape)#(600,)
            # print(self.A[self.output_class])
            self.output_class = np.random.choice(self.output_size, 1, p=self.A[self.output_class])[0]

        noise_intensity = 0
        if self.previous_output_class is None or self.previous_output_class == self.output_class:
            input_value = to_categorical(self.output_class, self.output_size) * np.exp(
                -0.1 * self.time_counter) + np.random.randn(self.output_size) * noise_intensity
            # print("input_value:", input_value)#(600,)
        else:
            input_value = to_categorical(self.output_class, self.output_size) * np.exp(
                -0.1 * self.time_counter) + np.random.randn(self.output_size) * noise_intensity + to_categorical(
                self.previous_output_class, self.output_size) * np.exp(-0.1 * (self.time_counter + self.time_delay))
            # print("input_value:", input_value)

        return input_value

    def random_walk_on_graph(self, sequence_size, reset_time=None):
        connection_matrix = self.A
        num_nodes = self.A.shape[0]

        # Find nodes with no outgoing connections
        no_outgoing = np.where(np.sum(connection_matrix, axis=1) == 0)[0]
        if len(no_outgoing) != 0:
            warnings.warn("Some nodes have no outgoing connections.")

        starting_node = np.random.choice(num_nodes)
        while starting_node in no_outgoing:
            warnings.warn("Starting node has no outgoing connections. Choosing another node.")
            starting_node = np.random.choice(num_nodes)

        trajectory = []
        one_hot_vectors = []

        current_node = starting_node
        steps_since_reset = 0

        print("Random walk starting node:", current_node)

        for _ in tqdm(range(sequence_size)):
            # Record current node index
            trajectory.append(current_node)

            # Generate one-hot vector for current node
            one_hot = np.zeros(num_nodes, dtype=np.bool_)
            one_hot[current_node] = True
            one_hot_vectors.append(one_hot)

            # Choose next node based on outgoing connections
            if np.sum(connection_matrix[current_node]) == 0 or (
                    reset_time is not None and steps_since_reset == reset_time):
                current_node = np.random.choice(num_nodes)
                warnings.warn("No outgoing connections from current node. Choosing another node.")
                steps_since_reset = 0
            else:
                prob = connection_matrix[current_node] / np.sum(connection_matrix[current_node])
                current_node = np.random.choice(num_nodes, p=prob)
                steps_since_reset += 1

        return np.array(trajectory), np.array(one_hot_vectors)

    def generate_memory_sequence(self, input_seq):
        print("generate sequence...")

        output_seq = []
        for i_state in tqdm(range(len(input_seq))):
            state = input_seq[i_state]
            self.working_memory.append(state)
            # convert to numpy
            current_working_mem = np.asarray(self.working_memory)
            current_working_mem = np.sum(current_working_mem, axis=0).astype(np.bool_)

            # append to output_seq
            output_seq.append(current_working_mem)

        return np.asarray(output_seq)


    def plotGraph(self, save=True):
        options = {
            'node_size': 100,
            'arrowstyle': '-|>',
            'arrowsize': 12,
        }
        nx.draw_networkx(self.G, arrows=True, **options)

        if save == True:
            plt.savefig("./results/graph_plot.png")

        plt.show()

    def plot(self, input_class, input_sequence=None, save=False):
        a = np.asarray(input_class)
        t = [i for i, value in enumerate(a)]

        plt.plot(t, a)

        if input_sequence != None:
            sequence = [np.argmax(x) for x in input_sequence]
            plt.plot(t, sequence)

        if save == True:
            plt.savefig("./results/plot.png")

        plt.show()
        plt.close()

    def plotSuperposed(self, input_class, input_sequence=None, save=False):
        input_sequence = np.asarray(input_sequence)
        t = [i for i, value in enumerate(input_sequence)]

        for i in range(input_sequence.shape[1]):
            a = input_sequence[:, i]
            plt.plot(t, a)

        a = np.asarray(input_class)
        plt.plot(t, a)

        if save == True:
            plt.savefig("plot.png")

        plt.show()
        plt.close()
