import copy
import pickle
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import DBSCAN


class SyncMap:
    def __init__(self, input_size, dimensions, space_scale=1.0, leaking_rate=1.0, dropout_positive=0.0,
                 dropout_negative=0.0,
                 movmean_window=2000,
                 movmean_interval=20,
                 is_symmetrical_activation=True, number_of_selected_node=2, adaptation_rate=0.01,
                 is_adaptive_LR=False, adaptive_LR_widrow_hoff=0.1):
        self.organized = False

        # Initial SyncMap
        self.space_scale = space_scale
        self.space_scale_dimensions_sqrt = space_scale * np.sqrt(dimensions)
        self.dimensions = dimensions
        self.input_size = input_size
        self.movmean_interval = movmean_interval
        self.syncmap_movmean_list = deque(maxlen=movmean_window)
        self.leaking_rate = leaking_rate
        self.dropout_positive = dropout_positive
        self.dropout_negative = dropout_negative

        self.syncmap = np.random.rand(self.input_size, self.dimensions) * self.space_scale
        self.syncmap = ((self.syncmap - np.mean(self.syncmap, axis=0)) / (
                    np.std(self.syncmap, axis=0) + 1e-12)) * self.space_scale
        # print("self.syncmap:", self.syncmap)


        # adaptive learning rate
        self.adaptation_rate = adaptation_rate

        self.is_adaptive_LR = is_adaptive_LR
        self.adaptive_LR = 1
        self.adaptive_LR_widrow_hoff = adaptive_LR_widrow_hoff


        # symmetrical activation
        self.is_symmetrical_activation = is_symmetrical_activation
        self.number_of_selected_node = number_of_selected_node

        self.draw_3d_nodes = []
        self.center_plus = []
        self.center_minus = []

    def inputGeneral(self, x):
        plus = x > 0.1
        minus = ~ plus
        # print("plus:", plus)#(100000, 9)
        # print("minus:", minus)

        sequence_size = x.shape[0]
        for i in range(sequence_size):
            vplus = plus[i, :]
            vminus = minus[i, :]

            plus_mass = vplus.sum()
            minus_mass = vminus.sum()

            if plus_mass <= 1:
                continue

            if minus_mass <= 1:
                continue

            # Calculate cp, cn
            center_plus = np.dot(vplus, self.syncmap) / plus_mass
            center_minus = np.dot(vminus, self.syncmap) / minus_mass

            # Calculate distance
            dist_plus = distance.cdist(center_plus[None, :], self.syncmap, 'euclidean')
            dist_minus = distance.cdist(center_minus[None, :], self.syncmap, 'euclidean')
            dist_plus = np.transpose(dist_plus)
            dist_minus = np.transpose(dist_minus)

            # The increment of each variable coordinate [9, 2]
            update_plus = vplus[:, np.newaxis] * (
                    (center_plus - self.syncmap) / dist_plus)
            update_minus = vminus[:, np.newaxis] * (
                    (center_minus - self.syncmap) / dist_minus)

            update = update_plus - update_minus
            self.syncmap += self.adaptation_rate * update

            maximum = self.syncmap.max()
            self.syncmap = self.space_size * self.syncmap / maximum

            self.draw_3d_nodes.append(self.syncmap)

            # Update center
            self.center_plus.append(center_plus)
            self.center_minus.append(center_minus)

    def get_syncmap(self, isMovMean=False):
        if isMovMean:
            self.syncmap_movmean = np.mean(np.asarray(self.syncmap_movmean_list), axis=0)
            return self.syncmap_movmean, np.array(self.draw_3d_nodes), np.array(self.center_plus), np.array(
                self.center_minus)
        else:
            return self.syncmap, np.array(self.draw_3d_nodes), np.array(self.center_plus), np.array(self.center_minus)

    def process_input(self, input_seq, current_state=None):
        for i_state in range(len(input_seq)):
            state = input_seq[i_state]
            # print("state:", state)
            current_state_idx = current_state[i_state] if current_state is not None else None
            self.adapt_chunking(state)  # Do the calculation
            if i_state % self.movmean_interval == 0 or i_state == len(input_seq) - 1:
                self.syncmap_movmean_list.append(self.syncmap.copy())

        self.syncmap_movmean = np.mean(np.asarray(self.syncmap_movmean_list), axis=0)

    # Core function: Do the calculation
    def adapt_chunking(self, input_state_vec):
        syncmap_previous = self.syncmap.copy()

        set_positive = input_state_vec == True
        set_negative = input_state_vec == False

        # symmetrical activation
        if self.is_symmetrical_activation:
            set_positive, set_negative = self.symmetrical_activation(set_positive.copy())

        if set_positive.sum() <= 1 or set_negative.sum() <= 1:
            return syncmap_previous

        centroid_positive = np.dot(set_positive, self.syncmap) / set_positive.sum()
        centroid_negative = np.dot(set_negative, self.syncmap) / set_negative.sum()

        dist_set2centroid_positive = distance.cdist(centroid_positive[None, :], self.syncmap, 'euclidean').T
        dist_set2centroid_negative = distance.cdist(centroid_negative[None, :], self.syncmap, 'euclidean').T

        # get dropout mask
        isDropout_positive = np.random.rand() > self.dropout_positive
        isDropout_negative = np.random.rand() > self.dropout_negative

        # update syncmap
        update_positive = set_positive[:, np.newaxis] * (
                centroid_positive - self.syncmap) / dist_set2centroid_positive
        update_negative = set_negative[:, np.newaxis] * (
                centroid_negative - self.syncmap) / dist_set2centroid_negative

        # adaptive learning rate regularization
        if self.is_adaptive_LR:
            adaptive_LR_positive, adaptive_LR_negative = self.update_adaptive_learning_rate(
                dist_set2centroid_positive, set_positive)
            update_positive = update_positive * adaptive_LR_positive
            update_negative = update_negative * adaptive_LR_negative

        self.syncmap += self.adaptation_rate * (
                update_positive * isDropout_positive - update_negative * isDropout_negative)

        # regularization
        # normalize the syncmap to have 0 mean and 1 std
        # self.syncmap = ((self.syncmap - np.mean(self.syncmap, axis=0)) / (np.std(self.syncmap, axis=0) + 1e-12)) * self.space_scale
        self.syncmap = (self.syncmap / self.syncmap.max()) * self.space_scale

        # leaking
        self.syncmap = self.leaking_rate * self.syncmap + (1 - self.leaking_rate) * syncmap_previous
        # print("self.syncmap:", self.syncmap)

        self.center_plus.append(centroid_positive)
        self.center_minus.append(centroid_negative)
        self.draw_3d_nodes.append(self.syncmap)

        return self.syncmap

    def symmetrical_activation(self, input_vector):
        state_vector_plus = input_vector.copy()

        ## version 2 ##
        # Consider those positive nodes which are not selected in positive stochastic selection process
        number_of_selected_node_temp = state_vector_plus.sum()
        state_vector_plus = self.stochastic_selection(input_vector=state_vector_plus,
                                                      number_of_selected_node_overwrite=number_of_selected_node_temp)
        # update number_of_selected_node_temp
        number_of_selected_node_temp = state_vector_plus.sum()
        state_vector_minus = self.stochastic_selection(input_vector=~state_vector_plus,
                                                       number_of_selected_node_overwrite=number_of_selected_node_temp)

        return state_vector_plus, state_vector_minus

    def stochastic_selection(self, input_vector, number_of_selected_node_overwrite=None):
        number_of_activated_node = input_vector.sum()

        if self.number_of_selected_node is None:
            number_of_selected_node = number_of_selected_node_overwrite
        else:
            number_of_selected_node = self.number_of_selected_node

        if number_of_activated_node == 0:
            Pr = 0
        elif number_of_selected_node < number_of_activated_node:  # randomly select a subset of nodes from the activated nodes
            Pr = number_of_selected_node / number_of_activated_node
        else:  # select all the activated nodes
            Pr = 1

        state_vector_after_masking = np.logical_and(input_vector, np.random.rand(len(input_vector)) < Pr)
        return state_vector_after_masking

    def update_adaptive_learning_rate(self, dist_set2centroid_positive, set_positive):
        dist_avg_positive = np.sum(dist_set2centroid_positive * set_positive[:, None]) / set_positive.sum()
        adaptive_LR_positive = dist_avg_positive / self.space_scale_dimensions_sqrt

        ### version 4:
        self.adaptive_LR += self.adaptive_LR_widrow_hoff * (adaptive_LR_positive - self.adaptive_LR)
        adaptive_LR_positive = copy.deepcopy(self.adaptive_LR)
        adaptive_LR_negative = 1

        adaptive_LR_negative_amplifier_a = 0.01
        adaptive_LR_negative_amplifier_b = 2
        adaptive_LR_negative_amplifier = self.input_size * adaptive_LR_negative_amplifier_a + adaptive_LR_negative_amplifier_b
        adaptive_LR_negative_threshold = 1.5
        adaptive_LR_negative = self.adaptive_LR * adaptive_LR_negative_amplifier
        if adaptive_LR_negative > adaptive_LR_negative_threshold:
            adaptive_LR_negative = adaptive_LR_negative_threshold

        adaptive_LR_positive_threshold = 0.05
        if adaptive_LR_positive < adaptive_LR_positive_threshold:
            return adaptive_LR_positive_threshold, adaptive_LR_negative
        else:
            return adaptive_LR_positive, adaptive_LR_negative

    def set_adaptation_rate(self, adaptation_rate):
        self.adaptation_rate = adaptation_rate

    def organize(self):
        # print("self.syncmap:", self.syncmap.shape)#(30, 3)
        self.organized = True
        self.labels = DBSCAN(eps=3, min_samples=2).fit_predict(self.syncmap)
        return self.labels, np.array(self.draw_3d_nodes), np.array(self.center_plus), np.array(self.center_minus)

    def activate(self, x):
        '''
        Return the label of the index with maximum input value
        '''
        if self.organized == False:
            print("Activating a non-organized SyncMap")
            return

        # maximum output
        max_index = np.argmax(x)

        return self.labels[max_index]

    def plotSequence(self, input_sequence, input_class, filename="plot.png"):
        input_sequence = input_sequence[1:500]
        input_class = input_class[1:500]

        a = np.asarray(input_class)
        t = [i for i, value in enumerate(a)]
        c = [self.activate(x) for x in input_sequence]

        plt.plot(t, a, '-g')
        plt.plot(t, c, '-.k')

        plt.savefig(filename, quality=1, dpi=300)
        plt.show()
        plt.close()

    def plot(self, color=None, save=False, filename="plot_map.png"):
        if color is None:
            color = self.labels
        # print(self.syncmap)

        if self.dimensions == 2:
            ax = plt.scatter(self.syncmap[:, 0], self.syncmap[:, 1], c=color)

        if self.dimensions == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            ax.scatter3D(self.syncmap[:, 0], self.syncmap[:, 1], self.syncmap[:, 2], c=color)

        if save == True:
            plt.savefig(filename)

        plt.show()
        plt.close()

    def save(self, filename):
        """save class as self.name.txt"""
        file = open(filename + '.txt', 'w')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self, filename):
        """try load self.name.txt"""
        file = open(filename + '.txt', 'r')
        dataPickle = file.read()
        file.close()

        self.__dict__ = pickle.loads(dataPickle)
