import numpy
from library import TNclass
import library.wheel_functions as wf


class MPS(TNclass.TensorNetwork):
    def __init__(self):
        # Prepare parameters
        TNclass.TensorNetwork.__init__(self)

    def mps_regularization(self, regular_center):
        if regular_center == -1:
            regular_center = self.tensor_info['n_length']-1
        if self.tensor_info['regular_center'] == 'unknown':
            self.tensor_info['regular_center'] = 0
            while self.tensor_info['regular_center'] < self.tensor_info['n_length']-1:
                self.move_regular_center2next()
        while self.tensor_info['regular_center'] < regular_center:
            self.move_regular_center2next()
        while self.tensor_info['regular_center'] > regular_center:
            self.move_regular_center2forward()

    def move_regular_center2next(self):
        tensor_index = self.tensor_info['regular_center']
        if self.tensor_info['move_label_mode'] == 'off':
            u, s, v = wf.tensor_svd(self.tensor_data[tensor_index], index_right=2)
        if self.tensor_info['normalization_mode'] == 'on':
            s /= numpy.linalg.norm(s)
        s = s[numpy.where(s > self.tensor_info['cutoff'])]
        dimension_middle = min([len(s), self.tensor_info['regular_bond_dimension']])
        u = u[:, 0:dimension_middle]
        s = s[0:dimension_middle]
        v = v[0:dimension_middle, :]
        self.tensor_data[tensor_index] = u.reshape(
            self.tensor_data[tensor_index].shape[0], self.tensor_data[tensor_index].shape[1], dimension_middle)
        self.tensor_data[tensor_index+1] = wf.tensor_contract(
            numpy.dot(numpy.diag(s), v), self.tensor_data[tensor_index+1], [-1, 0]).reshape(
            dimension_middle,
            self.tensor_data[tensor_index+1].shape[1],
            self.tensor_data[tensor_index+1].shape[2])
        self.tensor_info['regular_center'] += 1

    def move_regular_center2forward(self):
        tensor_index = self.tensor_info['regular_center']
        if self.tensor_info['move_label_mode'] == 'off':
            u, s, v = wf.tensor_svd(self.tensor_data[tensor_index], index_left=0)
        if self.tensor_info['normalization_mode'] == 'on':
            s /= numpy.linalg.norm(s)
        s = s[numpy.where(s > self.tensor_info['cutoff'])]
        dimension_middle = min([len(s), self.tensor_info['regular_bond_dimension']])
        u = u[:, 0:dimension_middle]
        s = s[0:dimension_middle]
        v = v[0:dimension_middle, :]
        self.tensor_data[tensor_index] = v.reshape(
            dimension_middle, self.tensor_data[tensor_index].shape[1], self.tensor_data[tensor_index].shape[2])
        self.tensor_data[tensor_index-1] = wf.tensor_contract(
            self.tensor_data[tensor_index-1], numpy.dot(u, numpy.diag(s)), [-1, 0]).reshape(
            self.tensor_data[tensor_index - 1].shape[0],
            self.tensor_data[tensor_index - 1].shape[1],
            dimension_middle)
        self.tensor_info['regular_center'] -= 1

    def measure_mps(self, operator=numpy.diag([1, -1])):
        # testing code
        measure_data = numpy.zeros(self.tensor_info['n_length'])
        if self.tensor_info['regular_center'] != 0:
            self.mps_regularization(-1)
            self.mps_regularization(0)
        self.tensor_data[0] /= numpy.linalg.norm(self.tensor_data[0])
        for ii in range(self.tensor_info['n_length']):
            measure_data[ii] = wf.tensor_contract(
                wf.tensor_contract(self.tensor_data[ii], self.tensor_data[ii], [[0, -1], [0, -1]]),
                operator, [[0, 1], [1, 0]])
            if ii < self.tensor_info['n_length'] - 1:
                self.move_regular_center2next()
        return measure_data

    def measure_images_from_mps(self):
        # testing code
        probability = numpy.empty((self.tensor_info['n_length'], 2))
        state = numpy.empty((self.tensor_info['n_length'], 2, 2))
        if self.tensor_info['regular_center'] != 0:
            self.mps_regularization(-1)
            self.mps_regularization(0)
        self.tensor_data[0] /= numpy.linalg.norm(self.tensor_data[0])
        for ii in range(self.tensor_info['n_length']):
            probability[ii], state[ii] = numpy.linalg.eigh(
                wf.tensor_contract(self.tensor_data[ii], self.tensor_data[ii], [[0, -1], [0, -1]]))
            if ii < self.tensor_info['n_length'] - 1:
                self.move_regular_center2next()
        return probability, state

    def reverse_mps(self):
        self.tensor_data.reverse()
        for ii in range(self.tensor_info['n_length']):
            self.tensor_data[ii] = self.tensor_data[ii].transpose(2, 1, 0)
        self.tensor_info['regular_center'] = self.tensor_info[
                                                 'n_length'] - self.tensor_info['regular_center'] - 1
