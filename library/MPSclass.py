import numpy
import torch
import operator
from library import TNclass
import library.wheel_functions as wf
import copy


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
        # self.reverse_mps()
        # self.move_regular_center2forward()
        # self.reverse_mps()
        tensor_index = self.tensor_info['regular_center']
        u, s, v = wf.tensor_svd(self.tensor_data[tensor_index], index_right=2)
        if self.tensor_info['normalization_mode'] == 'on':
            s /= s.norm()
        s = s[s > self.tensor_info['cutoff']]
        dimension_middle = min([len(s), self.tensor_info['regular_bond_dimension']])
        u = u[:, 0:dimension_middle].reshape(-1, dimension_middle)
        s = s[0:dimension_middle]
        v = v[:, 0:dimension_middle].reshape(-1, dimension_middle)
        self.tensor_data[tensor_index] = u.reshape(
            self.tensor_data[tensor_index].shape[0],
            self.tensor_data[tensor_index].shape[1], dimension_middle)
        self.tensor_data[tensor_index + 1] = torch.einsum(
            'ij,jkl->ikl',
            [(torch.diag(s)).mm(v.t()), self.tensor_data[tensor_index+1]])
        self.tensor_info['regular_center'] += 1

    def move_regular_center2forward(self):
        # self.reverse_mps()
        # self.move_regular_center2next()
        # self.reverse_mps()
        tensor_index = self.tensor_info['regular_center']
        u, s, v = wf.tensor_svd(self.tensor_data[tensor_index], index_left=0)
        if self.tensor_info['normalization_mode'] == 'on':
            s /= s.norm()
        s = s[s > self.tensor_info['cutoff']]
        dimension_middle = min([len(s), self.tensor_info['regular_bond_dimension']])
        u = u[:, 0:dimension_middle].reshape(-1, dimension_middle)
        s = s[0:dimension_middle]
        v = v[:, 0:dimension_middle].reshape(-1, dimension_middle)
        self.tensor_data[tensor_index] = v.t().reshape(
            dimension_middle, self.tensor_data[tensor_index].shape[1],
            self.tensor_data[tensor_index].shape[2])
        self.tensor_data[tensor_index - 1] = torch.einsum(
            'ijk,kl->ijl',
            [self.tensor_data[tensor_index - 1], u.mm(torch.diag(s))])
        self.tensor_info['regular_center'] -= 1

    def measure_mps(self, operator=numpy.diag([1, -1])):
        # testing code
        measure_data = numpy.zeros(self.tensor_info['n_length'])
        if self.tensor_info['regular_center'] != 0:
            self.mps_regularization(-1)
            self.mps_regularization(0)
        self.tensor_data[0] /= self.tensor_data[0].norm()
        for ii in range(self.tensor_info['n_length']):
            measure_data[ii] = wf.tensor_contract(
                wf.tensor_contract(self.tensor_data[ii], self.tensor_data[ii], [[0, -1], [0, -1]]),
                operator, [[0, 1], [1, 0]])
            if ii < self.tensor_info['n_length'] - 1:
                self.move_regular_center2next()
        return measure_data

    def measure_images_from_mps(self):
        # testing code
        probability = torch.empty((self.tensor_info['n_length'], 2))
        state = torch.empty((self.tensor_info['n_length'], 2, 2))
        if self.tensor_info['regular_center'] != 0:
            self.mps_regularization(-1)
            self.mps_regularization(0)
        self.tensor_data[0] /= (self.tensor_data[0]).norm()
        for ii in range(self.tensor_info['n_length']):
            probability[ii], state[ii] = torch.symeig(
                torch.einsum('ivj,iwj->vw', self.tensor_data[ii], self.tensor_data[ii]),
                eigenvectors=True)
            if ii < self.tensor_info['n_length'] - 1:
                self.move_regular_center2next()
        return probability, state

    def calculate_single_entropy(self, dot_index='all'):
        if operator.eq(dot_index, 'all'):
            dot_index = list(range(self.tensor_info['n_length']))
        elif isinstance(dot_index, int):
            dot_index = [dot_index]
        entropy = dict()
        for ii in dot_index:
            self.mps_regularization(ii)
            tmp_tensor = copy.deepcopy(self.tensor_data[ii])
            tmp_tensor = tmp_tensor / numpy.linalg.norm(tmp_tensor)
            tmp_tensor = wf.tensor_contract(tmp_tensor, tmp_tensor, [[0, -1], [0, -1]])
            tmp_spectrum = numpy.linalg.eigh(tmp_tensor)[0]
            tmp_spectrum /= numpy.sum(tmp_spectrum)
            tmp_spectrum[tmp_spectrum <= 0] = 1
            entropy[ii] = abs((tmp_spectrum * numpy.log(tmp_spectrum)).sum())
        return entropy
