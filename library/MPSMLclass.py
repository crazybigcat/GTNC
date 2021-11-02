import numpy
import os
import torch
import library.wheel_functions as wf
from library import BasicFunctions_szz
from library import Parameters
from library import MPSclass
from library import MLclass
from library import Programclass
import copy
import operator


class GTN(MPSclass.MPS, MLclass.MachineLearning, Programclass.Program):
    def __init__(self, para=Parameters.gtn(), debug_mode=False, device='cuda'):
        # Initialize Parameters
        Programclass.Program.__init__(self, device=device, dtype=para['dtype'])
        MLclass.MachineLearning.__init__(self, para, debug_mode)
        MPSclass.MPS.__init__(self)
        self.initialize_parameters_gtn()
        self.name_md5_generate()
        self.debug_mode = debug_mode
        # Initialize MPS and update info
        if not debug_mode:
            self.load_gtn()
        if len(self.tensor_data) == 0:
            self.initialize_dataset()
            # Initialize info
            self.generate_tensor_info()
            self.generate_update_info()
            self.initialize_mps_gtn()
        if not self.tensor_data[0].device == self.device:
            for ii in range(len(self.tensor_data)):
                self.tensor_data[ii] = torch.tensor(self.tensor_data[ii], device=self.device)
            torch.cuda.empty_cache()
        # Environment Preparation
        self.tensor_input = tuple()
        self.environment_left = tuple()
        self.environment_right = tuple()
        self.environment_zoom = tuple()

    def prepare_start_learning(self):
        if 'dealt_input' not in self.images_data.keys():
            self.initialize_dataset()
        if isinstance(self.tensor_input, tuple):
            self.tensor_input = self.feature_map(self.images_data['dealt_input'])
        self.environment_left = list(range(self.tensor_info['n_length']))
        self.environment_right = list(range(self.tensor_info['n_length']))
        self.environment_zoom = dict()
        self.initialize_environment()
        if self.update_info['loops_learned'] != 0:
            print('load mps trained ' + str(self.update_info['loops_learned']) + ' loops')

    def initialize_mps_gtn(self):
        if self.para['tensor_initialize_type'] == 'rand':
            torch.manual_seed(self.para['mps_rand_seed'])
        if self.para['tensor_initialize_type'] == 'rand':
            for ii in range(self.tensor_info['n_length']):
                self.tensor_data.append(torch.rand(
                    self.para['tensor_initialize_bond'],
                    self.tensor_info['physical_bond'],
                    self.para['tensor_initialize_bond'],
                    device=self.device, dtype=self.dtype))
            ii = 0
            self.tensor_data[ii] = torch.rand(
                1, self.tensor_info['physical_bond'],
                self.para['tensor_initialize_bond'],
                device=self.device, dtype=self.dtype)
            ii = -1
            self.tensor_data[ii] = torch.rand(
                self.para['tensor_initialize_bond'],
                self.tensor_info['physical_bond'], 1,
                device=self.device, dtype=self.dtype)
        elif self.para['tensor_initialize_type'] == 'ones':
            for ii in range(self.tensor_info['n_length']):
                self.tensor_data.append(torch.ones((
                    self.para['tensor_initialize_bond'],
                    self.tensor_info['physical_bond'], self.para['tensor_initialize_bond']),
                    device=self.device, dtype=self.dtype))
            ii = 0
            self.tensor_data[ii] = torch.ones((
                1, self.tensor_info['physical_bond'], self.para['tensor_initialize_bond']),
                device=self.device, dtype=self.dtype)
            ii = -1
            self.tensor_data[ii] = torch.ones((
                self.para['tensor_initialize_bond'], self.tensor_info['physical_bond'], 1),
                device=self.device, dtype=self.dtype)
        # Regularization
        self.mps_regularization(-1)
        self.mps_regularization(0)
        self.update_info['update_position'] = self.tensor_info['regular_center']

    def generate_tensor_info(self):
        self.tensor_info['regular_center'] = 'unknown'
        self.tensor_info['normalization_mode'] = self.para['mps_normalization_mode']
        self.tensor_info['move_label_mode'] = self.para['move_label_mode']
        self.tensor_info['cutoff'] = self.para['mps_cutoff']
        self.tensor_info['regular_bond_dimension'] = self.para['virtual_bond_limitation']
        self.tensor_info['n_length'] = self.data_info['n_feature']
        self.tensor_info['physical_bond'] = self.para['physical_bond']

    def update_mps_once(self):
        # Calculate gradient
        tmp_index1 = self.tensor_info['regular_center']
        tmp_tensor_current = self.tensor_data[tmp_index1]
        tmp_tensor1 = torch.einsum(
            'ni,nv,nj->nivj',
            self.environment_left[tmp_index1],
            self.tensor_input[:, tmp_index1, :],
            self.environment_right[tmp_index1]).reshape(self.data_info['n_training'], -1)
        tmp_inner_product = (tmp_tensor1.mm(tmp_tensor_current.view(-1, 1))).t()
        tmp_tensor1 = ((1/tmp_inner_product).mm(tmp_tensor1)).reshape(tmp_tensor_current.shape)
        self.tmp['gradient'] = 2 * (
                (tmp_tensor_current / (tmp_tensor_current.norm() ** 2))
                - tmp_tensor1 / self.data_info['n_training'])
        # Update MPS
        tmp_tensor_norm = (self.tmp['gradient']).norm()
        tmp_tensor_current -= self.update_info['step'] * self.tmp['gradient'] / (
                tmp_tensor_norm + self.para['tensor_acc'])
        self.tensor_data[self.update_info['update_position']] = tmp_tensor_current

    def update_one_loop(self):
        self.calculate_running_time(mode='start')
        if self.tensor_info['regular_center'] != 0:
            self.mps_regularization(-1)
            self.mps_regularization(0)
        self.update_info['update_position'] = self.tensor_info['regular_center']
        self.update_info['update_direction'] = +1
        while self.tensor_info['regular_center'] < self.tensor_info['n_length'] - 1:
            self.update_mps_once()
            self.mps_regularization(self.update_info['update_position'] + self.update_info['update_direction'])
            self.update_info['update_position'] = self.tensor_info['regular_center']
            self.calculate_environment_next(self.update_info['update_position'])
        self.update_info['update_direction'] = -1
        while self.tensor_info['regular_center'] > 0:
            self.update_mps_once()
            self.mps_regularization(self.update_info['update_position'] + self.update_info['update_direction'])
            self.update_info['update_position'] = self.tensor_info['regular_center']
            self.calculate_environment_forward(self.update_info['update_position'])
        self.calculate_running_time(mode='end')
        self.tensor_data[0] /= (self.tensor_data[0]).norm()
        self.calculate_cost_function()
        print('cost function = ' + str(self.update_info['cost_function'])
              + ' at ' + str(self.update_info['loops_learned'] + 1) + ' loops.')
        self.print_running_time()
        self.update_info['cost_function_loops'].append(self.update_info['cost_function'])
        self.update_info['loops_learned'] += 1

    def initialize_environment(self):
        self.environment_zoom['left'] = torch.zeros(
            (self.tensor_info['n_length'], self.data_info['n_training']),
            device=self.device, dtype=self.dtype)
        self.environment_zoom['right'] = torch.zeros(
            (self.tensor_info['n_length'], self.data_info['n_training']),
            device=self.device, dtype=self.dtype)
        ii = 0
        self.environment_left[ii] = torch.ones(self.data_info['n_training'], device=self.device, dtype=self.dtype)
        self.environment_left[ii].resize_(self.environment_left[ii].shape + (1,))
        ii = self.tensor_info['n_length'] - 1
        self.environment_right[ii] = torch.ones(self.data_info['n_training'], device=self.device, dtype=self.dtype)
        self.environment_right[ii].resize_(self.environment_right[ii].shape + (1,))
        # for ii in range(self.tensor_info['n_length'] - 1):
        #     self.calculate_environment_next(ii + 1)
        for ii in range(self.tensor_info['n_length']-1, 0, -1):
            self.calculate_environment_forward(ii - 1)

    def calculate_cost_function(self):
        if self.update_info['update_position'] != 0:
            print('go check your code')
        tmp_matrix = self.tensor_input[:, 0, :].mm(self.tensor_data[0][0, :, :])
        tmp_inner_product = ((tmp_matrix.mul(self.environment_right[0])).sum(1)).cpu()
        self.update_info['cost_function'] = 2 * numpy.log((
            self.tensor_data[0]).norm().cpu()) - numpy.log(self.data_info['n_training']) - 2 * sum(
            self.environment_zoom['right'][0, :].cpu() + numpy.log(abs(tmp_inner_product))) / self.data_info['n_training']

    def calculate_environment_forward(self, environment_index):
        self.environment_right[environment_index] = torch.einsum(
            'nv,ivj,nj->ni',
            self.tensor_input[:, environment_index + 1, :],
            self.tensor_data[environment_index + 1],
            self.environment_right[environment_index + 1])
        tmp_norm = (self.environment_right[environment_index]).norm(dim=1)
        self.environment_zoom['right'][environment_index, :] = \
            self.environment_zoom['right'][environment_index + 1, :] + torch.log(tmp_norm)
        self.environment_right[environment_index] = torch.einsum(
            'ij,i->ij', [self.environment_right[environment_index], 1/tmp_norm])

    def calculate_environment_next(self, environment_index):
        self.environment_left[environment_index] = torch.einsum(
            'ni,ivj,nv->nj',
            self.environment_left[environment_index - 1],
            self.tensor_data[environment_index - 1],
            self.tensor_input[:, environment_index - 1, :])
        tmp_norm = self.environment_left[environment_index].norm(dim=1)
        self.environment_zoom['left'][environment_index, :] = \
            self.environment_zoom['left'][environment_index - 1, :] + torch.log(tmp_norm)
        self.environment_left[environment_index] = torch.einsum(
            'ij,i->ij', self.environment_left[environment_index], 1/tmp_norm)

    def calculate_inner_product(self, images_mapped):
        n_images = images_mapped.shape[0]
        tmp_inner_product = torch.ones((n_images, 1), device=self.device, dtype=self.dtype)
        for ii in range(self.tensor_info['n_length']):
            tmp_inner_product = torch.einsum(
                'ni,ivj,nv->nj', tmp_inner_product, self.tensor_data[ii], images_mapped[:, ii, :])
        tmp_inner_product = tmp_inner_product.reshape(-1)
        return tmp_inner_product

    def initialize_parameters_gtn(self):
        self.program_info['program_name'] = self.para['classifier_type']
        self.program_info['path_save'] = \
            self.para['save_data_path'] + self.program_info['program_name'] + '/'
        if self.para['tensor_initialize_bond'] == 'max':
            self.para['tensor_initialize_bond'] = self.para['virtual_bond_limitation']

    def generate_image(self, known_indexes=tuple(), known_pixels=tuple(), generate_mode='average'):
        # testing code
        known_indexes, known_pixels = list(known_indexes), list(known_pixels)
        n_pixels = len(known_indexes)
        if operator.eq(list(range(n_pixels)), list(known_indexes)):
            if self.tensor_info['regular_center'] > n_pixels:
                self.mps_regularization(n_pixels)
            tmp_left_tensor = numpy.array([1])
            tmp_left_tensor.shape = (1, 1)
            index = 0
            while index < self.tensor_info['n_length']:
                if index >= n_pixels:
                    tmp_tensor = wf.tensor_contract(
                        tmp_left_tensor, self.tensor_data[index], [[1], [0]])
                    tmp_tensor /= numpy.linalg.norm(tmp_tensor)
                    probability, state = numpy.linalg.eigh(
                        wf.tensor_contract(tmp_tensor, tmp_tensor, [[0, -1], [0, -1]]))
                    pixels = self.anti_feature_map(state)
                    if generate_mode == 'max':
                        # stupid code
                        probability[numpy.where(probability == probability.min())] = 0
                        probability /= probability.max()
                    known_pixels.append((numpy.array(probability) * numpy.array(pixels)).sum())
                tmp_left_tensor = numpy.dot(
                    tmp_left_tensor,
                    wf.tensor_contract(
                        self.tensor_data[index],
                        self.feature_map([known_pixels[index]]).flatten(),
                        [[1], [0]]))
                tmp_left_tensor /= numpy.linalg.norm(tmp_left_tensor)
                index += 1
            return numpy.array(known_pixels)
        else:
            print('only work when i know first n pixels')

    def load_gtn(self):
        load_path = self.program_info['path_save'] + self.program_info['save_name']
        if os.path.isfile(load_path):
            self.tensor_data, self.tensor_info, self.update_info = \
                BasicFunctions_szz.load_pr(load_path, ['tensor_data', 'tensor_info', 'update_info'])

    def save_data(self):
        if not self.debug_mode:
            BasicFunctions_szz.save_pr(
                self.program_info['path_save'],
                self.program_info['save_name'],
                [self.tensor_data, self.tensor_info, self.update_info],
                ['tensor_data', 'tensor_info', 'update_info'])


class GTNC(Programclass.Program, MLclass.MachineLearning):
    def __init__(self, para=Parameters.gtnc(), debug_mode=False, device='cuda'):
        # Initialize Parameters
        Programclass.Program.__init__(self, device=device, dtype=para['dtype'])
        MLclass.MachineLearning.__init__(self, para, debug_mode)
        self.debug_mode = debug_mode
        self.initialize_parameters_gtnc()
        self.name_md5_generate()
        self.inner_product = dict()
        self.data_mapped = dict()
        self.right_label = dict()
        self.accuracy = dict()
        self.test_info = dict()
        self.is_all_gtn_trained = False
        if not self.debug_mode:
            self.load_accuracy()

    def training_gtn(self, learning_loops=30):
        para = copy.deepcopy(self.para)
        para['classifier_type'] = 'GTN'
        for mps_label in self.para['training_label']:
            para['training_label'] = mps_label
            print('begin to train label = ' + str(mps_label))
            tmp = GTN(copy.deepcopy(para), self.debug_mode, device=self.device)
            tmp.start_learning(learning_loops=learning_loops)

    def initialize_parameters_gtnc(self):
        self.program_info['program_name'] = self.para['classifier_type']
        self.program_info['path_save'] = \
            self.para['save_data_path'] + self.program_info['program_name'] + '/'
        if self.para['tensor_initialize_bond'] == 'max':
            self.para['tensor_initialize_bond'] = self.para['virtual_bond_limitation']

    def calculate_inner_product(self, data_type):
        self.inner_product[data_type] = dict()
        para = copy.deepcopy(self.para)
        para['classifier_type'] = 'GTN'
        for mps_label in self.para['training_label']:
            para['training_label'] = mps_label
            tmp = GTN(copy.deepcopy(para), self.debug_mode, device=self.device)
            self.inner_product[data_type][str(mps_label)] = tmp.calculate_inner_product(
                self.data_mapped[data_type]['all']).cpu().numpy()

    def calculate_accuracy(self, data_type='test', force_mode='off'):
        self.print_program_info(mode='start')
        if data_type in self.accuracy.keys():
            print('the accuracy has been calculated as ' + str(self.accuracy[data_type]))
            print(
                'test ' + str(self.test_info['n_test'])
                + ' images, got ' + str(self.test_info['n_right']) + ' right images')
            return self.accuracy[data_type]
        else:
            self.check_is_gtn_trained()
            self.test_info[data_type] = dict()
            if self.is_all_gtn_trained or (force_mode == 'on'):
                self.generate_data_mapped(data_type)
                self.calculate_inner_product(data_type)
                self.generate_right_label(data_type)
                tmp_test_inner_product = []
                for mps_label in self.para['training_label']:
                    tmp_test_inner_product.append(self.inner_product[data_type][str(mps_label)])
                tmp_test_inner_product = numpy.abs(numpy.array(tmp_test_inner_product)).argmax(axis=0)
                tmp_right_number = len(numpy.argwhere(tmp_test_inner_product == self.right_label[data_type]))
                self.test_info['n_test'] = self.right_label[data_type].shape[0]
                self.test_info['n_right'] = tmp_right_number
                self.accuracy[data_type] = tmp_right_number/tmp_test_inner_product.shape[0]
                print(
                    'test ' + str(self.test_info['n_test'])
                    + ' images, got ' + str(self.test_info['n_right']) + ' right images')
                print('the accuracy is ' + str(self.accuracy[data_type]))
                if self.is_all_gtn_trained:
                    self.save_data()
                else:
                    print('Warning!!! You are using forced mode. The result will not be saved.')
                return self.accuracy[data_type]
            elif (not self.is_all_gtn_trained) and (force_mode == 'off'):
                print("Warning!!! You can not calculate accuracy. Try turn force_mode = 'on' to force it run.")
                return 'None'
        self.print_program_info(mode='end')

    def generate_right_label(self, data_type):
        self.right_label[data_type] = []
        tmp_label = 0
        for mps_label in self.para['training_label']:
            for label in mps_label:
                if data_type == 'test':
                    self.right_label[data_type] += list(tmp_label *
                                                        numpy.ones(self.index['test']['divided'][label].shape))
                elif data_type == 'train':
                    self.right_label[data_type] += list(
                        tmp_label * numpy.ones(self.data_mapped[data_type][str(mps_label)].shape[0]))
            tmp_label += 1
        self.right_label[data_type] = numpy.array(self.right_label[data_type])

    def generate_data_mapped(self, data_type):
        self.initialize_dataset()
        self.divide_data(data_type)
        self.data_mapped[data_type] = dict()
        self.data_mapped[data_type]['all'] = list()
        for mps_label in self.para['training_label']:
            if data_type == 'test':
                self.data_mapped[data_type][str(mps_label)] = list()
                for label in mps_label:
                    self.data_mapped[data_type][str(mps_label)] += list(self.deal_data(
                        self.images_data['test'][self.index[data_type]['divided'][label]]))
                self.data_mapped[data_type][str(mps_label)] = (
                    self.feature_map(numpy.array(
                        self.data_mapped[data_type][str(mps_label)]))).cpu().numpy()
            elif data_type == 'train':
                para = copy.deepcopy(self.para)
                para['classifier_type'] = 'GTN'
                para['training_label'] = mps_label
                tmp = GTN(copy.deepcopy(para), device=self.device)
                                if 'dealt_input' not in tmp.images_data.keys():
                    tmp.initialize_dataset()
                if isinstance(tmp.tensor_input, tuple):
                    tmp.tensor_input = tmp.feature_map(tmp.images_data['dealt_input'])
                self.data_mapped[data_type][str(mps_label)] = tmp.tensor_input.cpu().numpy()
            self.data_mapped[data_type]['all'] += list(self.data_mapped[data_type][str(mps_label)])
        self.data_mapped[data_type]['all'] = torch.tensor(
            self.data_mapped[data_type]['all'], device=self.device, dtype=self.dtype)

    def check_is_gtn_trained(self):
        self.is_all_gtn_trained = True
        para = copy.deepcopy(self.para)
        para['classifier_type'] = 'GTN'
        for mps_label in self.para['training_label']:
            para['training_label'] = mps_label
            tmp = GTN(copy.deepcopy(para), self.debug_mode, device=self.device)
            if tmp.update_info['is_converged'] == 'untrained':
                self.is_all_gtn_trained = False
                print('Warning!!! MPS of which label equals to ' + str(mps_label) + " hasn't trained.")
            elif not tmp.update_info['is_converged']:
                self.is_all_gtn_trained = False
                print('Warning!!! MPS of which label equals to ' + str(mps_label) + ' still needs training')

    def load_accuracy(self):
        load_path = self.program_info['path_save'] + self.program_info['save_name']
        if os.path.isfile(load_path):
            self.accuracy, self.test_info = BasicFunctions_szz.load_pr(load_path, ['accuracy', 'test_info'])

    def save_data(self):
        BasicFunctions_szz.save_pr(
            self.program_info['path_save'],
            self.program_info['save_name'],
            [self.accuracy, self.test_info],
            ['accuracy', 'test_info'])

