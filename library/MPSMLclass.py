import numpy
import os
import library.wheel_functions as wf
from library import BasicFunctions_szz
import Parameters
from library import MPSclass
from library import MLclass
from library import Programclass


class GTN(MPSclass.MPS, MLclass.MachineLearning, Programclass.Program):
    def __init__(self, para=Parameters.gtn().copy(), debug_mode=False):
        # Initialize Parameters
        Programclass.Program.__init__(self)
        MLclass.MachineLearning.__init__(self, para, debug_mode)
        MPSclass.MPS.__init__(self)
        self.initialize_parameters_gtn()
        self.name_md5_generate()
        self.deug_mode = debug_mode
        # self.generate_contract_index()
        # Initialize info
        self.generate_tensor_info()
        self.generate_update_info()
        # Initialize MPS and update info
        if not debug_mode:
            self.load_gtn()
        if len(self.tensor_data) == 0:
            self.initialize_mps_gtn()
        # Environment Preparation
        self.tensor_input = self.feature_map(self.images_data['input'])
        self.environment_left = list(range(self.tensor_info['n_length']))
        self.environment_right = list(range(self.tensor_info['n_length']))
        self.environment_zoom = dict()

    def prepare_start_learning(self):
        self.initialize_environment()
        if self.update_info['loops_learned'] != 0:
            print('load mps trained ' + str(self.update_info['loops_learned']) + ' loops')

    def initialize_mps_gtn(self):
        if self.para['tensor_initialize_type'] == 'rand':
            numpy.random.seed(self.para['mps_rand_seed'])
        if self.para['tensor_initialize_type'] == 'rand':
            for ii in range(self.tensor_info['n_length']):
                self.tensor_data.append(numpy.random.rand(
                    self.para['tensor_initialize_bond'],
                    self.tensor_info['physical_bond'], self.para['tensor_initialize_bond']))
            ii = 0
            self.tensor_data[ii] = numpy.random.rand(
                1, self.tensor_info['physical_bond'], self.para['tensor_initialize_bond'])
            ii = -1
            self.tensor_data[ii] = numpy.random.rand(
                self.para['tensor_initialize_bond'], self.tensor_info['physical_bond'], 1)
        elif self.para['tensor_initialize_type'] == 'ones':
            for ii in range(self.tensor_info['n_length']):
                self.tensor_data.append(numpy.ones((
                    self.para['tensor_initialize_bond'],
                    self.tensor_info['physical_bond'], self.para['tensor_initialize_bond'])))
            ii = 0
            self.tensor_data[ii] = numpy.ones((
                1, self.tensor_info['physical_bond'], self.para['tensor_initialize_bond']))
            ii = -1
            self.tensor_data[ii] = numpy.ones((
                self.para['tensor_initialize_bond'], self.tensor_info['physical_bond'], 1))
        # Regularization
        self.mps_regularization(-1)
        self.mps_regularization(0)
        self.update_info['update_position'] = self.tensor_info['regular_center']

    # def generate_contract_index(self):
    #     self.tensor_info['contract_index'] = []
    #     for ii in range(self.tensor_info['n_length']):
    #         self.tensor_info['contract_index'].append([ii - 1, -1, ii + 1])
    #     self.tensor_info['contract_index'][0] = [-1, -1, 1]
    #     self.tensor_info['contract_index'][-1] = [self.tensor_info['n_length'] - 2, -1, -1]

    def generate_tensor_info(self):
        self.tensor_info['regular_center'] = 'unknown'
        self.tensor_info['normalization_mode'] = self.para['mps_normalization_mode']
        self.tensor_info['move_label_mode'] = self.para['move_label_mode']
        self.tensor_info['cutoff'] = self.para['mps_cutoff']
        self.tensor_info['regular_bond_dimension'] = self.para['virtual_bond_limitation']
        self.tensor_info['n_length'] = self.para['n_feature']
        self.tensor_info['physical_bond'] = self.para['physical_bond']

    def update_mps_once(self):
        # Calculate gradient
        tmp_index1 = self.tensor_info['regular_center']
        if self.update_info['update_mode'] == 'one_dot':
            tmp_tensor_current = self.tensor_data[tmp_index1]
            tmp_tensor1 = wf.outer_parallel(
                self.environment_left[tmp_index1],
                self.tensor_input[:, tmp_index1, :],
                self.environment_right[tmp_index1])
        elif self.update_info['update_mode'] == 'two_dots':
            tmp_index_left = min(0, self.update_info['update_direction']) + tmp_index1
            tmp_index_right = max(0, self.update_info['update_direction']) + tmp_index1
            tmp_tensor_current = wf.tensor_contract(
                self.tensor_data[tmp_index_left], self.tensor_data[tmp_index_right], [-1, 0])
            tmp_tensor1 = wf.outer_parallel(
                self.environment_left[tmp_index_left],
                self.tensor_input[:, tmp_index_left, :],
                self.tensor_input[:, tmp_index_right, :],
                self.environment_right[tmp_index_right])
        tmp_inner_product = numpy.dot(tmp_tensor1, tmp_tensor_current.flatten())
        tmp_tensor1 = numpy.dot((1/tmp_inner_product), tmp_tensor1).reshape(tmp_tensor_current.shape)
        self.tmp['gradient'] = 2 * (
                (self.tensor_data[tmp_index1] / (numpy.linalg.norm(self.tensor_data[tmp_index1]) ** 2))
                - tmp_tensor1 / self.para['n_training'])
        # Update MPS
        if self.para['update_method'] == 'SZZ':
            tmp_tensor_norm = numpy.linalg.norm(self.tmp['gradient'])
            if tmp_tensor_norm > self.para['tensor_acc']:
                tmp_tensor_current -= \
                    self.update_info['step'] * self.tmp['gradient'] / tmp_tensor_norm
        if self.update_info['update_mode'] == 'one_dot':
            self.tensor_data[self.update_info['update_position']] = tmp_tensor_current
        elif self.update_info['update_mode'] == 'two_dots':
            u, s, v = wf.tensor_svd(tmp_tensor_current, index_left=[0, 1])
            if self.tensor_info['normalization_mode'] == 'on':
                s /= numpy.linalg.norm(s)
            s = s[numpy.where(s > self.tensor_info['cutoff'])]
            dimension_middle = min([len(s), self.tensor_info['regular_bond_dimension']])
            u = u[:, 0:dimension_middle]
            s = s[0:dimension_middle]
            v = v[0:dimension_middle, :]
            if self.update_info['update_direction'] == 1:
                self.tensor_data[tmp_index1] = u.reshape(
                    self.tensor_data[tmp_index1].shape[0], 
                    self.tensor_data[tmp_index1].shape[1], dimension_middle)
                self.tensor_data[tmp_index1 + 1] = numpy.dot(numpy.diag(s), v).reshape(
                    dimension_middle,
                    self.tensor_data[tmp_index1 + 1].shape[1],
                    self.tensor_data[tmp_index1 + 1].shape[2])
            elif self.update_info['update_direction'] == -1:
                self.tensor_data[tmp_index1] = v.reshape(
                    dimension_middle, self.tensor_data[tmp_index1].shape[1], self.tensor_data[tmp_index1].shape[2])
                self.tensor_data[tmp_index1 - 1] = numpy.dot(u, numpy.diag(s)).reshape(
                    self.tensor_data[tmp_index1 - 1].shape[0],
                    self.tensor_data[tmp_index1 - 1].shape[1],
                    dimension_middle)
            self.tensor_info['regular_center'] += self.update_info['update_direction']

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
        self.tensor_data[0] /= numpy.linalg.norm(self.tensor_data[0])
        self.calculate_cost_function()
        print('cost function = ' + str(self.update_info['cost_function']))
        self.print_running_time()
        self.update_info['cost_function_loops'].append(self.update_info['cost_function'])
        self.update_info['loops_learned'] += 1

    def initialize_environment(self):
        self.environment_zoom['left'] = numpy.zeros((self.tensor_info['n_length'], self.para['n_training']))
        self.environment_zoom['right'] = numpy.zeros((self.tensor_info['n_length'], self.para['n_training']))
        ii = 0
        self.environment_left[ii] = numpy.ones(self.para['n_training'])
        self.environment_left[ii].shape = self.environment_left[ii].shape + (1,)
        ii = self.para['n_feature'] - 1
        self.environment_right[ii] = numpy.ones(self.para['n_training'])
        self.environment_right[ii].shape = self.environment_right[ii].shape + (1,)
        # for ii in range(self.para['n_feature'] - 1):
        #     self.calculate_environment_next(ii + 1)
        for ii in range(self.para['n_feature']-1, 0, -1):
            self.calculate_environment_forward(ii - 1)

    def calculate_cost_function(self):
        if self.update_info['update_position'] != 0:
            print('go check your code')
        tmp_matrix = numpy.dot(self.tensor_input[:, 0, :], self.tensor_data[0][0, :, :])
        tmp_inner_product = (tmp_matrix * self.environment_right[0]).sum(1)
        self.update_info['cost_function'] = 2 * numpy.log(numpy.linalg.norm(
            self.tensor_data[0])) - numpy.log(self.para['n_training']) - 2 * sum(
            self.environment_zoom['right'][0, :] + numpy.log(abs(tmp_inner_product))) / self.para['n_training']

    def calculate_environment_forward(self, environment_index):
        tmp_dimension = (self.tensor_data[environment_index]).shape[-1]
        self.environment_right[environment_index] = wf.tensor_contract(
            wf.outer_parallel(self.tensor_input[:, environment_index + 1, :],
                              self.environment_right[environment_index + 1])
            , self.tensor_data[environment_index + 1], [[1], [1, 2]])
        self.environment_zoom['right'][environment_index, :] = \
            self.environment_zoom['right'][environment_index + 1, :] + numpy.log(
                numpy.linalg.norm(self.environment_right[environment_index], axis=1))
        self.environment_right[environment_index] /= numpy.linalg.norm(
            self.environment_right[environment_index], axis=1).repeat(tmp_dimension).reshape(
            self.environment_right[environment_index].shape)

    def calculate_environment_next(self, environment_index):
        tmp_dimension = (self.tensor_data[environment_index]).shape[0]
        self.environment_left[environment_index] = wf.tensor_contract(
            wf.outer_parallel(self.environment_left[environment_index - 1],
                              self.tensor_input[:, environment_index - 1, :])
            , self.tensor_data[environment_index - 1], [[1], [0, 1]])
        self.environment_zoom['left'][environment_index, :] = \
            self.environment_zoom['left'][environment_index - 1, :] + numpy.log(
                numpy.linalg.norm(self.environment_left[environment_index], axis=1))
        self.environment_left[environment_index] /= numpy.linalg.norm(
            self.environment_left[environment_index], axis=1).repeat(tmp_dimension).reshape(
            self.environment_left[environment_index].shape)

    def calculate_inner_product(self, images_mapped):
        tmp_dv = self.tensor_info['physical_bond']
        n_images = images_mapped.shape[0]
        tmp_inner_product = numpy.ones(n_images)
        for jj in range(n_images):
            tmp_matrix = numpy.array([1])
            for ii in range(self.tensor_info['n_length']):
                tmp_matrix = numpy.dot(
                    tmp_matrix, self.tensor_data[ii].reshape(
                        len(tmp_matrix), -1)).reshape(tmp_dv, -1)
                tmp_matrix = numpy.dot(images_mapped[jj, ii, :], tmp_matrix)
            tmp_inner_product[jj] = tmp_matrix
        return tmp_inner_product

    def initialize_parameters_gtn(self):
        self.program_info['program_name'] = self.para['classifier_type']
        self.program_info['path_save'] = \
            self.para['save_data_path'] + self.program_info['program_name'] + '/'
        if self.para['tensor_initialize_bond'] == 'max':
            self.para['tensor_initialize_bond'] = self.para['virtual_bond_limitation']

    def load_gtn(self):
        self.name_md5_generate()
        load_path = self.program_info['path_save'] + self.program_info['save_name']
        if os.path.isfile(load_path):
            self.tensor_data, self.tensor_info, self.update_info = \
                BasicFunctions_szz.load_pr(load_path, ['tensor_data', 'tensor_info', 'update_info'])

    def save_data(self):
        self.name_md5_generate()
        if not self.deug_mode:
            BasicFunctions_szz.save_pr(
                self.program_info['path_save'],
                self.program_info['save_name'],
                [self.tensor_data, self.tensor_info, self.update_info],
                ['tensor_data', 'tensor_info', 'update_info'])


class GTNC(Programclass.Program, MLclass.MachineLearning):
    def __init__(self, para=Parameters.gtnc().copy(), debug_mode=False):
        # Initialize Parameters
        Programclass.Program.__init__(self)
        MLclass.MachineLearning.__init__(self, para, debug_mode)
        self.debug_mode = debug_mode
        self.initialize_parameters_gtnc()
        self.name_md5_generate()
        self.inner_product = dict()
        self.data_mapped = dict()
        self.right_label = dict()
        self.accuracy = dict()
        self.test_info = dict()
        self.accuracy_calculate_permission = False
        if not debug_mode:
            self.load_accuracy()

    def training_gtn(self):
        para = self.para.copy()
        para['classifier_type'] = 'GTN'
        for mps_label in self.para['training_label']:
            para['training_label'] = mps_label
            print('begin to train label = ' + str(mps_label))
            tmp = GTN(para.copy(), self.debug_mode)
            tmp.start_learning()

    def initialize_parameters_gtnc(self):
        self.program_info['program_name'] = self.para['classifier_type']
        self.program_info['path_save'] = \
            self.para['save_data_path'] + self.program_info['program_name'] + '/'
        if self.para['tensor_initialize_bond'] == 'max':
            self.para['tensor_initialize_bond'] = self.para['virtual_bond_limitation']

    def calculate_inner_product(self, data_type):
        self.inner_product[data_type] = dict()
        para = self.para.copy()
        para['classifier_type'] = 'GTN'
        for mps_label in self.para['training_label']:
            para['training_label'] = mps_label
            tmp = GTN(para.copy(), self.debug_mode)
            self.inner_product[data_type][str(mps_label)] = tmp.calculate_inner_product(
                self.data_mapped[data_type]['all'])

    def calculate_accuracy(self, data_type, force_mode='off'):
        self.print_program_info(mode='start')
        self.check_is_gtn_trained()
        self.test_info[data_type] = dict()
        if data_type in self.accuracy.keys():
            print('the accuracy has been calculated as ' + str(self.accuracy[data_type]))
            print(
                'test ' + str(self.test_info['n_test'])
                + ' images, got ' + str(self.test_info['n_right']) + ' right images')
            return self.accuracy[data_type]
        elif self.accuracy_calculate_permission or (force_mode == 'on'):
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
            if self.accuracy_calculate_permission:
                self.save_data()
            else:
                print('Warning!!! You are using forced mode. The result will not be saved.')
            return self.accuracy[data_type]
        elif (not self.accuracy_calculate_permission) and (force_mode == 'off'):
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
        self.divide_data(data_type)
        self.data_mapped[data_type] = dict()
        self.data_mapped[data_type]['all'] = list()
        for mps_label in self.para['training_label']:
            if data_type == 'test':
                self.data_mapped[data_type][str(mps_label)] = list()
                for label in mps_label:
                    self.data_mapped[data_type][str(mps_label)] += list(self.images_data['test'][
                        self.index[data_type]['divided'][label]])
                self.data_mapped[data_type][str(mps_label)] = self.feature_map(
                    numpy.array(self.data_mapped[data_type][str(mps_label)]))
            elif data_type == 'train':
                para = self.para.copy()
                para['classifier_type'] = 'GTN'
                para['training_label'] = mps_label
                tmp = GTN(para.copy())
                self.data_mapped[data_type][str(mps_label)] = tmp.tensor_input
            self.data_mapped[data_type]['all'] += list(self.data_mapped[data_type][str(mps_label)])
        self.data_mapped[data_type]['all'] = numpy.array(self.data_mapped[data_type]['all'])

    def check_is_gtn_trained(self):
        self.accuracy_calculate_permission = True
        para = self.para.copy()
        para['classifier_type'] = 'GTN'
        for mps_label in self.para['training_label']:
            para['training_label'] = mps_label
            tmp = GTN(para.copy(), self.debug_mode)
            if tmp.update_info['is_converged'] == 'untrained':
                self.accuracy_calculate_permission = False
                print('Warning!!! MPS of which label equals to ' + str(mps_label) + " hasn't trained.")
            elif not tmp.update_info['is_converged']:
                self.accuracy_calculate_permission = False
                print('Warning!!! MPS of which label equals to ' + str(mps_label) + ' still needs training')

    def load_accuracy(self):
        self.name_md5_generate()
        load_path = self.program_info['path_save'] + self.program_info['save_name']
        if os.path.isfile(load_path):
            self.accuracy, self.test_info = BasicFunctions_szz.load_pr(load_path, ['accuracy', 'test_info'])

    def save_data(self):
        self.name_md5_generate()
        BasicFunctions_szz.save_pr(
            self.program_info['path_save'],
            self.program_info['save_name'],
            [self.accuracy, self.test_info],
            ['accuracy', 'test_info'])



