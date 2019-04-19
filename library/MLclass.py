import numpy
import os
import math
import scipy.special
import time
from library import mnist_functions


class MachineLearning:

    def __init__(self, para, debug_mode=False):
        # initialize parameters
        self.para = para.copy()
        self.images_data = dict()
        self.index = dict()
        self.labels_data = dict()
        self.update_info = dict()
        self.tmp = {}
        self.generative_model = ['GTN']
        self.discriminative_model = ['DTNC']
        # Load the data set
        self.load_dataset()
        # Calculate the basic information of the data set
        self.calculate_dataset_info()
        # Arrange Data
        self.arrange_data()

    def start_learning(self, learning_loops=30):
        self.print_program_info(mode='start')
        self.prepare_start_learning()
        if self.update_info['is_converged'] == 'untrained':
            self.update_info['is_converged'] = False
        while self.update_info['loops_learned'] >= learning_loops:
            print('you have learnt too many loops')
            learning_loops = int(input("learning_loops = "))
        if self.update_info['loops_learned'] == 0:
            self.calculate_cost_function()
            self.update_info['cost_function_loops'].append(self.update_info['cost_function'])
            print('Initializing ... cost function = ' + str(self.update_info['cost_function']))
        print('start to learn to ' + str(learning_loops) + ' loops')
        while (self.update_info['loops_learned'] < learning_loops) and not(self.update_info['is_converged']):
            self.update_one_loop()
            self.is_converge()
            self.save_data()
        self.save_data()
        if self.update_info['is_converged']:
            self.print_converge_info()
        else:
            print('Training end, cost function = ' + str(self.update_info['cost_function']) + ', do not converge.')
        self.calculate_program_info_time(mode='end')
        self.print_program_info(mode='end')

    def load_dataset(self):
        if (self.para['dataset'] == 'mnist') or (self.para['dataset'] == 'fashion'):
            self.images_data['train'], self.labels_data['train'] = mnist_functions.\
                load_mnist(os.path.join(self.para['path_dataset'], self.para['dataset']), kind='train')
            self.images_data['test'], self.labels_data['test'] = mnist_functions.\
                load_mnist(os.path.join(self.para['path_dataset'], self.para['dataset']), kind='t10k')
            self.para['labels'] = sorted(set(self.labels_data['train']))

    def calculate_dataset_info(self):
        self.para['n_sample'] = {}
        for data_type in self.para['data_type']:
            self.para['n_sample'][data_type] = {}
            self.para['n_sample'][data_type] = (self.images_data[data_type]).shape[0]
        self.para['n_feature'] = (self.images_data['train']).shape[1]
        self.para['n_length'] = self.para['n_feature']
        for data_type in self.para['data_type']:
            self.index[data_type] = dict()
            self.index[data_type]['origin'] = numpy.arange(self.para['n_sample'][data_type])

    def divide_data(self, data_type='train'):
        self.index[data_type]['divided'] = dict()
        if self.para['divide_module'] == 'label':
            for label in self.para['labels']:
                self.index[data_type]['divided'][label] = numpy.where(label == self.labels_data[data_type])[0]

    def arrange_data(self):
        if self.para['sort_module'] == 'rand':
            self.images_data['train'], self.labels_data['train'] = \
                self.rand_sort_data(self.images_data['train'], self.labels_data['train'])
        self.divide_data(data_type='train')
        if self.para['classifier_type'] in self.generative_model:
            self.images_data['input'] = list()
            self.labels_data['input'] = list()
            for label in tuple(list(self.para['training_label'])):
                self.images_data['input'] += list((self.images_data['train'][
                                            self.index['train']['divided'][label], :]))
                self.labels_data['input'] += list((self.labels_data['train'][
                                        self.index['train']['divided'][label]]))
            self.images_data['input'] = numpy.array(self.images_data['input'])
            self.labels_data['input'] = numpy.array(self.labels_data['input'])
            self.images_data['input'], self.labels_data['input'] = \
                self.rand_sort_data(self.images_data['input'], self.labels_data['input'])
            if self.para['n_training'] == 'all':
                self.para['n_training'] = len(self.labels_data['input'])
            if self.para['n_training'] < len(self.labels_data['input']):
                self.images_data['input'] = self.images_data['input'][range(self.para['n_training']), :]
                self.labels_data['input'] = self.labels_data['input'][range(self.para['n_training'])]

    def feature_map(self, image_data_mapping):
        image_data_mapping = numpy.array(image_data_mapping)
        if self.para['map_module'] == 'many_body_Hilbert_space':
            image_data_mapping = image_data_mapping * self.para['theta']
            while numpy.ndim(image_data_mapping) < 2:
                image_data_mapping.shape = (1,) + image_data_mapping.shape
            image_data_mapped = numpy.zeros(image_data_mapping.shape + (self.para['mapped_dimension'],))
            for ii in range(self.para['mapped_dimension']):
                image_data_mapped[:, :, ii] = math.sqrt(
                    scipy.special.comb(self.para['mapped_dimension'] - 1, ii)) * (
                        numpy.sin(image_data_mapping) ** (self.para['mapped_dimension'] - ii - 1)) * (
                        numpy.cos(image_data_mapping) ** ii)
        return image_data_mapped

    def anti_feature_map(self, state):
        # test code
        state_shape = list(state.shape)
        state_shape.pop(-1)
        pixels = numpy.arcsin(numpy.abs(
            state.reshape(-1, self.para['mapped_dimension'])[:, 0])).reshape(state_shape) / self.para['theta']
        return pixels

    def rand_sort_data(self, image_data, image_label):
        numpy.random.seed(self.para['rand_index_seed'])
        rand_index = numpy.random.permutation(image_data.shape[0])
        image_data_rand_sorted = image_data[rand_index, :]
        image_label_rand_sorted = image_label[rand_index]
        return image_data_rand_sorted, image_label_rand_sorted

    def calculate_running_time(self, mode='end'):
        if mode == 'start':
            self.tmp['start_time_cpu'] = time.clock()
            self.tmp['start_time_wall'] = time.time()
        elif mode == 'end':
            self.tmp['end_time_cpu'] = time.clock()
            self.tmp['end_time_wall'] = time.time()
            self.update_info['cost_time_cpu'].append(self.tmp['end_time_cpu'] - self.tmp['start_time_cpu'])
            self.update_info['cost_time_wall'].append(self.tmp['end_time_wall'] - self.tmp['start_time_wall'])

    def print_running_time(self, print_type=('wall', 'cpu')):
        if ('cpu' in print_type) or ('cpu' == print_type):
            print('This loop consumes ' + str(self.tmp['end_time_cpu']
                                              - self.tmp['start_time_cpu']) + ' cpu seconds.')
        if ('wall' in print_type) or ('wall' == print_type):
            print('This loop consumes ' + str(self.tmp['end_time_wall']
                                              - self.tmp['start_time_wall']) + ' wall seconds.')

    def is_converge(self):
        if self.para['converge_type'] == 'cost function':
            loops_learned = self.update_info['loops_learned']
            cost_function_loops = self.update_info['cost_function_loops']
            self.update_info['is_converged'] = \
                ((cost_function_loops[loops_learned - 1] - cost_function_loops[loops_learned]) /
                 cost_function_loops[loops_learned - 1]) < self.para['converge_accuracy']
            if self.update_info['is_converged']:
                if self.update_info['step'] > self.para['step_accuracy']:
                    self.update_info['step'] /= self.para['step_decay_rate']
                    print('update step reduces to ' + str(self.update_info['step']))
                    self.update_info['is_converged'] = False

    def print_converge_info(self):
        print(self.para['converge_type'] + ' is converged at ' + str(self.update_info['cost_function'])
              + '. Program terminates')
        print('Train ' + str(self.update_info['loops_learned']) + ' loops')

    def generate_update_info(self):
        self.update_info['update_position'] = 'unknown'
        self.update_info['update_direction'] = +1
        self.update_info['loops_learned'] = 0
        self.update_info['cost_function_loops'] = list()
        self.update_info['cost_time_cpu'] = list()
        self.update_info['cost_time_wall'] = list()
        self.update_info['step'] = self.para['update_step']
        self.update_info['is_converged'] = 'untrained'
        self.update_info['update_mode'] = self.para['update_mode']


