import numpy


def gtn(para=dict()):
    para.update(ml().copy())
    para.update(program().copy())
    para.update(mps().copy())
    para.update(feature_map().copy())
    para.update(training().copy())
    # Machine Learning Parameter
    para['classifier_type'] = 'GTN'
    para['dataset'] = 'mnist'
    # Program Parameter
    # MPS Parameter
    para['physical_bond'] = 2
    para['virtual_bond_limitation'] = 8
    para['mps_cutoff'] = -1
    para['mps_rand_seed'] = 1
    # Feature Map Parameter
    para['theta'] = (numpy.pi/2) / 255
    para['mapped_dimension'] = 2
    # Training Parameter
    para['training_label'] = [9]  # para['training_label'] should be a list
    para['n_training'] = 50  # an int or 'all'
    para['update_mode'] = 'one_dot'  # one_dot or two_dots
    para['converge_accuracy'] = 1e-2
    para['rand_index_mode'] = 'on'
    para['rand_index_seed'] = 1
    para['tensor_acc'] = 1e-7
    return para


def gtnc(para=dict()):
    para.update(ml().copy())
    para.update(program().copy())
    para.update(mps().copy())
    para.update(feature_map().copy())
    para.update(training().copy())
    # Machine Learning Parameter
    para['classifier_type'] = 'GTNC'
    para['dataset'] = 'mnist'
    # Program Parameter
    # MPS Parameter
    para['physical_bond'] = 2
    para['virtual_bond_limitation'] = 4
    para['mps_cutoff'] = -1
    para['mps_rand_seed'] = 1
    # Feature Map Parameter
    para['theta'] = (numpy.pi/2) / 255
    para['mapped_dimension'] = 2
    # Training Parameter
    # para['training_label'] = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]  # para['training_label'] should be a list
    para['training_label'] = [[3], [8]]
    para['n_training'] = 'all'  # an int or 'all'
    para['update_mode'] = 'one_dot'  # one_dot or two_dots
    para['converge_accuracy'] = 1e-2
    para['rand_index_mode'] = 'on'
    para['rand_index_seed'] = 1
    para['tensor_acc'] = 1e-7
    return para


def ml(para=dict()):
    para['dataset'] = 'mnist'
    para['path_dataset'] = './dataset/'
    para['data_type'] = ['train', 'test']
    para['sort_module'] = 'rand'
    para['divide_module'] = 'label'
    para['save_data_path'] = './data_trained/'
    return para


def program(para=dict()):
    para['gpu_mode'] = 'off'
    return para


def mps(para=dict()):
    para['physical_bond'] = 2
    para['virtual_bond_limitation'] = 8
    para['tensor_network_type'] = 'MPS'
    para['mps_cutoff'] = -1
    para['mps_normalization_mode'] = 'on'
    para['move_label_mode'] = 'off'
    para['tensor_initialize_type'] = 'rand'
    para['tensor_initialize_bond'] = 'max'
    para['mps_rand_seed'] = 1
    return para


def feature_map(para=dict()):
    para['map_module'] = "many_body_Hilbert_space"
    para['theta'] = (numpy.pi/2) / 255
    para['mapped_dimension'] = 2
    return para


def training(para=dict()):
    para['training_label'] = [[3], [8]]
    para['n_training'] = 40  # an int or 'all'
    para['update_step'] = 2e-1
    para['step_decay_rate'] = 5
    para['step_accuracy'] = 1e-4
    para['normalization_update_mode'] = 'on'
    para['update_mode'] = 'one_dot'  # one_dot or two_dots
    para['update_method'] = 'SZZ'
    para['converge_type'] = 'cost function'
    para['converge_accuracy'] = 2e-3
    para['rand_index_mode'] = 'on'
    para['rand_index_seed'] = 1
    para['tensor_acc'] = 1e-8
    return para

