import pynvml
import os
import pickle
import hashlib
import operator
import torch
import datetime


def info_contact():
    """Return the information of the contact"""
    info = dict()
    info['name'] = 'Sun Zhengzhi'
    info['email'] = 'sunzhengzhi16@mails.ucas.ac.cn'
    info['affiliation'] = 'University of Chinese Academy of Sciences'
    return info
# These are from the original functions of Sun Zhengzhi


def get_best_gpu(device='cuda'):
    if isinstance(device, torch.device):
        return device
    elif device == 'cuda':
        pynvml.nvmlInit()
        num_gpu = pynvml.nvmlDeviceGetCount()
        memory_gpu = torch.zeros(num_gpu)
        for index in range(num_gpu):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_gpu[index] = memory_info.free
        max_gpu = int(torch.sort(memory_gpu, )[1][-1])
        return torch.device('cuda:' + str(max_gpu))
    elif device == 'cpu':
        return torch.device('cpu')


def sort_dict(a):
    b = dict()
    dict_index = sorted(a.keys())
    for index in dict_index:
        b[index] = a[index]
    return b


def issubdict(a, b):
    # check keys
    if set(a.keys()).issubset(set(b.keys())):
        flag = True
        for key in a.keys():
            if not operator.eq(a[key], b[key]):
                flag = False
        return flag
    else:
        return False


def save_pr_add_data(path, file, data, names):
    mkdir(path)
    if os.path.isfile(path+file):
        tmp = load_pr(path+file)
    else:
        tmp = {}
    s = open(path + file, 'wb')
    for ii in range(0, len(names)):
        tmp[names[ii]] = data[ii]
    pickle.dump(tmp, s)
    s.close()


def name_generator_md5(path, file, input_parameter, rough_mode='off'):
    file_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
    file_save = file + '_' + file_time
    file_path = path + 'code_book/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    tmp_save = dict()
    input_parameter = sort_dict(input_parameter)
    number_md5 = hashlib.md5(str(input_parameter).encode(encoding='utf-8')).hexdigest()
    tmp_save[number_md5] = input_parameter
    save_pr(file_path, file_save, [input_parameter], [number_md5])
    return number_md5


def integrate_codebook(path, file):
    path = path + 'code_book/'
    if not os.path.exists(path):
        os.makedirs(path)
    all_filename = os.listdir(path)
    tmp_save = dict()
    for filename in all_filename:
        if file in filename:
            tmp_load = load_pr(path + filename)
            tmp_save.update(tmp_load)
            os.remove(path + filename)
    save_pr(path, (file + '_codebook'), list(tmp_save.values()), list(tmp_save.keys()))


# These are from BasicFunctions of Ran Shi_ju with some changes
def info_contact_ran():
    """Return the information of the contact"""
    info = dict()
    info['name'] = 'S.J. Ran'
    info['email'] = 'ranshiju10@mail.s ucas.ac.cn'
    info['affiliation'] = 'ICFO – The Institute of Photonic Sciences'
    return info

def save_pr(path, file, data, names):
    """
    Save the data as a dict in a file located in the path
    :param path: the location of the saved file
    :param file: the name of the file
    :param data: the data to be saved
    :param names: the names of the data
    Notes: 1. Conventionally, use the suffix \'.pr\'. 2. If the folder does not exist, system will
    automatically create one. 3. use \'load_pr\' to load a .pr file
    Example:
    >>> x = 1
    >>> y = 'good'
    >>> save_pr('/test', 'ok.pr', [x, y], ['name1', 'name2'])
      You have a file '/test/ok.pr'
    >>> z = load_pr('/test/ok.pr')
      z = {'name1': 1, 'name2': 'good'}
      type(z) is dict
    """
    mkdir(path)
    # print(os.path.join(path, file))
    s = open(path+file, 'wb')
    tmp = dict()
    for i in range(0, len(names)):
        tmp[names[i]] = data[i]
    pickle.dump(tmp, s)
    s.close()


def load_pr(path_file, names=None):
    """
    Load the file saved by save_pr as a dict from path
    :param path_file: the path and name of the file
    :param names: the specific names of the data you want to load
    :return  the file you loaded
    Notes: the file you load should be a  \'.pr\' file.
    Example:
        >>> x = 1
        >>> y = 'good'
        >>> z = [1, 2, 3]
        >>> save_pr('.\\test', 'ok.pr', [x, y, z], ['name1', 'name2', 'name3'])
        >>> A = load_pr('.\\test\\ok.pr')
          A = {'name1': 1, 'name2': 'good'}
        >>> y, z = load_pr('\\test\\ok.pr', ['y', 'z'])
          y = 'good'
          z = [1, 2, 3]
    """
    if os.path.isfile(path_file):
        s = open(path_file, 'rb')
        if names is None:
            data = pickle.load(s)
            s.close()
            return data
        else:
            tmp = pickle.load(s)
            if type(names) is str:
                data = tmp[names]
                s.close()
                return data
            elif (type(names) is list) or (type(names) is tuple):
                nn = len(names)
                data = list(range(0, nn))
                for i in range(0, nn):
                    data[i] = tmp[names[i]]
                s.close()
                return tuple(data)
    else:
        return False


def mkdir(path):
    """
       Create a folder at your path
       :param path: the path of the folder you wish to create
       :return: the path of folder being created
       Notes: if the folder already exist, it will not create a new one.
    """
    path = path.strip()
    path = path.rstrip("\\")
    path_flag = os.path.exists(path)
    if not path_flag:
        os.makedirs(path)
    return path_flag


