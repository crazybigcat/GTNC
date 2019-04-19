import os
import pickle
import hashlib
import time


def info_contact():
    """Return the information of the contact"""
    info = dict()
    info['name'] = 'Sun Zhengzhi'
    info['email'] = 'sunzhengzhi16@mails.ucas.ac.cn'
    info['affiliation'] = 'University of Chinese Academy of Sciences'
    return info
# These are from the original functions of Sun Zhengzhi


def sort_dict(a):
    b = dict()
    dict_index = sorted(a.keys())
    for index in dict_index:
        b[index] = a[index]
    return b


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


def name_generator_md5(path, file, input_parameter):
    input_parameter = sort_dict(input_parameter)
    number_md5 = hashlib.md5(str(input_parameter).encode(encoding='utf-8')).hexdigest()
    tmp_save = {number_md5 : input_parameter}
    file_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    file_save = file + '_' + file_time
    if not os.path.exists(path):
        os.makedirs(path)
    all_filename = os.listdir(path)
    for filename in all_filename:
        if file in filename:
            tmp_load = load_pr(path + filename)
            tmp_save.update(tmp_load)
            os.remove(path + filename)
    save_pr(path, file_save, list(tmp_save.values()), list(tmp_save.keys()))
    return number_md5

# These are from BasicFunctions of Ran Shi_ju with some changes


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
            elif type(names) is list or type(names) is tuple:
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


def print_program_info(para, mode='start'):
    if mode == 'start':
        print('This program starts at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(sort_dict(para))
    elif mode == 'end':
        print('This program ends at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

