import numpy
import os
import gzip


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(
        path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(
        path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:labels = \
        numpy.frombuffer(lbpath.read(), dtype=numpy.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:images = numpy.frombuffer\
        (imgpath.read(), dtype=numpy.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


def select_images(x, y, labels=numpy.arange(10), n_select="all"):
    if n_select == "all":
        x_select = x
        y_select = y
    else:
        n_sample = numpy.size(x, 0)
        select = numpy.array([])
        select = select.astype(int)
        for label in labels:
            select_tem = numpy.arange(0, n_sample, 1)[y == label]
            total_tem = numpy.size(select_tem, 0)
            if n_select < total_tem:
                total_interval = total_tem // n_select
                select_tem = select_tem[numpy.arange(
                    total_interval - 1, (total_tem - (total_tem % n_select)), total_interval)]
            select = numpy.append(select, select_tem)
        x_select = x[select, :]
        y_select = y[select]
    return x_select, y_select


