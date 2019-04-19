import numpy


def outer_parallel(a, *matrix):
    # need optimization
    for b in matrix:
        a = (a.repeat(b.shape[1], 1).reshape(a.shape + (-1,))
             * b.repeat(a.shape[1], 0).reshape(a.shape + (-1,))).reshape(a.shape[0], -1)
    return a


def outer(a, *matrix):
    for b in matrix:
        a = numpy.outer(a, b).flatten()
    return a


def tensor_contract(a, b, index):
    ndim_a = numpy.array(a.shape)
    ndim_b = numpy.array(b.shape)
    order_a = numpy.arange(len(ndim_a))
    order_b = numpy.arange(len(ndim_b))
    order_a_contract = numpy.array(order_a[index[0]]).flatten()
    order_b_contract = numpy.array(order_b[index[1]]).flatten()
    order_a_hold = numpy.setdiff1d(order_a, order_a_contract)
    order_b_hold = numpy.setdiff1d(order_b, order_b_contract)
    hold_shape_a = ndim_a[order_a_hold].flatten()
    hold_shape_b = ndim_b[order_b_hold].flatten()
    return numpy.dot(
        a.transpose(numpy.concatenate([order_a_hold, order_a_contract])).reshape(hold_shape_a.prod(), -1),
        b.transpose(numpy.concatenate([order_b_contract, order_b_hold])).reshape(-1, hold_shape_b.prod()))\
        .reshape(numpy.concatenate([hold_shape_a, hold_shape_b]))


def tensor_svd(tmp_tensor, index_left='none', index_right='none'):
    tmp_shape = numpy.array(tmp_tensor.shape)
    tmp_index = numpy.arange(len(tmp_tensor.shape))
    if index_left == 'none':
        index_right = tmp_index[index_right].flatten()
        index_left = numpy.setdiff1d(tmp_index, index_right)
    elif index_right == 'none':
        index_left = tmp_index[index_left].flatten()
        index_right = numpy.setdiff1d(tmp_index, index_left)
    index_right = numpy.array(index_right).flatten()
    index_left = numpy.array(index_left).flatten()
    tmp_tensor = tmp_tensor.transpose(numpy.concatenate([index_left, index_right]))
    tmp_tensor = tmp_tensor.reshape(tmp_shape[index_left].prod(), tmp_shape[index_right].prod())
    u, l, v = numpy.linalg.svd(tmp_tensor, full_matrices=False)
    return u, l, v


