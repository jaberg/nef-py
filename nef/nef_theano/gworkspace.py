import theano
import numpy as np

workspace = None

def set_workspace(ws):
    global workspace
    workspace = ws

def add_ndarray(value, name, fixed_shape=True):
    value = np.asarray(value)
    dtype = value.dtype
    if fixed_shape:
        broadcastable = [si == 1 for si in value.shape]
    else:
        broadcastable = [False] * value.ndim
    vtype = theano.tensor.TensorType(dtype=dtype, broadcastable=broadcastable)
    var = vtype(name=name)
    workspace[var] = value
    return var

