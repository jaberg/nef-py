
import collections

import numpy as np
import theano
import theano.tensor as TT
import gworkspace

class Filter:
    """Filter a theano variable representing some state at time t"""

    def __init__(self, dt, pstc, name=None, source=None, shape=None):
        self.dt = dt
        self.pstc = pstc
        self.source = source

        ### try to find the shape from the given parameters (source and shape)
        if source is not None and hasattr(source, 'get_value'):
            value = source.get_value()
            if shape is not None:
                assert value.shape == shape
            if name is None: 
                name = 'filtered_%s' % source.name
        elif shape is not None:
            value = np.zeros(shape, dtype='float32')
        else:
            raise Exception("Either \"source\" or \"shape\" must define filter shape")
            
        ### create shared variable to store the filtered value
        if source:
            self.value_var = source.type(name=name)
        else:
            self.value_var = TT.TensorType(
                dtype=theano.config.floatX,
                broadcastable=[False] * len(shape))
        gworkspace.workspace[self.value_var] = value

    def set_source(self, source):
        self.source = source

    def update(self):
        if self.pstc > 0:
            decay = TT.cast(np.exp(-self.dt / self.pstc), self.value.dtype)
            value_new = decay*self.value + (1 - decay)*self.source
            return collections.OrderedDict([(self.value, value_new)])
        else:
            ### no filtering (pstc = 0), so just make the value the source
            return collections.OrderedDict([(self.value, self.source)])

