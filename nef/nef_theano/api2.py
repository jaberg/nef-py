
from collections import OrderedDict

import numpy as np
from theano import function, shared, tensor
from theano.printing import debugprint

from neuron.lif import LIFNeuron

class RandomLowRankConnection(object):
    def __init__(self, v1, v2, rank):
        self.v1 = v1 # view 1
        self.v2 = v2 # view 2
        self.rank = rank

        # -- weight matrix is factored dot(u, v)
        self.u = shared(np.random.randn(len(self.v1), rank).astype('float32'))
        self.v = shared(np.random.randn(len(self.v2), rank).astype('float32'))

    def add_to_updates(self, updates):
        v1uv = tensor.dot(
            tensor.dot(self.v1.output, self.u),
            self.v.T)

        return self.v2.add_to_updates(updates, v1uv)

class NEFConnection(RandomLowRankConnection):
    def __init__(self, v1, v2, rank, value):
        pass


class Simulator(object):
    def __init__(self, populations, connections):
        self.populations = populations
        self.connections = connections

        updates = OrderedDict()
        for p in populations:
            J = shared(np.random.randn(len(p)).astype('float32'))
            updates.update(p.update(J))
        for c in connections:
            c.add_to_updates(updates)
        self.f = function([], [], updates=updates)

    def step(self, n):
        for i in xrange(n):
            self.f()



