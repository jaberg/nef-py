
from collections import OrderedDict

import numpy as np
from theano import function, shared, tensor
from theano.printing import debugprint

from neuron.lif import LIFNeuron

############################
# Destined for connection.py
############################

class LowRankConnection(object):
    # -- weight matrix is factored dot(u, v)

    def __init__(self, v1, v2, u, v):
        self.v1 = v1 # view 1
        self.v2 = v2 # view 2
        self.u = u
        self.v = v

    def add_to_updates(self, updates):
        # TODO: size checking?
        v1uv = tensor.dot(
            tensor.dot(self.v1.output, self.u),
            self.v.T)

        return self.v2.add_to_updates(updates, v1uv)


class BatchedLowRankConnection(object):
    def __init__(self, connections):
        connections = list(connections)
        self.population = connections[0].v1.population
        if not all(c.v1.population is self.population
                   for c in connections):
            raise ValueError()
        if not all(c.v2.population is self.population
                   for c in connections):
            raise ValueError()
        connections.sort(lambda a, b: cmp(a.v2.start, b.v2.start))
        self.v1s = [c.v1 for c in connections]
        self.v2s = [c.v2 for c in connections]
        self.us = [c.u for c in connections]
        self.vs = [c.v for c in connections]

    def add_to_updates(self, updates):
        if 0:
            for v1, v2, u, v in zip(self.v1s, self.v2s, self.us, self.vs):
                v1uv = tensor.dot( tensor.dot(v1.output, u), v.T)
                updates = v2.add_to_updates(updates, v1uv)
        else:
            output = self.population.voltage
            newout = updates.get(output, tensor.zeros_like(output))
            for v1, v2, u, v in zip(self.v1s, self.v2s, self.us, self.vs):
                v1uv = tensor.dot( tensor.dot(v1.output, u), v.T)
                v2idx = v2.selection
                newout = tensor.inc_subtensor(newout[v2idx], v1uv)
            updates[output] = newout
        return updates

def random_low_rank_connection(v1, v2, rank, rng=None, dtype='float32'):
    if rng is None:
        rng = np.random
    u = shared(rng.randn(len(v1), rank).astype(dtype))
    v = shared(rng.randn(len(v2), rank).astype(dtype))
    return LowRankConnection(v1, v2, u, v)


def decoder_encoder(v1, v2, latent_v1, latent_v2, samples_v1, samples_v2, rng=None):
    # Given that neuron view v1 represents latent signal `latent_v1`
    # as
    pass


############################
# Destined for simulator.py
############################

class Simulator(object):
    def __init__(self, populations, connections):
        self.populations = populations
        self.connections = connections

        conns = OrderedDict()
        for c in connections:
            conns.setdefault(type(c), []).append(c)

        # compress the set of connections as much as possible
        # TODO: make this a registry or smth
        conns[LowRankConnection] = [BatchedLowRankConnection(
            conns[LowRankConnection])]

        updates = OrderedDict()
        for p in populations:
            J = shared(np.random.randn(len(p)).astype('float32'))
            updates.update(p.update(J))
        for ctype in conns:
            for c in conns[ctype]:
                c.add_to_updates(updates)
        self.f = function([], [], updates=updates)

    def step(self, n):
        for i in xrange(n):
            self.f()



