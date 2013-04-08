
from collections import OrderedDict

import numpy as np
from theano import function, shared, tensor
from theano.printing import debugprint

from neuron.lif import LIFNeuron

from batched_gemv import gemm_batched

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

    @property
    def rank(self):
        return self.u.get_value().shape[0]

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
        self.connections = connections

        pop = self.population
        conns = connections
        conns.sort(lambda a, b: cmp(a.v2.start, b.v2.start))

        if not all(c.v1.population is pop for c in conns):
            raise ValueError('not all view1s in same population')
        if not all(c.v2.population is pop for c in conns):
            raise ValueError('not all view2s in same population')
        # -- check all connections have same shape:
        if not 1 == len(set(c.rank for c in conns)):
            raise ValueError('multiple ranks')
        for ci, cj in zip(conns[:], conns[1:]):
            # -- reshaping into matrix requires all to have same len
            if len(ci.v1) != len(cj.v1):
                raise ValueError()
            if len(ci.v2) != len(cj.v2):
                raise ValueError()
            # -- too strict but easy to check. Ideally we just need to
            #    guarantee a regular spacing of the starts
            if cj.v1.start != ci.v1.start + len(ci.v1):
                raise ValueError()
            if cj.v2.start != ci.v2.start + len(ci.v2):
                raise ValueError()
        self.v1s = [c.v1 for c in connections]
        self.v2s = [c.v2 for c in connections]
        self.us = [c.u for c in connections]
        self.vs = [c.v for c in connections]
        Ustack = np.concatenate(
                [u.get_value().T[None, :, :] for u in self.us])
        Vstack = np.concatenate(
                [(v.get_value())[None, :, :] for v in self.vs])

        self.Ustack = shared(Ustack)
        self.Vstack = shared(Vstack)

    def add_to_updates(self, updates):
        if 0:
            for v1, v2, u, v in zip(self.v1s, self.v2s, self.us, self.vs):
                v1uv = tensor.dot( tensor.dot(v1.output, u), v.T)
                updates = v2.add_to_updates(updates, v1uv)
        elif 0:
            output = self.population.voltage
            newout = updates.get(output, tensor.zeros_like(output))
            for v1, v2, u, v in zip(self.v1s, self.v2s, self.us, self.vs):
                v1uv = tensor.dot( tensor.dot(v1.output, u), v.T)
                v2idx = v2.selection
                newout = tensor.inc_subtensor(newout[v2idx], v1uv)
            updates[output] = newout
        else:
            v1_start = self.connections[0].v1.start
            v1_rowlen = len(self.connections[0].v1)

            v2_start = self.connections[0].v2.start
            v2_rowlen = len(self.connections[0].v2)

            v1len = len(self.connections) * v1_rowlen
            v2len = len(self.connections) * v2_rowlen

            voltage = self.population.voltage
            output = self.population.output

            # -- multiply neuron outputs by weights to increment voltage

            output3 = output[v1_start:v1_start + v1len].reshape(
                    (len(self.connections), v1_rowlen, 1))

            decoded = gemm_batched(1.0, self.Ustack, output3)

            newvolt = updates.get(voltage)
            if newvolt is None:
                newvolt3 = None
            else:
                newvolt3 = newvolt[v2_start:v2_start + v2len].reshape(
                        (len(self.connections), v2_rowlen, 1))

            newvolt3 = gemm_batched(1.0, self.Vstack, decoded, 1.0, newvolt3)

            if newvolt is None:
                updates[voltage] = tensor.inc_subtensor(
                    voltage[v2_start:v2_start + v2len],
                    newvolt3.flatten())
            else:
                updates[voltage] = tensor.inc_subtensor(
                    newvolt[v2_start:v2_start + v2len],
                    newvolt3.flatten())
        return updates


def random_low_rank_connection(v1, v2, rank, rng=None, dtype='float32'):
    if rng is None:
        rng = np.random
    u = shared(
        np.asarray(rng.randn(len(v1), rank),
            dtype=dtype,
            order='F'))
    v = shared(
        np.asarray(rng.randn(len(v2), rank),
            dtype=dtype,
            order='F'))
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



