"""
Simple theano-based simulator
"""
from _collections import OrderedDict
import theano
import numpy as np
from simulator import Simulator
from ensemble import Ensemble
from filter import Filter

def spiking_ensembles_updates(members, dt):
    array_sizes = [m.array_size for m in members]
    dimensions = [m.dimensions for m in members]
    neurons_size = [m.neurons.size for m in members]
    neurons = [m.neurons for m in members]

    # -- reallocate origins
    origins = []
    for m in members:
        for oname, origin in m.origin.items():
            origins.append(origin)
    decoded_output = theano.shared(np.concatenate(
            [o.decoded_output.get_value()[None, ...] for o in origins]))
    decoders = theano.shared(np.concatenate(
            [o.decoders.get_value()[None, ...] for o in origins]))
    print 'decoded_output shape', decoded_output.shape
    print 'decoders shape', decoders.shape

    # -- reallocate filters
    decoded_input_sources = []

    filters = []
    for m in members:
        for name, filt in m.decoded_input.items():
            filters.append(filt)
    Dimshuffle = theano.tensor.elemwise.Dimshuffle
    # XXX make sure that the pstc values match
    filtered_decoded_output = Filter(
            filters[0].pstc,
            source=decoded_output)

    if filt.source.owner and isinstance(filt.source.owner.op, DimShuffle):
        print m.array_size, m.dimensions, m.neurons
        orig = filt.source.owner.inputs[0]
        if orig not in origins:
            raise NotImplementedError()

        #print name, node, node.source
        decoded_input_sources.append(filt.source)
    #print theano.printing.debugprint(decoded_input_sources)

    assert 0


class SimulatorEA(object):
    def __init__(self, network):
        self.network = network
        self.simulation_steps = 0

        if self.network.tick_nodes:
            raise ValueError('Simulator does not support',
                             ' networks with tick_nodes')

        spiking_ensembles = []
        rate_ensembles = []
        other_nodes = []
        for node in self.network.nodes.values():
            if isinstance(node, Ensemble):
                if node.mode == 'spiking':
                    spiking_ensembles.append(node)
                else:
                    rate_ensembles.append(node)
            else:
                other_nodes.append(node)



        # dictionary for all variables
        # and the theano description of how to compute them 
        updates = OrderedDict()

        updates.update(spiking_ensembles_updates(spiking_ensembles,
            network.dt))
        assert len(rate_ensembles) == 0
        # for every node in the network
        for node in other_nodes:
            updates.update(node.update(network.dt))

        # create graph and return optimized update function
        self.step = theano.function([simulation_time], [],
                                    updates=updates.items())

    def run_steps(self, N):
        for i in xrange(N):
            simulation_time = self.simulation_steps * self.network.dt
            self.step(simulation_time)
            self.simulation_steps += 1


    def run(self, approx_sim_time):
        n_steps = int(approx_sim_time / self.network.dt)
        return self.run_steps(n_steps)


