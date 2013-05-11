"""
simulator_concat: a theano simulator that merges together
populations and certain other ops for faster evaluation.

"""
import copy
from _collections import OrderedDict
import theano
import numpy as np

import simulator
import lif

from theano.compile import optdb
from theano.gof.opt import Optimizer

# -- support for the "const_shape" tag I made up
def shape_ir(self, i, r):
    """Return symbolic r.shape[i] for tensor variable r, int i"""
    if hasattr(r.type, "broadcastable") and r.type.broadcastable[i]:
        return self.lscalar_one
    elif hasattr(r, 'data'):
        return theano.tensor.as_tensor_variable(r.data.shape[i])
    elif hasattr(r.tag, 'const_shape'):
        return theano.tensor.as_tensor_variable(r.tag.const_shape[i])
    else:
        return theano.tensor.opt.Shape_i(i).make_node(r).outputs[0]

theano.tensor.opt.ShapeFeature.shape_ir = shape_ir


class LIF_Concat(Optimizer):
    """
    Merge multiple populations of LIF neurons
    """
    def apply(self, fgraph):
        lifs = [node for node in fgraph.toposort()
                if isinstance(node.op, lif.LIF_Op)]
        if len(lifs) < 2:
            return

        tau_rcs = [node.op.tau_rc for node in lifs]
        tau_refs = [node.op.tau_ref for node in lifs]
        upsamples = [node.op.upsample for node in lifs]
        dts = [node.inputs[-1] for node in lifs]

        if len(set(tau_rcs)) > 1:
            raise NotImplementedError('multiple tau_rcs')
        if len(set(tau_refs)) > 1:
            raise NotImplementedError('multiple tau_refs')
        if len(set(upsamples)) > 1:
            raise NotImplementedError('multiple upsamples')
        if len(set(dts)) > 1:
            raise NotImplementedError('multiple dts')

        getconst = theano.tensor.get_scalar_constant_value
        shape_of = fgraph.shape_feature.shape_of

        shapes = [[map(int, map(getconst, shape_of[vv]))
                   for vv in node.outputs]
            for node in lifs]

        sizes = [map(np.prod, shape) for shape in shapes]

        print shapes
        print sizes

        if all(shape[0][0] == 1 for shape in shapes):
            v = theano.tensor.concatenate(
                [node.inputs[0] for node in lifs])

            rt = theano.tensor.concatenate(
                [node.inputs[1] for node in lifs])

            ic = theano.tensor.concatenate(
                [node.inputs[1] for node in lifs])

            dt = lifs[0].inputs[-1]

            new_v, new_rt, new_spiked = lifs[0].op(v, rt, ic, dt)

            repls = []

            for ii, node in enumerate(lifs):
                repls.append((node.outputs[0], new_v[ii:ii + 1]))
                repls.append((node.outputs[1], new_rt[ii:ii + 1]))
                repls.append((node.outputs[2], new_spiked[ii:ii + 1]))
            fgraph.replace_all_validate(repls, reason='LIF_Concat')
        else:
            raise NotImplementedError()


optdb['canonicalize'].register('LIF_Concat', LIF_Concat(),
                               'fast_run', 'nengo')



class SimulatorConcat(simulator.Simulator):
    def __init__(self, network):
        self.network = network
        self.simulation_steps = 0

        if self.network.tick_nodes:
            raise ValueError('Simulator does not support',
                             ' networks with tick_nodes')

        # dictionary for all variables
        # and the theano description of how to compute them 
        updates = OrderedDict()

        # for every node in the network
        for node in self.network.nodes.values():
            # if there is some variable to update
            if hasattr(node, 'update'):
                # add it to the list of variables to update every time step
                updates.update(node.update(self.network.dt))

        # create graph and return optimized update function
        self.step = theano.function([simulator.simulation_time], [],
                                    updates=updates.items())


    def run_steps(self, N):
        fn = self.ifs.fn
        for i in xrange(N):
            simulation_time = self.simulation_steps * self.network.dt
            fn(simulation_time)
            self.simulation_steps += 1


    def run(self, approx_sim_time):
        n_steps = int(approx_sim_time / self.network.dt)
        return self.run_steps(n_steps)


