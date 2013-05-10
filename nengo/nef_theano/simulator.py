import copy
from _collections import OrderedDict
import theano
import numpy as np

class MapGemv(theano.Op):
    def __init__(self):
        pass

    def make_node(self, alpha, A, X, beta, J):
        inputs = map(theano.tensor.as_tensor_variable,
            [alpha, A, X, beta, J])
        return theano.Apply(self, inputs, [inputs[-1].type()])

    def perform(self, node, inputs, outstor):
        alpha, A, X, beta, J = inputs

        J = J.copy() # TODO: inplace

        J *= beta
        for i in range(len(J)):
            J[i] += alpha * np.dot(X[i], A[i].T)
        outstor[0][0] = J

map_gemv = MapGemv()


class IFS(object):
    """
    Iteratable Function System

    Like "FunctionGraph" in Theano, except that it is is designed around
    updates, so the number of inputs and outputs can actually change.
    """
    def __init__(self, fn):
        self.updates = fn.fn.updated_vars.items()
        self._nodes = fn.fn.nodes
        self.dag = None # networkx ?
        self.meta = {}
        for node in self._nodes:
            for vv in node.inputs + node.outputs:
                self.meta.setdefault(vv, copy.copy(vv.tag))

    def add_update(self, in_expr, out_expr):
        if in_expr.owner:
            raise ValueError(in_expr)
        self.updates.append((in_expr, out_expr))

    def replace(self, old_v, new_v):
        raise NotImplementedError()

    @property
    def nodes(self):
        return self._nodes

    @property
    def variables(self):
        tmp = []
        for node in self._nodes:
            tmp.extend(node.inputs)
            tmp.extend(node.outputs)
        seen = set()
        rval = []
        for vv in tmp:
            if vv not in seen:
                rval.append(vv)
            seen.add(vv)
        return rval


simulation_time = theano.tensor.fscalar(name='simulation_time')

scribe_buf_last_filled = theano.tensor.iscalar(name='scribe_buf_last_filled')


class Simulator(object):
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


