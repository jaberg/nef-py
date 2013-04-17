import pyopencl as cl
import pyopencl.array as cl_array
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import numpy as np

from .neuron import Neuron

class LIFNeuronView(object):
    def __init__(self, population, selection):
        self.population = population
        self.selection = selection

    def __len__(self):
        if isinstance(self.selection, slice):
            start, stop = self.selection.start, self.selection.stop 
            if start is None:
                start = 0
            if stop is None:
                stop = self.population.size
            return stop - start
        assert 0

    @property
    def start(self):
        return 0 if self.selection.start is None else self.selection.start

    @property
    def stop(self):
        if self.selection.stop is None:
            return self.population.size
        else:
            return self.selection.stop

    @property
    def step(self):
        return 1 if self.selection.step is None else self.selection.step

    @property
    def voltage(self):
        return self.population.voltage[self.selection]

    @property
    def refractory_time(self):
        return self.population.refractory_time[self.selection]

    @property
    def output(self):
        return self.population.output[self.selection]

    def add_to_updates(self, updates, v):
        var = self.population.voltage
        idx = self.selection
        newvar = updates.get(var, var)
        updates[var] = TT.inc_subtensor(newvar[idx], v)
        return updates



class LIFNeuron(Neuron):
    def __init__(self, queue, size, dt=0.001, tau_rc=0.02, tau_ref=0.002):
        """Constructor for a set of LIF rate neuron.

        :param int size: number of neurons in set
        :param float dt: timestep for neuron update function
        :param float tau_rc: the RC time constant
        :param float tau_ref: refractory period length (s)

        """
        Neuron.__init__(self, queue, size, dt)
        self.tau_rc = tau_rc
        self.tau_ref  = tau_ref
        self.V_threshold = 1.0
        self.voltage = cl_array.zeros(queue, (size,), 'float32')
        self.refractory_time = cl_array.zeros(queue, (size,), 'float32')
        self.input_current = cl_array.zeros(queue, (size,), 'float32')

        self._cl_fn = cl.Program(queue.context, """
            __kernel void foo(
                __global const float *J,
                __global float *voltage,
                __global float *refractory_time,
                __global float *output
                         )
            {
                const float dt = %(dt)s;
                const float tau_ref = %(tau_ref)s;
                const float tau_rc = %(tau_rc)s;
                const float V_threshold = %(V_threshold)s;

                int gid = get_global_id(0);
                float v = voltage[gid];
                float rt = refractory_time[gid];

                  float dV = dt / tau_rc * (J[gid] - v);
                  v += dV;
                  float post_ref = - rt / dt;
                  v = v > 0 ?
                      v * (post_ref < 0 ? 0 : post_ref < 1 ? post_ref : 1)
                      : 0;
                  int spiked = v > V_threshold;
                  float overshoot = (v - V_threshold) / dV;
                  float spiketime = dt * (1.0 - overshoot);

                  float new_voltage = v * (1.0 - spiked);
                  float new_rt = spiked ? spiketime + tau_ref : rt - dt;

                  output[gid] = spiked ? 1.0 : 0.0;
                  refractory_time[gid] = new_rt;
                  voltage[gid] = new_voltage;
            }
            """ % self.__dict__).build().foo
        
    #TODO: make this generic so it can be applied to any neuron model
    # (by running the neurons and finding their response function),
    # rather than this special-case implementation for LIF        

    def make_alpha_bias(self, max_rates, intercepts):
        """Compute the alpha and bias needed to get the given max_rate
        and intercept values.
        
        Returns gain (alpha) and offset (j_bias) values of neurons.

        :param float array max_rates: maximum firing rates of neurons
        :param float array intercepts: x-intercepts of neurons
        
        """
        x1 = intercepts
        x2 = 1.0
        z1 = 1
        z2 = 1.0 / (1 - TT.exp(
                (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        alpha = (z1 - z2) / (x1 - x2)
        j_bias = z1 - alpha * x1
        return alpha, j_bias

    # TODO: have a reset() function at the ensemble and network level
    #that would actually call this
    def reset(self):
        """Resets the state of the neuron."""
        Neuron.reset(self)

        self.voltage.set_value(np.zeros(self.size).astype('float32'))
        self.refractory_time.set_value(np.zeros(self.size).astype('float32'))

    def cl_update(self, queue):
        self._cl_fn(queue, (self.size,), None,
            self.input_current.data,
            self.voltage.data,
            self.refractory_time.data,
            self.output.data)

    def __len__(self):
        return self.size

    def __getitem__(self, foo):
        return LIFNeuronView(self, foo)
