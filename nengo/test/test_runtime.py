"""This test file is for checking the run time of the theano code."""

import math
import time
import unittest

import nose
from nengo import nef_theano as nef
from nengo.nef_theano.simulator import Simulator

# -- XXX rename to an optimization file
from nengo.nef_theano.simulator_concat import SimulatorConcat

from nengo.nef_theano.simulator_ocl import SimulatorOCL
from nengo.nef_theano.simulator_ea import SimulatorEA
#try:
#except ImportError:
#    pass

approx_time = 1.0 # second

class Runtime(unittest.TestCase):
    def setUp(self):

        net=nef.Network('Runtime Test', seed=123)
        net.make_input('in', value=math.sin)
        net.make('A', 1000, 1)
        net.make('B', 1000, 1)
        net.make('C', 1000, 1)
        net.make('D', 1000, 1)

        # some functions to use in our network
        def pow(x):
            return [xval**2 for xval in x]

        def mult(x):
            return [xval*2 for xval in x]

        net.connect('in', 'A')
        net.connect('A', 'B')
        net.connect('A', 'C', func=pow)
        net.connect('A', 'D', func=mult)
        net.connect('D', 'B', func=pow) # throw in some recurrency whynot

        self.net = net

    def test_run(self):
        self.net.run(0.001) # build the thing
        print "starting simulation (net.run)"
        start_time = time.time()
        self.net.run(approx_time)
        print "runtime: ", time.time() - start_time, "seconds"

    def test_simulator(self):
        sim = Simulator(self.net)
        start_time = time.time()
        print "starting simulation (Simulator)"
        sim.run(approx_time)
        print "runtime: ", time.time() - start_time, "seconds"

    def test_simulator_ea(self):
        sim = SimulatorEA(self.net)
        start_time = time.time()
        print "starting simulation (SimulatorConcat)"
        sim.run(approx_time)
        print "runtime: ", time.time() - start_time, "seconds"

    def test_simulator_concat(self):
        sim = SimulatorConcat(self.net)
        start_time = time.time()
        print "starting simulation (SimulatorConcat)"
        sim.run(approx_time)
        print "runtime: ", time.time() - start_time, "seconds"

    def test_simulator_ocl(self):
        if 'SimulatorOCL' in globals():
            sim3 = SimulatorOCL(self.net, profiling=False)
            start_time = time.time()
            print "starting simulation (OCL)"
            sim3.run(approx_time)
            print "runtime: ", time.time() - start_time, "seconds"
        else:
            raise nose.SkipTest()

    def test_simulator_ocl_profile(self):
        if 'SimulatorOCL' in globals():
            sim2 = SimulatorOCL(self.net, profiling=True)
            start_time = time.time()
            print "starting simulation (OCL with profiling)"
            sim2.run(approx_time)
            print "runtime: ", time.time() - start_time, "seconds"
            foo = [(t, n) for (n, t) in sim2.t_used.items()]
            foo.sort()
            foo.reverse()
            t_total = 0
            for t, n in foo:
                print t * 1e-9, n
                t_total += t * 1e-9
            print 'total time in OCL:', t_total
        else:
            raise nose.SkipTest()

    def test_simulator_ocl_debug_mode(self):
        if 'SimulatorOCL' in globals():
            sim4 = SimulatorOCL(self.net, profiling=False)
            start_time = time.time()
            print "starting simulation with error detection (OCL)"
            sim4.run(approx_time, run_theano_too=True)
        else:
            raise nose.SkipTest()
