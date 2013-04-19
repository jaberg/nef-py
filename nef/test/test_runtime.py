"""This test file is for checking the run time of the theano code."""

import math
import time

from .. import nef_theano as nef

# some functions to use in our network
def pow(x):
    return [xval**2 for xval in x]

def times2(x):
    return [xval*2 for xval in x]

def time_network(net):

    net.make_input('in', value=math.sin)
    net.make('A', 1000, 1)
    net.make('B', 1000, 1)
    net.make('C', 1000, 1)
    net.make('D', 1000, 1)

    net.connect('in', 'A')
    net.connect('A', 'B')
    net.connect('A', 'C', func=pow)
    net.connect('A', 'D', func=times2)
    net.connect('D', 'B', func=pow) # throw in some recurrency whynot

    start_time = time.time()
    print "starting simulation"
    net.run(0.1)
    print "runtime: ", time.time() - start_time, "seconds"





if 1:
    import pyopencl as cl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    from nef.nef_theano import api2

    time_network(api2.Network('Runtime Test (OCL)', queue))

if 1:
    time_network(nef.Network('Runtime Test'))

