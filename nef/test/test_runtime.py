"""This test file is for checking the run time of the theano code."""

import math
import time

from .. import nef_theano as nef

# some functions to use in our network
def pow(x):
    return [xval**2 for xval in x]

def times2(x):
    return [xval*2 for xval in x]


if 0:

    net=nef.Network('Runtime Test')
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

    from nef.nef_theano.api2 import (
        LIFNeuron,
        random_low_rank_connection,
        Simulator,
        ConnectionList,
        FuncInput,
        random_connection,
        decoder_encoder_connection,
        )

    dt = 0.001
    Q = queue

    signal = FuncInput(Q, function=math.sin)

    lifs = LIFNeuron(Q, size=4000)
    A = lifs[:1000]
    B = lifs[1000:2000]
    C = lifs[2000:3000]
    D = lifs[3000:]

    connl = ConnectionList('Runtime Test')
    connl.add(random_connection(Q, signal, A))
    connl.add(decoder_encoder_connection(Q, A, B, func=lambda x: x))
    connl.add(decoder_encoder_connection(Q, A, C, func=pow))
    connl.add(decoder_encoder_connection(Q, A, D, func=times2))
    connl.add(decoder_encoder_connection(Q, D, B, func=pow))

    connl.solve_decoder_encoders(Q)

    simulator = connl.simulator()

    start_time = time.time()
    print "starting simulation"
    dt = .0005
    simulator.step(queue, int(1.0 / dt), dt)
    print "runtime: ", time.time() - start_time, "seconds"


