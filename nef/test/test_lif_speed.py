import time
import theano

import pyopencl as cl
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
print cl.get_platforms()
print ctx.devices

from nef.nef_theano.api2 import (
    OCL_LIFNeuron,
    random_low_rank_connection,
    Simulator,
    )

p = OCL_LIFNeuron(queue, size=2048 * 1000)
sim = Simulator([p], [])
queue.flush()
t0 = time.time()
sim.step(queue, 1000, dt=0.0005)
queue.flush()
t1 = time.time()
elapsed = t1 - t0
print elapsed

