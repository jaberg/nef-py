Using the Neural Engineering Framework with Python
==================================================

This package contains an API for creating neural simulations
using the methods of the Neural Engineering Framework
(NEF_).

It also contains, at the moment, the beginnings of
a Theano_ implementation of the NEF API.

Documentation can be found at https://nef-py.readthedocs.org/en/latest/

.. _NEF: http://ctnsrv.uwaterloo.ca/cnrglab/node/215

.. _Theano: http://deeplearning.net/software/theano/

Temporary Theano Requirements
-----------------------------

The theano branch currently uses the "Workspace" code from

https://github.com/jaberg/theano_workspace
https://github.com/logpy/logpy

to set these up, type:

    git clone https://github.com/jaberg/theano_workspace \
    && (cd theano_workspace && python setup.py )

    git clone https://github.com/logpy/logpy \
    && (cd logpy && python setup.py )

