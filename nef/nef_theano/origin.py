from numbers import Number

import numpy as np
import theano
import gworkspace

class Origin(object):
    """An origin is an object that provides a signal. Origins project
    to terminations.

    This is a basic Origin, promising a set of instance variables
    to any accessing objects.
    """
    
    def __init__(self, func, initial_value=None, name='origin'):
        """
        :param function func: the function carried out by this origin
        :param array initial_value: the initial_value of the decoded_output
        
        """
        self.func = func

        if initial_value is None:
            # initial output value = function value with input 0.0
            initial_value = self.func(0.0)

        # if scalar, make it a list
        if isinstance(initial_value, Number):
            initial_value = [initial_value]
        initial_value = np.float32(initial_value)

        # theano internal state defining output value
        self.decoded_output_var = gworkspace.add_ndarray(
            initial_value, name='%s.decoded_output' % name) 

        # find number of parameters of the projected value
        self.dimensions = len(initial_value)
