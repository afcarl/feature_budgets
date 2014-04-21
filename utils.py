import csv
import re
import numpy as np
import numpy.ma as ma

def weighted_sample(weights):
    '''Randomly sample a value proportional to the given weights.'''
    probs = weights / weights.sum()
    u = np.random.random()
    cur = 0.
    for i,p in enumerate(probs):
        cur += p
        if u <= cur:
            return i
    raise Exception('Weights do not normalize properly! {0}'.format(weights))

def pretty_str(p, decimal_places=2):
    '''Pretty-print a matrix or vector.'''
    if len(p.shape) == 1:
        return vector_str(p, decimal_places)
    if len(p.shape) == 2:
        return matrix_str(p, decimal_places)
    raise Exception('Invalid array with shape {0}'.format(p.shape))

def matrix_str(p, decimal_places=2):
    '''Pretty-print the matrix.'''
    return '[{0}]'.format("\n  ".join([vector_str(a, decimal_places) for a in p]))

def vector_str(p, decimal_places=2):
    '''Pretty-print the vector values.'''
    style = '{0:.' + str(decimal_places) + 'f}'
    return '[{0}]'.format(", ".join([style.format(a) for a in p]))