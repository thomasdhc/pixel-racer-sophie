import os, argparse

import numpy as np

import tensorflow as tf

def reshape_state(state, dim, type='val'):
    if type == 'val':
        rows, cols = np.where(state == 0.5)
        row, col = rows[0], cols[0]
        return row * dim + col
    elif type == 'identity':
        rows, cols = np.where(state == 0.5)
        row, col = rows[0], cols[0]
        index = row * dim + col
        return np.identity(121)[index:index + 1]
    elif type == 'array':
        return np.reshape(state, [1, dim * dim])