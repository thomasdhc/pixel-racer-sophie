import numpy as np

def reshape_state(state, dim):
    rows, cols = np.where(state == 0.5)
    row, col = rows[0], cols[0]
    return row * dim + col