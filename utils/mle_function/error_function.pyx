"""
erro_function.py: Main Cython code for the MLE error. 
Authors       : mns
"""
import numpy as np
cimport cython

DTYPE = np.int64

@cython.boundscheck(False)
@cython.wraparound(False)
# n - len(x_true)
# pred_count - len(x_pred)
# l, m - d.shape
def offset_sum(long long [:, :] sorted_idx, double [:, :] d, long long n, long long m, long long max_dist):
    cdef Py_ssize_t i
    cdef long long p, q

    ## OPT: Just use a boolean array
    _closest_pred = -1 * np.ones(n, dtype=DTYPE)
    cdef long long [:] closest_pred = _closest_pred

    _closest_true = -1 * np.ones(m, dtype=DTYPE)
    cdef long long [:] closest_true = _closest_true

    cdef long long trues_left = n
    cdef long long preds_left = m

    cdef double offset_sum = 0

    cdef long long temp = n * m
    for i in range(temp):
        if preds_left <=0 or trues_left <= 0:
            break
        p = sorted_idx[0, i]
        q = sorted_idx[1, i]
        if d[p, q] >= max_dist:
            break
        if closest_pred[p] != -1 or closest_true[q] != -1:
            continue
        else:
            closest_pred[p] = q
            closest_true[q] = p
            offset_sum += d[p, q]
            preds_left -= 1
            trues_left -= 1

    offset_sum += max(preds_left, trues_left) * max_dist

    return offset_sum
