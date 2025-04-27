# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 13:41:54 2025

@author: enisc
"""

import numpy as np


def prox_norm_ell_2(tau, w):

    norm = np.linalg.norm(w)

    if norm <= 1e-9:
        return w
    else:
        return max(0, 1 - tau / norm) * w


def prox_norm_ell_1(tau, w):

    return np.sign(w) * np.maximum(np.abs(w) - tau, 0)


def prox_norm_ell_2_tilted(tau, w, tilt):
    '''
    computes the proximity operator of |w - tilt|_2
    '''

    return tilt + prox_norm_ell_2(tau, w - tilt)


def prox_norm_ell_1_tilted(tau, w, tilt):
    '''
    computes the proximity operator of |w - tilt|_1
    '''

    return tilt + prox_norm_ell_1(tau, w - tilt)


def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _negative_sigmoid(x):
    # Cache exp so you won't have to calculate it twice
    exp = np.exp(x)
    return exp / (exp + 1)


def sigmoid(x):
    positive = x >= 0
    # Boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains juke hence will be faster to allocate than zeros
    result = np.empty_like(x)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])

    return result


# def sigmoid(z):

#     return 1 / (1 + np.exp(-z))
