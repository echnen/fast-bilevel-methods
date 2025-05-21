# -*- coding: utf-8 -*-
#
#    Copyright (C) 2025 Radu Ioan Bot (radu.bot@univie.ac.at)
#                       Enis Chenchene (enis.chenchene@univie.ac.at)
#                       Robert Csetnek (robert.csetnek@univie.ac.at)
#                       David Hulett (david.hulett@univie.ac.at)
#
#    This file is part of the example code repository for the paper:
#
#      R. I. Bot, E. Chenchene, R. Csetnek, D. Hulett.
#      Accelerating Diagonal Methods for Bilevel Optimization:
#      Unified Convergence via Continuous-Time Dynamics
#      2025. DOI: 10.48550/arXiv.2505.14389.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
This file contains useful functions to run all numerical experiments in Section
5 of:

R. I. Bot, E. Chenchene, R. Csetnek, D. Hulett.
Accelerating Diagonal Methods for Bilevel Optimization:
Unified Convergence via Continuous-Time Dynamics.
2025. DOI: 10.48550/arXiv.2505.14389.

For any comment, please contact: enis.chenchene@gmail.com
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

    exp = np.exp(x)

    return exp / (exp + 1)


def sigmoid(x):

    positive = x >= 0
    # boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains juke hence will be faster to allocate than zeros
    result = np.empty_like(x)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])

    return result
