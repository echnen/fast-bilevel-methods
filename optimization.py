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
This file contains all implementations of the numerical methods used for
comparison in Section 5 of:

R. I. Bot, E. Chenchene, R. Csetnek, D. Hulett.
Accelerating Diagonal Methods for Bilevel Optimization:
Unified Convergence via Continuous-Time Dynamics.
2025. DOI: 10.48550/arXiv.2505.14389.

For any comment, please contact: enis.chenchene@gmail.com
"""

import numpy as np


def bi_FISTA(x_init, alpha, sigma_e, sigma_t, s, c, delta, Model, maxit):
    '''
    Algorithm 2 in Section 3 of our paper.
    '''

    # storage
    Res = []
    Fs = []
    Hs = []

    # initialize
    x_old = np.copy(x_init)
    x = np.copy(x_init)

    for k in range(maxit):

        alp_k = 1 - alpha / (k + sigma_t + 1)
        eps_k = c / (k + sigma_e + 1) ** delta

        y = x + alp_k * (x - x_old)
        x_old = np.copy(x)
        x = Model.Prox(s, eps_k, y - s * Model.Grad(eps_k, y))

        Res.append(Model.res(x, x_old))
        Fs.append(Model.obj(x))
        Hs.append(Model.obj_outer(x))

    return Res, Fs, Hs


def Bi_PG(x_init, sigma_e, s, c, delta, Model, maxit):
    '''
    Algorithm 1 in Section 2 of our paper.
    '''

    # storage
    Res = []
    Fs = []
    Hs = []

    # initialize
    x = np.copy(x_init)

    for k in range(maxit):

        eps_k = c / (k + sigma_e + 1) ** (delta / 2)

        x_old = np.copy(x)
        x = Model.Prox(2 * s, eps_k, x - 2 * s * Model.Grad(eps_k, x))

        Res.append(Model.res(x, x_old))
        Fs.append(Model.obj(x))
        Hs.append(Model.obj_outer(x))

    return Res, Fs, Hs


def FBi_PG(x_init, alpha, s, c, delta, Model, maxit):
    '''
    Fast Bi-level Proximal Gradient

    Merchav, Sabach, Teboulle,
    A Fast Algorithm for Convex Composite Bi-Level Optimization, 2024.

    Note : t_k = (k + a) / a, a >= 2, c = 1
    To standardize, we use: gamma = delta, a = alpha - 1

    '''
    # storage
    Res = []
    Fs = []
    Hs = []

    # initialize
    x_old = np.copy(x_init)
    x = np.copy(x_init)

    for k in range(maxit):

        alp_k = 1 - alpha / (k + alpha)
        eps_k = 1 / (k + alpha - 1) ** delta

        y = x + alp_k * (x - x_old)
        x_old = np.copy(x)
        x = Model.Prox(s, eps_k, y - s * Model.Grad(eps_k, y))

        Res.append(Model.res(x, x_old))
        Fs.append(Model.obj(x))
        Hs.append(Model.obj_outer(x))

    return Res, Fs, Hs


def staBiM(x_init, sigma, c, delta, Model, maxit):
    '''
    Static Bilevel Method

    Latafat, Themelis, Villa, Patrinos
    On the convergence of proximal gradient methods for convex simple
    bilevel optimization, '24
    '''

    # storage
    Res = []
    Fs = []
    Hs = []

    # initialize
    x = np.copy(x_init)

    for k in range(maxit):

        s = 0.99 / ((3 / 4) ** k * Model.L_1 + Model.L_2)
        eps_k = c / (k + sigma + 1) ** (delta / 2)

        x_old = np.copy(x)
        x = Model.Prox(s, eps_k, x - s * Model.Grad(eps_k, x))

        Res.append(Model.res(x, x_old))
        Fs.append(Model.obj(x))
        Hs.append(Model.obj_outer(x))

    return Res, Fs, Hs


def Bi_SG_II(x_init, c, delta, Model, maxit):
    '''
    Bi-Sub-Gradient - Version II

    Merchav, Sabach
    Convex Bi-level Optimization Problems with Non-smooth Outer Objective
    Function, '23
    '''

    # storage
    Res = []
    Fs = []
    Hs = []

    # initialize
    x = np.copy(x_init)

    s = 1 / Model.L_2

    for k in range(maxit):

        eps_k = c / (k + 1) ** (delta / 2)

        x_old = np.copy(x)
        y = x - s * Model.Grad(eps_k, x)
        x = Model.Prox(s, eps_k, y)

        Res.append(Model.res(x, x_old))
        Fs.append(Model.obj(x))
        Hs.append(Model.obj_outer(x))

    return Res, Fs, Hs
