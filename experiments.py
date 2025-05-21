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
This file contains the numerical experiments in:

R. I. Bot, E. Chenchene, R. Csetnek, D. Hulett.
Accelerating Diagonal Methods for Bilevel Optimization:
Unified Convergence via Continuous-Time Dynamics.
2025. DOI: 10.48550/arXiv.2505.14389.

For any comment, please contact: enis.chenchene@gmail.com
"""

import numpy as np
import logistic_regression as lr
import nemirovsky_example as nem
import optimization as opt
import plots as show
from tqdm import tqdm


def experiment_nemirovsky():

    dim = 7
    J = 4
    maxit = 2000
    cases = 20
    np.random.seed(0)

    # storage
    Res_Bi_PG = np.zeros((maxit, cases))
    Res_biFI = np.zeros((maxit, cases))
    Res_FBi_PG = np.zeros((maxit, cases))
    Res_staBiM = np.zeros((maxit, cases))
    Res_Bi_SG_II = np.zeros((maxit, cases))

    Obj_Bi_PG = np.zeros((maxit, cases))
    Obj_biFI = np.zeros((maxit, cases))
    Obj_FBi_PG = np.zeros((maxit, cases))
    Obj_staBiM = np.zeros((maxit, cases))
    Obj_Bi_SG_II = np.zeros((maxit, cases))

    Obj_H_Bi_PG = np.zeros((maxit, cases))
    Obj_H_biFI = np.zeros((maxit, cases))
    Obj_H_FBi_PG = np.zeros((maxit, cases))
    Obj_H_staBiM = np.zeros((maxit, cases))
    Obj_H_Bi_SG_II = np.zeros((maxit, cases))

    # contains delta
    Spects = np.linspace(1 + 1e-1, 2 - 1e-1, cases)

    # initializing model
    Model = nem.Nemirowki_Example(J, dim)

    # parameters
    alpha = 4
    sigma_e = 1e1
    sigma_t = 20
    c = 1e1
    s = 0.95 / Model.L_2

    # initializing
    x_init = np.zeros(Model.dim)

    for cs in tqdm(range(cases)):

        # step-size
        delta = Spects[cs]
        # Spects.append(delta)

        # Algorithm 1 (our paper)
        Res_Bi_PG[:, cs], Obj_Bi_PG[:, cs], Obj_H_Bi_PG[:, cs] = \
            opt.Bi_PG(x_init, sigma_e, s, c, delta, Model, maxit)

        # Algorithm 2 (our paper)
        Res_biFI[:, cs], Obj_biFI[:, cs], Obj_H_biFI[:, cs] = \
            opt.bi_FISTA(x_init, alpha, sigma_e, sigma_t, s, c, delta, Model,
                         maxit)

        # Fast Bi-level Proximal Gradient (Merchav, Sabach, Teboulle, '24)
        Res_FBi_PG[:, cs], Obj_FBi_PG[:, cs], Obj_H_FBi_PG[:, cs] =\
            opt.FBi_PG(x_init, alpha, s, c, delta, Model, maxit)

        # Static Bilevel Method (Latafat, Themelis, Villa, Patrinos, '24)
        Res_staBiM[:, cs], Obj_staBiM[:, cs], Obj_H_staBiM[:, cs] = \
            opt.staBiM(x_init, sigma_e, c, delta, Model, maxit)

        # Bi-Sub-Gradient - Version II (Merchav, Sabach, '23)
        Res_Bi_SG_II[:, cs], Obj_Bi_SG_II[:, cs], Obj_H_Bi_SG_II[:, cs] = \
            opt.Bi_SG_II(x_init, c, delta, Model, maxit)

    show.plot_nemirovsky(Res_Bi_PG, Res_biFI, Res_FBi_PG, Res_staBiM,
                         Res_Bi_SG_II, Obj_Bi_PG, Obj_biFI, Obj_FBi_PG,
                         Obj_staBiM, Obj_Bi_SG_II, Obj_H_Bi_PG, Obj_H_biFI,
                         Obj_H_FBi_PG, Obj_H_staBiM, Obj_H_Bi_SG_II,
                         maxit, Spects, cases)


def experiment_logistic():

    # initializing model
    X_train, y_train = lr.load_dataset(1)
    Model = lr.Logistic_Regression(X_train, y_train)
    print('Dataset downloaded')

    # parameter
    alpha = 4
    sigma_e = 1
    sigma_t = 1
    delta = 1.9
    c = 1e2
    s = 0.95 / Model.L_2
    maxit = 50000

    # initializing
    x_init = np.random.rand(Model.dim)

    # Algorithm 1 (our paper)
    print('Starting Algorithm 1 ...')
    Res_Bi_PG, Obj_Bi_PG, Obj_H_Bi_PG = \
        opt.Bi_PG(x_init, sigma_e, s, c, delta, Model, maxit)

    # Algorithm 2 (our paper)
    print('Starting Algorithm 2 ...')
    Res_biFI, Obj_biFI, Obj_H_biFI = \
        opt.bi_FISTA(x_init, alpha, sigma_e, sigma_t, s, c, delta, Model,
                     maxit)

    # Fast Bi-level Proximal Gradient (Merchav, Sabach, Teboulle, '24)
    print('Starting Fast Bi-Level Proximal Gradient ...')
    Res_FBi_PG, Obj_FBi_PG, Obj_H_FBi_PG = \
        opt.FBi_PG(x_init, alpha, s, c, delta, Model, maxit)

    # Static Bilevel Method (Latafat, Themelis, Villa, Patrinos, '24)
    print('Starting Static Bilevel Method ...')
    Res_staBiM, Obj_staBiM, Obj_H_staBiM = \
        opt.staBiM(x_init, sigma_e, c, delta, Model, maxit)

    # Bi-Sub-Gradient - Version II (Merchav, Sabach, '23)
    print('Starting Bi-Sub-Gradient Version II ...')
    Res_Bi_SG_II, Obj_Bi_SG_II, Obj_H_Bi_SG_II = \
        opt.Bi_SG_II(x_init, c, delta, Model, maxit)

    show.plot_logistic(Res_Bi_PG, Res_biFI, Res_FBi_PG, Res_staBiM,
                       Res_Bi_SG_II, Obj_Bi_PG, Obj_biFI, Obj_FBi_PG,
                       Obj_staBiM, Obj_Bi_SG_II, Obj_H_Bi_PG, Obj_H_biFI,
                       Obj_H_FBi_PG, Obj_H_staBiM, Obj_H_Bi_SG_II, maxit)
