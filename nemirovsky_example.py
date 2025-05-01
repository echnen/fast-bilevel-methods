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
#      Flexible and Fast Diagonal Schemes for Simple Bilevel Optimization.
#      2025. DOI: XX.YYYY/arXiv.XXXX.YYYYY.
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
This file contains useful functions to run the numerical experiment in Section
5.2 of:

R. I. Bot, E. Chenchene, R. Csetnek, D. Hulett.
Flexible and Fast Diagonal Schemes for Simple Bilevel Optimization.
2025. DOI: XX.YYYY/arXiv.XXXX.YYYYY.

For any comment, please contact: enis.chenchene@gmail.com
"""

import numpy as np
from scipy import sparse as sp
import structures as st


class Nemirowki_Example:
    '''
    Inner : np.sum((self.mat @ x - self.off_set) ** 2) / 2
    Outer : ell_1

    '''

    def __init__(self, J, dim):

        self.dim = dim
        self.scale = 50

        off_set = np.zeros(dim)
        off_set[0] = 1
        self.off_set = off_set

        # define function value matrix
        self.eta = lambda it: 1
        diag_lo = np.array([-1 for i in range(dim)])
        diag_ma = np.array([self.eta(i) for i in range(dim)])

        diag_ma[J:] = 0 * diag_ma[J:]
        diag_lo[(J - 1):] = 0 * diag_lo[(J - 1):]

        mat = sp.diags((diag_lo, diag_ma), (-1, 0), shape=(dim, dim),
                       format='csr')

        self.mat = mat
        self.mat_square = mat.T @ mat
        self.L_2 = sp.linalg.norm(self.mat_square, 2)
        self.L_1 = 0

        x_opt = np.ones(dim)
        x_opt[J:] = self.scale * x_opt[J:]
        self.x_opt = x_opt

    def Prox(self, tau, eps_k, in_prox):

        return st.prox_norm_ell_1_tilted(tau * eps_k, in_prox,
                                         self.scale * np.ones(self.dim))

    def Grad(self, eps_k, x):

        return self.mat_square @ x - self.mat.T @ self.off_set

    def res(self, x, x_old):

        return np.sum((x - self.x_opt) ** 2)

    def obj(self, x):

        return np.sum((self.mat @ x - self.off_set) ** 2) / 2

    def obj_outer(self, x):

        return np.abs(np.sum(np.abs(x - self.scale * np.ones(self.dim)))
                      - np.sum(np.abs(self.x_opt - self.scale * np.ones(self.dim))))
