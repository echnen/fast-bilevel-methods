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
5.3 of:

R. I. Bot, E. Chenchene, R. Csetnek, D. Hulett.
Flexible and Fast Diagonal Schemes for Simple Bilevel Optimization.
2025. DOI: XX.YYYY/arXiv.XXXX.YYYYY.

For any comment, please contact: enis.chenchene@gmail.com
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import structures as st


def load_dataset(dataset=1):
    '''
    Loading the dataset
    '''

    if dataset == 1:

        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        poly = PolynomialFeatures(degree=3, include_bias=False)
        X = df.drop(columns=['target'])
        X = poly.fit_transform(X)

        # split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, df['target'],
            test_size=0.2, random_state=42,
            stratify=df['target']
        )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # adding bias
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))

    return X_train, y_train


class Logistic_Regression:

    def __init__(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train
        self.m = X_train.shape[0]
        self.dim = X_train.shape[1]

        self.L_2 = np.linalg.norm(X_train.T @ X_train, 2) / self.m
        self.L_1 = 0

    def Prox(self, tau, eps_k, in_prox):

        return st.prox_norm_ell_1(tau * eps_k, in_prox)

    def Grad(self, eps_k, in_grad):

        y_pred = st.sigmoid(self.X_train @ in_grad)

        return 1 / self.m * (self.X_train.T @ (y_pred - self.y_train))

    def res(self, x, x_old):

        return np.sum((x - x_old) ** 2)

    def obj(self, x):

        y_pred = st.sigmoid(self.X_train @ x)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)

        return -np.mean(self.y_train * np.log(y_pred) +
                        (1 - self.y_train) * np.log(1 - y_pred))

    def obj_outer(self, x):

        return np.sum(np.abs(x))
