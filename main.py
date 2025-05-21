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
Run this file to reproduce all numerical experiments in:

R. I. Bot, E. Chenchene, R. Csetnek, D. Hulett.
Accelerating Diagonal Methods for Bilevel Optimization:
Unified Convergence via Continuous-Time Dynamics.
2025. DOI: 10.48550/arXiv.2505.14389.

For any comment, please contact: enis.chenchene@gmail.com
"""

import pathlib
import experiments as expm

if __name__ == "__main__":

    pathlib.Path("results").mkdir(parents=True, exist_ok=True)

    print('Starting experiment in Section 5.2 ...')
    expm.experiment_nemirovsky()

    print('Starting experiment in Section 5.3 ...')
    expm.experiment_logistic()
