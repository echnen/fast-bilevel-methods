# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:01:01 2025

@author: enisc
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.mstats import gmean

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font', **{'family' : 'serif', 'serif' : ['Times'], 'size':15})
rc('text', usetex=True)


def plot_nemirovsky(Res_Bi_PG, Res_biFI, Res_FBi_PG, Res_staBiM, Res_Bi_SG_II,
                    Obj_Bi_PG, Obj_biFI, Obj_FBi_PG, Obj_staBiM, Obj_Bi_SG_II,
                    Obj_H_Bi_PG, Obj_H_biFI, Obj_H_FBi_PG, Obj_H_staBiM, Obj_H_Bi_SG_II, maxit, Spects, cases):

    # plotting inner objectives
    plt.figure(figsize=(5, 5))

    plt.loglog(Obj_Bi_PG, color='y', alpha=0.1)
    plt.loglog(Obj_biFI, color='k', alpha=0.1)
    plt.loglog(Obj_FBi_PG, color='g', alpha=0.1)
    plt.loglog(Obj_staBiM, color='r', alpha=0.1)
    plt.loglog(Obj_Bi_SG_II, color='b', alpha=0.1)

    plt.loglog(gmean(Obj_Bi_PG, axis=1), color='y', label='Alg. 1', linewidth=2)
    plt.loglog(gmean(Obj_biFI, axis=1), color='k', label='Alg. 2', linewidth=2)
    plt.loglog(gmean(Obj_FBi_PG, axis=1), color='g', label='FBi-PG', linewidth=2)
    plt.loglog(gmean(Obj_staBiM, axis=1), color='r', label='staBiM', linewidth=2)
    plt.loglog(gmean(Obj_Bi_SG_II, axis=1), color='b', label='Bi-SG-II', linewidth=2)

    plt.loglog(range(maxit), [1e3 / (k + 1) ** 1 for k in range(maxit)], '--', alpha=.4, color='k')
    plt.loglog(range(maxit), [1 / (k + 1) ** 2 for k in range(maxit)], '--', alpha=.4, color='k')

    plt.xlim(1, maxit)
    plt.ylabel(r'$F(x^k) - \min F$')
    plt.xlabel(r'Iteration number $(k)$')
    plt.grid()
    plt.legend()
    plt.savefig('results/exp_nemirovsky_obj.pdf', bbox_inches='tight')
    plt.show()


    # plotting objectives outer (comparison)
    fig = plt.figure(figsize=(5, 5))
    plt.loglog(Obj_H_Bi_PG, color='y', alpha=0.05)
    plt.loglog(Obj_H_biFI, color='k', alpha=0.05)
    plt.loglog(Obj_H_FBi_PG, color='g', alpha=0.05)
    plt.loglog(Obj_H_staBiM, color='r', alpha=0.05)
    plt.loglog(Obj_H_Bi_SG_II, color='b', alpha=0.05)

    plt.loglog(gmean(Obj_H_Bi_PG, axis=1), color='y', label='Alg. 1', linewidth=2)
    plt.loglog(gmean(Obj_H_biFI, axis=1), color='k', label='Alg. 2', linewidth=2)
    plt.loglog(gmean(Obj_H_FBi_PG, axis=1), color='g', label='FBi-PG', linewidth=2)
    plt.loglog(gmean(Obj_H_staBiM, axis=1), color='r', label='staBiM', linewidth=2)
    plt.loglog(gmean(Obj_H_Bi_SG_II, axis=1), color='b', label='Bi-SG-II', linewidth=2)

    plt.xlim(1e1, maxit)
    plt.ylim(1, 500)
    plt.ylabel(r'$|H(x_k) - H(x^*)|$')
    plt.xlabel(r'Iteration number $(k)$')
    plt.grid()
    plt.savefig('results/exp_nemirovsky_obj_outer_comparison.pdf', bbox_inches='tight')
    plt.show()

    # plotting distance to solution
    plt.figure(figsize=(5, 5))

    plt.loglog(Res_Bi_PG, color='y', alpha=0.1)
    plt.loglog(Res_biFI, color='k', alpha=0.1)
    plt.loglog(Res_FBi_PG, color='g', alpha=0.1)
    plt.loglog(Res_staBiM, color='r', alpha=0.1)
    plt.loglog(Res_Bi_SG_II, color='b', alpha=0.1)

    plt.loglog(gmean(Res_Bi_PG, axis=1), color='y', linewidth=3, label='Alg. 1')
    plt.loglog(gmean(Res_biFI, axis=1), color='k', linewidth=3, label='Alg. 2')
    plt.loglog(gmean(Res_FBi_PG, axis=1), color='g', linewidth=3, label='FBi-PG')
    plt.loglog(gmean(Res_staBiM, axis=1), color='r', linewidth=3, label='staBiM')
    plt.loglog(gmean(Res_Bi_SG_II, axis=1), color='b', linewidth=3, label='Bi-SG-II')

    plt.xlim(1e1, maxit)
    plt.ylim(1, 50000)

    plt.ylabel(r'$\|x^k - x^*\|^2$')
    plt.xlabel(r'Iteration number $(k)$')
    plt.grid()
    plt.savefig('results/exp_nemirovsky_res.pdf', bbox_inches='tight')
    plt.show()


    # # plotting objectives outer (first order)
    # fig = plt.figure(figsize=(5, 5))
    # cmap = LinearSegmentedColormap.from_list("WhiteBlue", [(0, 0, 0),
    #                                                        (0, 0, 1),
    #                                                        (0.5, 0.9, 1)])

    # norm = Normalize(np.min(Spects) / 2, np.max(Spects) / 2)
    # sm = ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # fig = plt.figure(figsize=(5, 5))

    # for cs in range(cases):

    #     delta = Spects[cs] / 2
    #     color = cmap(norm(delta))
    #     plt.loglog(Obj_H_Bi_PG[:, cs], color=color, alpha=0.2)
    #     # plt.loglog(range(maxit), [1 / (k + 1) ** (1 - delta) for k in range(maxit)], '--', alpha=.4, color='k')

    # plt.xlim(1e1, maxit)
    # plt.ylabel(r'$|H(x^k) - H(x^*)|$')
    # plt.xlabel(r'Iteration number $(k)$')
    # plt.ylim(1e-2, 1e4)
    # plt.grid()
    # cbar_ax = fig.add_axes([0.38, 0.83, .5, 0.02])
    # plt.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=r'$\delta$')
    # # plt.legend()
    # plt.savefig('results/exp_nemirovsky_obj_outer_second_order.pdf', bbox_inches='tight')
    # plt.show()


    # plotting objectives outer (second order)
    fig = plt.figure(figsize=(5, 5))
    cmap = LinearSegmentedColormap.from_list("WhiteBlue", [(0, 0, 0),
                                                            (0, 0, 1),
                                                            (0.5, 0.9, 1)])
    norm = Normalize(np.min(Spects), np.max(Spects))
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig = plt.figure(figsize=(5, 5))

    for cs in range(cases):
        delta = Spects[cs]
        color = cmap(norm(Spects[cs]))
        plt.loglog(Obj_H_biFI[:, cs], color=color, alpha=0.5)
        # plt.loglog(range(maxit), [1 / (k + 1) ** (2 - delta) for k in range(maxit)], '--', alpha=.4, color='k')

    plt.xlim(1e1, maxit)
    plt.ylabel(r'$|H(x_k) - H(x^*)|$')
    plt.xlabel(r'Iteration number $(k)$')
    plt.ylim(1e-2, 1e3)
    plt.grid()
    # cbar_ax = fig.add_axes([0.38, 0.83, .5, 0.02])
    # plt.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=r'$\delta$')
    cbar_ax = fig.add_axes([.95, .15, .02, .7])
    plt.colorbar(sm, cax=cbar_ax, orientation="vertical", label=r'Value of $\delta$')
    plt.savefig('results/exp_nemirovsky_obj_outer_second_order.pdf', bbox_inches='tight')
    plt.show()

    # plotting objectives inner (second order)
    fig = plt.figure(figsize=(5, 5))
    cmap = LinearSegmentedColormap.from_list("WhiteBlue", [(0, 0, 0),
                                                            (0, 0, 1),
                                                            (0.5, 0.9, 1)])
    norm = Normalize(np.min(Spects), np.max(Spects))
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig = plt.figure(figsize=(5, 5))

    for cs in range(cases):
        delta = Spects[cs]
        color = cmap(norm(delta))
        plt.loglog(Obj_biFI[:, cs], color=color, alpha=0.5)
        # plt.loglog(range(maxit), [1 / (k + 1) ** (2 - delta) for k in range(maxit)], '--', alpha=.4, color='k')

    plt.xlim(1e1, maxit)
    plt.ylabel(r'$F(x_k) - \min F$')
    plt.xlabel(r'Iteration number $(k)$')
    # plt.ylim(1e-2, 1e4)
    plt.grid()
    # cbar_ax = fig.add_axes([.95, .15, .02, .7])
    # plt.colorbar(sm, cax=cbar_ax, orientation="vertical", label=r'Value of $\delta$')
    plt.savefig('results/exp_nemirovsky_obj_inner_second_order.pdf', bbox_inches='tight')
    plt.show()

    return


def plot_logistic(Res_Bi_PG, Res_biFI, Res_FBi_PG, Res_staBiM, Res_Bi_SG_II,
                  Obj_Bi_PG, Obj_biFI, Obj_FBi_PG, Obj_staBiM, Obj_Bi_SG_II,
                  Obj_H_Bi_PG, Obj_H_biFI, Obj_H_FBi_PG, Obj_H_staBiM, Obj_H_Bi_SG_II, maxit):

    # plotting residual
    plt.figure(figsize=(5, 5))

    plt.loglog(Res_Bi_PG,  color='y', linewidth=3, label='Alg. 1')
    plt.loglog(Res_biFI, color='k', linewidth=3, label='Alg. 2')
    plt.loglog(Res_FBi_PG, color='g', linewidth=3, label='FBi-PG')
    plt.loglog(Res_staBiM, color='r', linewidth=3, label='staBiM')
    plt.loglog(Res_Bi_SG_II, color='b', linewidth=3, label='Bi-SG-II')

    # plt.loglog(gmean(Res_Bi_PG, axis=1), color='y', linewidth=3, label='Alg. 1')
    # plt.loglog(gmean(Res_biFI, axis=1), color='k', linewidth=3, label='Alg. 2')
    # plt.loglog(gmean(Res_FBi_PG, axis=1), color='g', linewidth=3, label='FBi-PG')
    # plt.loglog(gmean(Res_staBiM, axis=1), color='r', linewidth=3, label='staBiM')
    # plt.loglog(gmean(Res_Bi_SG_II, axis=1), color='b', linewidth=3, label='Bi-SG-II')

    plt.xlim(1, maxit)
    plt.ylabel(r'$\|x^{k + 1} - x^{k}\|^2$')
    plt.xlabel(r'Iteration number $(k)$')
    plt.grid()
    plt.savefig('results/exp_logistic_res.pdf', bbox_inches='tight')
    plt.show()

    # plotting objectives inner
    plt.figure(figsize=(5, 5))

    min_f = min(np.min(Obj_Bi_PG), np.min(Obj_biFI), np.min(Obj_FBi_PG), np.min(Obj_staBiM),
                np.min(Obj_Bi_SG_II))

    plt.loglog(Obj_Bi_PG - min_f, color='y', label='Alg. 1', linewidth=3)
    plt.loglog(Obj_biFI - min_f, color='k', label='Alg. 2', linewidth=3)
    plt.loglog(Obj_FBi_PG - min_f, color='g', label='FBi-PG', linewidth=3)
    plt.loglog(Obj_staBiM - min_f, color='r', label='staBiM', linewidth=3)
    plt.loglog(Obj_Bi_SG_II - min_f, color='b', label='Bi-SG-II', linewidth=3)

    plt.loglog(range(maxit), [1e3 / (k + 1) ** (1.9 / 2) for k in range(maxit)], '--', alpha=0.4, color='k')
    plt.loglog(range(maxit), [1e3 / (k + 1) ** 1.9 for k in range(maxit)], '--', alpha=0.4, color='k')

    plt.xlim(1e2, maxit - int(maxit * 2 / 10))
    plt.ylabel(r'$f(x^k) - \min f$')
    plt.xlabel(r'Iteration number $(k)$')
    plt.grid()
    plt.legend()
    plt.savefig('results/exp_logistic_obj_inner.pdf', bbox_inches='tight')
    plt.show()

    # plotting objectives outer
    plt.figure(figsize=(5, 5))

    # outer optimal estimated
    opt_H = np.mean((Obj_H_Bi_PG[-1], Obj_H_biFI[-1], Obj_H_FBi_PG[-1], Obj_H_FBi_PG[-1], Obj_H_Bi_SG_II[-1]))

    # plt.loglog(np.abs(np.array(Obj_H_Bi_PG) - opt_H), color='y', label='Alg. 1', linewidth=3)
    # plt.loglog(np.abs(np.array(Obj_H_biFI) - opt_H), color='k', label='Alg. 2', linewidth=3)
    # plt.loglog(np.abs(np.array(Obj_H_FBi_PG) - opt_H), color='g', label='FBi-PG', linewidth=3)
    # plt.loglog(np.abs(np.array(Obj_H_FBi_PG) - opt_H), color='r', label='staBiM', linewidth=3)
    # plt.loglog(np.abs(np.array(Obj_H_Bi_SG_II) - opt_H), color='b', label='Bi-SG-II', linewidth=3)

    plt.loglog(Obj_H_Bi_PG, color='y', label='Alg. 1', linewidth=3)
    plt.loglog(Obj_H_biFI, color='k', label='Alg. 2', linewidth=3)
    plt.loglog(Obj_H_FBi_PG, color='g', label='FBi-PG', linewidth=3)
    plt.loglog(Obj_H_FBi_PG, color='r', label='staBiM', linewidth=3)
    plt.loglog(Obj_H_Bi_SG_II, color='b', label='Bi-SG-II', linewidth=3)

    plt.xlim(1, maxit - int(maxit * 2 / 10))
    plt.ylabel(r'$H(x_k)$')
    plt.xlabel(r'Iteration number $(k)$')
    plt.grid()
    # plt.legend()
    plt.savefig('results/exp_logistic_obj_outer.pdf', bbox_inches='tight')
    plt.show()

    return
