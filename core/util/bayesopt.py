# Copyright (c) 2020 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Author: Lukas P. Fröhlich, lukas.froehlich@de.bosch.com


import GPy
import numpy as np
import pickle
import os
from tqdm import tqdm
import time

import core.util.gp as gp_module
import core.util.misc as misc
from core.util.stats import calc_sigma_points_and_weights


def run_bo_nes(acq, objective, param, x_init, y_init, run_idx, res_dir,
               hyper_opt=False, hyper_opt_iter=1, timing_on=False):
    while True:
        try:
            # Set up the Gaussian processes
            k_f = GPy.kern.RBF(input_dim=param['input_dim'], variance=param['signal_var'],
                               lengthscale=param['lengthscale'], ARD=True)
            gp = gp_module.GP(k_f, x_init, y_init, param['noise_var'], normalize_Y=True)
            compute_durations = []

            # Store evaluated points
            X, Y = x_init, y_init
            x_belief = np.empty((param['max_iter'], param['input_dim']))
            g_belief = np.empty((param['max_iter'],))

            for it in tqdm(range(param['max_iter']), disable=False):
                # print("Iter #{} (idx {})".format(it, run_idx))
                if hyper_opt and not np.fmod(it, hyper_opt_iter):
                    gp = optimize_hyperparameters(gp, param)

                t0 = time.time()
                x_next = acq.next_point(gp)
                compute_durations.append(time.time() - t0)

                y_next = objective(x_next, param['noise_var'])
                X = np.vstack((X, x_next))
                Y = np.vstack((Y, y_next))
                gp.set_xy(X, Y)

                # Calculate current belief of optimum
                ngp = gp_module.NoisyInputGP.from_gp(gp, param['input_var'])
                x_guess, g_guess = misc.optimize_gp_2(ngp, acq.domain, n_restarts=100)
                x_belief[it] = x_guess
                g_belief[it] = g_guess

            sub_res_dir = res_dir + acq.__name__
            if not os.path.exists(sub_res_dir):
                os.mkdir(sub_res_dir)
            res = {'x_belief': x_belief, 'g_belief': g_belief, 'x_eval': X, 'f_eval': Y.squeeze()}
            pickle.dump(res, open(sub_res_dir + "/res_{0:03d}.pkl".format(run_idx), "wb"))
            pickle.dump(compute_durations, open(sub_res_dir + "/duration_{0:03d}.pkl".format(run_idx), "wb")) if timing_on else None

            return x_belief, g_belief, X, Y.squeeze()
        except ValueError:
            print("Warning: Need to restart due to error.")
            pass


def run_bo_unsc(acq, objective, param, x_init, y_init, run_idx, res_dir,
                hyper_opt=False, hyper_opt_iter=1, timing_on=False):
    """
    Bayesian Optimization with unscented transformation.
    """
    # Set up the Gaussian process
    k_f = GPy.kern.RBF(input_dim=param['input_dim'], variance=param['signal_var'],
                       lengthscale=param['lengthscale'], ARD=True)
    gp = gp_module.GP(k_f, x_init, y_init, param['noise_var'])
    compute_durations = []

    # Store evaluated points
    X, Y = x_init, y_init
    x_belief = np.empty((param['max_iter'], param['input_dim']))
    g_belief = np.empty((param['max_iter'],))
    for it in tqdm(range(param['max_iter']), disable=False):
        if hyper_opt and not np.fmod(it, hyper_opt_iter):
            gp = optimize_hyperparameters(gp, param)

        t0 = time.time()
        x_next = acq.next_point(gp)
        compute_durations.append(time.time() - t0)
        y_next = objective(x_next, param['noise_var'])
        X = np.vstack((X, x_next))
        Y = np.vstack((Y, y_next))
        gp.set_xy(X, Y)

        # Calculate current belief of optimum
        x_guess, g_guess = misc.optimize_gp_unsc(gp, acq.domain, acq, n_restarts=10)
        x_belief[it] = x_guess
        g_belief[it] = g_guess

    sub_res_dir = res_dir + acq.__name__
    if not os.path.exists(sub_res_dir):
        os.mkdir(sub_res_dir)
    res = {'x_belief': x_belief, 'g_belief': g_belief, 'x_eval': X, 'f_eval': Y.squeeze()}
    pickle.dump(res, open(sub_res_dir + "/res_{0:03d}.pkl".format(run_idx), "wb"))
    pickle.dump(compute_durations, open(sub_res_dir + "/duration_{0:03d}.pkl".format(run_idx), "wb")) if timing_on else None

    return x_belief, g_belief, X, Y.squeeze()


def run_bo_uu(acq, objective, param, x_init, y_init, run_idx, res_dir,
              hyper_opt=False, hyper_opt_iter=1, timing_on=False):
    """
    Bayesian Optimization under Uncertainty. All these methods only depend on
    the NoisyGP. Thus we can simply exchange the acquisition function.
    """
    # Set up the Gaussian processes
    k_f = GPy.kern.RBF(input_dim=param['input_dim'], variance=param['signal_var'],
                       lengthscale=param['lengthscale'], ARD=True)
    gp = gp_module.GP(k_f, x_init, y_init, param['noise_var'])
    ngp = gp_module.NoisyInputGP.from_gp(gp, param['input_var'])
    compute_durations = []

    # Store evaluated points
    X, Y = x_init, y_init
    x_belief = np.empty((param['max_iter'], param['input_dim']))
    g_belief = np.empty((param['max_iter'],))
    for it in tqdm(range(param['max_iter']), disable=False):
        if hyper_opt and not np.fmod(it, hyper_opt_iter):
            gp = optimize_hyperparameters(gp, param)
            ngp = gp_module.NoisyInputGP.from_gp(gp, param['input_var'])

        t0 = time.time()
        x_next = acq.next_point(ngp)
        compute_durations.append(time.time() - t0)
        y_next = objective(x_next, param['noise_var'])
        X = np.vstack((X, x_next))
        Y = np.vstack((Y, y_next))
        gp.set_xy(X, Y)
        ngp = gp_module.NoisyInputGP.from_gp(gp, param['input_var'])

        # Calculate current belief of optimum
        x_guess, g_guess = misc.optimize_gp_2(ngp, acq.domain, n_restarts=100)
        x_belief[it] = x_guess
        g_belief[it] = g_guess

    sub_res_dir = res_dir + acq.__name__ + "_uu"
    if not os.path.exists(sub_res_dir):
        os.mkdir(sub_res_dir)
    res = {'x_belief': x_belief, 'g_belief': g_belief, 'x_eval': X, 'f_eval': Y.squeeze()}
    pickle.dump(res, open(sub_res_dir + "/res_{0:03d}.pkl".format(run_idx), "wb"))
    pickle.dump(compute_durations, open(sub_res_dir + "/duration_{0:03d}.pkl".format(run_idx), "wb")) if timing_on else None

    return x_belief, g_belief, X, Y.squeeze()


def run_bo_vanilla(acq, objective, param, x_init, y_init, run_idx, res_dir,
                   hyper_opt=False, hyper_opt_iter=1, timing_on=False):
    """
    Vanilla Bayesian Optimization, i.e., no robustness considered.
    """
    # Set up the Gaussian process
    k_f = GPy.kern.RBF(input_dim=param['input_dim'], variance=param['signal_var'],
                       lengthscale=param['lengthscale'], ARD=True)
    gp = gp_module.GP(k_f, x_init, y_init, param['noise_var'], normalize_Y=True)
    compute_durations = []

    # Store evaluated points
    X, Y = x_init, y_init
    x_belief = np.empty((param['max_iter'], param['input_dim']))
    g_belief = np.empty((param['max_iter'],))
    for it in tqdm(range(param['max_iter']), disable=True):
        if hyper_opt and not np.fmod(it, hyper_opt_iter):
            gp = optimize_hyperparameters(gp, param)

        t0 = time.time()
        x_next = acq.next_point(gp)
        compute_durations.append(time.time() - t0)
        y_next = objective(x_next, param['noise_var'])
        X = np.vstack((X, x_next))
        Y = np.vstack((Y, y_next))
        gp.set_xy(X, Y)

        # Calculate current belief of optimum
        x_guess, g_guess = misc.optimize_gp_2(gp, acq.domain, n_restarts=100)
        x_belief[it] = x_guess
        g_belief[it] = g_guess

    sub_res_dir = res_dir + acq.__name__ + '_vanilla'
    if not os.path.exists(sub_res_dir):
        os.mkdir(sub_res_dir)
    res = {'x_belief': x_belief, 'g_belief': g_belief, 'x_eval': X, 'f_eval': Y.squeeze()}
    pickle.dump(res, open(sub_res_dir + "/res_{0:03d}.pkl".format(run_idx), "wb"))
    pickle.dump(compute_durations, open(sub_res_dir + "/duration_{0:03d}.pkl".format(run_idx), "wb")) if timing_on else None

    return x_belief, g_belief, X, Y.squeeze()


def optimize_hyperparameters(gp, param):
    # I did not want to implement hyperparameter optimization. Let GPy do the work.
    k_hyper = GPy.kern.RBF(input_dim=param['input_dim'], ARD=True)
    gp_hyper = GPy.models.GPRegression(gp.X, gp.Y, kernel=k_hyper)

    # Give enough room for optimization (4 orders of magnitude) but prevent extremely large noise
    # such that GP prediction is essentially flat with constant noise.
    gp_hyper.likelihood.constrain_bounded(0.01 * param['noise_var'], 100 * param['noise_var'], warning=False)

    for i in range(gp_hyper.kern.lengthscale.shape[0]):
        mu = np.log(np.sqrt(param['input_var']))
        mu = mu[i] if type(param['input_var']) == list else mu

        ell_prior = GPy.priors.LogGaussian(mu, 0.07)
        gp_hyper.kern.lengthscale[[i]].set_prior(ell_prior, warning=False)

    gp_hyper.optimize()

    gp.kernel.lengthscale[:] = gp_hyper.kern.lengthscale[:]
    gp.kernel.variance[:] = gp_hyper.kern.variance[:]
    gp.noise_var = gp_hyper.likelihood.variance[:]
    gp.set_xy(gp.X, gp.Y)  # Need this, updates the Gram matrix

    return gp

