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

import json
from multiprocessing import Process, Queue
import queue
import numpy as np
import pickle
import os
from scipy.optimize import Bounds
import time

from core.acquisitions import EI, UCB, UEI, MESNaive, MESGumbel
from core.acquisitions import NESRejectionSamplingGrid, NESExpProp
from core.util import objectives
import core.util.misc as misc
from core.util import bayesopt


def load_settings(objective_name):
    if objective_name == 'synthetic_1d_01':
        objective = objectives.synthetic_1d_01
        objective_config = './cfg/param_synth_01_1d.json'
    elif objective_name == 'rkhs_1d':
        objective = objectives.rkhs_synth
        objective_config = './cfg/param_rkhs_1d.json'
    elif objective_name == 'gmm_2d':
        objective = objectives.gmm_2d
        objective_config = './cfg/param_gmm_2d.json'
    elif objective_name == 'poly_2d':
        objective = objectives.synth_poly_2d_norm
        objective_config = './cfg/param_poly_2d.json'
    elif objective_name == 'hartmann_3d':
        objective = objectives.hartmann_3d
        objective_config = './cfg/param_hartmann_3d.json'
    else:
        raise ValueError("Objective does not exist. Please choose an existing one.")

    with open(objective_config) as f:
        param_obj = json.load(f)

    with open('cfg/exp_params_{}d.json'.format(param_obj['input_dim'])) as f:
        param_exp = json.load(f)
    param = {**param_exp, **param_obj}

    return objective, param


def run_all_experiments_queue(experiments_queue, param, objective, res_dir):
    while True:
        try:
            run_idx = experiments_queue.get_nowait()
        except queue.Empty:
            break
        else:
            domain = Bounds(np.array(param['lower_bound']), np.array(param['upper_bound']))
            print("Elements in queue: {} / {}".format(experiments_queue.qsize() + 1, param['n_runs']))

            # Setup acquisition functions
            acq_nes_grid = NESRejectionSamplingGrid(
                domain=domain, filter_width=np.sqrt(param['input_var']),
                n_max_value_samples=param['n_max_value_samples'],
                n_function_samples=param['n_rejection_samples'], dx_grid=param['dx'])

            acq_nes_ep = NESExpProp(
                domain=domain, filter_width=np.sqrt(param['input_var']),
                n_max_value_samples=param['n_max_value_samples'])

            acq_ucb = UCB(domain=domain)
            acq_ei = EI(domain=domain)
            acq_uei = UEI(domain=domain, input_var=param['input_var'], k=1.0)
            acq_mes = MESNaive(domain=domain, n_samples=param['n_max_value_samples'])
            acq_mes = MESGumbel(domain=domain, n_samples=param['n_max_value_samples'])

            np.random.seed(int(time.time()) + run_idx)  # Important for multiprocessing
            x_init = np.random.uniform(domain.lb, domain.ub, (param['n_init'], param['input_dim']))
            y_init = objective(x_init, param['noise_var'])

            hyper_opt = True
            hyper_opt_iter = 1
            bayesopt.run_bo_nes(acq_nes_grid, objective, param, x_init, y_init, run_idx, res_dir, hyper_opt=hyper_opt, hyper_opt_iter=hyper_opt_iter)
            bayesopt.run_bo_nes(acq_nes_ep, objective, param, x_init, y_init, run_idx, res_dir, hyper_opt=hyper_opt, hyper_opt_iter=hyper_opt_iter)
            bayesopt.run_bo_uu(acq_ei, objective, param, x_init, y_init, run_idx, res_dir, hyper_opt=hyper_opt, hyper_opt_iter=hyper_opt_iter)
            bayesopt.run_bo_uu(acq_ucb, objective, param, x_init, y_init, run_idx, res_dir, hyper_opt=hyper_opt, hyper_opt_iter=hyper_opt_iter)
            bayesopt.run_bo_uu(acq_mes, objective, param, x_init, y_init, run_idx, res_dir, hyper_opt=hyper_opt, hyper_opt_iter=hyper_opt_iter)
            bayesopt.run_bo_unsc(acq_uei, objective, param, x_init, y_init, run_idx, res_dir, hyper_opt=hyper_opt, hyper_opt_iter=hyper_opt_iter)
            bayesopt.run_bo_vanilla(acq_ei, objective, param, x_init, y_init, run_idx, res_dir, hyper_opt=hyper_opt, hyper_opt_iter=hyper_opt_iter)


def main(objective_name):
    # Load Settings
    objective, param = load_settings(objective_name)
    res_dir = misc.create_results_dir(objective_name)
    pickle.dump(param, open(res_dir + "param.pkl", "wb"))

    # Get maximum number of available cores
    number_of_processes = os.cpu_count()

    # Fill queue with experiment IDs
    experiments_queue = Queue()
    for i in range(param['n_runs']):
        experiments_queue.put(i)

    # Spawn processes and work of all experiments in the queue
    processes = []
    for w in range(number_of_processes):
        p = Process(target=run_all_experiments_queue,
                    args=(experiments_queue, param, objective, res_dir))
        processes.append(p)
        p.start()

    # Wait for everyone to be finished
    for p in processes:
        p.join()


if __name__ == '__main__':
    main('synthetic_1d_01')
    # main('rkhs_1d')
    # main('gmm_2d')
    # main('poly_2d')
    # main('hartmann_3d')

