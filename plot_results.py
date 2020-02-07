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

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from tqdm import tqdm

from core.util import objectives
from core.util import misc


def plot_convergence(res_dir, objective, show_true=False):

    # Processing the data may be time-consuming, thus save processed data
    res_dir_proc = res_dir[:-1] + '_proc/'
    processed_data_exists = False
    if os.path.exists(res_dir_proc):
        processed_data_exists = True
        print("There exists an already processes version of this directory.")

    method_keys = ['nes_sampling',
                   'nes_ep',
                   'ucb_uu',
                   'ei_uu',
                   'uei',
                   'ei_vanilla',
                   'ei_pseudo',
                   'ucb_vanilla',
                   'mes_naive_vanilla',
                   'mes_naive_uu',
                   'mes_gumbel_uu']

    lw = 5
    common_args = dict(capsize=1, capthick=2, errorevery=1)
    label_args = {method_keys[0]: dict(color='C0', label='NES-Grid', ls='--', lw=lw),
                  method_keys[1]: dict(color='C2', label='NES-EP', ls='--', lw=lw),
                  method_keys[2]: dict(color='C4', label='UCB-UU', ls='-.', lw=lw),
                  method_keys[3]: dict(color='C5', label='EI-UU', ls='-.', lw=lw),
                  method_keys[4]: dict(color='C6', label='Unsc. EI', ls='-.', lw=lw),
                  method_keys[5]: dict(color='C7', label='Vanilla EI', ls='-', lw=lw),
                  method_keys[6]: dict(color='C8', label='Pseudo EI', ls=':', lw=lw),
                  method_keys[7]: dict(color='C9', label='Vanilla UCB', ls='--', lw=lw),
                  method_keys[9]: dict(color='C8', label='MES-UU', ls='-.', lw=lw),
                  method_keys[10]: dict(color='C8', label='MES-UU', ls='-.', lw=lw)}

    # Get some general parameters for notational convenience
    param = pickle.load(open(res_dir + 'param.pkl', 'rb'))
    g_opt = param['g_opt']
    filter_width = np.sqrt(param['input_var'])

    # Wrapper for robust objective (depending on dimensionality, this takes some time to evaluate)
    def objective_filtered(x):
        return misc.conv_wrapper(objective, x, filter_width, 101, param['input_dim'])

    if objective == objectives.synthetic_1d_01:
        x_opt = np.array([0.31111868])
    elif objective == objectives.rkhs_synth:
        x_opt = np.array([0.3157128])
    elif objective == objectives.gmm_2d:
        x_opt = np.array([0.20029798, 0.20022463])
    elif objective == objectives.synth_poly_2d_norm or objective == objectives.synth_poly_2d:
        x_opt = np.array([0.27593822, 0.69643763])
        g_opt = objective_filtered(x_opt)
    elif objective == objectives.hartmann_3d:
        x_opt = np.array([0.11728554, 0.56940675, 0.83030156])
    else:
        raise ValueError("Error, error...")

    # Unpickle data if exist
    res = {}
    for method_key in method_keys:
        if method_key in os.listdir(res_dir):
            sub_res_dir = res_dir + method_key + "/"

            x_belief = np.zeros((param['max_iter'], param['input_dim'], param['n_runs']))
            g_belief = np.zeros((param['max_iter'], param['n_runs']))
            x_eval = np.zeros((param['max_iter'] + param['n_init'], param['input_dim'], param['n_runs']))
            f_eval = np.zeros((param['max_iter'] + param['n_init'], param['n_runs']))
            for i, res_file_name in enumerate(os.listdir(sub_res_dir)):
                print(i, sub_res_dir + res_file_name)
                tmp = pickle.load(open(sub_res_dir + res_file_name, 'rb'))
                x_belief[:, :, i] = tmp['x_belief']
                g_belief[:, i] = tmp['g_belief']
                x_eval[:, :, i] = tmp['x_eval']
                f_eval[:, i] = tmp['f_eval']
            res[method_key] = {'x_belief': x_belief, 'g_belief': g_belief,
                               'x_eval': x_eval, 'f_eval': f_eval}

    if processed_data_exists:
        gx_belief = pickle.load(open(res_dir_proc + "gx_belief.pkl", "rb"))
    else:
        # Calculate relevant statistics for plotting for each method
        gx_belief = {}
        for method_key, method_res in tqdm(res.items()):
            # Evaluate robust objective at current belief of the optimum
            gx_belief[method_key] = np.zeros((param['max_iter'], param['n_runs']))
            for i in range(param['n_runs']):
                gx_belief[method_key][:, i] = objective_filtered(method_res['x_belief'][:, :, i])

        os.mkdir(res_dir_proc)
        pickle.dump(gx_belief, open(res_dir_proc + "gx_belief.pkl", "wb"))

    # Calculate inference regret and corresponding uncertainty bounds
    inf_regret, inf_regret_bounds = {}, {}
    for method_key in gx_belief.keys():
        # Absolute distance to true robust optimum gives robust inference regret
        inf_regret[method_key] = np.abs(gx_belief[method_key] - g_opt)

        # Calculate confidence bounds for plotting
        percentiles = [25, 75]
        inf_regret_bounds[method_key] = np.percentile(inf_regret[method_key], percentiles, axis=1).T

    # Visualization
    n = np.arange(1, param['max_iter']+1)
    alpha = 0.3

    fig = plt.figure(figsize=(5.0, 5.0))
    ax = fig.add_subplot(111)
    for method_key in inf_regret.keys():
        y = np.median(inf_regret[method_key], axis=1)
        lower = inf_regret_bounds[method_key][:, 0]
        upper = inf_regret_bounds[method_key][:, 1]
        plt.fill_between(n, lower, upper, color=label_args[method_key]['color'], alpha=alpha)
        plt.plot(n, y, **label_args[method_key])

    ax.set_yscale('log')
    plt.xlabel('# Function evaluations')
    plt.tight_layout()

    if show_true:
        plt.show()


if __name__ == '__main__':
    res_dir = './Results/synthetic_1d_01/2020-01-27_10-51/'
    objective = objectives.synthetic_1d_01
    plot_convergence(res_dir, objective, show_true=True)

