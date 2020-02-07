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


from GPy.util.linalg import jitchol
import numpy as np
from scipy import optimize
from scipy import signal
from scipy import stats
import sobol_seq
import datetime
import os
import time


def compute_stable_cholesky(K):
    K = 0.5 * (K + K.T)
    L = jitchol(K, maxtries=5)
    return L, True


def filter_signal(sig, dx, filter_width, filter_type='gauss'):
    """
    Filter a signal with given window using convolution.

    :param sig: signal to be filtered. Needs to be defined on uniform grid.
    :param dx: resolution of signal, i.e., (x_max - x_min) / signal_length
    :param filter_width: width of the filter.
    :param filter_type: type of filter, either 'gauss' or 'uniform'.
    :return: filtered signal.
    """
    sig = sig.squeeze()
    dim = sig.ndim

    if filter_type in ['gauss', 'SE', 'squared_exp']:
        # Gaussian window is +/- 5 standard deviations large
        dw = 5 * filter_width
        window_range = np.arange(-dw, dw + dx, dx)
        window_range_list = dim * [window_range]
        x_grids = np.asarray(np.meshgrid(*window_range_list))
        pos = np.moveaxis(x_grids, 0, -1)  # Make first dimension last dimension

        mu = np.zeros((dim,))
        cov = filter_width ** 2 * np.eye(dim)  # Filter is isotropic
        window = stats.multivariate_normal.pdf(pos.squeeze(), mu, cov)
    elif filter_type in ['uniform', 'box', 'rect', 'rectangular']:
        n = dim * [int(filter_width / dx)]
        window = np.ones(n)
    else:
        raise NotImplementedError

    filtered_signal = signal.convolve(sig, window, mode='same')
    filtered_signal = filtered_signal / np.sum(window)

    if dim == 1:
        filtered_signal = filtered_signal[:, None]
    return filtered_signal


def create_meshgrid(x_min, x_max, dx, dim):
    if dim == 1 and type(x_min) is not np.ndarray:
        x_min = np.array([x_min])
        x_max = np.array([x_max])
    elif dim > 1 and type(x_min) is not np.ndarray:
        x_min = x_min * np.ones((dim,))
        x_max = x_max * np.ones((dim,))

    # Note: using this double reversing for meshgrid makes it the same as np.mgrid,
    #       however np.mgrid cannot deal with unpacked lists as arguments.
    x_meshes = np.meshgrid(*[np.arange(x_min[i], x_max[i] + dx, dx) for i in reversed(range(dim))])
    return x_meshes


def create_meshgrid_lin(x_min, x_max, n, dim):
    if dim == 1 and type(x_min) is not np.ndarray:
        x_min = np.array([x_min])
        x_max = np.array([x_max])
    elif dim > 1 and type(x_min) is not np.ndarray:
        x_min = x_min * np.ones((dim,))
        x_max = x_max * np.ones((dim,))

    # Note: using this double reversing for meshgrid makes it the same as np.mgrid,
    #       however np.mgrid cannot deal with unpacked lists as arguments.
    x_meshes = np.meshgrid(*[np.linspace(x_min[i], x_max[i], n) for i in reversed(range(dim))])
    return x_meshes


def create_flattened_meshgrid(x_min, x_max, dx, dim):
    x_meshes = create_meshgrid(x_min, x_max, dx, dim)
    mesh_shape = x_meshes[0].shape
    xs = np.vstack([x.flatten() for x in reversed(x_meshes)]).T
    return xs, mesh_shape


def create_flattened_meshgrid_lin(x_min, x_max, n, dim):
    x_meshes = create_meshgrid_lin(x_min, x_max, n, dim)
    mesh_shape = x_meshes[0].shape
    xs = np.vstack([x.flatten() for x in reversed(x_meshes)]).T
    return xs, mesh_shape


def optimize_gp_2(gp, domain, n_restarts=100):
    def f(x):
        val, _ = gp.predict(np.atleast_2d(x))
        return -1 * val

    dim = gp.kernel.input_dim
    x0_candidates = domain.lb + (domain.ub - domain.lb) * \
                    sobol_seq.i4_sobol_generate(dim, n_restarts) + \
                    np.random.randn(n_restarts, dim)
    x_opt_candidates = np.empty((n_restarts, dim))
    f_opt = np.empty((n_restarts,))
    for i, x0 in enumerate(x0_candidates):
        res = optimize.minimize(fun=f, x0=x0, bounds=domain)
        x_opt_candidates[i] = res['x']
        f_opt[i] = -1 * res['fun']

    idx_opt = np.argmax(f_opt)
    x_opt = x_opt_candidates[idx_opt]
    f_opt = f_opt[idx_opt]

    return x_opt, f_opt


def optimize_gp_unsc(gp, domain, acq_unsc, n_restarts=100):
    w_sig = acq_unsc._calc_sigma_weights()

    def f(x):
        x = np.atleast_2d(x)
        x_sig = acq_unsc._calc_sigma_points(x)
        return np.sum([wi*gp.predict(xi)[0] for wi, xi in zip(w_sig, x_sig)])

    x_guess, f_guess_unsc = maximize_function(f=f, domain=domain, n_restarts=n_restarts)
    return x_guess, f_guess_unsc


def maximize_function(f, domain, n_restarts=10):
    dim = domain.lb.shape[0]
    x_opt = np.empty((n_restarts, dim))
    f_opt = np.empty((n_restarts,))
    x0_candidates = domain.lb + (domain.ub - domain.lb) * \
                    sobol_seq.i4_sobol_generate(dim, n_restarts)
    for i, x0 in enumerate(x0_candidates):
        res = optimize.minimize(fun=neg(f), x0=x0, bounds=domain)
        x_opt[i, :] = res['x']
        f_opt[i] = -1 * res['fun']

    idx_guess = np.argmax(f_opt)
    x_guess = x_opt[idx_guess]
    f_guess = f_opt[idx_guess]
    return x_guess, f_guess


def conv_wrapper(func, x, filter_width, n, dim, filter_type='gauss'):
    """Wrapper around a function func s.t. we can evaluate the convolution at
    a given x.

    :param func: handle to function that shall be wrapped.
    :param x: evaluation points.
    :param filter_width: width of the filter to be used.
    :param n: number of points used for discrete convolution.
    :param dim: dimensionality of the input.
    :param filter_type: type of filter that is to be used.

    :return Value of the convolution between func and filter at x.
    """
    if x.ndim == 1 and dim == 1:
        x = np.atleast_2d(x).T
    elif x.ndim == 1 and dim > 1:
        x = np.atleast_2d(x)
    n_data = x.shape[0]

    if type(filter_width) == float or type(filter_width) == np.float64:
        filter_width = np.array(dim * [filter_width])
    elif type(filter_width) == np.ndarray:
        assert filter_width.shape[0] == dim
    elif type(filter_width) == list:
        assert len(filter_width) == dim
        filter_width = np.array(filter_width)

    if filter_type in ['gauss', 'SE', 'squared_exp']:
        # Gaussian window is +/- 5 standard deviations large
        window_range_list = []
        for wi in filter_width:
            dw = 5 * wi
            window_range_list.append(np.linspace(-dw, dw, n))

        x_grids = np.asarray(np.meshgrid(*window_range_list))
        pos = np.moveaxis(x_grids, 0, -1)  # Make first dimension last dimension

        mu = np.zeros((dim,))
        cov = np.diag(filter_width ** 2)

        window = stats.multivariate_normal.pdf(pos.squeeze(), mu, cov)
    elif filter_type in ['uniform', 'box', 'rect', 'rectangular']:
        window = np.ones(dim * [n])
        dw = filter_width
    else:
        raise NotImplementedError

    func_filt = np.empty(n_data)
    for i, xi in enumerate(x):
        xs, mesh_shape = create_flattened_meshgrid_lin(xi-dw, xi+dw, n, dim)
        fs = func(xs)
        fs = np.reshape(fs, mesh_shape)
        func_filt[[i]] = signal.convolve(fs, window, mode='valid')

    return func_filt / np.sum(window)


def generate_filtered_samples_on_grid(gp, x_min, x_max, dx, n_samples, filter_width, filter_type='gauss', with_unfiltered=False):
    dim = gp.kernel.input_dim

    # For the filtering of the samples we need a larger domain s.t. we can
    # neglect the boundary effects.
    if filter_type == 'gauss':
        x_increase = 5 * filter_width
    elif filter_type == 'uniform':
        x_increase = filter_width
    else:
        raise NotImplementedError
    x2_min = x_min - x_increase
    x2_max = x_max + x_increase

    # Create meshes on small and large domain
    x_meshes = create_meshgrid(x_min, x_max, dx, dim)
    x2_meshes = create_meshgrid(x2_min, x2_max, dx, dim)
    mesh_shape = x_meshes[0].shape
    mesh2_shape = np.array(x2_meshes[0].shape)
    xs2 = np.vstack([x.flatten() for x in reversed(x2_meshes)]).T
    dn2 = (0.5 * (mesh2_shape - mesh_shape)).astype(int)
    ranges = tuple([slice(dn2[i], dn2[i] + mesh_shape[i]) for i in range(dim)])

    # Sample and filter on large domain
    gp2_samples = gp.sample_posterior(xs2, n_samples)
    gp2_samples = np.reshape(gp2_samples, [*mesh2_shape, -1])
    gp2_samples_filtered = np.array(
        [filter_signal(gp2_samples[..., i], dx, filter_width) for i in range(n_samples)])

    # Bring back to small domain
    gp_samples_filtered = np.array(
        [gp2_samples_filtered[i, ...][ranges] for i in range(n_samples)]).squeeze()
    gp_samples = np.array(
        [gp2_samples[..., i][ranges] for i in range(n_samples)])

    if with_unfiltered:
        return gp_samples_filtered, gp_samples
    else:
        return gp_samples_filtered


def neg(f):
    return lambda x: -1 * f(x)


def timed_prefix(ts=None, str_format='%Y-%m-%d_%H-%M/'):
    ts = time.time() if ts is None else ts
    return datetime.datetime.fromtimestamp(ts).strftime(str_format)


def create_results_dir(script_name):
    results_dir = 'Results/' + script_name + '/' + timed_prefix()
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    else:
        i = 1
        while os.path.exists(results_dir[:-1] + '_%03d/' % i):
            i += 1
        results_dir = results_dir[:-1] + '_%03d/' % i
        os.makedirs(results_dir)
    print("Created: {}".format(results_dir))
    return results_dir


def project_in_domain(x, domain):
    return np.maximum(domain.lb, np.minimum(domain.ub, x))
