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


import numpy as np
from scipy import stats
from scipy import linalg
import sobol_seq

from core.acquisitions.acquisition_base import AcquisitionBase
from core.util import misc
from core.util import gp as gp_module
from core.util import stats as my_stats


class NESBase(AcquisitionBase):
    def __init__(self, domain, filter_width, n_max_value_samples, filter_type='gauss'):
        """
        Constructor for base class of  Noisy-Input Entropy Search acquisition function.
        """
        AcquisitionBase.__init__(self, domain=domain)

        self.filter_width = filter_width
        self.n_max_value_samples = n_max_value_samples
        self.robust_max_values = None
        self.f_acq_eval = None
        self.filter_type = filter_type

    def next_point(self, gp):
        raise NotImplementedError

    def _f_acq(self, x):
        raise NotImplementedError


class NESRejectionSamplingGrid(NESBase):
    """
    Class that implements NES based on rejection sampling.
    """
    def __init__(self, domain, filter_width, n_max_value_samples,
                 n_function_samples, dx_grid, filter_type='gauss'):
        NESBase.__init__(self, domain, filter_width, n_max_value_samples, filter_type)
        self.__name__ = 'nes_rs'

        self.n_function_samples = n_function_samples
        self.dx_grid = dx_grid

    def next_point(self, gp):
        self.gp = gp
        self.robust_max_values = self._sample_robust_max_values(gp)
        x_next = self._optimize_acq()
        return x_next

    def _f_acq(self, x):
        """
        Note: The input xs must be generated with misc.create_flattened_meshgrid.
        """
        # Calculate H[ p( y | D, x) ]
        _, var = self.gp.predict(x)
        H_pyDx = 0.5 * np.log(2 * np.pi * np.e * (var + self.gp.noise_var))

        # Calculate E[ H[ p( y | D, x, g* ) ] ]
        H_pyDxy = np.empty((x.shape[0], self.robust_max_values.shape[0]))
        for i, max_val in enumerate(self.robust_max_values):
            acc_samples = my_stats.robust_max_value_rejection_sampling_global(
                self.gp, self.domain.lb, self.domain.ub, self.dx_grid, max_val,
                self.n_function_samples, self.filter_width)
            acc_samples += np.sqrt(self.gp.noise_var) * np.random.randn(*acc_samples.shape)
            H_pyDxy[:, i] = my_stats.entropy_estimate(acc_samples)
        EH_pyDxy = np.mean(H_pyDxy, axis=1)

        # Final acquisition is the difference between the two above
        f_acq = H_pyDx - EH_pyDxy
        return f_acq

    def _optimize_acq(self):
        """
        Overwrite the optimization method from AcquisitionBase as the following
        implementation is based on grid evaluations.
        """
        xs, grid_shape = misc.create_flattened_meshgrid(
            self.domain.lb, self.domain.ub, self.dx_grid, self.gp.kernel.input_dim)
        f_acq = self._f_acq(xs)
        self.f_acq_eval = np.reshape(f_acq, grid_shape)
        x_opt = xs[np.argmax(f_acq)]
        return x_opt

    def _sample_robust_max_values(self, gp):
        n_samples = 50
        gp_samples_filtered = misc.generate_filtered_samples_on_grid(
            gp, self.domain.lb, self.domain.ub, self.dx_grid, n_samples,
            self.filter_width, filter_type=self.filter_type)
        robust_max_values = np.max(gp_samples_filtered.reshape([n_samples, -1]), axis=1)

        # For improved efficiency of the rejection sampling, we only consider
        # max-values between the 50th and 95th percentile.
        percentiles = np.linspace(50, 95, self.n_max_value_samples)
        reduced_maxes = np.percentile(robust_max_values, percentiles)
        return reduced_maxes


class NESExpProp(NESBase):
    """
    Class that implements NES based on Expectation Propagation.
    """
    def __init__(self, domain, filter_width, n_max_value_samples, filter_type='gauss', opt_grid=False):
        NESBase.__init__(self, domain, filter_width, n_max_value_samples, filter_type)
        self.__name__ = 'nes_ep'

        self.ngp = None
        self.opt_grid = opt_grid
        self.mu1_list, self.cov1_list = [], []  # Place holder for EP solution
        if opt_grid and domain.lb.shape[0] > 1:
            raise ValueError("Grid optimization is only possible for 1D")

    def next_point(self, gp):
        self.gp = gp
        self.ngp = gp_module.NoisyInputGP.from_gp(self.gp, self.filter_width**2)
        self.robust_max_values = self._sample_robust_max_values(grid=False)

        self.mu1_list, self.cov1_list = [], []
        for max_val in self.robust_max_values:
            mu1, cov1 = ep_approx(self.ngp, max_val)
            self.mu1_list.append(mu1)
            self.cov1_list.append(cov1)

        if self.opt_grid:
            print("Warning: You are optimizing the acquisition function on a grid!")
            xs = np.linspace(self.domain.lb[0], self.domain.ub[0], 1000)[:, None]
            f_acq = self._f_acq(xs)
            x_next = xs[np.argmax(f_acq)]
        else:
            x_next = self._optimize_acq()
        return x_next

    def _f_acq(self, x):
        # Calculate H[ p( y | D ) ]
        x = np.atleast_2d(x)
        _, var = self.gp.predict(x)
        H_pyDx = 0.5 * np.log(var + self.gp.noise_var)

        # Calculate E[ H[ p( y | D, g* ) ] ]
        H_pyDxy = np.empty((x.shape[0], self.robust_max_values.shape[0]))
        for i, max_val in enumerate(self.robust_max_values):

            if x.shape[0] > 1:
                print("Evaluating for {} points: # max_val {} / {}".format(
                    x.shape[0], i, len(self.robust_max_values)))
            nm3, nv3 = trunc_gauss_approx_g(x, self.ngp, max_val,
                                            self.mu1_list[i], self.cov1_list[i])
            m5, v5 = predictive_cond_dist(x, self.ngp, max_val, nm3, nv3)
            H_pyDxy[:, i] = 0.5 * np.log(v5.squeeze() + self.gp.noise_var)
        EH_pyDxy = np.mean(H_pyDxy, axis=1)

        # Final acquisition is the difference between the two above
        f_acq = H_pyDx - EH_pyDxy
        return f_acq.squeeze()

    def _sample_robust_max_values(self, grid=False):

        ssgp = gp_module.SparseSpectrumGP.from_gp(self.gp, 1000)
        ngp = gp_module.NoisyInputGP.from_gp(self.gp, self.filter_width**2)

        if grid:
            print("Warning: You are sampling the robust max-values on a sobol grid!")
            x_sobol = sobol_seq.i4_sobol_generate(self.gp.input_dim, 1000)
            samples = ngp.sample_posterior(x_sobol, 100)
            robust_max_values = np.max(samples, axis=0)
        else:
            robust_max_values = []
            n_samples = 50
            for _ in range(n_samples):
                _, g_handle = ssgp.sample_posterior_handle_with_filtered_counterpart(1, self.filter_width**2)
                _, rob_max_val = misc.maximize_function(g_handle, self.domain, n_restarts=10)
                robust_max_values.append(rob_max_val)
            robust_max_values = np.asarray(robust_max_values)

        percentiles = np.linspace(50, 95, self.n_max_value_samples)
        # percentiles = np.linspace(25, 75, self.n_max_value_samples)
        reduced_maxes = np.percentile(robust_max_values, percentiles)

        return reduced_maxes


def ep_approx(ngp, max_val):
    X = ngp.X
    n_data = X.shape[0]

    # Step 1: p( g | y, g*) \propto p( g | y ) \prod Indicator(g* > g_i)
    mu0, cov0 = ngp.predict(X, full_variance=True)  # p( g | y )
    mu1, cov1 = my_stats.expectation_propagation_trunc_gauss(mu0, cov0, ub=max_val * np.ones(n_data))
    return mu1, cov1


def trunc_gauss_approx_g(x, ngp, max_val, mu1, cov1):
    k_f = ngp.kernel.K
    k_fg = ngp.kernel_fg.K
    k_g = ngp.kernel_g.K
    X = ngp.X
    Y = gp_module.deshift_y(ngp.Y, ngp.y_mean) if ngp.normalize_Y else ngp.Y
    n_data = X.shape[0]

    # Step 1: p( g | y, g*) \propto p( g | y ) \prod Indicator(g* > g_i)
    # This is pre-computed and given via mu1, cov1

    # Step 2: p0( g(x) | y, g*) = \int p( g(x) | g, y ) p( g | y, g*) dg
    # Pre-calculate bunch of kernel matrices
    kfg_XX = k_fg(X)
    kf_XX_noisy = k_f(X) + ngp.noise_var * np.eye(n_data)
    K = np.vstack((np.hstack((k_g(X), kfg_XX)), np.hstack((kfg_XX, kf_XX_noisy))))
    k_xX = np.hstack((k_g(x, X), k_fg(X, x).T))

    # p( g(x) | g, y ) = N( g(x) | mu_g, cov_g )
    K_chol = misc.compute_stable_cholesky(K)
    s = linalg.cho_solve(K_chol, k_xX.T)

    covg = k_g(x) - s.T @ k_xX.T
    a1 = s.T[:, :n_data]
    a2 = s.T[:, n_data:]

    # p0(g(x) | y, g *) = N( g(x) | mu2, v2 )
    mu2 = a1 @ mu1[:, None] + a2 @ Y
    cov2 = covg + a1 @ cov1 @ a1.T

    m2 = mu2.squeeze()
    v2 = np.diag(cov2)

    # Step 3: p( g(x) | y, g*) = N( g(x) | mu3, v3 )
    alpha = (max_val - m2) / np.sqrt(v2)
    r = stats.norm.pdf(alpha) / stats.norm.cdf(alpha)
    m3 = m2 - np.sqrt(v2 * r)
    v3 = v2 - v2 * r * (r + alpha)

    return m3, v3


def predictive_cond_dist(x, ngp, g_star, nm3, nv3):
    n_eval = x.shape[0]
    X = ngp.X
    Y = gp_module.deshift_y(ngp.Y, ngp.y_mean) if ngp.normalize_Y else ngp.Y
    m5, v5 = np.empty((n_eval,)), np.empty((n_eval,))
    k_f = ngp.kernel.K
    k_g = ngp.kernel_g.K
    k_fg = ngp.kernel_fg.K
    for i, xi in enumerate(x):
        xi = np.atleast_2d(xi)
        kf_Xx = k_f(X, xi)
        kgf_xx = k_fg(xi)
        kz = np.vstack((kf_Xx, kgf_xx))

        kg_xx = k_g(xi)
        kf_XX_noise = k_f(X) + ngp.noise_var * np.eye(X.shape[0])
        kfg_Xx = k_fg(X, xi)

        tmp1 = np.hstack((kf_XX_noise, kfg_Xx))
        tmp2 = np.hstack((kfg_Xx.T, kg_xx))
        Kz = np.vstack((tmp1, tmp2))

        Kz_chol = misc.compute_stable_cholesky(Kz)
        s = linalg.cho_solve(Kz_chol, kz)  # s = inv(Kz) @ kz
        cov4 = k_f(xi) - s.T @ kz
        a1 = s.T[:, :-1]
        a2 = s.T[:, -1:]

        m5[i] = a1 @ Y + a2 * nm3[i]
        v5[i] = cov4 + a2**2 * nv3[i]

    return m5, v5
