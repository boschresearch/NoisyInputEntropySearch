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
from scipy import linalg
from scipy import stats

from core.util import misc


class GP:
    def __init__(self, kernel, X, Y, noise_var, normalize_Y=False):
        self.kernel = kernel
        self.input_dim = kernel.input_dim
        self.X = X
        self.Y = Y
        self.noise_var = noise_var

        self.normalize_Y = normalize_Y
        if self.normalize_Y:
            self.Y, self.y_mean = shift_y(Y)

        K = self.kernel.K(X) + self.noise_var * np.eye(X.shape[0])
        self.chol = misc.compute_stable_cholesky(K)
        self.alpha = linalg.cho_solve(self.chol, self.Y)

    def set_xy(self, X, Y):
        self.X = X
        self.Y = Y

        if self.normalize_Y:
            self.Y, self.y_mean = shift_y(Y)

        K = self.kernel.K(X) + self.noise_var * np.eye(X.shape[0])
        self.chol = misc.compute_stable_cholesky(K)
        self.alpha = linalg.cho_solve(self.chol, self.Y)

    def predict(self, Xs, full_variance=False):
        ks = self.kernel.K(self.X, Xs)
        kss = self.kernel.K(Xs)

        mu = ks.T @ self.alpha
        mu = deshift_y(mu, self.y_mean) if self.normalize_Y else mu

        try:
            V = linalg.solve_triangular(np.tril(self.chol[0]), ks, lower=True)
        except ValueError as err:
            eigs = np.linalg.eigvals(self.chol[0])
            np.save('crash_input_data', self.X)
            np.save('crash_input_eval', Xs)
            print("Minimum eigen value is: {}".format(np.min(eigs)))
            print("Cholesky contains nan: {}".format(np.isnan(np.tril(self.chol[0])).any()))
            print("Cholesky contains inf: {}".format(np.isinf(np.tril(self.chol[0])).any()))
            print("ks contains nan: {}".format(np.isnan(ks).any()))
            print("ks contains inf: {}".format(np.isinf(ks).any()))

            raise ValueError(err)
        var_full = kss - V.T @ V

        if full_variance:
            var_full = (var_full + var_full.T) / 2
            return mu, var_full
        else:
            var = np.diag(var_full)
            return mu, var

    def sample_posterior(self, Xs, n):
        mu, var_full = self.predict(Xs, full_variance=True)
        s = np.random.multivariate_normal(mean=mu.squeeze(), cov=var_full, size=n).T
        return s


class NoisyInputGP(GP):
    def __init__(self, kernel_f, kernel_fg, kernel_g, X, Y, output_noise_var, normalize_Y=False):
        super(NoisyInputGP, self).__init__(kernel=kernel_f, X=X, Y=Y, noise_var=output_noise_var, normalize_Y=normalize_Y)
        self.kernel_fg = kernel_fg
        self.kernel_g = kernel_g

    def predict(self, Xs, full_variance=False):
        ks = self.kernel_fg.K(self.X, Xs)
        kss = self.kernel_g.K(Xs)

        mu = ks.T @ self.alpha
        mu = deshift_y(mu, self.y_mean) if self.normalize_Y else mu

        V = linalg.solve_triangular(np.tril(self.chol[0]), ks, lower=True)
        var_full = kss - V.T @ V

        if full_variance:
            var_full = (var_full + var_full.T) / 2
            return mu, var_full
        else:
            var = np.diag(var_full)
            return mu, var

    @staticmethod
    def from_gp(gp, input_var):
        """Construct a NoisyInputGP from a GP"""
        k_g, k_fg = create_noisy_input_rbf_kernel(gp.kernel, input_var)
        Y = deshift_y(gp.Y, gp.y_mean) if gp.normalize_Y else gp.Y
        return NoisyInputGP(gp.kernel, k_fg, k_g, gp.X, Y, gp.noise_var, normalize_Y=gp.normalize_Y)


class SparseSpectrumGP:
    def __init__(self, kernel, X, Y, noise_var, n_features, normalize_Y=False):
        self.input_dim = kernel.input_dim
        self.lengthscale = kernel.lengthscale
        self.signal_var = kernel.variance
        self.noise_var = noise_var
        self.X = X
        self.Y = Y if Y.ndim == 2 else np.atleast_2d(Y).T

        self.normalize_Y = normalize_Y
        if self.normalize_Y:
            self.Y, self.y_mean = shift_y(Y)

        self.n_features = n_features
        self.w = None
        self.b = None
        self.phi = self._compute_phi()

        phi_train = self.phi(X)
        A = phi_train @ phi_train.T + self.noise_var * np.eye(self.n_features)
        chol_A = misc.compute_stable_cholesky(A)

        B = np.eye(self.X.shape[0]) + phi_train.T @ phi_train / self.noise_var
        chol_B = misc.compute_stable_cholesky(B)
        v = linalg.cho_solve(chol_B, phi_train.T)

        a_inv = (np.eye(self.n_features) - phi_train @ v / self.noise_var)

        self.theta_mu = linalg.cho_solve(chol_A, phi_train @ self.Y)
        self.theta_var = a_inv

    def predict(self, Xs, full_variance=False):
        phi_x = self.phi(Xs)

        mu = phi_x.T @ self.theta_mu
        mu = deshift_y(mu, self.y_mean) if self.normalize_Y else mu

        var_full = phi_x.T @ self.theta_var @ phi_x

        if full_variance:
            var_full = 0.5 * (var_full + var_full.T)
            return mu, var_full
        else:
            var = np.diag(var_full)
            return mu, var

    def sample_posterior(self, Xs, n_samples):
        fs_h = self.sample_posterior_handle(n_samples)
        return fs_h(Xs)

    def sample_posterior_handle(self, n_samples):
        """
        Generate handle to n_samples function samples that can be evaluated at x.
        """
        chol = misc.compute_stable_cholesky(self.theta_var)[0]
        theta_samples = self.theta_mu + chol @ np.random.randn(self.n_features, n_samples)

        def handle_to_function_samples(x):
            if x.ndim == 1 and self.input_dim == 1:
                x = np.atleast_2d(x).T
            elif x.ndim == 1 and self.input_dim > 1:
                x = np.atleast_2d(x)

            h = self.phi(x).T @ theta_samples
            return deshift_y(h, self.y_mean) if self.normalize_Y else h

        return handle_to_function_samples

    def sample_posterior_weights(self, n_samples):
        theta_samples = stats.multivariate_normal.rvs(mean=self.theta_mu.squeeze(),
                                                      cov=self.theta_var,
                                                      size=n_samples).T
        return theta_samples

    def sample_posterior_handle_with_filtered_counterpart(self, n_samples, input_var):
        """
        Generate handle to n_samples function samples that can be evaluated at x.
        Additionally generate a handle to the Gaussian-filtered counterpart of this
        sample.
        """
        chol = misc.compute_stable_cholesky(self.theta_var)[0]
        theta_samples = self.theta_mu + chol @ np.random.randn(self.n_features, n_samples)

        def handle_to_function_samples(x):
            if x.ndim == 1 and self.input_dim == 1:
                x = np.atleast_2d(x).T
            elif x.ndim == 1 and self.input_dim > 1:
                x = np.atleast_2d(x)

            h = self.phi(x).T @ theta_samples
            return deshift_y(h, self.y_mean) if self.normalize_Y else h

        def handle_to_filtered_function_samples(x):
            if x.ndim == 1 and self.input_dim == 1:
                x = np.atleast_2d(x).T
            elif x.ndim == 1 and self.input_dim > 1:
                x = np.atleast_2d(x)

            phi_filt = self._compute_phi_filt(input_var)
            h = phi_filt(x).T @ theta_samples
            return deshift_y(h, self.y_mean) if self.normalize_Y else h

        return handle_to_function_samples, handle_to_filtered_function_samples

    def _compute_phi(self):
        """
        Compute random features.
        """
        lin_3sigma = np.linspace(stats.norm.cdf(-3), stats.norm.cdf(3), self.n_features * self.input_dim)
        lin_inv_cdf = stats.norm.ppf(lin_3sigma)
        self.w = np.random.permutation(lin_inv_cdf).reshape(self.n_features, self.input_dim) / self.lengthscale
        self.b = np.random.permutation(np.linspace(0, 2 * np.pi, self.n_features))[:, None]
        return lambda x: np.sqrt(2 * self.signal_var / self.n_features) * np.cos(self.w @ x.T + self.b)

    def _compute_phi_filt(self, input_var):
        """
        Compute the corresponding random features for filtered/smoothed samples.
        Note: Needs to be called after _compute_phi(.)
        """
        v_x = _check_input_var_dim(input_var, self.input_dim)
        filt_factor = np.exp(-0.5 * np.linalg.norm(self.w * np.sqrt(v_x), axis=1, keepdims=True) ** 2)
        return lambda x: self.phi(x) * filt_factor

    @staticmethod
    def from_gp(gp, n_ssgp_features):
        """Construct a SparseSpectrumGP from a GP"""
        Y = deshift_y(gp.Y, gp.y_mean) if gp.normalize_Y else gp.Y
        return SparseSpectrumGP(gp.kernel, gp.X, Y, gp.noise_var, n_ssgp_features, normalize_Y=gp.normalize_Y)


def _check_input_var_dim(input_var, dim):
    if isinstance(input_var, float):
        v_x = input_var * np.ones((dim,))
    elif isinstance(input_var, np.ndarray):
        assert input_var.ndim == 1
        assert input_var.shape[0] == dim
        v_x = input_var
    elif isinstance(input_var, list):
        assert len(input_var) == dim
        v_x = np.array(input_var)
    else:
        raise ValueError("Please specify input variance as either float scalar or np.array")
    return v_x


def shift_y(Y):
    y_mean = np.mean(Y)
    Y = Y - y_mean
    return Y, y_mean


def deshift_y(Y, y_mean):
    return Y + y_mean


def create_noisy_input_rbf_kernel(kernel, input_var):
    dim = kernel.input_dim
    l_f = kernel.lengthscale
    v_f = kernel.variance

    v_x = _check_input_var_dim(input_var, dim=dim)

    l_g = np.sqrt(l_f ** 2 + 2 * v_x)
    v_g = v_f * np.sqrt(np.prod(l_f ** 2) / np.prod(l_f ** 2 + 2 * v_x))
    k_g = GPy.kern.RBF(input_dim=dim, variance=v_g, lengthscale=l_g, ARD=True)

    l_fg = np.sqrt(l_f ** 2 + v_x)
    v_fg = v_f * np.sqrt(np.prod(l_f ** 2) / np.prod(l_f ** 2 + v_x))
    k_fg = GPy.kern.RBF(input_dim=dim, variance=v_fg, lengthscale=l_fg, ARD=True)

    return k_g, k_fg
