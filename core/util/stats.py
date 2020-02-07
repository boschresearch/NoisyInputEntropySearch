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

from core.util import misc


def expectation_propagation_trunc_gauss(mean, cov, lb=None, ub=None, n_max_sweeps=50, abs_tol=1e-6):
    """
    Implements the Gaussian EP algorithm applied to a multivariate
    truncated Gaussian distribution of the following form:

    .. math::
        p(x) = \mathcal{N}(x | \mu, \Sigma) \prod_{i = 1}^{n} I_{lb_i < x_i < ub_i}

    Note: This implementation follows Algorithm 3 in this paper:
          https://www.microsoft.com/en-us/research/wp-content/uploads/2005/07/EP.pdf

    :param mean: Mean vector of the (non-truncated) Gaussian distribution.
    :param cov: Covariance of the (non-truncated) Gaussian distribution.
    :param lb: Lower bound for the truncation.
    :param ub: Upper bound for the truncation.
    :param n_max_sweeps: Maximum number of sweep over all factors.
    :param abs_tol: Tolerance below which a value is assumed to be converged.
    :return: Mean and covariance of an approximated Gaussian distribution.
    """
    def v(t, l, u):
        """
        Helper function for EP.
        """
        numerator = stats.norm.pdf(l - t) - stats.norm.pdf(u - t)
        denominator = stats.norm.cdf(u - t) - stats.norm.cdf(l - t)
        return numerator / denominator

    def w(t, l, u):
        """
        Helper function for EP.
        """
        numerator = (u - t) * stats.norm.pdf(u - t) - (l - t) * stats.norm.pdf(l - t)
        denominator = stats.norm.cdf(u - t) - stats.norm.cdf(l - t)
        return v(t, l, u) ** 2 + numerator / denominator

    # Dimension of the random variable
    mean = mean.squeeze()
    dim = mean.shape[0]

    # If no bound are defined, set them very large/small such that they become ineffective
    if lb is None and ub is None:
        return mean.copy(), cov.copy()
    elif lb is None:
        lb = -1e6 * np.ones(dim)
    elif ub is None:
        ub = 1e6 * np.ones(dim)

    # For numerical stability if bounds are very small / large
    jitter = 1e-10

    # Initialize approximating factors
    mu_n = np.zeros((dim,))
    pi_n = np.zeros((dim,))
    s_n = np.ones((dim,))

    # Initialize mean and variance of approximating Gaussian (follows from factor initialization)
    mean_hat = mean.copy()
    cov_hat = cov.copy()

    # Pick an index and perform updates
    for i_sweep in range(n_max_sweeps):
        # Check for convergence after each sweep over all factors
        mu_n_old = mu_n.copy()
        pi_n_old = pi_n.copy()
        s_n_old = s_n.copy()

        for j in range(dim):
            # Pre-computations
            t_j = cov_hat[:, j]
            d_j = pi_n[j] * cov_hat[j, j]
            e_j = 1 / (1 - d_j)

            phi_j = mean_hat[j] + d_j * e_j * (mean_hat[j] - mu_n[j])
            psi_j = cov_hat[j, j] * e_j

            phi_prime_j = phi_j / np.sqrt(psi_j)
            lb_prime_j = lb[j] / np.sqrt(psi_j)
            ub_prime_j = ub[j] / np.sqrt(psi_j)

            alpha_j = v(phi_prime_j, lb_prime_j, ub_prime_j) / np.sqrt(psi_j)
            beta_j = w(phi_prime_j, lb_prime_j, ub_prime_j) / psi_j

            # ADF update
            mean_hat += e_j * (pi_n[j] * (mean_hat[j] - mu_n[j]) + alpha_j) * t_j
            cov_hat += (pi_n[j] * e_j - e_j ** 2 * beta_j) * np.outer(t_j, t_j)

            # Factor update
            pi_n[j] = beta_j / (1 - beta_j * psi_j)
            mu_n[j] = alpha_j / (beta_j + jitter) + phi_j

            tmp1 = stats.norm.cdf(ub_prime_j - phi_prime_j) - stats.norm.cdf(lb_prime_j - phi_prime_j)
            tmp2 = np.exp(alpha_j ** 2 / (2 * beta_j + jitter)) / np.sqrt(1 - psi_j * beta_j)
            s_n[j] = tmp1 * tmp2

        # Calculate differences of factors before sweep
        mu_n_diff = np.max(np.abs(mu_n - mu_n_old))
        pi_n_diff = np.max(np.abs(pi_n - pi_n_old))
        s_n_diff = np.max(np.abs(s_n - s_n_old))

        if (np.array([mu_n_diff, pi_n_diff, s_n_diff]) <= abs_tol).all():
            # print("EP converged after {} iterations".format(i_sweep + 1))
            break

    return mean_hat, cov_hat


def robust_max_value_rejection_sampling_global(gp, x_min, x_max, dx, max_value, n_accept, filter_width, filter_type='gauss', batch_size=1000):
    """
    Perform global rejection sampling based on the robust max-value.

    Note: due to boundary effects from the convolution with a filter, the bounds
          are increased for the purpose of filtering. However, the rejection
          criterion and the final samples are based on the original domain.

    :param gp: GP from which samples are drawn.
    :param x_min: lower bound on domain (int or nd.array).
    :param x_max: upper bound on domain (int or nd.array).
    :param dx: grid resolution for domain.
    :param max_value: max-value for rejection criterion.
    :param n_accept: need at least n_accept accepted samples.
    :param filter_type: type of filter, either 'gauss' or 'uniform'.
    :param batch_size: number of samples per iteration.
    :return: Accepted samples.
    """
    n_samples = 0  # Counter for number of samples
    accepted_samples = []
    dim = gp.kernel.input_dim

    if dim == 1 and type(x_min) is not np.ndarray:
        x_min = np.array([x_min])
        x_max = np.array([x_max])
    elif dim > 1 and type(x_min) is not np.ndarray:
        x_min = x_min * np.ones((dim,))
        x_max = x_max * np.ones((dim,))

    if filter_type == 'gauss':
        x_increase = 5 * filter_width
    elif filter_type == 'uniform':
        x_increase = 1 * filter_width
    else:
        raise NotImplementedError

    x2_min = x_min - x_increase
    x2_max = x_max + x_increase
    x_meshes = misc.create_meshgrid(x_min, x_max, dx, dim)
    x2_meshes = misc.create_meshgrid(x2_min, x2_max, dx, dim)

    mesh2_shape = np.array(x2_meshes[0].shape)
    mesh_shape = np.array(x_meshes[0].shape)
    dn2 = (0.5 * (mesh2_shape - mesh_shape)).astype(int)
    ranges = tuple([slice(dn2[i], dn2[i] + mesh_shape[i]) for i in range(dim)])

    xs = np.vstack([x.flatten() for x in reversed(x_meshes)]).T
    xs2 = np.vstack([x.flatten() for x in reversed(x2_meshes)]).T

    # Perform sampling as long as we do not have enough accepted samples
    while n_samples < n_accept:
        # Sample and filter on large domain
        gp2_samples = gp.sample_posterior(xs2, batch_size)
        gp2_samples = np.reshape(gp2_samples, [*mesh2_shape, -1])

        gp2_samples_filtered = np.array(
            [misc.filter_signal(gp2_samples[..., i], dx, filter_width) for i in range(batch_size)])
        gp_samples_filtered = np.array(
            [gp2_samples_filtered[i, ...][ranges] for i in range(batch_size)])

        gp_samples = np.concatenate(
            [gp2_samples[..., i][ranges][..., None] for i in range(batch_size)], axis=-1)

        # Check for rejection criterion
        robust_maxes = np.max(gp_samples_filtered.reshape([batch_size, -1]), axis=1)
        keep_indices = robust_maxes <= max_value
        accepted_samples.append(gp_samples[..., keep_indices])
        n_samples += np.sum(keep_indices)

        if np.sum(keep_indices) / batch_size <= 0.01:
            print("Caution: ratio of accepted samples is below 1% ({}/{})".format(
                np.sum(keep_indices), batch_size))

    accepted_samples = np.concatenate(accepted_samples, axis=-1)
    return accepted_samples[..., :n_accept]


def entropy_estimate(samples):
    """
    Calculate the sample estimate for the entropy.

    :param samples: Samples on grid.
    :return: Entropy estimate on the same grid as the samples.
    """
    dim = samples.ndim - 1  # Assume samples to be on grid, e.g., 2d: n x n x n_samples

    data_std = np.std(samples, axis=-1).flatten()
    n_samples = samples.shape[-1]
    a_n = np.power(n_samples, -1. / (dim + 4))  # Scott's factor

    # Pre-calculate inverses for improved performance
    data_std_inv = 1 / data_std
    n_samples_inv = 1 / n_samples
    a_n_inv = 1 / a_n
    sqrt_2pi_inv = 1 / (np.sqrt(2 * np.pi))

    samples_flat = np.reshape(samples, [-1, n_samples])
    entropy = np.empty((samples_flat.shape[0],))
    for i, s in enumerate(samples_flat):
        y = s[:, None] * (data_std_inv[i] * a_n_inv)
        ysq = np.square(y)
        r2 = -2.*np.outer(y, y) + (ysq + ysq.T)
        k = np.exp(-0.5 * r2) * (sqrt_2pi_inv * data_std_inv[i])
        k_log_sum = np.log(np.sum(k, axis=1) * (n_samples_inv * a_n_inv))
        entropy[i] = - np.sum(k_log_sum) * n_samples_inv

    entropy = np.reshape(entropy, samples.shape[:-1])
    return entropy


def calc_sigma_points_and_weights(x, k_factor, input_var, domain):
    # Get dimensionality from input data
    x = np.atleast_2d(x)
    dim = x.shape[1]

    # Calculate sigma weights
    w0 = [k_factor / (dim + k_factor)]
    wi = [0.5 / (dim + k_factor)] * dim
    w_sig = wi + w0 + wi

    # Calculate sigma points
    x_sig = [x.copy() for _ in range(2*dim + 1)]
    for i in range(dim):
        ei = np.zeros(x.shape)
        ei[:, i] = np.sqrt((dim + k_factor) * input_var)
        x_sig[i] += ei
        x_sig[len(x_sig) - 1 - i] -= ei

    x_sig = [misc.project_in_domain(xi_sig, domain) for xi_sig in x_sig]
    return x_sig, w_sig
