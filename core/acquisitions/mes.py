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
from scipy.stats import norm
import sobol_seq
from tqdm import tqdm

from .acquisition_base import AcquisitionBase
from core.util import misc
from core.util import gp as gp_module


class MES(AcquisitionBase):
    def __init__(self, domain, n_samples):
        super(MES, self).__init__(domain=domain)
        self.n_samples = n_samples
        self.y_maxes = None

    def next_point(self, gp):
        self.gp = gp
        self.y_maxes = self._sample_maxes()
        x_next = self._optimize_acq()
        return x_next

    def _f_acq(self, x):
        x = np.atleast_2d(x)
        mu, var = self.gp.predict(x)
        std = np.sqrt(var)
        gamma_maxes = (self.y_maxes - mu) / std[:, None]
        tmp = 0.5 * gamma_maxes * norm.pdf(gamma_maxes) / norm.cdf(gamma_maxes) - \
            np.log(norm.cdf(gamma_maxes))
        mes = np.mean(tmp, axis=1, keepdims=True)

        return mes

    def _sample_maxes(self):
        raise NotImplementedError("Specify sampling method for MES.")


class MESNaive(MES):
    def __init__(self, domain, n_samples):
        super(MESNaive, self).__init__(domain=domain, n_samples=n_samples)
        self.__name__ = "mes_naive"

    def _sample_maxes(self, grid=True):
        if grid:
            x_sobol = sobol_seq.i4_sobol_generate(self.gp.input_dim, 1000)
            samples = self.gp.sample_posterior(x_sobol, 100)
            max_values = np.max(samples, axis=0)
        else:
            raise ValueError

        percentiles = np.linspace(50, 95, self.n_samples)
        reduced_maxes = np.percentile(max_values, percentiles)

        return reduced_maxes


class MESGumbel(MES):
    def __init__(self, domain, n_samples=1000):
        super(MESGumbel, self).__init__(domain=domain, n_samples=n_samples)
        self.__name__ = "mes_gumbel"

    def _sample_maxes(self):
        dim = self.domain.lb.shape[0]
        x_grid = sobol_seq.i4_sobol_generate(dim, 100)
        mu, var = self.gp.predict(x_grid)
        std = np.sqrt(var)

        def cdf_approx(z):
            z = np.atleast_1d(z)
            ret_val = np.zeros(z.shape)
            for i, zi in enumerate(z):
                ret_val[i] = np.prod(norm.cdf((zi - mu) / std))
            return ret_val

        lower = np.max(self.gp.Y)
        upper = np.max(mu + 5*std)
        if cdf_approx(upper) <= 0.75:
            upper += 1

        grid = np.linspace(lower, upper, 100)

        cdf_grid = cdf_approx(grid)
        r1, r2 = 0.25, 0.75

        y1 = grid[np.argmax(cdf_grid >= r1)]
        y2 = grid[np.argmax(cdf_grid >= r2)]

        b = (y1 - y2) / (np.log(-np.log(r2)) - np.log(-np.log(r1)))
        a = y1 + (b * np.log(-np.log(r1)))

        maxes = a - b*np.log(-np.log(np.random.rand(self.n_samples,)))
        return maxes
