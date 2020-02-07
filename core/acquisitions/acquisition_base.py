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
from scipy.optimize import minimize
import sobol_seq

from core.util import misc


class AcquisitionBase:
    def __init__(self, domain, n_restarts=10):
        self.domain = domain
        self.n_restarts = n_restarts
        self.gp = None

    def next_point(self, gp):
        raise NotImplementedError

    def _optimize_acq(self):
        dim = self.gp.kernel.input_dim
        x0_candidates = self.domain.lb + (self.domain.ub - self.domain.lb) * \
                        sobol_seq.i4_sobol_generate(dim, self.n_restarts) + \
                        np.random.randn(self.n_restarts, dim)
        x_opt_candidates = np.empty((self.n_restarts, dim))
        f_opt = np.empty((self.n_restarts,))
        for i, x0 in enumerate(x0_candidates):
            res = minimize(fun=misc.neg(self._f_acq), x0=x0, bounds=self.domain)
            x_opt_candidates[i] = res['x']
            f_opt[i] = -1 * res['fun']

        x_opt = x_opt_candidates[np.argmax(f_opt)]
        return x_opt

    def _optimize_acq_grid(self, n_grid):
        dim = self.gp.kernel.input_dim
        x0 = self.domain.lb + (self.domain.ub - self.domain.lb) * \
                        sobol_seq.i4_sobol_generate(dim, n_grid)
        f0 = self._f_acq(x0)
        return x0[np.argmax(f0)]

    def _f_acq(self, x):
        raise NotImplementedError
