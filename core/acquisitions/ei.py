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
from core.acquisitions.acquisition_base import AcquisitionBase
from core.util.misc import optimize_gp_2
from scipy.stats import norm


class EI(AcquisitionBase):
    def __init__(self, domain, n_restarts=10):
        super(EI, self).__init__(domain=domain, n_restarts=n_restarts)
        self.f_max = None  # Maximum of GP mean
        self.__name__ = 'ei'

    def next_point(self, gp):
        self.gp = gp
        _, self.f_max = optimize_gp_2(gp, self.domain)
        x_next = self._optimize_acq()
        return x_next

    def _f_acq(self, x):
        x = np.atleast_2d(x)
        mu, var = self.gp.predict(x)
        std = np.sqrt(var)[:, None]
        gamma = (self.f_max - mu) / std
        return std * ((norm.cdf(gamma) - 1) * gamma + norm.pdf(gamma))
