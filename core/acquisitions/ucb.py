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
from .acquisition_base import AcquisitionBase


class UCB(AcquisitionBase):
    def __init__(self, domain, n_restarts=10, exploration_factor=2.0):
        super(UCB, self).__init__(domain=domain, n_restarts=n_restarts)
        self.exploration_factor = exploration_factor
        self.__name__ = 'ucb'

    def next_point(self, gp):
        self.gp = gp
        x_next = self._optimize_acq()
        return x_next

    def _f_acq(self, x):
        x = np.atleast_2d(x)
        mu, var = self.gp.predict(x)
        std = np.sqrt(var)[:, None]
        return mu + self.exploration_factor * std
