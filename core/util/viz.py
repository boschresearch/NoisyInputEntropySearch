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


def gp_pred_with_bounds(x_plot, mean, variance, color='C0', label=None, std_factor=1.0, lw=1.5):
    x_plot = x_plot.squeeze()
    mean = mean.squeeze()
    std = np.sqrt(variance).squeeze()

    plt.plot(x_plot, mean, color=color, label=label, lw=lw)
    plt.fill_between(x_plot, mean + std_factor*std, mean - std_factor*std, color=color, alpha=0.3)
