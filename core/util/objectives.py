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
import rocket_sim

def synthetic_1d_01(x, noise_var=0.0):
    f = np.sin(5 * np.pi * x**2) + 0.5*x
    return f + np.sqrt(noise_var) * np.random.randn(*f.shape)


def rkhs_synth(x, noise_var=0.0):
    """
    from: https://github.com/iassael/bo-benchmark-rkhs/blob/master/rkhs.py
    """
    raise ValueError("Unfortunately, the authors decided to publish their cost function under GNU GPL.")


def gmm_2d(x, noise_var=0.0):
    x = np.atleast_2d(x)
    gmm_pos = np.array([[0.2, 0.2],
                        [0.8, 0.2],
                        [0.5, 0.7]])
    gmm_var = np.array([0.20, 0.10, 0.10])**2
    gmm_norm = 2 * np.pi * gmm_var * np.array([0.5, 0.7, 0.7])
    gaussians = [stats.multivariate_normal(mean=gmm_pos[i], cov=gmm_var[i]) for i in range(gmm_var.shape[0])]
    f = [gmm_norm[i] * g.pdf(x) for i, g in enumerate(gaussians)]
    f = np.atleast_1d(np.sum(np.asarray(f), axis=0))[:, None]

    return f + np.sqrt(noise_var) * np.random.randn(*f.shape)


def synth_poly_2d(x, noise_var=0.0):
    x = np.atleast_2d(x)
    x1 = np.minimum(np.maximum(x[:, 0], -0.95), 3.2)
    x2 = np.minimum(np.maximum(x[:, 1], -0.45), 4.4)
    x1_terms = -2*x1**6 + 12.2*x1**5 - 21.2*x1**4 + 06.4*x1**3 + 04.7*x1**2 - 06.2*x1
    x2_terms = -1*x2**6 + 11.0*x2**5 - 43.3*x2**4 + 74.8*x2**3 - 56.9*x2**2 + 10.0*x2
    x12_terms = 4.1*x1*x2 + 0.1*x1**2*x2**2 - 0.4*x1*x2**2 - 0.4*x1**2*x2
    f = x1_terms + x2_terms + x12_terms
    f = f[:, None]

    return f + np.sqrt(noise_var) * np.random.randn(*f.shape)


def synth_poly_2d_norm(x, noise_var=0.0):
    f = synth_poly_2d(x, noise_var)
    f = (f + 14.97) / 11.84
    return f + np.sqrt(noise_var) * np.random.randn(*f.shape)
    

def hartmann_3d(x, noise_var=0.0):
    x = np.atleast_2d(x)

    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
    P = np.array([[3689, 1170, 2673],
                  [4699, 4387, 7470],
                  [1091, 8732, 5547],
                  [ 381, 5743, 8828]]) * 1e-4
    f = np.zeros((x.shape[0], 1))
    for i in range(4):
        tmp = np.zeros(f.shape)
        for j in range(3):
            tmp += (A[i, j]*(x[:, j] - P[i, j])**2)[:, None]
        f += alpha[i] * np.exp(-tmp)
    return f + np.sqrt(noise_var) * np.random.randn(*f.shape)


def rocket_simulation(x, noise_var=0.0):
    x = np.atleast_2d(x)
    f = np.empty((x.shape[0], 1))

    world = rocket_sim.WorldModel()
    sim = rocket_sim.WorldSimulation(world.ode_rhs)

    # Set up the simulation and parameters
    dt = 0.001
    n_steps = 4000

    # Run simulation for each configuration
    for i, xi in enumerate(x):
        # Notational convenience for initial angle and initial speed
        a0 = xi[0]
        s0 = xi[1]

        # Construct initial state from configuration
        v0 = s0 * np.array([np.cos(a0), np.sin(a0)])
        p0 = np.array([0.1, 0.0])
        x0 = np.hstack((p0, v0))

        # Run simulation
        x, t = sim.run(x0, dt, n_steps)

        # Evaluate trajectory
        d = np.linalg.norm(x[:, :2] - world.target_planet.pos, axis=1)
        idx_min = np.argmin(d)
        closest_point = d[idx_min]
        f[i] = closest_point + 2 * s0

    f = np.log10(f)  # log-transformation makes value range nicer
    return -f + np.sqrt(noise_var)*np.random.randn(*f.shape)
