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
from scipy.integrate import ode


class Planet(object):
    def __init__(self, pos, mass, name):
        self.pos = pos
        self.mass = mass
        self.name = name


class WorldModel(object):
    def __init__(self):
        self.home_planet = Planet(np.array([0.0, 0.0]), 1.0, 'A')
        self.target_planet = Planet(np.array([10.0, 0.0]), 10.0, 'D')
        self.planets = [self.home_planet, self.target_planet]

        # Planets for sling-shot maneuver
        self.planets.append(Planet(np.array([4.0, 0.2]), 2.0, 'B'))
        self.planets.append(Planet(np.array([5.0, -1.5]), 8.0, 'C'))

        self.G = 1.0

    def ode_rhs(self, _, x):
        pos = x[:2]
        vel = x[2:]

        forces = []
        for planet in self.planets:
            r = planet.pos - pos
            d = np.linalg.norm(r)
            f = self.G * planet.mass * r / d ** 3
            forces.append(f)

        dpos = vel
        dvel = sum(forces)

        dx = np.hstack((dpos, dvel))
        return dx


class WorldSimulation(object):
    def __init__(self, ode_rhs):
        self.solver = ode(ode_rhs)
        self.solver.set_integrator('vode')

    def run(self, x0, dt, n_steps):
        self.solver.set_initial_value(x0, 0)

        t = np.zeros(n_steps+1)
        x = np.zeros((n_steps+1, 4))
        x[0, :] = x0
        i = 0
        while self.solver.successful() and self.solver.t < dt * n_steps:
            self.solver.integrate(self.solver.t + dt)
            t[i] = self.solver.t
            x[i] = self.solver.y
            i += 1

        return x, t


if __name__ == '__main__':
    from core.util.objectives import rocket_simulation

    # Bounds for input parameters
    alpha_bounds = np.deg2rad([-10, 45])
    speed_bounds = [4.1, 5.0]

    # Input variance used for the experiment
    input_var = np.array([np.deg2rad(3.0), 0.05]) ** 2
    
    # Run simulation
    f = rocket_simulation(np.array([[alpha_bounds[0], speed_bounds[0]]]))
    print(f)
    


