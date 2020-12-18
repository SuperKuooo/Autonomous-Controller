from enum import Enum
from numpy import array, append, dot
from math import cos, sin, tan, radians, degrees


# oversees the overall car control
class CarManager:
    def __init__(self, x, y, theta):
        """

        :param x: x coordinate
        :param y: y coordinate
        :param theta: relative to x axis in degrees
        """
        self.location = _CarLocation(x, y, theta)
        self.physics = _CarPhysics()
        self.velocity = 5
        self.turn_angle = 10
        self.look_ahead_dist = 5

    def set_velocity(self, velocity):
        self.velocity = velocity

    def set_turn_angle(self, angle):
        self.turn_angle = angle

    def get_states(self):
        car_points = self.location.get_points()
        center = self.location.get_centers()

        states = [center, car_points, ]
        return states

    def update(self, interval):
        self._update_location(interval)
        self._update_physics()

    def _update_location(self, interval):
        """

        :param interval: delta time in seconds
        :return:
        """
        dot_matrix = self.location.get_dot_matrix(self.velocity, self.turn_angle)
        d_x, d_y, d_theta = interval * dot_matrix  # meter, meter, radians
        delta = array([[d_x, d_y]]).transpose()

        self.location.set_center(delta, add=True)
        self.location.set_heading_angle(d_theta, add=True)

    def _update_physics(self):
        self.physics.update_physics()


# handles the car locations, transformation, and setups
class _CarLocation:
    def __init__(self, x, y, theta):
        self.center = array([[x, y]], dtype=float).transpose()

        # all points HERE are relative to the global frame
        self.car_points = array([0, 0])

        point1 = array([0.5, -0.5])
        point2 = array([-0.5, -0.5])
        point3 = array([-0.5, 0.5])
        point4 = array([0, 0.7])
        point5 = array([0.5, 0.5])
        point6 = array([0.5, -0.5])

        # car_points dimensions = [2, 6]
        self.car_points = append([self.car_points],
                                 [point1, point2, point3, point4, point5, point6],
                                 axis=0).transpose()

        # relative to global x axis
        self.heading_angle = theta

    def get_dot_matrix(self, velocity, wheel_angle):
        head_rad = radians(self.heading_angle)
        wheel_rad = radians(wheel_angle + CarConfig.wheel_deflection.value)

        x_dot = velocity * -sin(head_rad)
        y_dot = velocity * cos(head_rad)
        theta_dot = velocity / CarConfig.length.value * tan(wheel_rad)

        return array([x_dot, y_dot, theta_dot])

    def get_points(self):
        head_rad = radians(self.heading_angle)
        rotation_matrix = array([[cos(head_rad), -sin(head_rad)], [sin(head_rad), cos(head_rad)]])

        points = dot(rotation_matrix, self.car_points)
        points += self.center
        return points

    def get_centers(self):
        return self.center

    def set_heading_angle(self, angle, add=False):
        if add:
            self.heading_angle -= degrees(angle)
        else:
            self.heading_angle = degrees(angle)

    def set_center(self, center, add=False):
        if add:
            self.center += center
        else:
            self.center = center


# handles the performance index: cross track error, lateral acceleration
class _CarPhysics:
    def __init__(self):
        self.lat_g = 0  # lateral acceleration
        self.cross_track_e = 0  # cross track error

    def update_physics(self):
        pass


# sets the configurations of the car
class CarConfig(Enum):
    length = 1.2
    width = 1
    wheel_radius = 0.5
    wheel_deflection = 0
    max_turn_angle = 15  # degree


class WayPoints:
    def __init__(self, n_points, config):
        self.n_points = n_points
        self.config = config
        self.way_points = None

    def generate_way_points(self, length):
        if self.config == WayPointsConfig.straight_line:
            from numpy import linspace, ones
            start = -2
            points_x = ones((1, self.n_points)) * 0  # change if need different x
            points_y = linspace(start, start + length, self.n_points).reshape((1, self.n_points))
            self.way_points = append(points_x, points_y, axis=0)

    def get_way_points(self, row):
        if row == 'x':
            return self.way_points[0, :]
        elif row == 'y':
            return self.way_points[1, :]

    def get_near_way_points(self, center, search_dist):
        search_dist = search_dist ** 2
        temp_wp = (self.way_points - center) ** 2
        print()


class WayPointsConfig(Enum):
    straight_line = 0
    right_turn = 1
    left_turn = 2
