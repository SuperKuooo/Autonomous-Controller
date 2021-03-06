from models import CarManager, WayPoints, WayPointsConfig
from matplotlib import use
from numpy import concatenate
import matplotlib.pyplot as plt

# initialize matplotlib stuff
use("TkAgg")
plt.ion()
figure, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-20, 20)
ax.set_ylim(-5, 30)
plt.title("Car Controller Simulation", fontsize=25)

# initialize lines
car_line, = ax.plot(0, 0)
wp_line, = ax.plot(0, 0, 'ro', markersize=1)


class Simulator:
    og_x = 0
    og_y = -2
    og_theta = 0.001

    def __init__(self, delta_t):
        self.delta_t = delta_t  # in milliseconds

        # initialize car
        self.car = CarManager(self.og_x, self.og_y, self.og_theta)  # x, y, theta

        # initialize way points
        self.wp = WayPoints(30, WayPointsConfig.straight_line)  # number of points and configurations
        self.wp.generate_way_points(30)  # length of path in meters

    def step(self, delta_t, actions):
        """
        updates the simulation environment at given time step and actions
        :param delta_t: time step duration
        :param actions: actions taken at this time step
        :return:
        """

        # TODO: set a delay to turn angle and velocity
        v_clip = self.car.set_velocity(actions[0])
        turn_angle_clip = self.car.set_turn_angle(actions[1])

        # update car location
        self.car.update(delta_t / 1000)

        # get cross-trek error and goal error
        ct_e = self.wp.get_cross_trek(self.car)
        goal_e = self.wp.get_goal_error(self.car)
        done = True if goal_e < 0.3 else False

        # get velocity, turn_angle, heading angle
        states = self.car.get_states()
        velocity = states[0]

        # get nearest waypoints
        wp = self.wp.get_upcoming_waypoints(10)
        states = concatenate((states, wp[0, :], wp[1, :]))

        # calculate reward
        reward = -ct_e ** 2 - 100 * goal_e - 100 * (v_clip + turn_angle_clip) + done * 300000
        # reward = -ct_e ** 1.8 + self.delta_t * 1000 * velocity * 10 - goal_e - 15 * (v_clip + turn_angle_clip)
        """
        Rewards function explaination
        1. Cross trek error is heavily punished
        2. Reduce distance traveled so the car doesn't go back and forth
        3. The closer to the goal, the less punishment it is
        4. Punished when model hits max velocity or turn angle
        """
        return states, reward, done

    def render(self):
        """
        Plots the result on matplotlib
        :return:
        """
        center, car_points = self.car.get_points()

        # update car plot
        x = car_points[0, 1:]
        y = car_points[1, 1:]
        car_line.set_xdata(x)
        car_line.set_ydata(y)

        # update way points plot
        wp_line.set_xdata(self.wp.get_way_points('x'))
        wp_line.set_ydata(self.wp.get_way_points('y'))
        figure.canvas.draw()
        figure.canvas.flush_events()

    def reset(self):
        """
        resets the simulation to the starting positions
        :return:
        """
        self.car = CarManager(self.og_x, self.og_y, self.og_theta)
        states = self.car.get_states()
        wp = self.wp.get_upcoming_waypoints(10)
        return concatenate((states, wp[0, :], wp[1, :]))

    @staticmethod
    def plot_loss(loss):
        x = list(range(len(loss)))
        plt.plot(x, loss)
        plt.show()


if __name__ == "__main__":
    sim = Simulator(50)
    action = list()
    for i in range(100):
        sim.step(50 / 1000, action)
        sim.render()
