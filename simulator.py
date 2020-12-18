from models import CarManager, WayPoints, WayPointsConfig

from matplotlib import use
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# initialize matplotlib stuff
use("TkAgg")
fig, ax = plt.subplots()
ax.set_xlim(-20, 20)
ax.set_ylim(-5, 30)
car_line, = ax.plot(0, 0)
wp_line, = ax.plot(0, 0, 'ro', markersize=1)

# initialize car
delta_t = 50
car = CarManager(5, 5, 180)

# initialize way points
wp = WayPoints(30, WayPointsConfig.straight_line)
wp.generate_way_points(30)  # input length


def animate(i):
    car.update(delta_t / 1000)
    center, car_points = car.get_states()

    near_wp = wp.get_near_way_points(center, 8)
    x = car_points[0, 1:]
    y = car_points[1, 1:]
    car_line.set_xdata(x)
    car_line.set_ydata(y)

    wp_line.set_xdata(wp.get_way_points('x'))
    wp_line.set_ydata(wp.get_way_points('y'))
    return car_line,


ani = animation.FuncAnimation(fig, func=animate, frames=5, interval=delta_t, blit=True)
plt.show()
