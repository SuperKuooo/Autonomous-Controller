from simulator import Simulator
from RL_models import Pi
import torch


def main():
    sim_length = 10  # simulation length per episode in seconds (simulation time)
    delta_t = 200  # time step duration in millisecond (simulation time)

    sim = Simulator(delta_t)
    pi = Pi(25, 4, torch.load('./trained/straight_line'))

    try:
        while True:
            states = sim.reset()
            for i in range(int(sim_length / delta_t * 1000)):
                actions = pi.act(states)
                states, reward, done = sim.step(delta_t, actions)  # input step in millisecond
                # sim.render()
                pi.rewards.append(reward)
                if done:
                    break
            pi.on_policy_reset()
    except KeyboardInterrupt:
        print('Ended')

if __name__ == '__main__':
    main()
