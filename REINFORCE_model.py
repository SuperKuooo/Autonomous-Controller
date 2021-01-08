import numpy as np
import torch
import torch.optim as optim

from simulator import Simulator
from RL_models import Pi

gamma = 0.99  # used to calculate return

in_dim = 5 + 20  # number of observing states

# Output Dimensions
# Normal distribution, so two parameters each
# 1. velocity
# 2. turn angle
out_dim = 2 * 2
loss_list = list()


def train(pi, optimizer):
    T = len(pi.rewards)
    rets = np.empty(T)

    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret

    rets = torch.tensor(rets)
    v_log_probs = torch.stack(pi.v_log_probs)
    t_log_probs = torch.stack(pi.turn_log_probs)
    v_loss = -v_log_probs * rets
    t_loss = -t_log_probs * rets
    loss = torch.sum(v_loss) + torch.sum(t_loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_list.append(loss)
    return loss


def main():
    sim_length = 10  # simulation length per episode in seconds (simulation time)
    delta_t = 200  # time step duration in millisecond (simulation time)

    sim = Simulator(delta_t)
    pi = Pi(in_dim, out_dim)
    optimizer = optim.Adam(pi.parameters(), lr=0.01)

    try:
        for epoch in range(1000):
            states = sim.reset()
            for i in range(int(sim_length / delta_t * 1000)):
                actions = pi.act(states)
                states, reward, done = sim.step(delta_t, actions)  # input step in millisecond
                # sim.render()

                pi.rewards.append(reward)
                if done:
                    break

            loss = train(pi, optimizer)
            total_reward = sum(pi.rewards)
            pi.on_policy_reset()
            print(f'Epoch: {epoch}, loss: {loss}, total reward: {total_reward}')
    finally:
        print("Training has been stopped")
        torch.save(pi.model.state_dict(), './trained/straight_line')
        sim.plot_loss(loss_list)


if __name__ == '__main__':
    main()
