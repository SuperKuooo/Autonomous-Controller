import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Categorical
from simulator import Simulator
from time import sleep

gamma = 0.99
n_node = 64
in_dim = 5  # number of observing states
out_dim = 3  # number of changes


# Policy Class
class Pi(nn.Module):
    def __init__(self):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, n_node),
            nn.ReLU(),
            nn.Linear(n_node, out_dim)
        ]
        self.model = nn.Sequential(*layers)
        self.log_probs = list()
        self.rewards = list()
        self.on_policy_reset()
        self.train()

    def on_policy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    def act(self, state):
        x = torch.from_numpy(state)
        pdparam = self.forward(x)
        pd = Categorical(logits=pdparam)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item


def train(pi, optimizer):
    T = len(pi.rewards)
    rets = np.empty(T)

    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret

    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = -log_probs * rets
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main():
    pb_ratio = 0  # Reduce real time waiting time (fast < 1 < slow)
    sim_length = 10  # simulation length per episode in seconds (simulation time)
    delta_t = 50.0  # time step duration in millisecond (simulation time)

    actions = list()
    sim = Simulator(delta_t)

    for epoch in range(200):
        sim.reset()
        for i in range(int(sim_length / delta_t * 1000)):
            states, reward, done = sim.step(delta_t, actions)  # input step in millisecond
            sim.render()

            if done:
                break
            sleep(delta_t / 1000 * pb_ratio)


if __name__ == '__main__':
    main()
