import torch
import torch.nn as nn
from torch.distributions import Normal

n_node = 64


# Policy Class
class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, n_node),
            nn.ReLU(),
            nn.Linear(n_node, out_dim)
        ]
        self.model = nn.Sequential(*layers)
        self.v_log_probs = list()
        self.turn_log_probs = list()
        self.rewards = list()

        self.on_policy_reset()
        self.train()

    def on_policy_reset(self):
        self.v_log_probs = []
        self.turn_log_probs = []
        self.rewards = []

    def forward(self, x):
        # print(x)
        pdparam = self.model(x)
        return pdparam

    def act(self, state):  # this is like the inference state
        x = torch.from_numpy(state).float()
        pdparam = self.forward(x)

        # run network
        v_loc, v_scale = pdparam[0], pdparam[1]
        turn_loc, turn_scale = pdparam[2], pdparam[3]
        v_pd = Normal(loc=v_loc, scale=v_scale)
        turn_pd = Normal(loc=turn_loc, scale=turn_scale)

        # sample velocity and turn angle
        new_v = v_pd.sample()
        new_turn = turn_pd.sample()

        v_prob = v_pd.log_prob(new_v)  # a perfect certainty will have 0 log prob
        turn_prob = turn_pd.log_prob(new_turn)

        self.v_log_probs.append(v_prob)
        self.turn_log_probs.append(turn_prob)

        return new_v.item(), new_turn.item()
