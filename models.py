import torch
import torch.nn as nn
import torch.nn.functional as F

class ACNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        '''
        Initializes Actor Critic model
        params:
            - input_dim: dimension of state space
            - action_dim: dimension of action space
            - hidden_dim: number of neurons in hidden layers

        '''
        super(ACNetwork, self).__init__()
        
        # Fully connected actor network
        self.actor = nn.Sequential(nn.Linear(input_dim,hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim,hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, action_dim),
                                   nn.Tanh())
        
        # Fully connected critic network
        self.critic = nn.Sequential(nn.Linear(input_dim,hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim,hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, 1))
        
        
        # Extra tunable parameter for standard deviation of gaussian action distribution
        self.std = nn.Parameter(torch.zeros(action_dim))
        self.softplus = nn.Softplus()
        
    def forward(self, inputs, action=None):

        action_mean = self.actor(inputs)
        value = self.critic(inputs)

        # Create normal distribution to sample values from
        dist = torch.distributions.Normal(action_mean, self.softplus(self.std))
        
        if action is None:
            action = dist.sample()
       
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)        

        return action, log_prob, value

    def act_deterministic(self, inputs):
        return self.actor(inputs)

    