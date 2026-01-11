import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    def __init__(self, action_size, device, in_channels=4, input_height=84, input_width=84):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Compute conv output dim with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_height, input_width)
            conv_out = self.conv(dummy)
            conv_out_dim = conv_out.view(1, -1).size(1)

        # You’ll compute conv_out_dim using a dummy forward once you know H, W
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
        )

        # TODO: Create network

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """
        # Convert numpy to tensor if needed
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)

        x = observation.to(self.device).float()

        # Ensure batch dimension
        if x.dim() == 3:      # (C, H, W)
            x = x.unsqueeze(0)

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        q_values = self.fc(x)
        return q_values
        # TODO: Forward pass through the network

