import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """

    # TODO: Run single Q-learning step
    #Steps: 

    #    1. Sample transitions from replay_buffer

    # Expected: replay_buffer.sample(batch_size) -> (states, actions, rewards, next_states, dones)
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    #    2. Compute Q(s_t, a)
    states      = torch.from_numpy(states).float().to(device)
    next_states = torch.from_numpy(next_states).float().to(device)
    actions     = torch.from_numpy(actions).long().to(device)
    rewards     = torch.from_numpy(rewards).float().to(device)
    dones       = torch.from_numpy(dones.astype(np.float32)).float().to(device)  # 1.0 if done, else 0.0

    #    3. Compute \max_a Q(s_{t+1}, a) for all next states.

    #    Q(s_t, ·) has shape (B, num_actions)
    q_values = policy_net(states)
    #    Gather Q(s_t, a_t) for the taken actions
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # shape (B,)

    #    4. Mask next state values where episodes have terminated

    with torch.no_grad():
        next_q_values = target_net(next_states)                  # (B, num_actions)
        max_next_q, _ = next_q_values.max(dim=1)                 # (B,)

    #    5. Compute the target

    #    If done == 1, then there is no bootstrap term
        max_next_q = max_next_q * (1.0 - dones)

    #    6. Compute the loss

        target = rewards + gamma * max_next_q

    #    7. Calculate the gradients

    loss = F.mse_loss(q_sa, target)

    #    8. Clip the gradients

    optimizer.zero_grad()
    loss.backward()

    #    9. Optimize the model

    clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()
    

def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """

    # TODO: Update target network
    
    target_net.load_state_dict(policy_net.state_dict())

