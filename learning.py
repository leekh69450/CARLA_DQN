import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device,double_dqn = False):
    """
    Perform one Deep Q-Learning (DQN) optimization step using experience replay
    and a frozen target network.

    This implements the 1-step TD target:
        y = r + gamma * (1 - done) * max_a Q_target(s', a)

    Parameters
    ----------
    policy_net : torch.nn.Module
        Online Q-network Q(s, a; θ) used for selecting Q(s_t, a_t) and updated by SGD.
    target_net : torch.nn.Module
        Target Q-network Q(s, a; θ^-) used only to compute bootstrap targets (no gradients).
    optimizer : torch.optim.Optimizer
        Optimizer for updating policy_net parameters.
    replay_buffer : ReplayBuffer
        Buffer providing random minibatches of transitions.
        Must implement sample(batch_size) -> (states, actions, rewards, next_states, dones).
    batch_size : int
        Number of transitions sampled for this update.
    gamma : float
        Discount factor in [0, 1].
    device : torch.device
        Device to move sampled tensors to (e.g., CPU or CUDA).
    double_dqn : bool
        if True then perform double dqn
        if False then perform vanila dqn

    Returns
    -------
    float
        Scalar TD loss (MSE) as a Python float.
    """

    # TODO: Run single Q-learning step
    #Steps: 

    # 1. Sample a random minibatch of transitions from replay buffer
    #    Each transition: (state, action, reward, next_state, done)
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # 2. Move data to torch tensors on the correct device
    states      = torch.from_numpy(states).float().to(device)
    next_states = torch.from_numpy(next_states).float().to(device)
    actions     = torch.from_numpy(actions).long().to(device)
    rewards     = torch.from_numpy(rewards).float().to(device)
    dones       = torch.from_numpy(dones.astype(np.float32)).float().to(device)  
    # dones = 1.0 if terminal transition, else 0.0

    # 3. Compute Q(s_t, ·) from the online (policy) network
    #    Shape: (B, num_actions)
    q_values = policy_net(states)

    # 4. Select Q(s_t, a_t) for the actions actually taken
    #    Shape after gather: (B,)
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # shape (B,)

    # 5. Compute bootstrap target using the frozen target network
    #    No gradients should flow through the target network
    with torch.no_grad():
        if double_dqn:
            # Double DQN:
            # 1) select a* using online net
            next_q_online = policy_net(next_states)        # (B, A)
            next_actions = next_q_online.argmax(dim=1)     # (B,)

            # 2) evaluate Q_target(s', a*)
            next_q_target = target_net(next_states)        # (B, A)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # (B,)
        else:
            # Vanilla DQN:
            next_q_target = target_net(next_states)
            next_q, _ = next_q_target.max(dim=1)
    # Mask terminal states to stop bootstrapping
    # If done == 1, then future value contribution is zero
        next_q = next_q * (1.0 - dones)

    # TD target: y = r + gamma * max_a Q_target(s', a)
        target = rewards + gamma * next_q

    # 6. Compute TD error loss (Mean Squared Error)
    loss = F.mse_loss(q_sa, target)

    # 7. Backpropagate loss through the policy network
    optimizer.zero_grad()
    loss.backward() 

    # 8. Clip gradients to improve stability
    clip_grad_norm_(policy_net.parameters(), max_norm=1.0)

    #    9. Optimize the model
    optimizer.step()

    return loss.item()
    

def update_target_net(policy_net, target_net):
    """
    Hard-update the target network parameters by copying from the online network.

    θ^- ← θ

    Parameters
    ----------
    policy_net : torch.nn.Module
        Online Q-network whose parameters will be copied.
    target_net : torch.nn.Module
        Target Q-network to receive the copied parameters.
    """

    # TODO: Update target network
    
    target_net.load_state_dict(policy_net.state_dict())

