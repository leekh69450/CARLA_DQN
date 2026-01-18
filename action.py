import random
import torch


def select_greedy_action(state, policy_net, action_size):
    """ Select the greedy action
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    Returns
    -------
    int
        ID of selected action
    """

    # TODO: Select greedy action
    # Convert state (np.array) to a batch tensor on the same device as policy_net
    device = next(policy_net.parameters()).device
    state_t = torch.from_numpy(state).float().to(device)

    # Add batch dimension if needed
    if state_t.dim() == 3:  # (C, H, W)
        state_t = state_t.unsqueeze(0)  # (1, C, H, W)

    policy_net.eval()
    with torch.no_grad():
        q_values = policy_net(state_t)          # shape: (1, action_size)
        action = int(q_values.argmax(dim=1).item())

    return action

def select_exploratory_action(state, policy_net, action_size, exploration, t):
    """ Select an action according to an epsilon-greedy exploration strategy
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    exploration: LinearSchedule
        linear exploration schedule
    t: int
        current time-step
    Returns
    -------
    int
        ID of selected action
    """

    # TODO: Select exploratory action
    # Get current epsilon from the schedule
    epsilon = exploration.value(t)

    # With probability epsilon: random action (exploration)
    if random.random() < epsilon:
        return random.randrange(action_size)

    # Otherwise: greedy action (exploitation)
    return select_greedy_action(state, policy_net, action_size)

def get_action_set():
    """ Get the list of available actions
    Returns
    -------
    list
        list of available actions
    """
    return [
        [-0.7, 0.6, 0.0],
        [ 0.7, 0.6, 0.0],
        [ 0.0, 0.9, 0.0],
        [ 0.0, 0.0, 1.0],
        #[ 0.0, 0.0, 0.0],
    ]

