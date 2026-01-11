import numpy as np
import torch
import torch.optim as optim
import time
from action import get_action_set, select_exploratory_action, select_greedy_action
from learning import perform_qlearning_step, update_target_net
from model import DQN
from replay_buffer import ReplayBuffer
from schedule import LinearSchedule
from utils import get_state, visualize_training
from collections import deque
# initialize your carla env

class FrameStack:
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, frame):
        frame = self._to_hw(frame)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(frame)
        return self.get()

    def step(self, frame):
        frame = self._to_hw(frame)
        self.frames.append(frame)
        return self.get()

    def get(self):
        return np.stack(list(self.frames), axis=0)  # (k, H, W)

    @staticmethod
    def _to_hw(frame):
        frame = np.asarray(frame)

        # Remove ALL singleton dimensions:
        # (1,1,84,84) -> (84,84)
        # (1,84,84)   -> (84,84)
        frame = np.squeeze(frame)

        # Safety check: we must end with (H,W)
        if frame.ndim != 2:
            raise ValueError(f"Expected 2D frame after squeeze, got shape {frame.shape}")

        return frame

def evaluate(env, load_path='agent.pt'):
    """ Evaluate a trained model and compute your leaderboard scores

	NO CHANGES SHOULD BE MADE TO THIS FUNCTION

    Parameters
    -------
    env: Carla Env
        environment to evaluate on
    load_path: str
        path to load the model (.pt) from
    """
    episode_rewards = []
    actions = get_action_set()
    action_size = len(actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # These are not the final evaluation seeds, do not overfit on these tracks!
    seeds = [22597174, 68545857, 75568192, 91140053, 86018367,
             49636746, 66759182, 91294619, 84274995, 31531469]

    # Build & load network
    policy_net = DQN(action_size, device, in_channels=4).to(device)
    checkpoint = torch.load(load_path, map_location=device)
    policy_net.load_state_dict(checkpoint)
    policy_net.eval()

    # Iterate over a number of evaluation episodes
    for i in range(10):
        env.seed(seeds[i])
        obs_raw, done = env.reset(), False
        framestack = FrameStack(k=4)
        frame = get_state(obs_raw)
        obs = framestack.reset(frame)
        t = 0

        # Run each episode until episode has terminated or 600 time steps have been reached
        episode_rewards.append(0.0)
        while not done and t < 600:
            env.render()
            action_id = select_greedy_action(obs, policy_net, action_size)
            action = actions[action_id]
            obs_raw, rew, done, _ = env.step(action)
            frame = get_state(obs_raw)
            obs = framestack.step(frame)
            episode_rewards[-1] += rew
            t += 1
        print('episode %d \t reward %f' % (i, episode_rewards[-1]))

    print('---------------------------')
    print(' total score: %f' % np.mean(np.array(episode_rewards)))
    print('---------------------------')


def learn(env,
          lr=1e-4,
          total_timesteps = 100000,
          buffer_size = 50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          action_repeat=4,
          batch_size=32,
          learning_starts=1000,
          gamma=0.99,
          target_network_update_freq=500,
          model_identifier='agent'):
    """ Train a deep q-learning model.
    Parameters
    -------
    env: gym.Env
        environment to train on
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to take
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
    action_repeat: int
        selection action on every n-th frame and repeat action for intermediate frames
    batch_size: int
        size of a batched sampled from replay buffer for training
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    model_identifier: string
        identifier of the agent
    """
    episode_rewards = [0.0]
    training_losses = []
    actions = get_action_set()
    action_size = len(actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build networks
    policy_net = DQN(action_size, device,in_channels=4).to(device)
    target_net = DQN(action_size, device,in_channels=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Create replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Create optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize environment and get first state
    framestack = FrameStack(k=4)
    raw = env.reset()
    frame = get_state(raw)          # (1,H,W) or (H,W)
    print("DEBUG frame type:", type(frame))
    print("DEBUG frame shape:", np.array(frame).shape)
    obs = framestack.reset(frame)   # (4,H,W)
    print("DEBUG stacked obs shape:", obs.shape)

    start_time = time.time()
    print_every = 100

    # Iterate over the total number of time steps
    for t in range(total_timesteps):
        eps = exploration.value(t)

        # Select action
        action_id = select_exploratory_action(obs, policy_net, action_size, exploration, t)
        env_action = actions[action_id]

        total_rew = 0.0
        for f in range(action_repeat):
            new_raw, rew, done, _ = env.step(env_action)
            total_rew += rew
            episode_rewards[-1] += rew
            if done:
                break

        new_frame = get_state(new_raw)
        new_obs = framestack.step(new_frame)     # (4,H,W)

        replay_buffer.add(obs, action_id, total_rew, new_obs, float(done))
        obs = new_obs


        # --- compute ETA ---
        steps_done   = t + 1
        elapsed      = time.time() - start_time
        steps_left   = max(total_timesteps - steps_done, 0)
        if steps_done > 0:
            eta_seconds = (elapsed / steps_done) * steps_left
        else:
            eta_seconds = 0.0
        eta_min, eta_sec = divmod(int(eta_seconds), 60)

        # --- per-step logging ---
        if t % print_every == 0:
            # Note: rew is the last immediate reward from env.step
            print(
                f"[t={steps_done:6d}/{total_timesteps}] "
                f"eps={eps:.3f} "
                f"action_id={action_id} "
                f"env_action={env_action} "
                f"rew={rew:.3f} "
                f"ETA~{eta_min:02d}:{eta_sec:02d}",
                flush=True
            )

        if done:
            # Start new episode after previous episode has terminated
            print(
                f"Episode {len(episode_rewards)} finished at t={t} "
                f"R={episode_rewards[-1]:.2f} "
                f"(elapsed={elapsed/60:.1f} min, "
                f"ETA~{eta_min:02d}:{eta_sec:02d})",
                flush=True
            )
            obs_raw = env.reset()
            frame = get_state(obs_raw)
            obs = framestack.reset(frame)
            episode_rewards.append(0.0)

        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            loss = perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device)
            training_losses.append(loss)

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            update_target_net(policy_net, target_net)

    # Save the trained policy network
    torch.save(policy_net.state_dict(), model_identifier+'.pt')

    # Visualize the training loss and cumulative reward curves
    visualize_training(episode_rewards, training_losses, model_identifier)
