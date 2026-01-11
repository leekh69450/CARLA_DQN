import numpy as np

def reward_function(collision, speed, lane_invasion, current_position, next_waypoint, previous_distance, target_speed=15.0):
    """
    Calculate the reward for the agent based on its actions.

    Parameters:
    - collision (bool): Whether a collision occurred.
    - speed (float): The current speed of the vehicle.
    - lane_invasion (bool): Whether the vehicle has invaded a lane.
    - goal_reward
        - current_position (tuple): The current (x, y) position of the vehicle.
        - next_waypoint (tuple): The (x, y) position of the next waypoint.
    - target_speed (float): The target speed for the vehicle.

    Returns:
    - float: The calculated reward.
    """

    # 1. Hard penalty for collisions
    if collision:
        return -100.0

    # 2. Penalty for lane invasions
    if lane_invasion:
        return -10.0

    # 3. Speed reward (bell-shaped around target speed)
    #    speed_reward = 1 when speed == target_speed
    #    speed_reward = 0 when difference is large
    speed_reward = 1.0 - abs(speed - target_speed) / target_speed
    speed_reward = max(speed_reward, 0.0)

    # 4. Distance to next waypoint
    curr_pos = np.array(current_position, dtype=np.float32)
    next_wp = np.array(next_waypoint, dtype=np.float32)
    distance_to_waypoint = np.linalg.norm(curr_pos - next_wp)

    # Reward for being close to waypoint (bounded between 0 and 1)
    goal_reward = max(0.0, 1.0 - distance_to_waypoint)

    # 5. Directional progress penalty/reward
    #    If moving away from next waypoint → penalty
    if distance_to_waypoint > previous_distance:
        distance_penalty = -0.5
    else:
        distance_penalty = 0.0

    # 6. Combine all rewards
    total_reward = (
        10.0 * speed_reward +   # maintain good speed
        goal_reward +           # move toward waypoint
        distance_penalty        # penalize going backwards
    )

    # Optional: clip reward for training stability
    return float(np.clip(total_reward, -100.0, 100.0))

