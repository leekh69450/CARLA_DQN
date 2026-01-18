import argparse
import deepq
import glob
import sys
import os

carla_root = r"C:/Users/leekh/Downloads/newCARLA_0.9.10.1/WindowsNoEditor/PythonAPI/carla/dist"

try:
    sys.path.append(
        glob.glob(
            os.path.join(
                carla_root,
                "carla-*%d.%d-%s.egg" % (
                    sys.version_info.major,
                    sys.version_info.minor,
                    "win-amd64"
                )
            )
        )[0]
    )
except IndexError:
    raise RuntimeError("CARLA egg not found. Check your path and Python version.")

import carla
from carla_racing_env import CarlaRacingEnv


def make_env(host: str = "localhost", port: int = 2000, seed: int = 0):
    """Create the CARLA racing environment for evaluation."""
    # During evaluation you probably want rendering ON to watch the agent
    env = CarlaRacingEnv(
        host=host,
        port=port,
        seed=seed,       # deepq.evaluate will override seed per episode
        render = False, # e.g. render=True if your env supports it
    )
    return env
#The default parameters for training a agent can be found in deepq.py
def main():
    """ 
    Train a Deep Q-Learning agent 
    """ 
    #Initialize your carla env above and train the Deep Q-Learning agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost",
                        help="CARLA host (default: localhost)")
    parser.add_argument("--port", type=int, default=2000,
                        help="CARLA TCP port (default: 2000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for environment.")
    parser.add_argument("--model_id", type=str, default="agent",
                        help="Prefix/name for saved model and plots (e.g. agent.pt).")
    args = parser.parse_args()

    env = make_env(host=args.host, port=args.port, seed=args.seed)

    try:
        deepq.learn(
            env=env,
            model_identifier=args.model_id,
        )
    finally:
        env.close()

if __name__ == '__main__':
    main()

