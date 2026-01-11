# evaluate_racing.py
import argparse
import deepq
from carla_racing_env import CarlaRacingEnv


def make_env(host: str = "localhost", port: int = 2000, render: bool = True):
    """
    Create the CARLA racing environment for evaluation.

    During evaluation we usually want rendering ON so we can watch the agent.
    """
    env = CarlaRacingEnv(
        host=host,
        port=port,
        seed=0,
        render=render,
    )
    return env


def main():
    """
    Evaluate a trained Deep Q-Learning agent.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="CARLA host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA TCP port (default: 2000)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="dqn_hw3.pt",   # <- change this if your file is named differently
        help="Path to trained model (.pt) file",
    )
    args = parser.parse_args()

    env = make_env(host=args.host, port=args.port, render=True)

    try:
        # deepq.evaluate will load the model and run greedy policy episodes
        deepq.evaluate(env, load_path=args.model_path)
    finally:
        env.close()


if __name__ == "__main__":
    main()
