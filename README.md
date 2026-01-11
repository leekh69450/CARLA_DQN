\# CARLA DQN — Deep Reinforcement Learning for Autonomous Driving



This repository implements a \*\*Deep Q-Network (DQN)\*\* agent trained in the \*\*CARLA simulator\*\* for autonomous driving in a racing-style environment.



The goal of this project is not to introduce new algorithms, but to reproduce and apply the \*\*system-level techniques\*\* that made deep reinforcement learning stable in practice, as introduced in the \*Atari DQN\* paper. The focus is on \*\*engineering correctness, training stability, and clean code structure\*\*.



---



\## Project Overview



The agent learns a discrete driving policy directly from observations by interacting with a custom CARLA environment. Key reinforcement learning components such as \*\*experience replay\*\*, \*\*target networks\*\*, and \*\*epsilon-greedy exploration scheduling\*\* are implemented from scratch.



This repository serves as a \*\*baseline DQN implementation\*\* that can be extended to more advanced methods such as \*\*Double DQN\*\*, behavior-cloning initialization, or alternative reward and action space designs.



---



\## Implemented Features



\- Custom CARLA environment wrapper

\- Discrete action space for vehicle control

\- Reward shaping for lane following, stability, and progress

\- Deep Q-Network (DQN) implemented in PyTorch

\- Experience replay buffer

\- Fixed target network for stable learning

\- Epsilon-greedy exploration with scheduling

\- Training and evaluation scripts

\- Saved training curves and example checkpoint



---



\## Repository Structure



```text

CARLA\_DQN/

├── agent/              # DQN model, replay buffer, learning logic

├── environment/        # CARLA environment wrapper and reward function

├── plots/              # Training reward and loss curves

├── checkpoints/        # Saved model checkpoints

├── docs/               # Reserved for future documentation

├── train.py            # Training entry point

├── evaluate.py         # Evaluation script

├── requirements.txt    # Python dependencies

└── README.md



---



\## Requirements



Basic Python dependencies are listed in requirements.txt:



```bash

pip install -r requirements.txt



Note: CARLA must be installed separately.

Ensure the CARLA Python API is available on your PYTHONPATH before running training or evaluation.



---



\## Training



To train the DQN agent:



```bash

python train.py





During training:



Transitions are stored in a replay buffer



Mini-batches are sampled randomly to break temporal correlations



Target network updates are used to stabilize learning



Training reward and loss curves are saved to the plots/ directory.



---



\## Evaluation



To evaluate a trained agent:



```bash

python evaluate.py





An example pretrained checkpoint is provided in the checkpoints/ directory for reference.



---



\## Design Philosophy



This project emphasizes:



* Correct reinforcement learning system design



* Clear separation between environment, agent, and training logic



* Reproducibility and readability over novelty



It is intended as a solid foundation for further exploration of reinforcement learning methods applied to autonomous driving.



--- 



\## Future Work



Planned extensions include:



* Double DQN to reduce overestimation bias



* Alternative reward formulations



* Action space redesign and ablation studies



* Behavior cloning as an initialization baseline



* More complex driving scenarios

---



\## Author



Stan Lee

M.S. Robotics \& Autonomous Systems

Boston University



