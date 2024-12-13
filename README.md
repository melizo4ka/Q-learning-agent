# Reinforcement Learning Grid World

This project implements a grid-based reinforcement learning environment with Q-learning. The grid world is designed to train an agent to collect treasures while avoiding walls. The environment uses a graphical interface to visualize the agent's progress and rewards.
In this project, the Q-learning algorithm is enhanced with the epsilon-greedy policy for action selection. The Q-value update rule is given by:

$$\
Q(s, a) \leftarrow Q(s, a) + \beta \big( R + \alpha \max_{a'} Q(s', a') - Q(s, a) \big)
\$$

Where:
- $\Q(s, a)$: Q-value for state s and action a.
- \(\beta\): Learning rate.
- \(R\): Reward obtained after taking action \(a\) in state \(s\).
- \(\alpha\): Discount factor.
- \(s'\): Next state after taking action \(a\).
- \(\max_{a'} Q(s', a')\): Maximum Q-value for the next state \(s'\) over all possible actions.

The epsilon-greedy policy balances exploration and exploitation during training. At each step:
- With probability \(\epsilon\), the agent explores by choosing a random action.
- With probability \(1 - \epsilon\), the agent exploits by choosing the action with the highest Q-value for the current state (greedy action).

Initially, \(\epsilon\) is high to encourage exploration. It gradually decreases over time to prioritize exploitation, enabling the agent to converge to an optimal policy.



## Features

- **10x10 Grid World:** The environment consists of a 10x10 grid with cells containing treasures and walls.
- **Q-Learning Agent:** The agent learns using Q-learning with configurable learning rate (`beta`) and discount factor (`alpha`).
- **Treasure Collection:** The agent earns rewards for collecting treasures  (+10) and incurs penalties for movement (-1).
- **Walls and Obstacles:** Movement is restricted by walls defined at before training.
- **Visualization:** A graphical display shows the agent's actions and treasures collected.
- **Performance Tracking:** Rewards per step are plotted at the end of training cycle using `matplotlib` to visualize training progress.

## Installation

To use this project you need to install the required Python packages:
   ```bash
   pip install pygame matplotlib
   ```

### Controls and Parameters

- **Environment Parameters:**
  - `BOARD_ROWS` and `BOARD_COLS`: Size of the grid world (default is 10x10).
  - `TREASURE_POSITIONS`: Coordinates of treasures in the grid.
  - `WALLS`: List of horizontal and vertical walls defined by their start and end coordinates.
- **Agent Learning:**
  - `beta`: Learning rate for Q-learning (default is 0.6).
  - `alpha`: Discount factor for future rewards (default is 0.95).
- **Training Parameters (can be set by the user):**
  - `total_epochs`: Number of training epochs (default is 30).
  - `steps_per_epoch`: Number of steps per epoch (default is 30).
  - `max_moves_per_step`: Maximum moves allowed per step (default is 400).

## Code Structure

- **`main.py`:** The main script to run the grid world simulation.
- **Key Classes:**
  - `State`: Represents the agent’s state, treasures, and termination condition.
  - `Agent`: Implements Q-learning with action selection, state updates, and reward calculation.
  - `GridWorldDisplay`: Handles the graphical rendering of the grid world using `pygame`.
- **Helper Functions:**
  - `generate_wall_segments`: Converts wall coordinates into renderable segments.
  - `is_valid_move`: Validates movement actions based on the grid and walls.
  - `random_start_position`: Generates a random initial position for the agent.

## Output

- A `pygame` window showing:
  - Agent position in blue.
  - Treasures in gold (green if collected).
  - Walls in black.
- A matplotlib plot of rewards per step across epochs and steps.
- Printed Q-Table showing the learned policy.

