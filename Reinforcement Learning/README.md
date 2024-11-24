# FrozenLake Q-Learning Agent

This repository contains a Python implementation of a Q-learning agent to solve the FrozenLake-v1 environment using reinforcement learning. The notebook is designed to provide a hands-on demonstration of the Q-learning algorithm, covering the entire pipeline from environment setup and training to visualizing results.

# Overview

The FrozenLake environment is a classic grid-world problem where an agent must navigate from a start position to a goal position while avoiding holes. The grid is slippery, adding randomness to the agent's movements.

**This notebook implements:**

- **Q-learning Algorithm:** An off-policy reinforcement learning algorithm for training the agent.
- **Visualization:** Real-time rendering of the agent's performance using pyvirtualdisplay and Matplotlib.
- **Analysis:** Visualization of training progress with average rewards per episode.

# Dependencies

Before running the notebook, ensure the required dependencies are installed:
```python
!apt-get install python-opengl -y
!apt install xvfb -y
!pip install pyvirtualdisplay
!pip install piglet
!pip install -U colabgymrender
!pip install gymnasium
```

**The notebook also uses the following Python libraries:**

- gymnasium: For creating and interacting with the FrozenLake environment.
- numpy: For numerical computations and managing the Q-table.
- matplotlib: For visualizing training progress.
- time: For tracking training duration.
- pyvirtualdisplay: For rendering the environment.

# Key Features

- Customizable Training: Easily modify hyperparameters such as the number of episodes, learning rate, and epsilon decay.
- Real-time Rendering: Watch the agent navigate the FrozenLake environment after training.
- Performance Metrics: Track training progress with a plot of average rewards per episode.

# Notebook Highlights

**Environment Initialization**

The FrozenLake environment is initialized with the following configurations:

- is_slippery=True: Simulates slippery surfaces to add randomness.
- render_mode='rgb_array': Renders the environment as a visual grid.

**Q-learning Algorithm**

The notebook implements Q-learning with:

- Exploration-exploitation trade-off using an epsilon-greedy strategy.
- Incremental updates to the Q-table based on the Bellman equation.

**Visualization**

- Training Progress: A plot of average rewards over episodes provides insights into the agent's learning curve.
- Agent Path: Watch the trained agent navigate the FrozenLake environment in real time.

![image](https://github.com/user-attachments/assets/6f9e1e74-c1c8-4b49-95a3-5d671ee75368)

# Example Output

**Plot Average Rewards:**

![image](https://github.com/user-attachments/assets/ef4989d0-6324-4c37-823f-f4431eb32283)


**Agent Path Visualization:**

![image](https://github.com/user-attachments/assets/bdbd3d14-4611-4130-a88b-6caf58bbf108)

