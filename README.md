# MSDS684 Reinforcement Learning

This repository contains my weekly lab assignments for **MSDS684 Reinforcement Learning**. Each week introduces a new reinforcement learning topic and applies the theory through Python implementations, Gymnasium environments, experiments, visualizations, and written analysis.

The labs follow the progression of the course from basic reinforcement learning foundations to dynamic programming, Monte Carlo methods, temporal-difference learning, function approximation, policy gradients, planning methods, and final course synthesis.

## Repository Purpose

The purpose of this repository is to document my reinforcement learning coursework and implementations for MSDS684. Each weekly folder contains code and analysis related to that week’s topic. The assignments focus on implementing reinforcement learning algorithms from scratch where possible, using NumPy, Gymnasium, Matplotlib, and PyTorch.

The repository is organized by week, with each week focusing on a different reinforcement learning concept.

---

## Week 1: Multi-Armed Bandit Problems and MDP Foundations

Week 1 introduced the basic foundations of reinforcement learning through multi-armed bandit problems and Markov Decision Processes.

The lab focused on the exploration-exploitation tradeoff. A custom multi-armed bandit environment was implemented using the Gymnasium interface. The environment contained multiple arms, each producing stochastic rewards from unknown reward distributions. The agent had to learn which arm produced the highest expected reward through repeated interaction.

The main algorithms implemented were:

- Epsilon-greedy action selection
- Upper Confidence Bound action selection
- Random policy baselines for Gymnasium environments

The lab compared how different exploration strategies affected average reward and optimal action selection over time. It also introduced MDP structure by examining Gymnasium environments such as FrozenLake and Taxi. The purpose was to understand states, actions, rewards, transition dynamics, and how reinforcement learning problems are formally represented.

---

## Week 2: Dynamic Programming

Week 2 focused on Dynamic Programming methods for solving Markov Decision Processes when the full transition model is known.

The lab implemented planning algorithms for grid-based environments. A custom GridWorld environment was created with deterministic and stochastic transition dynamics. The transition model was explicitly represented so that Dynamic Programming algorithms could compute optimal values and policies.

The main algorithms and concepts covered were:

- Value iteration
- Policy iteration
- Bellman optimality updates
- Policy evaluation and policy improvement
- Deterministic versus stochastic transition dynamics
- Synchronous versus in-place value iteration

The lab included value-function heatmaps, policy visualizations, and convergence curves. The results showed how deterministic environments allow more direct policies and faster convergence, while stochastic environments produce lower values and more cautious policies. FrozenLake was also used to confirm that the implementation could generalize beyond the custom GridWorld.

---

## Week 3: Monte Carlo Methods

Week 3 focused on Monte Carlo methods for learning from complete episodes.

The lab implemented first-visit Monte Carlo control for Gymnasium’s Blackjack-v1 environment. Unlike Dynamic Programming, Monte Carlo methods do not require a full transition model. Instead, the agent learns action values by sampling full episodes and computing returns after each episode terminates.

The main algorithm implemented was:

- First-visit Monte Carlo control with epsilon-soft policies

The Blackjack state was represented as:

- Player sum
- Dealer showing card
- Usable ace indicator

The action space contained:

- Stick
- Hit

The agent learned an action-value function by averaging returns following the first visit to each state-action pair. The learned value function was visualized using 3D surface plots for usable ace and non-usable ace states. The learned policy was visualized and compared with simplified basic Blackjack strategy.

The lab also experimented with different epsilon values and epsilon decay schedules to study the effect of exploration on Monte Carlo control.

---

## Week 4: Temporal-Difference Learning

Week 4 focused on Temporal-Difference learning, especially the difference between on-policy and off-policy TD control.

The lab used Gymnasium’s CliffWalking-v0 environment to compare SARSA and Q-learning. Both algorithms used NumPy Q-tables and epsilon-greedy exploration. Unlike Monte Carlo methods, TD methods update after each step rather than waiting for the full episode to finish.

The main algorithms implemented were:

- SARSA
- Q-learning

The SARSA update used the next action actually selected by the epsilon-greedy policy. The Q-learning update used the maximum next action value. This difference produced different behavior in the CliffWalking environment.

The lab included:

- Learning curves across multiple random seeds
- 95% confidence intervals
- Learned policy arrows
- Value-function heatmaps
- Sample trajectories
- Step-size experiments
- Epsilon-decay experiments

The results demonstrated that SARSA tends to learn a safer path because it accounts for exploratory actions, while Q-learning tends to learn the shorter optimal path near the cliff, which can be riskier during training.

---

## Week 5: Function Approximation with Value Methods

Week 5 introduced the transition from tabular reinforcement learning to function approximation.

The lab used Gymnasium’s MountainCar-v0 environment, which has a continuous observation space consisting of position and velocity. Since continuous states cannot be represented directly with a finite table, the lab implemented tile coding as a feature representation.

The main algorithm implemented was:

- Semi-gradient SARSA with tile coding

The tile coder was implemented from scratch using NumPy. It converted continuous observations into sparse binary features by using multiple overlapping tilings. Each action maintained its own separate weight vector, and Q-values were computed as a linear function of the active tile features.

The lab included:

- Tile coding feature construction
- Semi-gradient SARSA updates
- Episode-length curves
- Return curves
- TD-error curves
- Learned value-function heatmaps
- Learned policy heatmaps
- Sample trajectories overlaid on the value function
- Feature configuration experiments
- Step-size experiments
- Epsilon-schedule experiments

The lab showed why function approximation is necessary for large or continuous state spaces and how feature design affects learning speed, final performance, and computational cost.

---

## Week 6: Policy Gradient Methods

Week 6 focused on direct policy optimization using policy gradient methods.

The lab implemented policy gradient algorithms using PyTorch. The first part used Gymnasium’s CartPole-v1 environment with discrete actions. The second part used Gymnasium’s Pendulum-v1 environment with continuous actions.

The main algorithms implemented were:

- REINFORCE
- REINFORCE with baseline
- Actor-Critic with TD(0)

For CartPole, a neural network policy output action logits, which were converted into probabilities using softmax. Actions were sampled using a categorical distribution. REINFORCE was trained using complete episode returns, while REINFORCE with baseline used a separate value network to reduce variance.

For Pendulum, the actor network output Gaussian policy parameters for continuous actions. The critic estimated state values. The actor and critic were updated online using TD error.

The lab included:

- PyTorch policy networks
- Value baseline networks
- Complete episode return computation
- Policy loss using log probabilities
- Actor-Critic updates
- Policy entropy plots
- TD-error plots
- Actor and critic loss curves
- Learned policy visualizations
- Model checkpointing with torch.save

The results showed how baselines reduce variance in policy gradients and how Actor-Critic methods combine policy optimization with value-based bootstrapping.

---

## Week 7: Planning and Learning Integration

Week 7 focused on integrating learning and planning through model-based reinforcement learning.

The lab used Gymnasium’s Taxi-v3 environment and implemented Dyna-style algorithms. The agent learned from real environment interactions and also used a learned model for simulated planning updates.

The main algorithms implemented were:

- Pure Q-learning
- Dyna-Q
- Dyna-Q+
- Prioritized Sweeping

Dyna-Q used three integrated components:

- Direct reinforcement learning through Q-learning updates
- Model learning with a Python dictionary storing observed transitions
- Planning updates from simulated model experience

The model was represented as:

```python
model[(state, action)] = (reward, next_state, terminated)
