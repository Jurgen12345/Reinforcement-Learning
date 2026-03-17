import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("QtAgg")


class MultiArmedBandit(gym.Env):
    def __init__(self, k = 10, reward_means = None, reward_stds=None,seed=42):
        super().__init__()
        self.k = k
        self.action_space = spaces.Discrete(k)
        self.observation_space = spaces.Discrete(1) 
        self.rng = np.random.default_rng(seed=seed)

        if reward_means == None:
            self.reward_means = self.rng.normal(loc=0.0, scale=0.1, size=k)
        else:
            self.reward_means = np.array(reward_means,dtype=float)

        if reward_stds == None:
            self.reward_stds = np.ones(k)
        else:
            self.reward_stds = np.array(reward_stds, dtype=float)
        
        self.best_arm = int(np.argmax(self.reward_means))
        self.state = 0
    
    def reset(self, seed = 42, options = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        self.state = 0
        info = {"best-arm":self.best_arm}
        return self.state, info 
    
    def step(self, action):
        reward = self.rng.normal(
            loc = self.reward_means[action],
            scale = self.reward_stds[action]
        )
        observation = 0
        termination = False
        truncation = False
        info = {
            "best-arm":self.best_arm,
            "optimal-arm":int(action == self.best_arm)
        }
        return observation, reward,termination, truncation, info 

class EpsilonGreedyAgent:
    def __init__(self, k = 10, epsilon = 0.1):
        self.k = k
        self.epsilon = epsilon

        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.rng = np.random.default_rng(seed =42)
    def select_action(self):
        if self.rng.random() < self.epsilon:
            return self.rng.integers(self.k)
        return np.argmax(self.Q)

    def update(self, action, reward):
        self.N[action] +=1
        n = self.N[action]
        self.Q[action] += (reward - self.Q[action]) /n


class UCBAgent:
    def __init__(self, k =10, c =2.0):
        self.k = k
        self.c = c
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.t = 0

    def select_action(self):
        self.t +=1

        for a in range(self.k):
            if self.N[a] == 0:
                return a

        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t)/ self.N)
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.N[action] +=1
        n = self.N[action]    
        self.Q[action] += (reward -self.Q[action]) /n



if __name__ == "__main__":
    epsilons = [0.01, 0.1, 0.2]
    runs = 1000
    steps = 2000

    for epsilon in epsilons:

        avg_rewards = np.zeros((runs, steps))
        optimal_actions = np.zeros((runs, steps))

        for run in range(runs):

            env = MultiArmedBandit(k=10, seed=run)
            epsilon_greedy_agent = EpsilonGreedyAgent(k=10, epsilon=epsilon)

            obs, info = env.reset(seed=run)

            for t in range(steps):

                action = epsilon_greedy_agent.select_action()

                obs, reward, termination, truncation, info = env.step(action)

                epsilon_greedy_agent.update(action, reward)

                avg_rewards[run, t] = reward
                optimal_actions[run, t] = info["optimal-arm"]

        mean_reward = avg_rewards.mean(axis=0)
        optimal_pct = optimal_actions.mean(axis=0) * 100

        plt.plot(mean_reward, label=f"ε={epsilon}", alpha =0.7)

    plt.title("Average Reward ε-Greedy")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.show()


    c_values = [0.5, 1.0, 2.0]
    runs = 1000
    steps = 2000

    for c_value in c_values:

        avg_rewards = np.zeros((runs, steps))
        optimal_actions = np.zeros((runs, steps))

        for run in range(runs):

            env = MultiArmedBandit(k=10, seed=run)
            ucb_agent = UCBAgent(k=10, c=c_value)

            obs, info = env.reset(seed=run)

            for t in range(steps):

                action = ucb_agent.select_action()

                obs, reward, termination, truncation, info = env.step(action)

                ucb_agent.update(action, reward)

                avg_rewards[run, t] = reward
                optimal_actions[run, t] = info["optimal-arm"]

        mean_reward = avg_rewards.mean(axis=0)
        optimal_pct = optimal_actions.mean(axis=0) * 100

        plt.plot(mean_reward, label=f"c={c_value}")

    plt.title("Average Reward UCB")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.show()