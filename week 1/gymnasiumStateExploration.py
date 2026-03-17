import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self):
        return self.action_space.sample()


def inspect_space(space, name):
    print(f"{name}: {space}")

    if isinstance(space, spaces.Discrete):
        print("Type: Discrete")
        print("n:", space.n)

    elif isinstance(space, spaces.Box):
        print("Type: Box")
        print("shape:", space.shape)
        print("low:", space.low)
        print("high:", space.high)


def inspect_environment(env_name):
    env = gym.make(env_name)
    print(f"\n=== {env_name} ===")
    inspect_space(env.observation_space, "Observation Space")
    inspect_space(env.action_space, "Action Space")
    env.close()


def evaluate_random_agent(env_name, episodes=100):
    env = gym.make(env_name)
    agent = RandomAgent(env.action_space)

    episode_rewards = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)

    env.close()

    episode_rewards = np.array(episode_rewards)
    print(f"\nPerformance on {env_name}")
    print("Average reward:", episode_rewards.mean())
    print("Std:", episode_rewards.std())
    print("Min:", episode_rewards.min())
    print("Max:", episode_rewards.max())

if __name__ == "__main__":
    inspect_environment("FrozenLake-v1")
    inspect_environment("Taxi-v3")

    evaluate_random_agent("FrozenLake-v1", episodes=100)
    evaluate_random_agent("Taxi-v3", episodes=100)