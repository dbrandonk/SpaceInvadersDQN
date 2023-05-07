import gymnasium as gym
import ray
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.tune.logger import pretty_print

# Define the configuration for the DQN agent
config = DQNConfig()
config.environment('SpaceInvaders-v4')
config.framework('torch')

# Initialize Ray
ray.init()

# Create train object.
trainer = DQN(config=config)

# Train the agent for 100 iterations
for i in range(100):
    result = trainer.train()
    print(pretty_print(result))

# Use the trained agent to play the game
env = gym.make('SpaceInvaders-v0')
obs = env.reset()
done = False
while not done:
    action = trainer.compute_action(obs)
    obs, reward, done, info = env.step(action)
    env.render()
env.close()

