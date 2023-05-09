import gymnasium as gym
import ray
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.tune.logger import pretty_print

# Define the configuration for the DQN agent
config = DQNConfig()

config.environment('SpaceInvaders-v4')
config.framework('torch')

config.training(gamma=0.99, lr=0.005, train_batch_size=32, optimizer={'type':'Adam'})

# Initialize Ray
ray.init()

# Create train object.
trainer = DQN(config=config)

# Train the agent
for i in range(1):
    result = trainer.train()
    print(pretty_print(result))

# Use the trained agent to play the game
env = gym.make('SpaceInvaders-v4', render_mode='human')
env.metadata['render_fps'] = 30
obs, info = env.reset()
done = False

while not done:
    action = trainer.compute_single_action(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
env.close()
