import gymnasium as gym
import torch
import ray
from ray.rllib.algorithms.dqn import DQN, DEFAULT_CONFIG

# Define the configuration for the DQN agent
config = DEFAULT_CONFIG.copy()
config['num_workers'] = 1
config['num_gpus'] = 0
config['env'] = 'SpaceInvaders-v0'
config['replay_buffer_size'] = 10000
config['train_batch_size'] = 32
config['timesteps_per_iteration'] = 1000
config['num_steps_sampled_before_learning_starts'] = 1000
config['exploration_fraction'] = 0.1
config['exploration_final_eps'] = 0.02
config['target_network_update_freq'] = 500
config['framework'] = 'torch'

# Initialize Ray and create the DQNTrainer object
ray.init()
trainer = DQN(config=config)

# Train the agent for 100 iterations
for i in range(100):
    result = trainer.train()
    print(result)

# Use the trained agent to play the game
env = gym.make('SpaceInvaders-v0')
obs = env.reset()
done = False
while not done:
    action = trainer.compute_action(obs)
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
