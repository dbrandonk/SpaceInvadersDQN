import gym
import torch
import ray
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print

# Define the configuration for the DQN agent
config = DEFAULT_CONFIG.copy()
config['num_workers'] = 1
config['num_gpus'] = 0
config['env'] = 'SpaceInvaders-v0'
config['model']['dim'] = 84
config['model']['conv_filters'] = [[32, 8, 4], [64, 4, 2], [64, 3, 1]]
config['model']['fcnet_hiddens'] = [256]
config['buffer_size'] = 10000
config['train_batch_size'] = 32
config['timesteps_per_iteration'] = 1000
config['learning_starts'] = 1000
config['exploration_fraction'] = 0.1
config['exploration_final_eps'] = 0.02
config['target_network_update_freq'] = 500

# Initialize Ray and create the DQNTrainer object
ray.init()
trainer = DQNTrainer(config=config)

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
