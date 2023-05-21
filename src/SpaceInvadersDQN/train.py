import gymnasium as gym
import ray
from ray.rllib.algorithms.dqn import DQN, DQNConfig

config = DQNConfig()

config.environment('SpaceInvaders-v4')
config.framework('torch')

space_invader_model = {
    "fcnet_hiddens": [256, 256],
    "fcnet_activation": "relu",
    "framestack": True, # True enables 4x stacking behavior.
    "dim": 84, # Final resized frame dimension
}

config.training(
    gamma=0.99,
    lr=0.005,
    model=space_invader_model,
    train_batch_size=32,
    optimizer={
        'type': 'Adam'})

trainer = DQN(config=config)

ray.init()
num_episodes = 100
for episode in range(num_episodes):
    result = trainer.train()
    print("Completed Episode:{episode} of {num_episodes}")

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
