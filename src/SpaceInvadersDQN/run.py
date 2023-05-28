import gymnasium as gym
from ray.rllib.algorithms.dqn import DQN

env = gym.make('SpaceInvaders-v4', render_mode='human')
env.metadata['render_fps'] = 30
obs, info = env.reset()
done = False

dqn = DQN.from_checkpoint('checkpoint')

while not done:
    action = dqn.compute_single_action(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
env.close()
