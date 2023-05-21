env = gym.make('SpaceInvaders-v4', render_mode='human')
env.metadata['render_fps'] = 30
obs, info = env.reset()
done = False

while not done:
    action = trainer.compute_single_action(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
env.close()
