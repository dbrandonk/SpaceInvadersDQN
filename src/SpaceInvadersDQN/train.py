import ray
from ray.rllib.algorithms.dqn import DQNConfig

config = DQNConfig()

config.environment(env='SpaceInvaders-v4')
config.framework('torch')
config.resources(num_learner_workers=8)

space_invader_model = {
    "fcnet_hiddens": [512, 512, 512],
    "fcnet_activation": "relu",
    "framestack": True,  # True enables 4x stacking behavior.
    "dim": 84,  # Final resized frame dimension
}

exploration_config = {
    'type': 'EpsilonGreedy',
    'initial_epsilon': 1.0,
    'final_epsilon': 0.01,
    'epsilon_timesteps': 100000}
config.exploration(explore=True, exploration_config=exploration_config)

config.checkpointing(export_native_model_files=True)

config.training(
    gamma=0.99,
    lr=0.005,
    model=space_invader_model,
    train_batch_size=32,
    optimizer={
        'type': 'Adam'})

dqn = config.build()

ray.init()
NUM_EPISODES = 200
CHECKPOINT_RATE = 10
for episode in range(NUM_EPISODES):
    result = dqn.train()
    print(f"Completed Episode:{episode} of {NUM_EPISODES}")

    if episode % CHECKPOINT_RATE == 0:
        dqn.save()

