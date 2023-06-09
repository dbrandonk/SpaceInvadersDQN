"""
Configures and trains a Deep Q-Network (DQN) agent using Ray's RLlib library.
The agent is trained on the 'SpaceInvaders-v4' environment.
The configuration includes settings for the DQN's exploration strategy, replay buffer, and etc.
"""

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
    'final_epsilon': 0.1,
    'epsilon_timesteps': 500000}
config.exploration(explore=True, exploration_config=exploration_config)

config.checkpointing(checkpoint_trainable_policies_only=True)

replay_buffer_config = {
    'type': 'MultiAgentPrioritizedReplayBuffer',
    'capacity': 100000,
    'replay_sequence_length': 1}

config.training(
    double_q=True,
    dueling=False,
    gamma=0.99,
    lr=0.0005,
    model=space_invader_model,
    n_step=1,
    noisy=False,
    num_atoms=1,
    optimizer={'type': 'Adam'},
    replay_buffer_config=replay_buffer_config,
    target_network_update_freq=10000,
    td_error_loss_fn='huber',
    train_batch_size=32,
    training_intensity=None
)

dqn = config.build()

ray.init()
NUM_EPISODES = 2000
CHECKPOINT_RATE = 50
for episode in range(NUM_EPISODES):
    result = dqn.train()
    print(f"Completed Episode:{episode} of {NUM_EPISODES}")

    if episode % CHECKPOINT_RATE == 0:
        dqn.save()
