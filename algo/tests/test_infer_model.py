import time
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

import gym_stag_hunt

from algo.Utils import convert_coords_to_tuples, counterfactual_infer_model
from algo.agent.Q_social_net import Q_net
from algo.fixed_policy import choice_action
from algo.strategy_infer_model import global_strategy_infer_stag_hunt

env_name = "StagHunt-Hunt-v0"
grid_size = (8, 8)
max_ep_len = 100

K_epochs = 8

eps_clip = 0.2
gamma = 0.99

lr_actor = 0.0003
lr_critic = 0.001

env = gym.make(
    id=env_name,
    grid_size=grid_size,
    screen_size=(600, 600),
    obs_type="coords",
    enable_multiagent=True,
    run_away_after_maul=False,
    forage_quantity=2,
    stag_reward=5,
    forage_reward=1,
    mauling_punishment=-1e-4,
)

state_dim = env.observation_space.shape
action_dim = env.action_space.n

infer_model = Q_net(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
infer_model.load_model(
"ppo_preTrain/stag_hunt/train_soical_net/StagHunt-Hunt-v0/agent_0/agent_0_PPO_StagHunt-Hunt-v0_0_30000000.pth"
)
# ppo_preTrain/stag_hunt/train_soical_net/StagHunt-Hunt-v0/agent_0/agent_0_PPO_StagHunt-Hunt-v0_0_15000000.pth

num_episodes = 1

for ep in range(num_episodes):
    obses, info = env.reset()
    pos_info = convert_coords_to_tuples(info)['entity_positions']

    agent0_pos_lst = [pos_info["agent_0"]]
    agent1_pos_lst = [pos_info["agent_1"]]

    agent0_last_strategy = 0  # start from cooperation
    agent1_last_strategy = 0
    agent0_curr_strategy = 0
    agent1_curr_strategy = 0

    agent0_strategy_history = []
    agent1_strategy_history = []
    ish_count = 0

    agent0_obses = []
    agent1_obses = []
    agent0_actions = []
    agent1_actions = []

    agent0_infer_res_lst = []

    opp_strategy = np.random.choice(["allc", "alld"])

    for t in range(max_ep_len):
        agent0_action = choice_action(
            agent_policy="allc",
            agent_id=0,
            pos_info=pos_info,
        )

        agent1_action = choice_action(
            agent_policy=opp_strategy,
            agent_id=1,
            pos_info=pos_info,
        )

        actions = [agent0_action, agent1_action]

        agent0_obses.append(obses[0])
        agent0_actions.append(agent0_action)
        agent1_obses.append(obses[1])
        agent1_actions.append(agent1_action)

        obses, rewards, terminated, truncated, info = env.step(actions)

        pos_info = convert_coords_to_tuples(info)["entity_positions"]

        stag_pos = pos_info["stag"]
        plants_pos = pos_info["plants"]

        agent0_pos_lst.append(pos_info["agent_0"])
        agent1_pos_lst.append(pos_info["agent_1"])

        ish_done = sum(rewards) > 0
        if ish_done:
            ish_count += 1

            agent0_curr_strategy = global_strategy_infer_stag_hunt(
                0, agent0_pos_lst, stag_pos, plants_pos, rewards, agent0_strategy_history, verbose=False
            )

            agent1_curr_strategy = global_strategy_infer_stag_hunt(
                1, agent1_pos_lst, stag_pos, plants_pos, rewards, agent1_strategy_history, verbose=False
            )

            agent0_infer_res = counterfactual_infer_model(
                agent0_obses, agent0_actions,
                agent1_obses, agent1_actions,
                infer_model, action_dim,
                t, verbose=True
            )

            print(f"agent0 infer model result: {agent0_infer_res}, "
                  f"global strategy infer result: {agent0_curr_strategy}\n"
                  f"agent1 global strategy infer result: {agent1_curr_strategy}")
            print(f"reward: {rewards}\n\n")

            agent0_infer_res_lst.append(agent0_infer_res)
            agent0_strategy_history.append(agent0_curr_strategy)
            agent1_strategy_history.append(agent1_curr_strategy)

            agent0_obses = []
            agent1_obses = []
            agent0_actions = []
            agent1_actions = []

            agent0_pos_lst = [pos_info["agent_0"]]
            agent1_pos_lst = [pos_info["agent_1"]]

        if ish_done:
            agent0_last_strategy = agent0_curr_strategy
            agent1_last_strategy = agent1_curr_strategy

    print(f"================= Episode {ep} =================")
    print(f"Number of ISH done: {ish_count} length of strategy history: {len(agent0_strategy_history)}")
    print(f"agent0 infer model results: \t{agent0_infer_res_lst}")
    print(f"agent0 strategy history: \t{agent0_strategy_history}")
    print(f"agent1 strategy history: \t{agent1_strategy_history}\n")

env.close()
