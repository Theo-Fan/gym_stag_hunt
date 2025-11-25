import time
from typing import Dict, List, Tuple

import gymnasium as gym
import gym_stag_hunt

from algo.Utils import convert_coords_to_tuples
from algo.fixed_policy import choice_action
from algo.strategy_infer_model import global_strategy_infer_stag_hunt

env = gym.make(
    "StagHunt-Hunt-v0",
    grid_size=(5, 5),
    screen_size=(600, 600),
    obs_type="image",  # coords or image
    enable_multiagent=True,
    run_away_after_maul=False,
    forage_quantity=2,
    stag_reward=5,
    forage_reward=1,
    mauling_punishment=-0.000001,
)

num_episodes = 1
max_steps_per_episode = 100

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

    for t in range(max_steps_per_episode):
        agent0_action = choice_action(
            agent_policy="allc",
            agent_id=0,
            pos_info=pos_info,
        )
        agent1_action = choice_action(
            agent_policy="alld",
            agent_id=1,
            pos_info=pos_info,
            opp_last=agent0_last_strategy
        )

        actions = [agent0_action, agent1_action]
        obs, rewards, terminated, truncated, info = env.step(actions)

        pos_info = convert_coords_to_tuples(info)["entity_positions"]

        stag_pos = pos_info["stag"]
        plants_pos = pos_info["plants"]

        agent0_pos_lst.append(pos_info["agent_0"])
        agent1_pos_lst.append(pos_info["agent_1"])
        print(f"rewards at step {t}: {rewards}")

        # time.sleep(0.2)
        # env.render()

        ish_done = sum(rewards) > 0
        if ish_done:
            ish_count += 1

            agent0_curr_strategy = global_strategy_infer_stag_hunt(
                0, agent0_pos_lst, stag_pos, plants_pos, rewards, agent0_strategy_history, verbose=False
            )

            agent1_curr_strategy = global_strategy_infer_stag_hunt(
                1, agent1_pos_lst, stag_pos, plants_pos, rewards, agent1_strategy_history, verbose=False
            )

            agent0_strategy_history.append(agent0_curr_strategy)
            agent1_strategy_history.append(agent1_curr_strategy)

            agent0_pos_lst = [pos_info["agent_0"]]
            agent1_pos_lst = [pos_info["agent_1"]]

        if ish_done:
            agent0_last_strategy = agent0_curr_strategy
            agent1_last_strategy = agent1_curr_strategy

    print(f"================= Episode {ep} =================")
    print(f"Number of ISH done: {ish_count} length of strategy history: {len(agent0_strategy_history)}")
    print(f"agent0 strategy history: {agent0_strategy_history}")
    print(f"agent1 strategy history: {agent1_strategy_history}\n")

env.close()
