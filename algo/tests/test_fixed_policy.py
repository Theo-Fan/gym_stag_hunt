import sys

import gymnasium as gym
import gym_stag_hunt
import time

from algo.Utils import convert_coords_to_tuples
from algo.fixed_policy import choice_action

env = gym.make(
    "StagHunt-Hunt-v0",
    grid_size=(5, 5),
    screen_size=(600, 600),
    obs_type="image",  # coords or image
    enable_multiagent=True,
    opponent_policy="random",
    # stag_follows="static",
    run_away_after_maul=True,
    forage_quantity=2,
    stag_reward=5,
    forage_reward=1,
    mauling_punishment=-2,
)  # you can pass config parameters here

obses, info = env.reset()

for iteration in range(500):
    agent0_action = choice_action(
        agent_policy="alld",
        agent_id=0,
        pos_info=convert_coords_to_tuples(info),
    )

    agent1_action = choice_action(
        agent_policy="allc",
        agent_id=1,
        pos_info=convert_coords_to_tuples(info),
    )

    actions = [agent0_action, agent1_action]  # env.action_space.sample()
    obs, rewards, terminated, truncated, info = env.step(actions)

    print(
        f"timestep: {iteration} action:{actions}, rewards: {rewards}, "
        f"done: {terminated}, truncated: {truncated}, info: {convert_coords_to_tuples(info)}"
    )

    env.render()
    time.sleep(.2)

env.close()
