import sys

import gymnasium as gym
import gym_stag_hunt
import time

env = gym.make(
    "StagHunt-Hunt-v0",
    grid_size=(6, 6),
    screen_size=(600, 600),
    obs_type="image",
    enable_multiagent=True,
    stag_follows="follow",
    run_away_after_maul=True,
    forage_quantity=2,
    stag_reward=5,
    forage_reward=1,
    mauling_punishment=-2,
)  # you can pass config parameters here

obses, _ = env.reset()
for iteration in range(100):
    time.sleep(.2)
    actions = [env.action_space.sample(), env.action_space.sample()]
    obs, rewards, terminated, truncated, info = env.step(actions)

    print(f"timestep: {iteration} action:{actions} obs: {obs}, rewards: {rewards}, info: {info}")
    print(f"done: {terminated}, truncated: {truncated}\n")
    env.render()

env.close()
