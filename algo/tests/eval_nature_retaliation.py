import math
import random
from datetime import datetime

import gymnasium as gym
import gym_stag_hunt

from algo.Utils import convert_coords_to_tuples
from algo.agent.ppo_agent_memory import PPO as PPO_Memory
from algo.fixed_policy import choice_action


def main():
    env_name = "StagHunt-Hunt-v0"
    grid_size = (8, 8)
    max_ep_len = 500

    K_epochs = 80
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

    STAG_REWARD = 5.0
    PLANT_REWARD = 1.0

    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    agent0 = PPO_Memory(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    agent_0_checkpoint_path = "save_model/agent0/TFT_model.pth"
    agent0.load_model(agent_0_checkpoint_path)

    print("============================================================================================")
    start_time = datetime.now().replace(microsecond=0)
    print("Start 'retaliation' evaluation (plants share) at (GMT): ", start_time)
    print("============================================================================================")

    total_self_positive = 0
    total_self_plants = 0
    total_self_stag = 0
    opp_strategy = "alld"
    test_episodes = 20

    for ep in range(test_episodes):
        seed = random.randint(0, 10_000)
        obses, infos = env.reset(seed=seed)

        pos_info = convert_coords_to_tuples(infos)["entity_positions"]

        agent0_last_strategy = 0
        agent1_last_strategy = 0

        ep_self_positive = 0
        ep_self_plants = 0
        ep_self_stag = 0

        for t in range(max_ep_len):
            agent0_action, _, _ = agent0.get_action(
                obses[0],
                agent1_last_strategy,
            )

            agent1_action = choice_action(
                agent_policy=opp_strategy,
                agent_id=1,
                pos_info=pos_info,
                opp_last=agent0_last_strategy,
            )

            actions = [agent0_action, agent1_action]

            ne_obses, rewards, dones, truncateds, infos = env.step(actions)
            rewards = list(rewards)
            env_r0, env_r1 = rewards[0], rewards[1]

            pos_info = convert_coords_to_tuples(infos)["entity_positions"]

            if env_r0 > 0:
                total_self_positive += 1
                ep_self_positive += 1

                if abs(env_r0 - PLANT_REWARD) < 1e-6:
                    total_self_plants += 1
                    ep_self_plants += 1
                elif abs(env_r0 - STAG_REWARD) < 1e-6:
                    total_self_stag += 1
                    ep_self_stag += 1
                else:
                    print(f"[Warn] Unexpected positive reward for agent0: {env_r0}")

            if env_r0 == STAG_REWARD:
                agent0_last_strategy = 0
            elif env_r0 == PLANT_REWARD:
                agent0_last_strategy = 1

            if env_r1 == STAG_REWARD:
                agent1_last_strategy = 0
            elif env_r1 == PLANT_REWARD:
                agent1_last_strategy = 1

            if dones or truncateds:
                break

            obses = ne_obses

        if ep_self_positive > 0:
            ep_plants_ratio = ep_self_plants / ep_self_positive
        else:
            ep_plants_ratio = float('nan')

        print(f"\n================ Episode {ep} summary ================")
        print(f"episode self positive cnt: {ep_self_positive}")
        print(f"episode self plants cnt  : {ep_self_plants}")
        print(f"episode self stag   cnt  : {ep_self_stag}")
        print(f"episode plants share (plants / positive): {ep_plants_ratio:.3f}")

    print("\n================ Global 'retaliation' (plants share) summary ================")
    print(f"Total self positive reward count : {total_self_positive}")
    print(f"Total self plants-reward count   : {total_self_plants}")
    print(f"Total self stag-reward count     : {total_self_stag}")

    if total_self_positive > 0:
        p_plants = total_self_plants / total_self_positive
        sd = math.sqrt(p_plants * (1 - p_plants))
        se = sd / math.sqrt(total_self_positive)
        half = 1.96 * se

        print(f"\nRetaliation metric (plants share) p = {p_plants:.4f}")
        print(f"SD  = {sd:.6f}")
        print(f"SE  = {se:.6f}")
        print(f"95% CI (Wald) ≈ [{p_plants - half:.4f}, {p_plants + half:.4f}]")
    else:
        print("\nAgent0 never got positive reward – cannot compute plants share.")

    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started evaluation at (GMT) : ", start_time)
    print("Finished evaluation at (GMT): ", end_time)
    print("Total eval time             : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    main()
