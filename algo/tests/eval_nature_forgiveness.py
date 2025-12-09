import math
import random
from statistics import mean, stdev
from math import sqrt
from pprint import pprint
from datetime import datetime

import gymnasium as gym
import gym_stag_hunt

from algo.Utils import convert_coords_to_tuples
from algo.agent.ppo_agent_memory import PPO as PPO_Memory
from algo.fixed_policy import choice_action
from algo.strategy_infer_model import global_strategy_infer_stag_hunt


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

    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    agent0 = PPO_Memory(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    agent_0_checkpoint_path = "ppo_preTrain/stag_hunt/Three_strategies_to_tft_r_0.0/StagHunt-Hunt-v0/agent_0/agent_0_PPO_StagHunt-Hunt-v0_0_30000000.pth"

    agent0.load_model(agent_0_checkpoint_path)

    print("============================================================================================")
    start_time = datetime.now().replace(microsecond=0)
    print("Start forgiveness evaluation (Stag Hunt) at (GMT): ", start_time)
    print("============================================================================================")

    steps_lst = []
    total_ish_cnt = 0
    test_episodes = 100
    MIN_ISH_BEFORE_SWITCH = 10

    for ep in range(test_episodes):
        seed = random.randint(0, 10_000)
        obses, infos = env.reset(seed=seed)

        pos_info = convert_coords_to_tuples(infos)["entity_positions"]

        opp_strategy = "alld"
        change_flag = False
        last_defect_during_alld = False

        forgiveness_timer = 0
        forgiveness_recorded = False
        ep_ish_records = []

        agent0_last_strategy = 0
        agent1_last_strategy = 0
        agent0_history_strategies = [agent0_last_strategy]
        agent1_history_strategies = [agent1_last_strategy]

        agent0_pos_lst = [pos_info["agent_0"]]
        agent1_pos_lst = [pos_info["agent_1"]]
        ep_ish_cnt = 0

        for t in range(1, max_ep_len + 1):
            agent0_action, _, _ = agent0.get_action(
                obses[0], agent1_last_strategy
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
            stag_pos = pos_info["stag"]
            plants_pos = pos_info["plants"]

            agent0_pos_lst.append(pos_info["agent_0"])
            agent1_pos_lst.append(pos_info["agent_1"])

            ish_done = (env_r0 + env_r1) > 0

            if ish_done and len(agent0_pos_lst) > 1:
                ep_ish_cnt += 1
                total_ish_cnt += 1

                agent0_curr_strategy = global_strategy_infer_stag_hunt(
                    agent_id=0,
                    trajectory=agent0_pos_lst,
                    stag_pos=stag_pos,
                    plants_pos=plants_pos,
                    rewards=[env_r0, env_r1],
                    agent_history=agent0_history_strategies,
                    verbose=False,
                )

                agent1_curr_strategy = global_strategy_infer_stag_hunt(
                    agent_id=1,
                    trajectory=agent1_pos_lst,
                    stag_pos=stag_pos,
                    plants_pos=plants_pos,
                    rewards=[env_r0, env_r1],
                    agent_history=agent1_history_strategies,
                    verbose=False,
                )

                agent0_history_strategies.append(agent0_curr_strategy)
                agent1_history_strategies.append(agent1_curr_strategy)

                ep_ish_records.append(
                    (agent0_curr_strategy, agent1_curr_strategy, env_r0, env_r1)
                )

                if not change_flag and opp_strategy == "alld":
                    if agent0_curr_strategy == 1:
                        last_defect_during_alld = True

                if (not change_flag) and opp_strategy == "alld":
                    if ep_ish_cnt >= MIN_ISH_BEFORE_SWITCH and last_defect_during_alld:
                        change_flag = True
                        opp_strategy = "allc"
                        forgiveness_timer = 0
                        forgiveness_recorded = False
                        print(
                            f"[Episode {ep}] switch opponent to ALLC at timestep {t}, "
                            f"ep_ish_cnt={ep_ish_cnt}"
                        )

                if change_flag and (not forgiveness_recorded):
                    forgiveness_timer += 1

                    if agent0_curr_strategy == 0:
                        steps_lst.append(forgiveness_timer)
                        forgiveness_recorded = True
                        print(f"[Episode {ep}] forgiving at timestep {t}, forgiveness_timer={forgiveness_timer}")
                        break

                agent0_last_strategy = agent0_curr_strategy
                agent1_last_strategy = agent1_curr_strategy

                agent0_pos_lst = [pos_info["agent_0"]]
                agent1_pos_lst = [pos_info["agent_1"]]

            if dones or truncateds:
                break

            obses = ne_obses

        print(f"\n================ Episode {ep} summary ================")
        print(f"ISH count in this episode: {ep_ish_cnt}")
        print("First 20 ISHs (a0_label, a1_label, r0, r1):")
        pprint(ep_ish_records[:20])
        print("change_flag:", change_flag, "forgiveness_recorded:", forgiveness_recorded)

    print("\n================ Global forgiveness (delay) summary ================")
    print(f"Total ISH count                 : {total_ish_cnt}")
    print(f"Forgiveness delay samples (ISH) : {steps_lst}")
    n = len(steps_lst)
    print(f"n = {n}")

    if n == 0:
        print("No forgiveness event observed: after the opponent switched to cooperation, "
              "the agent never switched from defect(plants) to cooperate(stag) in these episodes.")
    elif n == 1:
        xbar = steps_lst[0]
        print("Only 1 forgiveness event observed.")
        print(f"Forgiveness delay = {xbar} ISH events after switch.")
    else:
        xbar = mean(steps_lst)
        s = stdev(steps_lst)
        se = s / sqrt(n)
        z = 1.96
        half = z * se

        print(f"Sample mean X̄ = {xbar:.4f} ISH events")
        print(f"Sample std s  = {s:.4f}")
        print(f"Std error SE  = {se:.4f}")
        print(f"95% CI        = [{xbar - half:.3f}, {xbar + half:.3f}]")

    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started evaluation at (GMT) : ", start_time)
    print("Finished evaluation at (GMT): ", end_time)
    print("Total evaluation time       : ", end_time - start_time)
    print("============================================================================================")


if __name__ == "__main__":
    main()
