import os
import random
import csv
from collections import deque

import torch
import numpy as np
from datetime import datetime

import gym_stag_hunt
import gymnasium as gym

from algo.Utils import row2, convert_coords_to_tuples, row1, counterfactual_infer_model
from algo.agent.Q_social_net import Q_net
from algo.agent.ppo_agent_memory import PPO as PPO_Memory
from algo.fixed_policy import choice_action
from algo.opp_policy import sample_opponent_policy_three
from algo.strategy_infer_model import global_strategy_infer_stag_hunt


def compute_tft_reward(self_label, opp_last_label, w=1.0):
    if self_label == opp_last_label:
        return +w
    return -w


def total_reward(r_env, self_label, opp_last_label, lambda_tft=1.0, lambda_imp=0.1, violation=False):
    r_tft = compute_tft_reward(self_label, opp_last_label) if self_label is not None else 0.0
    r_imp = -1.0 if violation else 0.0
    return r_env + lambda_tft * r_tft + lambda_imp * r_imp


def main():
    env_name = "StagHunt-Hunt-v0"
    grid_size = (8, 8)
    max_ep_len = 500

    max_training_timesteps = int(3e7)
    print_freq = max_ep_len * 20
    save_model_freq = int(5e5)

    update_timestep = max_ep_len * 12
    K_epochs = 8
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 0.0003
    lr_critic = 0.001

    random_seed = 0

    ################ TFT hyperparameters ################
    lambda_tft = 10.0
    lambda_imp = 0.0

    WINDOW = 500
    coop_window_allc = deque(maxlen=WINDOW)
    coop_window_alld = deque(maxlen=WINDOW)

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

    USE_WANDB = True
    if USE_WANDB:
        import wandb
        wandb.init(
            project=env_name,
            tags=["Rule-based Opponent", "opp_allc", "opp_alld", "opp_tft", "Train Agent0"],
            name=f"TFT_r_{lambda_tft}_three",
            mode="online",
            config={
                "env": env_name,
                "eps_clip": eps_clip,
                "gamma": gamma,
                "update_timestep": update_timestep,
                "K_epochs": K_epochs,
                "max_ep_len": max_ep_len,
                "lr": lr_actor,
                "save_model_freq": save_model_freq,
                "random_seed": random_seed,
                "max_training_episodes": max_training_timesteps / max_ep_len,
                "max_training_timesteps": max_training_timesteps,
                "algorithm": "PPO",
                "trained_agent": "agent_0",
                "grid_size": grid_size,
                "r_align": lambda_tft,
                "r_imp": lambda_imp
            },
        )

    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    infer_model = Q_net(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    infer_model.load_model(
        "ppo_preTrain/stag_hunt/train_soical_net/StagHunt-Hunt-v0/agent_0/agent_0_PPO_StagHunt-Hunt-v0_0_10000000.pth"
    )

    directory = f"ppo_preTrain/stag_hunt/Three_strategies_to_tft_r_{lambda_tft}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    agent_0_dir = directory + '/' + env_name + '/agent_0/'
    if not os.path.exists(agent_0_dir):
        os.makedirs(agent_0_dir)

    agent_1_dir = directory + '/' + env_name + '/agent_1/'
    if not os.path.exists(agent_1_dir):
        os.makedirs(agent_1_dir)

    csv_path = os.path.join(directory, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    csv_fields = [
        "episode", "timestep", "collective_return",
        "agent0_avg_reward", "agent1_avg_reward",
        "agent0_avg_stag_cnt", "agent1_avg_stag_cnt",
        "agent0_avg_plants_cnt", "agent1_avg_plants_cnt",
        "agent0_avg_hunt_cnt", "agent1_avg_hunt_cnt",
        "coop_rate_allc", "coop_rate_alld",
        # ...
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    def write_csv_row(row: dict):
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(row)

    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
    print("--------------------------------------------------------------------------------------------")

    agent0 = PPO_Memory(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    print("============================================================================================")
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    agent0_print_running_reward = 0
    agent1_print_running_reward = 0
    agent0_print_stag_cnt = 0
    agent1_print_stag_cnt = 0
    agent0_print_plants_cnt = 0
    agent1_print_plants_cnt = 0
    agent0_print_hunt_cnt = 0
    agent1_print_hunt_cnt = 0

    time_step = 0
    i_episode = 0
    print_running_episodes = 0

    tmp_strategies_lst = []

    while time_step <= max_training_timesteps:
        obses, infos = env.reset()

        pos_info = convert_coords_to_tuples(infos)['entity_positions']

        agent0_current_ep_reward = 0
        agent1_current_ep_reward = 0
        agent0_current_ep_stag_cnt = 0
        agent1_current_ep_stag_cnt = 0
        agent0_current_ep_plants_cnt = 0
        agent1_current_ep_plants_cnt = 0
        agent0_current_ep_hunt_cnt = 0
        agent1_current_ep_hunt_cnt = 0

        agent0_last_strategy = 0  # start from cooperation
        agent1_last_strategy = 0
        agent0_curr_strategy = 0
        agent1_curr_strategy = 0

        agent0_history_strategies = []
        agent1_history_strategies = []

        agent0_obses = []
        agent1_obses = []
        agent0_actions = []
        agent1_actions = []

        agent0_pos_lst = [pos_info["agent_0"]]
        agent1_pos_lst = [pos_info['agent_1']]

        opp_cur_episode_policy = sample_opponent_policy_three()  # policy rollout: "allc", "alld", "tft"
        tmp_strategies_lst.append(opp_cur_episode_policy)

        for t in range(max_ep_len):
            agent0_action, _, agent0_action_log_prob = agent0.get_action(
                obses[0], agent1_last_strategy
            )

            agent1_action = choice_action(
                agent_policy=opp_cur_episode_policy,
                agent_id=1,
                pos_info=pos_info,
                opp_last=agent0_last_strategy
            )

            actions = [agent0_action, agent1_action]

            agent0_obses.append(obses[0])
            agent0_actions.append(agent0_action)
            agent1_obses.append(obses[1])
            agent1_actions.append(agent1_action)

            ne_obses, rewards, dones, truncateds, infos = env.step(actions)
            rewards = list(rewards)
            env_r0, env_r1 = rewards[0], rewards[1]

            pos_info = convert_coords_to_tuples(infos)["entity_positions"]
            stag_pos = pos_info["stag"]
            plants_pos = pos_info["plants"]

            agent0_pos_lst.append(pos_info["agent_0"])
            agent1_pos_lst.append(pos_info["agent_1"])

            if env_r0 < 0:
                agent0_current_ep_hunt_cnt += 1
            elif env_r0 == 1:
                agent0_current_ep_plants_cnt += 1
            elif env_r0 == 5:
                agent0_current_ep_stag_cnt += 1

            if env_r1 < 0:
                agent1_current_ep_hunt_cnt += 1
            elif env_r1 == 1:
                agent1_current_ep_plants_cnt += 1
            elif env_r1 == 5:
                agent1_current_ep_stag_cnt += 1

            trigger = (env_r0 + env_r1) > 0

            if trigger and len(agent0_pos_lst) > 1:
                # agent0_curr_strategy = global_strategy_infer_stag_hunt(
                #     0, agent0_pos_lst, stag_pos, plants_pos, rewards, agent0_history_strategies
                # )
                #
                # agent1_curr_strategy = global_strategy_infer_stag_hunt(
                #     1, agent1_pos_lst, stag_pos, plants_pos, rewards, agent1_history_strategies
                # )

                agent0_curr_strategy = counterfactual_infer_model(
                    agent0_obses, agent0_actions,
                    agent1_obses, agent1_actions,
                    infer_model, action_dim,
                )

                agent1_curr_strategy = counterfactual_infer_model(
                    agent1_obses, agent1_actions,
                    agent0_obses, agent0_actions,
                    infer_model, action_dim,
                )

                if rewards[0] == 5.0 and rewards[1] == 5.0:
                    agent0_curr_strategy = 0
                    agent1_curr_strategy = 0

                if rewards[0] == 1.0:
                    agent0_curr_strategy = 1

                if rewards[1] == 1.0:
                    agent1_curr_strategy = 1

                agent0_history_strategies.append(agent0_curr_strategy)
                agent1_history_strategies.append(agent1_curr_strategy)

                if opp_cur_episode_policy == "allc":
                    coop_window_allc.append(1 if env_r0 == 5 else 0)
                elif opp_cur_episode_policy == "alld":
                    coop_window_alld.append(1 if env_r0 == 5 else 0)

                rewards[0] = total_reward(
                    env_r0,
                    agent0_curr_strategy,
                    agent1_last_strategy,
                    lambda_tft=lambda_tft,
                    lambda_imp=lambda_imp,
                )

                rewards[1] = total_reward(
                    env_r1,
                    agent1_curr_strategy,
                    agent0_last_strategy,
                    lambda_tft=lambda_tft,
                    lambda_imp=lambda_imp,
                )

                agent0_obses = []
                agent1_obses = []
                agent0_actions = []
                agent1_actions = []

                agent0_pos_lst = [pos_info["agent_0"]]
                agent1_pos_lst = [pos_info["agent_1"]]

            agent0.buffer.add(
                obses[0],
                agent0_action,
                agent0_action_log_prob,
                rewards[0],
                ne_obses[0],
                dones,
                agent0_curr_strategy,
                agent1_last_strategy,
                agent1_curr_strategy
            )

            # agent1.buffer.add(
            #     obses[1],
            #     agent1_action,
            #     agent1_action_log_prob,
            #     sum(reward),
            #     ne_obses[1],
            #     dones,
            # )

            if trigger:
                agent0_last_strategy = agent0_curr_strategy
                agent1_last_strategy = agent1_curr_strategy

            obses = ne_obses
            time_step += 1

            agent0_current_ep_reward += env_r0
            agent1_current_ep_reward += env_r1

            if time_step % update_timestep == 0:
                agent0.update_net()
                # agent1.update_net()

            if time_step % print_freq == 0:
                agent0_print_avg_reward = round((agent0_print_running_reward / (print_running_episodes + 1e-9)), 2)
                agent1_print_avg_reward = round((agent1_print_running_reward / (print_running_episodes + 1e-9)), 2)

                agent0_print_avg_stag_cnt = round((agent0_print_stag_cnt / (print_running_episodes + 1e-9)), 2)
                agent1_print_avg_stag_cnt = round((agent1_print_stag_cnt / (print_running_episodes + 1e-9)), 2)

                agent0_print_avg_plants_cnt = round((agent0_print_plants_cnt / (print_running_episodes + 1e-9)), 2)
                agent1_print_avg_plants_cnt = round((agent1_print_plants_cnt / (print_running_episodes + 1e-9)), 2)

                agent0_print_avg_hunt_cnt = round((agent0_print_hunt_cnt / (print_running_episodes + 1e-9)), 2)
                agent1_print_avg_hunt_cnt = round((agent1_print_hunt_cnt / (print_running_episodes + 1e-9)), 2)

                print(f"\nEpisode: {i_episode + 1:<6}  Timestep: {time_step:.2e}")
                print(row2("avg_reward", agent0_print_avg_reward, agent1_print_avg_reward))
                print(row2("avg_stag_cnt", agent0_print_avg_stag_cnt, agent1_print_avg_stag_cnt))
                print(row2("avg_plants_cnt", agent0_print_avg_plants_cnt, agent1_print_avg_plants_cnt))
                print(row2("avg_hunt_cnt", agent0_print_avg_hunt_cnt, agent1_print_avg_hunt_cnt))

                print(row1("Opponent strategies lst:", tmp_strategies_lst))
                print(row1("Agent0 last_episode strategies lst:", agent0_history_strategies[-20:]))
                print(row1("Agent1 last_episode strategies lst:", agent1_history_strategies[-20:]))

                coop_rate_allc = sum(coop_window_allc) / max(1, len(coop_window_allc))
                coop_rate_alld = sum(coop_window_alld) / max(1, len(coop_window_alld))
                print(row1(f"coop_rate vs ALLC:", f"{coop_rate_allc:.2f}"))
                print(row1(f"coop_rate vs ALLD:", f"{coop_rate_alld:.2f}"))

                write_csv_row({
                    "episode": i_episode + 1,
                    "timestep": time_step,
                    "collective_return": round((agent0_print_avg_reward + agent1_print_avg_reward) / 2, 2),
                    "agent0_avg_reward": agent0_print_avg_reward,
                    "agent1_avg_reward": agent1_print_avg_reward,
                    "agent0_avg_stag_cnt": agent0_print_avg_stag_cnt,
                    "agent1_avg_stag_cnt": agent1_print_avg_stag_cnt,
                    "agent0_avg_plants_cnt": agent0_print_avg_plants_cnt,
                    "agent1_avg_plants_cnt": agent1_print_avg_plants_cnt,
                    "agent0_avg_hunt_cnt": agent0_print_avg_hunt_cnt,
                    "agent1_avg_hunt_cnt": agent1_print_avg_hunt_cnt,
                    "coop_rate_allc": round(coop_rate_allc, 3),
                    "coop_rate_alld": round(coop_rate_alld, 3),
                })

                if USE_WANDB:
                    wandb.log({
                        "global/episode": i_episode + 1,
                        "global/timestep": time_step,
                        "global/collective_return": round((agent0_print_avg_reward + agent1_print_avg_reward) / 2, 2),
                        "global/coop_rate_allc": coop_rate_allc,
                        "global/coop_rate_alld": coop_rate_alld,
                        "agent0/avg_reward": agent0_print_avg_reward,
                        "agent1/avg_reward": agent1_print_avg_reward,
                        "agent0/avg_stag_cnt": agent0_print_avg_stag_cnt,
                        "agent1/avg_stag_cnt": agent1_print_avg_stag_cnt,
                        "agent0/avg_plants_cnt": agent0_print_avg_plants_cnt,
                        "agent1/avg_plants_cnt": agent1_print_avg_plants_cnt,
                        "agent0/avg_hunt_cnt": agent0_print_avg_hunt_cnt,
                        "agent1/avg_hunt_cnt": agent1_print_avg_hunt_cnt,
                    }, step=time_step)

                # For logger init metircs
                agent0_print_running_reward = 0
                agent1_print_running_reward = 0
                agent0_print_stag_cnt = 0
                agent1_print_stag_cnt = 0
                agent0_print_plants_cnt = 0
                agent1_print_plants_cnt = 0
                agent0_print_hunt_cnt = 0
                agent1_print_hunt_cnt = 0
                print_running_episodes = 0

                tmp_strategies_lst = []

            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                agent_0_checkpoint_path = agent_0_dir + f"agent_0_PPO_{env_name}_{random_seed}_{time_step}.pth"
                print(f"agent_0 saving model at : {agent_0_checkpoint_path}")
                agent0.save_model(agent_0_checkpoint_path)

                # agent_1_checkpoint_path = agent_1_dir + f"agent_1_PPO_{env_name}_{random_seed}_{time_step}.pth"
                # print(f"agent_1 saving model at : {agent_1_checkpoint_path}")
                # agent1.save_model(agent_1_checkpoint_path)

                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            if dones or truncateds:
                break

        agent0_print_running_reward += agent0_current_ep_reward
        agent1_print_running_reward += agent1_current_ep_reward
        agent0_print_stag_cnt += agent0_current_ep_stag_cnt
        agent1_print_stag_cnt += agent1_current_ep_stag_cnt
        agent0_print_plants_cnt += agent0_current_ep_plants_cnt
        agent1_print_plants_cnt += agent1_current_ep_plants_cnt
        agent0_print_hunt_cnt += agent0_current_ep_hunt_cnt
        agent1_print_hunt_cnt += agent1_current_ep_hunt_cnt

        print_running_episodes += 1
        i_episode += 1

    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    main()
