import os
import random
import sys
import csv
import torch
from pathlib import Path
import numpy as np
from datetime import datetime

import gym_stag_hunt
import gymnasium as gym

from algo.agent.ppo_agent import PPO as PPO_Basic


def main():
    env_name = "StagHunt-Hunt-v0"
    max_ep_len = 500
    max_training_timesteps = int(3e7)

    print_freq = max_ep_len * 10
    log_freq = max_ep_len * 2

    save_model_freq = int(5e5)

    update_timestep = max_ep_len * 12
    K_epochs = 8

    eps_clip = 0.2
    gamma = 0.99

    lr_actor = 0.0003
    lr_critic = 0.001

    random_seed = 0

    env = gym.make(
        env_name,
        grid_size=(5, 5),
        screen_size=(600, 600),
        obs_type="coords",
        enable_multiagent=True,
        stag_follows=False,
        run_away_after_maul=True,
        forage_quantity=2,
        stag_reward=5,
        forage_reward=1,
        mauling_punishment=-2,
    )
    USE_WANDB = False
    if USE_WANDB:
        import wandb
        wandb.init(
            project=env_name,
            tags=["PPO", "Train Basic", "ALLC"],
            name=f"PPO_Train_Basic",
            mode="online",
            config={
                "env": env_name,
                "eps_clip": eps_clip,
                "K_epochs": K_epochs,
                "max_ep_len": max_ep_len,
                "lr": lr_actor,
                "max_training_timesteps": max_training_timesteps
            },
        )

    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    directory = "ppo_preTrain/stag_hunt/basic"
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
        # "agent0_avg_gold_mining_cnt", "agent1_avg_gold_mining_cnt",
        # "agent0_avg_iron_mining_cnt", "agent1_avg_iron_mining_cnt",
        # "agent0_avg_mining_total_ore", "agent1_avg_mining_total_ore",
        # "agent0_avg_gold_mining_ratio", "agent1_avg_gold_mining_ratio",
        # "agent0_avg_iron_mining_ratio", "agent1_avg_iron_mining_ratio",
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
    print("log frequency : " + str(log_freq) + " timesteps")
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

    agent0 = PPO_Basic(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    agent1 = PPO_Basic(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    print("============================================================================================")
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    agent_0_print_running_reward = 0
    agent_0_print_running_gold_cnt = 0
    agent_0_print_running_iron_cnt = 0

    agent_1_print_running_reward = 0
    agent_1_print_running_gold_cnt = 0
    agent_1_print_running_iron_cnt = 0

    print_running_episodes = 0

    time_step = 0
    i_episode = 0

    while time_step <= max_training_timesteps:

        obses, _ = env.reset()

        agent_0_current_ep_reward = 0
        agent_0_current_ep_gold_ore = 0
        agent_0_current_ep_iron_ore = 0

        agent_1_current_ep_reward = 0
        agent_1_current_ep_gold_ore = 0
        agent_1_current_ep_iron_ore = 0

        for t in range(1, max_ep_len + 1):
            agent0_action, _, agent0_action_log_prob = agent0.get_action(obses[0])
            agent1_action, _, agent1_action_log_prob = agent1.get_action(obses[1])
            actions = [agent0_action, agent1_action]

            ne_obses, reward, dones, truncateds, infos = env.step(actions)

            agent0.buffer.add(
                obses[0],
                agent0_action,
                agent0_action_log_prob,
                reward[0],
                ne_obses[0],
                dones,
            )

            agent1.buffer.add(
                obses[1],
                agent1_action,
                agent1_action_log_prob,
                reward[1],
                ne_obses[1],
                dones,
            )

            obses = ne_obses
            time_step += 1

            # agent_0_current_ep_reward += reward["agent_0"]
            # agent_0_current_ep_gold_ore += infos['agent_0']['is_mining_gold']
            # agent_0_current_ep_iron_ore += infos['agent_0']['is_mining_iron']
            #
            # agent_1_current_ep_reward += reward["agent_1"]
            # agent_1_current_ep_gold_ore += infos['agent_1']['is_mining_gold']
            # agent_1_current_ep_iron_ore += infos['agent_1']['is_mining_iron']

            if time_step % update_timestep == 0:
                agent0.update_net()
                agent1.update_net()
                print("okok")

            if time_step % print_freq == 0:
                agent_0_print_avg_reward = round((agent_0_print_running_reward / (print_running_episodes + 1e-9)), 2)
                agent_1_print_avg_reward = round((agent_1_print_running_reward / (print_running_episodes + 1e-9)), 2)



                print(f"Episode: {i_episode + 1:<6}  Timestep: {time_step:.2e}")
                # print(row2("avg_reward", agent_0_print_avg_reward, agent_1_print_avg_reward))
                # print(row2("avg_gold_mining_cnt",
                #            round((agent_0_print_running_gold_cnt / (print_running_episodes + 1e-9)), 2),
                #            round((agent_1_print_running_gold_cnt / (print_running_episodes + 1e-9)), 2),
                #            ))
                # print(row2("avg_iron_mining_cnt",
                #            round((agent_0_print_running_iron_cnt / (print_running_episodes + 1e-9)), 2),
                #            round((agent_1_print_running_iron_cnt / (print_running_episodes + 1e-9)), 2),
                #            ))
                # print(row2("avg_mining_total_ore",
                #            round((agent_0_total_mining / (print_running_episodes + 1e-9)), 2),
                #            round((agent_1_total_mining / (print_running_episodes + 1e-9)), 2),
                #            ))
                # print(row2("avg_gold_mining_ratio", agent_0_avg_gold_mining_ratio, agent_1_avg_gold_mining_ratio))
                # print(row2("avg_iron_mining_ratio", agent_0_avg_iron_mining_ratio, agent_1_avg_iron_mining_ratio))

                # write_csv_row({
                #     "episode": i_episode + 1,
                #     "timestep": time_step,
                #     "collective_return": round((agent_0_print_avg_reward + agent_1_print_avg_reward) / 2, 2),
                #     "agent0_avg_reward": agent_0_print_avg_reward,
                #     "agent1_avg_reward": agent_1_print_avg_reward,
                #     "agent0_avg_gold_mining_cnt": round(
                #         (agent_0_print_running_gold_cnt / (print_running_episodes + 1e-9)), 2),
                #     "agent1_avg_gold_mining_cnt": round(
                #         (agent_1_print_running_gold_cnt / (print_running_episodes + 1e-9)), 2),
                #     "agent0_avg_iron_mining_cnt": round(
                #         (agent_0_print_running_iron_cnt / (print_running_episodes + 1e-9)), 2),
                #     "agent1_avg_iron_mining_cnt": round(
                #         (agent_1_print_running_iron_cnt / (print_running_episodes + 1e-9)), 2),
                #     "agent0_avg_mining_total_ore": round(
                #         (agent_0_total_mining / (print_running_episodes + 1e-9)), 2),
                #     "agent1_avg_mining_total_ore": round(
                #         (agent_1_total_mining / (print_running_episodes + 1e-9)), 2),
                #     "agent0_avg_gold_mining_ratio": agent_0_avg_gold_mining_ratio,
                #     "agent1_avg_gold_mining_ratio": agent_1_avg_gold_mining_ratio,
                #     "agent0_avg_iron_mining_ratio": agent_0_avg_iron_mining_ratio,
                #     "agent1_avg_iron_mining_ratio": agent_1_avg_iron_mining_ratio,
                # })

                if USE_WANDB:
                    wandb.log({
                        "episode": i_episode + 1,
                        "timestep": time_step,
                        "collective_return": round((agent_0_print_avg_reward + agent_1_print_avg_reward) / 2, 2),
                        "agent0/avg_reward": agent_0_print_avg_reward,
                        "agent1/avg_reward": agent_1_print_avg_reward,
                    }, step=time_step)

                # For logger init metircs
                agent_0_print_running_reward = 0
                agent_0_print_running_iron_cnt = 0
                agent_0_print_running_gold_cnt = 0

                agent_1_print_running_reward = 0
                agent_1_print_running_iron_cnt = 0
                agent_1_print_running_gold_cnt = 0

                print_running_episodes = 0

            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                agent_0_checkpoint_path = agent_0_dir + f"agent_0_PPO_{env_name}_{random_seed}_{time_step}.pth"
                print(f"agent_0 saving model at : {agent_0_checkpoint_path}")
                agent0.save_model(agent_0_checkpoint_path)

                agent_1_checkpoint_path = agent_1_dir + f"agent_1_PPO_{env_name}_{random_seed}_{time_step}.pth"
                print(f"agent_1 saving model at : {agent_1_checkpoint_path}")
                agent1.save_model(agent_1_checkpoint_path)

                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            if dones:
                break

        agent_0_print_running_reward += agent_0_current_ep_reward
        agent_0_print_running_gold_cnt += agent_0_current_ep_gold_ore
        agent_0_print_running_iron_cnt += agent_0_current_ep_iron_ore

        agent_1_print_running_reward += agent_1_current_ep_reward
        agent_1_print_running_gold_cnt += agent_1_current_ep_gold_ore
        agent_1_print_running_iron_cnt += agent_1_current_ep_iron_ore
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
