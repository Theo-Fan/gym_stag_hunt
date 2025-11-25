import os
import random
import csv
import torch
import numpy as np
from datetime import datetime

import gym_stag_hunt
import gymnasium as gym

from algo.Utils import row2
from algo.agent.ppo_agent import PPO as PPO_Basic
from algo.fixed_policy import choice_action


def main():
    env_name = "StagHunt-Hunt-v0"
    grid_size = (8, 8)
    max_ep_len = 500

    max_training_timesteps = int(1e7)
    print_freq = max_ep_len * 20
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
            tags=["PPO", "Train ALLD", "Rule-Base"],
            name=f"rule-base_train_alld",
            mode="online",
            config={
                "env": env_name,
                "eps_clip": eps_clip,
                "K_epochs": K_epochs,
                "max_ep_len": max_ep_len,
                "lr": lr_actor,
                "max_training_timesteps": max_training_timesteps,
                "grid_size": grid_size,
            },
        )

    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    directory = "ppo_preTrain/stag_hunt/rulebase_alld"
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
    # agent1 = PPO_Basic(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

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
    print_running_episodes = 0

    time_step = 0
    i_episode = 0

    while time_step <= max_training_timesteps:

        agent0_current_ep_reward = 0
        agent1_current_ep_reward = 0
        agent0_current_ep_stag_cnt = 0
        agent1_current_ep_stag_cnt = 0
        agent0_current_ep_plants_cnt = 0
        agent1_current_ep_plants_cnt = 0
        agent0_current_ep_hunt_cnt = 0
        agent1_current_ep_hunt_cnt = 0

        obses, infos = env.reset()

        for t in range(max_ep_len):
            agent0_action, _, agent0_action_log_prob = agent0.get_action(obses[0])

            agent1_action = choice_action(
                agent_policy="alld",
                agent_id=1,
                pos_info=infos,
            )

            actions = [agent0_action, agent1_action]

            ne_obses, reward, dones, truncateds, infos = env.step(actions)

            if reward[0] < 0:
                agent0_current_ep_hunt_cnt += 1
            elif reward[0] == 1:
                agent0_current_ep_plants_cnt += 1
            elif reward[0] == 5:
                agent0_current_ep_stag_cnt += 1

            if reward[1] < 0:
                agent1_current_ep_hunt_cnt += 1
            elif reward[1] == 1:
                agent1_current_ep_plants_cnt += 1
            elif reward[1] == 5:
                agent1_current_ep_stag_cnt += 1

            agent0.buffer.add(
                obses[0],
                agent0_action,
                agent0_action_log_prob,
                sum(reward),
                ne_obses[0],
                dones,
            )

            # agent1.buffer.add(
            #     obses[1],
            #     agent1_action,
            #     agent1_action_log_prob,
            #     sum(reward),
            #     ne_obses[1],
            #     dones,
            # )

            obses = ne_obses
            time_step += 1

            agent0_current_ep_reward += reward[0]
            agent1_current_ep_reward += reward[1]

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

                })

                if USE_WANDB:
                    wandb.log({
                        "global/episode": i_episode + 1,
                        "global/timestep": time_step,
                        "global/collective_return": round((agent0_print_avg_reward + agent1_print_avg_reward) / 2, 2),
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

            if dones:
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
