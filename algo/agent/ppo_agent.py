import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

##### ====> set device
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


class PPOBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, obs, action, act_log_prob, reward, next_obs, done):
        self.buffer.append((obs, action, act_log_prob, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, actions, act_log_prob, rewards, next_obs, dones = zip(*batch)
        return np.array(obs), np.array(actions), np.array(act_log_prob), np.array(rewards), np.array(
            next_obs), np.array(dones)

    def get_all(self):
        obs, actions, act_log_prob, rewards, next_obs, dones = zip(*self.buffer)
        return np.array(obs), np.array(actions), np.array(act_log_prob), np.array(rewards), np.array(
            next_obs), np.array(dones)

    def get_random_batch(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))

        # get index in buffer
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)

        sampled = [self.buffer[i] for i in indices]
        obs, actions, act_log_prob, rewards, next_obs, dones = zip(*sampled)

        return (
            np.array(obs),
            np.array(actions),
            np.array(act_log_prob),
            np.array(rewards),
            np.array(next_obs),
            np.array(dones),
        )

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()

        # cnn: input size
        self.cnn = nn.Sequential(
            nn.Conv2d(obs_dim[-1], 32, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(32 * 11 * 11, 64),
        )

        cnn_out_dim = 64  # Output dimension after CNN layers
        self.action_dim = action_dim
        obs_size = obs_dim[0]

        # actor
        self.actor = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),  # 注意这里是 64 -> action_dim，不再是 32
            nn.Softmax(dim=-1)
        )

        # === critic：V(s) ===
        self.critic = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def reshape_cnn_input(self, obs):
        return obs.permute(0, 3, 1, 2)

    def forward(self):
        raise NotImplementedError


class PPO:
    def __init__(self, obs_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # self.optimizer = torch.optim.Adam([
        #     {'params': self.policy.actor.parameters(), 'lr': lr_actor},
        #     {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        # ])

        self.ac_net = ActorCritic(obs_dim, action_dim).to(device)
        self.ac_net_old = ActorCritic(obs_dim, action_dim).to(device)
        self.ac_net_old.load_state_dict(self.ac_net.state_dict())

        self.ac_optim = torch.optim.Adam(self.ac_net.parameters(), lr=lr_actor)

        self.MseLoss = nn.MSELoss()

        self.buffer = PPOBuffer(buffer_size=10000)

    def get_action(self, obs, is_eval=False):
        obs = torch.FloatTensor(obs.copy()).unsqueeze(0).to(device)
        with torch.no_grad():
            actor_out = self.ac_net_old.actor(obs)
        dist = Categorical(actor_out)
        probs = dist.probs.squeeze()

        if is_eval:
            action = torch.argmax(probs)
        else:
            action = dist.sample()

        return action.item(), probs, dist.log_prob(action).item()

    def get_q_value(self, obs, action):
        # act_one_hot = F.one_hot(action, num_classes=self.ac_net.action_dim).float()
        # features = torch.concat([obs, act_one_hot], dim=1)
        q_value = self.ac_net.critic(obs)
        return q_value.squeeze()

    def update_net(self):
        obses, actions, old_log_probs, rewards, ne_obses, dones = self.buffer.get_all()

        obses = torch.FloatTensor(obses).to(device)
        actions = torch.LongTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        ne_obses = torch.FloatTensor(ne_obses).to(device)
        dones = torch.FloatTensor(dones).to(device)

        with torch.no_grad():
            # get values
            old_values = self.ac_net.critic(obses).squeeze()      # [T]
            next_values = self.ac_net.critic(ne_obses).squeeze()  # [T]

            # calculate advantages using GAE
            advantages = []
            gae = 0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - old_values[t]
                gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae  # lambda: 0.95
                advantages.insert(0, gae)
                if dones[t]:
                    gae = 0.0

            advantages = torch.tensor(advantages, dtype=torch.float32).to(device)

        returns = advantages + old_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.K_epochs):
            probs = self.ac_net.actor(obses)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)

            values = self.ac_net.critic(obses).squeeze()

            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, returns)

            entropy = dist.entropy().mean()

            combined_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.ac_optim.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 0.5)
            self.ac_optim.step()

            total_loss += combined_loss.item()

        self.ac_net_old.load_state_dict(self.ac_net.state_dict())
        self.buffer.clear()

        avg_loss = total_loss / self.K_epochs
        return avg_loss

    def save_model(self, path):
        torch.save({
            'ac_net': self.ac_net.state_dict(),
            'ac_net_old': self.ac_net_old.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.ac_net.load_state_dict(checkpoint['ac_net'])
        self.ac_net_old.load_state_dict(checkpoint['ac_net_old'])
        print(f"Model loaded from {path}")
