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

    def add(self, obs, a1, a2, act_log_prob, reward, next_obs, done):
        self.buffer.append((obs, a1, a2, act_log_prob, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, a1, a2, act_log_prob, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs),
            np.array(a1, dtype=np.int64),  # [B, 2]
            np.array(a2, dtype=np.int64),  # [B, 2]
            np.array(act_log_prob),
            np.array(rewards),
            np.array(next_obs),
            np.array(dones),
        )

    def get_all(self):
        obs, a1, a2, act_log_prob, rewards, next_obs, dones = zip(*self.buffer)
        return (
            np.array(obs),
            np.array(a1, dtype=np.int64),  # [B, 2]
            np.array(a2, dtype=np.int64),  # [B, 2]
            np.array(act_log_prob),
            np.array(rewards),
            np.array(next_obs),
            np.array(dones),
        )

    def get_random_batch(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        sampled = [self.buffer[i] for i in indices]
        obs, a1, a2, act_log_prob, rewards, next_obs, dones = zip(*sampled)
        return (
            np.array(obs),
            np.array(a1, dtype=np.int64),
            np.array(a2, dtype=np.int64),
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
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_size + (self.action_dim ** 2), 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def reshape_cnn_input(self, obs):
        return obs.permute(0, 3, 1, 2)

    def forward(self):
        raise NotImplementedError


class Q_net:
    def __init__(self, obs_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.ac_net = ActorCritic(obs_dim, action_dim).to(device)
        self.ac_net_old = ActorCritic(obs_dim, action_dim).to(device)
        self.ac_net_old.load_state_dict(self.ac_net.state_dict())

        self.ac_optim = torch.optim.Adam(self.ac_net.parameters(), lr=lr_actor)

        self.MseLoss = nn.MSELoss()

        self.buffer = PPOBuffer(buffer_size=10000)

    def get_joint_onehot(self, a1: torch.LongTensor, a2: torch.LongTensor):
        a1 = a1.long().view(-1)
        a2 = a2.long().view(-1)
        A = self.ac_net.action_dim
        joint_idx = a1 * A + a2  # [B] LongTensor, 范围[0, A^2-1]
        joint_oh = F.one_hot(joint_idx, num_classes=A * A).float()  # [B, A^2]
        return joint_oh

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

    def get_q_value(self, obs: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor):
        joint_act_oh = self.get_joint_onehot(a1, a2)  # [B, action_dim^2]
        critic_in = torch.cat([obs, joint_act_oh], dim=1)  # [B, 64 + action_dim^2]
        q_value = self.ac_net.critic(critic_in)
        return q_value.squeeze()

    def eval_q(self, obs, a1: int, a2: int):
        self.ac_net.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # [1, obs_size]
            a1_t = torch.as_tensor([a1], dtype=torch.long, device=device)  # [1]
            a2_t = torch.as_tensor([a2], dtype=torch.long, device=device)  # [1]
            q = self.get_q_value(obs_t, a1_t, a2_t)
            return float(q.item())

    # def update_net(self):
    #     obses, a1, a2, old_log_probs, rewards, ne_obses, dones = self.buffer.get_all()
    #
    #     obses = torch.as_tensor(obses, dtype=torch.float32, device=device)  # [B, obs_size]
    #     self_actions = torch.as_tensor(a1, dtype=torch.long, device=device).view(-1)  # [B]
    #     other_actions = torch.as_tensor(a2, dtype=torch.long, device=device).view(-1)  # [B]
    #     old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=device).view(-1).detach()  # [B]
    #     rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).view(-1)  # [B]
    #     dones = torch.as_tensor(dones, dtype=torch.float32, device=device).view(-1)  # [B]
    #
    #     B = rewards.size(0)
    #     A = self.ac_net.action_dim
    #
    #     with torch.no_grad():
    #         returns = torch.zeros_like(rewards)  # [B]
    #         G = 0.0
    #         for t in reversed(range(B)):
    #             if dones[t] > 0.5:
    #                 G = 0.0
    #             G = rewards[t] + self.gamma * G
    #             returns[t] = G
    #         returns = returns.detach()
    #
    #         probs_old = self.ac_net_old.actor(obses)
    #         probs_old = torch.clamp(probs_old, 1e-8, 1.0)
    #         probs_old = probs_old / probs_old.sum(dim=-1, keepdim=True)
    #
    #         # enumerate all self actions
    #         all_self = torch.arange(A, device=device, dtype=torch.long).view(1, A).expand(B, A)  # [B, A]
    #         all_other = other_actions.view(B, 1).expand(B, A)  # [B, A]
    #
    #         # flatten to [B*A]
    #         a1_all = all_self.reshape(B * A)  # [B*A]
    #         a2_all = all_other.reshape(B * A)  # [B*A]
    #
    #         # repeat obs to match [B*A, obs_size]
    #         obs_rep = obses.unsqueeze(1).expand(B, A, obses.size(1)).reshape(B * A, obses.size(1))  # [B*A, obs_size]
    #
    #         # Q for all actions: [B*A] -> [B, A]
    #         q_all = self.get_q_value(obs_rep, a1_all, a2_all).view(B, A)  # [B, A]
    #
    #         v_baseline = (probs_old * q_all).sum(dim=1).detach()  # [B]
    #
    #         advantages = (returns - v_baseline).detach()
    #         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #
    #     # ========= 2) PPO 更新 =========
    #     total_loss = 0.0
    #     for _ in range(self.K_epochs):
    #         probs = self.ac_net.actor(obses)
    #         probs = torch.clamp(probs, 1e-8, 1.0)
    #         probs = probs / probs.sum(dim=-1, keepdim=True)
    #         dist = Categorical(probs)
    #
    #         log_probs = dist.log_prob(self_actions)  # [B]
    #         entropy = dist.entropy().mean()
    #
    #         ratios = torch.exp(log_probs - old_log_probs)  # [B]
    #
    #         surr1 = ratios * advantages
    #         surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
    #         policy_loss = -torch.min(surr1, surr2).mean()
    #
    #         values = self.get_q_value(obses, self_actions, other_actions)  # [B]
    #         value_loss = F.smooth_l1_loss(values, returns)
    #
    #         combined_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
    #
    #         self.ac_optim.zero_grad()
    #         combined_loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 0.5)
    #         self.ac_optim.step()
    #
    #         total_loss += combined_loss.item()
    #
    #     self.ac_net_old.load_state_dict(self.ac_net.state_dict())
    #     self.buffer.clear()
    #     return total_loss / self.K_epochs

    def update_net(self):
        obses, a1, a2, old_log_probs, rewards, ne_obses, dones = self.buffer.get_all()

        obses = torch.as_tensor(obses, dtype=torch.float32, device=device)  # [B, obs]
        ne_obses = torch.as_tensor(ne_obses, dtype=torch.float32, device=device)  # [B, obs]
        a1 = torch.as_tensor(a1, dtype=torch.long, device=device).view(-1)  # [B]
        a2 = torch.as_tensor(a2, dtype=torch.long, device=device).view(-1)  # [B]
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=device).view(-1).detach()
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).view(-1)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=device).view(-1)

        B = rewards.size(0)
        A = self.ac_net.action_dim
        not_done = (1.0 - dones).clamp(0.0, 1.0)

        # 最后一条没有 next_action（即使 next_obs 有），不要 bootstrap
        valid_next = torch.ones(B, device=device, dtype=torch.float32)
        if B > 0:
            valid_next[-1] = 0.0
        bootstrap_mask = not_done * valid_next  # [B]

        # --- helper: 用某个网络 net 计算 Q(s,a1,a2) ---
        def q_by_net(net, obs, aa1, aa2):
            joint_oh = self.get_joint_onehot(aa1, aa2)  # [B, A^2]
            critic_in = torch.cat([obs, joint_oh], dim=1)  # [B, obs + A^2]
            return net.critic(critic_in).squeeze(-1)  # [B]

        # ========= 1) no_grad: 计算 V_t(反事实baseline), GAE, 以及 critic TD target =========
        with torch.no_grad():
            # ---- (a) 计算 V_t = E_{a~pi_old} Q_old(s,(a,a2_t)) ----
            probs_old = self.ac_net_old.actor(obses)  # [B, A]
            probs_old = torch.clamp(probs_old, 1e-8, 1.0)
            probs_old = probs_old / probs_old.sum(dim=-1, keepdim=True)

            all_self = torch.arange(A, device=device, dtype=torch.long).view(1, A).expand(B, A)  # [B,A]
            all_other = a2.view(B, 1).expand(B, A)  # [B,A]

            obs_rep = obses.unsqueeze(1).expand(B, A, obses.size(1)).reshape(B * A, obses.size(1))
            a1_all = all_self.reshape(B * A)
            a2_all = all_other.reshape(B * A)

            q_all_old = q_by_net(self.ac_net_old, obs_rep, a1_all, a2_all).view(B, A)  # [B,A]
            V = (probs_old * q_all_old).sum(dim=1)  # [B]

            # ---- (b) 计算 V_{t+1}：需要 a2_{t+1}，用 shift 得到（并断开边界）----
            next_a2 = torch.zeros_like(a2)
            if B > 1:
                next_a2[:-1] = a2[1:]
            next_a2 = torch.where(dones > 0.5, torch.zeros_like(next_a2), next_a2)

            # 对 next state 也算 V_next = E_{a~pi_old(.|s')} Q_old(s',(a,next_a2))
            probs_old_next = self.ac_net_old.actor(ne_obses)
            probs_old_next = torch.clamp(probs_old_next, 1e-8, 1.0)
            probs_old_next = probs_old_next / probs_old_next.sum(dim=-1, keepdim=True)

            all_self_n = torch.arange(A, device=device, dtype=torch.long).view(1, A).expand(B, A)
            all_other_n = next_a2.view(B, 1).expand(B, A)
            ne_obs_rep = ne_obses.unsqueeze(1).expand(B, A, ne_obses.size(1)).reshape(B * A, ne_obses.size(1))
            a1_all_n = all_self_n.reshape(B * A)
            a2_all_n = all_other_n.reshape(B * A)

            q_all_old_next = q_by_net(self.ac_net_old, ne_obs_rep, a1_all_n, a2_all_n).view(B, A)
            V_next = (probs_old_next * q_all_old_next).sum(dim=1)  # [B]

            # ---- (c) GAE advantages: delta = r + gamma*V_next - V ----
            lam = 0.95
            advantages = torch.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(B)):
                delta = rewards[t] + self.gamma * bootstrap_mask[t] * V_next[t] - V[t]
                gae = delta + self.gamma * lam * bootstrap_mask[t] * gae
                advantages[t] = gae
                if dones[t] > 0.5:
                    gae = 0.0

            returns = (advantages + V).detach()  # 让 baseline V 回归 returns（稳定）
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ---- (d) critic TD target for Q(s,a1,a2): SARSA(0) using next (a1,a2) from rollout ----
            next_a1 = torch.zeros_like(a1)
            if B > 1:
                next_a1[:-1] = a1[1:]
            next_a1 = torch.where(dones > 0.5, torch.zeros_like(next_a1), next_a1)

            q_next_old = q_by_net(self.ac_net_old, ne_obses, next_a1, next_a2)  # [B]
            td_target_q = (rewards + self.gamma * bootstrap_mask * q_next_old).detach()

        # ========= 2) PPO update =========
        total_loss = 0.0
        for _ in range(self.K_epochs):
            probs = self.ac_net.actor(obses)
            probs = torch.clamp(probs, 1e-8, 1.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            dist = Categorical(probs)

            log_probs = dist.log_prob(a1)  # [B]
            entropy = dist.entropy().mean()

            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # --- critic loss: 让 Q(s,a1,a2) 拟合 td_target_q（用于动作区分） ---
            q_pred = self.get_q_value(obses, a1, a2)  # [B]
            value_loss = F.smooth_l1_loss(q_pred, td_target_q)

            combined_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.ac_optim.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 0.5)
            self.ac_optim.step()

            total_loss += float(combined_loss.item())

        self.ac_net_old.load_state_dict(self.ac_net.state_dict())
        self.buffer.clear()
        return total_loss / self.K_epochs

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
