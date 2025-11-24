import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ===================== device =====================
print("============================================================================================")
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print("Device set to :", torch.cuda.get_device_name(device))
else:
    print("Device set to : cpu")
print("============================================================================================")


def encode_opp_to_onehot(opp, device=device) -> torch.Tensor:
    """
    将对手上一段策略编码为 one-hot:
      'C'/0 -> [1,0],  'D'/1 -> [0,1]
    支持标量或一维列表/np.ndarray。返回形状 [B,2] 或 [1,2] 的 FloatTensor。
    """

    def _to_idx(x):
        if isinstance(x, str):
            x = x.strip().upper()
            if x == 'C': return 0
            if x == 'D': return 1
            return int(x)
        return int(x)

    if isinstance(opp, (list, tuple, np.ndarray)):
        idxs = torch.tensor([_to_idx(x) for x in opp], dtype=torch.long, device=device)
    else:
        idxs = torch.tensor([_to_idx(opp)], dtype=torch.long, device=device)

    return F.one_hot(idxs.clamp(0, 1), num_classes=2).float().to(device)


# ===================== Buffer =====================
class PPOBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, obs, action, act_log_prob, reward, next_obs, done,
            self_cur_strategy, opp_last_strategy, opp_cur_strategy):
        self.buffer.append((
            obs, action, act_log_prob, reward, next_obs, done,
            self_cur_strategy, opp_last_strategy, opp_cur_strategy
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        (obs, actions, act_log_prob, rewards, next_obs, dones,
         self_cur_strategy, opp_last_strategy, opp_cur_strategy) = zip(*batch)
        return (np.array(obs),
                np.array(actions),
                np.array(act_log_prob),
                np.array(rewards),
                np.array(next_obs),
                np.array(dones),
                np.array(self_cur_strategy),
                np.array(opp_last_strategy),
                np.array(opp_cur_strategy))

    def get_all(self):
        (obs, actions, act_log_prob, rewards, next_obs, dones,
         self_cur_strategy, opp_last_strategy, opp_cur_strategy) = zip(*self.buffer)
        return (np.array(obs),
                np.array(actions),
                np.array(act_log_prob),
                np.array(rewards),
                np.array(next_obs),
                np.array(dones),
                np.array(self_cur_strategy),
                np.array(opp_last_strategy),
                np.array(opp_cur_strategy))

    def clear(self):
        self.buffer.clear()


# ===================== Model =====================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, opp_dim: int = 2):
        super().__init__()

        # CNN：输入 (14, 11, 11)
        self.cnn = nn.Sequential(
            nn.Conv2d(obs_dim[-1], 32, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.Tanh(),
            nn.Flatten(),
            # nn.Linear(32 * 11 * 11, 64),  # coins (11, 11, 14)
            nn.Linear(32 * 6 * 6, 64), # coop mining (6, 6, 12)
        )
        cnn_out_dim = 64

        self.action_dim = action_dim
        self.opp_dim = opp_dim

        # Actor 
        self.actor = nn.Sequential(
            nn.Linear(cnn_out_dim + opp_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic 
        self.critic = nn.Sequential(
            nn.Linear(cnn_out_dim + opp_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    @staticmethod
    def reshape_cnn_input(obs: torch.Tensor) -> torch.Tensor:
        # [B, H, W, C] → [B, C, H, W]
        return obs.permute(0, 3, 1, 2)

    def forward(self):
        raise NotImplementedError


# ===================== PPO =====================
class PPO:
    def __init__(
        self,
        obs_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.ac_net = ActorCritic(obs_dim, action_dim).to(device)
        self.ac_net_old = ActorCritic(obs_dim, action_dim).to(device)
        self.ac_net_old.load_state_dict(self.ac_net.state_dict())

        self.ac_optim = torch.optim.Adam(self.ac_net.parameters(), lr=lr_actor)

        self.MseLoss = nn.MSELoss()
        self.buffer = PPOBuffer(buffer_size=10000)

    def get_action(self, obs, strategy, is_eval=False):
        obs_t = torch.FloatTensor(obs.copy()).unsqueeze(0).to(device)
        obs_t = self.ac_net_old.reshape_cnn_input(obs_t)
        with torch.no_grad():
            conv_obs = self.ac_net_old.cnn(obs_t)  # [1,64]
            opp_vec = encode_opp_to_onehot(strategy)  # [1,2]
            net_in = torch.cat([conv_obs, opp_vec], dim=1)  # [1,66]
            probs = self.ac_net_old.actor(net_in).squeeze(0)  # [A]
        dist = Categorical(probs)

        if is_eval:
            action = torch.argmax(probs)
        else:
            action = dist.sample()

        return action.item(), dist.probs.squeeze(), dist.log_prob(action).item()

    def update_net(self):

        (obses, actions, old_log_probs, rewards, ne_obses, dones,
         self_cur_strategy, opp_last_strategy, opp_cur_strategy) = self.buffer.get_all()

        obses = torch.FloatTensor(obses).to(device)
        ne_obses = torch.FloatTensor(ne_obses).to(device)
        actions = torch.LongTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        def cd_to_idx(arr) -> torch.Tensor:
            return torch.tensor([0 if (s in ['C', 'c', 0]) else 1 for s in arr],
                                dtype=torch.long, device=device)

        opp_prev = cd_to_idx(opp_last_strategy)  # [B]
        opp_next = cd_to_idx(opp_cur_strategy)

        opp_prev_vec = F.one_hot(opp_prev, num_classes=2).float()  # [B, 2]
        opp_next_vec = F.one_hot(opp_next, num_classes=2).float()

        with torch.no_grad():
            feat_s = self.ac_net.reshape_cnn_input(obses)
            feat_s = self.ac_net.cnn(feat_s)  # [B, 64]
            net_in = torch.cat([feat_s, opp_prev_vec], dim=1)
            values = self.ac_net.critic(net_in).squeeze(-1)  # [B]

            feat_n = self.ac_net.reshape_cnn_input(ne_obses)
            feat_n = self.ac_net.cnn(feat_n)
            net_in_next = torch.cat([feat_n, opp_next_vec], dim=1)
            next_values = self.ac_net.critic(net_in_next).squeeze(-1)  # [B]

            advantages = []
            gae = 0.0
            lam = 0.95
            T = len(rewards)
            for t in reversed(range(T)):
                delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * lam * (1 - dones[t]) * gae
                advantages.insert(0, gae)
                if dones[t]:
                    gae = 0.0
            advantages = torch.tensor(advantages, dtype=torch.float32, device=device)

        returns = advantages + values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.K_epochs):
            # -------- forward actor / critic --------
            feat_s = self.ac_net.reshape_cnn_input(obses)
            feat_s = self.ac_net.cnn(feat_s)  # [B, 64]
            net_in = torch.cat([feat_s, opp_prev_vec], dim=1)  # [B, 66]

            probs = self.ac_net.actor(net_in)  # [B, A]
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)  # [B]
            cur_values = self.ac_net.critic(net_in).squeeze(-1)  # [B]

            # -------- PPO clipped objective --------
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(cur_values, returns)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

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

    def save_model(self, path: str) -> None:
        torch.save({
            'ac_net': self.ac_net.state_dict(),
            'ac_net_old': self.ac_net_old.state_dict()
        }, path)

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=device)
        self.ac_net.load_state_dict(checkpoint['ac_net'])
        self.ac_net_old.load_state_dict(checkpoint['ac_net_old'])
        print(f"Model loaded from {path}")
