from abc import ABC

import numpy as np
# from gym import Env
from gymnasium import Env

from gym_stag_hunt.src.utils import print_matrix


class AbstractMarkovStagHuntEnv(Env, ABC):
    metadata = {
        "render_modes": ["human", "array"],
        "obs_types": ["image", "coords"],
        "render_fps": 4,
    }

    def __init__(self, grid_size=(5, 5), obs_type="image", enable_multiagent=False):
        """
        :param grid_size: A (W, H) tuple corresponding to the grid dimensions. Although W=H is expected, W!=H works also
        :param obs_type: Can be 'image' for pixel-array based observations, or 'coords' for just the entity coordinates
        """

        total_cells = grid_size[0] * grid_size[1]
        if total_cells < 3:
            raise AttributeError(
                "Grid is too small. Please specify a larger grid size."
            )
        if obs_type not in self.metadata["obs_types"]:
            raise AttributeError(
                'Invalid observation type provided. Please specify "image" or "coords"'
            )
        if grid_size[0] >= 255 or grid_size[1] >= 255:
            raise AttributeError(
                "Grid is too large. Please specify a smaller grid size."
            )

        super(AbstractMarkovStagHuntEnv, self).__init__()

        self.obs_type = obs_type
        self.done = False
        self.enable_multiagent = enable_multiagent

    # def step(self, actions):
    #     """
    #     Run one timestep of the environment's dynamics.
    #     :param actions: ints signifying actions for the agents. You can pass one, in which case the second agent does a
    #                     random move, or two, in which case each agent takes the specified action.
    #     :return: observation, rewards, is the game done, additional info
    #     """
    #     return self.game.update(actions)

    # def reset(self):
    #     """
    #     Reset the game state
    #     :return: initial observation
    #     """
    #     self.game.reset_entities()
    #     self.done = False
    #     return self.game.get_observation()

    def step(self, actions):
        result = self.game.update(actions)

        if len(result) == 3:
            obs, reward, done = result
            info = {}
        else:
            obs, reward, done, info = result

        # # ---- 处理多智能体 -> 单智能体 ----
        # if isinstance(obs, (tuple, list)):
        #     info["full_obs"] = obs
        #     obs = obs[0]
        #
        # if isinstance(reward, (tuple, list)):
        #     info["full_reward"] = reward
        #     reward = float(reward[0])

        obs = np.asarray(obs, dtype=self.observation_space.dtype)

        entity_positions = getattr(self.game, "ENTITY_POSITIONS", None)
        if entity_positions is not None:
            # 确保 info 是 dict
            info = dict(info) if info is not None else {}

            # 通用：整个 dict 都放进去，方便以后扩展/调试
            info["entity_positions"] = entity_positions

        terminated = bool(done)
        truncated = False

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        # gymnasium 规范
        super().reset(seed=seed)

        self.game.reset_entities()
        self.done = False

        base_obs = self.game.get_observation()

        enable_multi = getattr(self.game, "_enable_multiagent", False)
        obs_type = getattr(self.game, "_obs_type", None)

        if enable_multi:
            if obs_type == "coords":
                obs_0 = np.asarray(base_obs, dtype=self.observation_space.dtype)
                obs_1_raw = self.game._flip_coord_observation_perspective(base_obs)
                obs_1 = np.asarray(obs_1_raw, dtype=self.observation_space.dtype)

                obs = np.stack([obs_0, obs_1], axis=0)
            else:
                obs_0 = np.asarray(base_obs, dtype=self.observation_space.dtype)
                obs = np.stack([obs_0, obs_0], axis=0)  # 形状：(2, H, W, C)
        else:
            # 单智能体模式保持原样
            if isinstance(base_obs, (tuple, list)):
                base_obs = base_obs[0]
            obs = np.asarray(base_obs, dtype=self.observation_space.dtype)

        info = {}
        entity_positions = getattr(self.game, "ENTITY_POSITIONS", None)
        if entity_positions is not None:
            info["entity_positions"] = entity_positions

        return obs, info

    def render(self, mode="human", obs=None):
        """
        :param obs: observation data (passed for coord observations so we dont have to run the function twice)
        :param mode: rendering mode
        :return:
        """
        if mode == "human":
            if self.obs_type == "image":
                self.game.RENDERER.render_on_display()
            else:
                if self.game.RENDERER:
                    self.game.RENDERER.update()
                    self.game.RENDERER.render_on_display()
                else:
                    if obs is not None:
                        print_matrix(obs, self.game_title, self.game.GRID_DIMENSIONS)
                    else:
                        print_matrix(
                            self.game.get_observation(),
                            self.game_title,
                            self.game.GRID_DIMENSIONS,
                        )
        elif mode == "array":
            print_matrix(
                self.game._coord_observation(),
                self.game_title,
                self.game.GRID_DIMENSIONS,
            )

    def close(self):
        """
        Closes all needed resources
        :return:
        """
        if self.game.RENDERER:
            self.game.RENDERER.quit()
