# -*- coding: utf-8 -*-
"""
UR5e 避障到达环境（Gymnasium 接口）
====================================

任务定义（一句话版）：
    机械臂从 home 位形出发，把末端执行器(EE)移动到绿色目标球（距离 < 5cm 算成功），
    途中不能碰到红色球形障碍物（碰到 = 失败，回合终止）。

MDP 五要素：
    状态 S : 关节角(6) + 关节速度(6) + EE位置(3) + EE→目标向量(3)
             + 4个臂上关键点到障碍物的向量(12) + 障碍物半径(1)  → 共 31 维
    动作 A : 6 维，每维 ∈ [-1,1]，映射为 6 个关节位置控制目标的增量
    奖励 R : 见 _reward_info() 内的逐项注释
    终止   : 成功(EE 距目标<0.05m) / 撞障碍物 / 超时(150 步)

控制方式：
    ur5e.xml 里的 <general> 执行器 gainprm=2000, biasprm="0 -2000 -400"，
    展开就是  F = 2000*(ctrl - q) - 400*qdot  —— 一个刚度 2000、阻尼 400 的
    关节位置 PD 伺服。所以我们只需给"目标关节角" ctrl，底层力矩自动算好。
"""

import os
import numpy as np
import gymnasium as gym
import mujoco
from gymnasium import spaces

# 默认场景文件：本项目工作区内的副本（不改动你 menagerie 原件）
DEFAULT_SCENE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "universal_robots_ur5e", "ur5e_ppo_scene.xml",
)


class UR5eObstacleEnv(gym.Env):
    """UR5e 避障到达环境。

    Args:
        scene_xml: MuJoCo 场景文件路径。
        frame_skip: 每个环境步内仿真多少个物理步（控制周期 = frame_skip * 2ms）。
        max_episode_steps: 每回合最多环境步数。
        render_mode: "rgb_array"（离屏渲染，用于录像/可视化）或 None。
        difficulty: 0=简单(障碍小、离路径近), 1=中等, 2=难(障碍大、必经之路上)。
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        scene_xml: str = DEFAULT_SCENE,
        frame_skip: int = 5,
        max_episode_steps: int = 200,
        render_mode: str | None = None,
        difficulty: int = 1,
        seed: int | None = None,
    ):
        super().__init__()

        self.scene_xml = scene_xml
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.difficulty = difficulty

        # ---- MuJoCo 模型 ----
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        # 常用 id 缓存（用名字查 id，避免魔法数字）
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.obstacle_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle")
        self.target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # 臂上用于观测障碍物相对位置的 body（避开纯 EE 观测导致"手臂撞了但策略看不见"）
        self.obs_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")
        ]

        self.n_joints = 6
        # 关节位置控制目标的单步最大增量（弧度）。太大→动作抖，太小→到得太慢
        self.ctrl_delta = 0.25

        # home 关键帧的 EE 位置（reset 后正运动学算一次，作为采样目标的基准）
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.home_ee = self.data.site_xpos[self.ee_site_id].copy()
        self.home_qpos = self.data.qpos.copy()

        # ---- Gymnasium 空间 ----
        obs_dim = 6 + 6 + 3 + 3 + 12 + 1  # 31
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_joints,), dtype=np.float64)

        # 回合状态
        self._elapsed = 0
        self._dist_prev = None

        # 离屏渲染器（惰性创建，训练时不渲染不吃性能）
        self._renderer = None

    # ------------------------------------------------------------------ #
    #  目标 / 障碍物采样
    # ------------------------------------------------------------------ #
    def _sample_task(self):
        """随机采样本回合的目标点和障碍物（位置+半径）。"""
        # 目标点：以 home EE 为基准的箱型扰动，保证可达且不贴地
        target = self.home_ee + np.array([
            self.rng.uniform(0.10, 0.45),   # 向前（机器人前方 x+）
            self.rng.uniform(-0.30, 0.30),  # 左右
            self.rng.uniform(-0.15, 0.20),  # 上下
        ])
        target[2] = max(target[2], 0.15)  # 别钻到桌面以下

        # ★ 可达性约束：UR5e 水平臂展约 0.85m，留安全余量把目标压到 [0.30, 0.68]。
        #   否则会采到物理上够不着的点，那些回合永远不可能成功，白白污染训练信号。
        r_xy = np.linalg.norm(target[:2])
        if r_xy > 0.68:
            target[:2] *= 0.68 / r_xy
        elif r_xy < 0.30:
            target[:2] *= 0.30 / max(r_xy, 1e-6)
        target[2] = np.clip(target[2], 0.15, 0.65)

        # 障碍物半径：难度越高越大
        if self.difficulty == 0:
            r = self.rng.uniform(0.035, 0.055)
        elif self.difficulty == 1:
            r = self.rng.uniform(0.045, 0.075)
        else:
            r = self.rng.uniform(0.055, 0.09)

        # 障碍物位置：放在 home→target 连线附近（必经之路），加横向偏移
        for _ in range(50):
            t = self.rng.uniform(0.35, 0.65)  # 沿路径的参数
            center = (1 - t) * self.home_ee + t * target
            # 垂直于路径方向的随机横向偏移：难度高→偏移小→更挡路
            offset = self.rng.uniform(0.0, 0.13 if self.difficulty >= 2 else 0.18)
            theta = self.rng.uniform(0, 2 * np.pi)
            # 在 z 平面内做横向扰动（简化：竖直平面内偏移）
            obstacle = center + np.array([0.0, offset * np.cos(theta), offset * np.sin(theta)])
            obstacle[2] = max(obstacle[2], r + 0.05)  # 不与地面相交
            # 约束：离起点/目标都留出安全余量，保证任务可解
            if (np.linalg.norm(obstacle - target) > r + 0.09
                    and np.linalg.norm(obstacle - self.home_ee) > r + 0.08):
                return target, obstacle, r
        # 兜底（理论上几乎不会走到）
        return target, center, r

    def _apply_task(self, target, obstacle, radius):
        """把采样结果写进模型（geom 直接挂在世界坐标系下，可运行时改）。"""
        self.target = target
        self.model.geom_pos[self.target_geom_id] = target
        self.model.geom_pos[self.obstacle_geom_id] = obstacle
        self.model.geom_size[self.obstacle_geom_id] = [radius, 0, 0]

    # ------------------------------------------------------------------ #
    #  Gymnasium 标准接口
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # 起始位形加小扰动，防止策略只会死记一条轨迹
        self.data.qpos[:] = self.home_qpos + self.rng.uniform(-0.05, 0.05, self.n_joints)
        self.data.ctrl[:] = self.data.qpos[:]

        target, obstacle, radius = self._sample_task()
        self._apply_task(target, obstacle, radius)

        mujoco.mj_forward(self.model, self.data)
        self._elapsed = 0
        self._dist_prev = self._ee_distance()
        return self._get_obs(), {}

    def step(self, action):
        # 1) 动作 → 关节位置控制目标（增量式）
        ctrl = self.data.ctrl + self.ctrl_delta * np.clip(action, -1, 1)
        lo = self.model.actuator_ctrlrange[:, 0]
        hi = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(ctrl, lo, hi)

        # 2) 物理仿真 frame_skip 步（每步 2ms）
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._elapsed += 1
        reward, info = self._reward_info(action)
        terminated = info["collision"] or info["success"]
        truncated = (not terminated) and self._elapsed >= self.max_episode_steps
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  观测 / 奖励 / 碰撞
    # ------------------------------------------------------------------ #
    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def _ee_distance(self) -> float:
        return float(np.linalg.norm(self.ee_pos - self.target))

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos[: self.n_joints].copy()
        qvel = self.data.qvel[: self.n_joints].copy() / 5.0  # 粗略缩放，量级拉到 O(1)
        ee = self.ee_pos
        rel_target = self.target - ee
        # 4 个臂上点到障碍物的相对向量：让策略"看得见"障碍物与手臂的距离
        obs_pos = self.data.xpos[self.obs_body_ids]           # (4,3)
        rel_obstacle = (obs_pos - self.model.geom_pos[self.obstacle_geom_id]).reshape(-1)
        radius = float(self.model.geom_size[self.obstacle_geom_id, 0])
        return np.concatenate([qpos, qvel, ee, rel_target, rel_obstacle, [radius]]).astype(np.float64)

    def _check_collision(self) -> bool:
        """障碍物与机械臂任意碰撞 geom 是否发生接触。"""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if self.obstacle_geom_id in (g1, g2):
                other = g2 if g1 == self.obstacle_geom_id else g1
                if other != self.floor_geom_id and other != self.target_geom_id:
                    return True
        return False

    def _reward_info(self, action) -> tuple[float, dict]:
        d = self._ee_distance()
        collision = self._check_collision()
        success = d < 0.05

        # -------- 奖励各项（改这里 = 改任务，看 README 的调参表） --------
        # (1) 进度奖励：靠近目标就加分。势能差形式，只奖励"净前进"
        r_progress = 4.0 * (self._dist_prev - d)
        # (2) 距离惩罚：离目标越远每步扣得越多，催它快点到
        r_dist = -0.05 * d
        # (3) 时间惩罚：活着就有成本，避免磨蹭
        r_time = -0.02
        # (4) 动作代价：抑制大幅度乱甩，让轨迹平滑省力
        r_action = -0.005 * float(np.sum(np.square(action)))
        # (5) 终止事件
        r_success = 20.0 if success else 0.0
        r_collision = -20.0 if collision else 0.0

        reward = r_progress + r_dist + r_time + r_action + r_success + r_collision
        self._dist_prev = d

        info = {
            "success": success,
            "collision": collision,
            "distance": d,
            "reward_progress": r_progress,
            "reward_terminal": r_success + r_collision,
        }
        return float(reward), info

    # ------------------------------------------------------------------ #
    #  渲染（离屏）
    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ---------------------------------------------------------------------- #
#  快速自检：python ur5e_obstacle_env.py
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    env = UR5eObstacleEnv(seed=0)
    obs, _ = env.reset()
    print(f"观测维度: {obs.shape}, 期望: {env.observation_space.shape}")
    print(f"home EE 位置: {env.home_ee.round(3)}")
    print(f"目标点    : {env.target.round(3)}")
    print(f"障碍物    : {env.model.geom_pos[env.obstacle_geom_id].round(3)} "
          f"r={env.model.geom_size[env.obstacle_geom_id, 0]:.3f}")

    # 随机动作跑一回合，验证环境无崩溃
    total_r, steps = 0.0, 0
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total_r += r
        steps += 1
        done = term or trunc
    print(f"随机策略一回合: {steps} 步, 总奖励 {total_r:.2f}, "
          f"结果: {'碰撞' if info['collision'] else ('成功' if info['success'] else '超时')}")

    # 粗测仿真速度（决定训练要多久）
    import time
    env.reset()
    t0 = time.time()
    n = 2000
    for _ in range(n):
        env.step(env.action_space.sample())
        if env._elapsed >= env.max_episode_steps:
            env.reset()
    fps = n * env.frame_skip / (time.time() - t0)
    print(f"仿真速度: {fps:,.0f} 物理步/秒  (单环境)")
