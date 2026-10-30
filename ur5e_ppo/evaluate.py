# -*- coding: utf-8 -*-
"""
评估脚本：测试训练好的 PPO 策略
================================

用法：
    python evaluate.py                              # 100 回合统计 + 3 回合 GIF
    python evaluate.py --episodes 50 --seed 7       # 换随机种子再测（检验泛化）
    python evaluate.py --viewer                     # 打开 MuJoCo 实时窗口肉眼看
    python evaluate.py --model runs/ppo_ur5e/final_model.zip

统计口径：
    success   = EE 距目标 < 5cm 时回合结束
    collision = 机械臂任何碰撞 geom 碰到障碍物
    timeout   = 200 步内既没成功也没撞
"""

import argparse
import os

import numpy as np

from stable_baselines3 import PPO
from ur5e_obstacle_env import UR5eObstacleEnv

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "runs", "ppo_ur5e", "best_model.zip")


def run_stats(model, env, n_episodes: int) -> dict:
    n_success = n_collision = n_timeout = 0
    rewards, dists, steps_list = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done, total_r = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            done = term or trunc
        rewards.append(total_r)
        dists.append(info["distance"])
        steps_list.append(env._elapsed)
        if info["success"]:
            n_success += 1
        elif info["collision"]:
            n_collision += 1
        else:
            n_timeout += 1
    return {
        "episodes": n_episodes,
        "success_rate": n_success / n_episodes,
        "collision_rate": n_collision / n_episodes,
        "timeout_rate": n_timeout / n_episodes,
        "avg_reward": float(np.mean(rewards)),
        "avg_final_dist": float(np.mean(dists)),
        "avg_steps": float(np.mean(steps_list)),
    }


def run_gif(model, env, gif_path: str, n_episodes: int = 4, seed: int = 999):
    """跑几回合并离屏渲染成 GIF（每个环境步取 1 帧，慢放展示）。"""
    import imageio
    frames = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
            done = term or trunc
    # 物理步长 10ms，用 50ms/帧 ≈ 5 倍慢放，看得清动作
    imageio.mimsave(gif_path, frames, duration=0.05, loop=0)
    print(f"演示 GIF 已保存: {gif_path}  ({len(frames)} 帧, ~{len(frames)*0.05:.1f}s)")


def run_viewer(model, difficulty: int):
    """实时窗口模式：肉眼看策略表现（需要本机 OpenGL 正常）。"""
    import mujoco.viewer
    env = UR5eObstacleEnv(difficulty=difficulty, seed=999)
    obs, _ = env.reset()
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        while v.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            v.sync()
            if term or trunc:
                obs, _ = env.reset()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--difficulty", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--viewer", action="store_true", help="打开实时 3D 窗口")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")

    if args.viewer:
        run_viewer(model, args.difficulty)
        return

    env_kwargs = dict(difficulty=args.difficulty)
    if args.seed is not None:
        env_kwargs["seed"] = args.seed
    env = UR5eObstacleEnv(**env_kwargs)

    stats = run_stats(model, env, args.episodes)
    print("\n========== 评估结果 ==========")
    for k, v in stats.items():
        print(f"{k:>16}: {v:.3f}" if isinstance(v, float) else f"{k:>16}: {v}")

    if not args.no_gif:
        try:
            gif_env = UR5eObstacleEnv(difficulty=args.difficulty, seed=999, render_mode="rgb_array")
            gif_env.reset()
            gif_path = os.path.join(HERE, "runs", "ppo_ur5e", "demo.gif")
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            run_gif(model, gif_env, gif_path)
        except Exception as e:
            print(f"[警告] 离屏渲染失败（可能是本机 OpenGL/虚拟显卡问题）: {e}")
            print("       统计结果不受影响；也可以加 --viewer 实时观看。")


if __name__ == "__main__":
    main()
