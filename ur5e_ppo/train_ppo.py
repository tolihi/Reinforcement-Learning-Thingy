# -*- coding: utf-8 -*-
"""
PPO 训练脚本：UR5e 避障到达
============================

用法（用 WorkBuddy 托管环境的 Python）：
    python train_ppo.py                        # 默认 20 万步
    python train_ppo.py --timesteps 400000     # 想练得更久
    python train_ppo.py --difficulty 2         # 虐待自己：大障碍挡必经之路

产物（runs/ppo_ur5e/）：
    best_model.zip   训练过程中评估奖励最高的模型（用这个）
    final_model.zip  训练结束时的模型
    monitor_*.csv    每回合的奖励/长度/成功/碰撞记录（画学习曲线用）
    learning_curve.png  学习曲线 + 成功率曲线
"""

import argparse
import glob
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from ur5e_obstacle_env import UR5eObstacleEnv

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(HERE, "runs", "ppo_ur5e")


def make_env(difficulty: int, seed: int, rank: int, monitor_dir: str):
    """工厂函数：每个并行环境各自 Monitor 包装、各自独立随机种子。"""
    def _init():
        env = UR5eObstacleEnv(difficulty=difficulty, seed=seed + rank)
        env = Monitor(env, os.path.join(monitor_dir, f"monitor_{rank}"),
                      info_keywords=("success", "collision"))
        return env
    set_random_seed(seed + rank)
    return _init


def plot_curves(monitor_dir: str, out_png: str):
    """从 monitor.csv 画学习曲线：回合奖励 + 滑动成功率。"""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = []
    # SB3 Monitor 生成的文件名是 monitor_<rank>.monitor.csv
    for f in sorted(glob.glob(os.path.join(monitor_dir, "monitor_*.monitor.csv"))):
        frames.append(pd.read_csv(f, skiprows=1))  # 第 1 行是 SB3 的元信息注释
    if not frames:
        print("没找到 monitor 日志，跳过画图")
        return
    df = pd.concat(frames, ignore_index=True).sort_values("t")
    df["reward_ma"] = df["r"].rolling(50, min_periods=5).mean()
    df["success_ma"] = df["success"].rolling(50, min_periods=5).mean()
    df["collision_ma"] = df["collision"].rolling(50, min_periods=5).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(df["t"], df["r"], alpha=0.25, lw=0.8, color="steelblue", label="episode reward")
    ax1.plot(df["t"], df["reward_ma"], lw=2, color="steelblue", label="MA(50)")
    ax1.set_ylabel("episode reward")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(df["t"], df["success_ma"], lw=2, color="seagreen", label="success rate MA(50)")
    ax2.plot(df["t"], df["collision_ma"], lw=2, color="indianred", label="collision rate MA(50)")
    ax2.set_ylabel("rate"); ax2.set_xlabel("timesteps")
    ax2.set_ylim(-0.05, 1.05); ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("UR5e obstacle avoidance — PPO learning curves")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"学习曲线已保存: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--difficulty", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--resume", type=str, default=None,
                    help="从指定 .zip 模型继续训练（课程学习/加练用）")
    ap.add_argument("--run-dir", type=str, default=RUN_DIR,
                    help="产物输出目录（续训时换个目录，避免覆盖上一段的日志）")
    args = ap.parse_args()

    run_dir = args.run_dir
    monitor_dir = os.path.join(run_dir, "monitor")
    os.makedirs(monitor_dir, exist_ok=True)

    # ---- 训练环境（4 个并行）+ 评估环境（独立、确定性）----
    from stable_baselines3.common.vec_env import DummyVecEnv
    train_env = DummyVecEnv([make_env(args.difficulty, args.seed, i, monitor_dir)
                             for i in range(args.n_envs)])
    eval_env = Monitor(UR5eObstacleEnv(difficulty=args.difficulty, seed=12345),
                       info_keywords=("success", "collision"))

    # ---- PPO 超参数（每个都值得你动手改一改看效果，见 README）----
    if args.resume:
        model = PPO.load(args.resume, env=train_env, device=args.device,
                         learning_rate=3e-4, ent_coef=0.008, verbose=1)
        print(f"从 {args.resume} 继续训练")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=512,            # 每个环境采 512 步 → 一次 rollout = n_envs×512
            batch_size=256,
            n_epochs=10,            # 同一批数据重复学 10 遍
            gamma=0.99,             # 折扣：未来的 1 分奖励今天值 0.99 分
            gae_lambda=0.95,
            clip_range=0.2,         # PPO 的灵魂：策略单次最多变化 ±20%
            ent_coef=0.008,         # 熵奖励：鼓励探索，防过早收敛
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[128, 128]),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        eval_freq=20_000,       # 每 2 万步评 20 回合
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    print(f"开始训练: {args.timesteps:,} 步, difficulty={args.difficulty}, "
          f"device={args.device}, n_envs={args.n_envs}")
    model.learn(total_timesteps=args.timesteps, callback=eval_cb, progress_bar=False)
    model.save(os.path.join(run_dir, "final_model"))
    print("训练完成")

    plot_curves(monitor_dir, os.path.join(run_dir, "learning_curve.png"))


if __name__ == "__main__":
    main()
