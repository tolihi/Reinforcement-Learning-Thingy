"""A small, self-contained PPO example for the MuJoCo Shadow Hand scene."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


DEFAULT_MODEL = Path(
    r"D:\CS\code\.venv\training\mujoco_menagerie-main\shadow_hand\scene_right.xml"
)
OUT_DIR = Path(__file__).resolve().parent / "runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ShadowHandGraspEnv:
    """Grasp-and-lift task with five learned finger synergies.

    The XML has 20 position actuators. The policy controls five synergy values
    (thumb, index, middle, ring, little), while the wrist stays neutral. This
    keeps the first PPO exercise small enough to train and understand.
    """

    FINGER_TIP_BODIES = (
        "rh_thdistal",
        "rh_ffdistal",
        "rh_mfdistal",
        "rh_rfdistal",
        "rh_lfdistal",
    )
    ACTUATOR_GROUPS = {
        "thumb": ("THJ5", "THJ4", "THJ3", "THJ2", "THJ1"),
        "index": ("FFJ4", "FFJ3", "FFJ0"),
        "middle": ("MFJ4", "MFJ3", "MFJ0"),
        "ring": ("RFJ4", "RFJ3", "RFJ0"),
        "little": ("LFJ5", "LFJ4", "LFJ3", "LFJ0"),
    }
    CLOSED_TARGETS = {
        "thumb": (-0.35, 0.85, 0.0, 0.40, 0.65),
        "index": (0.16, 1.05, 2.30),
        "middle": (0.16, 1.05, 2.30),
        "ring": (0.16, 1.05, 2.30),
        "little": (0.35, 0.16, 1.05, 2.30),
    }

    def __init__(self, model_path: Path, frame_skip: int = 10, horizon: int = 160):
        self.model_path = Path(model_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.horizon = horizon
        self.dt = self.model.opt.timestep * frame_skip
        self.n_actions = 5

        self.object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        if self.object_body_id < 0:
            raise ValueError("The model scene must contain a body named 'object'.")
        free_joint_ids = [
            j for j in range(self.model.njnt)
            if self.model.jnt_bodyid[j] == self.object_body_id
            and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if len(free_joint_ids) != 1:
            raise ValueError("Expected one free joint on the scene's object body.")
        self.object_qpos_adr = int(self.model.jnt_qposadr[free_joint_ids[0]])
        self.object_qvel_adr = int(self.model.jnt_dofadr[free_joint_ids[0]])
        self.object_geom_ids = [
            g for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] == self.object_body_id
        ]
        self.tip_body_ids = [
            self._name_id(mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.FINGER_TIP_BODIES
        ]

        self.actuator_ids: dict[str, list[int]] = {}
        for finger, suffixes in self.ACTUATOR_GROUPS.items():
            self.actuator_ids[finger] = [
                self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"rh_A_{suffix}")
                for suffix in suffixes
            ]

        self.open_ctrl = np.zeros(self.model.nu, dtype=np.float64)
        self.closed_ctrl = np.zeros(self.model.nu, dtype=np.float64)
        # Neutral wrist; fingers start open at their zero-angle reference.
        for finger, suffixes in self.ACTUATOR_GROUPS.items():
            for actuator_id, suffix, value in zip(
                self.actuator_ids[finger], suffixes, self.CLOSED_TARGETS[finger]
            ):
                self.closed_ctrl[actuator_id] = value
        self.closed_ctrl = np.clip(
            self.closed_ctrl, self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

        self.step_count = 0
        self.start_object_z = 0.1
        self.prev_action = np.zeros(self.n_actions, dtype=np.float32)
        self.success_steps = 0

    def _name_id(self, kind: mujoco.mjtObj, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, kind, name)
        if idx < 0:
            raise ValueError(f"Model is missing expected MuJoCo object: {name}")
        return idx

    def reset(self, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        # Slight position variation gives the policy a little robustness.
        qadr = self.object_qpos_adr
        self.data.qpos[qadr:qadr + 3] = (
            0.30 + rng.uniform(-0.008, 0.008),
            rng.uniform(-0.008, 0.008),
            0.10 + rng.uniform(-0.003, 0.003),
        )
        self.data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.ctrl[:] = self.open_ctrl
        mujoco.mj_forward(self.model, self.data)
        self.start_object_z = float(self.data.xpos[self.object_body_id, 2])
        self.step_count = 0
        self.prev_action.fill(0)
        self.success_steps = 0
        return self._observation()

    def _observation(self) -> np.ndarray:
        object_pos = self.data.xpos[self.object_body_id]
        object_vel = self.data.qvel[self.object_qvel_adr:self.object_qvel_adr + 3]
        distances = [
            np.linalg.norm(self.data.xpos[body_id] - object_pos)
            for body_id in self.tip_body_ids
        ]
        # 24 hand joint angles + 24 hand joint velocities + object pose/velocity
        # relative to its spawn point + five fingertip distances + contact count.
        hand_q = np.clip(self.data.qpos[:24] / math.pi, -2.0, 2.0)
        hand_v = np.clip(self.data.qvel[:24] / 10.0, -5.0, 5.0)
        relative_object = np.clip(
            (object_pos - np.array([0.30, 0.0, self.start_object_z])) / 0.20,
            -5.0, 5.0,
        )
        object_v = np.clip(object_vel / 2.0, -5.0, 5.0)
        dist_obs = np.clip(np.asarray(distances) / 0.20, 0.0, 5.0)
        contact_count = min(self._object_hand_contacts(), 5) / 5.0
        return np.concatenate(
            [hand_q, hand_v, relative_object, object_v, dist_obs, [contact_count]]
        ).astype(np.float32)

    def _object_hand_contacts(self) -> int:
        touching_bodies: set[int] = set()
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body_a = int(self.model.geom_bodyid[contact.geom1])
            body_b = int(self.model.geom_bodyid[contact.geom2])
            if body_a == self.object_body_id and body_b != self.object_body_id:
                touching_bodies.add(body_b)
            elif body_b == self.object_body_id and body_a != self.object_body_id:
                touching_bodies.add(body_a)
        return sum(
            1 for body_id in touching_bodies
            if (name := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id))
            and name.startswith("rh_")
        )

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(self.n_actions)
        action = np.clip(action, -1.0, 1.0)
        closure = (action + 1.0) * 0.5
        finger_values = dict(zip(self.ACTUATOR_GROUPS, closure))
        targets = self.open_ctrl.copy()
        for finger, ids in self.actuator_ids.items():
            targets[ids] = self.closed_ctrl[ids] * finger_values[finger]
        self.data.ctrl[:] = targets
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obj_pos = self.data.xpos[self.object_body_id].copy()
        distances = np.array([
            np.linalg.norm(self.data.xpos[body_id] - obj_pos)
            for body_id in self.tip_body_ids
        ])
        proximity = np.exp(-np.maximum(0.0, distances - 0.055) / 0.05).sum()
        lift = max(0.0, float(obj_pos[2]) - self.start_object_z)
        contacts = self._object_hand_contacts()
        smoothness_cost = 0.01 * float(np.square(action - self.prev_action).mean())
        reward = 0.15 * float(proximity) + 25.0 * lift + 0.04 * min(contacts, 5)
        reward -= smoothness_cost
        self.prev_action = action.copy()

        dropped = bool(obj_pos[2] < 0.0)
        lifted = bool(obj_pos[2] > self.start_object_z + 0.06)
        self.success_steps = self.success_steps + 1 if lifted else 0
        success = self.success_steps >= 10
        terminated = dropped or success
        truncated = self.step_count >= self.horizon
        if dropped:
            reward -= 2.0
        if success:
            reward += 5.0

        info = {
            "object_height": float(obj_pos[2]),
            "lift": lift,
            "contacts": contacts,
            "success": success,
        }
        return self._observation(), float(reward), terminated, truncated, info


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.value = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def distribution_and_value(self, obs: torch.Tensor):
        features = self.shared(obs)
        mean = torch.tanh(self.mean(features))
        std = self.log_std.clamp(-2.0, 0.5).exp().expand_as(mean)
        return Normal(mean, std), self.value(features).squeeze(-1)

    @staticmethod
    def squashed_log_prob(dist: Normal, action: torch.Tensor) -> torch.Tensor:
        clipped = action.clamp(-0.999999, 0.999999)
        latent = torch.atanh(clipped)
        return (dist.log_prob(latent) - torch.log(1.0 - clipped.square() + 1e-6)).sum(-1)


class PPO:
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 3e-4):
        self.network = ActorCritic(obs_dim, action_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.entropy_coef = 0.005
        self.value_coef = 0.5

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        dist, value = self.network.distribution_and_value(obs_t)
        latent = dist.mean if deterministic else dist.sample()
        action = torch.tanh(latent)
        log_prob = ActorCritic.squashed_log_prob(dist, action)
        return action.squeeze(0).cpu().numpy(), float(log_prob.item()), float(value.item())

    def update(self, rollout: dict[str, list], epochs: int = 8, minibatch: int = 256):
        obs = torch.as_tensor(np.asarray(rollout["obs"]), dtype=torch.float32, device=DEVICE)
        actions = torch.as_tensor(np.asarray(rollout["actions"]), dtype=torch.float32, device=DEVICE)
        old_logp = torch.as_tensor(rollout["logp"], dtype=torch.float32, device=DEVICE)
        rewards = torch.as_tensor(rollout["rewards"], dtype=torch.float32, device=DEVICE)
        dones = torch.as_tensor(rollout["dones"], dtype=torch.float32, device=DEVICE)
        values = torch.as_tensor(rollout["values"], dtype=torch.float32, device=DEVICE)
        next_values = torch.as_tensor(rollout["next_values"], dtype=torch.float32, device=DEVICE)

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        count = len(rewards)
        for _ in range(epochs):
            indices = torch.randperm(count, device=DEVICE)
            for start in range(0, count, minibatch):
                idx = indices[start:start + minibatch]
                dist, new_values = self.network.distribution_and_value(obs[idx])
                new_logp = ActorCritic.squashed_log_prob(dist, actions[idx])
                ratio = torch.exp(new_logp - old_logp[idx])
                unclipped = ratio * advantages[idx]
                clipped = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * advantages[idx]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = (new_values - returns[idx]).square().mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

    def save(self, path: Path, steps: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"network": self.network.state_dict(), "steps": steps}, path)

    def load(self, path: Path):
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        return int(checkpoint.get("steps", 0))


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = ShadowHandGraspEnv(args.model, args.frame_skip, args.horizon)
    agent = PPO(obs_dim=60, action_dim=env.n_actions, lr=args.lr)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "training_log.csv"
    checkpoint_path = OUT_DIR / "shadow_hand_ppo.pt"

    obs = env.reset(seed=args.seed)
    rollout = {k: [] for k in ("obs", "actions", "logp", "rewards", "dones", "values", "next_values")}
    episode_return = 0.0
    episode_length = 0
    completed_returns: list[float] = []
    start_time = time.time()
    next_save = args.save_every

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(("steps", "episode_return", "episode_length", "success", "object_height"))
        for step in range(1, args.steps + 1):
            action, logp, value = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            _, _, next_value = agent.act(next_obs, deterministic=True)
            rollout["obs"].append(obs)
            rollout["actions"].append(action)
            rollout["logp"].append(logp)
            rollout["rewards"].append(reward)
            rollout["dones"].append(float(done))
            rollout["values"].append(value)
            rollout["next_values"].append(next_value)
            obs = next_obs
            episode_return += reward
            episode_length += 1

            if done:
                completed_returns.append(episode_return)
                writer.writerow((step, round(episode_return, 4), episode_length,
                                 int(info["success"]), round(info["object_height"], 4)))
                file.flush()
                obs = env.reset()
                episode_return = 0.0
                episode_length = 0

            if len(rollout["rewards"]) >= args.rollout_steps or step == args.steps:
                agent.update(rollout, epochs=args.epochs, minibatch=args.minibatch)
                rollout = {k: [] for k in rollout}

            if step >= next_save:
                agent.save(checkpoint_path, step)
                next_save += args.save_every

            if step % args.log_every == 0 or step == args.steps:
                recent = completed_returns[-20:]
                avg = float(np.mean(recent)) if recent else float("nan")
                print(f"steps={step:>7}  avg_return_20={avg:8.3f}  elapsed={time.time()-start_time:.1f}s")

    agent.save(checkpoint_path, args.steps)
    print(f"\n模型已保存: {checkpoint_path}")
    print(f"训练日志: {csv_path}")


def play(args):
    env = ShadowHandGraspEnv(args.model, args.frame_skip, args.horizon)
    agent = PPO(obs_dim=60, action_dim=env.n_actions)
    agent.load(args.checkpoint)
    import mujoco.viewer

    obs = env.reset(seed=args.seed)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action, _, _ = agent.act(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            viewer.sync()
            if terminated or truncated:
                print(f"回合结束: success={info['success']}, height={info['object_height']:.3f} m")
                obs = env.reset()


def main():
    parser = argparse.ArgumentParser(description="Shadow Hand MuJoCo PPO grasp-and-lift demo")
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train", help="train PPO")
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    train_parser.add_argument("--steps", type=int, default=100_000)
    train_parser.add_argument("--rollout-steps", type=int, default=2048)
    train_parser.add_argument("--horizon", type=int, default=160)
    train_parser.add_argument("--frame-skip", type=int, default=10)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--minibatch", type=int, default=256)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--log-every", type=int, default=2048)
    train_parser.add_argument("--save-every", type=int, default=10_000)
    train_parser.set_defaults(func=train)

    play_parser = sub.add_parser("play", help="view a trained policy")
    play_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    play_parser.add_argument("--checkpoint", type=Path, default=OUT_DIR / "shadow_hand_ppo.pt")
    play_parser.add_argument("--horizon", type=int, default=160)
    play_parser.add_argument("--frame-skip", type=int, default=10)
    play_parser.add_argument("--seed", type=int, default=7)
    play_parser.set_defaults(func=play)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
