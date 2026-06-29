"""propose: discrete-action sac implementation for atari (independent of stable-baselines3).
input: gymnasium environment with discrete action space, hyperparameters.
output: trained discrete sac policy with actor-critic networks and replay buffer."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.helpers import select_torch_device


@dataclass
class ReplayBatch:
    obs: Any
    actions: Any
    rewards: Any
    next_obs: Any
    dones: Any


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.pos] = obs
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int, device: Any) -> ReplayBatch:
        import torch

        idx = np.random.randint(0, len(self), size=batch_size)
        return ReplayBatch(
            obs=torch.as_tensor(self.obs[idx], device=device, dtype=torch.float32).div_(255.0),
            actions=torch.as_tensor(self.actions[idx], device=device, dtype=torch.long),
            rewards=torch.as_tensor(self.rewards[idx], device=device, dtype=torch.float32),
            next_obs=torch.as_tensor(self.next_obs[idx], device=device, dtype=torch.float32).div_(255.0),
            dones=torch.as_tensor(self.dones[idx], device=device, dtype=torch.float32),
        )


class NatureBackbone:
    def __init__(self, in_channels: int) -> None:
        import torch
        from torch import nn

        self.nn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, in_channels, 84, 84)
            self.features_dim = int(self.nn(sample).shape[1])


def build_policy(in_channels: int, action_dim: int):
    from torch import nn

    backbone = NatureBackbone(in_channels)
    return nn.Sequential(backbone.nn, nn.Linear(backbone.features_dim, 512), nn.ReLU(), nn.Linear(512, action_dim))


class DiscreteSAC:
    """Small discrete-action SAC implementation for Atari smoke/full runs.

    This is intentionally independent of Stable Baselines3 because SB3 SAC is
    continuous-action. Inputs are expected as unbatched CHW uint8 frame stacks.
    """

    def __init__(
        self,
        policy: str,
        env: Any,
        seed: int = 0,
        verbose: int = 0,
        device: str = "auto",
        tensorboard_log: str | None = None,
        learning_rate: float = 3e-4,
        buffer_size: int = 500000,
        learning_starts: int = 50000,
        batch_size: int = 32,
        gamma: float = 0.99,
        tau: float = 0.005,
        train_freq: int = 1,
        gradient_steps: int = 1,
        target_entropy_scale: float = 0.6,
        alpha_lr: float = 3e-4,
        max_grad_norm: float = 1.0,
        exploration_initial_eps: float = 0.0,
        exploration_final_eps: float = 0.0,
        exploration_fraction: float = 0.1,
        log_interval: int = 50000,
        diagnostic_interval: int = 50000,
        q_backup_mode: str = "min",
        q_clip_value: float | None = 10.0,
        entropy_penalty_coef: float = 0.0,
        **_: Any,
    ) -> None:
        import torch

        if policy != "CnnPolicy":
            raise ValueError(f"DiscreteSAC currently supports CnnPolicy only, got {policy}")
        if env.action_space.__class__.__name__ != "Discrete":
            raise ValueError(f"DiscreteSAC requires Discrete action space, got {env.action_space}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.env = env
        self.seed = seed
        self.verbose = verbose
        self.device = select_torch_device(device)
        self.action_dim = int(env.action_space.n)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.learning_starts = int(learning_starts)
        self.batch_size = int(batch_size)
        self.train_freq = int(train_freq)
        self.gradient_steps = int(gradient_steps)
        self.max_grad_norm = float(max_grad_norm)
        self.target_entropy = -target_entropy_scale * np.log(1.0 / self.action_dim)
        self.exploration_initial_eps = float(exploration_initial_eps)
        self.exploration_final_eps = float(exploration_final_eps)
        self.exploration_fraction = float(exploration_fraction)
        self.log_interval = int(log_interval)
        self.diagnostic_interval = int(diagnostic_interval)
        self.q_backup_mode = str(q_backup_mode).lower()
        if self.q_backup_mode not in {"min", "mean"}:
            raise ValueError(f"q_backup_mode must be 'min' or 'mean', got {q_backup_mode}")
        self.q_clip_value = None if q_clip_value is None else float(q_clip_value)
        self.entropy_penalty_coef = float(entropy_penalty_coef)
        self.action_counts: Counter[int] = Counter()
        self.interval_action_counts: Counter[int] = Counter()
        self.latest_update_stats: dict[str, Any] = {}

        obs, _ = env.reset(seed=seed)
        obs_shape = tuple(int(x) for x in obs.shape)
        if obs_shape != (4, 84, 84):
            raise ValueError(f"DiscreteSAC expected observation shape (4, 84, 84), got {obs_shape}")

        in_channels = obs_shape[0]
        self.actor = build_policy(in_channels, self.action_dim).to(self.device)
        self.q1 = build_policy(in_channels, self.action_dim).to(self.device)
        self.q2 = build_policy(in_channels, self.action_dim).to(self.device)
        self.q1_target = build_policy(in_channels, self.action_dim).to(self.device)
        self.q2_target = build_policy(in_channels, self.action_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=learning_rate)
        self.log_alpha = torch.zeros((), device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.buffer = ReplayBuffer(buffer_size, obs_shape)
        self.num_timesteps = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _safe_logits(self, logits: Any):
        import torch

        return torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

    def _safe_probs(self, logits: Any):
        import torch

        probs = torch.softmax(self._safe_logits(logits), dim=1)
        probs = torch.nan_to_num(probs, nan=1.0 / self.action_dim, posinf=1.0 / self.action_dim, neginf=0.0)
        probs = probs.clamp_min(1e-8)
        return probs / probs.sum(dim=1, keepdim=True)

    def _exploration_rate(self, total_timesteps: int) -> float:
        if self.exploration_initial_eps <= self.exploration_final_eps:
            return self.exploration_final_eps
        decay_steps = max(int(total_timesteps * self.exploration_fraction), 1)
        progress = min(self.num_timesteps / decay_steps, 1.0)
        return self.exploration_initial_eps + progress * (self.exploration_final_eps - self.exploration_initial_eps)

    def _obs_tensor(self, obs: np.ndarray):
        import torch

        return torch.as_tensor(obs[None], device=self.device, dtype=torch.float32).div_(255.0)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        import torch

        self.actor.eval()
        with torch.no_grad():
            logits = self._safe_logits(self.actor(self._obs_tensor(obs)))
            if deterministic:
                action = int(torch.argmax(logits, dim=1).item())
            else:
                probs = self._safe_probs(logits)
                action = int(torch.distributions.Categorical(probs=probs).sample().item())
        self.actor.train()
        return action, None

    def learn(
        self,
        total_timesteps: int,
        checkpoint_freq: int = 0,
        checkpoint_dir: str | Path | None = None,
        diagnostics_path: str | Path | None = None,
    ):
        from src.config.logger import logger

        diagnostics_file = Path(diagnostics_path) if diagnostics_path else None
        obs, _ = self.env.reset(seed=self.seed)
        for step in range(int(total_timesteps)):
            eps = self._exploration_rate(int(total_timesteps))
            if self.num_timesteps < self.learning_starts or random.random() < eps:
                action = int(self.env.action_space.sample())
            else:
                action, _ = self.predict(obs, deterministic=False)
            self.action_counts[action] += 1
            self.interval_action_counts[action] += 1

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated or truncated)
            self.buffer.add(np.asarray(obs), action, float(reward), np.asarray(next_obs), done)
            obs = next_obs
            self.num_timesteps += 1

            if len(self.buffer) >= self.learning_starts and step % self.train_freq == 0:
                for _ in range(self.gradient_steps):
                    self._update()

            if done:
                obs, _ = self.env.reset()

            if checkpoint_freq and checkpoint_dir and self.num_timesteps % int(checkpoint_freq) == 0:
                self.save(Path(checkpoint_dir) / f"discrete_sac_step_{self.num_timesteps}.pt")
            if self.log_interval and self.num_timesteps % self.log_interval == 0:
                logger.info(
                    "discretesac step=%d eps=%.3f actions=%s buffer=%d",
                    self.num_timesteps,
                    eps,
                    dict(sorted(self.action_counts.items())),
                    len(self.buffer),
                )
            if diagnostics_file and self.diagnostic_interval and self.num_timesteps % self.diagnostic_interval == 0:
                self._write_diagnostics(diagnostics_file, eps)
        return self

    def _step_optimizer(self, loss: Any, optimizer: Any, module: Any) -> bool:
        import torch

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            return False
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), self.max_grad_norm)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            return False
        optimizer.step()
        return True

    def _update(self) -> None:
        import torch
        import torch.nn.functional as F  # noqa: N812

        from src.config.logger import logger

        batch = self.buffer.sample(self.batch_size, self.device)

        with torch.no_grad():
            next_logits = self._safe_logits(self.actor(batch.next_obs))
            next_probs = self._safe_probs(next_logits)
            next_log_probs = torch.log(next_probs)
            next_q = self._combine_q(self.q1_target(batch.next_obs), self.q2_target(batch.next_obs))
            next_v = (next_probs * (next_q - self.alpha.detach() * next_log_probs)).sum(dim=1)
            target_q = batch.rewards + (1.0 - batch.dones) * self.gamma * next_v
            target_q = torch.nan_to_num(target_q, nan=0.0, posinf=10.0, neginf=-10.0)
            target_q = self._clip_q(target_q)

        q1_pred = self.q1(batch.obs).gather(1, batch.actions[:, None]).squeeze(1)
        q2_pred = self.q2(batch.obs).gather(1, batch.actions[:, None]).squeeze(1)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        if torch.isnan(q1_loss) or torch.isinf(q1_loss) or torch.isnan(q2_loss) or torch.isinf(q2_loss):
            logger.warning("nan/inf detected in q loss at step %d -- skipping update", self.num_timesteps)
            return

        q1_ok = self._step_optimizer(q1_loss, self.q1_opt, self.q1)
        q2_ok = self._step_optimizer(q2_loss, self.q2_opt, self.q2)
        if not (q1_ok and q2_ok):
            logger.warning("non-finite q gradients at step %d -- skipping policy update", self.num_timesteps)
            return

        logits = self._safe_logits(self.actor(batch.obs))
        probs = self._safe_probs(logits)
        log_probs = torch.log(probs)
        q1_all = self.q1(batch.obs)
        q2_all = self.q2(batch.obs)
        q = self._combine_q(q1_all, q2_all)
        entropy_per_obs = -(probs * log_probs).sum(dim=1)
        entropy_penalty = self.entropy_penalty_coef * (entropy_per_obs - self.target_entropy).pow(2).mean()
        actor_loss = (probs * (self.alpha.detach() * log_probs - q)).sum(dim=1).mean() + entropy_penalty

        actor_ok = self._step_optimizer(actor_loss, self.actor_opt, self.actor)
        if not actor_ok:
            logger.warning("non-finite actor gradients at step %d -- skipping alpha/target update", self.num_timesteps)
            return

        entropy = entropy_per_obs.detach().mean()
        alpha_loss = self.log_alpha * (entropy - self.target_entropy)
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-5.0, 2.0)

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        if self.verbose > 0 and self.num_timesteps % 1000 == 0:
            with torch.no_grad():
                mean_q = float(q.mean().cpu().item())
                mean_target_q = float(target_q.mean().cpu().item())
                alpha_val = float(self.alpha.cpu().item())
                log_alpha_val = float(self.log_alpha.cpu().item())
                q_gap = float((q1_all - q2_all).abs().mean().cpu().item())
                mean_probs = [float(x) for x in probs.mean(dim=0).cpu().tolist()]
                greedy_counts = Counter(int(x) for x in torch.argmax(logits, dim=1).cpu().tolist())
                self.latest_update_stats = {
                    "MeanQ": mean_q,
                    "MeanTargetQ": mean_target_q,
                    "MeanQGap": q_gap,
                    "ActorEntropy": float(entropy.cpu().item()),
                    "Alpha": alpha_val,
                    "LogAlpha": log_alpha_val,
                    "ActorLoss": float(actor_loss.detach().cpu().item()),
                    "Q1Loss": float(q1_loss.detach().cpu().item()),
                    "Q2Loss": float(q2_loss.detach().cpu().item()),
                    "AlphaLoss": float(alpha_loss.detach().cpu().item()),
                    "EntropyPenalty": float(entropy_penalty.detach().cpu().item()),
                    "PolicyMeanProbs": json.dumps(mean_probs),
                    "BatchGreedyActions": json.dumps(dict(sorted(greedy_counts.items())), sort_keys=True),
                }
            logger.info(
                "step=%d q=%.2f target_q=%.2f entropy=%.3f alpha=%.4f log_alpha=%.2f actor_loss=%.4f",
                self.num_timesteps,
                mean_q,
                mean_target_q,
                float(entropy.cpu().item()),
                alpha_val,
                log_alpha_val,
                float(actor_loss.cpu().item()),
            )
        else:
            with torch.no_grad():
                q_gap = float((q1_all - q2_all).abs().mean().cpu().item())
                mean_probs = [float(x) for x in probs.mean(dim=0).cpu().tolist()]
                greedy_counts = Counter(int(x) for x in torch.argmax(logits, dim=1).cpu().tolist())
                self.latest_update_stats = {
                    "MeanQ": float(q.mean().cpu().item()),
                    "MeanTargetQ": float(target_q.mean().cpu().item()),
                    "MeanQGap": q_gap,
                    "ActorEntropy": float(entropy.cpu().item()),
                    "Alpha": float(self.alpha.cpu().item()),
                    "LogAlpha": float(self.log_alpha.cpu().item()),
                    "ActorLoss": float(actor_loss.detach().cpu().item()),
                    "Q1Loss": float(q1_loss.detach().cpu().item()),
                    "Q2Loss": float(q2_loss.detach().cpu().item()),
                    "AlphaLoss": float(alpha_loss.detach().cpu().item()),
                    "EntropyPenalty": float(entropy_penalty.detach().cpu().item()),
                    "PolicyMeanProbs": json.dumps(mean_probs),
                    "BatchGreedyActions": json.dumps(dict(sorted(greedy_counts.items())), sort_keys=True),
                }

    def _soft_update(self, source: Any, target: Any) -> None:
        with __import__("torch").no_grad():
            for src_param, target_param in zip(source.parameters(), target.parameters(), strict=False):  # noqa: B905
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * src_param.data)

    def _combine_q(self, q1: Any, q2: Any) -> Any:
        q = (q1 + q2) * 0.5 if self.q_backup_mode == "mean" else __import__("torch").min(q1, q2)
        return self._clip_q(q)

    def _clip_q(self, q: Any) -> Any:
        if self.q_clip_value is None or self.q_clip_value <= 0:
            return q
        return q.clamp(-self.q_clip_value, self.q_clip_value)

    def _action_entropy(self, counts: Counter[int]) -> float:
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        return -sum((count / total) * math.log(count / total) for count in counts.values() if count > 0)

    def _write_diagnostics(self, path: Path, eps: float) -> None:
        fields = [
            "Timesteps",
            "Epsilon",
            "BufferSize",
            "TotalActions",
            "IntervalActions",
            "TotalActionEntropy",
            "IntervalActionEntropy",
            "TargetEntropy",
            "QBackupMode",
            "QClipValue",
            "EntropyPenaltyCoef",
            "MeanQ",
            "MeanTargetQ",
            "MeanQGap",
            "ActorEntropy",
            "Alpha",
            "LogAlpha",
            "ActorLoss",
            "Q1Loss",
            "Q2Loss",
            "AlphaLoss",
            "EntropyPenalty",
            "PolicyMeanProbs",
            "BatchGreedyActions",
        ]
        row: dict[str, Any] = {
            "Timesteps": self.num_timesteps,
            "Epsilon": round(float(eps), 6),
            "BufferSize": len(self.buffer),
            "TotalActions": json.dumps(dict(sorted(self.action_counts.items())), sort_keys=True),
            "IntervalActions": json.dumps(dict(sorted(self.interval_action_counts.items())), sort_keys=True),
            "TotalActionEntropy": round(self._action_entropy(self.action_counts), 6),
            "IntervalActionEntropy": round(self._action_entropy(self.interval_action_counts), 6),
            "TargetEntropy": round(float(self.target_entropy), 6),
            "QBackupMode": self.q_backup_mode,
            "QClipValue": "" if self.q_clip_value is None else self.q_clip_value,
            "EntropyPenaltyCoef": self.entropy_penalty_coef,
        }
        row.update(self.latest_update_stats)

        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self.interval_action_counts.clear()

    def save(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "log_alpha": self.log_alpha.detach(),
                "actor_opt": self.actor_opt.state_dict(),
                "q1_opt": self.q1_opt.state_dict(),
                "q2_opt": self.q2_opt.state_dict(),
                "alpha_opt": self.alpha_opt.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        import torch

        state = torch.load(Path(path), map_location=self.device, weights_only=True)
        self.actor.load_state_dict(state["actor"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.q1_target.load_state_dict(state["q1_target"])
        self.q2_target.load_state_dict(state["q2_target"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.q1_opt.load_state_dict(state["q1_opt"])
        self.q2_opt.load_state_dict(state["q2_opt"])
        self.alpha_opt.load_state_dict(state["alpha_opt"])
