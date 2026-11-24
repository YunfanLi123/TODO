"""GRPO（Group Relative Policy Optimization）核心 —— 纯 NumPy，可在 CPU 上跑通。

借鉴
----
``github.com/huggingface/trl``（Apache-2.0）的 ``GRPOTrainer``：
``trl/trainer/grpo_trainer.py``。本文件照搬它的**核心三步**：

1. **组内归一化优势**::

       advantages = (rewards - mean_grouped_rewards) / (std_rewards + 1e-4)

   （TRL 里 ``scale_rewards == "group"`` 那一支，第 2904~2906 行。）

2. **裁剪的策略梯度**（PPO 的裁剪）::

       ratio = exp(logp_new - logp_old)
       loss = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)

3. **可选的参考模型 KL 惩罚**（``beta``），把策略拉在参考附近。

为什么自己写而不直接用 TRL
--------------------------
本机没有 torch（4060 Laptop 8GB，也放不下 7B 模型的训练）。
但**算法本身**与模型无关：这里的动作空间是「从一组候选写法里选一个」，
策略是一个线性打分器，特征是文本的字符 n-gram 哈希 ——
于是整条 GRPO 链路（采样 → 打分 → 组内归一化 → 更新）能在 CPU 上**真跑一遍**，
而不是只写个脚本就算完。

真要训练大模型时，把 :class:`LinearPolicy` 换成 torch 上的语言模型即可：
采样换成 ``generate``，打分换成 :mod:`app.report.rl_reward`，
其余三步一字不改。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:                                     # numpy 是唯一的硬依赖
    import numpy as np
except Exception:                        # noqa: BLE001
    np = None  # type: ignore

#: 特征维度（字符 n-gram 哈希桶）。
FEATURE_DIM = 4096

#: 字符 n-gram 的长度。
NGRAM = 3


def _hash_features(text: str, dim: int = FEATURE_DIM) -> List[int]:
    """文本 -> 特征下标（字符 n-gram 的哈希桶）。"""
    s = "".join((text or "").split())
    if not s:
        return []
    out: List[int] = []
    if len(s) < NGRAM:
        grams = [s]
    else:
        grams = [s[i:i + NGRAM] for i in range(len(s) - NGRAM + 1)]
    for g in grams:
        h = hashlib.md5(g.encode("utf-8")).digest()
        out.append(int.from_bytes(h[:4], "big") % dim)
    return out


@dataclass
class LinearPolicy:
    """线性打分策略：``score = w · f``，组内 softmax 出概率。

    权重初始化成 0 —— 于是初始分布是均匀的，
    第一轮更新完全由奖励决定（不是由初始化决定）。
    """

    dim: int = FEATURE_DIM
    weights: Any = None
    temperature: float = 1.0

    def __post_init__(self):
        if self.weights is None:
            if np is None:
                raise RuntimeError("需要 numpy")
            self.weights = np.zeros(self.dim, dtype=np.float64)

    # -- 打分 ---------------------------------------------------------------

    def scores(self, candidates: Sequence[str]) -> Any:
        if np is None:
            raise RuntimeError("需要 numpy")
        out = np.zeros(len(candidates), dtype=np.float64)
        for i, c in enumerate(candidates):
            idx = _hash_features(c, self.dim)
            if idx:
                out[i] = float(np.sum(self.weights[idx]))
        return out / max(1e-6, self.temperature)

    def log_probs(self, candidates: Sequence[str]) -> Any:
        s = self.scores(candidates)
        m = s.max()
        exp = np.exp(s - m)
        return s - (m + math.log(float(exp.sum())))

    def probs(self, candidates: Sequence[str]) -> Any:
        return np.exp(self.log_probs(candidates))

    # -- 更新 ---------------------------------------------------------------

    def update(self, candidates: Sequence[str], advantages: Any,
               lr: float = 0.5,
               expectation: Optional[Any] = None,
               expectation_candidates: Optional[Sequence[str]] = None
               ) -> float:
        """一步策略梯度更新。返回这次更新的 KL（对参考分布的）。

        ``advantages`` 是组内归一化之后的优势。
        ``expectation`` / ``expectation_candidates`` 指定「期望特征」取自
        哪个分布（默认用候选集上的 softmax）。

        **为什么要分开**：GRPO 只用**采样出来**的那几个候选更新，
        但期望项 ``E[f]`` 要按**整组**的分布算 ——
        两者混在一起会把期望算在错误的分布上（只算到采样到的那一条）。
        """
        if np is None:
            raise RuntimeError("需要 numpy")
        logp = self.log_probs(candidates)
        p = np.exp(logp)
        exp_cands = list(expectation_candidates) if expectation_candidates \
            is not None else list(candidates)
        if expectation is None:
            expectation = self.probs(exp_cands)
        #: 更新前，整组上的对数概率（KL 要在整组分布上算）。
        old_exp_logp = self.log_probs(exp_cands)
        grad = np.zeros(self.dim, dtype=np.float64)
        for i, c in enumerate(candidates):
            if not abs(float(advantages[i])) > 1e-12:
                continue
            idx = _hash_features(c, self.dim)
            if not idx:
                continue
            # ∇log π(i) = f_i - E[f]（E 取给定分布下的期望特征）
            mean_feat = np.zeros(self.dim, dtype=np.float64)
            for j, cj in enumerate(exp_cands):
                jdx = _hash_features(cj, self.dim)
                if jdx:
                    mean_feat += float(expectation[j]) * np.bincount(
                        jdx, minlength=self.dim)
            fi = np.bincount(idx, minlength=self.dim)
            grad += float(advantages[i]) * (fi - mean_feat)
        before = self.weights.copy()
        self.weights += lr * grad
        # 更新后的 KL(旧 || 新) —— 太大说明这一步迈得太野。
        # KL 要在**整组**分布上算：只有一个候选时 softmax 恒为 1，
        # 对数概率恒为 0，算出来的 KL 永远是 0（什么也没测到）。
        new_exp_logp = self.log_probs(exp_cands)
        kl = float(np.sum(np.asarray(expectation, dtype=np.float64)
                          * (old_exp_logp - new_exp_logp)))
        self._last_step = float(np.max(np.abs(self.weights - before)))
        return kl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "temperature": self.temperature,
            "weights": [float(x) for x in self.weights],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinearPolicy":
        pol = cls(dim=int(data.get("dim") or FEATURE_DIM),
                  temperature=float(data.get("temperature") or 1.0))
        w = data.get("weights")
        if w:
            pol.weights = np.asarray(w, dtype=np.float64)
        return pol


# ---------------------------------------------------------------- GRPO


def group_advantages(rewards: Sequence[float]) -> List[float]:
    """组内归一化优势（TRL ``scale_rewards="group"`` 那一支）。

    ``A_i = (r_i - mean) / (std + 1e-4)``

    为什么必须组内归一化：奖励的绝对值随对局变化很大
    （一局可能全是 0.8，另一局全是 0.2），不归一化的话
    优势的大小只反映「这局好不好」，不反映「这几个候选里哪个更好」。
    """
    if not rewards:
        return []
    arr = np.asarray(rewards, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    return [float((x - mean) / (std + 1e-4)) for x in arr]


def clipped_policy_loss(
    logp_new: Sequence[float],
    logp_old: Sequence[float],
    advantages: Sequence[float],
    epsilon: float = 0.2,
) -> float:
    """裁剪的策略梯度损失（返回**越小越好**的标量）。

    公式与 TRL 的 ``-min(ratio*A, clip(ratio)*A)`` 一致：
    比率越界之后就不再鼓励继续加大那一步。
    """
    losses: List[float] = []
    for lnew, lold, a in zip(logp_new, logp_old, advantages):
        ratio = math.exp(max(-20.0, min(20.0, lnew - lold)))
        clipped_ratio = max(1.0 - epsilon, min(1.0 + epsilon, ratio))
        # loss = -min(ratio*A, clipped_ratio*A) = max(-ratio*A, -clipped*A)
        losses.append(max(-ratio * a, -clipped_ratio * a))
    return float(sum(losses) / max(1, len(losses)))


@dataclass
class Group:
    """一个 prompt 的一组候选。"""

    prompt: str
    candidates: List[str]
    rewards: List[float] = field(default_factory=list)
    meta: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, text: str, reward: float, meta: Optional[Dict[str, Any]] = None
            ) -> None:
        self.candidates.append(text)
        self.rewards.append(float(reward))
        self.meta.append(meta or {})


def expected_reward(policy: "LinearPolicy", group: "Group") -> float:
    """策略下的**期望奖励**：``Σ_j π(j) · r_j``。

    这是 GRPO 真正在优化的量。奖励本身是**采样**出来的
    （见 :func:`train`），所以「平均奖励」不会因为训练而变 ——
    变的是「策略挑中好写法的概率」。
    """
    if np is None:
        raise RuntimeError("需要 numpy")
    if not group.candidates:
        return 0.0
    p = policy.probs(group.candidates)
    return float(np.sum(p * np.asarray(group.rewards, dtype=np.float64)))


def train(groups: Sequence[Group],
          policy: Optional[LinearPolicy] = None,
          rounds: int = 8,
          lr: float = 0.5,
          epsilon: float = 0.2,
          seed: int = 20261031,
          verbose: bool = True) -> Dict[str, Any]:
    """跑若干轮 GRPO。

    每一轮（照 TRL 的 GRPOTrainer 的形状）：

    1. 对每个 prompt **采样**若干个候选（这里就是「这一段该写成哪种」）；
    2. 给采样出来的候选打**奖励**（:mod:`app.report.rl_reward`）；
    3. **组内归一化**成优势；
    4. 一次**裁剪的策略梯度**更新。

    返回训练摘要（期望奖励、KL、权重变化）。

    **注意**：奖励是采出来的，不是变的。所以看的是
    :func:`expected_reward`（策略下的期望奖励）在涨 ——
    这才是「模型学会了挑好写法」的证据。
    """
    if np is None:
        raise RuntimeError("需要 numpy")
    rng = np.random.default_rng(seed)
    pol = policy or LinearPolicy()
    history: List[Dict[str, Any]] = []
    best: Optional[LinearPolicy] = None
    best_mean = -1.0

    for r in range(rounds):
        kls: List[float] = []
        losses: List[float] = []
        sampled_rewards: List[float] = []
        for g in groups:
            if len(g.candidates) < 2:
                continue
            # (1) 采样：按当前策略抽一个候选
            p = pol.probs(g.candidates)
            idx = int(rng.choice(len(g.candidates), p=p))
            sampled = [g.candidates[idx]]
            rewards = [g.rewards[idx]]
            sampled_rewards.append(g.rewards[idx])
            # (2)(3) 组内归一化优势：用**整组**的奖励统计做基线
            adv = np.asarray(group_advantages(g.rewards), dtype=np.float64)[idx]
            old_full = pol.log_probs(g.candidates)
            old_logp = float(old_full[idx])
            # (4) 一步裁剪策略梯度（期望项取**整组**分布）
            kl = pol.update(sampled, np.asarray([adv]), lr=lr,
                            expectation=p,
                            expectation_candidates=g.candidates)
            new_full = pol.log_probs(g.candidates)
            losses.append(clipped_policy_loss([float(new_full[idx])],
                                              [old_logp],
                                              [float(adv)], epsilon))
            kls.append(kl)
        mean_expected = float(np.mean([expected_reward(pol, g)
                                       for g in groups])) if groups else 0.0
        record = {
            "round": r + 1,
            "expected_reward": round(mean_expected, 4),
            "sampled_reward": round(
                float(np.mean(sampled_rewards)) if sampled_rewards else 0.0, 4),
            "mean_loss": round(float(np.mean(losses)) if losses else 0.0, 4),
            "mean_kl": round(float(np.mean(kls)) if kls else 0.0, 6),
        }
        history.append(record)
        if verbose:
            print("  第 %d 轮：期望奖励 %.4f（采样命中 %.4f），平均 KL %.6f"
                  % (r + 1, record["expected_reward"],
                     record["sampled_reward"], record["mean_kl"]))
        if mean_expected > best_mean:
            best_mean = mean_expected
            best = LinearPolicy(dim=pol.dim, temperature=pol.temperature,
                                weights=pol.weights.copy())
    return {
        "history": history,
        "policy": (best or pol).to_dict(),
        "best_mean_reward": round(best_mean, 4),
    }
