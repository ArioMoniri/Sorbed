#!/usr/bin/env python3
"""Federated-learning client seam for the continual grader — build now, wire later.

The continual fine-tune (:mod:`training.continual`) produces a small, human-driven
update to the deployed ConvNeXt-V2 grader at each hospital. This module packages
that update as a **client delta** so a future FedAvg / FedProx server can average
updates *across hospitals without any image, mask, or feedback row ever leaving
the site*. Only the parameter delta (and non-PHI aggregate stats) travel.

What ships today (no network required)
--------------------------------------
* :class:`ClientUpdate` — the on-wire contract: round id, the base checkpoint sha
  the delta is relative to, the delta tensors (full weight-deltas or LoRA A/B
  matrices), the number of local examples, and quality metrics. ``serialize`` /
  ``load`` round-trip it to a single ``.pt`` file.
* :func:`compute_delta` — ``fine_tuned - base`` per matching parameter.
* :func:`apply_delta` — ``base + scale · delta`` (the inverse of the above at
  ``scale=1``).
* :func:`fedavg_aggregate` — a real, unit-checkable weighted average of several
  clients' deltas (FedAvg is the ``μ=0`` special case of FedProx).

What a real server adds later (documented, not yet wired)
---------------------------------------------------------
* **FedProx over FedAvg.** Hospital data is heavily non-IID (different stage mix,
  skin-tone distribution, camera/lighting). A proximal term ``μ/2·‖w−w_global‖²``
  is added to the *client* loss in :mod:`training.continual` (config flag) to keep
  divergent local updates anchored to the global weights; the *server* aggregation
  math is identical to FedAvg. ``fedavg_aggregate`` here is that server step.
* **Secure aggregation.** :attr:`ClientUpdate.tensors` would pass through a
  pairwise mask/unmask layer so the server only ever sees the masked sum, never an
  individual client's delta. The packaging is unchanged — masking wraps the bytes.
* **Differential privacy.** Per-example gradient clipping + Gaussian noise are
  applied client-side (recorded in :attr:`ClientUpdate.privacy`) so the server can
  enforce a floor. Shipped off (``dp_applied=False``); enabling it is config.

No raw data ever leaves the client: this module only ever touches parameter
tensors and aggregate, non-identifying statistics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Keys that carry structured metadata rather than tensors when serialized.
_META_KEYS = (
    "round_id",
    "base_sha",
    "adapter_mode",
    "num_samples",
    "metrics",
    "data_stats",
    "privacy",
    "client_id",
)


@dataclass
class ClientUpdate:
    """One hospital's federated update, relative to a named base checkpoint.

    ``tensors`` maps parameter name → delta tensor. For ``adapter_mode="delta"``
    these are full ``fine_tuned - base`` weight deltas; for ``adapter_mode="lora"``
    they are the low-rank ``A``/``B`` adapter matrices (keys suffixed ``.lora_A`` /
    ``.lora_B``). The server treats them opaquely and averages them in-place.
    """

    round_id: int
    base_sha: str
    tensors: dict[str, Any]
    num_samples: int
    adapter_mode: str = "delta"
    metrics: dict[str, float] = field(default_factory=dict)
    data_stats: dict[str, Any] = field(default_factory=dict)
    privacy: dict[str, Any] = field(
        default_factory=lambda: {
            "dp_applied": False,
            "noise_multiplier": None,
            "clip_norm": None,
        }
    )
    client_id: str = "local"

    def __post_init__(self) -> None:
        if self.adapter_mode not in {"delta", "lora"}:
            raise ValueError(f"adapter_mode must be 'delta' or 'lora', got {self.adapter_mode!r}")
        if self.num_samples < 0:
            raise ValueError("num_samples must be non-negative")

    def meta(self) -> dict[str, Any]:
        """The non-tensor, JSON-serialisable header (safe to log / send in clear)."""
        return {k: getattr(self, k) for k in _META_KEYS}

    def serialize(self, path: str | Path) -> Path:
        """Write the update (header + tensors) to a single ``.pt`` file."""
        import torch

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": self.meta(), "tensors": {k: v.cpu() for k, v in self.tensors.items()}}
        torch.save(payload, out)
        # A sibling JSON header keeps the metadata inspectable without torch.
        out.with_suffix(".json").write_text(
            json.dumps(self.meta(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> ClientUpdate:
        """Reconstruct a :class:`ClientUpdate` written by :meth:`serialize`."""
        import torch

        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        meta = dict(payload["meta"])
        return cls(tensors=dict(payload["tensors"]), **meta)


def compute_delta(
    base_state: dict[str, Any],
    fine_tuned_state: dict[str, Any],
    *,
    only_floats: bool = True,
) -> dict[str, Any]:
    """Per-parameter ``fine_tuned - base`` over the shared, same-shape keys.

    Integer buffers (e.g. ``num_batches_tracked``) are skipped when ``only_floats``
    so the delta stays a pure additive gradient in float space. Keys absent from
    either state, or whose shapes disagree, are skipped (the models must share an
    architecture for a meaningful delta).
    """
    delta: dict[str, Any] = {}
    for key, ft in fine_tuned_state.items():
        base = base_state.get(key)
        if base is None or base.shape != ft.shape:
            continue
        if only_floats and not ft.dtype.is_floating_point:
            continue
        delta[key] = (ft - base).detach().clone()
    if not delta:
        raise ValueError("no overlapping float parameters between base and fine-tuned states")
    return delta


def apply_delta(
    base_state: dict[str, Any],
    delta: dict[str, Any],
    *,
    scale: float = 1.0,
) -> dict[str, Any]:
    """Return a new state dict equal to ``base + scale · delta`` on delta's keys.

    Non-delta keys are carried through from ``base`` unchanged. This is the exact
    inverse of :func:`compute_delta` at ``scale=1.0`` and is how a client (or the
    server) reconstructs full weights from a base checkpoint plus an update.
    """
    merged: dict[str, Any] = {k: v.clone() for k, v in base_state.items()}
    for key, d in delta.items():
        if key not in merged:
            continue
        merged[key] = merged[key] + float(scale) * d
    return merged


def fedavg_aggregate(
    deltas: list[dict[str, Any]],
    weights: list[float] | None = None,
) -> dict[str, Any]:
    """Weighted mean of several clients' deltas — the FedAvg server step.

    ``weights`` defaults to uniform; pass each client's ``num_samples`` to reproduce
    canonical FedAvg (contributions proportional to local data size). Only keys
    present in *every* delta are aggregated, so a client that trained a subset of
    parameters cannot silently drop others to zero. FedProx uses this same server
    aggregation; its proximal term lives in the client loss, not here.
    """
    import torch

    if not deltas:
        raise ValueError("no deltas to aggregate")
    if weights is None:
        weights = [1.0] * len(deltas)
    if len(weights) != len(deltas):
        raise ValueError("weights and deltas must be the same length")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("sum of aggregation weights must be positive")
    shared = set(deltas[0])
    for d in deltas[1:]:
        shared &= set(d)
    if not shared:
        raise ValueError("clients share no common parameter keys")

    aggregated: dict[str, Any] = {}
    for key in shared:
        acc = torch.zeros_like(deltas[0][key])
        for d, w in zip(deltas, weights, strict=True):
            acc = acc + (float(w) / total) * d[key]
        aggregated[key] = acc
    return aggregated


def package_client_update(
    *,
    round_id: int,
    base_sha: str,
    base_state: dict[str, Any],
    fine_tuned_state: dict[str, Any],
    num_samples: int,
    metrics: dict[str, float] | None = None,
    data_stats: dict[str, Any] | None = None,
    client_id: str = "local",
) -> ClientUpdate:
    """Build a :class:`ClientUpdate` (full weight-delta) from two state dicts.

    This is the seam :mod:`training.continual` calls after a promoted round: it
    turns (base checkpoint, fine-tuned checkpoint) into the delta a future FedAvg
    server would consume, carrying the round's quality metrics and non-PHI stats.
    """
    delta = compute_delta(base_state, fine_tuned_state)
    return ClientUpdate(
        round_id=int(round_id),
        base_sha=str(base_sha),
        tensors=delta,
        num_samples=int(num_samples),
        adapter_mode="delta",
        metrics=dict(metrics or {}),
        data_stats=dict(data_stats or {}),
        client_id=str(client_id),
    )
