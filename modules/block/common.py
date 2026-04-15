"""Small shared helpers for model blocks."""

from __future__ import annotations

import collections.abc
from itertools import repeat
from typing import Any, Callable

import torch


def _ntuple(n: int) -> Callable[[Any], tuple[Any, ...]]:
    """Return a parser that converts scalars or iterables into an `n`-tuple."""

    def parse(x: Any) -> tuple[Any, ...]:
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_3tuple = _ntuple(3)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive shift-scale modulation to a token tensor."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
