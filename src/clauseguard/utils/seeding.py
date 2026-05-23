"""Reproducible seeding across numpy, torch, random, env vars.

Mirrors the seeding helpers used in paper-a, paper-b, and graphtm-cbr
so the 5-seed protocol is byte-for-byte reproducible.
"""
from __future__ import annotations

import os
import random


def seed_all(seed: int) -> None:
    """Seed numpy, random, torch, and PYTHONHASHSEED.

    Torch is imported lazily so this module remains usable in
    minimal environments without torch installed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
