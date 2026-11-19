"""MARLA Honeypot (Multi-Agent Reinforced Learning Adaptive Honeypot): pilot simulator and agents."""

from .pilot import (
    ATTACKERS,
    B,
    Env,
    AlertFollow,
    PG,
    Static,
    critical_groups,
    evaluate,
    main as run_seed,
    random_groups,
    run_episode,
)

__all__ = [
    "ATTACKERS",
    "B",
    "Env",
    "AlertFollow",
    "PG",
    "Static",
    "critical_groups",
    "evaluate",
    "random_groups",
    "run_episode",
    "run_seed",
]

__version__ = "0.1.0"
