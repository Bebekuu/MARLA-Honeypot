import numpy as np

from marla_honeypot import ATTACKERS, B, PG, AlertFollow, Env, Static, critical_groups, evaluate, run_episode
from marla_honeypot.pilot import N_SUB

OUTCOMES = {"trapped", "compromised", "timeout", "stuck"}


def test_reset_returns_valid_alert():
    env = Env(np.random.default_rng(0))
    alert = env.reset("Random")
    assert 0 <= alert <= N_SUB


def test_episode_ends_with_known_outcome():
    rng = np.random.default_rng(1)
    env = Env(rng)
    for attacker in ATTACKERS:
        assert run_episode(env, Static(critical_groups), rng, attacker) in OUTCOMES


def test_learner_picks_valid_subnets():
    rng = np.random.default_rng(2)
    pol = PG(shared=True)
    for agent in range(B):
        assert 0 <= pol.act(alert=1, cur=None, rng=rng, agent=agent, t=0) < N_SUB


def test_rates_are_probabilities():
    rng = np.random.default_rng(3)
    env = Env(rng)
    res = evaluate(env, AlertFollow(), rng, n=30)
    for attacker in ATTACKERS + ["Mix"]:
        total = sum(res[attacker].values())
        assert abs(total - 1.0) < 1e-9


def test_short_training_runs():
    rng = np.random.default_rng(4)
    env = Env(rng)
    pol = PG(shared=False)
    for _ in range(200):
        run_episode(env, pol, rng, ATTACKERS[rng.integers(3)], learn=True)
    assert np.isfinite(pol.theta).all()
