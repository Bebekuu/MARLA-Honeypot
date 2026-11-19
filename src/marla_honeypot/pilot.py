"""
Pilot experiment: adaptive honeypot placement with multi-agent policy gradient.

Small, NumPy-only prototype for a master's thesis poster.
- 20-host network in 5 subnets (Hawkeyes-style subnet groups).
- B = 2 honeypots, one defender agent per honeypot.
- Each step every agent picks the subnet group for its honeypot (move costs a little).
- Attacker type is drawn per episode: Random, Greedy (highest vulnerability), Targeted (shortest path to critical).
- Defender sees a noisy network-monitor alert (FNR / FPR), not the attacker's position.
- Shared team reward: +1 trap, -1 critical asset compromised, -0.01 per honeypot move.

Methods compared:
  Random-static    : random groups at episode start, never moved
  Critical-static  : honeypots in the critical and server subnets (Hawkeyes' "Critical" baseline idea)
  Alert-follow     : hand-written reactive rule (move honeypots to alerted subnet and its next hop)
  IPG              : independent policy-gradient learners (separate tables, running-mean baseline)
  MAPG-CC          : parameter-shared policy-gradient with a centralised critic baseline (CTDE-style)
"""
import json
import sys
import numpy as np

# ---------------------------------------------------------------- network
SUBNETS = ["DMZ", "OfficeA", "OfficeB", "Servers", "Critical"]
SIZES = [3, 5, 5, 4, 3]
N_SUB = len(SUBNETS)
# allowed lateral movement between subnets (directed)
SUB_EDGES = {0: [1, 3], 1: [2, 3], 2: [1, 3], 3: [4], 4: []}
CRITICAL_ASSETS = 2            # first 2 hosts of the Critical subnet are the crown jewels
B = 2                          # honeypot budget = number of defender agents
MAX_STEPS = 30
FNR, FPR = 0.10, 0.05
MOVE_COST = 0.01
HONEYPOT_APPARENT_VULN = 0.75  # how attractive a honeypot looks to a greedy attacker

host_subnet = np.concatenate([[s] * n for s, n in enumerate(SIZES)])
N_HOSTS = len(host_subnet)
crit_hosts = set(np.where(host_subnet == 4)[0][:CRITICAL_ASSETS].tolist())

# subnet-level distance to the Critical subnet (BFS on reversed edges)
dist = {4: 0}
frontier = [4]
while frontier:
    nxt = []
    for s, outs in SUB_EDGES.items():
        for o in outs:
            if o in frontier and s not in dist:
                dist[s] = dist[o] + 1
                nxt.append(s)
    frontier = nxt
SUB_DIST = np.array([dist[s] for s in range(N_SUB)])

ATTACKERS = ["Random", "Greedy", "Targeted"]


def reachable_subnets(s):
    return [s] + SUB_EDGES[s]


class Env:
    def __init__(self, rng):
        self.rng = rng
        self.vuln = rng.uniform(0.3, 0.85, size=N_HOSTS)  # fixed per network instance

    def reset(self, atk_type):
        self.atk = atk_type
        entry = self.rng.integers(0, SIZES[0])         # an entry host in the DMZ
        self.pos = entry
        self.comp = {entry}
        self.t = 0
        self.hp = None                                  # honeypot groups, set by defenders
        return self.alert()

    def alert(self):
        """Noisy monitor: index 0 = no alert, 1..N_SUB = alerted subnet."""
        if self.rng.random() > FNR:
            return 1 + host_subnet[self.pos]
        if self.rng.random() < FPR:
            return 1 + self.rng.integers(0, N_SUB)
        return 0

    def step(self, groups):
        """groups: list of B subnet indices. Returns (reward, done, outcome, next_alert)."""
        moves = 0 if self.hp is None else sum(int(a != b) for a, b in zip(groups, self.hp))
        self.hp = list(groups)
        r = -MOVE_COST * moves
        self.t += 1
        s = host_subnet[self.pos]
        cands = [h for h in range(N_HOSTS)
                 if host_subnet[h] in reachable_subnets(s) and h not in self.comp]
        hp_cands = [g for g in set(self.hp) if g in reachable_subnets(s)]
        options = [("h", h) for h in cands] + [("p", g) for g in hp_cands]
        if not options:
            return r, True, "stuck", 0
        if self.atk == "Random":
            choice = options[self.rng.integers(len(options))]
        elif self.atk == "Greedy":
            score = [self.vuln[o[1]] if o[0] == "h" else HONEYPOT_APPARENT_VULN for o in options]
            choice = options[int(np.argmax(score))]
        else:  # Targeted: go for crown jewels, else lowest subnet distance, ties random
            def d(o):
                if o[0] == "h" and o[1] in crit_hosts:
                    return -1
                sub = host_subnet[o[1]] if o[0] == "h" else o[1]
                return SUB_DIST[sub]
            ds = np.array([d(o) for o in options])
            best = np.flatnonzero(ds == ds.min())
            choice = options[self.rng.choice(best)]
        if choice[0] == "p":
            return r + 1.0, True, "trapped", 0
        h = choice[1]
        if self.rng.random() < self.vuln[h]:            # exploit succeeds
            self.comp.add(h)
            self.pos = h
            if h in crit_hosts:
                return r - 1.0, True, "compromised", 0
        if self.t >= MAX_STEPS:
            return r, True, "timeout", 0
        return r, False, None, self.alert()


# ---------------------------------------------------------------- policies
def softmax(x):
    z = np.exp(x - x.max())
    return z / z.sum()


class Static:
    def __init__(self, groups_fn):
        self.groups_fn = groups_fn

    def act(self, alert, cur, rng, agent, t):
        return cur if cur is not None else self.groups_fn(rng)[agent]


def random_groups(rng):
    return list(rng.integers(0, N_SUB, size=B))


def critical_groups(rng):
    return [4, 3]


class AlertFollow:
    """Agent 0 -> alerted subnet, agent 1 -> its next hop toward Critical."""

    def act(self, alert, cur, rng, agent, t):
        if alert == 0:
            return cur if cur is not None else [4, 3][agent]
        s = alert - 1
        if agent == 0:
            return s
        nxt = [o for o in SUB_EDGES[s] if SUB_DIST[o] < SUB_DIST[s]]
        return nxt[0] if nxt else 4


class PG:
    """Tabular softmax policy-gradient agents with a shared team reward.
    obs = (alert in 0..N_SUB, own current group in 0..N_SUB-1 or 'unplaced')
    shared=False -> IPG (independent tables, running-mean baseline per obs)
    shared=True  -> MAPG-CC (same actors, but the baseline is a centralised critic V(true attacker subnet, t)
                    that sees the true state during training only: the CTDE idea behind MAPPO)
    """

    def __init__(self, shared, lr=0.1, critic_lr=0.1, gamma=0.98):
        self.shared = shared
        self.lr, self.clr, self.g = lr, critic_lr, gamma
        n_obs = (N_SUB + 1) * (N_SUB + 1)
        self.theta = np.zeros((B, n_obs, N_SUB))
        self.base = np.zeros((B, n_obs))
        self.V = np.zeros((N_SUB, MAX_STEPS + 1))

    def obs_idx(self, alert, cur):
        return alert * (N_SUB + 1) + (N_SUB if cur is None else cur)

    def act(self, alert, cur, rng, agent, t, greedy=False):
        o = self.obs_idx(alert, cur)
        p = softmax(self.theta[agent, o])
        return int(np.argmax(p)) if greedy else int(rng.choice(N_SUB, p=p))

    def update(self, traj):
        # traj: list of (obs_idx per agent, actions, reward, true_attacker_subnet, t)
        G = 0.0
        for obs, acts, r, s_true, t in reversed(traj):
            G = r + self.g * G
            for a in range(B):
                o = obs[a]
                if self.shared:
                    b = self.V[s_true, t]
                else:
                    b = self.base[a, o]
                adv = G - b
                p = softmax(self.theta[a, o])
                grad = -p
                grad[acts[a]] += 1.0
                self.theta[a, o] += self.lr * adv * grad
                self.base[a, o] += self.clr * (G - self.base[a, o])
            if self.shared:
                self.V[s_true, t] += self.clr * (G - self.V[s_true, t])


def run_episode(env, policy, rng, atk, learn=False, greedy=False):
    alert = env.reset(atk)
    cur = [None] * B
    traj = []
    while True:
        t = env.t
        if isinstance(policy, PG):
            obs = [policy.obs_idx(alert, cur[a]) for a in range(B)]
            acts = [policy.act(alert, cur[a], rng, a, t, greedy=greedy) for a in range(B)]
        else:
            obs = None
            acts = [policy.act(alert, cur[a], rng, a, t) for a in range(B)]
        s_true = host_subnet[env.pos]
        r, done, outcome, alert = env.step(acts)
        cur = acts
        if learn:
            traj.append((obs, acts, r, s_true, min(t, MAX_STEPS)))
        if done:
            break
    if learn:
        policy.update(traj)
    return outcome


def evaluate(env, policy, rng, n=600, greedy=False):
    res = {}
    for atk in ATTACKERS:
        outs = [run_episode(env, policy, rng, atk, greedy=greedy) for _ in range(n)]
        res[atk] = {k: outs.count(k) / n for k in ["trapped", "compromised", "timeout", "stuck"]}
    res["Mix"] = {k: float(np.mean([res[a][k] for a in ATTACKERS])) for k in res["Random"]}
    return res


def main(seed, train_eps=40000, curve_every=2000):
    rng = np.random.default_rng(seed)
    env = Env(rng)
    out = {"seed": seed, "methods": {}, "curves": {}}
    out["methods"]["Random-static"] = evaluate(env, Static(random_groups), rng)
    out["methods"]["Critical-static"] = evaluate(env, Static(critical_groups), rng)
    out["methods"]["Alert-follow"] = evaluate(env, AlertFollow(), rng)
    for name, shared in [("IPG", False), ("MAPG-CC", True)]:
        pol = PG(shared=shared)
        curve = []
        for ep in range(1, train_eps + 1):
            atk = ATTACKERS[rng.integers(len(ATTACKERS))]
            run_episode(env, pol, rng, atk, learn=True)
            if ep % curve_every == 0:
                curve.append((ep, evaluate(env, pol, rng, n=150, greedy=True)["Mix"]["trapped"]))
        out["curves"][name] = curve
        out["methods"][name] = evaluate(env, pol, rng, greedy=True)
    return out


if __name__ == "__main__":
    seed = int(sys.argv[1])
    print(json.dumps(main(seed)))
