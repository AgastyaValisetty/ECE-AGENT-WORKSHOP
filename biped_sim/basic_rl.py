import sys, os, time, math, random, argparse
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.optim as optim

from botapi import BotAPI, ACTIONS, N_ACTIONS, STATE_DIM, build_state, calculate_reward, ReplayBuffer, MAX_STEPS

GAMMA, LR, BATCH_SIZE, REPLAY_CAP, TARGET_UPDATE, TRAIN_EVERY = 0.99, 1e-3, 256, 100000, 200, 4
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 2000


class QNet(nn.Module):
    def __init__(self, sd, na, h=128):
        super().__init__()
        # 3-layer architecture matching biped_dqn.pt checkpoint
        # Linear(input, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, output)
        self.net = nn.Sequential(
                nn.Linear(sd, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, na)
            )
    def forward(self, x): return self.net(x)


class DQNAgent:
    def __init__(self, sd, na, dev):
        self.dev = dev; self.q = QNet(sd, na).to(dev); self.t = QNet(sd, na).to(dev)
        self.t.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=LR)
        self.buf, self.g, self.steps = ReplayBuffer(REPLAY_CAP), GAMMA, 0

    def act(self, s, eps):
        if random.random() < eps: return random.randrange(N_ACTIONS)
        with torch.no_grad():
            return int(self.q(torch.tensor(s, dtype=torch.float32, device=self.dev).unsqueeze(0)).argmax().item())

    def remember(self, s, a, r, s2, d): self.buf.push(s, a, r, s2, d)

    def learn(self):
        if len(self.buf) < BATCH_SIZE: return
        s, a, r, s2, d = [x.to(self.dev) for x in self.buf.sample(BATCH_SIZE)]
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad(): target = r + self.g * self.t(s2).max(1)[0] * (1 - d)
        nn.functional.mse_loss(q, target).backward()
        self.opt.step(); self.opt.zero_grad(); self.steps += 1
        if self.steps % TARGET_UPDATE == 0: self.t.load_state_dict(self.q.state_dict())

    def save(self, p): torch.save(self.q.state_dict(), p)
    def load(self, p):
        self.q.load_state_dict(torch.load(p, map_location=self.dev))
        self.t.load_state_dict(self.q.state_dict())


def run_episode(api, agent, eps, train=True):
    obs = api.reset_standing(); s = build_state(obs)
    total_r, stp, done = 0.0, 0, False
    while stp < MAX_STEPS and not done:
        a = agent.act(s, eps); api.apply_action_vector(ACTIONS[a]); obs2 = api.step()
        done = bool(obs2['torso_contact']) or obs2['torso_y'] < 0.4
        r = calculate_reward(obs, obs2, a, done); s2 = build_state(obs2)
        if train:
            agent.remember(s, a, r, s2, done)
            if stp % TRAIN_EVERY == 0: agent.learn()
        total_r += r; s, obs = s2, obs2; stp += 1
    return total_r, stp, float(obs['displacement_x'])


def evaluate(api, agent, n=3, render=False):
    dist, steps = 0.0, 0
    for _ in range(n):
        obs = api.reset_standing(); s = build_state(obs); stp, done = 0, False
        while stp < MAX_STEPS and not done:
            s = build_state(obs); a = agent.act(s, 0.0); api.apply_action_vector(ACTIONS[a]); obs = api.step()
            if render: api.render()
            done = bool(obs['torso_contact']) or obs['torso_y'] < 0.4; stp += 1
        dist += float(obs['displacement_x']); steps += stp
    return dist / n, steps / n


def main():
    p = argparse.ArgumentParser(); pa = p.add_argument
    pa('--episodes', type=int, default=5000); pa('--render', action='store_true')
    pa('--load', default=None); pa('--save', default='biped_dqn.pt')
    args = p.parse_args(); torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[RL] Device: {device}")
    api, agent = BotAPI(render_mode='headless'), DQNAgent(STATE_DIM, N_ACTIONS, device)

    if args.load:
        agent.load(args.load); print(f"[RL] Loaded: {args.load}")
    else:
        eps, best = EPS_START, -math.inf; rewards, dists = deque(maxlen=50), deque(maxlen=50); t0 = time.time()
        print(f"[RL] Training {args.episodes} episodes...")
        for e in range(1, args.episodes + 1):
            r, st, d = run_episode(api, agent, eps)
            rewards.append(r); dists.append(d)
            if d > best: best = d; agent.save(args.save)
            eps = max(EPS_END, EPS_START + (EPS_END - EPS_START) * e / EPS_DECAY)
            if e % 50 == 1: print(f"[ep {e:5d}] r={float(np.mean(rewards)):8.1f}  d={float(np.mean(dists)):.2f}m  best={best:.2f}m  eps={eps:.3f}  {time.time()-t0:5.0f}s")
        print(f"[RL] Done. Best: {best:.2f}m -> {args.save}")

    eval_api = api if not args.render else BotAPI(render_mode='human')
    avg_d, avg_s = evaluate(eval_api, agent, render=args.render)
    print(f"[RL] Eval: {avg_d:.2f}m, {avg_s:.0f} steps ({avg_s / 60:.1f}s)")
    eval_api.close(); api.close()


if __name__ == '__main__':
    main()
