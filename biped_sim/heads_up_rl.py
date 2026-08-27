"""
Enhanced Deep Q-Learning: train biped for upright posture and straight legs.

This variant emphasizes:
1. Keeping the torso upright (low tilt angle deviation)
2. Keeping knees from being overly bent (straighter legs for stability)
3. Maintaining consistent joint positions for stable walking
"""

import sys, os, time, math, random, argparse
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.optim as optim

from botapi import BotAPI, STATE_FEATURES as ORIGINAL_STATE_FEATURES, STATE_SCALE as ORIGINAL_STATE_SCALE

# Use original state dimension for transfer learning from biped_dqn.pt
STATE_DIM = len(ORIGINAL_STATE_FEATURES)

# Enhanced reward weights
GAMMA, LR, BATCH_SIZE, REPLAY_CAP, TARGET_UPDATE, TRAIN_EVERY = 0.99, 1e-3, 256, 100000, 200, 4
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 2000

# Enhanced reward weights for posture
W_FORWARD = 30.0        # Forward movement (same as base)
W_ALIVE = 0.1           # Time penalty for survival
W_TORSO_UPRIGHT = -0.5  # Penalty for tilting (enhanced from -0.05)
W_KNEE_BEND = -0.3      # Penalty for bent knees (negative = penalty), knee angle in [-0.1, 2.5]
                        # Knee at 0 = straight leg, positive = bent, negative = hyperextended
W_ENERGY = -0.0005      # Energy consumption penalty
FALL_PENALTY = -100.0   # Large penalty for falling
MAX_STEPS = 300

# Constants from botapi
N_ACTIONS = 6
ACTIONS = [
    np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),         # 0 REST
    np.array([30.0, -20.0, 30.0, -20.0], dtype=np.float32),    # 1 SQUAT
    np.array([-30.0, 20.0, -30.0, 20.0], dtype=np.float32),    # 2 SPRING
    np.array([30.0, 20.0, -30.0, 20.0], dtype=np.float32),     # 3 STEP LEFT
    np.array([-30.0, 20.0, 30.0, 20.0], dtype=np.float32),     # 4 STEP RIGHT
    np.array([30.0, 0.0, 30.0, 0.0], dtype=np.float32),        # 5 LEAN FORWARD
]


class ReplayBuffer:
    """Circular memory of past experiences for stable training."""

    def __init__(self, capacity: int):
        self.mem = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done: bool):
        self.mem.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.mem, batch_size)
        states = np.stack([t[0] for t in batch])
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.stack([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        return (torch.tensor(states), torch.tensor(actions), torch.tensor(rewards),
                torch.tensor(next_states), torch.tensor(dones))

    def __len__(self):
        return len(self.mem)


def calculate_enhanced_reward(obs_before: dict, obs_after: dict, action_idx: int, done: bool) -> float:
    """
    Enhanced reward function emphasizing upright posture and straight legs.

    Components:
    - Forward progress (dx forward per step)
    - Survival bonus (time penalty, encourages longer episodes)
    - Enhanced torso upright penalty (larger weight for tilt deviation)
    - Knee straight penalty (penalty for knee angle deviation from 0, i.e., bent legs)
    - Energy consumption penalty (for large torque actions)
    - Fall penalty (large negative reward for falling)
    """
    # Forward progress - encourage moving forward
    dx = obs_after['torso_x'] - obs_before['torso_x']

    # Survival bonus - slight penalty per step (encourages shorter, stable episodes)
    alive_bonus = W_ALIVE

    # Enhanced torso upright reward - penalize deviation from vertical (angle = 0)
    torso_tilt_penalty = W_TORSO_UPRIGHT * abs(obs_after['torso_angle'])

    # Knee straight penalty - penalize any deviation from 0 (straight leg)
    # Knee angle 0 = fully straight, positive = bent backward, negative = hyperextended
    # We penalize abs(knee_angle) to encourage straighter legs
    left_knee_penalty = W_KNEE_BEND * abs(obs_after['left_knee'])
    right_knee_penalty = W_KNEE_BEND * abs(obs_after['right_knee'])

    # Energy cost - penalize large torque actions
    energy_cost = W_ENERGY * float(np.abs(ACTIONS[action_idx]).sum())

    # Calculate base reward
    r = W_FORWARD * dx + alive_bonus + torso_tilt_penalty + left_knee_penalty + right_knee_penalty + energy_cost

    # Add fall penalty if done (fallen)
    return r + FALL_PENALTY if done else r


def build_state_enhanced(obs: dict) -> np.ndarray:
    """Build state vector for neural network input.

    Uses original state features for transfer learning compatibility with
    pre-trained biped_dqn.pt, but still applies enhanced rewards for upright
    posture and straight legs.
    """
    return np.array([obs[f] / ORIGINAL_STATE_SCALE[f] for f in ORIGINAL_STATE_FEATURES], dtype=np.float32)


class QNet(nn.Module):
    """Q-network - architecture matches checkpoint for transfer learning compatibility.

    The checkpoint biped_dqn.pt was saved with a 3-layer architecture:
    - Linear(input_dim, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, output_dim)
    """

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

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """DQN agent with experience replay and target network."""

    def __init__(self, sd, na, dev):
        self.dev = dev
        self.q = QNet(sd, na).to(dev)
        self.t = QNet(sd, na).to(dev)
        self.t.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=LR)
        self.buf, self.g, self.steps = ReplayBuffer(REPLAY_CAP), GAMMA, 0

    def act(self, s, eps):
        """Epsilon-greedy action selection."""
        if random.random() < eps:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            return int(self.q(torch.tensor(s, dtype=torch.float32, device=self.dev).unsqueeze(0)).argmax().item())

    def remember(self, state, action: int, reward: float, next_state, done: bool):
        """Store experience in replay buffer."""
        self.buf.push(state, action, reward, next_state, done)

    def learn(self):
        """Update network from replay buffer."""
        if len(self.buf) < BATCH_SIZE:
            return
        s, a, r, s2, d = [x.to(self.dev) for x in self.buf.sample(BATCH_SIZE)]
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r + self.g * self.t(s2).max(1)[0] * (1 - d)
        nn.functional.mse_loss(q, target).backward()
        self.opt.step()
        self.opt.zero_grad()
        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.t.load_state_dict(self.q.state_dict())

    def save(self, path: str):
        torch.save(self.q.state_dict(), path)

    def load(self, path: str):
        self.q.load_state_dict(torch.load(path, map_location=self.dev))
        self.t.load_state_dict(self.q.state_dict())


def run_episode(api, agent, eps, train=True):
    """Run a single episode, optional training."""
    obs = api.reset_standing()
    s = build_state_enhanced(obs)
    total_r, stp, done = 0.0, 0, False

    while stp < MAX_STEPS and not done:
        a = agent.act(s, eps)
        api.apply_action_vector(ACTIONS[a])
        obs2 = api.step()

        # Check for fall
        done = bool(obs2['torso_contact']) or obs2['torso_y'] < 0.4 or abs(obs2['torso_angle']) > math.pi / 4

        r = calculate_enhanced_reward(obs, obs2, a, done)
        s2 = build_state_enhanced(obs2)

        if train:
            agent.remember(s, a, r, s2, done)
            if stp % TRAIN_EVERY == 0:
                agent.learn()

        total_r += r
        s, obs = s2, obs2
        stp += 1

    return total_r, stp, float(obs['displacement_x'])


def evaluate(api, agent, n=3, render=False):
    """Evaluate agent performance."""
    dist, steps = 0.0, 0
    for _ in range(n):
        obs = api.reset_standing()
        s = build_state_enhanced(obs)
        stp, done = 0, False

        while stp < MAX_STEPS and not done:
            s = build_state_enhanced(obs)
            a = agent.act(s, 0.0)  # Greedy
            api.apply_action_vector(ACTIONS[a])
            obs = api.step()

            if render:
                api.render()

            done = bool(obs['torso_contact']) or obs['torso_y'] < 0.4
            stp += 1

        dist += float(obs['displacement_x'])
        steps += stp

    return dist / n, steps / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--load', default=None)
    parser.add_argument('--save', default='heads_up_biped_dqn.pt')
    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[HeadsUp RL] Device: {device}")

    api = BotAPI(render_mode='headless')
    agent = DQNAgent(STATE_DIM, N_ACTIONS, device)

    # Transfer learning: load base biped_dqn.pt by default, override with --load
    base_model_path = args.load if args.load is not None else 'biped_dqn.pt'
    if os.path.exists(base_model_path):
        agent.load(base_model_path)
        if args.load is None:
            print(f"[HeadsUp RL] Transfer learning: Loaded base model from {base_model_path}")
        else:
            print(f"[HeadsUp RL] Loaded from: {base_model_path}")
    else:
        print(f"[HeadsUp RL] Model {base_model_path} not found, starting fresh")

    # Training loop
    eps = EPS_START
    best = -math.inf
    rewards = deque(maxlen=50)
    dists = deque(maxlen=50)
    t0 = time.time()

    print(f"[HeadsUp RL] Training {args.episodes} episodes...")
    for e in range(1, args.episodes + 1):
        r, st, d = run_episode(api, agent, eps)
        rewards.append(r)
        dists.append(d)

        if d > best:
            best = d
            agent.save(args.save)

        eps = max(EPS_END, EPS_START + (EPS_END - EPS_START) * e / EPS_DECAY)

        if e % 50 == 1:
            print(f"[ep {e:5d}] r={float(np.mean(rewards)):8.1f}  d={float(np.mean(dists)):.2f}m  best={best:.2f}m  eps={eps:.3f}  {time.time()-t0:5.0f}s")

    print(f"[HeadsUp RL] Done. Best: {best:.2f}m -> {args.save}")

    # Evaluation
    eval_api = api if not args.render else BotAPI(render_mode='human')
    avg_d, avg_s = evaluate(eval_api, agent, render=args.render)
    print(f"[HeadsUp RL] Eval: {avg_d:.2f}m, {avg_s:.0f} steps ({avg_s / 60:.1f}s)")
    eval_api.close()
    api.close()


if __name__ == '__main__':
    main()