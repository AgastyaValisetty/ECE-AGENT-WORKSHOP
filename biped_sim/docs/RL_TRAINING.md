# RL Training Guide — `basic_rl.py`

`basic_rl.py` trains the 2D biped to **walk forward** using **Deep Q-Learning
(DQN)** — a Q-learning algorithm where a neural network approximates the
Q-table. Everything runs through `BotAPI`, so the RL code never touches the
physics directly.

---

## 1. High-level idea (how Q-learning works)

Q-learning learns a function `Q(s, a)` = *"expected total future reward if I
take action `a` in state `s`, then keep acting well."*

At each step the agent:

1. Reads the **state** `s` (how the robot is posed / moving).
2. Picks an **action** `a` (a torque command for the joints) — usually the one
   with the highest `Q(s, a)`.
3. The physics runs one step → new state `s'`, and the world gives a **reward** `r`.
4. The agent updates `Q(s, a)` toward the **Bellman target**:

```
Q(s,a) ← Q(s,a) + α · [ r + γ·max_a' Q(s',a') − Q(s,a) ]
```

| Symbol | Meaning |
|--------|---------|
| α | learning rate (`LR`) — how fast the estimate moves |
| γ | discount factor (`GAMMA`) — how much future reward matters |
| r | immediate reward from the environment |
| max_a' Q(s',a') | best estimated value of the next state |

In **DQN**, `Q(s,a)` is computed by a neural network instead of a lookup table.
This lets us feed in continuous state (angles, velocities) without binning.

---

## 2. Environment interface (what the agent sees & does)

The agent interacts through **`BotAPI`** (headless mode):

- `api.reset_standing()` — start an episode with the robot upright on the ground.
- `api.apply_action_vector(torques)` — send a 4-value torque command.
- `api.step()` — advance physics one fixed step (1/60 s).
- `api.get_state()` — full observation dict (for building the RL state).

---

## 3. State (what is passed into the network)

We take **14 features** from the BotAPI observation and normalize each one so
the network gets well-scaled inputs.

| # | Feature | Description | Scale |
|---|---------|-------------|-------|
| 1 | `torso_angle` | torso lean (rad) | ÷ π |
| 2 | `torso_omega` | torso angular velocity (rad/s) | ÷ 10 |
| 3 | `torso_vx` | forward (x) velocity (m/s) | ÷ 5 |
| 4 | `torso_vy` | vertical velocity (m/s) | ÷ 5 |
| 5–8 | `left_hip, left_knee, right_hip, right_knee` | joint angles (rad) | ÷ π |
| 9–12 | `*_omega` for the 4 joints | joint angular velocities (rad/s) | ÷ 10 |
| 13–14 | `left_contact, right_contact` | is that leg touching the ground? | × 1 |

So the state vector is 14 floats, e.g.:

```
state = [torso_angle/π, torso_omega/10, torso_vx/5, torso_vy/5,
         left_hip/π, left_knee/π, right_hip/π, right_knee/π,
         left_hip_omega/10, left_knee_omega/10,
         right_hip_omega/10, right_knee_omega/10,
         left_contact, right_contact]
```

This is enough information for the policy to know how the robot is posed,
moving, and whether each leg is planted — i.e., a Markov-ish state for walking.

---

## 4. Actions (what the agent can command)

A **discrete** action space of **6 "motor programs"**. Each action is a torque
command (N·m) for the 4 joints in order
`[left_hip, left_knee, right_hip, right_knee]`:

| ID | Name | left_hip | left_knee | right_hip | right_knee | Intent |
|----|------|----------|-----------|-----------|------------|--------|
| 0 | REST | 0 | 0 | 0 | 0 | do nothing |
| 1 | SQUAT | +30 | −20 | +30 | −20 | crouch (bend knees) |
| 2 | SPRING | −30 | +20 | −30 | +20 | straighten / push up |
| 3 | STEP LEFT | +30 | +20 | −30 | +20 | swing left leg forward |
| 4 | STEP RIGHT | −30 | +20 | +30 | +20 | swing right leg forward |
| 5 | LEAN FWD | +30 | 0 | +30 | 0 | push forward |

All values are inside BotAPI's ±50 N·m clamp, so the commanded torques are
applied exactly. The agent chooses one action per step (`agent.act`).

---

## 5. Reward function (how "good" is calculated)

Each physics step produces one reward. The formula in `compute_reward` is:

```
r = W_FORWARD · Δx
  + W_ALIVE
  + W_TILT · |torso_angle|
  + W_ENERGY · Σ|torques|
  + (FALL_PENALTY  if the robot fell this step)
```

| Weight | Value | What it rewards / punishes |
|--------|-------|----------------------------|
| `W_FORWARD` | +30.0 | per meter the torso moves **forward** (Δx this step) |
| `W_ALIVE` | +0.1 | surviving another step |
| `W_TILT` | −0.05 | per radian of torso tilt (encourages staying upright) |
| `W_ENERGY` | −0.0005 | tiny penalty per N·m of torque (discourages flailing) |
| `FALL_PENALTY` | −50.0 | big penalty when the torso hits the ground (episode ends) |

An episode **ends** when the torso touches the ground (`torso_contact`) or the
torso drops below y = 0.4 m, or after `MAX_STEPS` = 300 steps (~5 s).

Why these numbers: forward progress is the dominant signal (so the agent must
move to get reward), falling is punished hard (so it must stay upright), and
energy/tilt penalties are mild shaping that keep the motion sensible.

---

## 6. Model (the neural network)

A small **MLP** (`QNetwork`):

```
state (14) ──► Linear(14→128) ──► ReLU ──► Linear(128→128) ──► ReLU ──► Linear(128→6)
                                                                          └──► Q(s, a0..a5)
```

The output layer has one value per action — the predicted future reward for
taking that action in the current state. Two copies are kept:

- **online network** (`q`) — updated every few steps by gradient descent.
- **target network** (`target`) — a frozen copy refreshed every
  `TARGET_UPDATE` steps, used to compute the Bellman target (this stabilizes
  learning).

Training uses **Adam** on the **mean squared error** between `Q(s,a)` and the
Bellman target.

---

## 7. Training hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Episodes | 5000 (default) | number of training runs |
| Steps per episode | ≤ 300 | ~5 s of simulated time |
| `GAMMA` (γ) | 0.99 | discount factor |
| `LR` (α) | 0.001 | Adam learning rate |
| Batch size | 64 | transitions per gradient step |
| Replay buffer | 100 000 | memory of past experiences |
| `TARGET_UPDATE` | 200 steps | target-network refresh period |
| `TRAIN_EVERY` | 4 steps | gradient step frequency |
| ε start | 1.0 | fully random exploration at first |
| ε end | 0.05 | 5% exploration after decay |
| ε decay | over first 2000 episodes | linear |

**Exploration vs exploitation (ε-greedy):** with probability ε the agent picks a
random action (exploration); otherwise it picks the action with the highest
Q-value (exploitation). ε starts at 1.0 (mostly random) and decays to 0.05.

**Replay buffer:** past `(s, a, r, s', done)` transitions are stored and sampled
randomly, which breaks the correlation between consecutive steps and makes
training stable.

---

## 8. Device auto-detection (GPU / CPU)

At startup `basic_rl.py` picks the best available device:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

If a CUDA-capable GPU is present, the network and every training tensor are
moved to it (`.to(device)`), so training is much faster. Otherwise it falls
back to CPU automatically — no code changes needed.

---

## 9. How to run

```bash
# Train for 5000 episodes (default), save the best model to biped_dqn.pt
python basic_rl.py

# Train fewer episodes
python basic_rl.py --episodes 2000

# Train then watch the result (opens the Pygame window)
python basic_rl.py --render

# Skip training and just watch a saved policy
python basic_rl.py --load biped_dqn.pt --render

# Reproduce a run
python basic_rl.py --seed 7
```

Progress is printed every 50 episodes: average reward, average distance,
best distance, current ε, and elapsed time.

---

## 10. Function reference (`basic_rl.py`)

| Function / class | Purpose |
|------------------|---------|
| `ACTIONS` | the 6 discrete torque programs |
| `STATE_FEATURES` | the 14 state features (order matters) |
| `QNetwork` | the small MLP: state → Q-values per action |
| `ReplayBuffer` | stores and samples past experiences |
| `extract_state(obs)` | builds the normalized 14-value state from a BotAPI observation |
| `compute_reward(obs_before, obs_after, action_idx, done)` | the one-step reward formula (§5) |
| `DQNAgent` | Q-learning agent: `act` (ε-greedy), `remember`, `learn` (Bellman update), `save`, `load` |
| `run_episode(api, agent, epsilon, train)` | one training episode; returns (reward, steps, distance) |
| `evaluate(api, agent, episodes, render)` | greedy (ε=0) rollouts to measure walking ability |
| `main()` | CLI, device detection, training loop, evaluation |

---

## 11. Expected behavior & honesty note

- Over 5000 episodes the robot typically learns to **shuffle / hop forward**,
  reaching **0.5–2+ meters** of forward distance before falling, and surviving
  several seconds upright. It is not a graceful human-like walk — that needs
  feet, ankles, and more training — but it is real, learned forward locomotion
  via Q-learning.
- If a run doesn't learn (bad luck / a difficult seed), just re-run with a
  different `--seed` or increase `--episodes`.
- `--render` shows the final policy; press `R`/`SPACE`/`ESC` in the window
  (handled by `BotAPI.handle_events`) or close it to finish.

---

## 12. Files

- `basic_rl.py` — the training script.
- `botapi.py` — the environment interface the agent uses.
- `biped_dqn.pt` — the saved best model (created during training).