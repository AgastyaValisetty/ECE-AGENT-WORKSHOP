# Deep Q-Learning for Biped Walking - Explanation

This document explains every function and concept in `basic_rl.py` for teaching Deep Q-Learning to workshop students.

## Overview

This script trains a 2D biped robot to walk using Deep Q-Network (DQN) reinforcement learning. The agent learns to apply torques to the robot's joints to achieve forward motion.

## Key Components

### 1. State Space (14 Features)

The observation vector provides the agent with:

```python
STATE_FEATURES = [
    'torso_angle', 'torso_omega', 'torso_vx', 'torso_vy',    # Torso pose & velocity
    'left_hip', 'left_knee', 'right_hip', 'right_knee',      # Joint angles (rad)
    'left_hip_omega', 'left_knee_omega',                     # Left joint velocities
    'right_hip_omega', 'right_knee_omega',                   # Right joint velocities
    'left_contact', 'right_contact',                          # Ground contact flags
]
```

These 14 continuous values are normalized before being fed to the neural network.

### 2. Action Space (6 Discrete Actions)

The agent chooses from 6 "motor programs":

| Action | Name | Left Hip | Left Knee | Right Hip | Right Knee | Purpose |
|--------|------|----------|-----------|-----------|------------|---------|
| 0 | REST | 0 | 0 | 0 | 0 | Do nothing |
| 1 | SQUAT | +30 | -20 | +30 | -20 | Bend knees |
| 2 | SPRING | -30 | +20 | -30 | +20 | Push up |
| 3 | STEP LEFT | +30 | +20 | -30 | +20 | Swing left leg |
| 4 | STEP RIGHT | -30 | +20 | +30 | +20 | Swing right leg |
| 5 | LEAN FORWARD | +30 | 0 | +30 | 0 | Push forward |

Each value is a torque in N·m.

### 3. Reward Function

The reward for each step is computed by `compute_reward()`:

```python
r = W_FORWARD * Δx + W_ALIVE + W_TILT * |angle| + W_ENERGY * Σ|torque| + FALL_PENALTY
```

- **W_FORWARD = 30.0**: Reward for forward displacement (Δx)
- **W_ALIVE = 0.1**: Small reward for surviving each step
- **W_TILT = -0.05**: Penalty for torso lean (encourages upright posture)
- **W_ENERGY = -0.0005**: Penalty for energy usage (discourages flailing)
- **FALL_PENALTY = -100.0**: Large penalty when robot falls

## Function Reference

### `QNetwork(state_dim, n_actions, hidden=128)`

A simple Multi-Layer Perceptron (MLP) that approximates the Q-function Q(s, a).

**Architecture:**
```
Input (14) → Linear(14→128) → ReLU → Linear(128→128) → ReLU → Linear(128→6) → Output (Q-values for 6 actions)
```

**Parameters:**
- `state_dim`: Size of state vector (14)
- `n_actions`: Number of discrete actions (6)
- `hidden`: Hidden layer size (default 128)

### `ReplayBuffer(capacity)`

A circular buffer storing past experiences for replay.

**Methods:**
- `push(s, a, r, s2, done)`: Store a transition
- `sample(batch_size)`: Get random minibatch
- `__len__()`: Current buffer size

**Why replay?** Sampling random minibatches breaks correlation between consecutive steps, making training more stable.

### `extract_state(obs)`

Converts a BotAPI observation dict into a normalized numpy array for the neural network.

```python
def extract_state(obs):
    return np.array([obs[f] / STATE_SCALE[f] for f in STATE_FEATURES], dtype=np.float32)
```

Each feature is divided by a scale factor to keep values roughly in [0, 1].

### `compute_reward(obs_before, obs_after, action_idx, done)`

Computes the reward for one step.

**Key insight:** The reward uses `Δx` (change in x position), not absolute position, so the agent learns to move forward regardless of starting position.

### `DQNAgent(state_dim, n_actions, device)`

The main agent implementing DQN:
- Uses an online Q-network (`self.q`) for action selection
- Uses a target network (`self.target`) for stable Q-targets
- Stores experiences in a replay buffer for learning

#### `act(state, epsilon)` - Action Selection

Uses epsilon-greedy policy:
- With probability ε: choose a random action (exploration)
- Otherwise: choose the action with highest Q-value (exploitation)

```python
def act(self, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)  # Explore
    with torch.no_grad():
        q = self.q(state)
        return int(q.argmax(dim=1).item())  # Exploit
```

#### `learn()` - Network Update

Implements the Bellman update:

```python
target_q = r + γ * max_a' Q_target(s', a') * (1 - done)
loss = MSE(Q_live(s, a), target_q)
```

- Computes targets using the target network (frozen copy)
- Updates the online network via gradient descent
- Periodically syncs target network with online network

### `run_episode(api, agent, epsilon, train=True, max_steps=MAX_STEPS)`

Runs one training episode.

**Flow:**
1. Reset robot to standing position
2. Loop for up to `max_steps` (300 steps = ~5 seconds):
   - Select action (ε-greedy)
   - Apply action and step physics
   - Compute reward
   - Store experience (if training)
   - Learn from replay buffer (every 4 steps)
   - Check termination (fell or timeout)
3. Return statistics: total reward, steps survived, distance walked

### `evaluate(api, agent, episodes=3, render=False)`

Evaluate the trained policy.

**Flow:**
1. Reset to standing position
2. For each episode:
   - Use greedy policy (ε=0)
   - Step simulation
   - Render if requested
   - Track distance and steps
3. Return average distance and survival time

### `main()`

The main training loop:

1. Parse command-line arguments
2. Initialize environment (BotAPI) and agent (DQN)
3. Training phase (if no `--load`):
   - Linearly decay ε from 1.0 to 0.05 over 2000 episodes
   - Print progress every 50 episodes
4. Evaluation phase:
   - Run greedy policy (ε=0)
   - Report average walking distance

## Key Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `GAMMA` | 0.99 | Discount factor for future rewards |
| `LR` | 1e-3 | Adam learning rate |
| `BATCH_SIZE` | 256 | Minibatch size for training |
| `REPLAY_CAPACITY` | 100,000 | Replay buffer size |
| `TARGET_UPDATE` | 200 | Steps between target network updates |
| `TRAIN_EVERY` | 4 | Learn every N steps |
| `EPSILON_START` | 1.0 | Initial exploration rate |
| `EPSILON_END` | 0.05 | Final exploration rate |
| `EPSILON_DECAY_EPISODES` | 2000 | Episodes to decay ε |

## Training Strategy

```
for episode in range(num_episodes):
    epsilon = linear_decay(episode)
    run_episode(epsilon)
    if step % 4 == 0:
        learn_from_replay_buffer()
    if step % 200 == 0:
        sync_target_network()
```

## Episode Termination

An episode ends when:
1. Torso touches the ground (`torso_contact`)
2. Torso drops below y = 0.4 meters
3. 300 steps have passed (natural timeout)

## Expected Learning Behavior

With sufficient episodes (5000+), the agent should learn to:
- Keep torso upright (avoid tilt penalty)
- Use periodic hip/knee torques for balance
- Generate forward motion through leg coordination
- Walk or shuffle forward at 0.5-2+ meters before falling

The learned policy isn't a graceful human walk (that requires feet and ankles), but it demonstrates successful reinforcement learning of dynamic locomotion.