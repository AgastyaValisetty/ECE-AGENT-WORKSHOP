# 2D Biped Physics Simulation

A minimal 4-joint planar biped robot under real rigid-body physics, built for the
Agentic AI + ECE workshop. This milestone establishes the physical simulation only —
no AI, RL, or control yet.

## Chosen Physics Engine: **Pymunk**

**Why Pymunk?**

| Criterion | Evaluation |
|-----------|-----------|
| Rigid bodies | ✅ Box polygons, circles, segments |
| Revolute joints | ✅ `PivotJoint` (pin joint) |
| Joint limits | ✅ `RotaryLimitJoint` for angular limits |
| Collision detection | ✅ Broad/narrow phase, shape filters, categories |
| Gravity | ✅ `space.gravity` |
| Friction & restitution | ✅ Per-shape `friction` / `elasticity` |
| Torque/motor control | ✅ Direct body torque + `SimpleMotor` available |
| Stability | ✅ Mature, battle-tested (Chipmunk2D) |
| Python-native | ✅ Pure pip install, no build tools needed |
| Headless | ✅ Works without display for training |

Box2D is also a strong option but requires separate compilation on some platforms.
Pymunk is a mature Python binding for Chipmunk2D with excellent joint support and
trivially simple installation.

## Robot Structure

```
             TORSO
           ┌────────┐
           │        │
           └────────┘
             /    \
           HIP    HIP
           /        \
       THIGH       THIGH
         |           |
       KNEE        KNEE
         |           |
        SHIN        SHIN
```

**Bodies (5 total):**
- 1 torso
- 2 thighs
- 2 shins

**Joints (4 total, all revolute):**
- left hip (torso ↔ left thigh)
- left knee (left thigh ↔ left shin)
- right hip (torso ↔ right thigh)
- right knee (right thigh ↔ right shin)

The legs end at the shins (no feet). The shins contact the ground directly.

## Physics Parameters

All parameters are centralized in `config/robot_config.py`:

| Parameter | Value |
|-----------|-------|
| Gravity | (0, -9.81) m/s² |
| Timestep | 1/60 s |
| Spawn height | 5.0 m |
| Torso | 0.4 × 0.6 m, 8.0 kg |
| Thigh | 0.15 × 0.45 m, 2.5 kg |
| Shin | 0.12 × 0.45 m, 1.5 kg |
| Hip limits | ±1.2 rad (~±69°) |
| Knee limits | -0.1 to 2.5 rad |
| Ground friction | 0.8 |
| Ground restitution | 0.1 |
| Body friction | 0.6 |
| Body restitution | 0.0 |

## Coordinate System

- **X**: right (positive), meters
- **Y**: up (positive), meters
- **Z**: out of screen (Pymunk is 2D)
- Origin (0,0) at the ground surface
- Angles in radians, CCW positive

## Installation

```bash
pip install pymunk pygame
```

Requires Python 3.8+.

## Running the Simulation

```bash
cd biped_sim
python main.py
```

**Interactive test scene (through BotAPI):**

```bash
python testscene.py                    # GUI: R reset, SPACE pause, UP/DOWN speed,
                                       #      T toggle demo torques, B torque burst,
                                       #      ESC exit
python testscene.py --frames 240 --headless   # headless, for CI/testing
```

**Train the biped to walk (Deep Q-Learning):**

```bash
python basic_rl.py                     # train 5000 episodes (auto GPU/CPU)
python basic_rl.py --render            # train, then watch the learned policy
python basic_rl.py --load biped_dqn.pt --render   # watch a saved policy
```

**Watch the trained bot walk:**

```bash
python show_walk.py                    # plays the learned policy in a window;
                                       # auto-stands-up after each fall
```

See `docs/RL_TRAINING.md` for the full explanation of state, actions, reward,
model, and hyperparameters.

**Controls:**
- `R` — reset the robot to spawn position
- `SPACE` — pause / unpause
- `UP` / `DOWN` — increase / decrease simulation speed
- `ESC` — exit

## High-Level RL API (`botapi.py`)

`botapi.py` is the single clean interface between an agent/RL system and the
physics simulation. It wraps the world, robot, contact detection, and rendering
into one facade — callers never touch Pymunk directly.

```python
from botapi import BotAPI

api = BotAPI(render_mode='headless')   # 'headless' for training, 'human' for GUI
obs = api.reset()                      # dict observation

for _ in range(1000):
    action = {'left_hip': 20.0, 'left_knee': -10.0,
              'right_hip': -20.0, 'right_knee': 10.0}
    obs = api.step(action)             # apply torques + advance physics
    vec = api.get_state_vector()       # flat float32 vector for RL policies
    print(obs['torso_y'], obs['left_hip_omega'], vec.shape)

api.close()                            # or use `with BotAPI(...) as api:`
```

**Observation state (28 values)** — `observe()` / `get_state_vector()`:

| Group | Fields |
|-------|--------|
| Pose | `time`, `torso_x`, `torso_y`, `torso_angle` |
| Displacement | `displacement_x`, `displacement_y` (from spawn) |
| Velocity | `torso_vx`, `torso_vy`, `torso_omega` |
| Joint angles (rad) | `left_hip`, `left_knee`, `right_hip`, `right_knee` |
| Joint angular velocity | `*_omega` (rad/s) |
| Joint angular acceleration | `*_alpha` (rad/s², finite difference) |
| Applied torque | `*_torque` (N·m, last action) |
| Contact flags | `left_contact`, `right_contact`, `torso_contact` |

**Actions (4 continuous torques):** `apply_torques({...})` or
`apply_action_vector(np.ndarray)`; clamped to `max_torque` (default ±50 N·m).
See `get_action_space()` for the space spec.

**Spawn poses:** `reset()` drops the robot from height (drop test);
`reset_standing()` places the straightened legs on the ground — useful for
control/demo scenarios where you want to see the limbs move (e.g., the
sinusoidal leg-drive demo in `testscene.py`).

**Torque burst (diagnostic):** `apply_torque_burst()` slams ALL joints with a
large alternating torque (`burst_torque`, default 200 N·m) so you can visually
confirm the torque channel works. In `testscene.py`, press **B** to toggle it.

Other helpers: `get_joint_angles()`, `get_joint_angular_velocities()`,
`get_joint_angular_accelerations()`, `get_joint_torques()`,
`get_torso_state()`, `get_displacement()`, `get_contacts()`, `get_sim_time()`,
`render()`.

Run the self-test: `python botapi.py`

## Running Tests

```bash
cd biped_sim
python -m unittest discover tests -v
```

Or run individually:

```bash
python tests/test_world.py
python tests/test_robot.py
python tests/test_drop.py
python tests/test_botapi.py
```

## How the Drop Test Works

1. Robot is spawned with its torso centered 5 m above the ground.
2. All initial velocities and angular velocities are zero.
3. No control forces are applied.
4. Gravity accelerates all bodies downward.
5. Joints keep the 5 bodies connected as the robot falls.
6. Shins/limbs collide with the ground; friction and restitution dissipate energy.
7. The robot tumbles naturally and eventually settles.

The simulation uses a fixed 1/60 s timestep, so physics is deterministic and
independent of rendering frame rate.

## Project Structure

```
biped_sim/
├── main.py                 # Entry point: world + robot + render loop
├── botapi.py               # High-level API (physics + input + RL interface)
├── testscene.py            # Interactive scene (thin loop over BotAPI)
├── basic_rl.py             # Train the biped to walk with Deep Q-Learning
├── show_walk.py            # Play the trained walking policy in a window
├── config/
│   ├── __init__.py
│   └── robot_config.py     # ALL physical parameters
├── physics/
│   ├── __init__.py
│   ├── world.py            # Pymunk space + ground
│   └── constants.py        # Shape/joint factory helpers
├── robot/
│   ├── __init__.py
│   └── biped.py            # Biped construction, joints, reset
├── rendering/
│   ├── __init__.py
│   ├── camera.py           # world <-> screen transform
│   └── renderer.py         # Pygame drawing
├── docs/
│   └── RL_TRAINING.md      # State/action/reward/model explanation for basic_rl.py
└── tests/
    ├── test_world.py
    ├── test_robot.py
    ├── test_drop.py
    └── test_botapi.py      # API behavior tests
```

## Collision Behavior

- Robot parts collide with the **ground** (shins support the body).
- Robot parts **do not collide with each other** (self-collision disabled via
  Pymunk shape filters). This prevents internal limb contacts from destabilizing
  the articulated structure during the drop and will keep the gait stable later.
  Only the ground shape is in the `CAT_GROUND` category; all robot parts are
  `CAT_ROBOT`.

## Future Extension Points for RL

- **`BotAPI` (botapi.py)** is the ready-made RL interface: reset/step/observe,
  dict + vector states, torque actions, contact flags, angular acceleration.
- `Biped.get_joint_angles()` — raw joint observation source
- `Biped.apply_joint_torques({...})` — direct torque application on bodies
- Joint motors via `pymunk.SimpleMotor` can be attached at each joint instead
- The renderer already separates camera/world transforms for camera-follow
- `BotAPI(render_mode='headless')` runs without a display for fast training
