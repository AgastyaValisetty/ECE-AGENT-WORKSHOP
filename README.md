# 2D Biped Physics Simulation & Reinforcement Learning

A minimal 4-joint planar biped robot under real rigid-body physics, built for the Agentic AI + ECE workshop. This project includes the physics simulation, a high-level RL API (`botapi.py`), and scripts to train the biped to walk using Deep Q-Learning.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the Simulation](#running-the-simulation)
- [Training the Biped to Walk (RL)](#training-the-biped-to-walk-rl)
- [Watching the Learned Policy](#watching-the-learned-policy)
- [API Reference (botAPI)](#api-reference-botapi)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
The simulation models a 2D biped with a torso, two thighs, and two shins (no feet) connected by revolute joints (hips and knees). The physics engine is **Pymunk** (a Python binding for Chipmunk2D), chosen for its ease of installation, headless operation, and excellent joint support.

The project provides:
- A deterministic physics simulation with configurable parameters.
- A clean RL-facing API (`botapi.py`) that abstracts away Pymunk and provides:
  - Reset and step functions.
  - Observations as dictionaries or flat NumPy vectors.
  - Torque-based actions (continuous or discrete via predefined actions).
  - Rendering options (headless for training, human for visualization).
- Training scripts using Deep Q-Learning (`basic_rl.py`).
- Pretrained models and utility scripts to visualize learned policies.

---

## Project Structure
```
.
├── biped_sim/                 # Core simulation and RL code
│   ├── main.py                # Entry point: simulation + rendering loop
│   ├── botapi.py              # High-level RL API (physics + input + RL interface)
│   ├── testscene.py           # Interactive scene (thin loop over BotAPI)
│   ├── basic_rl.py            # Train the biped to walk with Deep Q-Learning
│   ├── show_walk.py           # Play the trained walking policy in a window
│   ├── config/                # Configuration (robot dimensions, physics params)
│   ├── physics/               # Pymunk world and helpers
│   ├── robot/                 # Biped construction and joint definitions
│   ├── rendering/             # Pygame rendering and camera
│   ├── docs/                  # Explanation of RL training (states, actions, rewards)
│   └── tests/                 # Unit tests for the simulation and API
├── requirements.txt           # Core dependencies (numpy, torch, pymunk, pygame)
├── biped_dqn.pt               # Pretrained walking policy (from basic_rl.py)
├── heads_up_biped_dqn.pt      # Alternate pretrained policy (from heads_up_rl.py)
├── heads_up_rl.py             # Alternative RL training script (see comments)
└── README.md                  # This file
```

---

## Setup
> **Important:** The following steps will create an isolated virtual environment and install all required dependencies.

```bash
# 1. Install uv (if not already installed)
pip install uv

# 2. Install Python 3.12.6 using uv
uv python install 3.12.6

# 3. Create a virtual environment with Python 3.12.6
uv venv --python 3.12.6

# 4. Activate the virtual environment
#    - Windows PowerShell
.venv\Scripts\activate

# 5. Install project dependencies from the root requirements.txt
uv pip install -r requirements.txt

# 6. (Optional) Fix PowerShell execution policy if needed
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 7. Reactivate the virtual environment (helps when PowerShell blocks execution)
.venv\Scripts\activate
```

### Why the double activation?
If your PowerShell profile restricts script execution, the second `activate` ensures the environment is properly loaded and usable.

---

## Running the Simulation
To run the bare simulation (no RL, just physics visualization):

```bash
# Change into the simulation directory
cd biped_sim

# Run the main simulation with GUI
python main.py

# Or run the interactive test scene (more controls)
python testscene.py                    # GUI: R reset, SPACE pause, UP/DOWN speed,
                                       #      T toggle demo torques, B torque burst,
                                       #      ESC exit

# Run headless (for CI/testing or to capture frames)
python testscene.py --frames 240 --headless
```

**Controls in `testscene.py`:**
- `R` — reset the robot to spawn position (drop test)
- `SPACE` — pause / unpause
- `UP` / `DOWN` — increase / decrease simulation speed
- `T` — toggle sinusoidal demo torques (standing reset)
- `B` — toggle torque burst (standing reset)
- `ESC` — exit

---

## Training the Biped to Walk (RL)
Train a Deep Q-Learning agent to make the biped walk:

```bash
cd biped_sim

# Train for 5000 episodes (default), saving the best model to biped_dqn.pt
python basic_rl.py

# Train with visualization during the final evaluation
python basic_rl.py --render

# Train and save to a custom path
python basic_rl.py --save my_walking_policy.pt

# Continue training from a pretrained model
python basic_rl.py --load biped_dqn.pt --save biped_dqn_finetuned.pt
```

See `biped_sim/docs/RL_TRAINING.md` for a detailed explanation of:
- State representation (28 values)
- Action space (5 discrete torques combinations)
- Reward function
- Model architecture and hyperparameters

---

## Watching the Learned Policy
After training (or using a pretrained model), watch the biped walk:

```bash
cd biped_sim

# Watch the policy learned during training (uses biped_dqn.pt by default)
python show_walk.py

# Watch a specific model
python show_walk.py --model biped_dqn.pt

# The show_walk.py script will:
#   - Load the policy
#   - Repeatedly reset the robot (when it falls) and run the learned policy
#   - Display the simulation in a Pygame window
```

---

## API Reference (botAPI)
The `botapi.py` module is the single interface between an agent/RL system and the physics simulation. It abstracts away Pymunk and provides a minimal set of operations.

### Key Classes and Functions
- **`BotAPI(render_mode='headless', ...)`**
  - Creates the API instance.
  - `render_mode`: `'headless'` (no window, for training) or `'human'` (opens a Pygame window).
  - Other arguments: `spawn_x`, `spawn_y`, `max_torque`, `burst_torque`, `substeps`.

- **`reset()` → dict**
  - Resets the robot to its spawn pose (in the air for the drop test) and zeroes velocities.
  - Returns the initial observation dictionary.

- **`reset_standing()` → dict**
  - Resets the robot to a standing pose with shins on the ground (useful for control demos).

- **`step(action=None) → dict`**
  - Applies an action (torques per joint) and advances physics by one control step.
  - If `action` is `None`, zero torque is applied (free-fall).
  - Returns the updated observation dictionary.

- **`apply_torques(torques_dict)`**
  - Applies torque to each named joint (clamped to `max_torque`).
  - Example: `api.apply_torques({'left_hip': 20.0, 'right_knee': -10.0})`

- **`apply_action_vector(np_array)`**
  - Applies torques from a flat array in the order `['left_hip', 'left_knee', 'right_hip', 'right_knee']`.

- **`get_state_vector()` → np.ndarray**
  - Returns a flat float32 observation vector (59 values) ordered by `STATE_VECTOR_ORDER` (see below).
  - Convenient for feeding directly to an RL policy network.

- **`get_action_space()` → dict**
  - Describes the continuous action space (for RL policy setup).
  - Returns joint names, low/high torque bounds, shape, dtype, and description.

- **`close()`**
  - Releases the renderer (if any). Call when done with the API.

- **Context Manager Support**
  - `with BotAPI(...) as api:` automatically calls `close()` at the end.

### Observation State (Dictionary Keys)
The `observe()` method returns a dictionary with the following keys (all Python floats):

| Group | Fields |
|-------|--------|
| Pose | `time`, `torso_x`, `torso_y`, `torso_angle` |
| Displacement | `displacement_x`, `displacement_y` (from spawn) |
| Velocity | `torso_vx`, `torso_vy`, `torso_omega` |
| Joint angles (rad) | `left_hip`, `left_knee`, `right_hip`, `right_knee` |
| Joint angular velocity | `left_hip_omega`, `left_knee_omega`, `right_hip_omega`, `right_knee_omega` |
| Joint angular acceleration | `left_hip_alpha`, `left_knee_alpha`, `right_hip_alpha`, `right_knee_alpha` |
| Applied torque | `left_hip_torque`, `left_knee_torque`, `right_hip_torque`, `right_knee_torque` |
| Contact flags | `left_contact`, `right_contact`, `torso_contact` (1.0 = touching, 0.0 = airborne) |

### State Vector Order (for `get_state_vector()`)
The flat vector returns values in this exact order:
1. `time`
2. `torso_x`, `torso_y`, `torso_angle`
3. `displacement_x`, `displacement_y`
4. `torso_vx`, `torso_vy`, `torso_omega`
5. `left_hip`, `left_knee`, `right_hip`, `right_knee`
6. `left_hip_omega`, `left_knee_omega`, `right_hip_omega`, `right_knee_omega`
7. `left_hip_alpha`, `left_knee_alpha`, `right_hip_alpha`, `right_knee_alpha`
8. `left_hip_torque`, `left_knee_torque`, `right_hip_torque`, `right_knee_torque`
9. `left_contact`, `right_contact`, `torso_contact`

Total length: **59** values.

### Action Space
- **Continuous**: Torque per joint in `[-max_torque, max_torque]` (default ±50 N·m).
- **Discrete (used in `basic_rl.py`)**: 5 predefined actions:
  0. `[0, 0, 0, 0]` (REST)
  1. `[30, -20, 30, -20]` (SQUAT)
  2. `[-30, 20, -30, 20]` (SPRING)
  3. `[30, 20, -30, 20]` (STEP LEFT)
  4. `[-30, 20, 30, 20]` (STEP RIGHT)
  5. `[30, 0, 30, 0]` (LEAN FORWARD)

---

## Contributing
Contributions are welcome! Please follow the standard fork‑branch‑pull‑request workflow.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feat/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feat/your-feature`).
5. Open a Pull Request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

--- 

*Happy training!* 🚶‍♂️💨