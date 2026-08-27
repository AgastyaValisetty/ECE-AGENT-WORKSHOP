"""
botapi.py — High-level RL-facing API for the 2D biped.

This module is the SINGLE clean interface between an agent/RL system and the
underlying physics simulation. It abstracts away Pymunk, world creation, robot
construction, contact detection, and rendering.

Design principles:
- Single responsibility: this file is the only place the agent talks to.
- Small surface: reset / step / observe / apply_torques / render / close.
- No leaked internals: callers never touch Pymunk bodies or the Space directly.
- Two observation views: a readable dict (for humans/debugging) and a flat
  NumPy vector (for RL policies), both derived from one source of truth.

Usage:
    from botapi import BotAPI

    api = BotAPI(render_mode='headless')
    obs = api.reset()
    for _ in range(1000):
        action = {'left_hip': 10.0, 'left_knee': -5.0,
                  'right_hip': -10.0, 'right_knee': 5.0}
        obs = api.step(action)          # returns updated state dict
    api.close()

    # Or flat-vector style for RL:
    vec = api.get_state_vector()        # np.ndarray, see STATE_VECTOR_ORDER
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pygame
import pymunk

from config.robot_config import config
from physics.constants import CAT_GROUND, CAT_ROBOT
from physics.world import PhysicsWorld
from robot.biped import Biped
from rendering.camera import Camera
from rendering.renderer import Renderer

# The four controllable joints, in canonical order.
JOINT_NAMES: List[str] = ['left_hip', 'left_knee', 'right_hip', 'right_knee']

# Canonical order of the flat observation vector (see BotAPI.get_state_vector).
STATE_VECTOR_ORDER: List[str] = [
    'time',
    'torso_x', 'torso_y', 'torso_angle',
    'displacement_x', 'displacement_y',
    'torso_vx', 'torso_vy', 'torso_omega',
    'left_hip', 'left_knee', 'right_hip', 'right_knee',
    'left_hip_omega', 'left_knee_omega', 'right_hip_omega', 'right_knee_omega',
    'left_hip_alpha', 'left_knee_alpha', 'right_hip_alpha', 'right_knee_alpha',
    'left_hip_torque', 'left_knee_torque', 'right_hip_torque', 'right_knee_torque',
    'left_contact', 'right_contact', 'torso_contact',
]


# ---------------------------------------------------------------------------
# RL environment: action definitions, state spec, reward, and replay buffer.
# Everything an RL agent needs to interact with the biped environment.
# ---------------------------------------------------------------------------

import random
from collections import deque
import torch
import torch.nn as nn

ACTIONS = [
    np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),         # 0 REST
    np.array([30.0, -20.0, 30.0, -20.0], dtype=np.float32),    # 1 SQUAT
    np.array([-30.0, 20.0, -30.0, 20.0], dtype=np.float32),    # 2 SPRING
    np.array([30.0, 20.0, -30.0, 20.0], dtype=np.float32),     # 3 STEP LEFT
    np.array([-30.0, 20.0, 30.0, 20.0], dtype=np.float32),     # 4 STEP RIGHT
    np.array([30.0, 0.0, 30.0, 0.0], dtype=np.float32),        # 5 LEAN FORWARD
]
N_ACTIONS = len(ACTIONS)

STATE_FEATURES = [
    'torso_angle', 'torso_omega', 'torso_vx', 'torso_vy',
    'left_hip', 'left_knee', 'right_hip', 'right_knee',
    'left_hip_omega', 'left_knee_omega', 'right_hip_omega', 'right_knee_omega',
    'left_contact', 'right_contact',
]
STATE_DIM = len(STATE_FEATURES)
STATE_SCALE = {
    'torso_angle': math.pi, 'torso_omega': 10.0, 'torso_vx': 5.0, 'torso_vy': 5.0,
    'left_hip': math.pi, 'left_knee': math.pi, 'right_hip': math.pi, 'right_knee': math.pi,
    'left_hip_omega': 10.0, 'left_knee_omega': 10.0, 'right_hip_omega': 10.0, 'right_knee_omega': 10.0,
    'left_contact': 1.0, 'right_contact': 1.0,
}
W_FORWARD, W_ALIVE, W_TILT, W_ENERGY, FALL_PENALTY, MAX_STEPS = 30.0, 0.1, -0.05, -0.0005, -100.0, 300


def build_state(obs: dict) -> np.ndarray:
    """Normalize observation dict into a flat float32 array for neural network input."""
    return np.array([obs[f] / STATE_SCALE[f] for f in STATE_FEATURES], dtype=np.float32)


def calculate_reward(obs_before: dict, obs_after: dict, action_idx: int, done: bool) -> float:
    """Reward = forward progress + survival - tilt - energy. Big penalty if fallen."""
    dx = obs_after['torso_x'] - obs_before['torso_x']
    tilt = abs(obs_after['torso_angle'])
    energy = float(np.abs(ACTIONS[action_idx]).sum())
    r = W_FORWARD * dx + W_ALIVE + W_TILT * tilt + W_ENERGY * energy
    return r + FALL_PENALTY if done else r


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


class _ContactSensor:
    """Tracks which robot parts are currently touching the ground.

    Uses Pymunk collision callbacks, so contact state is derived from real
    collision events rather than geometric approximations. This is internal to
    the API and not exposed to callers.
    """

    def __init__(self, space: pymunk.Space) -> None:
        self._contacting: set = set()
        space.on_collision(
            CAT_ROBOT, CAT_GROUND,
            begin=self._begin,
            separate=self._separate,
        )

    def _begin(self, arbiter, space, data) -> bool:
        for shape in arbiter.shapes:
            if shape.collision_type == CAT_ROBOT:
                self._contacting.add(shape)
        return True

    def _separate(self, arbiter, space, data) -> None:
        for shape in arbiter.shapes:
            if shape.collision_type == CAT_ROBOT:
                self._contacting.discard(shape)

    def clear(self) -> None:
        """Reset contact tracking (e.g. when the robot is teleported by reset)."""
        self._contacting.clear()

    def is_body_in_contact(self, body: pymunk.Body) -> bool:
        return any(shape in self._contacting for shape in body.shapes)


class BotAPI:
    """High-level API for controlling the biped.

    This is the RL-facing facade. It owns the physics world, the robot, the
    contact sensor, and (optionally) the renderer, and exposes a minimal set of
    operations: reset, step (apply action + advance physics), observe, and
    render.
    """

    def __init__(
        self,
        render_mode: str = 'headless',
        spawn_x: float = 0.0,
        spawn_y: Optional[float] = None,
        max_torque: float = 50.0,
        burst_torque: float = 200.0,
        substeps: int = 1,
    ) -> None:
        """
        Args:
            render_mode: 'headless' (no window, for training) or
                'human' (opens a Pygame window).
            spawn_x: X spawn position of the torso center (m).
            spawn_y: Y spawn position of the torso center (m); defaults to the
                configured spawn height.
            max_torque: Torque clamp per joint (N·m) for normal actions. Actions
                are clipped to [-max_torque, max_torque].
            burst_torque: Torque clamp (N·m) for ``apply_torque_burst`` — a
                diagnostic that slams ALL joints with a large torque to verify
                the action channel works.
            substeps: Number of physics substeps per control step (each substep
                advances the world by the fixed config timestep).
        """
        if render_mode not in ('headless', 'human'):
            raise ValueError(
                f"render_mode must be 'headless' or 'human', got {render_mode!r}"
            )

        self.render_mode: str = render_mode
        self.max_torque: float = float(max_torque)
        self.burst_torque: float = float(burst_torque)
        self.substeps: int = max(1, int(substeps))

        # Owned resources
        self.world: PhysicsWorld = PhysicsWorld()
        self.biped: Biped = Biped(self.world.space, spawn_x=spawn_x, spawn_y=spawn_y)
        self._contact = _ContactSensor(self.world.space)
        self._renderer: Optional[Renderer] = None
        if render_mode == 'human':
            self._renderer = Renderer(Camera())

        # Internal state bookkeeping
        self._spawn_x: float = self.biped.spawn_x
        self._spawn_y: float = self.biped.spawn_y
        self._applied_torques: Dict[str, float] = {name: 0.0 for name in JOINT_NAMES}
        self._joint_alpha: Dict[str, float] = {name: 0.0 for name in JOINT_NAMES}

        # Interactive-scene state (used by handle_event / handle_events and the
        # scene loop; ignored by pure RL usage).
        self.paused: bool = False
        self.speed: float = 1.0
        self.running: bool = True
        self.demo_mode: bool = False
        self.burst_mode: bool = False

    # ------------------------------------------------------------------
    # Scene lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        spawn_x: Optional[float] = None,
        spawn_y: Optional[float] = None,
    ) -> Dict[str, float]:
        """Reset the robot to its spawn pose and zero all velocities.

        Args:
            spawn_x, spawn_y: Optional new spawn position (m). If omitted, the
                previous spawn position is reused.

        Returns:
            The initial observation dict (see ``observe``).
        """
        if spawn_x is not None:
            self._spawn_x = float(spawn_x)
        if spawn_y is not None:
            self._spawn_y = float(spawn_y)

        self.biped.reset(spawn_x=self._spawn_x, spawn_y=self._spawn_y)
        self.world.time = 0.0
        self._contact.clear()
        self._applied_torques = {name: 0.0 for name in JOINT_NAMES}
        self._joint_alpha = {name: 0.0 for name in JOINT_NAMES}
        return self.observe()

    def reset_standing(self) -> Dict[str, float]:
        """Reset the robot to a standing pose with shins on the ground.

        Unlike ``reset`` (which spawns the robot in the air for the drop test),
        this places the straightened legs on the floor so torque/control demos
        produce visible limb motion. The robot may still topple since it stands
        on narrow shins — that is expected physics.

        Returns:
            The initial observation dict.
        """
        dims = config.dimensions
        torso_y = dims.torso[1] / 2 + dims.thigh[1] + dims.shin[1]
        return self.reset(spawn_y=torso_y)

    def close(self) -> None:
        """Release the renderer (if any). Call when done with the API."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "BotAPI":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Action interface
    # ------------------------------------------------------------------

    def apply_torques(self, torques: Dict[str, float]) -> None:
        """Apply torque to each named joint, clamped to ``max_torque``.

        Args:
            torques: Mapping of joint name -> torque (N·m). Unknown joint names
                raise a KeyError. Joints not listed get zero torque this step.

        Example:
            api.apply_torques({'left_hip': 20.0, 'right_knee': -10.0})
        """
        validated: Dict[str, float] = {}
        for name, value in torques.items():
            if name not in JOINT_NAMES:
                raise KeyError(
                    f"Unknown joint {name!r}. Valid joints: {JOINT_NAMES}"
                )
            validated[name] = float(
                np.clip(value, -self.max_torque, self.max_torque)
            )

        self.biped.apply_joint_torques(validated)
        self._applied_torques = {
            name: validated.get(name, 0.0) for name in JOINT_NAMES
        }

    def apply_action_vector(self, action: np.ndarray) -> None:
        """Apply torques from a flat array in JOINT_NAMES order."""
        if len(action) != len(JOINT_NAMES):
            raise ValueError(
                f"Action vector length {len(action)} != {len(JOINT_NAMES)} "
                f"(expected one torque per joint)"
            )
        self.apply_torques(dict(zip(JOINT_NAMES, action)))

    def get_action_space(self) -> Dict[str, object]:
        """Describe the continuous action space (for RL policy setup).

        Returns a dict with the joint order, per-joint low/high torque bounds,
        shape, dtype, and a human description.
        """
        return {
            'names': list(JOINT_NAMES),
            'low': [-self.max_torque] * len(JOINT_NAMES),
            'high': [self.max_torque] * len(JOINT_NAMES),
            'shape': (len(JOINT_NAMES),),
            'dtype': 'float32',
            'description': 'Torque (N·m) applied to each joint every control step.',
        }

    def apply_torque_burst(self, magnitude: Optional[float] = None) -> None:
        """Apply a large alternating torque to ALL joints (diagnostic).

        Slams every joint with ``magnitude`` (default ``burst_torque``, clamped
        to it) so you can visually confirm the torque channel works. Hip and
        knee signs alternate, making the limbs visibly flail.

        Args:
            magnitude: Torque magnitude in N·m. If None, uses ``burst_torque``.
        """
        m = self.burst_torque if magnitude is None else abs(float(magnitude))
        m = min(m, self.burst_torque)
        burst = {
            'left_hip': m,
            'left_knee': -m,
            'right_hip': -m,
            'right_knee': m,
        }
        self.biped.apply_joint_torques(burst)
        self._applied_torques = dict(burst)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(
        self,
        action: Optional[Dict[str, float]] = None,
        n_substeps: Optional[int] = None,
        use_scene_mode: bool = False,
    ) -> Dict[str, float]:
        """Apply an action (if given) and advance physics by one control step.

        Args:
            action: Optional dict of joint torques. If None, zero torque is
                applied (free-fall / passive dynamics).
            n_substeps: Optional override of the number of physics substeps.
                Defaults to the API's configured ``substeps``.
            use_scene_mode: If True, the interactive-scene mode decides the
                action instead of ``action``: torque burst if ``burst_mode``,
                sinusoidal demo if ``demo_mode``, otherwise passive (zero).

        Returns:
            The updated observation dict after the step.
        """
        if use_scene_mode:
            if self.burst_mode:
                self.apply_torque_burst()
            elif self.demo_mode:
                action = self._demo_action()
            else:
                action = None
        if action is not None:
            self.apply_torques(action)

        n = self.substeps if n_substeps is None else max(1, int(n_substeps))
        dt = n * self.world.dt

        omegas_before = self._compute_joint_omegas()
        for _ in range(n):
            self.world.step()
        omegas_after = self._compute_joint_omegas()

        # Angular acceleration via finite difference of joint angular velocity.
        self._joint_alpha = {
            name: (omegas_after[name] - omegas_before[name]) / dt
            for name in JOINT_NAMES
        }
        return self.observe()

    def _compute_joint_omegas(self) -> Dict[str, float]:
        """Joint angular velocities = child body omega - parent body omega."""
        bp = self.biped
        return {
            'left_hip': bp.left_thigh.angular_velocity - bp.torso.angular_velocity,
            'left_knee': bp.left_shin.angular_velocity - bp.left_thigh.angular_velocity,
            'right_hip': bp.right_thigh.angular_velocity - bp.torso.angular_velocity,
            'right_knee': bp.right_shin.angular_velocity - bp.right_thigh.angular_velocity,
        }

    # ------------------------------------------------------------------
    # Observation interface
    # ------------------------------------------------------------------

    def observe(self) -> Dict[str, float]:
        """Build the full observation state of the robot and scene.

        Includes torso pose (position, displacement from spawn, orientation),
        linear/angular velocities, the four joint angles, joint angular
        velocities and accelerations, the last applied torques, ground-contact
        flags, and simulation time. All values are Python floats.
        """
        bp = self.biped
        joint_angles = bp.get_joint_angles()
        omegas = self._compute_joint_omegas()
        torso = bp.torso

        return {
            'time': float(self.world.time),
            # Pose
            'torso_x': float(torso.position.x),
            'torso_y': float(torso.position.y),
            'torso_angle': float(torso.angle),
            'displacement_x': float(torso.position.x - self._spawn_x),
            'displacement_y': float(torso.position.y - self._spawn_y),
            # Velocity
            'torso_vx': float(torso.velocity.x),
            'torso_vy': float(torso.velocity.y),
            'torso_omega': float(torso.angular_velocity),
            # Joint angles (rad)
            'left_hip': joint_angles['left_hip'],
            'left_knee': joint_angles['left_knee'],
            'right_hip': joint_angles['right_hip'],
            'right_knee': joint_angles['right_knee'],
            # Joint angular velocities (rad/s)
            'left_hip_omega': omegas['left_hip'],
            'left_knee_omega': omegas['left_knee'],
            'right_hip_omega': omegas['right_hip'],
            'right_knee_omega': omegas['right_knee'],
            # Joint angular accelerations (rad/s^2)
            'left_hip_alpha': self._joint_alpha['left_hip'],
            'left_knee_alpha': self._joint_alpha['left_knee'],
            'right_hip_alpha': self._joint_alpha['right_hip'],
            'right_knee_alpha': self._joint_alpha['right_knee'],
            # Last applied torques (N·m)
            'left_hip_torque': self._applied_torques['left_hip'],
            'left_knee_torque': self._applied_torques['left_knee'],
            'right_hip_torque': self._applied_torques['right_hip'],
            'right_knee_torque': self._applied_torques['right_knee'],
            # Ground contact flags (1.0 = touching ground, 0.0 = airborne)
            'left_contact': self._contact_flag(bp.left_shin),
            'right_contact': self._contact_flag(bp.right_shin),
            'torso_contact': self._contact_flag(bp.torso),
        }

    def _contact_flag(self, body: pymunk.Body) -> float:
        return 1.0 if self._contact.is_body_in_contact(body) else 0.0

    def get_state(self) -> Dict[str, float]:
        """Alias for ``observe`` (readable dict form)."""
        return self.observe()

    def get_state_vector(self) -> np.ndarray:
        """Flat float32 observation vector, ordered by STATE_VECTOR_ORDER.

        Convenient for feeding directly to an RL policy network.
        """
        state = self.observe()
        return np.array([state[key] for key in STATE_VECTOR_ORDER],
                        dtype=np.float32)

    def get_state_vector_order(self) -> List[str]:
        """Return the names/order of entries in ``get_state_vector``."""
        return list(STATE_VECTOR_ORDER)

    def get_joint_angles(self) -> Dict[str, float]:
        """Current joint angles in radians."""
        return self.biped.get_joint_angles()

    def get_joint_angular_velocities(self) -> Dict[str, float]:
        """Current joint angular velocities in rad/s."""
        return self._compute_joint_omegas()

    def get_joint_angular_accelerations(self) -> Dict[str, float]:
        """Last computed joint angular accelerations in rad/s^2."""
        return dict(self._joint_alpha)

    def get_joint_torques(self) -> Dict[str, float]:
        """Last applied joint torques in N·m."""
        return dict(self._applied_torques)

    def get_torso_state(self) -> Dict[str, float]:
        """Torso pose and velocity (position, orientation, linear/angular vel)."""
        torso = self.biped.torso
        return {
            'x': float(torso.position.x),
            'y': float(torso.position.y),
            'angle': float(torso.angle),
            'vx': float(torso.velocity.x),
            'vy': float(torso.velocity.y),
            'omega': float(torso.angular_velocity),
        }

    def get_displacement(self) -> Dict[str, float]:
        """Displacement of the torso from its spawn position (m)."""
        torso = self.biped.torso
        return {
            'x': float(torso.position.x - self._spawn_x),
            'y': float(torso.position.y - self._spawn_y),
        }

    def get_contacts(self) -> Dict[str, bool]:
        """Which robot parts are currently touching the ground."""
        return {
            'left': self._contact.is_body_in_contact(self.biped.left_shin),
            'right': self._contact.is_body_in_contact(self.biped.right_shin),
            'torso': self._contact.is_body_in_contact(self.biped.torso),
        }

    def get_sim_time(self) -> float:
        """Current simulation time in seconds."""
        return float(self.world.time)

    # ------------------------------------------------------------------
    # Interactive scene / keyboard input
    # ------------------------------------------------------------------

    def _demo_action(self) -> Dict[str, float]:
        """Sinusoidal anti-phase leg drive used by the demo mode.

        Left/right legs are driven in anti-phase with moderate amplitudes
        (inside the ±max_torque clamp) so the kicking is a smooth sinusoid.
        """
        t = self.world.time
        return {
            'left_hip': 30.0 * math.sin(t * 3.0),
            'left_knee': 20.0 * math.sin(t * 5.0),
            'right_hip': 30.0 * math.sin(t * 3.0 + math.pi),
            'right_knee': 20.0 * math.sin(t * 5.0 + math.pi),
        }

    def handle_event(self, event) -> Optional[str]:
        """Map one pygame event to a scene command.

        Handles:
            QUIT / ESC  -> exit the scene
            R           -> reset the robot to its drop-test spawn
            SPACE       -> pause / unpause
            UP / DOWN   -> simulation speed
            T           -> toggle sinusoidal demo torques (standing reset)
            B           -> toggle torque burst (standing reset)

        Returns a short human-readable message describing what happened, or
        None if the event was not relevant. This is the ONLY place that needs
        to understand keyboard codes — scenes just call ``handle_events``.
        """
        if event.type == pygame.QUIT:
            self.running = False
            return None
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            self.running = False
            return None
        if event.key == pygame.K_r:
            self.reset()
            self.paused = False
            return "Reset"
        if event.key == pygame.K_SPACE:
            self.paused = not self.paused
            return "Paused" if self.paused else "Unpaused"
        if event.key == pygame.K_UP:
            self.speed = min(self.speed * 2.0, 8.0)
            return f"Speed: {self.speed:.1f}x"
        if event.key == pygame.K_DOWN:
            self.speed = max(self.speed / 2.0, 0.25)
            return f"Speed: {self.speed:.1f}x"
        if event.key == pygame.K_t:
            self.demo_mode = not self.demo_mode
            self.burst_mode = False  # modes are mutually exclusive
            if self.demo_mode:
                self.reset_standing()  # ground the robot so kicks are visible
                self.paused = False
                return "Demo torques ON (reset to standing)"
            return "Demo torques OFF"
        if event.key == pygame.K_b:
            self.burst_mode = not self.burst_mode
            self.demo_mode = False  # modes are mutually exclusive
            if self.burst_mode:
                self.reset_standing()
                self.paused = False
                return f"TORQUE BURST ON ({self.burst_torque:.0f} N·m on all joints)"
            return "Torque burst OFF"
        return None

    def handle_events(self) -> List[str]:
        """Drain the pygame event queue and apply all scene commands.

        Returns a list of messages describing what happened (print them to the
        console if you like). Safe in headless mode — returns an empty list.
        """
        if self._renderer is None:
            return []
        messages = []
        for event in pygame.event.get():
            msg = self.handle_event(event)
            if msg is not None:
                messages.append(msg)
        return messages

    def is_running(self) -> bool:
        """Whether the interactive scene should keep running."""
        return self.running

    def print_state_summary(self) -> None:
        """Print a compact one-line state summary (for scene feedback)."""
        st = self.observe()
        contacts = int(st['left_contact'] + st['right_contact'] + st['torso_contact'])
        print(
            f"t={st['time']:6.2f}s  torso_y={st['torso_y']:5.2f}  "
            f"left_hip={st['left_hip']:5.2f}  right_hip={st['right_hip']:5.2f} rad  "
            f"left_hip_omega={st['left_hip_omega']:6.2f} rad/s  contacts={contacts}"
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, paused: bool = False, speed: float = 1.0) -> None:
        """Draw one frame. No-op in headless mode.

        Args:
            paused: Show PAUSED in the HUD.
            speed: Show the current simulation speed in the HUD.
        """
        if self._renderer is not None:
            self._renderer.render(
                self.world, self.biped, self.world.time,
                paused=paused, speed=speed,
            )

    def tick(self, fps: int = 60) -> None:
        """Pace the display frame rate. No-op in headless mode."""
        if self._renderer is not None:
            self._renderer.tick(fps)

    def is_headless(self) -> bool:
        return self.render_mode == 'headless'


if __name__ == '__main__':
    # Quick self-test: drop the robot, apply some torques, print a state sample.
    api = BotAPI(render_mode='headless')
    api.reset()
    print('State vector order:', api.get_state_vector_order())
    print('Action space:', api.get_action_space())
    for i in range(120):
        obs = api.step({'left_hip': 10.0, 'left_knee': -5.0,
                        'right_hip': -10.0, 'right_knee': 5.0})
        if i % 30 == 0:
            vec = api.get_state_vector()
            print(f"t={obs['time']:.2f}s torso_y={obs['torso_y']:.2f} "
                  f"vec_len={vec.shape[0]} nan={bool(np.isnan(vec).any())}")
    print('OK')