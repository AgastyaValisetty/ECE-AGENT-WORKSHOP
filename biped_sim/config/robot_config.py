"""
Centralized configuration for the 2D biped simulation.
All physics parameters, dimensions, and simulation settings in one place.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PhysicsConfig:
    """Physics engine configuration."""
    gravity: Tuple[float, float] = (0.0, -9.81)  # (x, y) - negative y is down
    timestep: float = 1.0 / 60.0
    velocity_iterations: int = 8
    position_iterations: int = 3


@dataclass
class RobotDimensions:
    """Robot body dimensions (width, height) in meters."""
    torso: Tuple[float, float] = (0.4, 0.6)
    thigh: Tuple[float, float] = (0.15, 0.45)
    shin: Tuple[float, float] = (0.12, 0.45)


@dataclass
class RobotMasses:
    """Robot body masses in kg."""
    torso: float = 8.0
    thigh: float = 2.5
    shin: float = 1.5


@dataclass
class JointLimits:
    """Joint angle limits in radians."""
    # Hip limits: (min, max) relative to torso upright
    hip_min: float = -1.2   # ~ -69 degrees (forward)
    hip_max: float = 1.2    # ~  69 degrees (backward)
    
    # Knee limits: (min, max) - knee bends backward (positive = extended)
    knee_min: float = -0.1   # slight hyperextension
    knee_max: float = 2.5    # ~ 143 degrees (fully bent)


@dataclass
class CollisionConfig:
    """Collision and friction settings."""
    ground_friction: float = 0.8
    ground_restitution: float = 0.1
    body_friction: float = 0.6
    body_restitution: float = 0.0
    
    # Collision categories for filtering
    CAT_GROUND: int = 0x1
    CAT_ROBOT: int = 0x2


@dataclass
class SimConfig:
    """Simulation control settings."""
    spawn_height: float = 5.0  # meters above ground
    initial_velocity: Tuple[float, float] = (0.0, 0.0)
    initial_angular_velocity: float = 0.0


@dataclass
class RenderConfig:
    """Rendering configuration."""
    width: int = 1024
    height: int = 768
    ppm: float = 80.0  # pixels per meter
    camera_offset_x: float = 0.0
    camera_offset_y: float = 2.0
    bg_color: Tuple[int, int, int] = (30, 30, 40)
    ground_color: Tuple[int, int, int] = (60, 60, 70)
    body_colors: dict = None
    joint_color: Tuple[int, int, int] = (255, 255, 100)
    joint_radius: int = 6
    
    def __post_init__(self):
        if self.body_colors is None:
            self.body_colors = {
                'torso': (100, 180, 255),
                'thigh': (100, 220, 180),
                'shin': (180, 220, 100),
            }


# Combined config object for easy access
class Config:
    physics = PhysicsConfig()
    dimensions = RobotDimensions()
    masses = RobotMasses()
    joint_limits = JointLimits()
    collision = CollisionConfig()
    sim = SimConfig()
    render = RenderConfig()


# Global config instance
config = Config()