"""
Biped robot construction.
Creates all bodies, shapes, and joints for the 4-joint biped.
"""

import pymunk
from config.robot_config import config
from physics.constants import (
    create_body_mass_properties, create_box_shape, create_revolute_joint,
    local_to_world
)


class Biped:
    """4-joint planar biped robot."""

    def __init__(self, space: pymunk.Space, spawn_x: float = 0.0, spawn_y: float = None):
        """
        Create the biped at the given spawn position.

        Args:
            space: Pymunk space to add bodies/joints to
            spawn_x: X position of torso center
            spawn_y: Y position of torso center (defaults to config sim spawn_height)
        """
        self.space = space
        self.spawn_x = spawn_x
        self.spawn_y = spawn_y if spawn_y is not None else config.sim.spawn_height

        # Body references
        self.torso = None
        self.left_thigh = None
        self.left_shin = None
        self.right_thigh = None
        self.right_shin = None

        # Joint references for later actuation
        self.left_hip = None
        self.left_knee = None
        self.right_hip = None
        self.right_knee = None
        self.left_hip_limit = None
        self.left_knee_limit = None
        self.right_hip_limit = None
        self.right_knee_limit = None

        # Build the robot
        self._build()

    def _build(self):
        """Construct all bodies and joints."""
        dims = config.dimensions
        masses = config.masses
        limits = config.joint_limits

        # ---- TORSO ----
        torso_mass, torso_moment = create_body_mass_properties(
            masses.torso, dims.torso[0], dims.torso[1]
        )
        self.torso = pymunk.Body(torso_mass, torso_moment)
        self.torso.position = (self.spawn_x, self.spawn_y)
        self.torso.angle = 0.0
        torso_shape = create_box_shape(self.torso, dims.torso[0], dims.torso[1])
        self.space.add(self.torso, torso_shape)

        # Leg chain vertical offsets (straight-leg configuration)
        thigh_y = self.spawn_y - dims.torso[1] / 2 - dims.thigh[1] / 2
        shin_y = thigh_y - dims.thigh[1] / 2 - dims.shin[1] / 2

        # ---- LEGS ----
        self._add_leg('left', dims, masses, thigh_y, shin_y)
        self._add_leg('right', dims, masses, thigh_y, shin_y)

        # ---- JOINTS ----
        self._add_joints(dims, limits)

        # Zero initial velocities
        for body in self.get_all_bodies():
            body.velocity = config.sim.initial_velocity
            body.angular_velocity = config.sim.initial_angular_velocity

    def _add_leg(self, side: str, dims, masses, thigh_y, shin_y):
        """Add thigh and shin bodies for one leg."""
        x = -dims.torso[0] / 4 if side == 'left' else dims.torso[0] / 4

        thigh_mass, thigh_moment = create_body_mass_properties(
            masses.thigh, dims.thigh[0], dims.thigh[1]
        )
        thigh = pymunk.Body(thigh_mass, thigh_moment)
        thigh.position = (x, thigh_y)
        thigh.angle = 0.0
        self.space.add(thigh, create_box_shape(thigh, dims.thigh[0], dims.thigh[1]))

        shin_mass, shin_moment = create_body_mass_properties(
            masses.shin, dims.shin[0], dims.shin[1]
        )
        shin = pymunk.Body(shin_mass, shin_moment)
        shin.position = (x, shin_y)
        shin.angle = 0.0
        self.space.add(shin, create_box_shape(shin, dims.shin[0], dims.shin[1]))

        if side == 'left':
            self.left_thigh = thigh
            self.left_shin = shin
        else:
            self.right_thigh = thigh
            self.right_shin = shin

    def _add_joints(self, dims, limits):
        """Create the four main joints with limits."""
        hip_offset_y = -dims.torso[1] / 2
        left_hip_anchor = (-dims.torso[0] / 4, hip_offset_y)
        right_hip_anchor = (dims.torso[0] / 4, hip_offset_y)

        # Left Hip: torso <-> left_thigh
        lhj = create_revolute_joint(
            self.torso, self.left_thigh,
            left_hip_anchor, (0, dims.thigh[1] / 2),
            min_angle=limits.hip_min, max_angle=limits.hip_max,
            collide_bodies=False
        )
        self.left_hip = lhj[0]
        if len(lhj) > 1:
            self.left_hip_limit = lhj[1]
        self.space.add(*lhj)

        # Right Hip: torso <-> right_thigh
        rhj = create_revolute_joint(
            self.torso, self.right_thigh,
            right_hip_anchor, (0, dims.thigh[1] / 2),
            min_angle=limits.hip_min, max_angle=limits.hip_max,
            collide_bodies=False
        )
        self.right_hip = rhj[0]
        if len(rhj) > 1:
            self.right_hip_limit = rhj[1]
        self.space.add(*rhj)

        # Left Knee: left_thigh <-> left_shin
        lkj = create_revolute_joint(
            self.left_thigh, self.left_shin,
            (0, -dims.thigh[1] / 2), (0, dims.shin[1] / 2),
            min_angle=limits.knee_min, max_angle=limits.knee_max,
            collide_bodies=False
        )
        self.left_knee = lkj[0]
        if len(lkj) > 1:
            self.left_knee_limit = lkj[1]
        self.space.add(*lkj)

        # Right Knee: right_thigh <-> right_shin
        rkj = create_revolute_joint(
            self.right_thigh, self.right_shin,
            (0, -dims.thigh[1] / 2), (0, dims.shin[1] / 2),
            min_angle=limits.knee_min, max_angle=limits.knee_max,
            collide_bodies=False
        )
        self.right_knee = rkj[0]
        if len(rkj) > 1:
            self.right_knee_limit = rkj[1]
        self.space.add(*rkj)

    # ================================================================
    # Query / utility methods
    # ================================================================

    def get_all_bodies(self):
        """Return list of all robot bodies."""
        return [
            self.torso,
            self.left_thigh, self.left_shin,
            self.right_thigh, self.right_shin
        ]

    def get_all_shapes(self):
        """Return list of all robot shapes."""
        shapes = []
        for body in self.get_all_bodies():
            shapes.extend(body.shapes)
        return shapes

    def get_all_joints(self):
        """Return list of all main joints (for actuation later)."""
        return [
            self.left_hip, self.left_knee,
            self.right_hip, self.right_knee
        ]

    def get_joint_angles(self):
        """Get current joint angles in radians (relative to parent body)."""
        return {
            'left_hip': self.left_thigh.angle - self.torso.angle,
            'left_knee': self.left_shin.angle - self.left_thigh.angle,
            'right_hip': self.right_thigh.angle - self.torso.angle,
            'right_knee': self.right_shin.angle - self.right_thigh.angle,
        }

    def get_body_positions(self):
        """Get world positions of all bodies."""
        return {
            'torso': self.torso.position,
            'left_thigh': self.left_thigh.position,
            'left_shin': self.left_shin.position,
            'right_thigh': self.right_thigh.position,
            'right_shin': self.right_shin.position,
        }

    def get_joint_world_positions(self):
        """Get world positions of joint anchors."""
        dims = config.dimensions
        return {
            'left_hip': local_to_world(self.torso,
                                       (-dims.torso[0] / 4, -dims.torso[1] / 2)),
            'right_hip': local_to_world(self.torso,
                                        (dims.torso[0] / 4, -dims.torso[1] / 2)),
            'left_knee': local_to_world(self.left_thigh,
                                        (0, -dims.thigh[1] / 2)),
            'right_knee': local_to_world(self.right_thigh,
                                         (0, -dims.thigh[1] / 2)),
        }

    def apply_joint_torques(self, torques: dict):
        """Apply torques to joints (for future actuation)."""
        if 'left_hip' in torques:
            t = torques['left_hip']
            self.torso.torque -= t
            self.left_thigh.torque += t
        if 'right_hip' in torques:
            t = torques['right_hip']
            self.torso.torque -= t
            self.right_thigh.torque += t
        if 'left_knee' in torques:
            t = torques['left_knee']
            self.left_thigh.torque -= t
            self.left_shin.torque += t
        if 'right_knee' in torques:
            t = torques['right_knee']
            self.right_thigh.torque -= t
            self.right_shin.torque += t

    def reset(self, spawn_x: float = None, spawn_y: float = None):
        """Reset robot to initial position and zero velocities."""
        if spawn_x is not None:
            self.spawn_x = spawn_x
        if spawn_y is not None:
            self.spawn_y = spawn_y

        dims = config.dimensions
        thigh_y = self.spawn_y - dims.torso[1] / 2 - dims.thigh[1] / 2
        shin_y = thigh_y - dims.thigh[1] / 2 - dims.shin[1] / 2

        # Reset all bodies to spawn configuration
        reset_data = [
            (self.torso, (self.spawn_x, self.spawn_y)),
            (self.left_thigh, (-dims.torso[0] / 4, thigh_y)),
            (self.left_shin, (-dims.torso[0] / 4, shin_y)),
            (self.right_thigh, (dims.torso[0] / 4, thigh_y)),
            (self.right_shin, (dims.torso[0] / 4, shin_y)),
        ]
        for body, pos in reset_data:
            body.position = pos
            body.angle = 0.0
            body.velocity = (0, 0)
            body.angular_velocity = 0.0
            body.force = (0, 0)
            body.torque = 0.0