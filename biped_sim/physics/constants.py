"""
Physics constants and helper functions.
"""

import pymunk
import math
from config.robot_config import config


# Collision categories
CAT_GROUND = config.collision.CAT_GROUND
CAT_ROBOT = config.collision.CAT_ROBOT


def create_body_mass_properties(mass: float, width: float, height: float) -> tuple:
    """
    Calculate moment of inertia for a rectangular box.
    Returns (mass, moment).
    """
    # Moment of inertia for rectangle about center: (1/12) * m * (w^2 + h^2)
    moment = (1.0 / 12.0) * mass * (width * width + height * height)
    return mass, moment


def create_box_shape(body: pymunk.Body, width: float, height: float, 
                     friction: float = None, restitution: float = None,
                     collision_type: int = CAT_ROBOT,
                     collide_with_self: bool = False) -> pymunk.Poly:
    """Create a box-shaped collision polygon centered on the body.

    By default, robot parts do NOT collide with each other (collide_with_self=False);
    they only collide with the ground. This avoids internal limb collisions that
    destabilize the articulated structure.
    """
    if friction is None:
        friction = config.collision.body_friction
    if restitution is None:
        restitution = config.collision.body_restitution
    
    # Box vertices centered at origin
    hw, hh = width / 2.0, height / 2.0
    verts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    
    shape = pymunk.Poly(body, verts)
    shape.friction = friction
    shape.elasticity = restitution
    shape.collision_type = collision_type
    
    mask = CAT_GROUND
    if collide_with_self:
        mask |= CAT_ROBOT
    
    shape.filter = pymunk.ShapeFilter(
        categories=CAT_ROBOT,
        mask=mask  # Only collide with ground (and optionally other robot parts)
    )
    return shape


def create_revolute_joint(body_a: pymunk.Body, body_b: pymunk.Body,
                          anchor_a: tuple, anchor_b: tuple,
                          min_angle: float = None, max_angle: float = None,
                          collide_bodies: bool = False) -> pymunk.RotaryLimitJoint:
    """
    Create a revolute joint with optional angle limits.
    Returns the joint (and limit joint if limits specified).
    """
    # Main pivot joint
    pivot = pymunk.PivotJoint(body_a, body_b, anchor_a, anchor_b)
    pivot.collide_bodies = collide_bodies
    
    joints = [pivot]
    
    # Add rotary limit joint if limits specified
    if min_angle is not None or max_angle is not None:
        if min_angle is None:
            min_angle = -math.inf
        if max_angle is None:
            max_angle = math.inf
        limit = pymunk.RotaryLimitJoint(body_a, body_b, min_angle, max_angle)
        limit.collide_bodies = collide_bodies
        joints.append(limit)
    
    return joints


def world_to_local(body: pymunk.Body, world_pos: tuple) -> tuple:
    """Convert world coordinates to body-local coordinates."""
    return body.world_to_local(world_pos)


def local_to_world(body: pymunk.Body, local_pos: tuple) -> tuple:
    """Convert body-local coordinates to world coordinates."""
    return body.local_to_world(local_pos)