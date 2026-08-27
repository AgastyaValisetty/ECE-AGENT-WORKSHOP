"""
Physics world wrapper for Pymunk.
Handles world creation, stepping, and ground body.
"""

import pymunk
from config.robot_config import config


class PhysicsWorld:
    """Wrapper around pymunk.Space for the simulation."""
    
    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = config.physics.gravity
        # NOTE: space.damping is the fraction of velocity RETAINED each step.
        # 1.0 = no damping (default). 0.0 = total damping.
        self.space.damping = 1.0
        
        # Create static ground
        self.ground_body = self._create_ground()
        
        # Track simulation time
        self.time = 0.0
        self.dt = config.physics.timestep
        
    def _create_ground(self) -> pymunk.Body:
        """Create a static ground plane with collision geometry.

        The ground is a segment with radius 0.5. The body is placed at y=-0.5
        so the segment's TOP collision surface sits exactly at y=0, matching the
        rendered floor line. (A segment at y=0 with radius 0.5 would put the
        collision surface at y=0.5, leaving a visible gap.)
        """
        ground = pymunk.Body(body_type=pymunk.Body.STATIC)
        ground.position = (0, -0.5)

        # Ground segment - wide enough for the robot to fall and move.
        # Top surface lands at y = -0.5 + 0.5 = 0 (the drawn floor line).
        ground_shape = pymunk.Segment(ground, (-50, 0), (50, 0), 0.5)
        ground_shape.friction = config.collision.ground_friction
        ground_shape.elasticity = config.collision.ground_restitution
        ground_shape.collision_type = config.collision.CAT_GROUND
        ground_shape.filter = pymunk.ShapeFilter(
            categories=config.collision.CAT_GROUND,
            mask=pymunk.ShapeFilter.ALL_MASKS()
        )
        
        self.space.add(ground, ground_shape)
        return ground
    
    def step(self):
        """Advance physics by one timestep."""
        self.space.step(self.dt)
        self.time += self.dt
    
    def reset(self):
        """Reset the physics world - recreate ground, clear all dynamic bodies."""
        self.space = pymunk.Space()
        self.space.gravity = config.physics.gravity
        self.space.damping = 1.0  # 1.0 = no damping
        self.ground_body = self._create_ground()
        self.time = 0.0
    
    def add_body(self, body: pymunk.Body, *shapes: pymunk.Shape):
        """Add a body and its shapes to the space."""
        self.space.add(body, *shapes)
    
    def add_joint(self, joint: pymunk.Constraint):
        """Add a joint/constraint to the space."""
        self.space.add(joint)
    
    def remove_body(self, body: pymunk.Body, *shapes: pymunk.Shape):
        """Remove a body and its shapes from the space."""
        self.space.remove(body, *shapes)
    
    def remove_joint(self, joint: pymunk.Constraint):
        """Remove a joint from the space."""
        self.space.remove(joint)