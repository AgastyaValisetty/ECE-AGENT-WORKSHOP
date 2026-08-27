"""
Tests for the physics world.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pymunk

from physics.world import PhysicsWorld
from config.robot_config import config


class TestWorld(unittest.TestCase):
    
    def test_world_creation(self):
        """Physics world initializes correctly."""
        world = PhysicsWorld()
        self.assertIsInstance(world.space, pymunk.Space)
        self.assertEqual(world.space.gravity, config.physics.gravity)
        self.assertEqual(world.time, 0.0)
    
    def test_ground_creation(self):
        """Ground body is created with collision geometry."""
        world = PhysicsWorld()
        ground_shapes = [s for s in world.ground_body.shapes]
        self.assertEqual(len(ground_shapes), 1)
        self.assertIsInstance(ground_shapes[0], pymunk.Segment)
    
    def test_step(self):
        """Stepping advances time."""
        world = PhysicsWorld()
        world.step()
        self.assertEqual(world.time, config.physics.timestep)
    
    def test_reset(self):
        """Reset clears bodies and recreates ground."""
        world = PhysicsWorld()
        world.step()
        world.step()
        
        # Add a dynamic body
        body = pymunk.Body(1, 1)
        body.position = (0, 5)
        shape = pymunk.Circle(body, 0.5)
        world.add_body(body, shape)
        
        world.reset()
        self.assertEqual(world.time, 0.0)
        # Ground should be recreated
        self.assertEqual(len(world.ground_body.shapes), 1)


if __name__ == '__main__':
    unittest.main()