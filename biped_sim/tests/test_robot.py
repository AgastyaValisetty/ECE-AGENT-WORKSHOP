"""
Tests for the biped robot construction.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pymunk

from physics.world import PhysicsWorld
from robot.biped import Biped
from config.robot_config import config


class TestRobot(unittest.TestCase):
    
    def setUp(self):
        self.world = PhysicsWorld()
        self.biped = Biped(self.world.space)
    
    def test_robot_creation(self):
        """Biped has all expected body parts."""
        self.assertIsNotNone(self.biped.torso)
        self.assertIsNotNone(self.biped.left_thigh)
        self.assertIsNotNone(self.biped.left_shin)
        self.assertIsNotNone(self.biped.right_thigh)
        self.assertIsNotNone(self.biped.right_shin)
    
    def test_expected_body_count(self):
        """Robot has exactly 5 bodies (torso + 2 thigh + 2 shin)."""
        self.assertEqual(len(self.biped.get_all_bodies()), 5)
    
    def test_joint_count(self):
        """Robot has exactly 4 main joints (2 hips + 2 knees)."""
        joints = self.biped.get_all_joints()
        self.assertEqual(len(joints), 4)
    
    def test_joint_types(self):
        """All joints should be PivotJoint (revolute)."""
        for joint in self.biped.get_all_joints():
            self.assertIsInstance(joint, pymunk.PivotJoint)
    
    def test_joint_limits_present(self):
        """Each joint should have a corresponding limit constraint."""
        self.assertIsNotNone(self.biped.left_hip_limit)
        self.assertIsNotNone(self.biped.right_hip_limit)
        self.assertIsNotNone(self.biped.left_knee_limit)
        self.assertIsNotNone(self.biped.right_knee_limit)
    
    def test_initial_positions(self):
        """Robot should be at spawn height."""
        spawn_y = config.sim.spawn_height
        self.assertAlmostEqual(self.biped.torso.position.y, spawn_y, places=2)
    
    def test_initial_velocity_zero(self):
        """All bodies should have zero initial velocity."""
        for body in self.biped.get_all_bodies():
            self.assertEqual(body.velocity, (0, 0))
            self.assertEqual(body.angular_velocity, 0.0)
    
    def test_reset_restores_position(self):
        """Reset restores robot to spawn position."""
        # Simulate some movement
        for _ in range(10):
            self.world.step()
        
        self.biped.reset()
        
        spawn_y = config.sim.spawn_height
        self.assertAlmostEqual(self.biped.torso.position.y, spawn_y, places=2)
        self.assertEqual(self.biped.torso.angle, 0.0)
        self.assertEqual(self.biped.torso.velocity, (0, 0))
        self.assertEqual(self.biped.torso.angular_velocity, 0.0)


if __name__ == '__main__':
    unittest.main()