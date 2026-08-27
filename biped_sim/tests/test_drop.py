"""
Drop test: verify the biped falls under gravity, collides with ground,
remains connected, and stays numerically stable.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import math
import pymunk

from physics.world import PhysicsWorld
from robot.biped import Biped
from config.robot_config import config


class TestDrop(unittest.TestCase):
    
    def setUp(self):
        self.world = PhysicsWorld()
        self.biped = Biped(self.world.space)
    
    def test_falls_with_gravity_acceleration(self):
        """Robot should accelerate under gravity, not fall at constant velocity."""
        # After ~0.5s of free fall, speed should be ~4.9 m/s (g*t)
        for _ in range(30):
            self.world.step()
        torso_speed = math.hypot(
            self.biped.torso.velocity.x, self.biped.torso.velocity.y
        )
        # Allow some deviation as the robot may still be intact falling
        self.assertGreater(torso_speed, 3.0,
                           f"Torso only falling at {torso_speed:.2f} m/s - damping may be wrong")
    
    def test_falls_under_gravity(self):
        """Robot torso should move downward after spawning."""
        initial_y = self.biped.torso.position.y
        for _ in range(30):  # 0.5 seconds
            self.world.step()
        current_y = self.biped.torso.position.y
        self.assertLess(current_y, initial_y)
    
    def test_ground_collision(self):
        """Robot should not pass through the ground."""
        for _ in range(300):  # 5 seconds
            self.world.step()
        
        for body in self.biped.get_all_bodies():
            self.assertGreater(body.position.y, -0.5, 
                               f"Body {body} passed through ground")
    
    def test_joint_integrity(self):
        """Robot bodies should remain connected (no flying apart)."""
        for _ in range(300):  # 5 seconds
            self.world.step()
        
        # All bodies should be within a reasonable distance of each other
        positions = [b.position for b in self.biped.get_all_bodies()]
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i < j:
                    dist = math.hypot(pos1.x - pos2.x, pos1.y - pos2.y)
                    # Max separation should be less than total body length
                    self.assertLess(dist, 3.0, 
                                    f"Bodies {i} and {j} separated by {dist:.2f}m")
    
    def test_numerical_stability(self):
        """No NaN, inf, or exploding velocities after 5 seconds."""
        for _ in range(300):  # 5 seconds
            self.world.step()
        
        for body in self.biped.get_all_bodies():
            pos = body.position
            vel = body.velocity
            # No NaN
            self.assertFalse(math.isnan(pos.x))
            self.assertFalse(math.isnan(pos.y))
            self.assertFalse(math.isnan(vel.x))
            self.assertFalse(math.isnan(vel.y))
            self.assertFalse(math.isnan(body.angle))
            self.assertFalse(math.isnan(body.angular_velocity))
            # No infinity
            self.assertTrue(math.isfinite(pos.x))
            self.assertTrue(math.isfinite(pos.y))
            self.assertTrue(math.isfinite(vel.x))
            self.assertTrue(math.isfinite(vel.y))
            # No exploding velocities
            self.assertLess(abs(vel.x), 100.0, "Velocity X exploded")
            self.assertLess(abs(vel.y), 100.0, "Velocity Y exploded")
            self.assertLess(abs(body.angular_velocity), 50.0, "Angular velocity exploded")
    
    def test_robot_eventually_settles(self):
        """Robot should eventually come to rest (low velocities)."""
        for _ in range(1200):  # 20 seconds
            self.world.step()
        
        max_speed = 0.0
        for body in self.biped.get_all_bodies():
            speed = math.hypot(body.velocity.x, body.velocity.y)
            max_speed = max(max_speed, speed)
        
        # Robot should be mostly settled (small residual jitter allowed)
        self.assertLess(max_speed, 1.0, f"Robot still moving fast: {max_speed:.3f} m/s")


if __name__ == '__main__':
    unittest.main()