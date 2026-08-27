"""
Tests for the high-level BotAPI (botapi.py).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np

from botapi import BotAPI, JOINT_NAMES, STATE_VECTOR_ORDER
from config.robot_config import config


class TestBotAPI(unittest.TestCase):

    def setUp(self):
        self.api = BotAPI(render_mode='headless')
        self.api.reset()

    def tearDown(self):
        self.api.close()

    # ------------------------------------------------------------------
    # Scene lifecycle
    # ------------------------------------------------------------------

    def test_reset_returns_observation(self):
        obs = self.api.reset()
        self.assertIsInstance(obs, dict)
        self.assertAlmostEqual(obs['time'], 0.0)
        self.assertAlmostEqual(obs['torso_y'], config.sim.spawn_height, places=2)

    def test_reset_zeroes_velocity(self):
        self.api.reset()
        for _ in range(10):
            self.api.step()
        self.api.reset()
        st = self.api.get_torso_state()
        self.assertEqual(st['vx'], 0.0)
        self.assertEqual(st['vy'], 0.0)
        self.assertEqual(st['omega'], 0.0)

    def test_reset_standing_places_robot_on_ground(self):
        """reset_standing puts the shin bottoms at the ground surface (y=0)."""
        self.api.reset_standing()
        # Direct check: the lowest point of any robot shape should be ~0
        lows = []
        for body in self.api.biped.get_all_bodies():
            for shape in body.shapes:
                for v in shape.get_vertices():
                    lows.append(body.local_to_world(v).y)
        self.assertAlmostEqual(min(lows), 0.0, delta=0.05)

    def test_standing_torques_produce_joint_oscillation(self):
        """Sinusoidal hip torques from a standing pose swing the joints."""
        self.api.reset_standing()
        left_hip_samples = []
        for _ in range(120):  # 2 seconds
            action = self.api._demo_action()
            self.api.step(action)
            left_hip_samples.append(self.api.get_joint_angles()['left_hip'])
        swing = max(left_hip_samples) - min(left_hip_samples)
        self.assertGreater(swing, 1.0,
                           f"Expected visible hip oscillation, swing was {swing:.2f} rad")

    # ------------------------------------------------------------------
    # State / observation
    # ------------------------------------------------------------------

    def test_state_dict_has_all_keys(self):
        obs = self.api.observe()
        expected = set(STATE_VECTOR_ORDER)
        self.assertEqual(set(obs.keys()), expected)

    def test_state_vector_length_and_order(self):
        vec = self.api.get_state_vector()
        self.assertEqual(len(vec), len(STATE_VECTOR_ORDER))
        # Values must match the dict view
        obs = self.api.observe()
        for i, key in enumerate(STATE_VECTOR_ORDER):
            self.assertAlmostEqual(float(vec[i]), obs[key], places=4)

    def test_state_vector_no_nan(self):
        for _ in range(300):
            self.api.step({'left_hip': 10.0, 'right_hip': -10.0})
        vec = self.api.get_state_vector()
        self.assertFalse(np.isnan(vec).any())
        self.assertTrue(np.isfinite(vec).all())

    def test_joint_angles_available(self):
        angles = self.api.get_joint_angles()
        self.assertEqual(set(angles.keys()), set(JOINT_NAMES))

    def test_displacement_from_spawn(self):
        d = self.api.get_displacement()
        self.assertAlmostEqual(d['x'], 0.0, places=2)
        self.assertAlmostEqual(d['y'], 0.0, places=2)

    # ------------------------------------------------------------------
    # Action interface
    # ------------------------------------------------------------------

    def test_apply_torques_clamps(self):
        self.api.apply_torques({'left_hip': 1000.0})
        t = self.api.get_joint_torques()
        self.assertLessEqual(t['left_hip'], self.api.max_torque)

    def test_apply_torques_unknown_joint_raises(self):
        with self.assertRaises(KeyError):
            self.api.apply_torques({'fake_joint': 1.0})

    def test_torque_burst_hits_all_joints(self):
        """apply_torque_burst applies the burst magnitude to all 4 joints."""
        self.api.reset_standing()
        self.api.apply_torque_burst()
        t = self.api.get_joint_torques()
        self.assertEqual(set(t.keys()), set(JOINT_NAMES))
        for name in JOINT_NAMES:
            self.assertEqual(abs(t[name]), self.api.burst_torque)

    def test_torque_burst_moves_limbs(self):
        """A torque burst from a standing pose visibly swings the joints."""
        self.api.reset_standing()
        max_swing = 0.0
        for _ in range(120):  # 2 seconds
            self.api.apply_torque_burst()
            self.api.step()
            for v in self.api.get_joint_angles().values():
                max_swing = max(max_swing, abs(v))
        self.assertGreater(max_swing, 1.0,
                           f"Expected limbs to move, max swing was {max_swing:.2f} rad")

    def test_action_vector(self):
        vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        self.api.apply_action_vector(vec)
        t = self.api.get_joint_torques()
        self.assertEqual(t['left_hip'], 1.0)
        self.assertEqual(t['right_knee'], 4.0)

    def test_action_space_shape(self):
        space = self.api.get_action_space()
        self.assertEqual(space['shape'], (4,))
        self.assertEqual(space['names'], JOINT_NAMES)
        self.assertEqual(len(space['low']), 4)
        self.assertEqual(len(space['high']), 4)

    # ------------------------------------------------------------------
    # Physics behavior through the API
    # ------------------------------------------------------------------

    def test_robot_falls_and_settles(self):
        for _ in range(1200):  # 20 seconds
            self.api.step()
        speeds = [
            abs(v) for v in [
                self.api.get_torso_state()['vx'],
                self.api.get_torso_state()['vy'],
                self.api.get_torso_state()['omega'],
            ]
        ]
        self.assertLess(max(speeds), 0.5, f"Robot did not settle: {speeds}")

    def test_contact_detected_when_landed(self):
        self.api.reset()
        # Initially airborne
        self.assertEqual(self.api.get_contacts()['left'], False)
        for _ in range(120):  # let it land
            self.api.step()
        contacts = self.api.get_contacts()
        self.assertTrue(
            contacts['left'] or contacts['right'],
            f"Expected leg contact after landing, got {contacts}"
        )

    def test_angular_acceleration_computed(self):
        self.api.reset()
        self.api.step({'left_hip': 30.0})
        alphas = self.api.get_joint_angular_accelerations()
        for name in JOINT_NAMES:
            self.assertIsInstance(alphas[name], float)
        self.assertFalse(np.isnan(alphas['left_hip']))

    def test_apply_torques_changes_motion(self):
        self.api.reset()
        # Apply strong hip torques and confirm the robot starts rotating
        self.api.step({'left_hip': 40.0, 'right_hip': -40.0})
        self.api.step({'left_hip': 40.0, 'right_hip': -40.0})
        self.api.step({'left_hip': 40.0, 'right_hip': -40.0})
        omega = self.api.get_torso_state()['omega']
        # Torso should have gained some angular velocity from hip reactions
        self.assertNotEqual(omega, 0.0)

    # ------------------------------------------------------------------
    # Time / stepping
    # ------------------------------------------------------------------

    def test_time_advances(self):
        self.api.reset()
        self.api.step()
        self.api.step()
        self.assertAlmostEqual(self.api.get_sim_time(), 2 * config.physics.timestep)


if __name__ == '__main__':
    unittest.main()