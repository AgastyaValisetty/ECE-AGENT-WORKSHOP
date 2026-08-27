"""
Simple Pygame renderer for the biped simulation.
Draws ground, robot bodies (rectangles), and joint locations.
"""

import math
import pygame
import pymunk
from config.robot_config import config
from rendering.camera import Camera


class Renderer:
    """Renders the physics world using Pygame."""
    
    def __init__(self, camera: Camera = None):
        pygame.init()
        self.camera = camera if camera is not None else Camera()
        self.screen = pygame.display.set_mode((self.camera.width, self.camera.height))
        pygame.display.set_caption("2D Biped Simulation - Drop Test")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
    
    def clear(self):
        """Clear the screen."""
        self.screen.fill(config.render.bg_color)
    
    def draw_ground(self, ground_body: pymunk.Body):
        """Draw the ground as a filled rectangle at the world origin."""
        # Draw ground as a wide rectangle from y=0 down
        top_world_y = 0.0
        sx, sy = self.camera.world_to_screen(-20, top_world_y)
        ex, _ = self.camera.world_to_screen(20, top_world_y)
        
        ground_rect = pygame.Rect(sx, sy, ex - sx, 200)
        pygame.draw.rect(self.screen, config.render.ground_color, ground_rect)
        
        # Draw a line at the ground surface
        pygame.draw.line(self.screen, (255, 255, 255), (sx, sy), (ex, sy), 2)
    
    def draw_poly(self, body: pymunk.Body, shape: pymunk.Poly, color, outline=True):
        """Draw a pymunk polygon body."""
        verts_world = [body.local_to_world(v) for v in shape.get_vertices()]
        verts_screen = [self.camera.world_to_screen(v[0], v[1]) for v in verts_world]
        pygame.draw.polygon(self.screen, color, verts_screen)
        if outline:
            pygame.draw.polygon(self.screen, (20, 20, 20), verts_screen, 1)
    
    def draw_bodies(self, biped):
        """Draw all robot bodies."""
        colors = config.render.body_colors
        
        # Draw each body as its collision shape
        body_color_map = [
            (biped.torso, colors['torso']),
            (biped.left_thigh, colors['thigh']),
            (biped.left_shin, colors['shin']),
            (biped.right_thigh, colors['thigh']),
            (biped.right_shin, colors['shin']),
        ]
        
        for body, color in body_color_map:
            for shape in body.shapes:
                if isinstance(shape, pymunk.Poly):
                    self.draw_poly(body, shape, color)
    
    def draw_joints(self, biped):
        """Draw joint locations as circles."""
        for name, pos in biped.get_joint_world_positions().items():
            sx, sy = self.camera.world_to_screen(pos[0], pos[1])
            pygame.draw.circle(self.screen, config.render.joint_color,
                               (sx, sy), config.render.joint_radius)
            # Label the joint
            text = self.font.render(name.split('_')[-1].upper()[:1], True, (0, 0, 0))
            self.screen.blit(text, (sx - 3, sy - 6))
    
    def draw_info(self, time: float, paused: bool, speed: float):
        """Draw simulation info text."""
        info_lines = [
            f"Time: {time:.2f}s",
            f"Speed: {speed:.2f}x",
            "PAUSED" if paused else "Running",
        ]
        for i, line in enumerate(info_lines):
            color = (255, 200, 100) if line == "PAUSED" else (200, 200, 200)
            text = self.font.render(line, True, color)
            self.screen.blit(text, (10, 10 + i * 22))
        
        # Controls help
        help_lines = [
            "R: Reset    SPACE: Pause    ESC: Exit",
            "UP/DOWN: Speed"
        ]
        for i, line in enumerate(help_lines):
            text = self.font.render(line, True, (150, 150, 150))
            self.screen.blit(text, (10, self.camera.height - 50 + i * 20))
    
    def render(self, world, biped, sim_time: float, paused: bool, speed: float):
        """Render the complete frame."""
        self.clear()
        self.draw_ground(world.ground_body)
        self.draw_bodies(biped)
        self.draw_joints(biped)
        self.draw_info(sim_time, paused, speed)
        pygame.display.flip()
    
    def tick(self, fps: int = 60):
        """Keep frame rate steady."""
        self.clock.tick(fps)
    
    def close(self):
        """Close the pygame window."""
        pygame.quit()