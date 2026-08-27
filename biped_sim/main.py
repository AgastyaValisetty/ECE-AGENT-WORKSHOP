"""
Main entry point for the 2D biped simulation.
Runs the drop test: spawn robot above ground, let it fall, observe physics.
"""

import sys
import os
import pygame

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.robot_config import config
from physics.world import PhysicsWorld
from robot.biped import Biped
from rendering.renderer import Renderer
from rendering.camera import Camera


def main():
    """Run the biped drop test simulation."""
    # Create physics world
    world = PhysicsWorld()
    
    # Create the biped above ground
    biped = Biped(world.space)
    
    # Create renderer
    renderer = Renderer(Camera())
    
    # Simulation state
    paused = False
    running = True
    speed = 1.0
    
    print("=== 2D Biped Drop Test ===")
    print("Spawn height:", config.sim.spawn_height, "m")
    print("Gravity:", config.physics.gravity)
    print("Timestep:", config.physics.timestep)
    print("Controls: R=Reset, SPACE=Pause, ESC=Exit, UP/DOWN=Speed")
    print("-------------------------")
    
    while running:
        # Process input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    print("Resetting robot...")
                    biped.reset()
                    world.time = 0.0
                    paused = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Unpaused")
                elif event.key == pygame.K_UP:
                    speed = min(speed * 2.0, 8.0)
                    print(f"Speed: {speed:.1f}x")
                elif event.key == pygame.K_DOWN:
                    speed = max(speed / 2.0, 0.25)
                    print(f"Speed: {speed:.1f}x")
        
        # Physics step (fixed timestep, decoupled from render FPS)
        if not paused:
            for _ in range(int(round(speed))):
                world.step()
        
        # Render
        renderer.render(world, biped, world.time, paused, speed)
        renderer.tick(60)
    
    renderer.close()
    print("Simulation ended.")


if __name__ == "__main__":
    main()