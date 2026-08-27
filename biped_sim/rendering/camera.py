"""
Camera for world-to-screen coordinate conversion.
Simulation coordinates (meters) are converted to screen coordinates (pixels).
"""

from config.robot_config import config


class Camera:
    """Simple fixed camera for world-to-screen transform."""
    
    def __init__(self, width: int = None, height: int = None,
                 ppm: float = None, offset_x: float = None, offset_y: float = None):
        """
        Args:
            width, height: Screen size in pixels
            ppm: Pixels per meter (zoom)
            offset_x, offset_y: World point that appears at screen center (meters)
        """
        self.width = width if width is not None else config.render.width
        self.height = height if height is not None else config.render.height
        self.ppm = ppm if ppm is not None else config.render.ppm
        self.offset_x = offset_x if offset_x is not None else config.render.camera_offset_x
        self.offset_y = offset_y if offset_y is not None else config.render.camera_offset_y
    
    def world_to_screen(self, world_x: float, world_y: float):
        """Convert world coordinates (meters) to screen coordinates (pixels)."""
        sx = int((world_x - self.offset_x) * self.ppm + self.width / 2)
        sy = int(self.height - ((world_y - self.offset_y) * self.ppm + self.height / 2))
        return sx, sy
    
    def screen_to_world(self, sx: float, sy: float):
        """Convert screen coordinates (pixels) to world coordinates (meters)."""
        wx = (sx - self.width / 2) / self.ppm + self.offset_x
        wy = (self.height - sy - self.height / 2) / self.ppm + self.offset_y
        return wx, wy
    
    def world_scale(self):
        """Return scale factor (pixels per meter)."""
        return self.ppm