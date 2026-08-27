"""
testscene.py — Interactive biped test scene built on BotAPI only.

All keyboard handling, pause/speed/reset/demo/burst logic, and rendering are
abstracted inside BotAPI (see BotAPI.handle_event / handle_events). This file
is just a thin loop, deliberately kept tiny for students.

Controls (handled inside BotAPI):
    R reset · SPACE pause · UP/DOWN speed · T demo · B torque burst · ESC exit

CLI:
    python testscene.py                         # GUI scene
    python testscene.py --frames 240 --headless # headless, for CI/testing
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from botapi import BotAPI


def main() -> None:
    parser = argparse.ArgumentParser(description='Interactive biped test scene (BotAPI)')
    parser.add_argument('--frames', type=int, default=None,
                        help='Exit after N rendered frames (for testing).')
    parser.add_argument('--headless', action='store_true',
                        help='Run without a window (no rendering, no input).')
    args = parser.parse_args()

    api = BotAPI(render_mode='headless' if args.headless else 'human')
    frame_count = 0

    try:
        api.reset()
        while api.is_running():
            # Input -> scene commands (reset / pause / speed / demo / burst / exit)
            for msg in api.handle_events():
                print(msg)

            # Physics
            if not api.paused:
                for _ in range(int(round(api.speed))):
                    api.step(use_scene_mode=True)

            # Render + pacing
            api.render(paused=api.paused, speed=api.speed)
            api.tick(60)
            frame_count += 1

            # Periodic one-line state summary
            if frame_count % 60 == 0:
                api.print_state_summary()

            # Optional frame limit for automated testing
            if args.frames is not None and frame_count >= args.frames:
                break
    finally:
        api.close()

    print("Test scene closed.")


if __name__ == '__main__':
    main()