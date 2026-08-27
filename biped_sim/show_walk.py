"""show_walk.py — Watch the trained biped walk (greedy policy)."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from botapi import BotAPI

from basic_rl import DQNAgent, STATE_DIM, N_ACTIONS, ACTIONS, build_state, MAX_STEPS


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Watch trained biped walk')
    parser.add_argument('--model', default='biped_dqn.pt')
    parser.add_argument('--frames', type=int, default=None)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[show_walk] model not found: {args.model}. Train with: python basic_rl.py")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[show_walk] device: {device}")

    api = BotAPI(render_mode='headless' if args.headless else 'human')
    agent = DQNAgent(STATE_DIM, N_ACTIONS, device)
    agent.load(args.model)

    fc = run = 0
    try:
        api.reset_standing()
        start_x = api.get_state()['torso_x']

        while api.is_running():
            for msg in api.handle_events():
                print(msg)
            obs = api.get_state()
            s = build_state(obs)
            a = agent.act(s, 0.0)
            api.apply_action_vector(ACTIONS[a])
            obs2 = api.step()
            api.render(paused=api.paused, speed=api.speed)
            api.tick(60)
            fc += 1

            if bool(obs2['torso_contact']) or obs2['torso_y'] < 0.4:
                run += 1
                print(f"[run {run}] walked {obs2['torso_x']-start_x:.2f}m -> resetting")
                api.reset_standing()
                start_x = api.get_state()['torso_x']

            if args.frames and fc >= args.frames:
                break
    finally:
        api.close()
    print(f"[show_walk] done: {run} runs, {fc} frames")


if __name__ == '__main__':
    main()
