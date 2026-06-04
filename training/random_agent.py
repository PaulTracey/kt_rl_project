#training/random_agent.py
# --------------------------------------------------------------
# Random Test Agent: runs a random-action agent inside the TAP
# environment. Used for testing environment reset(), step(),
# and UI updates without a trained model.
#
# Used for debugging and development.
# --------------------------------------------------------------

import pygame

def run_random_agent(env, timesteps=1000, live_ui_mode=True):
    """
    Randon action agent that takes random actions in the environment.

    Args:
        env (gym.Env): The Gymnasium environment to test.
        timesteps (int): The total number of steps to run.
        live_ui_mode (bool): If True, the Pygame UI will be updated after each step.
    """

    obs, _ = env.reset()
    total_reward = 0
    steps = 0

    while steps < timesteps:
        # Sample a random action from the environment
        action = env.action_space.sample()
        obs, reward, done, trunc, _ = env.step(action)
        total_reward += reward
        steps += 1
        # Reset when episode finishes
        if done or trunc:
            obs, _ = env.reset()
        # Update the UI for debugging
        if live_ui_mode and hasattr(env, "game_manager") and hasattr(env.game_manager, "ui_manager"):
            env.game_manager.ui_manager.update(1 / 30.0)
            env.game_manager.redraw()
            pygame.display.update()

    print(f"Run complete - {timesteps} timesteps - Total reward: {total_reward:.2f}")


