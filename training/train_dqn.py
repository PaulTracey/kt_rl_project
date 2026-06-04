# train_dqn.py
# ----------------------------------------------------------------------------------
# Training script for SARL DQN agent in the TAP project.
#
# Sources for implementation:
# - Based on the Stable-Baselines3 DQN implementation:
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py
# - Callback pattern (_on_step) adapted from Stable-Baselines3:
# https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
# - Episode statistics provided by the Monitor wrapper:
# https://stable-baselines3.readthedocs.io/en/master/guide/monitor.html
#
# Used for:
#   - LiveUICallback to integrate with pygame UI (GameManager) during training.
#   - TensorBoard logging of per-episode reward, win/draw rates.
#   - UI panel updates with last-step reward and episode counters.
#   - Configurable seeding and structured log directory per run.
#
# Gymnasium compatibility:
#   Wrap environments with RecordEpisodeStatistics (Gymnasium) or SB3's Monitor
#   so infos[0]["episode"]["r"] is available at episode end for logging.
# -------------------------------------------------------------------------------
import pygame
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from loguru import logger
from config import Config



# ----------------------------------------------------------------------------------
# Custom callback class: allows integration of training with UI and logging
# ----------------------------------------------------------------------------------

class LiveUICallback(BaseCallback):
    def __init__(self, env, log_dir, live_ui_mode=True, verbose=0, log_enabled=True):
        """
        A custom callback to integrate SB3 training with the Pygame UI and TensorBoard.
        """
        super().__init__(verbose)
        self.env = env
        self.live_ui_mode = live_ui_mode
        self.log_enabled = log_enabled

        # Track episode level statistics
        self.episode_count = 0
        self.total_wins = 0
        self.total_draws = 0
        self.kills_this_episode = 0

        # Create directory and TensorBoard writer
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

    def _on_step(self) -> bool:
        """
        Called by SB3 trainer after each step in the environment.
        """
        # Refresh the game UI on every step if live_ui_mode is enabled
        # Only try to update the UI if live_ui_mode is enabled.
        if self.live_ui_mode and hasattr(self.env, "game_manager"):
            self.env.game_manager.ui_manager.update(1 / 30.0)
            self.env.game_manager.redraw()
            pygame.display.update()

        # Access the step information provided by the SB3 callback system (infos dict).
        info = self.locals["infos"][0]

        # Check if an enemy was killed in this step.
        if info.get("enemy_incapacitated"):
            self.kills_this_episode += 1

        # When an episode ends (dones=True), log reward and outcomes
        if self.locals.get("dones") and self.locals["dones"][0]:
            info = self.locals["infos"][0]

            # Episode reward comes from SB3's Monitor wrapper
            if "episode" in info:
                self.episode_count += 1
                episode_reward = info["episode"]["r"]

                # print out testing totals
                print(f"DQN Episode {self.episode_count} finished. Total Reward: {episode_reward}")
                self.writer.add_scalar("Training/Episode_Reward", episode_reward, self.episode_count)

            # Get Win/draw information based on GameManager’s score
            game_manager = getattr(self.env, "game_manager", None)
            if game_manager:
                p1_score = game_manager.score.points.get("drop_p1", 0)
                p2_score = game_manager.score.points.get("drop_p2", 0)
                team = game_manager.current_team
                # Increment counters
                if ((team == "drop_p1" and p1_score > p2_score) or
                        (team == "drop_p2" and p2_score > p1_score)):
                    self.total_wins += 1
                elif p1_score == p2_score:
                    self.total_draws += 1

                # Log win/draw rates to TensorBoard
                if self.episode_count > 0:
                    self.writer.add_scalar("Training/WinRate", self.total_wins / self.episode_count, self.episode_count)
                    self.writer.add_scalar("Training/DrawRate", self.total_draws / self.episode_count, self.episode_count)

                # Update the UI panel
                last_step_reward = self.locals["rewards"][0]
                game_manager.update_training_stats(
                    reward=last_step_reward,
                    total_episodes=self.episode_count,
                    total_wins=self.total_wins,
                )

        return True # continue training

    def _on_training_end(self) -> None:
        """
            Automatically called by the Stable-Baselines3 trainer  once the .learn() process is complete. Used for cleanup actions.
        """
        # Close TensorBoard writer
        self.writer.close()

# ----------------------------------------------------------------------
# Main training
# ----------------------------------------------------------------------
def train(env, timesteps: int = 100_000, live_ui_mode: bool = False, log_enabled: bool = False, seed: int = None):
    """
    Initialise and train the DQN agent

    Args:
        env (gym.Env): The Gymnasium environment to train on.
        timesteps (int, optional): The total number of training steps. Defaults to 100,000.
        live_ui_mode (bool, optional): If True, renders the UI during training. Defaults to False.
        log_enabled (bool, optional): If True, enables detailed logging. Defaults to False.
        seed (int, optional): A random seed for reproducibility. Defaults to None.

    Returns:
        DQN: The trained Stable-Baselines3 DQN model.
    """
    # 1) Seed everything
    seed = seed or Config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)

    # 2) Prepare log directory
    run_name = f"DQN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("logs/dqn_logs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if log_enabled:
        logger.info(f"[DQN] Training for {timesteps} steps")
        logger.info(f"[DQN] Logs → {run_dir.resolve()}")

    # 3) Build and train the model
    model = DQN(
        "MlpPolicy", # Standard multilayer perceptron policy
        env,   # Training environment
        seed=seed,
        tensorboard_log=str(run_dir / "tensorboard"),
        verbose=0,  # 0 = silent, 1 = more logs

        # Core DQN hyperparameters -----
        learning_rate=1e-4,  # tried 1e-4 vs 3e-4 in the sweep
        buffer_size=200_000,  # tried 100k vs 200k
        learning_starts=20000,  # warmup steps before learning begins
        batch_size=32,  # standard for DQN
        gamma=0.99,  # discount factor

        train_freq=4,  # learn every 4 env steps (with gradient_steps below)
        gradient_steps=1,  # number of gradient updates per train step
        target_update_interval=20_000,  # hard update of target network, tried 10k vs 20k

        # Exploration (epsilon greedy)
        exploration_fraction=0.2,  # % of total timesteps to go from initial_eps to final_eps
        exploration_initial_eps=1.0,  # start fully random
        exploration_final_eps=0.05,  # end exploration rate; tried 0.05 vs 0.10

        #
        max_grad_norm=10.0,  # safeguard; SB3 default is 10.0
        device="auto", # Use a GPU if available
        # policy_kwargs=dict(net_arch=[256, 256]),  # optional for larger network
    )

    # Attach custom callback
    callback = LiveUICallback(env, log_dir=run_dir, live_ui_mode=live_ui_mode, log_enabled=log_enabled)
    # Train the model
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    # Save the model
    model.save(run_dir / "dqn_killteam_model")

    # Training complete messages
    if log_enabled:
        logger.info(f"[DQN] Training complete, model saved to: {run_dir.resolve()}")
    if live_ui_mode:
        print(f"DQN training UI updated in real time; logs under {run_dir}")

    return model

