# train_ppo.py
# ---------------------------------------------------------
# Training script for SARL PPO agent in the TAP project.
#
# Sources for implementation:
# - Based on the Stable-Baselines3 PPO implementation:
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
# - Callback pattern (_on_step) adapted from Stable-Baselines3:
# https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
# - Uses episode statistics provided by the Monitor wrapper:
# https://stable-baselines3.readthedocs.io/en/master/guide/monitor.html
#
# used for:
# Added LiveUICallback to integrate with pygame UI (GameManager).
# Added TensorBoard logging for rewards, win/draw rates.
# Custom logging using loguru for debugging and logs.
# Script is Gymnasium-compatible when environments are wrapped
# with RecordEpisodeStatistics or SB3's Monitor.
# ---------------------------------------------------------
# Future work:
#   - Add PPO Action mask
#   - Refactor for repetition
# ---------------------------------------------------------------------------
import pygame
import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed
import random
from pathlib import Path
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
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
        self.kills_this_episode = 0  #

        # Create directory and TensorBoard writer
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

    def _on_step(self) -> bool:
        """
        Called by SB3 trainer after each step in the environment.
        """
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

            # Episode reward comes from SB3's Monitor wrapper
            if "episode" in info:
                self.episode_count += 1
                episode_reward = info["episode"]["r"]
                print(f"Episode {self.episode_count} finished. Total Reward: {episode_reward}")

                self.writer.add_scalar("Training/Episode_Reward", episode_reward, self.episode_count)
                self.writer.add_scalar("Training/Kills_Per_Episode", self.kills_this_episode, self.episode_count)

                self.kills_this_episode = 0



            game_manager = getattr(self.env, "game_manager", None)
            if game_manager:
                p1_score = game_manager.score.points.get("drop_p1", 0)
                p2_score = game_manager.score.points.get("drop_p2", 0)
                team = game_manager.current_team

                # W# Get Win/draw information based on GameManager’s score
                if ((team == "drop_p1" and p1_score > p2_score) or
                        (team == "drop_p2" and p2_score > p1_score)):
                    self.total_wins += 1
                    if self.log_enabled:
                        logger.info(f"[WIN] Episode {self.episode_count} won by {team}")
                elif p1_score == p2_score:
                    self.total_draws += 1
                    if self.log_enabled:
                        logger.info(f"[DRAW] Episode {self.episode_count} drawn on {p1_score} points")

                # Log win/draw rates to TensorBoard using the correct episode counter
                if self.episode_count > 0:
                    self.writer.add_scalar("Training/WinRate", self.total_wins / self.episode_count, self.episode_count)
                    self.writer.add_scalar("Training/DrawRate", self.total_draws / self.episode_count,
                                           self.episode_count)

                # Update the UI training panel with the latest info.
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
    """Initialises and trains a PPO model.

    Args:
        env (gym.Env): The Gymnasium environment to train on.
        timesteps (int, optional): The total number of training steps. Defaults to 100,000.
        live_ui_mode (bool, optional): If True, renders the Pygame UI during training. Defaults to False.
        log_enabled (bool, optional): If True, enables detailed logging. Defaults to False.
        seed (int, optional): A random seed for reproducibility. Defaults to None.

    Returns:
        PPO: The trained Stable-Baselines3 PPO model.
    """
    # Add global seeds
    seed = seed or Config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)


    # Logs folder with timestamped run subfolder
    run_name = f"PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("logs/ppo_logs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if log_enabled:
        logger.info(f"[PPO] Training for {timesteps} timesteps")
        logger.info(f"[UI MODE] live_ui_mode = {live_ui_mode}")
        logger.info(f"[LOGGING] All logs will be saved in: {run_dir.resolve()}")

    model = PPO(
        "MlpPolicy", # "MlpPolicy" means a standard neural network (multi-layer perceptron).
        env,
        seed=seed, # The starting number for the random number generator.
        tensorboard_log=str(run_dir / "tensorboard"),

        # Core PPO settings
        n_steps=512,  # Steps collected before each learning phase. (sweep: 2048, 1024, 512)
        # (Try 1024 in the sweep if episodes are very short.)
        batch_size=64,  # Mini-batch size for gradient updates.
        # Must divide n_steps * n_envs (with 1 env, divide n_steps).
        n_epochs=10,  # How many passes over the rollout data each update.
        learning_rate=3e-4,  # How fast the model learns (higher = riskier, lower = slower).
        clip_range=0.2,  # PPO safety clamp to prevent overly big policy changes.
        gamma=0.99,  # How much future rewards matter (close to 1 = cares about the future).
        gae_lambda=0.95,  # Smoother advantage estimates for more stable learning.
        ent_coef=0.0,  # Entropy bonus (encourages exploration).
        # Sweep values: 0.0, 0.005, 0.01, 0.02.
        vf_coef=0.5,  # Weight of the value-function loss in the total loss.
        max_grad_norm=0.5,  # Caps the size of gradient updates (prevents instability).
        normalize_advantage=True,  # Standardize advantages for steadier updates.
        verbose=0,  # 0 = quiet; 1 = some logs.
    )
    # Attach custom callback
    callback = LiveUICallback(env, log_dir=run_dir, live_ui_mode=live_ui_mode, log_enabled=log_enabled)
    # Train the model
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    # Save the model
    model.save(run_dir / "ppo_killteam_model")
    # Training complete messages
    if log_enabled:
        logger.info(f"[PPO] Training complete, model saved to: {run_dir.resolve()}")

    return model
