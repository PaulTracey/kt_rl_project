# ui/saliency_map.py
# ---------------------------------------------------------
# Saliency map generator module for TAP project.
#
# Sources for implementation:
# - Saliency method (Simonyan et al., 2013), gradients of output wrt input:
# https://arxiv.org/abs/1312.6034
# - PyTorch autograd for computing gradients:
# https://pytorch.org/docs/stable/autograd.html
# - Stable-Baselines3 PPO and DQN policies (for logits / Q-values):
# https://stable-baselines3.readthedocs.io/en/master/
# - Pygame Surface API (for heatmap overlay drawing):
# https://www.pygame.org/docs/ref/surface.html
#
# Used for:
# Implemented saliency map for PPO (returns saliency, confidence, value).
# Implemented saliency map for DQN (returns saliency, action index).
# Added drawing overlay with Pygame (heatmap on board tiles).
#
# Future work:
# - Extend to MARL IPPO
# ---------------------------------------------------------
import torch
import numpy as np
import pygame


# --------------------------------------------------------------
# PPO: saliency + confidence
# --------------------------------------------------------------
def get_saliency_map_ppo(model, obs, action_index=None, device=None):
    """Calculates the saliency map for a PPO model.

    Args:
        model: The trained Stable-Baselines3 PPO model.
        obs (np.ndarray): A 1D numpy array of the environment observation.
        action_index (int, optional): Action to analyze. Defaults to the highest probability action.
        device (str, optional): PyTorch device ('cpu' or 'cuda'). Defaults to the model's device.

    Returns:
        tuple: A tuple containing the saliency map (np.ndarray), action index (int),
        action confidence (float), and state value V(s) (float).
    """
    policy = model.policy
    dev = policy.device if (device is None and hasattr(policy, "device")) else (device or "cpu")
    policy.eval()

    with torch.enable_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
        obs_tensor.requires_grad_(True)

        # Extract features -> latent -> logits/value
        try:
            features = policy.extract_features(obs_tensor)
        except TypeError:
            features = policy.extract_features(obs_tensor, getattr(policy, "features_extractor", None))
        if features is None:
            features = policy.features_extractor(obs_tensor)

        latent_pi, latent_vf = policy.mlp_extractor(features)
        logits = policy.action_net(latent_pi)
        values = policy.value_net(latent_vf).squeeze(1)
        probs  = torch.softmax(logits, dim=1)

        if action_index is None:
            action_index = int(torch.argmax(logits, dim=1).item())

        confidence = float(probs[0, action_index].item())
        value      = float(values[0].item())

        # Clear stale grads
        for param in policy.parameters():
            if param.grad is not None:
                param.grad = None

        # Backprop to inputs
        logits[0, action_index].backward()
        saliency = obs_tensor.grad.detach().abs().squeeze().cpu().numpy()

        return saliency, action_index, confidence, value


# --------------------------------------------------------------
# DQN: saliency only does not have action probabilities
# --------------------------------------------------------------
def get_saliency_map(model, obs, action_index=None, device=None):
    """Calculates the saliency map for a DQN model.

    Args:
        model: The trained Stable-Baselines3 DQN model.
        obs (np.ndarray): A 1D numpy array of the environment observation.
        action_index (int, optional): Action to analyze. Defaults to the highest Q-value action.
        device (str, optional): PyTorch device ('cpu' or 'cuda'). Defaults to the model's device.

    Returns:
        A tuple containing the saliency map (np.ndarray) and action index (int), or (None, None) on error.

    """
    try:
        policy = model.policy
        dev = policy.device if (device is None and hasattr(policy, "device")) else (device or "cpu")

        with torch.enable_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
            obs_tensor.requires_grad_(True)

            q_network = policy.q_net
            try:
                action_scores = q_network(obs_tensor)  # (1, A)
            except Exception:
                feats = policy.features_extractor(obs_tensor)
                action_scores = q_network(feats)

            if action_index is None:
                action_index = int(torch.argmax(action_scores, dim=1).item())

            action_scores[0, action_index].backward()
            grad_tensor = obs_tensor.grad

            grad_detached = grad_tensor.detach()
            grad_abs = grad_detached.abs()
            grad_squeezed = grad_abs.squeeze()
            grad_cpu = grad_squeezed.cpu()
            saliency = grad_cpu.numpy()

            return saliency, action_index

    except Exception as e:
        print(f"[Saliency Error DQN] {e}")
        return None, None


# --------------------------------------------------------------
# Drawing overlay (shared by PPO + DQN)
# --------------------------------------------------------------
def draw_saliency_overlay(surface, saliency, grid_rows, grid_cols, cell_size, top_offset, left_padding):
    """
    Paints a transparent red heatmap over the game grid.

    The intensity of each tile's color is determined by its corresponding
    saliency value, highlighting the most influential parts of the observation.

    Args:
        surface (pygame.Surface): The main Pygame surface to draw on.
        saliency (np.ndarray): A 1D numpy array of saliency values.
        grid_rows (int): The number of rows in the grid.
        grid_cols (int): The number of columns in the grid.
        cell_size (int): The size of each grid cell in pixels.
        top_offset (int): The vertical offset of the grid from the top of the screen.
        left_padding (int): The horizontal offset of the grid from the left of the screen.
    """
    if saliency is None:
        return

    grid_values = saliency[:grid_rows * grid_cols].reshape((grid_rows, grid_cols))
    abs_vals = np.abs(grid_values)

    # Normalize the saliency values to a 0-1 range based on a percentile clip for better contrast.
    low_percentile = float(np.percentile(abs_vals, 60)) # Saliency values below this are ignored (transparent).
    high_percentile = float(np.percentile(abs_vals, 98)) # Saliency values above this are max intensity.
    # Use max() to prevent a division-by-zero error if all saliency values are the same.
    denominator = max(high_percentile -  low_percentile, 1e-8)
    normalized_values = np.clip((abs_vals -  low_percentile) / denominator, 0.0, 1.0)
    normalized_values = normalized_values ** 1.8  # brighten contrast

    alpha_max = 160
    for row in range(grid_rows):
        for col in range(grid_cols):
            alpha = int(alpha_max * normalized_values[row, col])
            if alpha <= 0:
                continue

            # Create a temporary surface for this tile that supports transparency.
            overlay = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
            # Fill the overlay with red using the calculated alpha for its transparency.
            overlay.fill((255, 0, 0, alpha))
            pixel_x = left_padding + col * cell_size
            pixel_y = top_offset + row * cell_size
            surface.blit(overlay, (pixel_x, pixel_y))
