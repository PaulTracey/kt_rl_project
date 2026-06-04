## train_marl_ippo.py
# -----------------------------------------------------------------------------
# IPPO-style multi-agent PPO training for the TAP.
#
# Sources for implementation:
# - CleanRL PPO (single-agent, vector obs / CartPole style MLP):
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
# - CleanRL multi-agent PPO example (PettingZoo Atari):
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_pettingzoo_ma_atari.py
# - PyTorch PPO tutorial (CartPole, MLP actor-critic):
# https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html
#
# Used for:
# CleanRL-style PPO PettingZoo multi-agent adapted to IPPO with TAP env + MLP
# Baseline structure inspired by CleanRL PPO (single-agent).
# Adapted for PettingZoo parallel multi-agent via IPPO-style data packing.
# Additions from CleanRL marked with [ADDED] and [CHANGED] in comments.
# -----------------------------------------------------------------------------
# Future work:
#   - Refactor to combine repeated training code into shared helper functions.
#   - Expose internal policy and value outputs so saliency can be used.
#   - Add Action mask or legality filter.
# ---------------------------------------------------------------------------
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
# [ADDED] PettingZoo Parallel wrapper for TAP Environment
from training.make_training_env import make_training_env

def parse_args():
    """
    Parses command-line arguments for the training script.
    """
    # ------------------------------------------------------------------------------------
    # Command Line args
    # -------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="marl_ippo_cleanrl_style") # label name
    parser.add_argument("--seed", type=int, default=1) # radom seed
    parser.add_argument("--torch-deterministic", type=bool, default=True) # if true make reproducible PyTorch operations (slower)
    parser.add_argument("--cuda", type=bool, default=True)  # if true, use GPU if available
    parser.add_argument("--capture-video", type=bool, default=False) # Allow video for gameplay playback. Not used.

    #------------------------------------------------------------------------------------
    # Main PPO settings
    #-------------------------------------------------------------------------------------
    parser.add_argument("--total-timesteps", type=int, default=500_000) # total agent steps shared across agents
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)  # optimizer step size
    parser.add_argument("--anneal-lr", type=bool, default=True)  # linearly decay LR over training
    parser.add_argument("--num-envs", type=int, default=8)  # number of parallel environments
    parser.add_argument("--num-steps", type=int, default=128)  # steps per env
    parser.add_argument("--gamma", type=float, default=0.99)  # discount factor
    parser.add_argument("--gae-lambda", type=float, default=0.95) # Generalized Advantage Estimation: trade-off between bias vs variance
    parser.add_argument("--num-minibatches", type=int, default=4) # minibatches per PPO update
    parser.add_argument("--update-epochs", type=int, default=4)  # gradient update epochs per batch
    parser.add_argument("--norm-adv", type=bool, default=True)  # normalize advantages before update
    parser.add_argument("--clip-coef", type=float, default=0.2)   # policy clip parameter
    parser.add_argument("--clip-vloss", type=bool, default=True)  # clip value loss updates
    parser.add_argument("--ent-coef", type=float, default=0.02)  # entropy bonus for exploration
    parser.add_argument("--vf-coef", type=float, default=0.5)  #  Value function coefficient for the loss calculation
    parser.add_argument("--max-grad-norm", type=float, default=0.5) # Maximum norm of the gradients for gradient clipping
    parser.add_argument("--target-kl", type=float, default=None)  # Kullback–Leibler divergence, the distance between old and new policies
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)  # [ADDED] Total capacity of the experience buffer.
    return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialises a linear layer with orthogonal initialisation. Using as default.

    Args:
        layer: an nn.Module which is the PyTorch layer to initialise.
        std: The standard deviation for the weights.
        bias_const: The constant value for the bias.

    Returns:
         The nn.Module initialised layer.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ------------------------------------------------------------------------------------
# Agent network: Actor-Critic (MLP)
# ------------------------------------------------------------------------------------
# This module defines a shared-parameter actor-critic network:
# - Critic: outputs a scalar value estimate V(s).
# - Actor: outputs logits over discrete actions (used to form a Categorical policy).
#
# Removed CNN (used in image-based PPO for Atari in CleanRL)
# and replaced with a Multilayer Perceptron (MLP) to process vector observations
# from the TAP environment.
#
# References:
# - PyTorch PPO tutorial (MLP actor + critic):
#   https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html
# - TorchRL “Getting started with modules” (MLP backbones, probabilistic actors):
#   https://docs.pytorch.org/rl/0.7/tutorials/getting-started-1.html
# Adapted here for IPPO (Independent PPO) in a PettingZoo multi-agent setup:
# each agent shares this same network.
# ------------------------------------------------------------------------------------
class Agent(nn.Module):  # [ADDED] MLP instead of CNN
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        # -------------------------------------------------------------------------
        # Critic network: estimates state-value function V(s)
        # Input: observation vector of length obs_dim
        # Output: scalar value
        # -------------------------------------------------------------------------
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # -------------------------------------------------------------------------
        # Actor network: outputs logits for each action
        # Input: observation vector
        # Output: unnormalized logits over 'act_dim' discrete actions
        # -------------------------------------------------------------------------
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    # -------------------------------------------------------------------------
    # Helper: query critic only
    # -------------------------------------------------------------------------
    def get_value(self, x): return self.critic(x)

    # -------------------------------------------------------------------------
    # Helper: sample from policy and return action, log-probability, entropy, and value
    # - logits are passed to a Categorical distribution:
    #     https://pytorch.org/docs/stable/distributions.html#categorical
    # -------------------------------------------------------------------------
    def get_action_and_value(self, x, action=None):
        """
        Returns:
          action: integer action to send to the environment
          log_prob: log probability of the action, used in PPO update
          entropy: policy entropy, used for the exploration bonus
          value: state value V(s), used in advantage estimation and value loss
        """
        logits = self.actor(x); dist = Categorical(logits=logits)
        if action is None: action = dist.sample()
        # action goes to env.step(); others are stored for PPO training
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)



def make_envs(n: int):  # [ADDED]
    """
    Create (--num-envs) independent PettingZoo TAP environments.
    Each environment returns observations as a dictionary:
    - keys: agent IDs (e.g. "p1", "p2")
    - values: NumPy arrays (float32 vectors) of shape (obs_dim)
    """
    env_list = []
    for _ in range(n):
        env = make_training_env()
        env_list.append(env)
    return env_list

def main():
    """Main training loop."""
    args = parse_args()

    # ---------------------------------------------------------------------
    # Seeding and determinism
    # ---------------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # ---------------------------------------------------------------------
    # Device and run naming
    # ---------------------------------------------------------------------
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")


    # ---------------------------------------------------------------------
    # Logging setup
    # [ADDED] Custom logging root (under logs/marl_ppo_logs/<run_name>)
    # [CHANGED] to use specific project path
    # ---------------------------------------------------------------------
    run_name = f"KT_IPPO__{args.exp_name}__{args.seed}__{int(time.time())}"
    save_dir = os.path.join("logs", "marl_ppo_logs", run_name)
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(os.path.join(save_dir, "tensorboard"))
    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        for k, value in vars(args).items(): f.write(f"{k} = {value}\n")

    # ---------------------------------------------------------------------
    # [ADDED] Create multiple PettingZoo TAP environments
    # [CHANGED] CleanRL uses envs.single_observation_space / action_space
    #           Grab the first agent’s obs/action from the dict instead.
    # ---------------------------------------------------------------------
    envs = make_envs(args.num_envs)
    next_obs_dicts = []
    for i, env in enumerate(envs):
        obs, _ = env.reset(seed=args.seed + i)
        next_obs_dicts.append(obs)
    first_agent = list(next_obs_dicts[0].keys())[0]
    obs_dim = next_obs_dicts[0][first_agent].shape[0]
    act_dim = envs[0].action_space(first_agent).n

    # ---------------------------------------------------------------------
    # Agent and optimiser (same as CleanRL)
    # ---------------------------------------------------------------------
    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    base_lr = args.learning_rate

    # ---------------------------------------------------------------------
    # [CHANGED] To Flat IPPO buffers
    # CleanRL stores [num_steps, num_envs] (single agent per env).
    # Here everything is flattened into one big table of rows
    # as there are multiple agents per env (IPPO).
    # Each row = one agent acting at one timestep.
    # ---------------------------------------------------------------------
    batch_size = args.batch_size
    obs_buf      = torch.zeros((batch_size, obs_dim), device=device)  # current obs
    act_buf      = torch.zeros((batch_size, 1), device=device, dtype=torch.long)  # chosen action
    logp_buf     = torch.zeros((batch_size,), device=device)  # log-prob of action
    val_buf      = torch.zeros((batch_size,), device=device)  # value estimate V(s)
    rew_buf      = torch.zeros((batch_size,), device=device)  # reward
    done_buf     = torch.zeros((batch_size,), device=device)  # done flag
    next_obs_buf = torch.zeros((batch_size, obs_dim), device=device)  # next obs (if agent continues)
    next_msk_buf = torch.zeros((batch_size,), device=device)  # mask (1 = next_obs exists, 0 = agent terminated)

    # [ADDED] per-env episodic return for logging
    epi_returns = [0.0 for _ in envs]
    global_step = 0
    start_time = time.time()

    # [UNCHANGED] number of PPO updates (formula same, interpretation is flat batch)
    num_updates = max(1, args.total_timesteps // max(1, args.batch_size))
    for update in range(1, num_updates + 1):
        # -----------------------------------------------------------------
        # [UNCHANGED] learning-rate annealing (shrinks learning rate over training)
        # -----------------------------------------------------------------
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            for pg in optimizer.param_groups: pg["lr"] = base_lr * frac

        # -----------------------------------------------------------------------
        # [CHANGED] step each env, for each live agent, write one row to buffers
        #           handle dict observations, per-agent rewards/dones, and resets
        # CleanRL loops over [num_steps, num_envs] with single agent per env
        # Here the flat index loops over all live agents in each env step
        # -------------------------------------------------------------------------
        #  This will count how many agent experiences collected so far.
        idx = 0

        while idx < batch_size:
            for env_idx, env in enumerate(envs):
                if idx >= batch_size: break
                if not env.agents:
                    nobs, _ = env.reset(); next_obs_dicts[env_idx] = nobs
                cur_obs = next_obs_dicts[env_idx]

                agent_data_for_step = []
                actions = {}
                for agent_id, agent_obs in cur_obs.items():
                    obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        # The 'log_prob' (log probability) is essential for PPO.
                        # It represents the logarithm of the probability of choosing a specific action.
                        # This value is used in the loss function to determine whether to increase (positive reward)
                        # or decrease (negative reward) the likelihood of this action in the future.
                        # We use logarithms for numerical stability when dealing with very small probabilities.
                        action_tensor, log_prob, _, value = agent.get_action_and_value(obs_tensor)

                    a_item = int(action_tensor.squeeze(0).item())
                    actions[agent_id] = a_item
                    agent_data_for_step.append((agent_id, obs_tensor, a_item, log_prob.squeeze(0), value.squeeze(0)))

                start_row = idx
                for (aid, obs_tensor, a_item, log_prob, value) in agent_data_for_step:
                    if idx >= batch_size: break
                    obs_buf[idx]   = obs_tensor.squeeze(0)
                    act_buf[idx,0] = a_item
                    logp_buf[idx]  = log_prob
                    val_buf[idx]   = value
                    idx += 1
                # step env with the dict of actions
                nobs, rewards, terms, truncs, infos = env.step(actions)

                # [ADDED] Process the auxiliary 'info' dictionary from the environment.
                # Aavailable for debugging and for future work to log custom metrics.
                # for agent_id, agent_info in infos.items():
                #     if agent_info.get("game_over", False):
                #         print(f"DEBUG: Game over for agent {agent_id} in env {env_idx}!")


                # [ADDED] Track episodic return per environment
                # In CleanRL there is only one agent per env so the reward is just a single number.
                # In this multi-agent case, each step gives a dict of rewards (one per agent),
                # so the values are summed before logging. When any agent in the env finishes,
                # the total is logged and the counter is reset.
                epi_returns[env_idx] += float(sum(rewards.values()))
                # if episode ended, log once and reset
                if any(terms.values()) or any(truncs.values()):
                    writer.add_scalar("charts/episodic_return", epi_returns[env_idx], global_step)
                    epi_returns[env_idx] = 0.0

                # -----------------------------------------------------------------------
                # Back-fill the rows just written with reward/done/next_obs info
                # [CHANGED] Multi-agent detail: match each written row to its agent id.
                #           If an agent terminated this step, there is no next_obs so set mask = 0.
                # -----------------------------------------------------------------------
                wrote = idx - start_row
                wrote_agents = [pa[0] for pa in agent_data_for_step][:wrote]
                for j, aid in enumerate(wrote_agents):
                    row = start_row + j
                    rew = float(rewards.get(aid, 0.0))
                    done = bool(terms.get(aid, False) or truncs.get(aid, False))
                    rew_buf[row]  = rew
                    done_buf[row] = 1.0 if done else 0.0
                    if aid in nobs:
                        next_obs_buf[row] = torch.tensor(nobs[aid], dtype=torch.float32, device=device)
                        next_msk_buf[row] = 1.0  # next state exists for this agent
                    else:
                        next_msk_buf[row] = 0.0 # agent terminated this step (no next state)

                # Save the latest obs dict for this env for the next iteration
                # and increment the global step counter
                next_obs_dicts[env_idx] = nobs
                global_step += 1

        # -----------------------------------------------------------------------
        # Slice out the valid portion of the flat buffers (last chunk may be short)
        # [UNCHANGED IDEA] Same as CleanRL: work only on collected rows.
        # -----------------------------------------------------------------------
        valid = idx
        b_obs, b_act, b_logp = obs_buf[:valid], act_buf[:valid], logp_buf[:valid]
        b_val, b_rew, b_done = val_buf[:valid], rew_buf[:valid], done_buf[:valid]
        b_next_obs, b_next_msk = next_obs_buf[:valid], next_msk_buf[:valid]

        # -----------------------------------------------------------------------
        # [UNCHANGED] Generalized Advantage Estimation (GAE): compute advantages and returns
        # Same formulas as CleanRL. It's a way of estimating how good an action was compared to average.
        # -----------------------------------------------------------------------
        with torch.no_grad():
            next_vals = torch.zeros_like(b_val)
            mask_idx = b_next_msk > 0.5
            if mask_idx.any():
                next_vals[mask_idx] = agent.get_value(b_next_obs[mask_idx]).squeeze(-1)
            deltas = b_rew + args.gamma * (1.0 - b_done) * next_vals - b_val  # Temporal-Difference error
            advantages = torch.zeros_like(deltas); gae = 0.0
            for t in reversed(range(valid)):
                nonterminal = 1.0 - b_done[t]
                gae = deltas[t] + args.gamma * args.gae_lambda * nonterminal * gae  # Smooth advantages by blending current TD error with discounted future ones
                advantages[t] = gae
            returns = advantages + b_val  # final advantage + baseline value

        # -----------------------------------------------------------------------
        # [UNCHANGED] PPO update: clipped policy loss, value loss, entropy bonus
        # Standard CleanRL structure; minibatches taken from the flat buffers.
        # -----------------------------------------------------------------------
        inds = np.arange(valid)
        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            mb_size = max(1, valid // args.num_minibatches)
            for start in range(0, valid, mb_size):
                end = min(valid, start + mb_size)
                mb = inds[start:end]
                if len(mb) == 0: continue

                obs_mb = b_obs[mb]
                act_mb = b_act[mb].long().squeeze(-1)
                adv_mb = advantages[mb]
                ret_mb = returns[mb]
                val_mb = b_val[mb]


                # Gets the log probability of the action that was taken, as recorded during
                # the data collection phase. This is the 'old' policy's log_prob.
                old_logp_mb = b_logp[mb]

                if args.norm_adv and adv_mb.numel() > 1:
                    adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

                # Get the log_prob of the same action under the 'new' (current) policy.
                _, new_logp, entropy, new_values = agent.get_action_and_value(obs_mb, act_mb)
                # -----------------------------------------------------------------------
                # 1. Policy Loss (L_CLIP)
                # -----------------------------------------------------------------------
                # The ratio between the new and old policies' probabilities is the core of PPO.
                logratio = new_logp - old_logp_mb
                ratio = torch.exp(logratio)

                pg_loss1 = -adv_mb * ratio
                pg_loss2 = -adv_mb * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                # -----------------------------------------------------------------------
                # 2. Value Function Loss (L_VF)
                # -----------------------------------------------------------------------
                new_values = new_values.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (new_values - ret_mb) ** 2
                    v_clipped = val_mb + torch.clamp(new_values - val_mb, -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - ret_mb) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_values - ret_mb) ** 2).mean()

                entropy_loss = entropy.mean()
                # -----------------------------------------------------------------------
                # 3. Entropy Bonus (S)
                # -----------------------------------------------------------------------
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                # -----------------------------------------------------------------------
                # Optimization Step
                # -----------------------------------------------------------------------
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            # KL Divergence Check (Early Stopping)
            with torch.no_grad():
                approx_kl = (ratio - 1 - logratio).mean()
            if args.target_kl is not None and approx_kl.item() > args.target_kl:
                break

        # -----------------------------------------------------------------------
        # Logging
        # [CHANGED] SPS line to fix risk of divide by zero error.
        #   - added max(1e-6) so will never divide by zero.
        #   - removed unwanted variables
        # -----------------------------------------------------------------------
        sps = int(global_step / max(1e-6, time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)

    # Save everything in save_dir
    torch.save(agent.state_dict(), os.path.join(save_dir, "final_model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "final_optimizer.pt"))
    writer.close()
    for env in envs: env.close()

if __name__ == "__main__":
    main()
