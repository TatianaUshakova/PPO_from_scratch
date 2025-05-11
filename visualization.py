"""Visualization utilities for PPO training and evaluation.

This module provides functions for:
1. Plotting learning curves during training
2. Visualizing agent performance (live or recorded)
3. Saving training metrics and videos
"""

import matplotlib.pyplot as plt # type: ignore
import numpy as np
import time
import os
from datetime import datetime
from gym.wrappers import RecordVideo
from typing import List, Dict, Callable, Any, Union

def get_timestamped_folder(base_folder: str = "training_runs") -> str:
    """Create and return a timestamped folder path for saving training outputs.
    now the plots and models are saved to new folders every time and not overwrites
    
    Args:
        base_folder: Base folder name for all training runs
        
    Returns:
        Path to the timestamped folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(base_folder, timestamp)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def plot_learning_curves(metrics_list: List[Dict[str, List[float]]], 
                        labels: List[str], 
                        save_folder: str = None,
                        show: bool = True) -> None:
    """Plot learning curves for multiple agents.
    
    Args:
        metrics_list: List of metrics dictionaries, each containing:
            - 'timesteps': List of timestep values
            - 'episode_rewards': List of reward values
            - 'episode_lengths': List of episode length values
            - 'policy_losses': List of policy loss values (optional)
            - 'value_losses': List of value loss values (optional)
        labels: List of labels for each agent
        save_folder: Folder to save the plot (if None, uses timestamped folder)
        show: Whether to display the plot
    """
    if save_folder is None:
        save_folder = get_timestamped_folder()
    
    plt.figure(figsize=(15,5))

    # Rewards
    plt.subplot(141)
    for metrics, label in zip(metrics_list, labels):
        plt.plot(metrics['timesteps'], metrics['episode_rewards'], label=label)
        plt.title('Episode rewards')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.legend()

    # Episode length
    plt.subplot(142)
    for metrics, label in zip(metrics_list, labels):
        plt.plot(metrics['timesteps'], metrics['episode_lengths'], label=label)
        plt.title('Episode Lengths')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Length')
    plt.legend()

    # Policy loss 
    plt.subplot(143)
    for metrics, label in zip(metrics_list, labels):
        if 'policy_losses' in metrics:
            plt.plot(metrics['timesteps'], metrics['policy_losses'], label=label)
    plt.title('Policy Loss')
    plt.xlabel('Timesteps')
    plt.ylabel('Loss')
    plt.legend()

    # Value loss 
    plt.subplot(144)
    for metrics, label in zip(metrics_list, labels):
        if 'value_losses' in metrics:
            plt.plot(metrics['timesteps'], metrics['value_losses'], label=label)
    plt.title('Value Loss')
    plt.xlabel('Timesteps')
    plt.ylabel('Loss')
    plt.legend()
        
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_folder, 'learning_curves.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    if show:
        plt.show(block=False)
        plt.pause(2)#seconds
        plt.close()

def visualize_agent(env: Any, 
                   agent_name: str, 
                   action_fn: Callable[[np.ndarray], int], 
                   num_episodes: int = 3, 
                   save_folder: str = None,
                   record_video: bool = False) -> None:
    """Show live performance or record video of agent's performance.
    
    Args:
        env: The environment to run the agent in
        agent_name: Name of the agent for video naming
        action_fn: Function that returns actions given observations
        num_episodes: Number of episodes to record/show
        save_folder: Folder to save videos (if None, uses timestamped folder)
        record_video: Whether to record video (False for live performance)
    """
    if record_video:
        if save_folder is None:
            save_folder = get_timestamped_folder()
        video_folder = os.path.join(save_folder, "videos")
        os.makedirs(video_folder, exist_ok=True)
        agent_video_folder = os.path.join(video_folder, f"{agent_name.lower().replace(' ', '_')}")
        os.makedirs(agent_video_folder, exist_ok=True)
        # Wrap environment with RecordVideo
        env = RecordVideo(env, video_folder=agent_video_folder, episode_trigger=lambda x: True)
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            action = action_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            time.sleep(0.02)
        print(f"{agent_name} Episode {ep+1}: Reward = {total_reward}")
    
    env.close()
    if record_video:
        print(f"Videos saved in {agent_video_folder}")
