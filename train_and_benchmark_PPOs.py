"""Training and benchmarking utilities for PPO implementations.

This module provides functions for:
1. Training my PPO implementation
2. Training Stable Baselines3 PPO
3. Comparing performance between implementations
4. Visualizing and recording agent performance
5. Saving the models
"""

#TO DO: I guess would be reasnable to also set the same params for policy and value nets and tehy are for sb3 for fair comparison

import torch
import gym
import pickle
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback # type: ignore
from visualization import plot_learning_curves, visualize_agent
from PPO_from_scratch import PPO_trainer, PolicyNet, ValueNet
from parallelized_PPO import ParallelizedPPO
from gym_style_snake import GymStyleSnake

def benchmark(env, policy_net, value_net, gamma=0.99, num_train_steps=100, total_timesteps=50000, save_progress=True, render=False, show_live_performance=True, record_video=True, parallel=False):
    """Train and compare my PPO with Stable Baselines3 PPO
    
    Args:
        env: The environment to train on
        policy_net: My PPO policy network (ValueNet) or any other
        value_net: My PPO value network (PolicyNet) or any other
        gamma: Discount factor
        num_train_steps: Number of training steps
        total_timesteps: Total timesteps for training
        save_progress: Whether to save models and metrics
        render: Whether to render the environment during training (slows down training)
        show_live_performance: Whether to show live performance of agents after training
        record_video: Whether to record videos of agent performance (requires rgb_array support - TO DO: add it for snake)
        parallel: Whether to use parallel PPO implementation
    """
    try:
        from stable_baselines3 import PPO as SB3_PPO # type: ignore
    except:
        raise ImportError('Stable baseline is not installed')
    
    # Create timestamped folder for this run
    from visualization import get_timestamped_folder
    save_folder = get_timestamped_folder() #now the plots and models are saved to new folders every time and not overwrites
    print(f"\nSaving all outputs to: {save_folder}")
    
    #---training and saving of my ppo-------------------------------------------
    render_mode = 'none' if not render else 'human'
    if parallel:
        # Use parallelized PPO
        def create_env():
            if isinstance(env, GymStyleSnake):
                return GymStyleSnake(aware_length=10, disallow_backward=True, render_mode=render_mode)
            else:
                return gym.make(env.spec.id, render_mode=render_mode)
        
        my_ppo = ParallelizedPPO(create_env, policy_net, value_net, gamma, num_envs=4, render=render)
        my_ppo.train(num_train_steps, num_trajectories_per_step=10, max_timesteps=total_timesteps)
    else:
        # non parallilized PPO
        my_ppo = PPO_trainer(env, policy_net, value_net, gamma, render=render)
        my_ppo.train(num_train_steps, max_timesteps=total_timesteps)

    if save_progress:
        torch.save(policy_net.state_dict(), os.path.join(save_folder, "my_ppo_policy.pt"))
        torch.save(value_net.state_dict(), os.path.join(save_folder, "my_ppo_value.pt"))
        print("Saved custom PPO policy and value networks.")

    #--stable baseline ppo-------------------------------------------------------
    sb3_metrics = {
        'timesteps': [],
        'episode_rewards': [],
        'episode_lengths': []
    }
    sb3_ppo = SB3_PPO('MlpPolicy', env, verbose=1, gamma=gamma)
    sb3_callback = SB3MetricsCallback(sb3_metrics)
    sb3_ppo.learn(total_timesteps=total_timesteps, callback=sb3_callback)

    if save_progress:
        sb3_ppo.save(os.path.join(save_folder, "sb3_ppo_agent"))
        print("Saved SB3 PPO agent.")

    #---comparison ------------------------------------------------------------
    plot_learning_curves(
        [my_ppo.metrics, sb3_metrics],
        ['My PPO', 'SB3 PPO'],
        save_folder=save_folder
    )

    if save_progress:
        with open(os.path.join(save_folder, 'ppo_metrics.pkl'), 'wb') as f:
            pickle.dump(my_ppo.metrics, f)
        with open(os.path.join(save_folder, 'sb3_metrics.pkl'), 'wb') as f:
            pickle.dump(sb3_metrics, f)
        print("Progress saved to metrics files.")

    if record_video:
        # Create environment for video recording (need to have env with diff render mode for recording)
        if isinstance(env, GymStyleSnake):
            record_env = GymStyleSnake(aware_length=10, disallow_backward=True, render_mode="rgb_array")
        else:
            record_env = gym.make(env.spec.id, render_mode="rgb_array")
            
        print("Recording videos of agent performance...")
        visualize_agent(record_env, "SB3 PPO", lambda obs: sb3_ppo.predict(obs, deterministic=True)[0].item(), 
                       save_folder=save_folder, record_video=True)
        visualize_agent(record_env, "Custom PPO", lambda obs: policy_net.get_action(torch.tensor(obs, dtype=torch.float32), from_distribution=False).item(), 
                       save_folder=save_folder, record_video=True)
        record_env.close()
        print("Videos recorded successfully!")

    if show_live_performance:
        # Create environment for live visualization (need to have env with diff render mode for visualization if was trained with render=False)
        if isinstance(env, GymStyleSnake):
            live_env = GymStyleSnake(aware_length=10, disallow_backward=True, render_mode="human")
        else:
            live_env = gym.make(env.spec.id, render_mode="human")

        # Wait for user to react before showing the performance game       
        input("Training finished! Ready to see the live performance of the trained agents? Stable baseline goes first. (Press Enter to continue)")
        visualize_agent(live_env, "SB3 PPO", lambda obs: sb3_ppo.predict(obs, deterministic=True)[0].item(), record_video=False)
    
        input("Watching custom PPO (Press Enter to continue)")
        visualize_agent(live_env, "Custom PPO", lambda obs: policy_net.get_action(torch.tensor(obs, dtype=torch.float32), from_distribution=False).item(), record_video=False)
        
        live_env.close()


def main(render=False, show_live_performance=True, snake_env=False, record_video=True, parallel=False):
    """Main function to run PPO training and comparison"""
    print('\n\n')
    print('------------------new run------------------------------------------------------')
    render_mode = 'none' if not render else 'human'
    
    if snake_env:
        from gym_style_snake import GymStyleSnake
        env = GymStyleSnake(aware_length=10, disallow_backward=True, render_mode=render_mode)
    else:
        import gym
        env = gym.make("MountainCar-v0", render_mode=render_mode)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_net = PolicyNet(state_dim, action_dim)
    value_net = ValueNet(state_dim)
    
    benchmark(env, policy_net, value_net, 
        gamma=0.99, 
        num_train_steps=200, #200,  # was 100 for "CartPole-v1"
        total_timesteps=50000,  # was 30000 for "CartPole-v1"
        render=render,
        show_live_performance=show_live_performance,
        record_video=record_video, 
        parallel=parallel)  
    env.close()


class SB3MetricsCallback(BaseCallback):
    def __init__(self, metrics):
        super().__init__()
        self.metrics = metrics

    def _on_step(self) -> bool:
        # Check if episode is done
        if len(self.locals['infos']) > 0 and 'episode' in self.locals['infos'][0]:
            reward = self.locals['infos'][0]['episode']['r']
            length = self.locals['infos'][0]['episode']['l']
            self.metrics['timesteps'].append(self.num_timesteps)
            self.metrics['episode_rewards'].append(reward)
            self.metrics['episode_lengths'].append(length)
        return True


if __name__ == '__main__':
    # To use snake: main(snake_env=True, record_video=False)
    # To use MountainCar: main(snake_env=False, record_video=True)
    # To use parallel training: main(parallel=True)
    
    main(render=False, show_live_performance=True, snake_env=False, record_video=True, parallel=False)
