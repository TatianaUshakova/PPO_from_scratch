import pygame # type: ignore
import torch
import torch.nn as nn
import numpy as np
import gym
from stable_baselines3.common.callbacks import BaseCallback #this is for bench with ppo stable baseline - getting info from this ppo
import pickle


class PPO_trainer():
    def __init__(self, env, policy_net, value_net, gamma):
        self.policy_net = policy_net
        self.value_net = value_net
        self.game = env
        self.gamma= gamma #disc factor 
        self.policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=3e-4)
        self.value_optimizer =torch.optim.Adam(value_net.parameters(), lr=3e-4)
        
        self.metrics = {
            'timesteps': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': []
        }

        self.total_timesteps = 0
        
    def collect_trajectories(self, num_trajectories): 
        #PPO algo: 'collect set of trajectories by running polcy in the environment'

        values = [] #predictions of expected reward from value_net to train value net
        rews_to_go = []
        actions = []
        states = []
        #log_prob = []#to do - see if the log prob increase num quality
        probs = []
        episode_returns = []
        episode_lengths = []

        def rewards_to_go(rewards, gamma):
                for i in reversed(range(len(rewards)-1)):
                    rewards[i] += gamma*rewards[i+1]
                return rewards

        for traj in range(num_trajectories): 
            #to do - num of traj should be modified to limited amount of totmal data
                
            state, _ = self.game.reset() 
            done = False
            rewards = []

            while not done: 

                if traj % 10 == 0:
                    self.game.render()

                state_tensor = torch.tensor(state, dtype=torch.float32)
                action = self.policy_net.get_action(state_tensor)
                prob = self.policy_net.get_prob(state_tensor, action=action)
                next_state, reward, terminated, truncated, _ = self.game.step(action.item())
                #action is torch tensor and should be converted to number to interact with env =>.item()
                
                done = terminated or truncated
                
                value = self.value_net(state_tensor)

                values.append(value)
                actions.append(action)
                rewards.append(reward)
                states.append(state_tensor)
                probs.append(prob)

                state = next_state

            #compute at the end of each game
            #rews_to_go+=rewards_to_go(rewards, self.gamma)
            episode_returns.append(sum(rewards))
            episode_lengths.append(len(rewards))
            rews_to_go += rewards_to_go(rewards, self.gamma)
    
        # Stack all tensor lists for correct furthure batch processing - these are already tensors from collection
        states = torch.stack(states)         
        actions = torch.stack(actions)       # Shape: (num_steps,)
        actions = actions.unsqueeze(1)       # Shape: (num_steps, 1) 
        values = torch.stack(values)         
        probs = torch.stack(probs)          

        rews_to_go = torch.tensor(rews_to_go, dtype=torch.float32)  # These were numbers as rewards come from env

        #compute advantages on all data
        advantages = rews_to_go - values.detach()  #do not optimize values in policy training

        return states, actions, rews_to_go, advantages, probs, episode_returns, episode_lengths

    def update_policy(self, actions, states, advantages, probs, epsilon_clip):       
        'update policy by PPO clip and fit value net by MSE'
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)#normalize advantages for better training
        
        print("\nDebugging probability calculations:") #TO DO - delete all debagging printing 
                                                    #or rewrite in an optional/some normal way
        print(f"Old probs sample: {probs[:5]}")
        print(f"Any NaN in old probs? {torch.isnan(probs).any()}")
        print(f"Any zeros in old probs? {(probs == 0).any()}")
        print(f"Min old prob: {probs.min()}")
        
        new_probs = self.policy_net.get_prob(states, action=actions)
        
        print(f"\nNew probs sample: {new_probs[:5]}")
        print(f"Any NaN in new probs? {torch.isnan(new_probs).any()}")
        print(f"Min new prob: {new_probs.min()}")
        
        old_probs = probs.detach()
        probs_ratio = new_probs / (old_probs + 1e-8)
        print(f"\nRatio calculation:")
        print(f"Sample division: {new_probs[:5]} / {old_probs[:5]} = {probs_ratio[:5]}")
        print(f"Any NaN in ratio? {torch.isnan(probs_ratio).any()}")
        print(f"Any inf in ratio? {torch.isinf(probs_ratio).any()}")
        print(f"Ratio range: {probs_ratio.min()} to {probs_ratio.max()}")
        print(f"Advantages range: min={advantages.min().item():.6f}, max={advantages.max().item():.6f}")
        
        term1 = probs_ratio * advantages
        term2 = torch.clamp(probs_ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantages
        
        print(f"Term1 range: min={term1.min().item():.6f}, max={term1.max().item():.6f}")
        print(f"Term2 range: min={term2.min().item():.6f}, max={term2.max().item():.6f}")
        
        loss = -torch.minimum(term1, term2).mean()
        print(f"Final loss: {loss.item():.6f}")

        self.policy_optimizer.zero_grad()
        loss.backward()
        
        for name, param in self.policy_net.named_parameters():
            if param.grad is not None:
                print(f"Gradient for {name}: min={param.grad.min().item():.6f}, max={param.grad.max().item():.6f}")
            else:
                print(f"No gradient for {name}")
        
        self.policy_optimizer.step()
        return loss.item()
    
    def update_value_nn(self, rews_to_go, states):
        loss = ((rews_to_go - self.value_net(states))**2).mean()

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        return loss.item()
    
    def train_one_step(self, num_trajectories, epsilon_clip, num_policy_updates = 5 ):
        '''
        Includes collecting the trajectories witht the current policy, make one update of the value nn and 
        make num_policy_updates of policy
        '''
        
        states, actions, rews_to_go, advantages, old_probs, episode_returns, episode_lengths = self.collect_trajectories(num_trajectories)
        
        value_loss = self.update_value_nn(rews_to_go, states)

        policy_losses = []
        
        for update in range(num_policy_updates):
            policy_loss = self.update_policy(actions, states, advantages, old_probs, epsilon_clip)
            policy_losses.append(policy_loss)
            #print(f"Policy update {update + 1}/{num_policy_updates}, Loss: {policy_loss:.6f}")
        
        return {
            #'mean_reward': rews_to_go.mean().item(),
            #'mean_length': len(states)/num_trajectories,
            'mean_reward': np.mean(episode_returns),
            'mean_length': np.mean(episode_lengths),
            'policy_loss': np.mean(policy_losses),  
            'value_loss': value_loss
        }
    
    def train(self, num_train_steps=100, num_trajectories_per_step=10, epsilon_clip=0.2, print_freq: None|int = 5, max_timesteps = None):

        for step_num in range(num_train_steps):

            one_step_metrics =  self.train_one_step(num_trajectories=num_trajectories_per_step, epsilon_clip=epsilon_clip)

            self.metrics['episode_rewards'].append(one_step_metrics['mean_reward'])
            self.metrics['episode_lengths'].append(one_step_metrics['mean_length'])
            self.metrics['policy_losses'].append(one_step_metrics['policy_loss'])
            self.metrics['value_losses'].append(one_step_metrics['value_loss'])

            self.total_timesteps += one_step_metrics['mean_length'] * num_trajectories_per_step
            self.metrics['timesteps'].append(self.total_timesteps)
            

            if print_freq is not None:
                if step_num % print_freq == 0:
                    print(f"Training step (Episode) number {step_num}: "
                      f"Mean Reward: {one_step_metrics['mean_reward']:.2f}, "
                      f"Mean Length: {one_step_metrics['mean_length']:.1f}, "
                      f"Policy Loss: {one_step_metrics['policy_loss']:.4f}, "
                      f"Value Loss: {one_step_metrics['value_loss']:.4f}")
                    plot_learning_curves([self.metrics], ['PPO'], img_name = f'learning_curves_{step_num//print_freq}.png', save=True)

            if max_timesteps and self.total_timesteps>max_timesteps:
                print(f'training finished after reaching the max number of training steps after {step_num} trajectories updates')
                break



class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Initialize the last layer with smaller weights to make initial policy more random
        nn.init.uniform_(self.model[-2].weight, -0.01, 0.01)
    def forward(self, x):
        return self.model(x)
        
    def get_action(self, state, from_distribution=True):
        '''
        if 'from_distribution' is selected, the sampling happen from the distribution corresponding to probabilities from the policy, othervise max prob action is selected
        '''
        probs = self.forward(state)
        if from_distribution:
            action = torch.multinomial(probs, 1).squeeze()  # Remove the extra dimension
        else:
            action = torch.argmax(probs)

        return action            
        
    def get_prob(self, state, *, action=None, max_prob=False):
        '''
        Function returns probability of action if action is specified or the probability of the most probable action if max_prob is selected
        '''
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        probs = self.forward(state)
        
        if len(probs.shape) == 2:  # Batch 
            if max_prob and action is not None:
                raise ValueError('Cannot specify both action and max_prob')
            
            if max_prob:
                return torch.max(probs, dim=1)[0]
            elif action is not None:
                action = action.long()
                # action is (batch_size, 1)
                return probs.gather(1, action).squeeze(1)
            else:
                raise ValueError('Must either provide action or set max_prob=True')
        else:  # Single input (no batch)
            if max_prob and action is not None:
                raise ValueError('Cannot specify both action and max_prob')
            
            if max_prob:
                return torch.max(probs)
            elif action is not None:
                return probs[action]
            else:
                raise ValueError('Must either provide action or set max_prob=True')


class ValueNet(nn.Module):
    ''' Neural network that estimates the average expected future return from a given state
      (averged across actions and future states)

    Input: state_dim: Dimention of the state space
    Output: Expected cummulative discounted return from the current state 

    '''
    def __init__(self, state_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, current_state):
        if not isinstance(current_state, torch.Tensor):
            current_state = torch.tensor(current_state, dtype=torch.float32)
            
        return self.model(current_state).squeeze(-1)    # shape: (batch_size,)






def plot_learning_curves(metrics_list, labels, img_name='learning_curves.png', show=True, save=True):
    """
    metrics_list: list of metrics dicts, containing 'timesteps', 'episode_rewards', 
    'episode_lengths', 'policy_losses', 'value_losses'
    labels: list of labels for each agent
    """
    import matplotlib.pyplot as plt # type: ignore
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
    if save:
        plt.savefig(img_name)
        #print(f"Plot saved to {img_name}")
    if show:
        plt.show(block=False)
        plt.pause(2)#seconds
        plt.close()



def benchmark(env, policy_net, value_net, gamma=0.99, num_train_steps=100, total_timesteps=50000, save_progress=True, visualize=True):
    try:
        from stable_baselines3 import PPO as SB3_PPO # type: ignore
    except:
        raise ImportError('Stable baseline is not installed')
    
    #---training and saving of my ppo-------------------------------------------
    my_ppo = PPO_trainer(env, policy_net, value_net, gamma)
    my_ppo.train(num_train_steps, max_timesteps=total_timesteps)

    if save_progress:
        torch.save(policy_net.state_dict(), "my_ppo_policy.pt")
        torch.save(value_net.state_dict(), "my_ppo_value.pt")
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
        sb3_ppo.save("sb3_ppo_agent")
        print("Saved SB3 PPO agent.")

    #---comparison ------------------------------------------------------------
    plot_learning_curves(
        [my_ppo.metrics, sb3_metrics],
        ['My PPO', 'SB3 PPO'],
        img_name='ppo_comparison.png',
        save=True
    )

    if save_progress:
        with open('ppo_metrics.pkl', 'wb') as f:
            pickle.dump(my_ppo.metrics, f)
        with open('sb3_metrics.pkl', 'wb') as f:
            pickle.dump(sb3_metrics, f)
        print("Progress saved to 'ppo_metrics.pkl' and 'sb3_metrics.pkl'.")

    # Visualize both agents' play after training for visual comparison 
    # Wait for user to react before showing the performance game
    
    if visualize:
        input("Training finished! Ready to see the performance of the trained agents? Stable baseline goes first. (Press Enter to continue)")
        visualize_agent(env, "SB3 PPO", lambda obs: sb3_ppo.predict(obs, deterministic=True)[0].item())
    
        input("Watching custom PPO (Press Enter to continue)")
        visualize_agent(env, "Custom PPO", lambda obs: policy_net.get_action(torch.tensor(obs, dtype=torch.float32), from_distribution=False).item())

def visualize_agent(env, agent_name, action_fn, num_episodes=3):
    import time
    #print(f"\n--- Watching {agent_name} ---")
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

def main(visualize=True, snake_env=False):
    print('\n\n')
    print('------------------new run------------------------------------------------------')
    if snake_env:
        from gym_style_snake import GymStyleSnake
        # Set render_mode="none" for training (no rendering)
        env = GymStyleSnake(aware_length=10, disallow_backward=True, render_mode="none")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    else:
        import gym
        env = gym.make("CartPole-v1", render_mode="human")  
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

    policy_net = PolicyNet(state_dim, action_dim)
    value_net = ValueNet(state_dim)
    
    benchmark(env, policy_net, value_net, gamma=0.99, num_train_steps=100, total_timesteps=30000, visualize=visualize)
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


if __name__ == "__main__":
    # To use snake: main(snake_env=True)
    # To use CartPole: main(snake_env=False)
    main(visualize=True, snake_env=True)
