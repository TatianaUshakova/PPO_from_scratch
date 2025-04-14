#from snake_rl import Game
import pygame
import torch
import torch.nn as nn
import numpy as np
from typing import TypeVar, Union, List
import matplotlib.pyplot as plt
import gym
from gym.spaces import Discrete, Box
   

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
        #convert to tensor if its not already
        #if not isinstance(state, torch.Tensor): -- no this is wrong - if not tensor already it will break comp graph
        #    state = torch.tensor(state, dtype=torch.float32)

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
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        probs = self.forward(state)
        
        if len(probs.shape) == 2:  # Batch case
            if max_prob and action is not None:
                raise ValueError('Cannot specify both action and max_prob')
            
            if max_prob:
                return torch.max(probs, dim=1)[0]
            elif action is not None:
                # Make sure action is a LongTensor for indexing
                action = action.long()
                # No need for additional unsqueeze since action is already (batch_size, 1)
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

#definition of state sype for clarity
State = TypeVar('State', torch.Tensor, np.ndarray, List[float]) 
ExpectedReturn = float   #or torch.Tensor if returning batches

class ValueNet(nn.Module):
    ''' Neural network that estimates the average expected future return from a given state (averged across actions and future states)

    Args:
        state_dim: Dimention of the state space
    
        Returns:
            Expected cummulative discounted reward from the current state 
    '''
    def __init__(self, state_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, current_state: State) -> ExpectedReturn:
        if not isinstance(current_state, torch.Tensor):
            current_state = torch.tensor(current_state, dtype=torch.float32)
            
        return self.model(current_state).squeeze(-1)    # shape: (batch_size,)


class PPO_trainer():
    def __init__(self, game, policy_net, value_net, gamma):
        #ppo algo: first have policy and values nets and set thier initial params
        self.policy_net = policy_net
        self.value_net = value_net
        self.game = game
        self.gamma= gamma #disc factor for reward
        self.policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-2)
        self.value_optimizer =torch.optim.Adam(value_net.parameters(), lr=1e-3)
        
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': []
        }
        
    def collect_trajectories(self, num_trajectories): #correct things should be converted to tensors
        #PPO algo: 'collect set of trajectories by running polcy in the environment'

        values = [] #predictions of expected reward after each timestep according to value_net and to ttrain value net
        rews_to_go = []
        actions = []
        states = []
        log_prob = []
        probs = []

        def rewards_to_go(rewards, gamma):
                for i in reversed(range(len(rewards)-1)):
                    rewards[i] += gamma*rewards[i+1]
                return rewards

        for traj in range(num_trajectories): 
            #num of traj should be modified to limited amount of totmal data
                
            
            state, _ = self.game.reset() #self.game.gamestart() -rewrite my game for consitency w gym!
            done = False
            rewards = []

            while not done: #not self.game.gameover:

                if traj % 10 == 0:
                    self.game.render()

                #these should be tensors with computational graph:
                #for value net optimization with respect to value net - rew to go can be constant
                #for policy net optimization optimization with respect to policy params that affect
                #advantages and policy probabilities. So we should save theis comp relation to policy nn
                #but: for efficient transfer to gpu all the things between input of nn and the all the things computed for loss should be tensor to transfer sll this part of computations to gpu
                    
                state_tensor = torch.tensor(state, dtype=torch.float32)

                action = self.policy_net.get_action(state_tensor)
                prob = self.policy_net.get_prob(state_tensor, action=action)
                #print(f"Prob requires grad: {prob.requires_grad}")  
                
                #self.game.action = action#needed for game to play based on action
                #reward = self.game.play_step() -- need to rewrite my game consistent with gym
                
                next_state, reward, terminated, truncated, _ = self.game.step(action.item())
                    #action is torch tensor and should be converted to number 
                    # to interact with env hence .item()
                done = terminated or truncated
                
                value = self.value_net(state_tensor)
                
                values.append(value)
                actions.append(action)
                rewards.append(reward)
                states.append(state_tensor)
                probs.append(prob)

                state = next_state
                #хуево у меня игра написана - у меня состояние определется считая что есть опр действие которое не определено с самого начала

            #compute at the end of each game
            rews_to_go+=rewards_to_go(rewards, self.gamma)
    
        # Stack all tensor lists for correct furthure batch processing - these are already tensors from collection
        states = torch.stack(states)         
        actions = torch.stack(actions)       # Shape: (num_steps,)
        actions = actions.unsqueeze(1)       # Shape: (num_steps, 1) 
        values = torch.stack(values)         
        probs = torch.stack(probs)          

        # Convert regular number lists to tensors
        rews_to_go = torch.tensor(rews_to_go, dtype=torch.float32)  # These were numbers as rewards come from env

        #compute advantages on all data
        advantages = rews_to_go - values.detach()  #do not optimize values in policy training

        return states, actions, rews_to_go, advantages, probs

    def update_policy(self, actions, states, advantages, probs, epsilon_clip):       
        'update policy by PPO clip and fit value net by MSE'
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)#normalize advantages for better training
        
        print("\nDebugging probability calculations:")
        print(f"Old probs sample: {probs[:5]}")
        print(f"Any NaN in old probs? {torch.isnan(probs).any()}")
        print(f"Any zeros in old probs? {(probs == 0).any()}")
        print(f"Min old prob: {probs.min()}")
        
        new_probs = self.policy_net.get_prob(states, action=actions)
        print(f"\nNew probs sample: {new_probs[:5]}")
        print(f"Any NaN in new probs? {torch.isnan(new_probs).any()}")
        print(f"Min new prob: {new_probs.min()}")
        
        # Calculate and inspect ratio
        old_probs = probs.detach()
        probs_ratio = new_probs / (old_probs + 1e-8)
        print(f"\nRatio calculation:")
        print(f"Sample division: {new_probs[:5]} / {old_probs[:5]} = {probs_ratio[:5]}")
        print(f"Any NaN in ratio? {torch.isnan(probs_ratio).any()}")
        print(f"Any inf in ratio? {torch.isinf(probs_ratio).any()}")
        print(f"Ratio range: {probs_ratio.min()} to {probs_ratio.max()}")
        
        # Debug advantages
        print(f"Advantages range: min={advantages.min().item():.6f}, max={advantages.max().item():.6f}")
        
        # Calculate loss terms separately for debugging
        term1 = probs_ratio * advantages
        term2 = torch.clamp(probs_ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantages
        
        print(f"Term1 range: min={term1.min().item():.6f}, max={term1.max().item():.6f}")
        print(f"Term2 range: min={term2.min().item():.6f}, max={term2.max().item():.6f}")
        
        loss = -torch.minimum(term1, term2).mean()
        print(f"Final loss: {loss.item():.6f}")
        
        # Check if gradients are flowing
        self.policy_optimizer.zero_grad()
        loss.backward()
        
        # Debug gradients
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
    
    def train_one_step(self, num_trajectories, epsilon_clip):
        # Collect trajectories with current policy
        states, actions, rews_to_go, advantages, old_probs = self.collect_trajectories(num_trajectories)
        
        policy_loss = self.update_policy(actions, states, advantages, old_probs.detach(), epsilon_clip)
        value_loss = self.update_value_nn(rews_to_go, states)
        
        return {
            'mean_reward': rews_to_go.mean().item(),
            'mean_length': len(states)/num_trajectories,
            'policy_loss': policy_loss,
            'value_loss': value_loss
        }
    
    def train(self, 
              num_train_steps=100, num_trajectories_per_step=10, epsilon_clip=0.2, print_freq = 10):

        for step_num in range(num_train_steps):
            #do one step update
            one_step_metrics =  self.train_one_step(num_trajectories=num_trajectories_per_step, epsilon_clip=epsilon_clip)

            self.metrics['episode_rewards'].append(one_step_metrics['mean_reward'])
            self.metrics['episode_lengths'].append(one_step_metrics['mean_length'])
            self.metrics['policy_losses'].append(one_step_metrics['policy_loss'])
            self.metrics['value_losses'].append(one_step_metrics['value_loss'])

            #print progress
            if step_num % print_freq == 0:
                print(f"Training step (Episode) number {step_num}: "
                      f"Mean Reward: {one_step_metrics['mean_reward']:.2f}, "
                      f"Mean Length: {one_step_metrics['mean_length']:.1f}, "
                      f"Policy Loss: {one_step_metrics['policy_loss']:.4f}, "
                      f"Value Loss: {one_step_metrics['value_loss']:.4f}")
                self.plot_learning_curves(save=True)


    def plot_learning_curves(self, save=False):
        plt.figure(figsize = (15,5) )

        #rewards
        plt.subplot(131)
        plt.plot(self.metrics['episode_rewards'])
        plt.title('Episode rewards')
        plt.xlabel('Episode')
        plt.ylabel('Mean Revard')

        #episode length
        plt.subplot(132)
        plt.plot(self.metrics['episode_lengths'])
        plt.title('Episode Lenghts')
        plt.xlabel('Episode')
        plt.ylabel('Mean Length')

        # losses
        plt.subplot(133)
        plt.plot(self.metrics['policy_losses'], label='Policy')
        plt.plot(self.metrics['value_losses'], label='Value')
        plt.title('Losses')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        if save:
            plt.savefig('learning_curves.png')
        
        plt.close()



def main():
    print('\n\n\n\n')
    env = gym.make("CartPole-v1", render_mode="human")  
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNet(state_dim, action_dim)
    value_net = ValueNet(state_dim)
    
    trainer = PPO_trainer(
        policy_net=policy_net,
        value_net=value_net,
        game=env,  
        gamma=0.99
    )

    #env.render()

    trainer.train()

    env.close()

if __name__ == "__main__":
    main()







        


            













        
# class default_Trainer():
#     def __init__(self, game):
#         self.game = game

#     def train_step():
#         pass

#     def train(self):
#         game.game_number = 0




# global_memory = []
# max_games = 250
# running_reward = 0
# print_every = 10

# while agent.game_num < max_games:
#     local_memory = []
#     game.game_number = agent.game_num  # Update game number before restart
#     game.gamestart()
#     episode_reward = 0
    
#     while not game.gameover:
#         state = agent.get_state(game)
#         action = agent.get_action(state)
#         game.action = action
#         reward = game.play_step()
#         episode_reward += reward  # Accumulate all rewards
#         action_idx = agent._point_to_idx(action)
#         local_memory.append((state, action_idx, reward))
    
#     metrics = agent.trainer.train_step(local_memory)
#     global_memory.append(metrics)
    
#     # Save model if score improves
#     agent.save_model(game.score)
    
#     # Update running reward using total episode reward
#     running_reward = 0.95 * running_reward + 0.05 * episode_reward
    
#     if agent.game_num % print_every == 0:
#         print(f'Game {agent.game_num}, Score: {game.score}, Running Reward: {running_reward:.2f}, '
#                 f'Total Reward: {episode_reward:.2f}, Length: {metrics["episode_length"]}, '
#                 f'Loss: {metrics["loss"]:.3f}, Best Score: {agent.best_score}')
    
#     agent.game_num += 1

#     # Optional: Early stopping if we reach good performance
#     if running_reward > 1000:  # Much higher threshold
#         print(f"Solved at game {agent.game_num}!")
#         break

#pygame.quit()
# new_game = True
# game = Game()
# while new_game:
#     while not game.gameover:
#         game.get_action()
#         reward = game.play_step()  # Get the reward value
#         #print(f"Current reward: {reward:.2f}")  # Optional: print to console too
#     #print if want to play again and then reinitialize the game    
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_n):
#             pygame.quit()
#             new_game = False
#         elif (event.type == pygame.KEYDOWN and event.key == pygame.K_y):
#             game.gamestart() 