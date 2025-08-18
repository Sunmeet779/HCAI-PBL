import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import json
from .environment import MouseEnvironment

class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE algorithm"""
    def __init__(self, state_size=150, hidden_size=128, action_size=4):  # Updated to 150 for one-hot encoded 6x5x5
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten grid
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        action_probs = F.softmax(self.fc3(x), dim=1)
        return action_probs

class RewardNetwork(nn.Module):
    """Reward model for learning from human feedback (Task 2)"""
    def __init__(self, state_size=150, hidden_size=64):  # Updated to 150 for one-hot encoded 6x5x5
        super(RewardNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten grid
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        reward = self.fc3(x)
        return reward

class REINFORCEAgent:
    def __init__(self, state_size=150, action_size=4, lr=0.001):  # Updated to 150 for one-hot encoded 6x5x5
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        
        # Policy network
        self.policy_net = PolicyNetwork(state_size, 128, action_size)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Reward model for RLHF
        self.reward_net = RewardNetwork(state_size, 64)
        self.reward_optimizer = optim.Adam(self.reward_net.parameters(), lr=0.001)
        
        # Storage for trajectory data
        self.reset_trajectory()
    
    def reset_trajectory(self):
        """Reset trajectory storage"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
    def get_action(self, state):
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_net(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
    
    def store_transition(self, state, action, reward, log_prob):
        """Store trajectory data"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
    
    def compute_returns(self, rewards, gamma=0.95):  # Increased gamma for longer-term thinking
        """Compute discounted returns with organic cheese bonus"""
        returns = []
        R = 0
        
        # Add bonus for collecting organic cheese
        enhanced_rewards = []
        for reward in rewards:
            if reward > 20:  # Organic cheese reward (25)
                enhanced_rewards.append(reward * 1.5)  # 1.5x multiplier for organic cheese!
            elif reward > 2 and reward < 10:  # Regular cheese reward (3-5)
                enhanced_rewards.append(reward * 0.5)  # Reduce regular cheese reward
            else:
                enhanced_rewards.append(reward)
        
        for reward in reversed(enhanced_rewards):
            R = reward + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        # Lighter normalization to preserve organic cheese preference
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        return returns
    
    def train_policy(self, gamma=0.99):
        """Train policy using REINFORCE algorithm (Task 1)"""
        if len(self.rewards) == 0:
            return
            
        returns = self.compute_returns(self.rewards, gamma)
        
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        self.policy_optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item()
    
    def generate_trajectory(self, env, max_steps=50):
        """Generate a complete trajectory for human feedback"""
        env.reset()
        states = []
        actions = []
        rewards = []
        total_reward = 0
        
        for step in range(max_steps):
            state = env._get_state()
            states.append(state.copy())
            
            action, _ = self.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            actions.append(env.ACTIONS[action])
            rewards.append(reward)
            total_reward += reward
            
            if done:
                break
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'total_reward': total_reward,
            'steps': len(states)
        }
    
    def train_initial_policy(self, env, num_episodes=500):
        """Train initial policy using standard rewards (Task 1)"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            self.reset_trajectory()
            total_reward = 0
            
            for step in range(50):  # Max 50 steps per episode
                action, log_prob = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                self.store_transition(state, action, reward, log_prob)
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Train policy after each episode
            loss = self.train_policy()
            episode_rewards.append(total_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
        
        return self.policy_net.state_dict()
    
    def train_reward_model(self, trajectory_pairs, preferences, epochs=100):
        """Train reward model using Bradley-Terry model (Task 2)"""
        self.reward_net.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for (traj1, traj2), preference in zip(trajectory_pairs, preferences):
                # Convert trajectory states to tensors
                states1 = torch.FloatTensor(traj1['states'])
                states2 = torch.FloatTensor(traj2['states'])
                
                # Get predicted rewards for each trajectory
                rewards1 = self.reward_net(states1).squeeze()
                rewards2 = self.reward_net(states2).squeeze()
                
                # Sum rewards over trajectory
                traj1_reward = rewards1.sum()
                traj2_reward = rewards2.sum()
                
                # Bradley-Terry model: P(traj1 > traj2) = exp(r1) / (exp(r1) + exp(r2))
                # Loss is negative log likelihood of observed preference
                if preference == 0:  # traj1 preferred
                    loss = -F.logsigmoid(traj1_reward - traj2_reward)
                else:  # traj2 preferred
                    loss = -F.logsigmoid(traj2_reward - traj1_reward)
                
                total_loss += loss
            
            if len(trajectory_pairs) > 0:
                avg_loss = total_loss / len(trajectory_pairs)
                
                self.reward_optimizer.zero_grad()
                avg_loss.backward()
                self.reward_optimizer.step()
                
                if epoch % 20 == 0:
                    print(f"Reward Model Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        return self.reward_net
    
    def train_with_learned_rewards(self, env, episodes=300, kl_penalty=0.1):
        """Train policy with learned rewards + KL penalty (Task 3)"""
        # Save original policy for KL penalty
        original_policy = PolicyNetwork(self.state_size, 128, self.action_size)
        original_policy.load_state_dict(self.policy_net.state_dict())
        original_policy.eval()
        
        episode_rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            self.reset_trajectory()
            total_reward = 0
            kl_total = 0
            
            for step in range(50):
                action, log_prob = self.get_action(state)
                next_state, _, done, _ = env.step(action)  # Ignore env reward
                
                # Use learned reward instead
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                learned_reward = self.reward_net(state_tensor).item()
                
                # Add KL penalty to keep policy close to original
                with torch.no_grad():
                    old_probs = original_policy(state_tensor)
                    new_probs = self.policy_net(state_tensor)
                    kl_div = F.kl_div(F.log_softmax(new_probs, dim=1), 
                                     F.softmax(old_probs, dim=1), 
                                     reduction='batchmean')
                    kl_total += kl_div.item()
                
                # Combine learned reward with KL penalty
                final_reward = learned_reward - kl_penalty * kl_div.item()
                
                self.store_transition(state, action, final_reward, log_prob)
                state = next_state
                total_reward += learned_reward
                
                if done:
                    break
            
            # Train policy with learned rewards
            loss = self.train_policy()
            episode_rewards.append(total_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_kl = kl_total / max(len(self.states), 1)
                print(f"RLHF Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg KL: {avg_kl:.4f}")
        
        return self.policy_net.state_dict()
    
    def evaluate_trajectory(self, trajectory):
        """Evaluate a trajectory using learned reward model"""
        states = torch.FloatTensor(trajectory['states'])
        with torch.no_grad():
            rewards = self.reward_net(states).squeeze()
            total_reward = rewards.sum().item()
        return total_reward