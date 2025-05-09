from hackaikit.core.base_module import BaseModule
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import gym
from collections import deque
import pandas as pd

class QNetwork(nn.Module):
    """Simple Q-Network for discrete action spaces"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PolicyNetwork(nn.Module):
    """Simple Policy Network for discrete action spaces"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
        
    def __len__(self):
        return len(self.buffer)

class ReinforcementLearningModule(BaseModule):
    """
    Module for reinforcement learning tasks.
    Supports Q-learning, DQN, and Policy Gradient methods.
    """
    def __init__(self, config_manager=None):
        super().__init__(config_manager)
        self.env = None
        self.algorithm = None
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.q_table = None
        self.state_dim = None
        self.action_dim = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = None
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.exploration_min = 0.01
        self.gamma = 0.99
        
    def process(self, data=None, task="train", **kwargs):
        """Main processing method that routes to appropriate task"""
        if task == "train":
            return self.train_agent(**kwargs)
        elif task == "evaluate":
            return self.evaluate_agent(**kwargs)
        elif task == "test":
            return self.test_agent(**kwargs)
        else:
            return f"Reinforcement learning task '{task}' not supported."
    
    def setup_environment(self, env_name, **kwargs):
        """
        Set up the reinforcement learning environment
        
        Args:
            env_name (str): Name of the Gym environment or 'custom'
            
        Returns:
            dict: Environment information
        """
        try:
            if env_name == 'custom':
                # For custom environments, they should be passed as objects
                if 'custom_env' in kwargs:
                    self.env = kwargs['custom_env']
                else:
                    return "For custom environments, provide the environment object as 'custom_env'"
            else:
                # Use gym environment
                self.env = gym.make(env_name)
            
            # Get state and action dimensions
            if isinstance(self.env.observation_space, gym.spaces.Discrete):
                self.state_dim = self.env.observation_space.n
            else:
                self.state_dim = self.env.observation_space.shape[0]
                
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                self.action_dim = self.env.action_space.n
            else:
                self.action_dim = self.env.action_space.shape[0]
            
            return {
                "env_name": env_name,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "observation_space": str(self.env.observation_space),
                "action_space": str(self.env.action_space)
            }
        
        except Exception as e:
            return f"Error setting up environment: {str(e)}"
    
    def train_agent(self, algorithm="dqn", env_name="CartPole-v1", episodes=500, 
                    max_steps=200, batch_size=64, update_target_freq=10, **kwargs):
        """
        Train a reinforcement learning agent
        
        Args:
            algorithm (str): RL algorithm to use (q_learning, dqn, policy_gradient)
            env_name (str): Name of the Gym environment
            episodes (int): Number of episodes to train
            max_steps (int): Maximum steps per episode
            batch_size (int): Batch size for training (DQN)
            update_target_freq (int): Frequency to update target network (DQN)
            
        Returns:
            dict: Training results
        """
        # Set up environment
        env_setup = self.setup_environment(env_name, **kwargs)
        if isinstance(env_setup, str):  # Error message
            return env_setup
            
        # Set algorithm
        self.algorithm = algorithm
        
        # Set hyperparameters from kwargs
        self.gamma = kwargs.get('gamma', 0.99)
        self.exploration_rate = kwargs.get('exploration_rate', 1.0)
        self.exploration_decay = kwargs.get('exploration_decay', 0.995)
        self.exploration_min = kwargs.get('exploration_min', 0.01)
        learning_rate = kwargs.get('learning_rate', 0.001)
        hidden_dim = kwargs.get('hidden_dim', 64)
        
        # Set up replay buffer for DQN
        buffer_capacity = kwargs.get('buffer_capacity', 10000)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Initialize algorithm-specific models and parameters
        if algorithm == "q_learning":
            # For discrete state and action spaces
            if isinstance(self.env.observation_space, gym.spaces.Discrete) and \
               isinstance(self.env.action_space, gym.spaces.Discrete):
                self.q_table = np.zeros((self.state_dim, self.action_dim))
            else:
                return "Q-learning requires discrete state and action spaces."
        
        elif algorithm == "dqn":
            # Initialize Q-network
            self.model = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
            self.target_model = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
            self.target_model.load_state_dict(self.model.state_dict())
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
        elif algorithm == "policy_gradient":
            # Initialize Policy network
            self.model = PolicyNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
        else:
            return f"Reinforcement learning algorithm '{algorithm}' not supported."
        
        # Track training progress
        rewards = []
        avg_rewards = []
        
        # Training loop
        for episode in range(episodes):
            state = self.env.reset()
            if isinstance(state, tuple):  # Handle newer Gym versions that return (state, info)
                state = state[0]
                
            episode_reward = 0
            done = False
            
            # For policy gradient
            if algorithm == "policy_gradient":
                states = []
                actions = []
                rewards_list = []
            
            for step in range(max_steps):
                # Select action
                if algorithm == "q_learning":
                    action = self._select_action_q_learning(state)
                elif algorithm == "dqn":
                    action = self._select_action_dqn(state)
                elif algorithm == "policy_gradient":
                    action = self._select_action_policy_gradient(state)
                    states.append(state)
                    actions.append(action)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                if isinstance(info, dict) and 'terminal_observation' in info:  # New Gym API
                    next_state, done = info['terminal_observation'], info.get('terminated', done)
                
                episode_reward += reward
                
                # Store experience
                if algorithm == "q_learning":
                    self._update_q_table(state, action, reward, next_state, done)
                elif algorithm == "dqn":
                    self.replay_buffer.push(state, action, reward, next_state, done)
                    
                    # Train from replay buffer when enough samples
                    if len(self.replay_buffer) > batch_size:
                        self._train_dqn(batch_size)
                        
                    # Update target network
                    if episode % update_target_freq == 0:
                        self.target_model.load_state_dict(self.model.state_dict())
                elif algorithm == "policy_gradient":
                    rewards_list.append(reward)
                
                # Move to next state
                state = next_state
                
                if done:
                    break
            
            # Update policy for policy gradient
            if algorithm == "policy_gradient" and len(rewards_list) > 0:
                self._train_policy_gradient(states, actions, rewards_list)
            
            # Decay exploration rate
            self.exploration_rate = max(self.exploration_min, 
                                       self.exploration_rate * self.exploration_decay)
            
            # Record rewards
            rewards.append(episode_reward)
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            avg_rewards.append(avg_reward)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}, Exploration: {self.exploration_rate:.4f}")
                
            # Early stopping if environment is solved
            if avg_reward >= max_steps-1 and len(rewards) >= 100:
                print(f"Environment solved in {episode+1} episodes!")
                break
        
        return {
            "algorithm": algorithm,
            "env_name": env_name,
            "episodes": episode + 1,
            "rewards": rewards,
            "avg_rewards": avg_rewards,
            "final_exploration_rate": self.exploration_rate
        }
    
    def evaluate_agent(self, env_name=None, episodes=10, max_steps=1000, render=False, **kwargs):
        """
        Evaluate a trained reinforcement learning agent
        
        Args:
            env_name (str): Name of the Gym environment (if different from training)
            episodes (int): Number of episodes to evaluate
            max_steps (int): Maximum steps per episode
            render (bool): Whether to render the environment
            
        Returns:
            dict: Evaluation results
        """
        if (self.model is None and self.q_table is None) or self.env is None:
            return "No trained agent found. Train an agent first."
            
        # Set up new environment if specified
        if env_name and env_name != self.env.spec.id:
            env_setup = self.setup_environment(env_name, **kwargs)
            if isinstance(env_setup, str):  # Error message
                return env_setup
        
        # Track evaluation progress
        rewards = []
        
        # Evaluation loop
        for episode in range(episodes):
            state = self.env.reset()
            if isinstance(state, tuple):  # Handle newer Gym versions that return (state, info)
                state = state[0]
                
            episode_reward = 0
            done = False
            
            for step in range(max_steps):
                # Select action (no exploration)
                if self.algorithm == "q_learning":
                    action = np.argmax(self.q_table[state])
                elif self.algorithm == "dqn":
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        q_values = self.model(state_tensor)
                    action = q_values.argmax().item()
                elif self.algorithm == "policy_gradient":
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        probs = self.model(state_tensor)
                    action = probs.argmax().item()
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                if isinstance(info, dict) and 'terminal_observation' in info:  # New Gym API
                    next_state, done = info['terminal_observation'], info.get('terminated', done)
                
                episode_reward += reward
                state = next_state
                
                if render:
                    self.env.render()
                
                if done:
                    break
            
            rewards.append(episode_reward)
            print(f"Evaluation Episode {episode+1}/{episodes}, Reward: {episode_reward:.2f}")
        
        # Close environment rendering
        if render:
            self.env.close()
            
        return {
            "algorithm": self.algorithm,
            "env_name": self.env.spec.id,
            "episodes": episodes,
            "rewards": rewards,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards)
        }
    
    def test_agent(self, state, **kwargs):
        """
        Test the agent on a single state
        
        Args:
            state: Environment state
            
        Returns:
            dict: Action and confidence information
        """
        if (self.model is None and self.q_table is None):
            return "No trained agent found. Train an agent first."
        
        # Preprocess state
        if isinstance(state, list):
            state = np.array(state)
        
        # Select action
        if self.algorithm == "q_learning":
            state_idx = state if isinstance(state, int) else state.argmax()
            q_values = self.q_table[state_idx]
            action = np.argmax(q_values)
            confidence = q_values[action] / (np.sum(np.abs(q_values)) + 1e-6)  # Normalized confidence
            
            return {
                "action": int(action),
                "q_values": q_values.tolist(),
                "confidence": float(confidence)
            }
            
        elif self.algorithm == "dqn":
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            q_values_np = q_values.cpu().numpy()[0]
            action = q_values.argmax().item()
            confidence = np.exp(q_values_np[action]) / np.sum(np.exp(q_values_np))  # Softmax confidence
            
            return {
                "action": int(action),
                "q_values": q_values_np.tolist(),
                "confidence": float(confidence)
            }
            
        elif self.algorithm == "policy_gradient":
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = self.model(state_tensor)
            probs_np = probs.cpu().numpy()[0]
            action = probs.argmax().item()
            
            return {
                "action": int(action),
                "probabilities": probs_np.tolist(),
                "confidence": float(probs_np[action])
            }
        
        return "Algorithm not recognized."
    
    def save_agent(self, path):
        """Save agent to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Create state dict
            state_dict = {
                "algorithm": self.algorithm,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "exploration_rate": self.exploration_rate,
                "gamma": self.gamma
            }
            
            # Save algorithm-specific data
            if self.algorithm == "q_learning":
                state_dict["q_table"] = self.q_table
                np.save(f"{path}_q_table.npy", self.q_table)
            else:
                # For DQN and policy gradient, save model
                state_dict["model_state_dict"] = self.model.state_dict()
                if self.optimizer:
                    state_dict["optimizer_state_dict"] = self.optimizer.state_dict()
                
                # Save target model for DQN
                if self.algorithm == "dqn" and self.target_model:
                    state_dict["target_model_state_dict"] = self.target_model.state_dict()
                
                torch.save(state_dict, path)
            
            return f"Agent saved to {path}"
        except Exception as e:
            return f"Error saving agent: {str(e)}"
    
    def load_agent(self, path, env_name=None, **kwargs):
        """Load agent from disk"""
        try:
            # Set up environment if specified
            if env_name:
                env_setup = self.setup_environment(env_name, **kwargs)
                if isinstance(env_setup, str):  # Error message
                    return env_setup
            
            # For Q-learning, load Q-table
            if path.endswith("_q_table.npy"):
                self.q_table = np.load(path)
                self.algorithm = "q_learning"
                return "Q-table loaded successfully."
            
            # For DQN and policy gradient, load model
            checkpoint = torch.load(path, map_location=self.device)
            
            # Set parameters
            self.algorithm = checkpoint.get("algorithm")
            self.state_dim = checkpoint.get("state_dim", self.state_dim)
            self.action_dim = checkpoint.get("action_dim", self.action_dim)
            self.exploration_rate = checkpoint.get("exploration_rate", 0.01)
            self.gamma = checkpoint.get("gamma", 0.99)
            
            # Initialize model based on algorithm
            hidden_dim = kwargs.get("hidden_dim", 64)
            
            if self.algorithm == "dqn":
                self.model = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
                self.target_model = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
                
                # Load model weights
                self.model.load_state_dict(checkpoint["model_state_dict"])
                
                # Load target model weights if available
                if "target_model_state_dict" in checkpoint:
                    self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
                else:
                    self.target_model.load_state_dict(self.model.state_dict())
                
            elif self.algorithm == "policy_gradient":
                self.model = PolicyNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
                
                # Load model weights
                self.model.load_state_dict(checkpoint["model_state_dict"])
            
            # Load optimizer if available
            if "optimizer_state_dict" in checkpoint and self.model:
                learning_rate = kwargs.get("learning_rate", 0.001)
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            return f"Agent loaded successfully with algorithm: {self.algorithm}"
        except Exception as e:
            return f"Error loading agent: {str(e)}"
    
    def visualize_training(self, save_path=None, figsize=(10, 6), **kwargs):
        """
        Visualize the training progress
        
        Args:
            save_path (str): Path to save the visualization
            figsize (tuple): Figure size
            
        Returns:
            Path to the saved visualization or None
        """
        if not hasattr(self, 'train_rewards') or not self.train_rewards:
            return "No training data available for visualization."
            
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot episode rewards
        plt.plot(self.train_rewards, alpha=0.7, label="Episode Reward")
        
        # Plot moving average
        window_size = min(100, len(self.train_rewards))
        moving_avg = np.convolve(self.train_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(self.train_rewards)), moving_avg, 'r', label=f"{window_size}-Episode Moving Average")
        
        plt.title(f"Training Progress - {self.algorithm} on {self.env.spec.id}", fontsize=15)
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return f"Visualization saved to {save_path}"
            except Exception as e:
                plt.close()
                return f"Error saving visualization: {str(e)}"
        
        # If not saving, show the plot
        plt.show()
        return "Visualization displayed"
    
    def _select_action_q_learning(self, state):
        """Select action for Q-learning"""
        # Exploration-exploitation trade-off
        if np.random.random() < self.exploration_rate:
            return self.env.action_space.sample()  # Explore: random action
        
        # Exploit: best action based on Q-table
        return np.argmax(self.q_table[state])
    
    def _select_action_dqn(self, state):
        """Select action for DQN"""
        # Exploration-exploitation trade-off
        if np.random.random() < self.exploration_rate:
            return self.env.action_space.sample()  # Explore: random action
        
        # Exploit: best action based on Q-network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()
    
    def _select_action_policy_gradient(self, state):
        """Select action for Policy Gradient"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.model(state_tensor).cpu().detach().numpy()[0]
        
        # Sometimes, use greedy action for evaluation
        if np.random.random() < self.exploration_rate:
            return np.random.choice(self.action_dim, p=probs)
        else:
            return np.argmax(probs)
    
    def _update_q_table(self, state, action, reward, next_state, done):
        """Update Q-table for Q-learning"""
        # Get the maximum Q-value for the next state
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Update Q-value for the current state-action pair
        learning_rate = 0.1  # Fixed learning rate for simplicity
        self.q_table[state, action] = (1 - learning_rate) * self.q_table[state, action] + learning_rate * q_target
    
    def _train_dqn(self, batch_size):
        """Train the DQN network"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q-values
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get next Q-values with target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        
        # Calculate target Q-values
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Calculate loss
        loss = F.mse_loss(q_values, q_targets)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _train_policy_gradient(self, states, actions, rewards):
        """Train the policy network with policy gradients"""
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        
        # Calculate discounted rewards
        discounted_rewards = self._calculate_discounted_rewards(rewards)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        # Get action probabilities
        probs = self.model(states)
        
        # Calculate loss: -log(p_a) * R
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1))
        loss = -(log_probs * discounted_rewards).mean()
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _calculate_discounted_rewards(self, rewards):
        """Calculate discounted rewards for policy gradient"""
        discounted_rewards = []
        cumulative_reward = 0
        
        # Calculate discounted rewards in reverse order
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
            
        return discounted_rewards
