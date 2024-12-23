# Standard
from collections import namedtuple, deque
import imageio
from IPython.display import Image as IPImage
import os
from PIL import Image, ImageDraw, ImageFont
import random
from time import sleep

# Third-party
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

device = torch.device("cpu")
print(f"Using device: {device}") # l'accelerazione GPU con pytorch non funziona su apple silicon

class QNet(nn.Module):
    # Policy Network
    def __init__(self, n_state_vars, n_actions, dim_hidden=64):
        super(QNet, self).__init__()

        # Define a feedforward neural network with hidden layers, ReLU
        #  activations, and an output layer that maps to the number of actions
        self.fc = nn.Sequential(
            nn.Linear(n_state_vars, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, n_actions)
        )

    def forward(self, x):
        # Passes the input through the network layers to output Q-values
        return self.fc(x)
    

class ReplayBuffer:
    def __init__(self, n_actions, memory_size, batch_size):
        # Initialize actions, batch and experience template
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # Initialize the memory
        self.memory = deque(maxlen=memory_size)

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        # Store experience in memory
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        # Sample a batch of experiences
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert to tensors for training
        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(device)

        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).long().to(device)

        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(device)

        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(device)

        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(device)

        # Return the tuple with all tensors
        return (states, actions, rewards, next_states, dones)
    

class DQN:
    def __init__(
        self, n_states, n_actions, batch_size=64, learning_rate=1e-4,
        learn_step=5, gamma=0.99, mem_size=int(1e5), tau=1e-3
    ):
        # Core parameters for learning and updating the Q-network
        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma  # Discount factor for future rewards
        self.learn_step = learn_step  # Frequency of learning steps
        self.tau = tau  # Rate for soft updating the target network

        # Initialize the policy network (net_eval) and target network (net_target)
        self.net_eval = QNet(n_states, n_actions).to(device)
        self.net_target = QNet(n_states, n_actions).to(device)
        self.optimizer = optim.Adam(self.net_eval.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Initialize memory for experience replay
        self.memory = ReplayBuffer(n_actions, mem_size, batch_size)
        self.counter = 0  # Tracks learning steps for periodic updates

    def getAction(self, state, epsilon):
        # Select action using an epsilon-greedy strategy to balance exploration
        #  and exploitation
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.net_eval.eval()  # Set network to evaluation mode
        with torch.no_grad():
            action_values = self.net_eval(state)
        self.net_eval.train()  # Return to training mode

        # Choose random action with probability epsilon, otherwise choose best
        #  predicted action
        if random.random() < epsilon:
            action = random.choice(np.arange(self.n_actions))
        else:
            action = np.argmax(action_values.cpu().data.numpy())
        return action

    def save2Memory(self, state, action, reward, next_state, done):
        # Save experience to memory and, if ready, sample from memory and
        #  update the network
        self.memory.add(state, action, reward, next_state, done)
        self.counter += 1  # Increment step counter

        # Perform learning every 'learn_step' steps if enough experiences are
        #  in memory
        if (
            self.counter % self.learn_step == 0 and
            len(self.memory) >= self.batch_size
        ):
            experiences = self.memory.sample()
            self.learn(experiences)
    
    def learn(self, experiences):
        # Perform a learning step by minimizing the difference between
        #  predicted and target Q-values
        states, actions, rewards, next_states, dones = experiences

        # Compute target Q-values from net_target for stability in training
        q_target = self.net_target(next_states).detach().max(1)[0].unsqueeze(1)
        y_j = rewards + (self.gamma * q_target * (1 - dones))
            # Bellman equation for target Q-value
        q_eval = self.net_eval(states).gather(1, actions)
            # Q-value predictions from policy network

        # Compute loss and backpropagate to update net_eval
        loss = self.criterion(q_eval, y_j)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network with soft update for smooth learning
        self.targetUpdate()

    def targetUpdate(self):
        # Soft update to gradually shift target network parameters toward
        #  policy network parameters
        params = zip(self.net_eval.parameters(), self.net_target.parameters())
        for eval_param, target_param in params:
            target_param.data.copy_(
                self.tau * eval_param.data + (1.0 - self.tau) * target_param.data
            )

CHECKPOINT_NAME = 'checkpoint.pth'
RECENT_EPISODES = 100  # Number of episodes for average score in early stopping
MIN_EPISODES_FOR_STOP = 100  # Ensures enough episodes before evaluating target

def train(
        env, agent, n_episodes, max_steps,
        eps_start, eps_end, eps_decay,
        target_score, do_store_checkpoint
):
    # Initialize score history and epsilon (exploration rate)
    score_hist = []
    epsilon = eps_start

    # Progress bar format for tracking training progress
    bar_format = '{l_bar}{bar:10}| {n:4}/{total_fmt}'\
                 ' [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]'
    pbar = trange(n_episodes, unit="ep", bar_format=bar_format, ascii=True)

    for idx_epi in pbar:
        # Reset the environment for a new episode
        state, _ = env.reset()
        score = 0

        for idx_step in range(max_steps):
            # Select an action based on the current policy (epsilon-greedy)
            action = agent.getAction(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)

            # Store experience in memory and update the agent
            agent.save2Memory(state, action, reward, next_state, done)
            state = next_state  # Move to the next state
            score += reward

            # Check if the episode is finished
            if done:
                break

        # Track scores and decay epsilon for less exploration over time
        score_hist.append(score)
        score_avg = np.mean(score_hist[-RECENT_EPISODES:])
        epsilon = max(eps_end, epsilon * eps_decay)  # Decay epsilon

        # Update the progress bar with the current score and average
        pbar.set_postfix_str(
            f"Score: {score: 7.2f}, 100 score avg: {score_avg: 7.2f}"
        )
        pbar.update(0)

        # Early stopping condition if target score is achieved
        if len(score_hist) >= MIN_EPISODES_FOR_STOP and score_avg >= target_score:
            print("\nTarget Reached!")
            break

    # Print completion message based on early stopping or max episodes
    if (idx_epi + 1) < n_episodes:
        print("\nTraining complete - target reached!")
    else:
        print("\nTraining complete - maximum episodes reached.")

    # Save the trained model if specified
    if do_store_checkpoint:
        torch.save(agent.net_eval.state_dict(), CHECKPOINT_NAME)

    return score_hist