# Standard
from collections import namedtuple, deque
import imageio
from IPython.display import Image as IPImage
import os
from PIL import Image, ImageDraw, ImageFont
import random
from time import sleep

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from yahtzee_api import YahtzeeGame, ACTION_INDEX_LIMIT

device = torch.device("cpu")
print(f"Using device: {device}")

# Classe QNet per la rete neurale
class QNet(nn.Module):
    def __init__(self, n_state_vars, n_actions, dim_hidden=64):
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_state_vars, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, n_actions)
        )

    def forward(self, x):
        return self.fc(x)


# Replay Buffer per esperienze di apprendimento
class ReplayBuffer:
    def __init__(self, n_actions, memory_size, batch_size):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.memory = deque(maxlen=memory_size)

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)
    

# Classe principale per il DQN
class DQN:
    def __init__(self, n_states, n_actions, batch_size=64, learning_rate=0.04, learn_step=5, gamma=0.4, mem_size=int(1e5), tau=1e-3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_step = learn_step
        self.tau = tau

        self.net_eval = QNet(n_states, n_actions).to(device)
        self.net_target = QNet(n_states, n_actions).to(device)
        self.optimizer = optim.Adam(self.net_eval.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.memory = ReplayBuffer(n_actions, mem_size, batch_size)
        self.counter = 0

    def getAction(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.net_eval.eval()
        with torch.no_grad():
            action_values = self.net_eval(state)
        self.net_eval.train()

        if random.random() < epsilon:
            action = random.choice(np.arange(self.n_actions))
        else:
            action = np.argmax(action_values.cpu().data.numpy())
        return action

    def save2Memory(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.counter += 1

        if self.counter % self.learn_step == 0 and len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        q_target = self.net_target(next_states).detach().max(1)[0].unsqueeze(1)
        y_j = rewards + (self.gamma * q_target * (1 - dones))
        q_eval = self.net_eval(states).gather(1, actions)

        loss = self.criterion(q_eval, y_j)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.targetUpdate()

    def targetUpdate(self):
        params = zip(self.net_eval.parameters(), self.net_target.parameters())
        for eval_param, target_param in params:
            target_param.data.copy_(
                self.tau * eval_param.data + (1.0 - self.tau) * target_param.data
            )


# Funzione per allenare l'agente con l'API di Yahtzee
CHECKPOINT_NAME = 'checkpoint.pth'
RECENT_EPISODES = 100  # Number of episodes for average score in early stopping
MIN_EPISODES_FOR_STOP = 100  # Ensures enough episodes before evaluating target

def train(
        game, agent, n_episodes, max_steps,
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
        # Reset the game for a new episode
        game.newGame()  # Start a new game using Yahtzee API's newGame
        state = np.array(game.getDiceValues() + [game.rollLeft()] + game.completed_rows, dtype=np.float32)
        score = 0

        for idx_step in range(max_steps):
            # Select an action based on the current policy (epsilon-greedy)
            action = agent.getAction(state, epsilon)

            # Log the action and number of rolls left for debugging
            # print(f"Episode {idx_epi}, Step {idx_step}: Action {action}, Rolls Left: {game.rollLeft()}")

            if game.rollLeft() == 0:  # Se i rolls sono finiti puoi solo selezionare una riga
                # Trova la prima riga disponibile
                available_rows = [i for i, completed in enumerate(game.completed_rows) if not completed]
                
                if not available_rows:
                    raise ValueError("Nessuna riga libera, il gioco dovrebbe essere terminato.")
                
                # Scegli la prima riga disponibile
                row_index = available_rows[0]
                row_name = list(game.scorecard.keys())[row_index]  # Ottieni il nome della riga
                game.chooseRow(row_name)  # Usa chooseRow per completare la riga
            else:
                game.chooseAction(action)

            # Reward is given by the most recent completed row
            reward = game.getLastReward()  # Get the score of the last completed row
            score += reward

            # Store experience in memory and update the agent
            next_state = np.array(game.getDiceValues() + [game.rollLeft()] + game.completed_rows, dtype=np.float32)
            agent.save2Memory(state, action, reward, next_state, game.hasFinished())

            state = next_state  # Move to the next state

            # Check if the episode is finished (game over or target reached)
            if game.hasFinished():  # Check if the game is finished
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
            print("\nTarget Reached! (Average score)")
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