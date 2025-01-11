from api import YahtzeeGame 
from DQN_yahtzee import DQN
from train import train
import torch

n_states = 19  # 5 dadi, 1 tiri rimasti, 13 righe completate
n_actions = 45  # tutte le azioni possibili
device = torch.device("cpu")



agent = DQN(n_states=n_states, 
            n_actions=n_actions,
            batch_size=128, 
            learning_rate=1e-4,
            gamma=0.9, 
            learn_step=5, 
            mem_size=int(1e6), 
            tau=1e-3,
            device=device)

# Parametri di training
n_episodes = 2000  # Numero massimo di episodi
max_steps = 52  # Numero massimo di passi per episodio
epsilon_start = 1.0  # Epsilon iniziale
epsilon_end = 0.01  # Epsilon finale
epsilon_decay = 0.99  # Decadimento (epsilon-greedy policy)
target_score = 150  # Punteggio target per terminare l'addestramento (punteggio medio su ultimi 100 episodi)
game = YahtzeeGame()


scores = train(
    agent, game, n_episodes, max_steps, target_score, epsilon_start, epsilon_end, 
    epsilon_decay, save_model=True)