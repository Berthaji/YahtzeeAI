from yahtzee_api import YahtzeeGame 
from DQN_yahtzee import DQN
from train import train

n_states = 19  # 5 dadi, 1 tiri rimasti, 13 righe completate
n_actions = 18  # 15 azioni di rilancio, 3 azioni di selezione riga

# agente DQN
agent = DQN(n_states=n_states, 
            n_actions=n_actions,
            batch_size=128, 
            learning_rate=5e-4,
            gamma=0.9, 
            learn_step=5, 
            mem_size=int(1e5), 
            tau=1e-3)

# Parametri di training
n_episodes = 60000  # Numero di episodi per l'addestramento
max_steps = 52  # Numero massimo di passi per episodio
eps_start = 1.0  # Epsilon iniziale
eps_end = 0.01  # Epsilon finale
eps_decay = 0.995  # Decadimento di epsilon
target_score = 180  # Punteggio target per terminare l'addestramento (punteggio medio su ultimi 100 episodi)
game = YahtzeeGame()


scores = train(
    game, agent, n_episodes, max_steps, eps_start, eps_end, 
    eps_decay, target_score, store_checkpoint=True)