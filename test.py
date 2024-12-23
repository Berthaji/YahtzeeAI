from yahtzee_api import YahtzeeGame  # Assicurati di importare correttamente l'API
from DQN_yahtzee import train, DQN

n_states = 5 + 1 + 13  # 5 dadi, 1 tiri rimasti, 13 righe completate
n_actions = 18  

# Crea l'agente DQN
agent = DQN(n_states=n_states, 
            n_actions=n_actions,
            batch_size=64, 
            learning_rate=1e-4,
            gamma=0.4, 
            learn_step=5, 
            mem_size=int(1e5), 
            tau=1e-3)

# Parametri di training
n_episodes = 60000  # Numero di episodi per l'addestramento
max_steps = 52  # Numero massimo di passi per episodio
eps_start = 1.0  # Epsilon iniziale
eps_end = 0.01  # Epsilon finale
eps_decay = 0.995  # Decadimento di epsilon
target_score = 130  # Punteggio target per terminare l'addestramento
game = YahtzeeGame()

# Avvia il training
train(game, agent, n_episodes, max_steps, eps_start, eps_end, eps_decay, target_score, do_store_checkpoint=True)