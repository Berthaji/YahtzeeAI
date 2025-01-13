from collections import namedtuple, deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from api import YahtzeeGame, ACTION_INDEX_LIMIT

MODEL_NAME = 'yahtzee_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# In ordine, le classi QNet, Replay Buffer e DQN.
# Implementata una funzione fetch_actions per impedire all'agente di scegliere azioni non valide.
# Implementazione funzione play per l'esecuzione di N partite alla riga 197.
# Entry point per l'esecuzione alla riga 246.


class QNet(nn.Module):
    """
        Implementa una rete neurale completamente connessa progettata per apprendere la funzione Q.

        La rete è composta dai seguenti livelli:
            - Un livello di input che accetta il vettore degli stati.
            - Due livelli nascosti con attivazione ReLU.
            - Un livello di output che restituisce i Q-values per tutte le azioni.
    """
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




class ReplayBuffer:
    """
        Buffer di esperienze per il training.
        Implementa una struttura dati per memorizzare e gestire le esperienze raccolte durante il training.

        Ogni esperienza memorizzata è rappresentata da un namedtuple "Experience" con i seguenti campi:
            - state: Lo stato corrente.
            - action: L'azione eseguita.
            - reward: La ricompensa ricevuta.
            - next_state: Lo stato successivo risultante dall'azione.
            - done: Flag che indica se l'episodio è terminato.
    """
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
    


class DQN:
    """
        Classe principale per l'implementazione dell'agente DQN.

        La classe utilizza due reti neurali:
            - net_eval: La rete principale che apprende la funzione Q.
            - net_target: La rete obiettivo, utilizzata per calcolare i target durante l'addestramento.

        Metodi principali:
            - getAction: Seleziona un'azione in base alla politica epsilon-greedy.
            - save2Memory: Memorizza le esperienze raccolte nel replay buffer.
            - learn: Aggiorna i parametri della rete principale.
            - targetUpdate: Aggiorna i parametri della rete obiettivo.

        Args:
            n_states: Numero di stati.
            n_actions: Numero di azioni possibili.
            batch_size: Dimensione del batch usato durante l'addestramento.
            learning_rate: Tasso di apprendimento.
            learn_step: Numero di passi tra due aggiornamenti della rete.
            gamma: Fattore di sconto per i reward.
            mem_size: Dimensione massima del replay buffer.
            tau: Fattore per l'aggiornamento della rete obiettivo.
            device: Dispositivo di esecuzione ("cpu" su Apple Silicon)
    """
    def __init__(self, n_states, n_actions, batch_size=64, learning_rate=1e-4, learn_step=5, gamma=0.9, mem_size=int(1e5), tau=1e-3, device=device):
        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_step = learn_step
        self.tau = tau
        self.device = device

        self.net_eval = QNet(n_states, n_actions).to(device)
        self.net_target = QNet(n_states, n_actions).to(device)
        self.optimizer = optim.Adam(self.net_eval.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.memory = ReplayBuffer(n_actions, mem_size, batch_size)
        self.counter = 0

    def getAction(self, state, epsilon, actions):
        if random.random() < epsilon:
        # Scelta casuale tra le azioni (esplorazione)
            return random.choice(actions)
        else:
            # Calcolo Q-values (sfruttamento)
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            self.net_eval.eval()
            with torch.no_grad():
                q_values = self.net_eval(state)[0]  # tensore dei Q-values
            self.net_eval.train()

            # Filtra per le azioni valide
            valid_q_values = [(action, q_values[action]) for action in actions] # coppie (azione – Q-value)
            best_action, _ = max(valid_q_values, key=lambda x: x[1]) # azione con Q-value massimo

            return best_action

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



def fetch_actions(game):
    """
        Restituisce le azioni che l'agente può compiere, in base allo stato corrente del gioco.

        Args:
            game: Oggetto che rappresenta il gioco.

        Returns:
            list or None: Se i tiri sono finiti, può restituire:
                - Un elenco delle righe non ancora completate.
                - None, se non ci sono righe disponibili.
            Se ci sono tiri rimasti, ritorna un intervallo per le azioni di rilancio.
    """
    if game.rollLeft() == 0:
        available_rows = []
        for i, row in enumerate(game.scorecard.keys()):
            if game.scorecard[row] is None:
                available_rows.append(i + ACTION_INDEX_LIMIT)
        
        # Se non ci sono righe valide, restituisce None
        return available_rows if available_rows else None
    
    # Intervallo per le azioni di rilancio
    return [i for i in range(ACTION_INDEX_LIMIT)]




def play(agent, game, n_games):
    """
        Esegue N partite di Yahtzee utilizzando il modello addestrato, 
        stampando il punteggio di ogni partita e la media finale.

        Args:
            agent: L'agente utilizzato durante il gioco.
            game: L'oggetto che rappresenta il gioco.
            n_games: Il numero di partite da giocare.

        Note:
            L'agente utilizza la politica appresa senza esplorazione (solo sfruttamento).
            Se non ci sono azioni valide, la partita termina.
    """
    score_list = [] 

    for game_index in range(n_games):
        game.newGame()  # Nuova partita
        state = np.array(game.getDiceValues() + [game.rollLeft()] + game.completed_rows, dtype=np.float32)
        total_score = 0

        while not game.hasFinished():

            # azioni valide 
            valid_actions = fetch_actions(game)

            if not valid_actions:
                print("Partita terminata. Nessuna azione valida disponibile.")
                break

            # Scelta dell'azione
            action = agent.getAction(state, epsilon=0.0, actions=valid_actions)
            game.chooseAction(action)
            reward = float(game.getLastReward())
            next_state = np.array(game.getDiceValues() + [game.rollLeft()] + game.completed_rows, dtype=np.float32)

            state = next_state
            total_score += reward

        # Store del punteggio della partita
        score_list.append(total_score)
        print(f"Partita {game_index + 1}/{n_games} - Punteggio: {total_score}")

    # Media dei punteggi
    average_score = np.mean(score_list)
    print(f"\nPunteggio medio su {n_games} partite: {average_score}")



if __name__ == "__main__":
    n_states = 19 
    n_actions = 45  
    n_games = 500

    # Creazione dell'agente e caricamento del modello
    agent = DQN(
            n_states=n_states, 
            n_actions=n_actions,
            batch_size=128, 
            learning_rate=1e-4,
            gamma=0.9, 
            learn_step=5, 
            mem_size=int(1e6), 
            tau=1e-3
    )
    agent.net_eval.load_state_dict(torch.load(MODEL_NAME, map_location=device, weights_only=True))
    agent.net_eval.eval()

    game = YahtzeeGame()
    print(f"Esecuzione su: {device}")
    play(agent, game, n_games)
