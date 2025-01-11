from collections import namedtuple, deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cpu")    # pytorch non supporta l'accelerazione GPU su apple silicon
print(f"Using device: {device}")

"""
    @class QNet
    @brief Rete neurale per il Q-Learning.
    Implementa una rete neurale completamente connessa pensata per 
    apprendere la funzione Q.

    @note È composta dai seguenti livelli:
        - Un livello di input che accetta il vettore degli stati.
        - Due livelli nascosti con attivazione ReLU.
        - Un livello di output che restituisce i Q-values per tutte le azioni.
"""
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



"""
    @class ReplayBuffer
    @brief Buffer di esperienza per il training nel Q-Learning.
    Implementa una struttura dati per memorizzare e gestire 
    le esperienze raccolte durante il training.

    @details
    Ogni esperienza memorizzata è rappresentata da un namedtuple "Experience":
    - @c state: Lo stato corrente.
    - @c action: L'azione eseguita.
    - @c reward: La ricompensa ricevuta.
    - @c next_state: Lo stato successivo risultante dall'azione.
    - @c done: Flag che indica se l'episodio è terminato.
"""
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
    



"""
    @class DQN
    @brief Classe principale per l'implementazione dell'agente di reinforcement learning
    sulla base del concetto di Q-Learning.
    - @c net_eval: Rete principale, che apprende la funzione Q.
    - @c net_target: Rete obiettivo, utilizzata per calcolare i target durante l'addestramento.

    @notem La classe include i seguenti metodi:
    - @c getAction: Selezione di un'azione.
    - @c save2Memory: Memorizzazione delle esperienze.
    - @c learn: Aggiornamento dei parametri della rete.
    - @c targetUpdate: Aggiornamento dei parametri della rete obiettivo.

    @param n_states: Numero di variabili di stato (dimensione dello stato di input).
    @param n_actions: Numero di azioni possibili (dimensione dell'output della rete).
    @param batch_size: Dimensione del batch usato durante l'addestramento.
    @param learning_rate: Tasso di apprendimento.
    @param learn_step: Numero di passi tra due aggiornamenti della rete.
    @param gamma: Fattore di sconto per i futuri reward.
    @param mem_size: Dimensione massima del replay buffer.
    @param tau: Fattore per l'aggiornamento della rete obiettivo.
    @param device: Dispositivo di esecuzione ("cpu" su Apple Silicon).
"""
class DQN:
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
