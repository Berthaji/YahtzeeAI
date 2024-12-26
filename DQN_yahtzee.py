from collections import namedtuple, deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from yahtzee_api import YahtzeeGame, ACTION_INDEX_LIMIT

device = torch.device("cpu")    # pytorch non supporta l'accelerazione GPU su apple silicon
print(f"Using device: {device}")

"""
    @class QNet
    @brief Rete neurale per il Deep Q-Learning.
    La classe QNet implementa una rete neurale completamente connessa progettata per 
    apprendere la funzione Q. Questa funzione stima i valori Q per ciascuna azione 
    possibile dato uno stato di input.

    @note La rete è composta da tre livelli:
        - Un livello di input che accetta il vettore stato (n_state_vars).
        - Due livelli nascosti con attivazione ReLU per catturare le relazioni non lineari.
        - Un livello di output che restituisce i valori Q per tutte le azioni (n_actions).
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
    @brief Buffer di esperienza per l'addestramento del Deep Q-Learning.

    La classe ReplayBuffer implementa una struttura dati per memorizzare e gestire 
    le esperienze raccolte durante l'apprendimento.

    @details
    Ogni esperienza memorizzata è rappresentata da un namedtuple "Experience":
    - @c state: Lo stato corrente.
    - @c action: L'azione eseguita.
    - @c reward: La ricompensa ricevuta.
    - @c next_state: Lo stato successivo risultante dall'azione.
    - @c done: booleano che indica se l'episodio è terminato.
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
    @brief Classe principale per l'implementazione dell'agente Deep Q-Network.
    La classe DQN implementa l'agente di reinforcement learning basato sull'algoritmo 
    Deep Q-Learning.
    - @c net_eval: La rete di valutazione, che apprende la funzione Q.
    - @c net_target: La rete obiettivo, utilizzata per calcolare i target durante l'addestramento.

    @note
    L'agente utilizza un buffer di replay per memorizzare le esperienze e 
    campionarle durante l'addestramento. La classe offre metodi per selezionare
    azioni (@c getAction), memorizzare esperienze (@c save2Memory), e aggiornare 
    i parametri delle reti (@c learn e @c targetUpdate).

    @param n_states Numero di variabili di stato (dimensione dello stato di input).
    @param n_actions Numero di azioni possibili (dimensione dell'output della rete).
    @param batch_size Dimensione del batch usato durante l'addestramento.
    @param learning_rate Tasso di apprendimento.
    @param learn_step Numero di passi tra due aggiornamenti della rete.
    @param gamma Fattore di sconto per i futuri reward.
    @param mem_size Dimensione massima del replay buffer.
    @param tau Fattore per l'aggiornamento della rete obiettivo.
"""
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
