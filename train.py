import numpy as np
import torch
from tqdm import trange

CHECKPOINT_NAME = 'checkpoint_yahtzee.pth'
RECENT_EPISODES = 100  # Numero di episodi di riferimento per l'average score
MIN_EPISODES_FOR_STOP = 100 


"""
    @brief Funzione di addestramento per un agente DQN nel gioco Yahtzee.
    Gestisce l'intero ciclo di addestramento dell'agente. L'addestramento termina quando 
    l'agente raggiunge il punteggio target medio definito o quando il numero massimo 
    di episodi è stato completato.

    @param game Oggetto che rappresenta il gioco
    @param agent L'agente DQN da addestrare, che prende decisioni basate sugli stati del gioco.
    @param n_episodes Numero massimo di episodi per l'addestramento.
    @param max_steps Numero massimo di passi per ogni episodio.
    @param eps_start Valore iniziale di epsilon (probabilità di esplorazione).
    @param eps_end Valore finale di epsilon.
    @param eps_decay Fattore di decadimento di epsilon ad ogni episodio.
    @param target_score Punteggio medio target.
    @param store_checkpoint booleano che determina se il modello dell'agente deve essere 
    salvato al termine dell'addestramento.

    @return Una lista dei punteggi ottenuti in ciascun episodio durante l'addestramento.

    @note La funzione include una barra di progresso per monitorare l'andamento dell'addestramento.
"""
def train(
        game, agent, n_episodes, max_steps,
        eps_start, eps_end, eps_decay,
        target_score, store_checkpoint
):
    score_history = []
    epsilon = eps_start

    # Progression bar per tenere traccia degli sviluppi dell'addestramento
    bar_format = '{l_bar}{bar:10}| {n:4}/{total_fmt}'\
                 ' [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]'
    progression_bar = trange(n_episodes, unit="ep", bar_format=bar_format, ascii=True)

    for episode_index in progression_bar:
        game.newGame()  # reset del gioco per un nuovo episodio
        state = np.array(game.getDiceValues() + [game.rollLeft()] + game.completed_rows, dtype=np.float32)
        score = 0

        for current_step in range(max_steps):
            # seleziona un'azione seguendo la logica epsilon-greedy
            action = agent.getAction(state, epsilon)

            if game.rollLeft() == 0:  # Se i rolls sono finiti si può solo selezionare una riga
                
                available_rows = [i for i, completed in enumerate(game.completed_rows) if not completed]
                
                if not available_rows:
                    raise ValueError("Nessuna riga libera, il gioco dovrebbe essere terminato.")
                
                # Scegli la prima riga disponibile
                row_index = available_rows[0]
                row_name = list(game.scorecard.keys())[row_index]  # nome della riga
                
                ## utilizzo di chooseRow() per il completamento della riga 
                # per evitare un roll aggiuntivo previsto nella funzione chooseAction e 
                # quindi un'eccezione, dato che i roll sono finiti
                game.chooseRow(row_name)  
            else:
                game.chooseAction(action)

            reward = game.getLastReward() 
            score += reward

            # salva esperienza
            next_state = np.array(game.getDiceValues() + [game.rollLeft()] + game.completed_rows, dtype=np.float32)
            agent.save2Memory(state, action, reward, next_state, game.hasFinished())

            state = next_state 

            if game.hasFinished(): 
                break

        # Traccia degli score e riduzione di epsilon
        score_history.append(score)
        average_score = np.mean(score_history[-RECENT_EPISODES:])
        epsilon = max(eps_end, epsilon * eps_decay)  # Decay epsilon

        # Update della progression bar
        progression_bar.set_postfix_str(
            f"Score: {score: 7.2f}, 100 score avg: {average_score: 7.2f}"
        )
        progression_bar.update(0)

        # Stopping condition nel caso in cui il target sia raggiunto prima del limite massimo di episodi
        if len(score_history) >= MIN_EPISODES_FOR_STOP and average_score >= target_score:
            print("\nObiettivo raggiunto.")
            break

    if (episode_index + 1) < n_episodes:
        print("\nTraining completato - obiettivo raggiunto")
    else:
        print("\nTraining completato - max episodi raggiunto.")

    # Salvataggio del modello
    if store_checkpoint:
        torch.save(agent.net_eval.state_dict(), CHECKPOINT_NAME)

    return score_history