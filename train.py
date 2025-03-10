import numpy as np
import torch
from tqdm import trange
from api import ACTION_INDEX_LIMIT


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
        valid_rows = []
        for i, row in enumerate(game.scorecard.keys()):
            if game.scorecard[row] is None:
                valid_rows.append(i + ACTION_INDEX_LIMIT)
        
        # Se non ci sono righe valide, restituisce None
        return valid_rows if valid_rows else None
    
    # Intervallo per le azioni di rilancio
    return [i for i in range(ACTION_INDEX_LIMIT)]




def train(
        agent, 
        game, 
        n_episodes, 
        max_steps, 
        target_score,
        epsilon_start, 
        epsilon_end, 
        epsilon_decay, 
        save_model): 
    """
        Funzione per l'addestramento di un agente DQN.

        L'agente apprende a selezionare azioni in base allo stato del gioco e al proprio comportamento nel tempo.
        La funzione gestisce il ciclo di episodi, il calcolo dei punteggi e l'aggiornamento della policy 
        utilizzata (epsilon-greedy). Supporta l'early stopping quando viene raggiunto il target di
        punteggio medio (prima del raggiungimento degli episodi massimi) e il salvataggio del modello.

        Args:
            agent: L'agente DQN da addestrare.
            game: Il gioco da utilizzare per l'addestramento.
            n_episodes: Numero massimo di episodi da eseguire.
            max_steps: Numero massimo di passi per episodio.
            target_score: Punteggio target per l'early stopping.
            epsilon_start: Valore iniziale di epsilon (per esplorazione).
            epsilon_end: Valore finale di epsilon.
            epsilon_decay: Fattore di decadimento di epsilon per episodio.
            save_model (bool): Flag che determina se il modello deve essere salvato alla fine dell'addestramento.

        Returns:
            list: Una lista contenente i punteggi ottenuti durante l'addestramento.
    """
    
    score_history = []
    epsilon = epsilon_start

    # Progression bar
    bar_format = '{l_bar}{bar:10}| {n:4}/{total_fmt}'\
                 ' [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]'
    progression_bar = trange(n_episodes, unit="ep", bar_format=bar_format, ascii=True)

    for episode_index in progression_bar:
        game.newGame()
        state = np.array(game.getDiceValues() + [game.rollLeft()] + game.completed_rows, dtype=np.float32)
        episode_score = 0

        for _ in range(max_steps):

            actions = fetch_actions(game)
            if not actions:
                print("Nessuna azione valida disponibile.")
                break

            action = agent.getAction(state, epsilon, actions)  # Scelta dell'azione 
            game.chooseAction(action)
            reward = float(game.getLastReward())
            next_state = np.array(game.getDiceValues() + [game.rollLeft()] + game.completed_rows, dtype=np.float32)

            agent.save2Memory(state, action, reward, next_state, game.hasFinished())
            state = next_state
            episode_score += reward

            if game.hasFinished():
                break

        score_history.append(episode_score)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)  # Decadimento di epsilon

        # Aggiornamento progression bar
        average_score = np.mean(score_history[-100:])
        progression_bar.set_postfix_str(
            f"Score: {episode_score: 7.2f}, 100 avg: {average_score: 7.2f}"
        )
        progression_bar.update(0)

        # Early stopping
        if len(score_history) >= 100 and np.mean(score_history[-100:]) >= target_score:
            print("\nAddestramento completato - obiettivo raggiunto.")
            progression_bar.close()
            break

    if episode_index == n_episodes - 1:
        print("\nAddestramento completato - max episodi raggiunto.")

    # Salvataggio del modello 
    if save_model:
        torch.save(agent.net_eval.state_dict(), 'yahtzee_model.pth')

    return score_history