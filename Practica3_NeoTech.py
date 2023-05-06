import enum
import typing
import urllib
import numpy as np
import collections
import requests

class Action(enum.Enum):
    """Cada una de las posibles figuras."""
    ROCK = '🪨'
    PAPER = '🧻'
    SCISSORS = '✂️'
    LIZARD = '🦎'
    SPOCK = '🖖'

MOVES_AND_REWARDS = {
    (Action.ROCK, Action.ROCK): 0, (Action.ROCK, Action.PAPER): -1,
    (Action.ROCK, Action.SCISSORS): 1, (Action.ROCK, Action.LIZARD): 1,
    (Action.ROCK, Action.SPOCK): -1,
    (Action.PAPER, Action.ROCK): 1, (Action.PAPER, Action.PAPER): 0,
    (Action.PAPER, Action.SCISSORS): -1, (Action.PAPER, Action.LIZARD): -1,
    (Action.PAPER, Action.SPOCK): 1,
    (Action.SCISSORS, Action.ROCK): -1, (Action.SCISSORS, Action.PAPER): 1,
    (Action.SCISSORS, Action.SCISSORS): 0, (Action.SCISSORS, Action.LIZARD): 1,
    (Action.SCISSORS, Action.SPOCK): -1,
    (Action.LIZARD, Action.ROCK): -1, (Action.LIZARD, Action.PAPER): 1,
    (Action.LIZARD, Action.SCISSORS): -1, (Action.LIZARD, Action.LIZARD): 0,
    (Action.LIZARD, Action.SPOCK): 1,
    (Action.SPOCK, Action.ROCK): 1, (Action.SPOCK, Action.PAPER): -1,
    (Action.SPOCK, Action.SCISSORS): 1, (Action.SPOCK, Action.LIZARD): -1,
    (Action.SPOCK, Action.SPOCK): 0,
}

print("\nEjemplo: ")
class Game:
    RENDER_MODE_HUMAN = 'human'
    
    def __init__(self, render_mode=None):
        self.render_mode = render_mode

    def play(self, p1_action, p2_action):
        result = MOVES_AND_REWARDS[(p1_action, p2_action)]
        if self.render_mode == 'human':
            self.render(p1_action, p2_action, result)
        return result
    
    @staticmethod
    def render(p1_action, p2_action, result):
        if result == 0:
            print(f'{p1_action.value} tie!')
        elif result == 1:
            print(f'{p1_action.value} beats {p2_action.value}')
        elif result == -1:
            print(f'{p2_action.value} beats {p1_action.value}')
        else:
            raise ValueError(f'{p1_action}, {p2_action}, {result}')

game = Game(render_mode='human')
game.play(np.random.choice(list(Action)), np.random.choice(list(Action)))
print("\n")

class Transition(typing.NamedTuple):
    """Representa la transición de un estado al siguiente"""
    prev_state: int              # Estado origen de la transición
    next_state: int              # Estado destino de la transición
    action: Action               # Acción que provocó esta transición
    reward: typing.SupportsFloat # Recompensa obtenida


# EJERCICIO 1
class Agent:
    def __init__(self, name: str, q_table: typing.Any = None):
        self.name = name
        if q_table:
            self.q_table = q_table
        else:
            # Inicializa la Q-table a 0
            self.q_table = {b.value: 0.0 for b in Action}
            self.updated_table = self.q_table.copy()

    def decide(self, state: int, epsilon: typing.SupportsFloat = 0) -> Action:
        if np.random.random() < epsilon:
            # Con probabilidad epsilon, elegir una acción random
            return np.random.choice(list(Action))
        if state in self.q_table:
            # Si el estado está en la Q-table, elige el que tenga el valor más alto
            argmax = np.argmax(list(self.q_table[state].values()))
            return list(Action)[argmax]
        else:
            # Si no lo está, elige uno random
            self.q_table[state] = {action.value: 0.0 for action in Action}
            return np.random.choice(list(Action))

    def update(self, t: Transition, alpha=0.1, gamma=0.95):
        if t.prev_state not in self.q_table:
            # Si el prev estado no está, añadirlo
            self.q_table[t.prev_state] = {action.value: 0.0 for action in Action}
        current_q = self.q_table[t.prev_state][t.action.value]
        next_q = max(list(self.q_table[t.next_state].values()))
        td_target = t.reward + gamma * next_q
        td_error = td_target - current_q
        new_q = current_q + alpha * td_error
        self.q_table[t.prev_state][t.action.value] = new_q
        self.updated_table = self.q_table.copy()

        # Printear valores para testear
        #print(f"Q-value update:")
        #print(f"  Prev State: {t.prev_state}, Action: {t.action.value}")
        #print(f"  Current Q-value: {current_q}")
        #print(f"  Next Q-value: {next_q}")
        #print(f"  TD Target: {td_target}")
        #print(f"  TD Error: {td_error}")
        #print(f"  New Q-value: {new_q}")

    def __str__(self) -> str:
        table_str = "Q-Table tras entrenar:\n"
        for state, actions in self.q_table.items():
            table_str += f"State: {state}\n"
            if isinstance(actions, float):
                table_str += f"  {Action(state).name}: {actions}\n"
            else:
                for action, value in actions.items():
                    table_str += f"  {action}: {value}\n"
        return table_str



#ENTRENAMIENTO
dataset_url = 'https://blazaid.github.io/Aprendizaje-profundo/Datasets/rock-paper-scissors-lizard-spock.trn'

player2_actions = []
with urllib.request.urlopen(dataset_url) as f:
    for line in f:
        move = line.decode('utf-8').strip().upper()
        if move:
            player2_actions.append(Action[move])

𝜀 = 1
𝛿𝜀 = 𝜀 / len(player2_actions)

game = Game()
agent = Agent(name='Agent')

state = 0  # El entorno (juego) no tiene estado, así que siempre será el mismo
for p2_action in player2_actions:
    p1_action = agent.decide(state, 𝛆)
    reward = game.play(p1_action, p2_action)

    # Printear las acciones
    #print("\n")
    #print(f"Player 1 Action: {p1_action.value}")
    #print(f"Player 2 Action: {p2_action.value}")
    #print(f"Reward: {reward}")


    # Actualizamos el agente
    agent.update(Transition(
        prev_state=state,
        next_state=state,
        action=p1_action,
        reward=reward
    ))

    # Actualizamos 𝜀
    𝜀 -= 𝛿𝜀 if 𝜀 > 0 else 0

print(agent)


dataset_url = 'https://blazaid.github.io/Aprendizaje-profundo/Datasets/rock-paper-scissors-lizard-spock.tst'

player2_actions = []
with urllib.request.urlopen(dataset_url) as f:
    for line in f:
        move = line.decode('utf-8').strip().upper()
        if move:
            player2_actions.append(Action[move])

stats = collections.defaultdict(int)
state = 0
for p2_action in player2_actions:
    p1_action = agent.decide(state)
    reward = game.play(p1_action, p2_action)
    if reward == 1:
        stats['wins'] += 1
    elif reward == -1:
        stats['loses'] += 1
    else:
        stats['ties'] += 1

print(dict(stats))