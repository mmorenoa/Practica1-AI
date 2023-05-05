import enum
import typing
import urllib
import requests

import numpy as np
import collections

# - La piedra aplasta la tijera y aplasta al lagarto
# - La tijera corta el papel y decapita al lagarto
# - El papel envuelve la piedra y desautoriza a Spock
# - El lagarto envenena a Spock y come el papel
# - Spock rompe las tijeras y vaporiza la piedra

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

class Game:
    RENDER_MODE_HUMAN = 'human'
    
    def __init__(self, render_mode=None):
        self.render_mode = render_mode

    def play(self, p1_action, p2_action):
        result = MOVES_AND_REWARDS[Action(p1_action), Action(p2_action)]
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

# Va a guardar la info de una transicion (definida por el estado origen, dest, accion por la que ocurre transicion y recompensa)

class Transition(typing.NamedTuple):
    """Representa la transición de un estado al siguiente"""
    prev_state: int              # Estado origen de la transición
    next_state: int              # Estado destino de la transición
    action: Action               # Acción que provocó esta transición
    reward: typing.SupportsFloat # Recompensa obtenida
    

class Agent:
    def __init__(self, name: str, q_table: typing.Any=None):
        """Inicializa este objeto.
        
        :param name: El nombre del agente, para identificarle.
        :param q_table: Una tabla q de valores. Es opcional.
        """
        #SI se proporciona val q table al crear Objecto AGent se asigna (self.name)
        # En caso contrario, se inicializa como diccionario vacio y se le asigna 0
        # para cada par de estados posibles. (0,0) el primero representa el valor de la funcion de valor de accion
        # y el segundo 0 es el numero de veces que se ha tomado la accion en ese estado
        self.name = name
        if q_table:
            self.q_table = q_table
        else:
            self.q_table = {}  
            self.q_table = {b.value: 0.0 for b in Action}
                
    def decide(self, state:int, 𝜀: typing.SupportsFloat=0.0) -> Action:
        """Decide la acción a ejecutar.
        
        :param state: El estado en el que nos encontramos.
        :param 𝜀: Un valor entre 0 y 1 que representa, según la estrategia
            ε-greedy, la probabilidad de que la acción sea elegida de manera
            aleatoria de entre todas las opciones posibles. Es opcional, y si
            no se especifica su valor es 0 (sin probabilidades de que se elija
            una acción aleatoria).
        :param returns: La acción a ejecutar.
        """
        # Si cumple if, se elige una accion aleatoria de todas las posibles
        # De lo contrario, argmax se usa para obtener la max recompensa esperada en el estado actual
        # y devuelve esa accion
        
        if np.random.random() < 𝛆:
            return np.random.choice(list(Action))
        if state in self.q_table:
            argmax = np.argmax(self.q_table[state])
            return list(Action)[argmax]
        else:
            self.q_table[state] = {}
            for action in Action:
                self.q_table[state][action.value] = 0.0
            return np.random.choice(list(Action))

       

    def update(self, t: Transition, 𝛼=0.1, 𝛾=0.95):
        """Actualiza el estado interno de acuerdo a la experiencia vivida.
        
        :param transition: Toda la información correspondiente a la transición
            de la que queremos aprender.
        :param 𝛼: El factor de aprendizaje del cambio en el valor q. Por
            defecto es 0.1
        :param 𝛾: La influencia de la recompensa a largo plazo en el valor q a
            actualizar. Va de 0 (sin influencia) a 1 (misma influencia que el
            valor actual). Por defecto es 0.95.
        """
        
        if t.prev_state not in self.q_table:
            self.q_table[t.prev_state] = {}
        if t.action not in self.q_table[t.prev_state]:
            self.q_table[t.prev_state][t.action.value] = 0.0
        # se obtiene el valor Q actual del par estado-acción previo, que corresponde al estado y la acción que el agente eligió en el paso anterior.
        current_q = self.q_table[t.prev_state][t.action.value]
        # se calcula el valor máximo de la función de valor Q para todas las acciones posibles en el estado siguiente.
        next_q = max(list(self.q_table[t.next_state].values()))
        # se calcula el objetivo de la actualización de Q, que es la suma de la recompensa recibida en el paso actual y el valor esperado de recompensa futura a largo plazo, descontado por el factor de descuento γ.
        td_target = t.reward + 𝛾 * next_q
        #se calcula el error de temporal-diferencia, que es la diferencia entre el objetivo de actualización y el valor Q actual.
        td_error = td_target - current_q
        # se actualiza el valor Q para el par estado-acción previo, multiplicando el error de temporal-diferencia por una tasa de aprendizaje α y sumándolo al valor Q actual.
        new_q = current_q + 𝛼 * td_error
        # se actualiza la tabla Q con el nuevo valor Q para el par estado-acción previo.
        self.q_table[t.prev_state][t.action.value] = new_q
        # Guardamos la tabla actualizada en una variable de cadena
        # self.updated_table = self.q_table.copy()
        print(self.q_table)
    
    def __str__(self) -> str:
        """Representación textual de la tabla Q del agente.
        
        :returns: Una cadena indicando la estructura interna de la tabla Q.
        """
        
        table_str = "Agent - Q Table:\n"
        for actions in self.q_table.items():
            table_str += f"{actions}\n"
        return table_str
        
dataset_url = 'https://blazaid.github.io/Aprendizaje-profundo/Datasets/rock-paper-scissors-lizard-spock.trn'

player2_actions = []
with urllib.request.urlopen(dataset_url) as f:
    for line in f:
        move = line.decode('utf-8').strip().upper()
        if move:
            player2_actions.append(Action[move])

#Comenzaremos a realizar el entrenamiento    
𝜀 = 1
𝛿𝜀 = 𝜀 / len(player2_actions)

game = Game()
agent = Agent(name='Agent')

state = 0  # El entorno (juego) no tiene estado, así que siempre será el mismo
for p2_action in player2_actions:
    p1_action = agent.decide(state, 𝛆)
    reward = game.play(p1_action, p2_action)

    # Actualizamos el agente
    agent.update(Transition(
        prev_state=state,
        next_state=state,
        action=p1_action,
        reward=reward
    ))

    # Actualizamos 𝜀
    𝜀 -= 𝛿𝜀 if 𝜀 > 0 else 0

#Tras entrenarlo, veremos como estan repartidos los valores de la tabla Q
print("tras entrenarlo el valor es: \n")
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
game = Game()
agent = Agent(name='Agent')
for p2_action in player2_actions:
    p1_action = agent.decide(state)
    reward = game.play(p1_action, p2_action)
    agent.update(Transition(
         prev_state=state,
         next_state=state,
         action=p1_action,
         reward=reward
     ))
    if reward == 1:
         stats['wins'] += 1
    elif reward == -1:
         stats['loses'] += 1
    else:
         stats['ties'] += 1

print(dict(stats))