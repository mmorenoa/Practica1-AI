import gym
import random
import numpy as np
from operator import attrgetter
import matplotlib.pyplot as plt
import copy

class Poblacion:
    def __init__(self):
        self.individuos = []

    def calcular_probabilidad_seleccion(self):
        recompensas_normalizadas = []
        recompensas_minimas = float("inf")
        suma_recompensas = 0

        for individuo in self.individuos:
            if individuo.valor_fitness <= recompensas_minimas:
                recompensas_minimas = individuo.valor_fitness

        for individuo in self.individuos:
            suma_recompensas += individuo.valor_fitness + abs(recompensas_minimas) + 1

        for individuo in self.individuos:
            recompensas_normalizadas.append((individuo.valor_fitness + abs(recompensas_minimas) + 1) / suma_recompensas)

        for i in range(len(recompensas_normalizadas)):
            self.individuos[i].set_probabilidad_seleccion(recompensas_normalizadas[i])

    def seleccionar_pares_individuos(self):
        probabilidades_acumulativas = []
        suma_parcial = 0

        for individuo in self.individuos:
            suma_parcial += individuo.probabilidad_seleccion
            probabilidades_acumulativas.append(suma_parcial)

        numero_aleatorio_1 = random.uniform(0.0, 1.0)
        numero_aleatorio_2 = random.uniform(0.0, 1.0)
        indice_seleccionado_1 = -1
        indice_seleccionado_2 = -1

        for i in range(len(probabilidades_acumulativas)):
            if numero_aleatorio_1 <= probabilidades_acumulativas[i]:
                indice_seleccionado_1 = i
                break

        for i in range(len(probabilidades_acumulativas)):
            if numero_aleatorio_2 <= probabilidades_acumulativas[i]:
                indice_seleccionado_2 = i
                break

        return indice_seleccionado_1, indice_seleccionado_2

    def realizar_playout(self, num_individuos, num_acciones, num_playout, env):
        for i in range(num_individuos):
            env.reset()
            suma_recompensas = 0

            for _ in range(num_playout // num_acciones):
                for j in range(num_acciones):
                    accion = self.individuos[i].acciones[j]
                    _, recompensa, _, _ = env.step(accion)
                    suma_recompensas += recompensa
                    env.render()
            self.individuos[i].calcular_valor_fitness(suma_recompensas)

    def mutacion(self, num_acciones, porcentaje_mutacion, num_mutaciones):
        for individuo in self.individuos:
            probabilidad_mutacion = random.randint(0, 100)

            if probabilidad_mutacion >= porcentaje_mutacion * 100:
                continue

            indices_mutacion = random.sample(range(num_acciones), num_mutaciones)

            for indice in indices_mutacion:
                individuo.acciones[indice] = env.action_space.sample()

    def cruce(self, N_ACCIONES, elite):
        nueva_poblacion = []
        nueva_poblacion.extend(elite)

        # Ignorar a la élite, iterar hasta el final de los individuos
        indices_aleatorios = list(range(len(elite)))
        random.shuffle(indices_aleatorios)

        for i in range(0, len(elite), 2):
            # Etapa de selección: seleccionar dos individuos de la élite SOLAMENTE
            indice_1 = indices_aleatorios[i]
            indice_2 = indices_aleatorios[i+1]

            # Cruce
            nuevo_individuo_1 = self.individuos[indice_1].cruce_entre_cromosomas(self.individuos[indice_2], N_ACCIONES)
            nuevo_individuo_2 = self.individuos[indice_2].cruce_entre_cromosomas(self.individuos[indice_1], N_ACCIONES)

            nueva_poblacion.append(nuevo_individuo_1)
            nueva_poblacion.append(nuevo_individuo_2)

        self.individuos = copy.deepcopy(nueva_poblacion)
    
    def seleccionar_elite(self, PORCENTAJE_ELITISMO, N_cromosomas):
        return self.individuos[:int(N_cromosomas * PORCENTAJE_ELITISMO)]

    def ejecutar_algoritmo_genetico(self, num_generaciones, num_individuos, num_acciones, num_playout, porcentaje_mutacion, num_mutaciones, env):
        mejores_fitness = []

        for _ in range(num_generaciones):
            print('Generación', _ + 1)
            self.realizar_playout(num_individuos, num_acciones, num_playout, env)
            self.calcular_probabilidad_seleccion()
            self.individuos.sort(key=lambda x: x.valor_fitness, reverse=True)
            elite = self.seleccionar_elite(porcentaje_elitismo, num_individuos)
            self.cruce(num_acciones, elite)
            self.mutacion(num_acciones, porcentaje_mutacion, num_mutaciones)

            mejor_individuo = max(self.individuos, key=attrgetter('valor_fitness'))
            mejores_fitness.append(mejor_individuo.valor_fitness)
            print("Mejor Individuo = ")
            self.individuos[0].imprime()

        return mejores_fitness



class Individuo:
    def __init__(self, acciones, N_ACCIONES):
        self.acciones = copy.deepcopy(acciones)
        self.probabilidad_seleccion = 0
        self.valor_fitness = 0

    def set_probabilidad_seleccion(self, probabilidad_seleccion):
        self.probabilidad_seleccion = probabilidad_seleccion

    def calcular_valor_fitness(self, valor_fitness):
        self.valor_fitness = valor_fitness

    def cruce_entre_cromosomas(self, individuo, N_ACCIONES):
        nuevas_acciones = []
        # 75% del self y 25% del otro
        proporcion = (3 * N_ACCIONES) // 4

        for i in range(proporcion):
            aux = self.acciones[i]
            nuevas_acciones.append(aux)

        for i in range(proporcion, N_ACCIONES):
            aux = individuo.acciones[i]
            nuevas_acciones.append(aux)

        nuevo_individuo = Individuo(nuevas_acciones, N_ACCIONES)
        return nuevo_individuo

    def imprime(self):
        print('Valor de fitness: ', self.valor_fitness)





# Configuración de parámetros
num_generaciones = 100
num_individuos = 100
num_acciones = 40
num_playout = 500
porcentaje_mutacion = 0.5
num_mutaciones = 1
porcentaje_elitismo = 0.50


#Crear el entorno de Gym
env = gym.make('BipedalWalker-v3')

# Crear la población inicial
env.reset()
poblacion = Poblacion()
for i in range(num_individuos):
    acciones_aleatorias = []

    for _ in range(num_acciones):
        acciones_aleatorias.append(env.action_space.sample())

    individuo = Individuo(acciones_aleatorias, num_acciones)
    suma_recompensas = 0

    for i in range(num_acciones):
        _, recompensa, _, _ = env.step(individuo.acciones[i])
        suma_recompensas += recompensa

    individuo.calcular_valor_fitness(suma_recompensas)
    poblacion.individuos.append(individuo)
env.reset()

#Ejecutar el algoritmo genético
mejores_fitness = poblacion.ejecutar_algoritmo_genetico(num_generaciones, num_individuos, num_acciones, num_playout, porcentaje_mutacion, num_mutaciones, env)

# Mostrar resultados
plt.plot(range(1, num_generaciones+1), mejores_fitness)
plt.xlabel('Generación')
plt.ylabel('Mejor Fitness')
plt.title('Evolución del Mejor Fitness')
plt.show()
