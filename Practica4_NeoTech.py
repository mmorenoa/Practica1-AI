import gym
import random
import numpy as np
from operator import attrgetter
import matplotlib.pyplot as plt

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

            self.individuos[i].calcular_valor_fitness(suma_recompensas)

    def mutacion(self, num_acciones, porcentaje_mutacion, num_mutaciones):
        for individuo in self.individuos:
            probabilidad_mutacion = random.randint(0, 100)

            if probabilidad_mutacion >= porcentaje_mutacion * 100:
                continue

            indices_mutacion = random.sample(range(num_acciones), num_mutaciones)

            for indice in indices_mutacion:
                individuo.acciones[indice] = env.action_space.sample()

    def cruce(self, num_acciones, elite):
        nueva_poblacion = []
        nueva_poblacion.extend(elite)

        for i in range(len(elite), len(self.individ.uos), 2):
            padre1, padre2 = self.individuos[i], self.individuos[i + 1]
            hijo1 = Individuo(num_acciones)
            hijo2 = Individuo(num_acciones)
            punto_corte = random.randint(1, num_acciones - 1)

            hijo1.acciones = padre1.acciones[:punto_corte] + padre2.acciones[punto_corte:]
            hijo2.acciones = padre2.acciones[:punto_corte] + padre1.acciones[punto_corte:]

            nueva_poblacion.extend([hijo1, hijo2])
        self.individuos = nueva_poblacion

    def ejecutar_algoritmo_genetico(self, num_generaciones, num_individuos, num_acciones, num_playout, porcentaje_mutacion, num_mutaciones, env):
        mejores_fitness = []

        for _ in range(num_generaciones):
            elite = self.seleccionar_elite(num_individuos)
            self.calcular_probabilidad_seleccion()
            self.realizar_playout(num_individuos, num_acciones, num_playout, env)
            self.mutacion(num_acciones, porcentaje_mutacion, num_mutaciones)
            self.cruce(num_acciones, elite)

            mejor_individuo = max(self.individuos, key=attrgetter('valor_fitness'))
            mejores_fitness.append(mejor_individuo.valor_fitness)

        return mejores_fitness


class Individuo:
    def init(self, num_acciones):
        self.acciones = [env.action_space.sample() for _ in range(num_acciones)]
        self.valor_fitness = 0
        self.probabilidad_seleccion = 0

    def calcular_valor_fitness(self, recompensa_total):
        self.valor_fitness = recompensa_total

    def set_probabilidad_seleccion(self, probabilidad):
        self.probabilidad_seleccion = probabilidad





# Configuración de parámetros
num_generaciones = 100
num_individuos = 50
num_acciones = 10
num_playout = 100
porcentaje_mutacion = 0.1
num_mutaciones = 2

#Crear el ambiente de Gym
env = gym.make('BipedalWalker-v3')

#Crear la población inicial
poblacion = Poblacion()
for _ in range(num_individuos):
    individuo = Individuo(num_acciones)
    poblacion.individuos.append(individuo)

#Ejecutar el algoritmo genético
mejores_fitness = poblacion.ejecutar_algoritmo_genetico(num_generaciones, num_individuos, num_acciones, num_playout, porcentaje_mutacion, num_mutaciones, env)

#Mostrar resultados
plt.plot(range(num_generaciones), mejores_fitness)
plt.xlabel('Generación')
plt.ylabel('Mejor Fitness')
plt.title('Evolución del Mejor Fitness')
plt.show()

# TERMINAR ESTAPARTE PARA QUE IMPRIMa ALS COSAS Y TAL






# SI NO VA LO DE ARRIBA, ESTA ES LA VERSION ORIGINAL PERO EN ESPAÑOL
# --------------------------------------------------------------------------------------------------------------------------------

# import gym
# import random
# import copy
# import numpy as np
# import itertools
# import heapq
# import time
# from operator import attrgetter
# import matplotlib.pyplot as plt

# class Poblacion:
# 	def __init__(self):
# 		self.cromosomas = []

# 	def calcular_seleccion_probabilidad(self):
# 		# Normalizar las recompensas a [0,1]
# 		recompensas_normalizadas = []
# 		recompensas_minimas = float("inf")
# 		suma_recompensas = 0

# 		for i in self.cromosomas:
# 			if i.valor_fitness <= recompensas_minimas:
# 				recompensas_minimas = i.valor_fitness

# 		for i in self.cromosomas:
# 			suma_recompensas += i.valor_fitness + abs(recompensas_minimas) + 1

# 		for i in self.cromosomas:
# 			recompensas_normalizadas.append((i.valor_fitness + abs(recompensas_minimas) + 1) / suma_recompensas)

# 		# Crear la distribución
# 		for i in range(len(recompensas_normalizadas)):
# 			self.cromosomas[i].set_probabilidad_seleccion(recompensas_normalizadas[i])

# 	def seleccionar_pares_cromosomas(self):
# 		# Crear una lista de probabilidades acumulativas para facilitar
# 		# la selección de pares
# 		probabilidades_acumulativas = []
# 		suma_parcial = 0

# 		for i in range(len(self.cromosomas)):
# 			suma_parcial += self.cromosomas[i].probabilidad_seleccion
# 			probabilidades_acumulativas.append(suma_parcial)

# 		numero_aleatorio_1 = random.uniform(0.0, 1.0)
# 		numero_aleatorio_2 = random.uniform(0.0, 1.0)
# 		cromosoma_seleccionado_1 = -1
# 		cromosoma_seleccionado_2 = -1

# 		# Iterar a través de las listas para obtener los cromosomas
# 		for i in range(len(probabilidades_acumulativas)):
# 			if numero_aleatorio_1 <= probabilidades_acumulativas[i]:
# 				cromosoma_seleccionado_1 = i
# 				break

# 		for i in range(len(probabilidades_acumulativas)):
# 			if numero_aleatorio_2 <= probabilidades_acumulativas[i]:
# 				cromosoma_seleccionado_2 = i
# 				break

# 		return cromosoma_seleccionado_1, cromosoma_seleccionado_2

# 	def realizar_playout(self, N_cromosomas, N_ACCIONES, N_PLAYOUT, env):
# 		for i in range(N_cromosomas):
# 			env.reset()
# 			suma_recompensas = 0

# 			for _ in range(N_PLAYOUT // N_ACCIONES):
# 				for j in range(N_ACCIONES):
# 					accion = self.cromosomas[i].acciones[j]
# 					_, recompensa, _, _ = env.step(accion)
# 					suma_recompensas += recompensa

# 			self.cromosomas[i].calcular_valor_fitness(suma_recompensas)

# 	def mutacion(self, N_ACCIONES, PORCENTAJE_MUTACION, N_MUTACIONES):
# 		for cromosoma in self.cromosomas:
# 			probabilidad_mutacion = random.randint(0, 100)

# 			if probabilidad_mutacion >= PORCENTAJE_MUTACION * 100:
# 				continue

# 			# Generar los índices de acciones a mutar completamente
# 			indices_mutacion = random.sample(range(N_ACCIONES), N_MUTACIONES)

# 			for m in indices_mutacion:
# 				cromosoma.acciones[m] = env.action_space.sample()

# 	# Este cruce implica a todos los cromosomas (cada uno tiene una oportunidad de ser seleccionado)
# 	def cruce_1(self, N_ACCIONES, elite):
# 		nueva_poblacion = []
# 		nueva_poblacion.extend(elite)

# 		# Ignorar a la élite, iterar hasta el final de los cromosomas
# 		for i in range(len(elite), len(self.cromosomas), 2):
# 			# Etapa de selección: seleccionar dos cromosomas para el cruce (la élite PUEDE ser seleccionada)
# 			indice_1, indice_2 = self.seleccionar_pares_cromosomas()

# 			# Cruce
# 			nuevo_cromosoma_1 = self.cromosomas[indice_1].cruce_entre_cromosomas(self.cromosomas[indice_2], N_ACCIONES)
# 			nuevo_cromosoma_2 = self.cromosomas[indice_2].cruce_entre_cromosomas(self.cromosomas[indice_1], N_ACCIONES)

# 			nueva_poblacion.append(nuevo_cromosoma_1)
# 			nueva_poblacion.append(nuevo_cromosoma_2)

# 		self.cromosomas = copy.deepcopy(nueva_poblacion)

# 	def cruce_2(self, N_ACCIONES, elite):
# 		nueva_poblacion = []
# 		nueva_poblacion.extend(elite)

# 		# Ignorar a la élite, iterar hasta el final de los cromosomas
# 		indices_aleatorios = list(range(len(elite)))
# 		random.shuffle(indices_aleatorios)

# 		for i in range(0, len(elite), 2):
# 			# Etapa de selección: seleccionar dos cromosomas de la élite SOLAMENTE
# 			indice_1 = indices_aleatorios[i]
# 			indice_2 = indices_aleatorios[i+1]

# 			# Cruce
# 			nuevo_cromosoma_1 = self.cromosomas[indice_1].cruce_entre_cromosomas(self.cromosomas[indice_2], N_ACCIONES)
# 			nuevo_cromosoma_2 = self.cromosomas[indice_2].cruce_entre_cromosomas(self.cromosomas[indice_1], N_ACCIONES)

# 			nueva_poblacion.append(nuevo_cromosoma_1)
# 			nueva_poblacion.append(nuevo_cromosoma_2)

# 		self.cromosomas = copy.deepcopy(nueva_poblacion)

# 	def elitismo(self, PORCENTAJE_ELITISMO, N_cromosomas):
# 		return self.cromosomas[:int(N_cromosomas * PORCENTAJE_ELITISMO)]



# class Cromosoma:
# 	def __init__(self, acciones, N_ACCIONES):
# 		self.acciones = copy.deepcopy(acciones)
# 		self.probabilidad_seleccion = 0
# 		self.valor_fitness = 0

# 	def set_probabilidad_seleccion(self, probabilidad_seleccion):
# 		self.probabilidad_seleccion = probabilidad_seleccion

# 	def calcular_valor_fitness(self, valor_fitness):
# 		self.valor_fitness = valor_fitness

# 	def cruce_entre_cromosomas(self, cromosoma, N_ACCIONES):
# 		nuevas_acciones = []
# 		# 75% del self y 25% del otro
# 		proporcion = (3 * N_ACCIONES) // 4

# 		for i in range(proporcion):
# 			aux = self.acciones[i]
# 			nuevas_acciones.append(aux)

# 		for i in range(proporcion, N_ACCIONES):
# 			aux = cromosoma.acciones[i]
# 			nuevas_acciones.append(aux)

# 		nuevo_cromosoma = Cromosoma(nuevas_acciones, N_ACCIONES)
# 		return nuevo_cromosoma

# 	def imprime(self):
# 		print('Valor de fitness: ', self.valor_fitness)

# N_GENERACIONES = 100
# N_cromosomas = 100 # Número par
# N_ACCIONES = 40 # Tiene que ser un número múltiplo de 4 (debido a la mutación - 0.75)
# N_PLAYOUT = 500 # Número de veces que se jugará un cromosoma
# PORCENTAJE_MUTACION = 0.50 # Probabilidad de que un cromosoma mute
# N_MUTACIONES = 1 # Número de mutaciones dentro de un cromosoma - Debe ser un valor menor que N_ACCIONES
# PORCENTAJE_ELITISMO = 0.50 # El resultado de la multiplicación de este valor y N_cromosomas debe ser par

# env = gym.make('BipedalWalker-v3')
# poblacion = Poblacion()

# eje_x = list(range(1, N_GENERACIONES + 1))
# eje_y = []

# env.reset()


# #La población inicial se inicializa aleatoriamente
# for i in range(N_cromosomas):
#     acciones_aleatorias = []
#     for _ in range(N_ACCIONES):
#         acciones_aleatorias.append(env.action_space.sample())
#     cromosoma = Cromosoma(acciones_aleatorias, N_ACCIONES)
#     suma_recompensas = 0
#     for i in range(N_ACCIONES):
#         _, recompensa, _, _ = env.step(cromosoma.acciones[i])
#         suma_recompensas += recompensa
#     cromosoma.calcular_valor_fitness(suma_recompensas)
#     poblacion.cromosomas.append(cromosoma)
# env.reset()

# #Bucle principal
# for contador_generacion in range(N_GENERACIONES):
#     print('Generación', contador_generacion + 1)
#     contador_generacion += 1
#     poblacion.playout(N_cromosomas, N_ACCIONES, N_PLAYOUT, env)
#     poblacion.calcular_seleccion_probabilidad() # Necesario para el cruce_1
#     poblacion.cromosomas.sort(key=lambda x: x.valor_fitness, reverse=True)
#     # Se grafica el mejor cromosoma
#     eje_y.append(poblacion.cromosomas[0].valor_fitness)
#     print('Mejores cromosomas')
#     poblacion.cromosomas[0].imprime()
#     poblacion.cromosomas[1].imprime()
#     poblacion.cromosomas[2].imprime()
#     print('Peores cromosomas')
#     poblacion.cromosomas[-3].imprime()
#     poblacion.cromosomas[-2].imprime()
#     poblacion.cromosomas[-1].imprime()
#     # Elitismo
#     elite = poblacion.elitismo(PORCENTAJE_ELITISMO, N_cromosomas)
#     # Cruce
#     poblacion.cruce_2(N_ACCIONES, elite)
#     # Mutación
#     poblacion.mutacion(N_ACCIONES, PORCENTAJE_MUTACION, N_MUTACIONES)
#     print()

# # Graficar el gráfico con respecto al mejor cromosoma en cada generación
# plt.plot(eje_x, eje_y)
# plt.xlabel("Generaciones")
# plt.ylabel("Valor de fitness")
# plt.savefig("results_plot")

# # En caso de que sea necesario graficar nuevamente el gráfico
# with open('results.txt', 'w') as f:
#     f.write("[")
#     for item in eje_x:
#         f.write("%s, " % item)
#     f.write("]\n\n")
#     f.write("[")
#     for item in eje_y:
#         f.write("%s, " % item)
#     f.write("]")