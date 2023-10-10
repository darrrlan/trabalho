import pygad
from Demo import FlappyBird, FlappyBird_GAME
from MLP import MLP
import numpy as np
import random

# Defina a função de aptidão

def save_generation_data(generation_number, max_fitness):
    with open("generation_data.txt", "a") as data_file:
        data_file.write(f"Geração: {generation_number}, Maior Fitness: {max_fitness}\n")

def fitness_function(ga_instance, solution, solution_idx):
    flappy_bird = FlappyBird()
    mlp = MLP(5, 8, 1)
    mlp.set_weights(solution)


    while not flappy_bird.isDead():
        flappy_bird.tick()  # Atualize o estado do jogo
    fit = flappy_bird.TotalDistance() 
    return fit




def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    max_fitness = ga_instance.best_solution()[1]
    print("Fitness of the best solution :", max_fitness)
    
    # Salve os dados da geração no arquivo de texto
    save_generation_data(ga_instance.generations_completed, max_fitness)

# Parâmetros do algoritmo genético
num_generations = 1000
num_parents_mating = 10
mutation_rate = 0.01  # Probabilidade de mutação por gene
init_range_low = -0.5
init_range_high = 0.5
num_genes = 48  # Número de genes na solução (ajuste de acordo com o tamanho do MLP)
sol_per_pop = 30
parent_selection_type = "tournament"  # Use a seleção por torneio
crossover_type = "single_point"
mutation_type = "swap"

# Crie uma instância do objeto PyGAD
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    parent_selection_type=parent_selection_type,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_rate,  # Defina a taxa de mutação aqui
    on_generation=on_gen,
    initial_population=np.random.uniform(init_range_low, init_range_high, (sol_per_pop, num_genes)),
    keep_elitism=2,random_seed=43
)

# Execute o algoritmo genético
ga_instance.run()

ga_instance.plot_fitness()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

flappy_bird = FlappyBird_GAME()
mlp = MLP(5, 8, 1)
mlp.set_weights(solution)
while not flappy_bird.isDead():
        flappy_bird.tick()  # Atualize o estado do jogo
        