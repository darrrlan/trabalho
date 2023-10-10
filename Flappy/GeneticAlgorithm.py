import pygad
from Demo import FlappyBird_Human
from MLP import MLP
import numpy as np
import random

# Defina a função de aptidão

def save_generation_data(generation_number, max_fitness):
    with open("generation_data.txt", "a") as data_file:
        data_file.write(f"Geração: {generation_number}, Maior Fitness: {max_fitness}\n")

def fitness_function(ga_instance,solution, solution_idx):
    flappy_bird = FlappyBird_Human()
    mlp = MLP(5, 8, 1)
    mlp.set_weights(solution)

    while True:
        flappy_bird.run_game_with_mlp(mlp)  # Obtenha a pontuação do jogo.

        if flappy_bird.dead:
            score_on_death = flappy_bird.get_score()
            distance_x_on_death = flappy_bird.get_distance_x()
            fit = score_on_death*1000 + distance_x_on_death
            if flappy_bird.birdY <= 0 or flappy_bird.birdY >= 700:
                score_on_death -= 100

            break
    return fit

def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    max_fitness = ga_instance.best_solution()[1]
    print("Fitness of the best solution :", max_fitness)
    
    # Salve os dados da geração no arquivo de texto
    save_generation_data(ga_instance.generations_completed, max_fitness)

# Parâmetros do algoritmo genético
num_generations = 1500
num_parents_mating = 5
mutation_rate = 0.01  # Probabilidade de mutação por gene
init_range_low = -0.5
init_range_high = 0.5
num_genes = 48  # Número de genes na solução (ajuste de acordo com o tamanho do MLP)
sol_per_pop = 20
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
