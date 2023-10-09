import pygad
from MLP import MLP  # Certifique-se de que a classe MLP esteja no mesmo diretório ou no PYTHONPATH
from Demo import FlappyBird_Human

# Criar uma instância da classe MLP
mlp = MLP(5, 8, 1)

# Criar uma instância da classe FlappyBird_Human passando a instância da MLP
flappy_bird = FlappyBird_Human(mlp)

# Definir a função de aptidão
def fitness_function(solution, solution_idx):
    # Simule o jogo Flappy Bird e obtenha o valor do contador (score)
    score = flappy_bird.get_score()

    # Retorne o score como aptidão
    return score

# Configuração do algoritmo genético
num_generations = 100
num_parents_mating = 10
mutation_rate = 0.01
num_genes = 1
sol_per_pop = 20
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"

# Crie uma instância do objeto PyGAD
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_rate)

# Executa o algoritmo genético
ga_instance.run()

# Obtém a melhor solução encontrada
best_solution, best_fitness = ga_instance.best_solution()
print("Melhor Solução:", best_solution)
print("Melhor Fitness (Score):", best_fitness)
