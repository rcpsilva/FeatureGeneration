import pygad as pg

# Define the function to optimize (in this case, a simple binary problem)
def binary_problem(solution):
    x = solution[0]
    y = solution[1]
    return x*y


# Set the parameters for the genetic algorithm
num_generations = 100
num_parents_mating = 4

# Initialize the genetic algorithm
ga = pg.genetic_algorithm(binary_problem, num_generations=num_generations, num_parents_mating=num_parents_mating)

# Define the binary problem's parameters
ga.params.binary_mut_prob = 0.1
ga.params.binary_cross_prob = 0.8
ga.params.num_bits = [50]

# Run the genetic algorithm
for generation in range(ga.params.num_generations):
    ga.evolve()
    best_solution = ga.best_solution()
    best_fitness = ga.best_fitness()
    print("Generation: ", generation)
    print("Best solution: ", best_solution)
    print("Best fitness: ", best_fitness)

# Print the final solution
print("\nFinal best solution found: ", ga.best_solution())

