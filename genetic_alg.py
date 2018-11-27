import numpy as np
from simulation import evaluate

    
def perturb_individual(x0, perturb_type, perturb):
    assert perturb_type in ['relative', 'absolute']
    if perturb_type == 'absolute':
        x = x0 + np.random.uniform(-perturb, perturb, size=len(x0))
    else:
        x = x0 * (1 + np.random.uniform(-perturb, perturb, size=len(x0)))
    return x

def generate_first_population(x0, population_size, perturb_type, perturb):
    population = []
    i = 0
    while i < population_size:
        population.append(perturb_individual(x0, perturb_type, perturb))
        i += 1
    return np.array(population)

def evaluate_population(population, syn_func):
    performance = []
    for individual in population:
        airfoil = syn_func(individual)
        performance.append(evaluate(airfoil))
    return performance

def select_best(population, n_best, syn_func):
    performance = evaluate_population(population, syn_func)
    ranking = np.flip(np.argsort(performance), axis=0)
    ranked_population = population[ranking]
    return ranked_population[:n_best], np.max(performance)

def select_random(population, n_random):
    ind = range(population.shape[0])
    return population[np.random.choice(ind, size=n_random)]

def select(population, n_best, n_random, syn_func):
    best, best_perf = select_best(population, n_best, syn_func)
    best_individual = best[0]
    random = select_random(population, n_random)
    selected = np.vstack((best, random))
    np.random.shuffle(selected)
    return selected, best_perf, best_individual

def create_child(individual1, individual2):
    # Crossover
    child = np.zeros_like(individual1)
    ind = np.random.binomial(1, 0.5, size=len(child))
    ind = ind.astype(bool)
    child[ind] = individual1[ind]
    child[np.logical_not(ind)] = individual2[np.logical_not(ind)]
    return child

def create_children(breeders, n_children):
    next_population = []
    for i in range(breeders.shape[0]/2):
        for j in range(n_children):
            next_population.append(create_child(breeders[i], breeders[-i-1]))
    return np.array(next_population)

def mutate_individual(individual, perturb_type, perturb):
    mutate_idx = np.random.choice(len(individual))
    if perturb_type == 'absolute':
        individual[mutate_idx] += np.random.uniform(-perturb, perturb)
    else:
        individual[mutate_idx] *= (1 + np.random.uniform(-perturb, perturb))
    return individual
	
def mutate_population(population, chance_of_mutation, perturb_type, perturb):
    for i, individual in enumerate(population):
        if np.random.rand(1) < chance_of_mutation:
            population[i] = mutate_individual(individual, perturb_type, perturb)
    return population

