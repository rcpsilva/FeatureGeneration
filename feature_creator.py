from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,cross_val_score
from sklearn.base import clone
import numpy as np
import pygad
from copy import copy

def get_feature_values(feature,X):
    return feature.predict(X)


def fitness(new_feature,X_temp,y_temp,model):
    X = np.hstack((X_temp,np.array([new_feature]).T))
    model.fit(X,y_temp)
    scores = cross_val_score(model, X, y_temp, cv=4, scoring='f1')
    return np.mean(scores)

def feature_creator(model,feature_model,X,y,n_features=5,batch_size=0.05):
    feature_models = [0 for i in range(n_features)]

    idxs = get_subsample_idxs(X,y,n_features,batch_size)

    ga_fitness = []

    feature_model_idx = 0
    for batch in idxs:
        X_temp = X[batch,:]
        y_temp = y[batch]

        fitness_function = lambda x,sol_idx : fitness(x,X_temp,y_temp,model)

        # GA parameters
        num_generations = 50
        num_parents_mating = 2

        sol_per_pop = 30
        num_genes = len(y_temp)

        init_range_low = 0
        init_range_high = 2

        parent_selection_type = "sss"
        keep_parents = 1

        crossover_type = "two_points"

        mutation_type = "random"
        mutation_percent_genes = 10

        ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       gene_type=int)

        # Run GA
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        # Persist fitness curves
        ga_fitness.append(copy(ga_instance.best_solutions_fitness))

        # fit model feature model

        feature_models[feature_model_idx] = clone(feature_model)
        feature_models[feature_model_idx].fit(X_temp,solution)

        feature_model_idx+=1

    return feature_models, ga_fitness


def get_subsample_idxs(X,y,n_features,batch_size):

    idx = []
    sss = StratifiedShuffleSplit(n_splits=n_features, test_size=batch_size)
    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        idx.append(test_index)

    return idx 
