from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,cross_val_score,cross_val_predict
from sklearn.base import clone
import numpy as np
from copy import copy
from sklearn.metrics import f1_score
from scipy.optimize import minimize, direct, Bounds, shgo

def get_feature_values(feature,X):
    return feature.predict(X)

def fitness(new_feature,X_temp,y_temp,model):
    X = np.hstack((X_temp,np.array([new_feature]).T))
    model.fit(X,y_temp)
    scores = cross_val_score(model, X, y_temp, cv=5, scoring='f1')
    return np.mean(scores)

def feature_creator(model,feature_model,X,y,n_features=5,batch_size=0.05):
    feature_models = [0 for i in range(n_features)]

    idxs = get_subsample_idxs(X,y,n_features,batch_size)

    results = []

    feature_model_idx = 0
    for batch in idxs:
        X_temp = X[batch,:]
        y_temp = y[batch]

        fitness_function = lambda x: -fitness(x,X_temp,y_temp,model)

        # Nelder mead
        #bounds = Bounds(np.zeros(len(y_temp)), np.ones(len(y_temp)))
        x0 = np.random.rand(len(y_temp)) # Initial guess for the optimization
        options = {'maxiter': 1000, 'xatol': 1e-2, 'fatol': 1e-2}
        res = minimize(fitness_function, x0, method='Nelder-Mead',options=options)

        #res = minimize(fitness_function, x0, method='Powell',options=options,bounds=bounds)

        #bounds = Bounds(np.zeros(len(y_temp)), np.ones(len(y_temp)))
        #res = shgo(fitness_function, bounds)
        
        results.append(res)
        # fit model feature model

        feature_models[feature_model_idx] = clone(feature_model)
        feature_models[feature_model_idx].fit(X_temp,res.x)

        feature_model_idx+=1

    return feature_models, results


def get_subsample_idxs(X,y,n_features,batch_size):

    idx = []
    sss = StratifiedShuffleSplit(n_splits=n_features, test_size=batch_size)
    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        idx.append(test_index)

    return idx 
