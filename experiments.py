import feature_creator as fc
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,cross_val_score
from sklearn.base import clone
import numpy as np
from copy import copy,deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pickle
from tqdm import tqdm

X_data = []
y_data = []

df = pd.read_csv('spambase.csv')
X = df.drop(columns=['spam']).to_numpy()
y = df['spam'].to_numpy()

X_data.append(deepcopy(X))
y_data.append(deepcopy(y))

df = pd.read_csv('spect_train.csv')
X = df.drop(columns=['OVERALL_DIAGNOSIS']).to_numpy()
y = df['OVERALL_DIAGNOSIS'].to_numpy()

X_data.append(deepcopy(X))
y_data.append(deepcopy(y))

df = pd.read_csv('ionosphere_data.csv')
X = df.drop(columns=['column_ai']).to_numpy()
y = df['column_ai'].to_numpy()

X_data.append(deepcopy(X))
y_data.append(deepcopy(y))

model = XGBClassifier()

feature_models = [XGBRegressor(),DecisionTreeRegressor(max_depth=5),DecisionTreeRegressor(max_depth=3)]
n_features = [1,2,5]
batch_sizes = [25,50,100]

results = []

for i in tqdm(range(len(y_data))):
    for fm in tqdm(feature_models):
        for n in tqdm(n_features):
            for b in tqdm(batch_sizes):
                features, ga_fitness = fc.feature_creator(model,
                                            fm,
                                            X_data[i],
                                            y_data[i],
                                            n_features=n,
                                            batch_size=(b/y_data))

                new_X = copy(X_data[i])
                for f in features:
                    new_X = np.hstack((new_X,np.array([fc.get_feature_values(f,X_data[i])]).T))

                base_scores = cross_val_score(model, X_data[i], y_data[i], cv=10, scoring='f1')
                scores = cross_val_score(model, new_X, y_data[i], cv=10, scoring='f1')

                row = [i,fm.__str__()[:15],n,b,np.mean(base_scores),np.std(base_scores),np.mean(scores),np.std(scores)]
                results.append(row)

                with open('results_continuous_fetures_2023_02_01.pkl', 'wb') as f:
                    pickle.dump(results, f)

print(results)