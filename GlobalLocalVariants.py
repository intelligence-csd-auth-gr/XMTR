from sklearn.metrics import mean_absolute_error
import random
import statistics
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestRegressor
from mlxtend.preprocessing import TransactionEncoder
from sklearn.multioutput import MultiOutputRegressor
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

class GlobalSurrogateTree:
    def __init__(self, x, y, feature_names, random_state=10):
        
        self.feature_names = feature_names
        
        dtree = DecisionTreeRegressor(random_state = random_state)
        parameters = [{
            'criterion': ['mse', 'mae'],#, 'poisson'],
            'splitter': ['best','random'],
            'max_depth': [1, 2, 5],#, 10, None],
            'max_features': ['sqrt', 'log2', 0.75, None],#['sqrt', 'log2', 0.75, None], #'sqrt', 'log2', 0.75, None
            'min_samples_leaf' : [1, 2, 5, 10],#[1, 2, 5, 10, 0.10], #1, 2, 5, 10, 0.10
        }]
        clf = GridSearchCV(estimator=dtree, param_grid=parameters, cv=10, n_jobs=-1, verbose=0, scoring='neg_mean_absolute_error')
        clf.fit(x, y)
        self.accuracy = clf.best_score_
        self.model = clf.best_estimator_
        
    def rule(self,instance):
        path = self.model.decision_path([instance])
        leq = {}  # leq: less equal ex: x <= 1
        b = {}  # b: bigger ex: x > 0.6
        local_range = {}
        for node in path.indices:
            feature_id = self.model.tree_.feature[node]
            feature = self.feature_names[feature_id]
            threshold = self.model.tree_.threshold[node]
            if threshold != -2.0:
                if instance[feature_id] <= threshold:
                    leq.setdefault(feature, []).append(threshold)
                else:
                    b.setdefault(feature, []).append(threshold)
        for k in leq:
            local_range.setdefault(k, []).append(['<=', min(leq[k])])  # !!
        for k in b:
            local_range.setdefault(k, []).append(['>', max(b[k])])  # !!
        return local_range, self.model.predict([instance])[0]


class LocalSurrogateTree:
    def __init__(self, x, y, feature_names, neighbours=None, random_state=10):
        self.x = x
        self.y = y
        self.neighbours = neighbours
        if neighbours is None:
            self.neighbour = int(len(x)/10)
        self.feature_names = feature_names
        
        neighbours_generator = KNeighborsRegressor(n_neighbors=self.neighbours, weights="distance", metric="minkowski", p=2)
        neighbours_generator.fit(self.x, self.y)
        self.neighbours_generator = neighbours_generator
        dtree = DecisionTreeRegressor(random_state = random_state)
        parameters = [{
            'criterion': ['mse', 'mae'],
            'splitter': ['best','random'],
            'max_depth': [1, 2, 5],#, 10, None],
            'max_features': ['sqrt', 'log2', 0.75, None],#['sqrt', 'log2', 0.75, None], #'sqrt', 'log2', 0.75, None
            'min_samples_leaf' : [1, 2, 5, 10],#[1, 2, 5, 10, 0.10], #1, 2, 5, 10, 0.10
        }]
        self.clf = GridSearchCV(estimator=dtree, param_grid=parameters, cv=10, n_jobs=-1, verbose=0, scoring='neg_mean_absolute_error')

    def _generate_neighbours(self,instance):
        x = [instance]
        ys = self.neighbours_generator.kneighbors(x, n_neighbors=self.neighbours, return_distance=False) # ----> self.x on github
        new_x_train = []
        new_y_train = []
        for i in ys[0]:
            new_x_train.append(self.x[i])
            new_y_train.append(self.y[i])
        return new_x_train, new_y_train
    
    def rule(self,instance):
        local_x, local_y = self._generate_neighbours(instance)
        self.clf.fit(local_x, local_y)
        model = self.clf.best_estimator_
        
        path = model.decision_path([instance])
        leq = {}  # leq: less equal ex: x <= 1
        b = {}  # b: bigger ex: x > 0.6
        local_range = {}
        for node in path.indices:
            feature_id = model.tree_.feature[node]
            feature = self.feature_names[feature_id]
            threshold = model.tree_.threshold[node]
            if threshold != -2.0:
                if instance[feature_id] <= threshold:
                    leq.setdefault(feature, []).append(threshold)
                else:
                    b.setdefault(feature, []).append(threshold)
        for k in leq:
            local_range.setdefault(k, []).append(['<=', min(leq[k])])  # !!
        for k in b:
            local_range.setdefault(k, []).append(['>', max(b[k])])  # !!
        return local_range, model.predict([instance])[0]
