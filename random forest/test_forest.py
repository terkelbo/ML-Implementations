# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 13:04:03 2017

@author: s144299
"""

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(return_X_y = True)

import os
os.chdir('C:/Users/s144299/OneDrive for Business 1/ML/implementations/random forest')

from randomforest import predict, random_forest
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

features = pd.DataFrame(data[0])
target = pd.DataFrame(data[1])

data = pd.concat((features,target),axis=1)
msk = np.random.rand(len(data)) < 0.8
train = data[msk].as_matrix()
test = data[~msk].as_matrix()



    
# evaluate algorithm
max_depth = 10
min_size = 1
sample_size = 1.0
n_trees=20
n_features = int(np.sqrt(len(train[0])-1)) #~5

trees = random_forest(train, max_depth, min_size, sample_size, n_trees, n_features)
predictions = predict(trees,test)

accuracy_score(test[:,-1],predictions)

