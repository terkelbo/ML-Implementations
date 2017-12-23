# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import division #float division without stating float
from random import seed         #randomness control
from random import randrange    #selecting features
from math import sqrt           #number of features selected

### first create the decision tree algorithm ###

#for classification the gini score is used as the cost function
def gini_index(groups,classes):
    #input: groups: list of groups of size n_rows
    #       classes: list of class values, i.e. binary => len(classes)=2
    
    ####################################################################
    #Count all sample at split point
    n_instances = sum([len(group) for group in groups])
    
    #calculate gini
    gini = 0 #initialize
    for group in groups:
        size = len(group)
        if size == 0: #check for zero division
            continue
        score = 0 #initialize score for gini calc
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p #aggregate score
        #aggregate gini
        gini += (1 - score) * (size / n_instances)
    return gini

#now a function to do a split of the dataset when called
def test_split(index, value, dataset):
    left, right = list(), list() #initialize
    for row in dataset:
        if row[index] < value: #check if rows variable is below or above value
            left.append(row)
        else:
            right.append(row)
    return left, right #return groups

#now a function that tries all possible splits and return the best single split
def get_split(dataset, n_features):
    #input: dataset with last column being the class values
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None #pre allocate values
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1) #randomly select feature (index)
        if index not in features:            #check if it was already included
            features.append(index)           #if not; include it in feature list
    for index in features: 
        for row in dataset: #try all features and rows in dataset
            groups = test_split(index, row[index], dataset) #order the split
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups #if best score then update variables
    return {'index':b_index, 'value':b_value, 'groups':b_groups} #return ordered dict

#now functions to build a tree:
#first a function to output the final node (terminal)
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

#second a function to build our tree
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups']) #potentially to save ram 
    #check if either left or right is empty
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    #check if max depth has been reached
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
   #first build left child complete
    if len(left) <= min_size:
       node['left'] = to_terminal(left)
    else:
         node['left'] = get_split(left, n_features)
         split(node['left'], max_depth, min_size, n_features, depth+1)
   
   #then build right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1) 
        
#now a master function to build a tree
def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root

# Make a single prediction
def single_predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return single_predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return single_predict(node['right'], row)
        else:
            return node['right']
            
#define subsample function
def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample
    
#create a function that bags the predictions from a range of trees
def bagging_predict(trees, row):
	predictions = [single_predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count) #do majority voting

#define master random forest function that creates the trees
def random_forest(train, max_depth, min_size, sample_size, n_trees, n_features):
	trees = list()
	for i in range(n_trees):
		sample = subsample(train, sample_size)
		tree = build_tree(sample, max_depth, min_size, n_features)
		trees.append(tree)
	return(trees)
    
# Predict a full dataset
def predict(trees,rows):
    predictions = list()
    for row in rows:
        predictions = [bagging_predict(trees, row) for row in rows]
    return(predictions)
    