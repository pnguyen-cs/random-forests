import random
import csv
from readmission import stdev, loadCsv, splitDataset, pca
import numpy as np

class Node:
    def __init__(self, dataset):
        self.dataset = dataset
        self.right = None
        self.left = None
        self.index = None
        self.value = None
    
    def best_split(self, n_features):
        class_values = [0, 1]
        b_index, b_value, b_score, b_groups = None, None, float("inf"), None
        # Random array of feature indices
        features = np.random.randint(self.dataset.shape[1], size=n_features)
        for index in features:
            for row in range(self.dataset.shape[0]):
                groups = split_data(self.dataset, index, self.dataset[row, index])
                gini = gini_score(groups, class_values)
                if gini < b_score:
                    b_index = index
                    b_value = self.dataset[row, index]
                    b_score = gini
                    b_groups = groups
        self.index = b_index
        self.value = b_value
        self.groups = b_groups
    
    
    def split(self, max_depth, min_size, n_features, curr_depth):
        l_group, r_group = self.groups
        print("splitting")
        # Tree should have no notion of the data that was used to
        # produce the tree
        # Check if one of the groups is empty
        # this means that there was no split that could occur with a better gini score
        if l_group.size == 0:
            self.left = most_common_label(r_group)
            self.right = self.left
            #stop recursion
            return
        elif r_group.size == 0:
            self.left = most_common_label(l_group)
            self.right = self.left
            # Stop recursion
            return
        # if we have reached our desired tree depth, stop
        if curr_depth >= max_depth:
            self.left, self.right = most_common_label(l_group), most_common_label(r_group)
            return
        # Split left and right group if larger than min size
        if len(l_group) <= min_size:
            self.left = most_common_label(l_group)
        else:
            self.left = Node(l_group)
            self.left.best_split(n_features)
            self.left.split(max_depth, min_size, n_features, curr_depth + 1)
        if len(l_group) <= min_size:
            self.left = most_common_label(l_group)
        else:
            self.right = Node(r_group)
            self.right.best_split(n_features)
            self.right.split(max_depth, min_size, n_features, curr_depth + 1)
    

def build_tree(train_data, max_depth, min_size, n_features):
    root = Node(train_data)
    root.best_split(n_features)
    root.split(max_depth, min_size, n_features, 1)
    return root

def predict(node, row):
    print(node.index, row)
    if not row:
        print("row is empty")
        return
    if row[node.index] < node.value:
        if node.left in [0, 1]:
            return node.left
        else:
            predict(node.left, row)
    else:
        if node.right in [0, 1]:
            return node.right
        else:
            predict(node.right, row)
            

def most_common_label(group):
    if group.size == 0:
        raise Exception("Group is empty")
    labels = group[-1,]
    # np function for counting occurences of each label
    vals, counts = np.unique(labels, return_counts=True)
    # find the index of the most frequent one
    max_index = np.argmax(counts)
    # return the most frequent value
    return vals[max_index]




def split_data(data, index, val):
    """splits a dataset based on the index of a feature and a value for that feature"""
    left = []
    right = []
    feature_split = data[:, index]
    for i in range(len(feature_split)):
        if feature_split[i] < val:
            left.append(data[i])
        else:
            right.append(data[i])
    return [np.array(left), np.array(right)]

def gini_score(groups_formed, classes):
    total_samples = 0
    for group in groups_formed:
        total_samples += len(group)
    # must initialize in case group size is 0
    gini_score = 0
    for group in groups_formed:
        if not (len(group) == 0):
            current_score = 0
            for val in classes:
                #get labels for each row
                labels = group[:, -1]
                labels = list(labels).count(val)
                prob = labels / float(len(group))
                current_score += prob**2
            gini_score += (1.0 - current_score)*(float(len(group)/float(total_samples)))
    return gini_score

#def main():
#    print(best_split(data, 4))




# data = loadCsv("readmissionTest.csv")
#should use stratification to split data
# np.random.seed(2)
# pca outputs min num of principal components to cover the given variance ratio with the readmission data as the last column
# data, num_features = pca(0.7)
# split = splitDataset(data, 0.2)
# small_train = split[0]   
#use k-folds to get size of each original tree
#main()
