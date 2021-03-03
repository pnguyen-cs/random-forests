import random
import csv
from readmission import stdev, loadCsv, splitDataset, pca, evaluate
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import math

class Node:
    def __init__(self, dataset, features_left):
        self.dataset = dataset
        self.right = None
        self.left = None
        self.index = None
        self.value = None
        self.features_left = features_left
    
    def best_split(self, n_features):
        class_values = [0, 1]
        b_index, b_value, b_score, b_groups = None, None, float("inf"), None
        # Random array of feature indices without the class label
        random.shuffle(self.features_left)
        features = []
        if len(self.features_left) < n_features:
            features = self.features_left
            self.features_left = None
        else:
            features = self.features_left[:3]
            self.features_left = self.features_left[3:]
        for index in features:
            if b_score == 0:
                break
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
        print("splitting")
        l_group, r_group = self.groups[0], self.groups[1]
        # Tree should have no notion of the data that was used to
        # produce the tree
        del self.groups
        del self.dataset
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
        # if we have reached our desired tree depth or if we have run out of features, stop
        if (curr_depth >= max_depth) or not self.features_left:
            self.left, self.right = most_common_label(l_group), most_common_label(r_group)
            return
        # Split left and right group if larger than min size
        if len(l_group) <= min_size:
            self.left = most_common_label(l_group)
        else:
            self.left = Node(l_group, self.features_left)
            self.left.best_split(n_features)
            self.left.split(max_depth, min_size, n_features, curr_depth + 1)
        if len(r_group) <= min_size:
            self.right = most_common_label(r_group)
        else:
            self.right = Node(r_group, self.features_left)
            self.right.best_split(n_features)
            self.right.split(max_depth, min_size, n_features, curr_depth + 1)

    def __repr__(self):
        string = ""
        string += "Value: " + str(self.value) +"  "
        string += "Index: " + str(self.index)
        return string

    def printTree(self, leadspace):
        print(leadspace, self)
        children = [self.left, self.right]
        for child in children:
            if isinstance(child, float):
                print(leadspace + str(child))
            else:
                child.printTree(leadspace + "  ")
    

def build_tree(train_data, max_depth, min_size, n_features):
    root = Node(train_data, list(range(train_data.shape[1])))
    root.best_split(n_features)
    root.split(max_depth, min_size, n_features, 1)
    return root

def predict(node, row):
    if row[node.index] < node.value:
        if node.left in [0.0, 1.0]:
            return node.left
        else:
            predict(node.left, row)
    else:
        if node.right in [0.0, 1.0]:
            return node.right
        else:
            predict(node.right, row)
            

def most_common_label(group):
    if group.size == 0:
        raise Exception("Group is empty")
    labels = group[:, -1]
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
    for i in range(data.shape[0]):
        if data[i, index] < val:
            left.append(data[i,])
        else:
            right.append(data[i,])
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
                labels = np.count_nonzero(labels == val)
                prob = labels / float(len(group))
                current_score += prob**2
            gini_score += (1.0 - current_score)*(float(len(group)/float(total_samples)))
    return gini_score


def random_forest(train_data, n_trees, n_features, sample_ratio, max_depth, min_leaf=5):
    trees = []
    for i in range(n_trees):
        trees.append(build_tree(sample_data(train_data, sample_ratio), max_depth, min_leaf, n_features))
    for tree in trees:
        tree.printTree("")
        print(" ")
    return trees

def final_predict(trees, row):
    predictions = []
    for tree in trees:
        prediction = predict(tree, row)
        if prediction == None:
            prediction = 0
        predictions.append(prediction)
    # np function for counting occurences of each label
    vals, counts = np.unique(predictions, return_counts=True)
    # find the index of the most frequent one
    max_index = np.argmax(counts)
    # return the most frequent value
    return vals[max_index]

def sample_data(dataset, ratio):
  n_sample = round(dataset.shape[0] * ratio)
  rows = np.random.randint(dataset.shape[0], size=n_sample)
  return dataset[rows]

data, num_features = pca(0.8)
split = splitDataset(data, 0.8)
small_train = split[0]

def sample_data(dataset, ratio):
    trainSize = int(len(dataset) * ratio)
    copy = np.copy(dataset)
    np.random.seed(10)
    np.random.shuffle(copy)
    
    # list that stores row indices of only either label 0 or label 1
    label_indices = [np.where(copy[:,-1] == 0), np.where(copy[:,-1] == 1)]
    
    # change the numbers to change class distribution in sample
    zeros = math.floor(0.3 * trainSize)
    ones = math.floor(0.3 * trainSize)
    
    train_indices = np.concatenate((label_indices[0][:zeros], label_indices[1][:ones]), axis=None)
    
    return copy[train_indices]

def data_synthesis(dataset, num_features):
    """ synthesizes data and adds random noise to randomly selected features"""
    copy = np.copy(dataset)
    # np.where returns a tuple, need to use [0] to access actual array of indices
    zero_indices, one_indices = np.where(copy[:, -1] == 0)[0], np.where(copy[:, -1] == 1)[0]
    zero_size = zero_indices.size
    one_size = one_indices.size
    while one_size < zero_size:
        for row in copy[one_indices]:
            if random.randint(0, 1) > 0.5:
                features_to_modify = np.random.randint(0, high=num_features, size=random.randrange(0, num_features, 1))
                row_copy = np.copy(row)
                for feature in features_to_modify:
                    value = row_copy[feature]
                    row_copy[feature] = value + random.uniform(-value / 2, value / 2)
                copy = np.vstack((copy, row_copy))
                one_size += 1
            if one_size > zero_size:
                break
    return copy


data, num_features = pca(0.8)
# data = loadCsv("readmissionTest.csv")
# num_features = data.shape[1]-1
data = data_synthesis(data, num_features)
split = splitDataset(data, 0.8)
small_train = split[0]

forest = random_forest(small_train, 5, num_features, 0.10, 10, min_leaf=5)
small_test = split[1]
predictions = []
for row in small_test:
  predictions.append(final_predict(forest, row))
evaluate(small_test, predictions)
