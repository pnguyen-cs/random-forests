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
        self.groups = groups
    
    
    def split(self, max_depth, min_size, n_features, curr_depth):
        l_group, r_group = self.groups
        print("splitting")
        # Tree should have no notion of the data that was used to
        # produce the tree
        del (self.groups)
        # Check if one of the groups is empty
        if not l_group or not r_group:
            self.left = most_common_label(l_group + r_group)
            self.right = self.left
            # Stop recursion
            return
        if curr_depth >= max_depth:
            self.left, self.right = most_common_label(l_group), most_common_label(r_group)
            return
        # Split left and right group if larger than min size
        if len(l_group) <= min_size:
            self.left = most_common_label(l_group)
        else:
            self.left = Node(l_group).best_split
            split(self.left, max_depth, min_size, n_features, curr_depth + 1)
        if len(l_group) <= min_size:
            self.left = most_common_label(l_group)
        else:
            self.right = Node(r_group).best_split
            split(self.right, max_depth, min_size, n_features, curr_depth + 1)
    

def build_tree(train_data, max_depth, min_size, n_features):
    root = Node(train_data)
    root.split(max_depth, min_size, n_features, 1)
    return root

def predict(node, row):
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
    labels = group[-1,]
    return max([0, 1], key=labels.count())

class decision_tree():
    def __init__(self, num_features, sample_data, class_col, depth, working_rows, working_cols):
        self.depth = depth
        self.num_features = num_features
        self.working_cols = working_cols
        self.working_rows = working_rows
        self.class_col = class_col
        self.sample_data = sample_data
        self.val = max(list(class_col))
        #starting it as big as possible so any new gini score calculated will be accepted
        self.gini_index = float("inf")
        # self.best_split()

    def leaf(self):
        return self.gini_index == float("inf") or self.depth <= 0

    def column_data(self):
        data = []
        for index in self.working_rows:
            data.append(self.sample_data[index,self.feature_index])
        return np.array(data)

    def get_prediction(self, row):
        if self.leaf():
            return self.val
        if row[self.feature_index] > self.feature_split:
            new_tree = self.right
        else:
            new_tree = self.left
        return new_tree.predict(row)

    
    def best_split(self):
        """ should calculate split point with lowest cost 
        input is the portion of the dataset from the training data and the number of features we want
        used on training data once it has been chosen""" 
        #determined by gini score for 0/1 label features and stdev for numerical features
        
        groups_formed = []
        #generating random features to use for split
        #minus two to account for label column
        
        #my_features = list(np.random.choice(len(self.sample_data[0]-2), self.num_features, replace = False))
        #print(my_features)
        #classes are possible final label values
        classes = [0, 1]
        
        #find_categorical_data(small_train)
            
        for i in self.working_cols:

            for r in self.sample_data:

                possible_groups_formed = split_data(self.sample_data, i, r[i])
                #using gini rn but should find a way to determine when to swap with stdev
                #alternativley just use stdev
                
                dev = gini_score(possible_groups_formed, classes)
                #print("gini done")
                if self.gini_index == float("inf") or dev < self.gini_index:
                    self.feature_index = i
                    self.feature_split = r[i]
                    self.gini_index = dev
                    groups_formed = possible_groups_formed
        if self.leaf():
            return None
        #get column data for specific rows
        if groups_formed[0] == groups_formed[1]:
            return None
        split_vals = self.column_data()
        left_tree = np.nonzero(split_vals<=self.feature_split)[0]
        print("Left tree: ", left_tree)
        left_tree_indices = np.random.choice(len(self.sample_data[0]-1), self.num_features, replace = False)
        right_tree = np.nonzero(split_vals>=self.feature_split)[0]
        print("Right tree: ", right_tree)
        right_tree_indices = np.random.choice(len(self.sample_data[0] - 1), self.num_features, replace=False)
        self.left = decision_tree(self.num_features, self.sample_data, self.class_col, self.depth -1, left_tree, left_tree_indices)
        self.right = decision_tree(self.num_features, self.sample_data, self.class_col, self.depth -1, right_tree, right_tree_indices)
        #left_data = list(np.random.choice(len(self.sample_data[0]-2), self.num_features, replace = False))
        return groups_formed
        

        



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
    #print(gini_score)
    return gini_score

#categorical columns based on og dataset without pca changes
#categorical_columns = [51, 52, 54, 56, 58, 59, 60, 61, 65, 68]

def find_categorical_data(small_train):
    #small_train should be a numpy array
    categorical_columns = []
    for i in range(len(small_train[0])):
        data_types = set(small_train[:, i])
        if len(data_types) == 2:
            categorical_columns.append(i)
    return categorical_columns



#def main():
#    print(best_split(data, 4))




# data = loadCsv("readmissionTest.csv")
#should use stratification to split data
np.random.seed(2)
# pca outputs min num of principal components to cover the given variance ratio with the readmission data as the last column
data, num_features = pca(0.7)
split = splitDataset(data, 0.2)
small_train = split[0]   
#use k-folds to get size of each original tree
test_tree = decision_tree(13, data[:100, :-1], data[:100, -1], 10, np.arange(0, 100), np.arange(0, 12))
#main()
