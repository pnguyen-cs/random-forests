from random_forest import build_tree, predict
from readmission import loadCsv, pca, splitDataset, evaluate
import numpy as np

def random_forest(train_data, n_trees, n_features, sample_ratio, max_depth, min_leaf=5):

    trees = []
    for i in range(n_trees):
      trees.append(build_tree(train_data, max_depth, min_leaf, n_features))
    return trees

def final_predict(trees, row):
    # return np.mean([predict(t, row) for t in trees], axis=0)
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

data, num_features = pca(0.7)
split = splitDataset(data, 0.8)
small_train = split[0]

forest = random_forest(small_train, 5, num_features, 0.5, 10)

small_test = split[1]
predictions = []
for row in small_test:
  predictions.append(final_predict(forest, row))
print(predictions)
evaluate(small_test, predictions)