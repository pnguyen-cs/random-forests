from random_forest.py import build_tree, predict
from readmission.py import loadCsv, pca

def random_forest(train_data, n_trees, n_features, sample_ratio, max_depth, min_leaf=5):
    np.random.seed(13)
    trees = []
    for i in range(n_trees):
      trees.append(build_tree(train_data, max_depth, min_leaf, n_features))

def final_predict(trees, row):
    return np.mean([predict(t, row) for t in trees], axis=0)



def sample_data(dataset, ratio):
  n_sample = round(dataset.shape[0] * ratio)
  rows = np.random.randint(dataset.shape[0], size=n_sample)
  return dataset[rows]

data, num_features = pca(0.7)
split = splitDataset(data, 0.2)
small_train = split[0]

forest = random_forest(small_train, 1, num_features, 0.25, 10)

