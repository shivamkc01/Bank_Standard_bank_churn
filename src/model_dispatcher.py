from sklearn import tree 
from sklearn import linear_model


models = {
    "lr" : linear_model.LogisticRegression(penalty='l2', C=1.0, solver='liblinear', class_weight={0:1, 1:2}),
    "dt" : tree.DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=5)
}