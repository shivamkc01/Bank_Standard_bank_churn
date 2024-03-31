from sklearn import tree 
from sklearn import linear_model
from sklearn import ensemble


models = {
    "lr" : linear_model.LogisticRegression(penalty='l2', C=1.0, solver='liblinear', class_weight={0:1, 1:2}),
    "dt" : tree.DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=5),
    "rf" : ensemble.RandomForestClassifier(bootstrap=  True, 
                                           max_depth=10,
                                           max_features= 'sqrt', 
                                           min_samples_leaf= 1, 
                                           min_samples_split= 2, 
                                           n_estimators=500)
}