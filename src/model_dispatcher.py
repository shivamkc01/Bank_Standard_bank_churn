from sklearn import tree 
from sklearn import linear_model
from sklearn import ensemble
import xgboost as xgb


models = {
    "lr" : linear_model.LogisticRegression(penalty='l2', C=1.0, solver='liblinear', class_weight={0:1, 1:2}),
    "dt" : tree.DecisionTreeClassifier(max_depth=100, min_samples_split=100, min_samples_leaf=5),
    "rf" : ensemble.RandomForestClassifier(bootstrap=  True, 
                                           max_depth=10,
                                           max_features= 'sqrt', 
                                           min_samples_leaf= 1, 
                                           min_samples_split= 2, 
                                           n_estimators=500),
    "xgb" : xgb.XGBClassifier(n_estimators= 200, 
                              max_depth= 3, 
                              learning_rate= 0.08369736219437933,
                              subsample= 1.0, 
                              colsample_bytree= 0.8, 
                              gamma= 0.0, 
                              reg_alpha=0.1, 
                              reg_lambda= 0.1, 
                              scale_pos_weight=1)
}