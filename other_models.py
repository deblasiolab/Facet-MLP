import argparse
import numpy as np
from sklearn import metrics, linear_model, ensemble, tree

from project_functions import fun

def facet_regression(x, y, fold, model):

    # training data
    training_data = fun().split_benchmarks("train", fold)
    x_train = x[training_data["indices"]]
    y_train = y[training_data["indices"]]
            
    # define & train model
    models = {
        "LR": linear_model.LinearRegression(),
        "RF": ensemble.RandomForestRegressor(criterion="squared_error", max_depth=35, random_state=1731),
        "DT": tree.DecisionTreeRegressor(criterion="squared_error", max_depth=35, random_state=1731)
    }
    models[model].fit(x_train, y_train.ravel())
    
    # testing data
    testing_data = fun().split_benchmarks("test", fold)
    x_test = x[testing_data["indices"]]
    y_test = y[testing_data["indices"]]
            
    # evaluate
    predictions = models[model].predict(x_test)    
    print("train r2", metrics.r2_score(y_train, model.predict(x_train)))
    print("test r2", metrics.r2_score(y_test, predictions))
    
    # run advisor
    estimator = models[model].predict(x)
    fun().save_estimator(model, fold, estimator)
    fun().evaluate_estimator(model, fold)
    
    return 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", type=int)
    parser.add_argument("model", type=str)
    args = parser.parse_args()
    
    # load data
    x = np.load("/mnt/disk023/lcedillo/Facet-MLP/Data/facet_features.npy", allow_pickle=True)
    y = np.load("/mnt/disk023/lcedillo/Facet-MLP/Data/alignments_accuracy.npy", allow_pickle=True)
    
    facet_regression(x, y, args.fold, args.model)
    
    