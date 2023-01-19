import argparse
import numpy as np
import pandas as pd
from sklearn import metrics, neural_network

from project_functions import fun

def find_best_architecture(fold):
    
    best_architecture, best_loss = None, 1
    
    for architecture in range(64):
        path = f"/mnt/disk023/lcedillo/Facet-MLP/Search_A/architecture_{architecture}.out"
        search_a = pd.read_csv(path, header=None, names=["fold", "k_fold", "top", "batch_size", "loss1", "loss2", "loss3"], sep=r"[\t]", engine="python")
        search_a = search_a.loc[search_a["fold"] == fold]
        
        if len(search_a) < 11: 
            print("Incomplete")
            return None
        
        loss = np.mean([np.mean(search_a["loss1"].values), np.mean(search_a["loss2"].values), np.mean(search_a["loss3"].values)])
            
        if loss < best_loss:
            best_loss = loss
            best_architecture = architecture
    
    return best_architecture

def find_best_batch_size(fold, architecture):
    
    path = f"/mnt/disk023/lcedillo/Facet-MLP/Search_B/architecture_{architecture}.out"
    search_b = pd.read_csv(path, header=None, names=["fold", "architecture", "batch_size", "1731", "2001", "1371", "test_1", "test_2", "test_3"], sep=r"[\t]", engine="python")
    search_b = search_b.loc[search_b["fold"] == fold]
    
    best_scores = {"1731": 0, "2001": 0, "1371": 0}
    best_batch_sizes = {"1731": 0, "2001": 0, "1371": 0}
    
    best_scores["1731"] = search_b["1731"].max(axis=0)
    best_batch_sizes["1731"] = search_b["batch_size"][search_b["1731"].idxmax(axis=0)]
    
    best_scores["2001"] = search_b["2001"].max(axis=0)
    best_batch_sizes["2001"] = search_b["batch_size"][search_b["2001"].idxmax(axis=0)]
    
    best_scores["1371"] = search_b["1371"].max(axis=0)
    best_batch_sizes["1371"] = search_b["batch_size"][search_b["1371"].idxmax(axis=0)]
    
    best_seed = max(best_scores, key=best_scores.get)
    
    return best_batch_sizes[best_seed], int(best_seed)

def facet_mlp(x, y, fold, best_architecture, best_batch_size, best_seed):

    # training data
    training_data = fun().split_benchmarks("train", fold)
    x_train = x[training_data["indices"]]
    y_train = y[training_data["indices"]]
            
    # define & train model
    model = neural_network.MLPRegressor(hidden_layer_sizes=best_architecture, 
                                        activation="relu", 
                                        solver="adam", 
                                        alpha=0.1,                          
                                        batch_size=best_batch_size,             
                                        learning_rate="adaptive", 
                                        learning_rate_init=0.1,             
                                        max_iter=500, 
                                        shuffle=True, 
                                        random_state=best_seed, 
                                        early_stopping=True, 
                                        validation_fraction=0.35)       
    model.fit(x_train, y_train.ravel())
    
    # testing data
    testing_data = fun().split_benchmarks("test", fold)
    x_test = x[testing_data["indices"]]
    y_test = y[testing_data["indices"]]
            
    # evaluate 
    print("train r2", metrics.r2_score(y_train, model.predict(x_train)))
    print("test r2", metrics.r2_score(y_test, model.predict(x_test)))
    
    # run advisor
    estimator = model.predict(x)
    fun().save_estimator("MLP", fold, estimator)
    fun().evaluate_estimator("MLP", fold)
    
    return 
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", type=int)
    args = parser.parse_args()
        
    # load data
    x = np.load("/mnt/disk023/lcedillo/Facet-MLP/Data/facet_features.npy", allow_pickle=True)
    y = np.load("/mnt/disk023/lcedillo/Facet-MLP/Data/alignments_accuracy.npy", allow_pickle=True)
    
    # search space
    search_space = [(10, 6), (3, 3), (3, 15), (16, 16), (14, 11), (15, 9), (10, 14), (4, 12), (2, 6), (5, 12), (5, 13), (3, 10), (10, 5), (5, 14), (6, 16), (13, 10), (3, 6), (13, 12), (9, 15), (15, 11), (10, 8), (15, 8), (9, 7), (4, 9), (10, 2), (8, 6), (9, 5), (16, 10), (15, 16), (11, 7), (9, 10), (4, 8), (14, 12), (11, 8), (2, 12), (9, 13), (14, 14), (15, 10), (15, 13), (2, 14), (9, 3), (2, 11), (16, 13), (13, 4), (16, 15), (2, 13), (7, 14), (14, 7), (10, 12), (11, 11), (2, 15), (5, 16), (11, 15), (6, 6), (10, 15), (2, 7), (9, 2), (15, 11), (8, 4), (11, 4), (5, 5), (10, 4), (5, 13), (8, 15), (3,4), (5,3)]
    
    best_architecture = find_best_architecture(args.fold)
    best_batch_size, best_seed = find_best_batch_size(args.fold, best_architecture)
    facet_mlp(x, y, args.fold, search_space[best_architecture], best_batch_size, best_seed)