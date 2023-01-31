import os, argparse
import numpy as np
import pandas as pd
from sklearn import metrics, neural_network

from project_functions import fun

def find_best_architecture(fold):
    
    best_architecture, best_loss = None, 1
    
    for architecture in range(64):
        path = f"/mnt/disk023/lcedillo/Lead_Data/Search_A/architecture_{architecture}.out"
        search_a = pd.read_csv(path, header=None, names=["fold", "k_fold", "architecture", "batch_size", "loss1", "loss2", "loss3"], sep=r"[\t]", engine="python")
        search_a = search_a.loc[search_a["fold"] == fold]
        
        if len(search_a) < 11: 
            print("Incomplete")
            return None
        
        loss = np.mean([np.mean(search_a["loss1"].values), np.mean(search_a["loss2"].values), np.mean(search_a["loss3"].values)])
            
        if loss < best_loss:
            best_loss = loss
            best_architecture = architecture
    
    return best_architecture

def evaluate_batch_size(x, y, fold, architecture, batch_size, rand):
    
    # training data
    training_data = fun().split_benchmarks("train", fold)
    x_train = x[training_data["indices"]]
    y_train = y[training_data["indices"]]
                    
    # define & train model
    model = neural_network.MLPRegressor(hidden_layer_sizes=architecture, 
                                        activation="relu", 
                                        solver="adam", 
                                        alpha=0.1, 
                                        batch_size=batch_size, 
                                        learning_rate="adaptive", 
                                        learning_rate_init=0.1, 
                                        max_iter=500, 
                                        shuffle=True, 
                                        random_state=rand, 
                                        early_stopping=True, 
                                        validation_fraction=0.35)
    model.fit(x_train, y_train.ravel())
            
    # evaluate
    predictions = model.predict(x_train)
    score = metrics.r2_score(y_train, predictions)
    
    return score

def batch_size_search(fold, architecture):
        
    # search space
    search_space = [(10, 6), (3, 3), (3, 15), (16, 16), (14, 11), (15, 9), (10, 14), (4, 12), (2, 6), (5, 12), (5, 13), (3, 10), (10, 5), (5, 14), (6, 16), (13, 10), (3, 6), (13, 12), (9, 15), (15, 11), (10, 8), (15, 8), (9, 7), (4, 9), (10, 2), (8, 6), (9, 5), (16, 10), (15, 16), (11, 7), (9, 10), (4, 8), (14, 12), (11, 8), (2, 12), (9, 13), (14, 14), (15, 10), (15, 13), (2, 14), (9, 3), (2, 11), (16, 13), (13, 4), (16, 15), (2, 13), (7, 14), (14, 7), (10, 12), (11, 11), (2, 15), (5, 16), (11, 15), (6, 6), (10, 15), (2, 7), (9, 2), (15, 11), (8, 4), (11, 4), (5, 5), (10, 4), (5, 13), (8, 15), (3,4), (5,3)]
    
    # load data
    x = np.load("/mnt/disk023/lcedillo/Lead_Data/facet_features.npy", allow_pickle=True)
    y = np.load("/mnt/disk023/lcedillo/Lead_Data/alignments_accuracy.npy", allow_pickle=True)
    
    # explore all    
    sizes = [32, 64, 128, 256, 512]  
    for batch_size in sizes:
            
        # retreive training evaluation
        score_1731 = evaluate_batch_size(x, y, fold, search_space[architecture], batch_size, 1731)
        score_2001 = evaluate_batch_size(x, y, fold, search_space[architecture], batch_size, 2001)
        score_1371 = evaluate_batch_size(x, y, fold, search_space[architecture], batch_size, 1371)
        print(search_space[architecture], fold, score_1731, score_2001, score_1371, end=" ")
            
        # write results
        write = f"{fold}\t{architecture}\t{batch_size}\t{score_1731}\t{score_2001}\t{score_1371}"
        os.makedirs(f"/mnt/disk023/lcedillo/Lead_Data/Search_B/", exist_ok=True)
        fpath = f"/mnt/disk023/lcedillo/Lead_Data/Search_B/architecture_{architecture}.out"
        with open(fpath, "a") as fl:
            fl.write(write+"\n")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", type=int)
    args = parser.parse_args()
    
    batch_size_search(args.fold, find_best_architecture(args.fold))