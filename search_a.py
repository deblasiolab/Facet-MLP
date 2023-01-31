import os, argparse
import numpy as np
from sklearn import metrics, neural_network

from project_functions import fun

def evaluate_architecture(x, y, fold, k_fold, architecture, rand):
    
    # training data
    training_data = fun().split_benchmarks("train", fold, k_fold)
    x_train = x[training_data["indices"]]
    y_train = y[training_data["indices"]]
                
    # define & train model
    model = neural_network.MLPRegressor(hidden_layer_sizes=architecture, 
                                        activation="relu", 
                                        solver="adam", 
                                        alpha=0.01, 
                                        batch_size=256, 
                                        learning_rate="adaptive", 
                                        learning_rate_init=0.1, 
                                        max_iter=500, 
                                        shuffle=True, 
                                        random_state=rand, 
                                        early_stopping=True)
    model.fit(x_train, y_train.ravel())
    
    # validation data
    validation_data = fun().split_benchmarks("val", fold, k_fold)
    x_val = x[validation_data["indices"]]
    y_val = y[validation_data["indices"]]
        
    # continous target to discrete
    y_val_binary = np.zeros(y_val.shape)
    y_val_binary[y_val > 0.5] = 1
    
    # evaluate
    predictions = model.predict(x_val)
    predictions = np.clip(predictions, 0, 1)
    loss = metrics.mean_squared_error(y_val_binary, predictions)
    
    return loss

def architecture_search(fold, k_fold):
        
    # search space
    search_space = [(10, 6), (3, 3), (3, 15), (16, 16), (14, 11), (15, 9), (10, 14), (4, 12), (2, 6), (5, 12), (5, 13), (3, 10), (10, 5), (5, 14), (6, 16), (13, 10), (3, 6), (13, 12), (9, 15), (15, 11), (10, 8), (15, 8), (9, 7), (4, 9), (10, 2), (8, 6), (9, 5), (16, 10), (15, 16), (11, 7), (9, 10), (4, 8), (14, 12), (11, 8), (2, 12), (9, 13), (14, 14), (15, 10), (15, 13), (2, 14), (9, 3), (2, 11), (16, 13), (13, 4), (16, 15), (2, 13), (7, 14), (14, 7), (10, 12), (11, 11), (2, 15), (5, 16), (11, 15), (6, 6), (10, 15), (2, 7), (9, 2), (15, 11), (8, 4), (11, 4), (5, 5), (10, 4), (5, 13), (8, 15), (3,4), (5,3)]
    
    # load data
    x = np.load("/mnt/disk023/lcedillo/Lead_Data/facet_features.npy", allow_pickle=True)
    y = np.load("/mnt/disk023/lcedillo/Lead_Data/alignments_accuracy.npy", allow_pickle=True)
        
    # explore all    
    for architecture in range(len(search_space)):
            
        # retreive validation evaluation
        loss_1731 = evaluate_architecture(x, y, fold, k_fold, search_space[architecture], 1731)
        loss_2001 = evaluate_architecture(x, y, fold, k_fold, search_space[architecture], 2001)
        loss_1371 = evaluate_architecture(x, y, fold, k_fold, search_space[architecture], 1371)
        print(search_space[architecture], fold, k_fold, loss_1731, loss_2001, loss_1371, end=" ")
            
        # write results
        write = f"{fold}\t{k_fold}\t{architecture}\t{256}\t{loss_1731}\t{loss_2001}\t{loss_1371}"
        os.makedirs(f"/mnt/disk023/lcedillo/Lead_Data/Search_A/", exist_ok=True)
        fpath = f"/mnt/disk023/lcedillo/Lead_Data/Search_A/architecture_{architecture}.out"
        with open(fpath, "a") as fl:
            fl.write(write+"\n")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", type=int)
    parser.add_argument("k_fold", type=int)
    args = parser.parse_args()
    
    architecture_search(args.fold, args.k_fold)