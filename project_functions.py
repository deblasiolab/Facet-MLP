import os
import pandas as pd

from advisor import advise_estimator

class fun:
    
    def split_benchmarks(self, split, fold, k_fold=None):
        
        if k_fold == None:
            
            benchmarks_path = f"/mnt/disk023/lcedillo/Lead_Data/1028_paramadvisor_data_transfer/{split}_VTML200.45.11.42.40-12-{fold}"
            benchmarks = pd.read_csv(benchmarks_path, header=None, sep=r"[\t]", engine="python", names=["benchmarks", "count"]).drop(["count"], axis=1)
            parameters_benchmarks = pd.read_pickle("/mnt/disk023/lcedillo/Lead_Data/parameters_benchmarks.pkl")

            return pd.merge(parameters_benchmarks, benchmarks)
            
        if split == "train":
            
            train_bench = f"/mnt/disk023/lcedillo/Lead_Data/1028_paramadvisor_data_transfer/train_VTML200.45.11.42.40-12-{fold}"
            train_bench = pd.read_csv(train_bench, header=None, sep=r"[\t]", engine="python", names=["benchmarks", "count"]).drop(["count"], axis=1)
        
            test_bench = f"/mnt/disk023/lcedillo/Lead_Data/1028_paramadvisor_data_transfer/test_VTML200.45.11.42.40-12-{k_fold}"
            test_bench = pd.read_csv(test_bench, header=None, sep=r"[\t]", engine="python", names=["benchmarks", "count"]).drop(["count"], axis=1)

            benchmarks = pd.concat([train_bench, test_bench]).drop_duplicates(keep=False)
            
            parameters_benchmarks = pd.read_pickle("/mnt/disk023/lcedillo/Lead_Data/parameters_benchmarks.pkl")

            return pd.merge(parameters_benchmarks, benchmarks)
        
        if split == "val":
            
            benchmarks_path = f"/mnt/disk023/lcedillo/Lead_Data/1028_paramadvisor_data_transfer/test_VTML200.45.11.42.40-12-{k_fold}"
            benchmarks = pd.read_csv(benchmarks_path, header=None, sep=r"[\t]", engine="python", names=["benchmarks", "count"]).drop(["count"], axis=1)
            parameters_benchmarks = pd.read_pickle("/mnt/disk023/lcedillo/Lead_Data/parameters_benchmarks.pkl")

            return pd.merge(parameters_benchmarks, benchmarks)       
    
    def save_estimator(self, experiment_name, fold, predictions):
        
        # estimator path
        project_path = f"/mnt/disk023/lcedillo/Lead_Data/{experiment_name}"
        os.makedirs(f"{project_path}/estimator/", exist_ok=True)

        # prepare predictions
        predictions = predictions.reshape(-1,)
        parameters_benchmarks = pd.read_pickle("/mnt/disk023/lcedillo/Lead_Data/parameters_benchmarks.pkl")

        # estimator file
        estimator_path = f"{project_path}/estimator/estm_fold_{fold}.out"
        estimator = [str(parameters_benchmarks["parameters"][i])+"/"+str(parameters_benchmarks["benchmarks"][i])+"\t"+str(predictions[i]) for i in range(len(parameters_benchmarks))]
        
        # save estimator
        with open(estimator_path, "w") as fl:
            for line in estimator:
                fl.write(line+"\n")
   
    def evaluate_estimator(self, experiment_name, fold, delta=0.0, verbose=1):

        # results path
        project_path = f"/mnt/disk023/lcedillo/Lead_Data/{experiment_name}"
        os.makedirs(f"{project_path}/advisor_results/", exist_ok=True)

        # run advisor
        results = advise_estimator(experiment_name, fold, delta)
        save_results = f"{experiment_name} {fold} {results.training_results} {results.testing_results}"
        
        # print results
        if verbose != 0: print(save_results)

        # save results
        save_results_path = f"{project_path}/advisor_results/res_fold_{fold}.out"
        with open(save_results_path , "a") as fl: 
            fl.write(save_results+"\n")

        return float(results.training_results), float(results.testing_results)