import os
import pickle
import sys
import csv
import numpy as np

class advise_estimator:

    def __init__(self, experiment_name, fold, delta):
        project_path = f"/mnt/disk023/lcedillo/Lead_Data/{experiment_name}"
        os.makedirs(f"{project_path}/sets/delta_{delta}/", exist_ok=True)

        self.paths = {}
        self.paths["sets"] = f"{project_path}/sets/delta_{delta}/train_fold_{fold}"
        self.paths["estimator"] = f"{project_path}/estimator/estm_fold_{fold}.out"

        self.learn_greedy_sets(fold, delta)

        self.training_results = self.run_advisor("train", fold)
        self.testing_results = self.run_advisor("test", fold)

    def learn_greedy_sets(self, fold, delta): 
        benchmark_fname = f"/mnt/disk023/lcedillo/Lead_Data/1028_paramadvisor_data_transfer/train_VTML200.45.11.42.40-12-{fold}"
        parameter_fname = "/mnt/disk023/lcedillo/Lead_Data/parameters_no_struct.txt"
        accuracies_fname = "/mnt/disk023/lcedillo/Lead_Data/accuracy_combined.tsv"
        k = 25   
        AVERAGE_MAX = True

        # Building benchmarks
        benchmark_map = {}
        f=open(benchmark_fname)
        lines = f.readlines()
        benchmark_map = {}
        benchmark_class_size = np.full(len(lines),len(lines))

        for benchline in lines:
            spl = benchline.strip().split("\t")
            benchmark_map[spl[0]] = len(benchmark_map)
            if(len(spl) > 1):
                benchmark_class_size[benchmark_map[spl[0]]] = int(spl[1])

        # Building parameters list, mapping params to an index value in a dictionary
        parameter_map = {}
        f=open(parameter_fname)
        lines = f.readlines()
        parameter_map = {lines[i].strip():i for i in range(len(lines))}
        parameters = ["" for i in range(len(parameter_map))]
        for key in parameter_map.keys():
            parameters[parameter_map[key]] = key

        # Dinamically size benchmarks and parameters size
        BENCHMARKS = len(benchmark_map)
        PARAMETERS = len(parameter_map)

        # opens and reads accuracy file using csv reader
        ac_file = open(accuracies_fname)
        read_ac = list(csv.reader(ac_file, delimiter = "\t")) 

        # Table of accuracies (matrix) with 861 columns and 16896 rows  PARAMETERS -> COLS  BENCHMARKS -> ROWS
        accuracy = np.full([PARAMETERS, BENCHMARKS], -1.0)

        # building the matrix, putting accuracies in based on dictonaries mapping
        for line in range(len(read_ac)):
            (param, bench) = read_ac[line][0].split("/")
            if param in parameter_map and bench in benchmark_map:
                accuracy[parameter_map[param]][benchmark_map[bench]] = float(read_ac[line][1])

        for i in range(0, PARAMETERS, 1):
            for j in range(0, BENCHMARKS, 1):
                if(accuracy[i][j] == -1):
                    print("Error, accuracy is -1",i,j)
                    sys.exit(2)

        # Creating and populating estimated accuracies matrix
        estimated_file = open(self.paths["estimator"])
        estimator = np.full([PARAMETERS, BENCHMARKS], -1.0)

        for line in estimated_file:
            splitted = line.split("\t")
            (param, bench) = splitted[0].split("/")
            if param in parameter_map and bench in benchmark_map:
                estimator[parameter_map[param]][benchmark_map[bench]] = float(splitted[-1])

        # 1D array of BENCHMARKS size initialized with -1 in all entries
        currently_used = np.full(BENCHMARKS, -1) 
        max_est = np.full(BENCHMARKS, -1.0)
        default_accuracy = np.full(BENCHMARKS, -1.0)

        # 1D array of booleans that represents whether or not the parameters is in set
        in_set = np.full(PARAMETERS, False)

        start_index = 1
        best_default = -1
        best_accuracy = -1
    
        for i in range(0, PARAMETERS, 1):
            temp_acc = 0.0
            for j in range(0, BENCHMARKS, 1):
                temp_acc = temp_acc + accuracy[i][j]
            if(temp_acc > best_accuracy):
                best_accuracy = temp_acc
                best_default = i

        default_parameter_index = best_default
        in_set[default_parameter_index] = True
        parameterSet = [None]*k 
        # Set default (set of size 1)
        parameterSet[0] = default_parameter_index

        for j in range(0, BENCHMARKS):
            default_accuracy[j] = accuracy[best_default][j]
            currently_used[j] = default_parameter_index
            max_est[j] = default_parameter_index
    
        # accuracies_for_estimator = [-1 for _ in range(k)]
        accuracies_for_estimator = np.full(k, -1.0)
        nested_fold_max_est = -1

        # Increasing set size
        for itteration in range(start_index, k):
     
            max_diff = -999999999
            max_index = -1
            max_accuracy = -99999999

            # Using the ones not in set
            for i in range(0, PARAMETERS):
                if(not in_set[i]):
                    temp_advisor_accuracy = 0
                    diff = 0.0
                  
                    for j in range(0, BENCHMARKS):
                        acc_to_use = 0 if(AVERAGE_MAX) else 1
                        acc_ct = 0
                        temp_max_est = -1
                        temp_max_est = i

                        for ip in parameterSet[0:itteration]:
                            if((estimator[ip][j]) > (estimator[temp_max_est][j])):
                                temp_max_est = ip

                        if(AVERAGE_MAX):
                            if(estimator[temp_max_est][j]-delta <= estimator[i][j]):
                                acc_to_use = acc_to_use + accuracy[i][j]
                                acc_ct += 1
                        elif((estimator[temp_max_est][j]-delta) <= (estimator[i][j]) and accuracy[i][j] < acc_to_use):
                            acc_to_use = accuracy[ip][j]

                        for ip in parameterSet[0:itteration]:
                            if(AVERAGE_MAX):
                                if(estimator[temp_max_est][j]-delta <= estimator[ip][j]):
                                    acc_to_use = acc_to_use + accuracy[ip][j]
                                    acc_ct += 1
                            elif(estimator[temp_max_est][j]-delta <= estimator[ip][j] and accuracy[ip][j] < acc_to_use):
                                acc_to_use = accuracy[ip][j]
                        if(AVERAGE_MAX):
                            acc_to_use /= float(acc_ct)
                        temp_advisor_accuracy += acc_to_use/benchmark_class_size[j]
                    if(max_accuracy == -999999999 or temp_advisor_accuracy > max_accuracy):
                        max_accuracy = temp_advisor_accuracy
                        max_index = i
    
            # We found a max one so we put in the set
            in_set[max_index] = True
            parameterSet[itteration] = max_index
    
            advisor_accuracy = 0.0
    
            for j in range(0, BENCHMARKS, 1):
                max_est_loop = -1
                for ip in range(0, PARAMETERS, 1):
                    if(in_set[ip]):
                        if(estimator[ip][j] > max_est_loop):
                            max_est_loop = estimator[ip][j]
                            max_est[j] = ip
                acc_to_use = 0 if(AVERAGE_MAX) else 1
                acc_ct = 0
                for ip in range(0, PARAMETERS, 1):
                    if(in_set[ip]):
                        if(AVERAGE_MAX):
                            if(max_est_loop-delta <= estimator[ip][j]):
                                acc_to_use += accuracy[ip][j]
                                acc_ct+=1
                        elif(max_est_loop-delta <= estimator[ip][j] and accuracy[ip][j] < acc_to_use):
                            acc_to_use = accuracy[ip][j]
                if(AVERAGE_MAX):
                    acc_to_use = acc_to_use / float(acc_ct)
                advisor_accuracy += acc_to_use/benchmark_class_size[j]

            avg = ".avg" if AVERAGE_MAX else ""
            parameters_file = open(self.paths["sets"] + "." + str(itteration+1) + str(".set"), "w")
            for i in range(0, itteration+1):
                parameters_file.write(parameters[parameterSet[i]]+"\n")

    def run_advisor(self, split, fold): 
        benchmarks_fname = f"/mnt/disk023/lcedillo/Lead_Data/1028_paramadvisor_data_transfer/{split}_VTML200.45.11.42.40-12-{fold}"
        accuracies_fname = "/mnt/disk001/dfdeblasio/Greedy_Algo/accuracy_combined.q.pickle"
        accuracy_ftype = "pickle"
        estimator_ftype = "tsv"
       
        # Read the accuracy file, should be a TSV with 3 columns: alignment name, Q score, TC score
        accuracy = {}
        if(accuracy_ftype == "pickle"):
            acc_f = open(accuracies_fname,"rb")
            accuracy = pickle.loads(acc_f.read())
        else:
            print("Accuracy file type not Pickle or TSV.")
            exit(20)

        # Read the estimator file, should be a TSV with 2 columns: alignment name, estimator score
        estimator = {}
        if(estimator_ftype == "tsv"):
            est_f = open(self.paths["estimator"])
            lines = csv.reader(est_f, delimiter="\t")
            for row in lines:
                estimator[row[0]] = float(row[-1])
        elif(estimator_ftype == "pickle"):
            est_f = open(self.paths['estimator'], "rb")
            estimator = pickle.loads(est_f.read())
        else:
            print("Accuracy file type not Pickle or TSV.")
            exit(20)

        # Read the accuracy file, should be a TSV with 2 columns: benchmark name, number of benchmarks in bin
        benchmarks = {}
        benchmarks_f = open(benchmarks_fname)
        lines = csv.reader(benchmarks_f, delimiter="\t")
        for row in lines:
            benchmarks[row[0]] = float(row[1])

        # define results
        advisor_results = np.zeros(24)

        # args.set will contain at least one advisor set file
        # each one will be a plain list of parameter vecotrs, one per line
        for cardinality in range(2, 26):
            set = []
            set_fname = open(self.paths["sets"]+f".{cardinality}.set")
            for row in set_fname:
                set.append(row.strip())
       
            acc_total = 0
            acc_count = 0
            for benchmark in benchmarks.keys():
                max_est = -1
                max_acc = -1
                max_acc_count = 1
                for param in set:
                    # For ties in estimator, get expected accuracy (i.e. average accuracy)
                    if estimator[param+"/"+benchmark] > max_est:
                        max_est = estimator[param+"/"+benchmark]
                        max_acc = accuracy[param+"/"+benchmark]
                        max_acc_count = 1
                    if estimator[param+"/"+benchmark] == max_est:
                        max_acc += accuracy[param+"/"+benchmark]
                        max_acc_count += 1
                if max_acc > -1:
                    acc_total += (1.0/benchmarks[benchmark])*(max_acc/max_acc_count)
                    acc_count += (1.0/benchmarks[benchmark])

            advisor_results[cardinality-2] = (acc_total/acc_count)

        # fold advising results results
        return np.mean(advisor_results, axis=0)