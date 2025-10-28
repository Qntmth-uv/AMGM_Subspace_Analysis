#Standard libraries
import numpy as np
import os
import time
from datetime import datetime
import pandas as pd
import argparse

# Example of usage
# program_directory --info 1 --savedir dist1_test_30 --sys 0 --dist 1 --nexp 30 --conf 2 --tol 1e-12

# Custom libraries
import Linear_tools as lt

#Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# Execution time
exc_date_str = datetime.now().strftime("%d%m%Y_%H%M")

# Get the directory of the current file
parent_dir = os.path.dirname(os.path.abspath(__file__))

# Experiment parameters (Diagonal type 1,2  & Dense type 1, 2)
condition_numbers = [100, 1000, 10000, 100000]
dimensions = [100, 1000, 10000]

# Experiment parameters (Specific for dense type 3)
condition_numbers_exponential = [5, 10, 15]
dimensions_exponential = [1000, 5000, 10000]

# Experiment parameters (Specific for test the exc algorithm)
# test_condition_numbers = [100, 1000, 10000, 100000]
# test_dimensions = [10, 100, 150, 200, 300]

#Set the seeds (randomtly generated, to reproduce the experiments)
seeds = [1832465322, 491551257, 165801770, 538562692, 1188286841, 194181317, 411101327, 738070179, 85041792, 357109069,
            707019774, 1179433513, 1241599407, 1726030522, 1853137160, 708250218, 1792206966, 739902302, 188114852, 1407803070, 
            570173993, 113434413, 1448529842, 1669513936, 1466364963, 1738574529, 1566692701, 741001325, 1012659799, 1832919952]

# Arguments of the experiments (for multiple runs)
parser = argparse.ArgumentParser(description='Execution of the experiments for the AMGM method.')
parser.add_argument('--iters', type=int, default=10000, help='Maximum number of iterations (default: 1000)')
parser.add_argument('--dist', type=int, default=1, help='Type of distribution (default: 1 (diagonal type 1))')
parser.add_argument('--info', type=int, default=0, help='Show progress information (default: 0 (False))')
parser.add_argument('--sys', type=int, default=7, help='System of equations being used (default: 7)')
parser.add_argument('--savedir', type=str, default=f"experiments", help='Directory where to save the data frame result (default: "no_name")')
parser.add_argument('--tol', type=float, default=1e-8, help='Tolerance for the stopping criterion (default: 1e-8)')
parser.add_argument('--nexp', type=int, default=10, help='Number of experiments to cover (default: 10)')
parser.add_argument('--conf', type=int, default=1, help='Configuration of the set of problems (default: 1 = [0:8])')

def unitary_test(dimension, condition_number, distribution: int, system_to_use: int, seeds,
                 show_info: bool, max_iters: int, directory_name: str, tolerance: float = 1e-8):
    """Execution of a set of experiments for a array of dimensions, and an array of condition numbers, and a specfic
    distribution of matrices, also a fixed system of equations (subspace algorithm), an array of seed for the reproducibilty of the experiments.
    Show info shows the evolution of the results on each problem, but not the evolution on the optimization process.
    Max_iters is the maximum numbers of iterations that we allow to execute the optimization process.
    Directory_name is the name of a directory being created or if it exist, then save the results of the set of experiments
    Toleranc is the minimum norm of the gradient that we allow.
    """
    # Total number of experiments
    total_experiments = len(dimension) * len(condition_number)
    
    # Write the results directory's
    write_resutls_directory = os.path.join(parent_dir, f"results_csv/{directory_name}/")
    write_std_directory = os.path.join(write_resutls_directory, f"std/")
    write_rawData_directory = os.path.join(write_resutls_directory, f"raw_data/")

    # Creation of the directorys
    os.makedirs(write_resutls_directory, exist_ok=True)
    os.makedirs(write_std_directory, exist_ok=True)
    os.makedirs(write_rawData_directory, exist_ok=True)

    # Matrix to save the results
    results = np.empty(shape=(total_experiments, 5))
    results_std = np.empty(shape=(total_experiments, 3))

    #Counter 
    counter = 0

    # Mu values
    mu_global = []

    #Move over the dimensions
    for dim in dimension:
        #Move over the condition numbers
        for cond in condition_number:
            if show_info:
                print(f"Progress: Dim={dim}, Cond={cond}, Exp={counter+1}/{total_experiments}")

            # Matrix to save the local results
            local_results = np.empty(shape=(len(seeds), 3))

            #Move over the seeds
            for it, seed in enumerate(seeds):
                # Lap-time
                start_time = time.time()

                # Quadratic function to use
                general_cuadratic = lt.general_Convex_Quadratic(dim, random_generator_seed=seed)

                ## Diagonal category
                if distribution <= 1:
                    general_cuadratic.diagonal_notDense_type1(cond)    #Diagonal Type 1
                elif distribution == 2:
                    general_cuadratic.diagonal_notDense_type2(cond)    #Diagonal Type 1

                ## Dense category
                elif distribution == 3:
                    general_cuadratic.dense_matrix_type1(cond)         # Dense Type 1
                elif distribution == 4:
                    general_cuadratic.dense_matrix_type2(cond)         # Dense Type 2
                elif distribution == 5:
                    general_cuadratic.dense_matrix_type3(cond)         # Dense Type 3

                # Method to use
                if system_to_use in [1, 2, 3, 4, 5, 6, 7]:
                    # Método AMGM con i=system_to_use
                    results_amgm = lt.AMGM_method(general_cuadratic.init_point,
                                                  general_cuadratic.matrix,
                                                  general_cuadratic.b,
                                                  tolerance,
                                                  max_iters,
                                                  system_to_use=system_to_use,
                                                  show_info=False)
                    local_results[it, 0:3], mu = results_amgm[0:3], results_amgm[-1]
                    mu_global.extend(mu)
                elif system_to_use == 0:
                    # Método del gradiente conjugado
                    local_results[it, 0:3] = lt.linear_CGM(general_cuadratic.init_point,
                                                           general_cuadratic.matrix,
                                                           general_cuadratic.b,
                                                           tolerance,
                                                           max_iters)

                # End of execution
                execution_time = time.time() - start_time

            # Get current date in DDMMYYYY format
            date_str_raw = datetime.now().strftime("%d%m%Y_%H%M")

            # Write the results
            results[counter, 0], results[counter, 1] = dim, cond
            results[counter, 2:] = np.mean(local_results, axis=0)
            results_std[counter, :] = np.std(local_results, axis=0)

            # Write the results in a csv file
            ## Baseline for column names
            baseline = ["Dimension", "K(A)", "Niters", "ExcTime", "GNorm"]
            baseline_rawdata = ["Niters", "ExcTime", "GNorm"]
            df_raw_data = pd.DataFrame(local_results, columns=baseline_rawdata)

            # Write the raw result for the present 30 experiments
            results_raw_directory = write_rawData_directory + f"raw_{dim}d_{cond}k_results_sys_{sys}_type_{distribution}_{date_str_raw}.csv"
            df_raw_data.to_csv(results_raw_directory, index=False, float_format='%1.8e')

            # Increment the counter
            counter += 1

    # Baseline of std dataframe
    baseline_std = ["Niters std", "ExcTime std", "GNorm std"]

    # Create DataFrame
    df = pd.DataFrame(results, columns=baseline)
    df_sdt = pd.DataFrame(results_std, columns=baseline_std)
    df_mu = pd.DataFrame(mu_global)

    # Get current date in DDMMYYYY format
    date_str_end = datetime.now().strftime("%d%m%Y_%H%M")

    # Set the directory to save the files
    result_write_directory = write_resutls_directory + f"results_sys_{system_to_use}_type_{distribution}_{date_str_end}.csv"
    results_std_directory = write_std_directory + f"std_results_sys_{system_to_use}_type_{distribution}_{date_str_end}.csv"
    results_mu_directory = write_resutls_directory + f"mu_results_sys_{system_to_use}_type_{distribution}_{date_str_end}.csv"
    
    # Write the dataframes
    df.to_csv(result_write_directory, index=False, float_format='%1.4e')
    df_sdt.to_csv(results_std_directory, index=False, float_format='%1.4e')
    df_mu.to_csv(results_mu_directory, index=True, float_format='%1.6e')
    
    # Print the info if is need it
    if show_info: 
        print(f"Results directory.   : {result_write_directory}")
        print(f"Results std directory: {results_std_directory}")


def set_of_test(subspaces, dimension, condition_number, distribution: int, seeds,
                 show_info: bool, max_iters: int, directory_name: str, tolerance: float = 1e-8):
    """Execute a  set of experiments for a array of subspaces (optimzation methods)"""
    for method in subspaces:
        unitary_test(dimension, condition_number, distribution, method, seeds, show_info, max_iters, directory_name, tolerance)
        print(f"Ended set of experiments: {method}")

def main():
    """Main function to run the experiments."""
    args = parser.parse_args()
    
    print(f"Running experiments with:")
    print(f"|  Max iterations: {args.iters}")
    print(f"|  Show info: {bool(args.info)}")
    print(f"|  System type: {args.sys}")
    print(f"|  Save directory: {args.savedir}")
    print(f"|  Distribution type: {args.dist}")
    print(f"|  Tolerance: {args.tol}")
    print(f"|  Directory: {parent_dir}")
    print(f"|  Num of exp: {args.nexp}")
    print(f"|  Configuration: {args.conf}")

    #Execution of only one experimient
    #unitary_test(dimensions, test_condition_numbers, args.dist, args.sys, seeds[:args.nexp], bool(args.info), args.iters, args.savedir, args.tol)

    #Execution of a set of experimetns of different methods
    #Run a set of experiments w/o cprofile using a specific subespaces (default 0-7, else 0,4,7)
    #Here the 0 is the Conjugate gradient method, 4 is the bidimensional subespacie generated by the gradient and momentum
    #The subspace 7 is exactly the method given by Harry Oviedo. 
    if args.conf == 1: conf = [0, 4, 7]
    else: conf = [4, 5, 6, 7]
    set_of_test(conf, dimensions_exponential, condition_numbers_exponential, args.dist, seeds[:args.nexp], bool(args.info), args.iters, args.savedir, args.tol)

if __name__ == "__main__": main()