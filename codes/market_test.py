#Importamos las librerias que provienen de otro directorio
import os
import argparse

# Import the library with the convex quadratic functions problems
from Linear_tools import general_Convex_Quadratic, AMGM_method, linear_CGM

# Import the matrix market library
import ssgetpy

# Import the matrix market read function
from scipy.io import mmread
import numpy as np
import pandas as pd

#Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

#Execution example
#python market_test.py --iters 150000 --info 0 --matNameDir matrices/bcsstk13/bcsstk13.mtx --savedir bcsstk13 
#    <file>      <iterations> <show info> <Directory of the matrix with tht mtx format> <save results directory>
#As an standard rule the savedir name is the name of the matrix.

def download_MM(id_dic: dict, dest_folder = "./matrices"):
    """Download a set of Market Matrices and save them in a given folder.
    
    Note. If the given folder does not exist then it's created at the moment
    """
    
    # List of matrix IDs
    IDS = list(id_dic.keys())

    # Create the folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)  
    print("The matrices are being saved at:", dest_folder)

    # Download each matrix
    for matrix_id in IDS:
        results = ssgetpy.search(name_or_id=matrix_id)
        if results:
            matrix = results[0]  # Get the first result (should be unique for a given ID)
            print(f"Downloading matrix {matrix_id}: {test_matrices[matrix_id]}")
            matrix.download(destpath=dest_folder, extract=True)

    print(f"All matrices downloaded to {dest_folder}")


def load_single_MM(matrix_path: str):
    """
    Load a single .mtx file by its name from the destination folder.

    Parameters:
    - matrix_path (str): The path of the matrix (with the .mtx extension).

    Returns:
    - The loaded sparse matrix, or raise an error if the file does not exist.
    """
    if os.path.exists(matrix_path):
        print(f"Loading matrix in path: {matrix_path}")
        matrix = mmread(matrix_path) # Load the matrix as a sparse matrix 
        return matrix
    else:
        raise ValueError; print(f"Matrix {matrix} not founded")

# Tested matrices
test_matrices = {
    1919: "2Cubes_Sphere", 1580: "af_0_k101", 1581: "af_1_k101", 1582: "af_2_k101",
    1583: "af_3_k101", 1584: "af_4_k101", 1585: "af_5_k101", 942: "af_shell3",
    946: "af_shell7", 1422: "apache1", 1423: "apache2", 35: "bcsstk13", 37: "bcsstk15",
    39: "bcsstk17", 40: "bcsstk18", 45: "bcsstk23", 46: "bcsstk24", 47: "bcsstk25",
    341: "bcsstk36", 343: "bcsstk38", 2544: "Flan_1565", 1625: "mhd4800b", 362: "msc23052",
    760: "nasa4704", 761: "nasasrb", 1605: "s1rmq4m1", 1605: "s2rmq4m1", 1607: "s3rmq4m1",
    1609: "s2rmt3m1", 1610: "s3rmt3m1", 1611: "s3rmt3m3", 1214: "sts4098"
}

if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Run MM test on AMGM.2A|AMGM.3A|CG .')
    parser.add_argument('--iters', type=int, default=1000, help='Maximum number of iterations (default: 1000)')
    parser.add_argument('--info', type=int, default=0, help='Show progress information (default: 0 (False))')
    parser.add_argument('--matNameDir', type=str, default="", help='Directory of the matrix (default: "")')
    parser.add_argument('--dMM', type=bool, default=False, help='Download Matrix Market set of matrices (default: False)')
    parser.add_argument('--savedir', type=str, default="no_name", help='Directory where to save the data frame result (default: 7)')
    
    
    # Parse the arguments
    args = parser.parse_args()

    #Download the the market matrices
    #if args.dMM: download_MM(test_matrices)
    
    #Load the to test matrix
    A = load_single_MM(args.matNameDir)

    # Check if A is not a dense array
    if not isinstance(A, np.ndarray):  
        # Convert sparse matrix to dense
        A = A.toarray()  

    #Creation of the b vector (acording to the original paper)
    x_aux = np.arange(1, A.shape[0]+1, dtype=float)
    b = np.matmul(A, x_aux)
    print(f"B norm: {np.linalg.norm(b)}")

    #Creation of the init point
    x0 = np.ones(A.shape[0], dtype=float)

    #Tolerance (according to the original paper)
    t = 1e-9 
    print(f"Tolerance: {t: 1.8e}")
    
    # Generate the quadratic convex function.
    Q = general_Convex_Quadratic(dimension=A.shape[0], matrix = A, b_vector=b, init_point=x0)

    #Place where to save the results
    results = np.empty(shape=(11), dtype=float)

    #Call the methods (the second entry is the condition number)
    results[0], results[1] = A.shape[0], 0
    results[2:5] = AMGM_method(Q.init_point, Q.matrix, Q.b, t, args.iters, system_to_use=4, show_info=args.info)[0:3]
    results[5:8] = AMGM_method(Q.init_point, Q.matrix, Q.b, t, args.iters, system_to_use=7, show_info=args.info)[0:3]
    results[8:] = linear_CGM(Q.init_point, Q.matrix, Q.b, t, args.iters, show_info=args.info)
    print("Ended process")

    # Header of the df
    #baseline = ["Dimension", "K(A)", "Niters", "ExcTime", "GNorm", "Niters", "ExcTime", "GNorm", "Niters", "ExcTime", "GNorm"]

    #Create DataFrame
    df = pd.DataFrame(results)

    #Write the CSV file with the results.
    df.T.to_csv(f"matrices/results_methods/{args.savedir}.csv", index=False, float_format='%1.5e')
    