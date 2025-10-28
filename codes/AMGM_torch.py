import torch
import numpy as np
import scipy
import time


def __AMGM_Solver__(gk:np.ndarray, wk:np.ndarray, yk:np.ndarray, vk:np.ndarray, system_to_use:int = 7):
    """Creation and solver of equations to the given set of vectors"""
    
    #Matrix of the system of equations
    matrix = np.array([[np.inner(wk, wk), np.inner(wk, yk), np.inner(wk, vk)], 
                       [np.inner(wk, yk), np.inner(yk, yk), np.inner(yk, vk)], 
                       [np.inner(wk, vk), np.inner(yk, vk), np.inner(vk, vk)]])
    
    #Response vector
    S = np.array([np.inner(gk, wk), np.inner(gk, yk), np.inner(gk, vk)])

    #Solutions of the system (Initializated)
    alpha, beta, gamma = 0, 0, 0

    #Different kind of solution depending on set of vectors used    
    ##Solution of the system usign Steepest gradient decent
    if(system_to_use <= 1):
        alpha = S[0]/matrix[0, 0]
    ##Solution of the system using the vector of momentum 
    elif(system_to_use == 2):
        beta = S[1]/matrix[1, 1]
    ##Solution of the system using the vector of curvature
    elif(system_to_use ==3):
        gamma = S[2]/matrix[2, 2]
    #Solution of the system using the gradient (gk) & momentum (s_{k-1})
    elif(system_to_use == 4):
        alpha, beta = scipy.linalg.solve(matrix[0:2, 0:2], S[0:2], assume_a="sym")
    #Solution of the system using the gradient (gk) & curvature (y_{k-1})
    elif(system_to_use == 5):
        submatrix = np.array([[matrix[0,0], matrix[0, 2]], [matrix[2,0], matrix[2, 2]]])
        subb = np.array([S[0], S[2]])
        alpha, gamma = scipy.linalg.solve(submatrix, subb, assume_a="sym")
    #Solution of the system using the momentum (s_k-1}) & curvauture (y_{k-1})
    elif(system_to_use == 6):
        beta, gamma = scipy.linalg.solve(matrix[1:, 1:], S[1:], assume_a="sym")
    #Solution of the system using the three vectors
    else:
        alpha, beta, gamma = scipy.linalg.solve(matrix, S, assume_a="sym")

    #Convert the sulution into a np.array
    coeficients = np.array([alpha, beta, gamma], dtype=np.float32) 
    return coeficients


def AMGM_method_Torch(x0: np.ndarray, A: np.ndarray, b: np.ndarray, tolerance: float, max_iters: int, device: str = "mps",
                      system_to_use: int = 7, use_device: str = "cpu"):
    """Generalized optimization method to solve the algorithm"""
    # Record the start time
    start_time = time.time()  

    #Convert the given input into a standar datatype
    x0 = x0.astype(np.float32)
    A = A.astype(np.float32)
    b = b.astype(np.float32)

    #Convert the input into a Torch tensor and send them into the GPU
    x0 = torch.from_numpy(x0).to(use_device)
    A = torch.from_numpy(A).to(use_device)
    b = torch.from_numpy(b).to(use_device)

    #Make the first iteration
    gk = A @ x0
    wk = A @ gk
    alpha = torch.inner(gk, wk)/(torch.inner(wk, wk))

    #Creation of the first element of the sequence
    sk = -alpha*gk
    xk = x0 + sk
    yk = -alpha * wk
    gk = gk + yk

    #Itearation process
    for k in range(1, max_iters):
        #If the norm of the function is small enough, we end the process
        gnorm = torch.linalg.vector_norm(gk)
        if (gnorm < tolerance): break

        #Computation of the needed vectors for the eq system.
        wk_old = wk
        wk = A @ gk
        vk = wk-wk_old

        #Compute the coeficients for the linear combination using a given method
        coef = torch.from_numpy(__AMGM_Solver__(gk.cpu().numpy(), wk.cpu().numpy(), yk.cpu().numpy(), vk.cpu().numpy(), system_to_use=system_to_use)).to(device=use_device)

        #Update the point xk
        sk = -coef[0]*gk - coef[2]*yk - coef[1]*sk
        xk = xk + sk

        #Update the gradient value
        yk = -coef[0]*vk - coef[1]*yk - coef[2]*(wk-wk_old);
        gk = gk + yk
 
    #Send back the variables into the CPU and convert them in numpy arrays
    x0 = x0.cpu().numpy()
    A = A.cpu().numpy()
    b = b.cpu().numpy()

    #Record the end time
    execution_time = time.time() - start_time

    #Move the gnorm into the CPU and get the number.
    gnorm = gnorm.cpu().item()

    #Return the number of iteration, the execution time and the norm
    return k+1, execution_time, gnorm
