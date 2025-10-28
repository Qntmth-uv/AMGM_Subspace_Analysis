import torch
import numpy as np
import scipy
import time


def __AMGM_Solver__(gk:np.ndarray, wk:np.ndarray, yk:np.ndarray, vk:np.ndarray, system_to_use:int = 7):
    """Solver de coeficientes para distintos tipos de sistemas."""
    
    #Matriz que respresneta el sistema
    matrix = np.array([[np.inner(wk, wk), np.inner(wk, yk), np.inner(wk, vk)], 
                       [np.inner(wk, yk), np.inner(yk, yk), np.inner(yk, vk)], 
                       [np.inner(wk, vk), np.inner(yk, vk), np.inner(vk, vk)]])
    
    #Vector de respuesta
    S = np.array([np.inner(gk, wk), np.inner(gk, yk), np.inner(gk, vk)])

    #Soluciones del sistema
    alpha, beta, gamma = 0, 0, 0

    #Distintos tipos de soluciones
    if(system_to_use <= 1):
        alpha = S[0]/matrix[0, 0]
    elif(system_to_use == 2):
        beta = S[1]/matrix[1, 1]
    elif(system_to_use ==3):
        gamma = S[2]/matrix[2, 2]
    
    #Espacio generado por gk y s_{k-1}
    elif(system_to_use == 4):
        alpha, beta = scipy.linalg.solve(matrix[0:2, 0:2], S[0:2], assume_a="sym")
    elif(system_to_use == 5):
        submatrix = np.array([[matrix[0,0], matrix[0, 2]], [matrix[2,0], matrix[2, 2]]])
        subb = np.array([S[0], S[2]])
        alpha, gamma = scipy.linalg.solve(submatrix, subb, assume_a="sym")
    elif(system_to_use == 6):
        beta, gamma = scipy.linalg.solve(matrix[1:, 1:], S[1:], assume_a="sym")
    #Espacio generado por la terna de QuasiNewton
    else:
        alpha, beta, gamma = scipy.linalg.solve(matrix, S, assume_a="sym")

    coeficients = np.array([alpha, beta, gamma], dtype=np.float32) 
    return coeficients


def AMGM_method_Torch(x0: np.ndarray, A: np.ndarray, b: np.ndarray, tolerance: float, max_iters: int, device: str = "mps",
                      system_to_use: int = 7, use_device: str = "cpu"):
    
    #Tiempo de inicio
    start_time = time.time()  # Record the start time

    #Convertimos el tipo de dato
    x0 = x0.astype(np.float32)
    A = A.astype(np.float32)
    b = b.astype(np.float32)

    #Pasamos a formato tensor los arrays para operarlos con la GPU
    x0 = torch.from_numpy(x0).to(use_device)
    A = torch.from_numpy(A).to(use_device)
    b = torch.from_numpy(b).to(use_device)

    #Realizamos la primera iteración
    gk = A @ x0
    wk = A @ gk
    alpha = torch.inner(gk, wk)/(torch.inner(wk, wk))

    #Creamos el primer elemento de la secuencia
    sk = -alpha*gk
    xk = x0 + sk
    yk = -alpha * wk
    gk = gk + yk

    for k in range(1, max_iters):
        gnorm = torch.linalg.vector_norm(gk)
        if (gnorm < tolerance): break

        #Calculamos los valores correspondientes
        wk_old = wk
        wk = A @ gk
        vk = wk-wk_old

        #Calculamos la terna de valores
        coef = torch.from_numpy(__AMGM_Solver__(gk.cpu().numpy(), wk.cpu().numpy(), yk.cpu().numpy(), vk.cpu().numpy(), system_to_use=system_to_use)).to(device=use_device)

        #Actualizamos el punto
        sk = -coef[0]*gk - coef[2]*yk - coef[1]*sk
        xk = xk + sk

        #Actualizamos el gradiente
        yk = -coef[0]*vk - coef[1]*yk - coef[2]*(wk-wk_old);
        gk = gk + yk
 
    #Regresamos las variables a la cpu
    x0 = x0.cpu().numpy()
    A = A.cpu().numpy()
    b = b.cpu().numpy()

    #Tiempo que en ejecutarse
    execution_time = time.time() - start_time

    #Pasamos la norma a la CPU y convertimos en numpy array
    gnorm = gnorm.cpu().numpy()

    return k, execution_time, gnorm
