import numpy as np
import scipy
import time

#Fucnión cuadratica
quadratic_function = lambda x, A, b: 0.5*(x.T @ (A @ x)) - np.inner(x, b)

class general_Convex_Quadratic:
    def __init__(self, dimension: int, matrix: np.ndarray = None, b_vector: np.ndarray = None, 
                random_generator_seed: int = None, condition_number:float = None, init_point:np.ndarray=None):
 
        # Dimension and generator of matrices
        self.dimension = dimension
        self.rnd = np.random.default_rng() if random_generator_seed is None else np.random.default_rng(random_generator_seed)
        self.init_point = init_point
        self.condition_number = None
        
        #Si no se da una función entonces creamos una matriz aleatoria (Convexa)
        if matrix is None:
                #Creamos una matriz aleatoria
                m =  self.rnd.random((dimension,dimension))

                #Multiplicamos por su transpuesta para hacerla definida positiva
                self.matrix = np.matmul(m.T, m)
                
                #Guardamos el numero de condición
                self.condition_number = condition_number
        else:
            self.matrix = matrix
            self.condition_number = condition_number

        #Si no se da ningun vector, entonces lo tomamos como el vector de zeros
        self.b = np.zeros(self.dimension) if b_vector is None else b_vector

        #If is not given a vector, the we take the vector zero
        self.init_point = np.zeros(self.dimension) if init_point is None else init_point

        print(f"The configuration for the quadratic convex is:\nDimensions - A:{self.matrix.shape} | B:{self.b.shape} | x0:{self.init_point.shape}")

    def function(self, x:np.ndarray)->float:
        """Función cuadratica general"""
        return 0.5 * np.matmul(x.T, np.matmul(self.matrix, x)) - np.inner(self.b, x)

    def gradient(self, x:np.ndarray)->np.ndarray:
        """Gradiente de la función cuadratica general"""
        return np.matmul(self.matrix, x)- self.b
    
    def hessian(self, x:np.ndarray)->np.ndarray:
        """Hessiana de la función cuadratica"""
        return self.matrix
    
    def diagonal_notDense_type1(self, cond_number: float):
        #Creamos una matriz diagonal para rellenar
        diagonal_matrix = np.zeros((self.dimension, self.dimension), dtype=float)

        #Creamos el espectro de la matriz
        self.spec = self.rnd.uniform(1, cond_number, self.dimension)
        self.spec[0], self.spec[-1] = cond_number, 1

        #Rellenamos la matriz
        np.fill_diagonal(diagonal_matrix, self.spec)

        #Intercambiamos con la matriz que deseamos y actualizamos el valor del numero de condición
        self.matrix = diagonal_matrix
        self.condition_number = cond_number

        #Establecemos el valor de b de acuerdo a la forma que fue tomada
        self.b = self.rnd.uniform(-10, 10, self.dimension)
        self.init_point = self.rnd.uniform(-5, 5, self.dimension)

    def diagonal_notDense_type2(self, cond_number: float):

        #Creamos una matriz diagonal para rellenar
        diagonal_matrix = np.zeros((self.dimension, self.dimension), dtype=float)

        #Creamos el espectro de la matriz
        tem = [10]*self.dimension
        self.spec = self.rnd.uniform(1, np.log10(cond_number), self.dimension)
        self.spec = np.power(tem, self.spec)
        self.spec[0], self.spec[-1] = cond_number, 1

        #Rellenamos la matriz
        np.fill_diagonal(diagonal_matrix, self.spec)

        #Intercambiamos con la matriz que deseamos y actualizamos el valor del numero de condición
        self.matrix = diagonal_matrix
        self.condition_number = cond_number

        #Establecemos el valor de b de acuerdo a la forma que fue tomada
        self.b = np.matmul(self.matrix, self.rnd.random(self.dimension))
        self.init_point = np.zeros(self.dimension)


    def dense_matrix_type1(self, cond_number: float):
        """"This is the correct matrix distribution for Harry Ovideo Dense type 1"""
        # Descomponemos la matriz original en parte ortogonal y triangular
        q, _ = np.linalg.qr(self.matrix)

        # Creamos una matriz diagonal para rellenar
        diagonal_matrix = np.zeros((self.dimension, self.dimension), dtype=float)

        # Creamos el espectro de la matriz
        self.spec = np.empty(self.dimension)
        self.spec[0], self.spec[-1] = cond_number, 1.0

        # Create the eigenvalues for the distribution
        for i in range(1, self.dimension-1):
            self.spec[i] = np.power(cond_number, (self.dimension-i)/(self.dimension - 1))

        # Rellenamos la matriz
        np.fill_diagonal(diagonal_matrix, self.spec)

        # Intercambiamos con la matriz que deseamos y actualizamos el valor del numero de condición
        self.matrix = np.matmul(q, np.matmul(diagonal_matrix, q.T))
        self.condition_number = cond_number

        # Establecemos el valor de b de acuerdo a la forma que fue tomada
        self.b = np.matmul(self.matrix, self.rnd.random(self.dimension), dtype=float)
        self.init_point = np.zeros(self.dimension)


    def dense_matrix_type2(self, cond_number: float, sort_eigenval:bool = False):
        #Descomponemos la matriz original en parte ortogonal y triangular
        q, r = np.linalg.qr(self.matrix)

        #Creamos una matriz diagonal para rellenar
        diagonal_matrix = np.zeros((self.dimension, self.dimension), dtype=float)

        #Creamos el espectro de la matriz
        self.spec = np.empty(self.dimension)
        self.spec[0], self.spec[-1] = cond_number, 1.0

        
        #Rellenamos la primera mitad
        for i in range(1, self.dimension//2):
            self.spec[i] = self.spec[0] + (self.spec[-1]-self.spec[0])*self.rnd.uniform(0, 0.2)

        #Rellenamos la segunda mitad
        for i in range(self.dimension//2, self.dimension-1):
            self.spec[i] = self.spec[0] + (self.spec[-1]-self.spec[0])*self.rnd.uniform(0.8, 1.0)

        #Ordenamos los valores propios (no es necesario)
        if sort_eigenval: self.spec = np.sort(self.spec)
        
        #Rellenamos la matriz
        np.fill_diagonal(diagonal_matrix, self.spec)

        #Intercambiamos con la matriz que deseamos y actualizamos el valor del numero de condición
        self.matrix = np.matmul(q, np.matmul(diagonal_matrix, q.T))
        self.condition_number = cond_number

        #Establecemos el valor de b de acuerdo a la forma que fue tomada
        self.b = np.matmul(self.matrix, self.rnd.random(self.dimension))
        self.init_point = np.zeros(self.dimension)


    def dense_matrix_type3(self, cond_number: float):
        #Descomponemos la matriz original en parte ortogonal y triangular
        q, r = np.linalg.qr(self.matrix)

        #Creamos una matriz diagonal para rellenar
        diagonal_matrix = np.zeros((self.dimension, self.dimension), dtype=float)

        #Creamos el espectro de la matriz
        self.spec = np.empty(self.dimension)
        self.spec[0], self.spec[-1] = np.exp(cond_number), 1.0

        for i in range(1, self.dimension-1):
            self.spec[i] = np.exp(((i-1)/(self.dimension-1))*cond_number)

        #Rellenamos la matriz
        np.fill_diagonal(diagonal_matrix, self.spec)

        #Intercambiamos con la matriz que deseamos y actualizamos el valor del numero de condición
        self.matrix = np.matmul(q, np.matmul(diagonal_matrix, q.T))
        self.condition_number = np.exp(cond_number)

        #Establecemos el valor de b de acuerdo a la forma que fue tomada
        self.b = np.matmul(self.matrix, self.rnd.random(self.dimension), dtype=float)
        self.init_point = np.zeros(self.dimension)


    def eigendist_matrix_type1(self, cond_number: float):
        """Primera distribución de matrices con espectro reducido"""

        #Descomponemos la matriz original en parte ortogonal y triangular
        q, r = np.linalg.qr(self.matrix)

        #Creamos una matriz diagonal para rellenar
        diagonal_matrix = np.zeros((self.dimension, self.dimension), dtype=float)

        #Creamos el espectro de la matriz
        self.spec = [cond_number/self.dimension for _ in range(0, self.dimension)]
        self.spec[0], self.spec[-1] = cond_number, 1

        #Rellenamos la matriz
        np.fill_diagonal(diagonal_matrix, self.spec)

        #Intercambiamos con la matriz que deseamos y actualizamos el valor del numero de condición
        self.matrix = np.matmul(q, np.matmul(diagonal_matrix, q.T))
        self.condition_number = cond_number

        #Establecemos el valor de b de acuerdo a la forma que fue tomada
        self.b = np.matmul(self.matrix, self.rnd.random(self.dimension))
        self.init_point = np.zeros(self.dimension)


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


conf_epsilon = 1e-10

def AMGM_method(x0:np.ndarray, A: np.ndarray, b: np.ndarray, tolarance: float, max_iters: int, 
                show_info: bool = False, system_to_use: int = 7):
    """Algoritmo de Acelerated Minimal Gradient Method para funciones cuadraticas convexas
    (se supone que A es definida positiva y que los vectores x0, b tiene la misma dimensión y que la multiplicación de x0 con A
    está bien definida)"""

    #Tiempo de inicio
    start_time = time.time()  # Record the start time

    #Inicializamos las variables
    xk = x0.copy()
    gk = np.matmul(A, x0) - b
    wk = np.matmul(A, gk)

    alpha  = (np.inner(gk, wk))/(np.inner(wk, wk) + conf_epsilon)
    sk = -alpha*gk
    xk += sk
    yk = -alpha*wk
    gk += yk

    #Lista donde guardar la norma de los gradientes
    # List to save some values
    #gNorm_historial = []
    #alpha_historial = []
    #beta_historial = []
    mu_historial = [0]
    
    for k in range(max_iters):

        #Si la norma del gradiente ya satisface nuestro criterio, entonces paramos  
        if np.linalg.norm(gk) < tolarance or np.isnan(np.linalg.norm(gk)): break

        #Calculamos los vectores principales
        wk_old = wk
        wk = np.matmul(A, gk)
        vk = wk-wk_old

        #Calculamos la terna de valores
        coef = __AMGM_Solver__(gk, wk, yk, vk, system_to_use)
        mu_historial.append(coef[2])

        #Actualizamos el valor x
        sk = -coef[0]*gk - coef[2]*yk -coef[1]*sk
        xk += sk

        #Actualizamos el gradiente
        yk = -coef[0]*wk - coef[2]*(wk - wk_old)- coef[1]*yk
        gk += yk
        #gNorm_historial.append(np.linalg.norm(gk))

        if show_info: 
            fk = quadratic_function(xk, A, b)
            print(f"{k} | F-value: {fk} Gk-Norm Value: {np.linalg.norm(gk)}")
    
    #print(f"AMGM - f(x_last) = {quadratic_function(xk, A, b): 1.4e} - Last Gradient- {np.linalg.norm(gk):1.4e} - Last iter: {k+1}")
    
    #Tiempo que tardo al final.
    execution_time = time.time() - start_time

    #Verification of non NaN values (to resolve the nIters and Exctime bias)
    if np.isnan(np.linalg.norm(gk)):
        execution_time = k = np.nan

    #Regresamos dependiendo de lo que se pide
    return k+1, execution_time, np.linalg.norm(gk), mu_historial

def linear_CGM(init_point: np.ndarray, matrix: np.ndarray, b_value: np.ndarray, tolerance: float = 1e-8,
               max_iters: int = None, show_info: bool = False):
    """
    Conjugate Gradient Method for solving linear systems Ax = b.
    Based on the algorithm from Nocedal.
    Returns:
        x_k: Solution vector
        num_iters: Number of iterations performed
        execution_time: Time taken in seconds
        last_residual_norm: Norm of the last residual

    """
    #Inicio de conteo del tiempo
    start_time = time.time()

    #Calculamos los primeros valores
    x_k = np.copy(init_point)
    residual = b_value - np.matmul(matrix, x_k) #gk
    direction = np.copy(residual)

    #Si el número de iteraciones no es dado lo calculamos usando n + (25% * n)
    n = init_point.shape[0]
    if max_iters is None: max_iters = int(n + (n * 0.25))

    for i in range(0, max_iters):
        #Calculamos la norma del gradiente (residuo)
        res_norm = np.linalg.norm(residual)
        
        #Vemos si la norma ya pasa la tolerancia
        if res_norm < tolerance or np.isnan(np.linalg.norm(res_norm)): break

        #Calculamos el tamaño de paso exacto
        Ad = np.matmul(matrix, direction)
        alpha = (residual.T @ residual) / (direction.T @ Ad)
        x_k += alpha * direction
        new_residual = residual - alpha * Ad

        #Calculamos el coeficiente de conjugación y la nueva dirección
        beta = (new_residual.T @ new_residual) / (residual.T @ residual)
        direction = new_residual + beta * direction

        #Actualizamos el residual
        residual = new_residual

        # Print info if there is need it.
        if show_info: print(f"Iter {i+1}: Residual norm = {res_norm:.3e}")

    execution_time = time.time() - start_time
    return i+1, execution_time, res_norm