import numpy as np
import scipy
import time

#Quadratic standard function
quadratic_function = lambda x, A, b: 0.5*(x.T @ (A @ x)) - np.inner(x, b)

class general_Convex_Quadratic:
    def __init__(self, dimension: int, matrix: np.ndarray = None, b_vector: np.ndarray = None, 
                random_generator_seed: int = None, condition_number:float = None, init_point:np.ndarray=None):
        """ Class to create Quadratic functions (QF) of certain type or a general given some parameters 
        -Does not verify that the given matriz is positive definide.""""
        
        # Dimension and generator of matrices
        self.dimension = dimension
        self.rnd = np.random.default_rng() if random_generator_seed is None else np.random.default_rng(random_generator_seed)
        self.init_point = init_point
        self.condition_number = None
        
        #If there is not given a definite positive matrix, then we create one using our random generator
        if matrix is None:
                #Creation of a random matrix
                m =  self.rnd.random((dimension,dimension))

                #We multiply the matriz by his transpose to make it positive definite
                self.matrix = np.matmul(m.T, m)
                
                #Save the condition number
                self.condition_number = condition_number
        else:
        #Otherwise we add the given values to the matrix
            self.matrix = matrix
            self.condition_number = condition_number

        #If it is not given any vector, then we take the vector b as a zero vector, otherwise we fix that vector
        self.b = np.zeros(self.dimension) if b_vector is None else b_vector

        #If it is not given any vector, then we take again the zero vector
        self.init_point = np.zeros(self.dimension) if init_point is None else init_point

        print(f"The configuration for the quadratic convex is:\nDimensions - A:{self.matrix.shape} | B:{self.b.shape} | x0:{self.init_point.shape}")

    def function(self, x:np.ndarray)->float:
        """Quadratic function formula"""
        return 0.5 * np.matmul(x.T, np.matmul(self.matrix, x)) - np.inner(self.b, x)

    def gradient(self, x:np.ndarray)->np.ndarray:
        """Gradient of the Quadratic function"""
        return np.matmul(self.matrix, x)- self.b
    
    def hessian(self, x:np.ndarray)->np.ndarray:
        """Hessian of the Quadratic function"""
        return self.matrix

    #------------------ FROM HERE ARE DIFFERENT TYPE OF STRUTURED MATRICES ------------------#
    #------------------------  Diagonal matrices ------------------------#
    def diagonal_notDense_type1(self, cond_number: float):
        """"This is the matrix distribution for Harry Ovideo Diagonal type 1"""
        #We create an empty matrix to fill it with a spectrum
        diagonal_matrix = np.zeros((self.dimension, self.dimension), dtype=float)

        #Creation of the spectrum of the matrix
        self.spec = self.rnd.uniform(1, cond_number, self.dimension)
        self.spec[0], self.spec[-1] = cond_number, 1

        #Fill the diagonal matrix
        np.fill_diagonal(diagonal_matrix, self.spec)

        #Set the new matrix and their condition number into the info of the QF.
        self.matrix = diagonal_matrix
        self.condition_number = cond_number

        #Take b and the init point as it is appears on the original paper and update the info of the QF.
        self.b = self.rnd.uniform(-10, 10, self.dimension)
        self.init_point = self.rnd.uniform(-5, 5, self.dimension)

    def diagonal_notDense_type2(self, cond_number: float):
        """"This is the matrix distribution for Harry Ovideo Diagonal type 2"""
        #We create an empty matrix to fill it with a spectrum
        diagonal_matrix = np.zeros((self.dimension, self.dimension), dtype=float)

        #Creation of the spectrum of the matrix
        tem = [10]*self.dimension
        self.spec = self.rnd.uniform(1, np.log10(cond_number), self.dimension)
        self.spec = np.power(tem, self.spec)
        self.spec[0], self.spec[-1] = cond_number, 1

        #Fill the diagonal matrix
        np.fill_diagonal(diagonal_matrix, self.spec)

        #Set the new matrix and their condition number into the info of the QF.
        self.matrix = diagonal_matrix
        self.condition_number = cond_number

        #Take b and the init point as it is appears on the original paper and update the info of the QF.
        self.b = np.matmul(self.matrix, self.rnd.random(self.dimension))
        self.init_point = np.zeros(self.dimension)


    def dense_matrix_type1(self, cond_number: float):
        """"This is the matrix distribution for Harry Ovideo Dense type 1"""
        #Decompose the original (random matrix) in his QR Decomposition, to get and Orghonormal matrix
        q, _ = np.linalg.qr(self.matrix)

        #Create an empty matrix to fill it with a spectrum
        diagonal_matrix = np.zeros((self.dimension, self.dimension), dtype=float)

        #Creation of the spectrum of the matrix
        self.spec = np.empty(self.dimension)
        self.spec[0], self.spec[-1] = cond_number, 1.0

        # Create the eigenvalues for the distribution
        for i in range(1, self.dimension-1):
            self.spec[i] = np.power(cond_number, (self.dimension-i)/(self.dimension - 1))

        #Fill the diagonal matrix
        np.fill_diagonal(diagonal_matrix, self.spec)

        #Set the new matrix and their condition number into the info of the QF.
        self.matrix = np.matmul(q, np.matmul(diagonal_matrix, q.T))
        self.condition_number = cond_number

        #Take b and the init point as it is appears on the original paper and update the info of the QF.
        self.b = np.matmul(self.matrix, self.rnd.random(self.dimension), dtype=float)
        self.init_point = np.zeros(self.dimension)


    def dense_matrix_type2(self, cond_number: float, sort_eigenval:bool = False):
        """"This is the matrix distribution for Harry Ovideo Dense type 2"""
        #Decompose the original (random matrix) in his QR Decomposition, to get and Orghonormal matrix
        q, _ = np.linalg.qr(self.matrix)

        #Create an empty matrix to fill it with a spectrum
        diagonal_matrix = np.zeros((self.dimension, self.dimension), dtype=float)

        #Creation of the spectrum of the matrix
        self.spec = np.empty(self.dimension)
        self.spec[0], self.spec[-1] = cond_number, 1.0

        #Create the eigenvalues for the distribution (first part)
        for i in range(1, self.dimension//2):
            self.spec[i] = self.spec[0] + (self.spec[-1]-self.spec[0])*self.rnd.uniform(0, 0.2)

        #Create the eigenvalues for the distribution (second part)
        for i in range(self.dimension//2, self.dimension-1):
            self.spec[i] = self.spec[0] + (self.spec[-1]-self.spec[0])*self.rnd.uniform(0.8, 1.0)

        #Sort the eigenvalues (it's not necesary)
        if sort_eigenval: self.spec = np.sort(self.spec)
        
        #Fill the diagonal matrix
        np.fill_diagonal(diagonal_matrix, self.spec)

        #Set the new matrix and their condition number into the info of the QF.
        self.matrix = np.matmul(q, np.matmul(diagonal_matrix, q.T))
        self.condition_number = cond_number

        #Take b and the init point as it is appears on the original paper and update the info of the QF.
        self.b = np.matmul(self.matrix, self.rnd.random(self.dimension))
        self.init_point = np.zeros(self.dimension)


    def dense_matrix_type3(self, cond_number: float):
        """"This is the matrix distribution for Harry Ovideo Dense type 3 (exponential)"""

        #Decompose the original (random matrix) in his QR Decomposition, to get and Orghonormal matrix
        q, _ = np.linalg.qr(self.matrix)

        #Create an empty matrix to fill it with a spectrum
        diagonal_matrix = np.zeros((self.dimension, self.dimension), dtype=float)

        #Creation of the spectrum of the matrix
        self.spec = np.empty(self.dimension)
        self.spec[0], self.spec[-1] = np.exp(cond_number), 1.0

        #Create the eigenvalues
        for i in range(1, self.dimension-1):
            self.spec[i] = np.exp(((i-1)/(self.dimension-1))*cond_number)

        #Fill the diagonal matrix
        np.fill_diagonal(diagonal_matrix, self.spec)

        #Set the new matrix and their condition number into the info of the QF.
        self.matrix = np.matmul(q, np.matmul(diagonal_matrix, q.T))
        self.condition_number = np.exp(cond_number)

        #Take b and the init point as it is appears on the original paper and update the info of the QF.
        self.b = np.matmul(self.matrix, self.rnd.random(self.dimension), dtype=float)
        self.init_point = np.zeros(self.dimension)


    def eigendist_matrix_type1(self, cond_number: float):
        """This is not a Harry Oviedo distribution, this was created to test a problem with a matrix with a degenerated
        spectrum (3 or 2 different eigenvalues), it is a Dense type"""

        #Decompose the original (random matrix) in his QR Decomposition, to get and Orghonormal matrix
        q, _ = np.linalg.qr(self.matrix)

        #Create an empty matrix to fill it with a spectrum
        diagonal_matrix = np.zeros((self.dimension, self.dimension), dtype=float)

        #Creation of the spectrum of the matrix (constant eigenvalues)
        self.spec = [cond_number/self.dimension for _ in range(0, self.dimension)]
        self.spec[0], self.spec[-1] = cond_number, 1

        #Fill the diagonal matrix
        np.fill_diagonal(diagonal_matrix, self.spec)

        #Set the new matrix and their condition number into the info of the QF.
        self.matrix = np.matmul(q, np.matmul(diagonal_matrix, q.T))
        self.condition_number = cond_number

        #Establecemos el valor de b de acuerdo a la forma que fue tomada
        #Take b and the init point as it is appears on the original paper and update the info of the QF.
        self.init_point = np.zeros(self.dimension)


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

#Constant epsilon value
conf_epsilon = 1e-10

def AMGM_method(x0:np.ndarray, A: np.ndarray, b: np.ndarray, tolarance: float, max_iters: int, 
                show_info: bool = False, system_to_use: int = 7):
    """Algoritmo de Acelerated Minimal Gradient Method para funciones cuadraticas convexas
    (se supone que A es definida positiva y que los vectores x0, b tiene la misma dimensión y que la multiplicación de x0 con A
    está bien definida)"""

    # Record the start time
    start_time = time.time()  # Record the start time

    #Initialization of the variables (and make the first iteration)
    xk = x0.copy()
    gk = np.matmul(A, x0) - b
    wk = np.matmul(A, gk)
    alpha  = (np.inner(gk, wk))/(np.inner(wk, wk) + conf_epsilon)

    #Creation of the first element of the sequence
    sk = -alpha*gk
    xk += sk
    yk = -alpha*wk
    gk += yk

    # List to save some values
    #gNorm_historial = []
    #alpha_historial = []
    #beta_historial = []
    mu_historial = [0]
    
    #Itearation process
    for k in range(max_iters):

        #If the norm of the function is small enough, we end the process
        if np.linalg.norm(gk) < tolarance or np.isnan(np.linalg.norm(gk)): break

        #Computation of the needed vectors for the eq system.
        wk_old = wk
        wk = np.matmul(A, gk)
        vk = wk-wk_old

        #Compute the coeficients for the linear combination using a given method
        coef = __AMGM_Solver__(gk, wk, yk, vk, system_to_use)
        mu_historial.append(coef[2])

        #Update the point xk
        sk = -coef[0]*gk - coef[2]*yk -coef[1]*sk
        xk += sk

        #Update the gradient value
        yk = -coef[0]*wk - coef[2]*(wk - wk_old)- coef[1]*yk
        gk += yk
        
        #gNorm_historial.append(np.linalg.norm(gk))

        #Show information about the optimization process if it is requiered
        if show_info: 
            fk = quadratic_function(xk, A, b)
            print(f"{k} | F-value: {fk} Gk-Norm Value: {np.linalg.norm(gk)}")
    
    #Final inforation
    #print(f"AMGM - f(x_last) = {quadratic_function(xk, A, b): 1.4e} - Last Gradient- {np.linalg.norm(gk):1.4e} - Last iter: {k+1}")
    
    #Record the end time
    execution_time = time.time() - start_time

    #Verification of non NaN values (to solve the nIners and Exc Time bias)
    if np.isnan(np.linalg.norm(gk)):
        execution_time = k = np.nan

    #Return the number of iteration, the execution time and the norm, and the historial of the third value
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
    # Record the start time
    start_time = time.time()

    #Compute the first values
    x_k = np.copy(init_point)
    residual = b_value - np.matmul(matrix, x_k) #gk
    direction = np.copy(residual)

    #If the numbers of iterations is not given then, d
    n = init_point.shape[0]
    if max_iters is None: max_iters = int(n + (n * 0.25))

    for i in range(0, max_iters):
        #If the norm of the function is small enough, we end the process
        res_norm = np.linalg.norm(residual)
        if res_norm < tolerance or np.isnan(np.linalg.norm(res_norm)): break

        #Compute the exact stepseize
        Ad = np.matmul(matrix, direction)
        alpha = (residual.T @ residual) / (direction.T @ Ad)
        x_k += alpha * direction
        new_residual = residual - alpha * Ad

        #Compute the conjugation coeficient and the new direction
        beta = (new_residual.T @ new_residual) / (residual.T @ residual)
        direction = new_residual + beta * direction

        #Update the residual
        residual = new_residual

        # Print info if there is need it.
        if show_info: print(f"Iter {i+1}: Residual norm = {res_norm:.3e}")

    #Record the end time
    execution_time = time.time() - start_time
    return i+1, execution_time, res_norm