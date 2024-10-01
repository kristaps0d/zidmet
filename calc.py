# AX = B ar precizitāti E=0.00001
import numpy as np, time
from datetime import datetime

# 01-10-2024
# Variants F9
A = np.array([
    [2.9730, 0.1344, -0.1391, 0.3690, 0.1169, 0.3030],
    [-0.4202, 2.7231, 0.4066, 0.4768, 0.0202, -0.0293],
    [0.1209, 0.1579, 2.5712, 0.2687, -0.4671, -0.0042],
    [-0.1389, -0.2742, 0.0149, 2.6547, -0.2835, 0.0166],
    [0.3263, 0.3522, 0.3368, -0.2264, 3.0903, 0.1224],
    [-0.2858, 0.3518, 0.2237, -0.1947, -0.4574, 2.9803]
])

B = np.array([
    [3.8052], 
    [0.6746], 
    [2.1557], 
    [4.2964], 
    [2.3349], 
    [1.4311]
])

# 6x6 *  6x1  = 6x1
# A   *  X    = B   

# Uzdevums:
# Jaiegūst X

def Calculate(_A, _X):
    # A * X = _T
    _T = np.matmul(_A, _X)
    return _T

def Error(_T, _B):
    _E = (_T / _B)
    return _E

def Adj_X(_X, _E):
    _X = _X / _E
    return _X

def find_minimum(goal_error:float=0.0001, print_per_iterations:int=99999, log_loc_min:bool=True):
    X = np.random.rand(6, 1)

    last = 0    # last dE value
    iter = 0
    while True: 

        iter += 1

        # A * X = T
        T = Calculate(A, X)

        # Check overall system error
        # E(t) = |∑(T_i(t) / B_i) - B_i.matrix.height|
        E = Error(T, B)
        S_E = np.abs(np.sum(E) - np.shape(E)[0])
        
        # If E'(t) ~ 0: then assume local minimum and restart with new starting array values
        if S_E == last:
            
            if log_loc_min:
                print("[Loc. Min. hit]: Restarting loop [Error:", str(S_E) + "]: Hit at iteration:", iter)

            return False # flag to restart loop

        # Historical error context for comparing error rate of change (E'(t))
        last = S_E

        # Debugging
        if iter % print_per_iterations == 0:
            print("Iteration: ", iter, " error:", S_E)
        
        if S_E <= goal_error:

            print("\n")
            print("[Success]: Final error:", S_E, "\n")
            print(X, "\n")

            return True

        # X(t+1) = X(t) / E(t)
        # divide errors respectivly to their corresponding weights
        X = Adj_X(X, E)

while True:

    ret = find_minimum()
    if ret == True:
        print('[info]: Exiting!')
        break