# AX = B ar precizitāti E=0.00001
import numpy as np, time
from datetime import datetime

# 13-10-2024
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

# https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
# Formulas no ↑

# A matriksu jadekompoze uz A = L + U, 
# kur L zemākā tristūra
# un U augšējā trijstūra bez diognāles

U = np.triu(A, k=1) # izlaista diognāle
L = np.tril(A, k=0) # ir diognāle

# KKadu sakotnēju X matriksu- rand cipari
X = np.random.rand(6, 1)

def X_Next(_inv_L, _U, _B, _X):
    UX = np.matmul(_U, _X)
    BmUX = _B - UX
    
    return np.matmul(_inv_L, BmUX)

def calc_T(_inv_L, _U):
    return (-1) * np.matmul(_inv_L, _U)

def calc_c(_inv_L, _B):
    return (-1) * np.matmul(_inv_L, _B)

def calc_error(_X, _X_Next):
    error_m = np.abs(_X_Next - _X)
    return np.max(error_m)

while True:
    inv_L = np.linalg.inv(L)
    N_X = X_Next(inv_L, U, B, X)
    
    max_abs_e = calc_error(X, N_X)
    X = N_X

    if max_abs_e <= 0.00001: # target error
        print("[info]: Success! Error:", max_abs_e)
        print(X)
        break
