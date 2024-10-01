import numpy as np
# Test script, to showcase and test calculated X array values

# Given values

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

# Calculated X value with 5.279925012757758e-07 error
X = np.array([
    [ 0.99821247],
    [-0.00860085],
    [ 0.73365621],
    [ 1.73243565],
    [ 0.66895002],
    [ 0.73849618]
 ])

# Prints arrays
print("Given A Array:\n", A, "\n")
print("Test X Array:\n", X, "\n")
print("Given B Array:\n", B, "\n")

# Prints resulting B array
print("A * X = _B Result array:\n", np.matmul(A, X))