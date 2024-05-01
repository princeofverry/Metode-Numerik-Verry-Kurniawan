import numpy as np


def crout_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for j in range(n):
        U[j, j] = 1

        for i in range(j, n):
            sum_val = sum(L[i, k] * U[k, j] for k in range(i))
            L[i, j] = A[i, j] - sum_val

        for i in range(j, n):
            sum_val = sum(L[j, k] * U[k, i] for k in range(j))
            if L[j, j] == 0:
                return None, None  # Matriks tidak bisa didekomposisi
            U[j, i] = (A[j, i] - sum_val) / L[j, j]

    return L, U


# Contoh penggunaan
A = np.array([[1, 1, -1],
              [-1, 1, 1],
              [2, 2, 1]])

L, U = crout_decomposition(A)
print("Matrix L:")
print(L)
print("Matrix U:")
print(U)


# A = np.array([[1, 1, -1], [-1, 1, 1], [2, 2, 1]])

# seharusnya U[[1, 1, -1], [0, 2, 0], [0, 0, 3]]
# seharusnya L[[1, 0, 0], [-1, 1, 0], [2, 0, 1]]
