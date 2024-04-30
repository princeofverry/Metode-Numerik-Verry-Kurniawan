import numpy as np

# Fungsi untuk dekomposisi Crout


def crout_decomposition(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        U[i][i] = 1

        for j in range(i, n):
            sum = 0
            for k in range(i):
                sum += L[j][k] * U[k][i]
            L[j][i] = matrix[j][i] - sum

        for j in range(i + 1, n):
            sum = 0
            for k in range(i):
                sum += L[i][k] * U[k][j]
            U[i][j] = (matrix[i][j] - sum) / L[i][i]

    return L, U


# Contoh penggunaan
A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
L, U = crout_decomposition(A)
print("Matriks L:")
print(L)
print("Matriks U:")
print(U)
